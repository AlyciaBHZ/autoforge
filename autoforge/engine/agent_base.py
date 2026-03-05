"""Agent base class — the core agentic tool-use loop.

Enhanced with:
  - Mid-task checkpoints (process reward model style)
  - Search tree integration for branching decisions
  - Dynamic constitution support
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

from autoforge.engine.config import ForgeConfig
from autoforge.engine.llm_router import BudgetExceededError, LLMRouter, TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """A tool that an agent can use."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Coroutine[Any, Any, str]] | None = None


@dataclass
class AgentResult:
    """Result of an agent's execution."""

    agent_name: str
    success: bool
    output: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    total_turns: int = 0
    duration_seconds: float = 0.0
    files_written: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)


class FileToolsMixin:
    """Mixin providing standard file tool handlers for agents with a working_dir.

    Subclasses must set ``self.working_dir`` (a :class:`Path`) before using
    these handlers.  Each handler returns a JSON string suitable for the
    agentic tool-use loop.
    """

    # 2 MB per file limit to prevent runaway writes
    MAX_FILE_SIZE = 2 * 1024 * 1024

    # --- handlers -----------------------------------------------------------

    async def _handle_read_file(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data["path"]
        try:
            full_path = AgentBase.validate_path(rel_path, self.working_dir)  # type: ignore[attr-defined]
        except ValueError:
            return json.dumps({"error": "Path traversal not allowed"})
        if not full_path.exists():
            return json.dumps({"error": f"File not found: {rel_path}"})
        try:
            return full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return json.dumps({"error": f"Cannot read binary file: {rel_path}"})

    async def _handle_write_file(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data["path"]
        content = input_data["content"]
        if len(content) > self.MAX_FILE_SIZE:
            return json.dumps({
                "error": f"File too large ({len(content)} bytes). Max is {self.MAX_FILE_SIZE} bytes.",
            })
        try:
            full_path = AgentBase.validate_path(rel_path, self.working_dir)  # type: ignore[attr-defined]
        except ValueError:
            return json.dumps({"error": "Path traversal not allowed"})
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        self._artifacts[rel_path] = str(full_path)  # type: ignore[attr-defined]
        return json.dumps({"status": "ok", "path": rel_path, "bytes": len(content)})

    async def _handle_list_files(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data.get("path", ".")
        try:
            full_path = AgentBase.validate_path(rel_path, self.working_dir)  # type: ignore[attr-defined]
        except ValueError:
            return json.dumps({"error": "Path traversal not allowed"})
        if not full_path.is_dir():
            return json.dumps({"error": f"Not a directory: {rel_path}"})
        base_resolved = self.working_dir.resolve()  # type: ignore[attr-defined]
        files = []
        for p in sorted(full_path.rglob("*")):
            if p.is_file() and ".git" not in p.parts:
                # Prevent symlinks from escaping the workspace
                if not p.resolve().is_relative_to(base_resolved):
                    continue
                try:
                    files.append(str(p.relative_to(self.working_dir)))  # type: ignore[attr-defined]
                except ValueError:
                    continue
        return json.dumps(files)

    async def _handle_run_command(self, input_data: dict[str, Any]) -> str:
        command = input_data["command"]
        if getattr(self, "sandbox", None):
            result = await self.sandbox.exec(command)  # type: ignore[attr-defined]
            return json.dumps({
                "exit_code": result.exit_code,
                "stdout": result.stdout[:8000],
                "stderr": result.stderr[:4000],
                "timed_out": getattr(result, "timed_out", False),
            })
        else:
            return json.dumps({
                "warning": "No sandbox available, command not executed",
                "command": command,
            })


class AgentBase(ABC):
    """Base class for all AutoForge agents.

    Implements the agentic tool-use loop:
        1. Send messages to LLM (with system prompt + tools)
        2. If stop_reason == "end_turn", done
        3. If stop_reason == "tool_use", execute tools, append results
        4. Repeat until done or MAX_TURNS exceeded

    Supports two modes via config.mode:
        - "developer": Full read-write access (default)
        - "research": Read-only; write tools return errors
    """

    # Subclasses must set these
    ROLE: str = ""
    COMPLEXITY: TaskComplexity = TaskComplexity.STANDARD
    MAX_TURNS: int = 25

    @staticmethod
    def validate_path(path_str: str, base_dir: Path) -> Path:
        """Validate and resolve a path, preventing path traversal.

        Uses Path.is_relative_to() which correctly handles:
        - Case-insensitive comparisons on Windows
        - Prefix collisions (e.g. /proj vs /project)
        - Symlink resolution

        Args:
            path_str: The relative path string to validate.
            base_dir: The base directory the path must stay within.

        Returns:
            The resolved absolute Path.

        Raises:
            ValueError: If the resolved path escapes base_dir.
        """
        resolved = (base_dir / path_str).resolve()
        base_resolved = base_dir.resolve()
        if not resolved.is_relative_to(base_resolved):
            raise ValueError(f"Path traversal detected: {path_str}")
        return resolved

    # Tools that modify state — blocked in research mode
    WRITE_TOOLS: set[str] = {"write_file", "run_command", "delete_file"}

    # Anti-spin detection: turns without write_file before warning/failure
    SPIN_WARN_TURNS: int = 10
    SPIN_FAIL_TURNS: int = 20

    def __init__(self, config: ForgeConfig, llm: LLMRouter) -> None:
        self.config = config
        self.llm = llm
        self.mode: str = getattr(config, "mode", "developer")
        self._system_prompt: str = ""
        self._dynamic_supplement: str = ""  # Dynamic constitution patches
        self._tools: list[ToolDefinition] = []
        self._output_parts: list[str] = []
        self._artifacts: dict[str, Any] = {}
        self._checkpoint_mgr: Any = None  # Lazy-initialized CheckpointManager
        self._load_system_prompt()
        self._register_tools()

    def set_dynamic_constitution(self, supplement: str) -> None:
        """Set dynamic constitution supplement (project-specific instructions)."""
        self._dynamic_supplement = supplement
        logger.debug(f"[{self.ROLE}] Dynamic constitution set ({len(supplement)} chars)")

    def _load_system_prompt(self) -> None:
        """Load system prompt from constitution/agents/{role}.md."""
        prompt_path = self.config.constitution_dir / "agents" / f"{self.ROLE}.md"
        if prompt_path.exists():
            self._system_prompt = prompt_path.read_text(encoding="utf-8")
            logger.debug(f"[{self.ROLE}] Loaded system prompt from {prompt_path}")
        else:
            logger.warning(f"[{self.ROLE}] No system prompt at {prompt_path}, using default")
            self._system_prompt = f"You are the {self.ROLE} agent in the AutoForge system."

    @abstractmethod
    def _register_tools(self) -> None:
        """Subclasses register their available tools here."""

    @abstractmethod
    def build_prompt(self, context: dict[str, Any]) -> str:
        """Build the user message for this agent's task."""

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        """Convert ToolDefinitions to internal format (provider conversion in LLMRouter)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools
        ]

    async def _execute_tool(self, name: str, input_data: dict[str, Any]) -> str:
        """Execute a tool by name and return the result string.

        In research mode, write tools are blocked and return an error message.
        """
        # Block write operations in research mode
        if self.mode == "research" and name in self.WRITE_TOOLS:
            logger.info(f"[{self.ROLE}] Blocked '{name}' in research mode")
            return json.dumps({
                "error": f"Tool '{name}' is disabled in research mode. "
                "Research mode is read-only. Switch to developer mode to make changes."
            })

        for tool in self._tools:
            if tool.name == name and tool.handler is not None:
                try:
                    result = await tool.handler(input_data)
                    return str(result)
                except Exception as e:
                    logger.error(f"[{self.ROLE}] Tool '{name}' failed: {e}")
                    return json.dumps({"error": str(e)})
        return json.dumps({"error": f"Unknown tool: {name}"})

    def _log_action(self, action: str, detail: str = "") -> None:
        """Log an agent action."""
        msg = f"[{self.ROLE}] {action}"
        if detail:
            msg += f": {detail}"
        logger.info(msg)

    async def run(self, context: dict[str, Any]) -> AgentResult:
        """Execute the agentic tool-use loop.

        Enhanced with mid-task checkpoints and dynamic constitution support.

        Args:
            context: Dictionary with task-specific data for build_prompt().

        Returns:
            AgentResult with success status, output text, artifacts, and metrics.
        """
        self._output_parts = []
        self._artifacts = {}
        start_time = time.monotonic()

        # Build effective system prompt (static + dynamic)
        effective_system = self._system_prompt
        if self._dynamic_supplement:
            effective_system += self._dynamic_supplement

        user_prompt = self.build_prompt(context)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        tool_defs = self._get_tool_definitions()
        turns = 0

        # --- Metrics tracking (Section D) ---
        tool_counts: dict[str, int] = {}
        files_written = 0
        files_written_list: list[str] = []
        # --- Anti-spin tracking (Section B) ---
        last_write_turn = 0  # last turn that called write_file
        has_write_tools = bool(self.WRITE_TOOLS & {t.name for t in self._tools})

        # --- Checkpoint setup (Section E) ---
        checkpoint_enabled = has_write_tools and self.MAX_TURNS >= 15
        if checkpoint_enabled:
            from autoforge.engine.checkpoints import CheckpointManager
            self._checkpoint_mgr = CheckpointManager(
                config=self.config, llm=self.llm,
                checkpoint_interval=8,
            )

        self._log_action("started", f"prompt length={len(user_prompt)}")

        def _build_result(success: bool, error: str = "") -> AgentResult:
            elapsed = time.monotonic() - start_time
            return AgentResult(
                agent_name=self.ROLE,
                success=success,
                output="\n".join(self._output_parts),
                artifacts=self._artifacts,
                error=error,
                total_turns=turns,
                duration_seconds=elapsed,
                files_written=files_written,
                tool_calls=dict(tool_counts),
            )

        try:
            while turns < self.MAX_TURNS:
                turns += 1

                response = await self.llm.call(
                    complexity=self.COMPLEXITY,
                    system=effective_system,
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                )

                # Collect text output
                for block in response.content:
                    if block.type == "text":
                        self._output_parts.append(block.text)

                # If model is done (no more tool calls), break
                if response.stop_reason == "end_turn":
                    self._log_action("completed", f"turns={turns}")
                    break

                # Process tool calls
                if response.stop_reason == "tool_use":
                    tool_results = []
                    wrote_this_turn = False

                    for block in response.content:
                        if block.type == "tool_use":
                            self._log_action("tool_call", f"{block.name}")
                            result = await self._execute_tool(block.name, block.input)

                            # Track metrics (Section D)
                            tool_counts[block.name] = tool_counts.get(block.name, 0) + 1
                            if block.name == "write_file":
                                files_written += 1
                                wrote_this_turn = True
                                # Track file paths for checkpoint system
                                if block.input.get("path"):
                                    files_written_list.append(block.input["path"])

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            })

                    # Update last write turn for spin detection
                    if wrote_this_turn:
                        last_write_turn = turns

                    # --- Anti-spin detection (Section B) ---
                    if has_write_tools and self.mode != "research":
                        idle_turns = turns - last_write_turn

                        # Hard fail: too many turns without writing
                        if idle_turns >= self.SPIN_FAIL_TURNS:
                            spin_msg = (
                                f"Agent spinning: no file output in {idle_turns} turns"
                            )
                            self._log_action("spin_detected", spin_msg)
                            return _build_result(False, error=spin_msg)

                        # Warning nudge: inject reminder into last tool result
                        if idle_turns >= self.SPIN_WARN_TURNS and tool_results:
                            nudge = (
                                f"\n\nWARNING: You have not written any files in "
                                f"{idle_turns} turns. Focus on writing code now. "
                                f"If blocked, report the issue explicitly."
                            )
                            tool_results[-1]["content"] += nudge
                            self._log_action(
                                "spin_warning",
                                f"nudge injected at turn {turns} ({idle_turns} idle)",
                            )

                    # --- Mid-task checkpoint (Section E) ---
                    if (checkpoint_enabled and self._checkpoint_mgr
                            and self._checkpoint_mgr.should_check(turns)):
                        task_desc = context.get("task", {}).get("description", user_prompt[:500])
                        verdict = await self._checkpoint_mgr.check_direction(
                            task_description=task_desc,
                            messages_so_far=messages,
                            files_written=files_written_list,
                            turn=turns,
                        )
                        if verdict.suggested_action == "adjust" and verdict.feedback:
                            # Inject course-correction guidance into tool results
                            correction = (
                                f"\n\n[CHECKPOINT REVIEW — Turn {turns}]\n"
                                f"Direction score: {verdict.score:.1f}/1.0\n"
                                f"Feedback: {verdict.feedback}\n"
                                f"Action: Adjust your approach based on this feedback."
                            )
                            if tool_results:
                                tool_results[-1]["content"] += correction
                            self._log_action("checkpoint_adjust", verdict.feedback[:100])
                        elif verdict.should_reset:
                            # Rollback to last good checkpoint
                            good_cp = self._checkpoint_mgr.get_last_good_checkpoint()
                            if good_cp:
                                messages = good_cp.messages_snapshot
                                # Reset stale tracking state to avoid
                                # spin-detection false positives and
                                # accumulated metric drift after rollback.
                                files_written = 0
                                files_written_list = []
                                last_write_turn = 0
                                tool_counts = {}
                                reset_msg = (
                                    f"\n\n[CHECKPOINT RESET — Turn {turns}]\n"
                                    f"Your previous approach scored {verdict.score:.1f}/1.0.\n"
                                    f"Feedback: {verdict.feedback}\n"
                                    f"You have been reset to turn {good_cp.turn}. "
                                    f"Try a DIFFERENT approach this time."
                                )
                                messages.append({
                                    "role": "user",
                                    "content": reset_msg,
                                })
                                self._log_action("checkpoint_reset",
                                    f"Reset to turn {good_cp.turn}, score={verdict.score:.2f}")
                                continue  # Skip appending, restart loop with reset messages

                    # Append assistant response + tool results to conversation
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
            else:
                # MAX_TURNS exceeded — this is a failure
                self._log_action("max_turns_reached", f"turns={turns}")
                return _build_result(
                    False,
                    error=f"Agent exceeded maximum turns ({self.MAX_TURNS})",
                )

            # Log metrics summary (Section D)
            total_calls = sum(tool_counts.values())
            self._log_action(
                "metrics",
                f"{files_written} files written, {total_calls} tool calls in {turns} turns",
            )

            return _build_result(True)

        except BudgetExceededError:
            raise  # Propagate budget errors to the orchestrator
        except Exception as e:
            self._log_action("failed", str(e))
            return _build_result(False, error=str(e))
