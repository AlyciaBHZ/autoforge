"""Agent base class — the core agentic tool-use loop."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

from engine.config import ForgeConfig
from engine.llm_router import LLMRouter, TaskComplexity

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


class AgentBase(ABC):
    """Base class for all AutoForge agents.

    Implements the agentic tool-use loop:
        1. Send messages to LLM (with system prompt + tools)
        2. If stop_reason == "end_turn", done
        3. If stop_reason == "tool_use", execute tools, append results
        4. Repeat until done or MAX_TURNS exceeded
    """

    # Subclasses must set these
    ROLE: str = ""
    COMPLEXITY: TaskComplexity = TaskComplexity.STANDARD
    MAX_TURNS: int = 25

    def __init__(self, config: ForgeConfig, llm: LLMRouter) -> None:
        self.config = config
        self.llm = llm
        self._system_prompt: str = ""
        self._tools: list[ToolDefinition] = []
        self._output_parts: list[str] = []
        self._artifacts: dict[str, Any] = {}
        self._load_system_prompt()
        self._register_tools()

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
        """Convert ToolDefinitions to Anthropic API format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools
        ]

    async def _execute_tool(self, name: str, input_data: dict[str, Any]) -> str:
        """Execute a tool by name and return the result string."""
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

        Args:
            context: Dictionary with task-specific data for build_prompt().

        Returns:
            AgentResult with success status, output text, and artifacts.
        """
        self._output_parts = []
        self._artifacts = {}
        start_time = time.monotonic()

        user_prompt = self.build_prompt(context)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        tool_defs = self._get_tool_definitions()
        turns = 0

        self._log_action("started", f"prompt length={len(user_prompt)}")

        try:
            while turns < self.MAX_TURNS:
                turns += 1

                response = await self.llm.call(
                    complexity=self.COMPLEXITY,
                    system=self._system_prompt,
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
                    for block in response.content:
                        if block.type == "tool_use":
                            self._log_action("tool_call", f"{block.name}")
                            result = await self._execute_tool(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            })

                    # Append assistant response + tool results to conversation
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
            else:
                # MAX_TURNS exceeded — this is a failure
                self._log_action("max_turns_reached", f"turns={turns}")
                elapsed = time.monotonic() - start_time
                return AgentResult(
                    agent_name=self.ROLE,
                    success=False,
                    output="\n".join(self._output_parts),
                    artifacts=self._artifacts,
                    error=f"Agent exceeded maximum turns ({self.MAX_TURNS})",
                    total_turns=turns,
                    duration_seconds=elapsed,
                )

            elapsed = time.monotonic() - start_time
            return AgentResult(
                agent_name=self.ROLE,
                success=True,
                output="\n".join(self._output_parts),
                artifacts=self._artifacts,
                total_turns=turns,
                duration_seconds=elapsed,
            )

        except Exception as e:
            elapsed = time.monotonic() - start_time
            self._log_action("failed", str(e))
            return AgentResult(
                agent_name=self.ROLE,
                success=False,
                output="\n".join(self._output_parts),
                error=str(e),
                total_turns=turns,
                duration_seconds=elapsed,
            )
