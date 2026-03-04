"""Builder Agent — code implementation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from autoforge.engine.agent_base import AgentBase, ToolDefinition
from autoforge.engine.llm_router import TaskComplexity
from autoforge.engine.sandbox import SandboxBase

logger = logging.getLogger(__name__)


class BuilderAgent(AgentBase):
    """Implements specific tasks by writing production-quality code.

    Each Builder works in its own working directory (either a git worktree
    or the main project dir) and has access to file and command tools.
    """

    ROLE = "builder"
    COMPLEXITY = TaskComplexity.STANDARD  # Uses Sonnet for cost-effective coding
    MAX_TURNS = 30  # Builders may need more iterations

    def __init__(
        self,
        config,
        llm,
        working_dir: Path,
        sandbox: SandboxBase | None = None,
        agent_id: str = "builder-00",
    ) -> None:
        self.working_dir = working_dir
        self.sandbox = sandbox
        self.agent_id = agent_id
        super().__init__(config, llm)

    def _register_tools(self) -> None:
        self._tools = [
            ToolDefinition(
                name="write_file",
                description=(
                    "Write content to a file. Creates parent directories automatically. "
                    "Path is relative to the project root."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path from project root",
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete file content",
                        },
                    },
                    "required": ["path", "content"],
                },
                handler=self._handle_write_file,
            ),
            ToolDefinition(
                name="read_file",
                description="Read the content of a file in the project.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path from project root",
                        },
                    },
                    "required": ["path"],
                },
                handler=self._handle_read_file,
            ),
            ToolDefinition(
                name="list_files",
                description="List all files in a directory of the project.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative directory path (default: project root)",
                        },
                    },
                },
                handler=self._handle_list_files,
            ),
            ToolDefinition(
                name="run_command",
                description=(
                    "Execute a shell command in the project sandbox. "
                    "Use for installing dependencies, running build tools, etc."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute",
                        },
                    },
                    "required": ["command"],
                },
                handler=self._handle_run_command,
            ),
        ]

        # Add grep_search for finding code patterns
        from functools import partial

        from autoforge.engine.tools.search import (
            GREP_SEARCH_TOOL_SCHEMA,
            handle_grep_search,
        )

        self._tools.append(ToolDefinition(
            name="grep_search",
            description=(
                "Search project files for a pattern (regex). "
                "Use to find imports, function definitions, existing patterns, "
                "and integration points before writing code."
            ),
            input_schema=GREP_SEARCH_TOOL_SCHEMA,
            handler=partial(handle_grep_search, working_dir=self.working_dir),
        ))

        # Add fetch_url and GitHub tools if web tools enabled
        if getattr(self.config, "web_tools_enabled", True):
            from autoforge.engine.tools.web import (
                FETCH_URL_TOOL_SCHEMA,
                handle_fetch_url,
            )

            self._tools.append(ToolDefinition(
                name="fetch_url",
                description=(
                    "Fetch a web page and return its text content. "
                    "Use to read API documentation or code examples while implementing."
                ),
                input_schema=FETCH_URL_TOOL_SCHEMA,
                handler=handle_fetch_url,
            ))

            # GitHub search for discovering libraries during implementation
            import os
            from autoforge.engine.tools.github_search import (
                SEARCH_GITHUB_TOOL_SCHEMA,
                handle_search_github,
            )

            github_token = os.environ.get("GITHUB_TOKEN", "")

            self._tools.append(ToolDefinition(
                name="search_github",
                description=(
                    "Search GitHub for libraries when you need a specific package. "
                    "Use to find the right npm/pip package for a sub-task."
                ),
                input_schema=SEARCH_GITHUB_TOOL_SCHEMA,
                handler=partial(handle_search_github, github_token=github_token),
            ))

    def _validate_path(self, rel_path: str) -> Path:
        """Validate and resolve a relative path, preventing path traversal."""
        return self.validate_path(rel_path, self.working_dir)

    # 2 MB per file limit to prevent runaway writes
    MAX_FILE_SIZE = 2 * 1024 * 1024

    async def _handle_write_file(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data["path"]
        content = input_data["content"]
        if len(content) > self.MAX_FILE_SIZE:
            return json.dumps({
                "error": f"File too large ({len(content)} bytes). Max is {self.MAX_FILE_SIZE} bytes.",
            })
        full_path = self._validate_path(rel_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        self._artifacts[rel_path] = str(full_path)
        return json.dumps({"status": "ok", "path": rel_path, "bytes": len(content)})

    async def _handle_read_file(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data["path"]
        full_path = self._validate_path(rel_path)
        if not full_path.exists():
            return json.dumps({"error": f"File not found: {rel_path}"})
        content = full_path.read_text(encoding="utf-8")
        return content

    async def _handle_list_files(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data.get("path", ".")
        full_path = self._validate_path(rel_path)
        if not full_path.is_dir():
            return json.dumps({"error": f"Not a directory: {rel_path}"})
        base_resolved = self.working_dir.resolve()
        files = []
        for p in sorted(full_path.rglob("*")):
            if p.is_file() and ".git" not in p.parts:
                # Prevent symlinks from escaping the workspace
                if not p.resolve().is_relative_to(base_resolved):
                    continue
                try:
                    files.append(str(p.relative_to(self.working_dir)))
                except ValueError:
                    continue
        return json.dumps(files)

    async def _handle_run_command(self, input_data: dict[str, Any]) -> str:
        command = input_data["command"]
        if self.sandbox:
            result = await self.sandbox.exec(command)
            return json.dumps({
                "exit_code": result.exit_code,
                "stdout": result.stdout[:5000],
                "stderr": result.stderr[:2000],
            })
        else:
            logger.warning(f"[{self.agent_id}] No sandbox available, skipping: {command[:80]}")
            return json.dumps({
                "warning": "No sandbox available — command was NOT executed. "
                           "This is expected when running without Docker.",
                "command": command,
            })

    def build_prompt(self, context: dict[str, Any]) -> str:
        task = context.get("task", {})
        spec = context.get("spec", {})
        architecture = context.get("architecture", "")
        existing_files = context.get("existing_files", [])

        parts = [
            f"You are implementing a task for the project '{spec.get('project_name', '')}'.\n",
            f"## Task\n",
            f"- ID: {task.get('id', 'N/A')}\n",
            f"- Description: {task.get('description', 'N/A')}\n",
            f"- Files to create/modify: {task.get('files', [])}\n",
        ]

        if task.get("acceptance_criteria"):
            parts.append(f"- Acceptance criteria: {task['acceptance_criteria']}\n")

        if task.get("fix_strategy"):
            parts.append(f"\n## Fix Strategy\n{task['fix_strategy']}\n")

        parts.append(f"\n## Project Spec\n```json\n{json.dumps(spec, indent=2, ensure_ascii=False)}\n```\n")

        if architecture:
            parts.append(f"\n## Architecture Notes\n{architecture}\n")

        if existing_files:
            parts.append(f"\n## Existing Files\nThese files already exist in the project:\n")
            for f in existing_files:
                parts.append(f"- {f}\n")
            parts.append(
                "\nRead any relevant existing files before implementing "
                "to ensure consistency.\n"
            )

        parts.append(
            "\n## Instructions\n"
            "Use the write_file tool to create each required file. "
            "Write complete, production-ready code. "
            "Include proper imports, type hints, error handling, and comments.\n"
        )

        return "".join(parts)
