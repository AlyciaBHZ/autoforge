"""Gardener Agent — code refactoring and quality improvement."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from autoforge.engine.agent_base import AgentBase, ToolDefinition
from autoforge.engine.llm_router import TaskComplexity

logger = logging.getLogger(__name__)


class GardenerAgent(AgentBase):
    """Improves code quality through targeted refactoring.

    Only refactors when there are specific, actionable issues from the Reviewer.
    Makes minimal changes to avoid introducing bugs.
    """

    ROLE = "gardener"
    COMPLEXITY = TaskComplexity.STANDARD

    def __init__(self, config, llm, working_dir: Path) -> None:
        self.working_dir = working_dir
        super().__init__(config, llm)

    def _register_tools(self) -> None:
        self._tools = [
            ToolDefinition(
                name="write_file",
                description="Update a file with improved code.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path",
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete updated file content",
                        },
                    },
                    "required": ["path", "content"],
                },
                handler=self._handle_write_file,
            ),
            ToolDefinition(
                name="read_file",
                description="Read a file to assess its quality.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path",
                        },
                    },
                    "required": ["path"],
                },
                handler=self._handle_read_file,
            ),
            ToolDefinition(
                name="list_files",
                description="List files in a directory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative directory path",
                        },
                    },
                },
                handler=self._handle_list_files,
            ),
        ]

        # Add grep_search for finding code patterns to refactor
        from functools import partial

        from autoforge.engine.tools.search import (
            GREP_SEARCH_TOOL_SCHEMA,
            handle_grep_search,
        )

        self._tools.append(ToolDefinition(
            name="grep_search",
            description=(
                "Search project files for a pattern (regex). "
                "Use to find code smells, dead code, TODOs, and duplications."
            ),
            input_schema=GREP_SEARCH_TOOL_SCHEMA,
            handler=partial(handle_grep_search, working_dir=self.working_dir),
        ))

        # Add fetch_url if web tools enabled
        if getattr(self.config, "web_tools_enabled", True):
            from autoforge.engine.tools.web import (
                FETCH_URL_TOOL_SCHEMA,
                handle_fetch_url,
            )

            self._tools.append(ToolDefinition(
                name="fetch_url",
                description=(
                    "Fetch a web page and return its text content. "
                    "Use to read best-practice guides and code quality references."
                ),
                input_schema=FETCH_URL_TOOL_SCHEMA,
                handler=handle_fetch_url,
            ))

    def _validate_path(self, rel_path: str) -> Path:
        full_path = (self.working_dir / rel_path).resolve()
        if not str(full_path).startswith(str(self.working_dir.resolve())):
            raise ValueError(f"Path traversal detected: {rel_path}")
        return full_path

    async def _handle_write_file(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data["path"]
        content = input_data["content"]
        full_path = self._validate_path(rel_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        self._artifacts[rel_path] = str(full_path)
        return json.dumps({"status": "ok", "path": rel_path})

    async def _handle_read_file(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data["path"]
        full_path = self._validate_path(rel_path)
        if not full_path.exists():
            return json.dumps({"error": f"File not found: {rel_path}"})
        return full_path.read_text(encoding="utf-8")

    async def _handle_list_files(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data.get("path", ".")
        full_path = self._validate_path(rel_path)
        if not full_path.is_dir():
            return json.dumps({"error": f"Not a directory: {rel_path}"})
        files = []
        for p in sorted(full_path.rglob("*")):
            if p.is_file() and ".git" not in p.parts:
                try:
                    files.append(str(p.relative_to(self.working_dir)))
                except ValueError:
                    continue
        return json.dumps(files)

    def build_prompt(self, context: dict[str, Any]) -> str:
        review = context.get("review", {})
        spec = context.get("spec", {})
        issues = review.get("issues", [])

        parts = [
            f"Refactor the project '{spec.get('project_name', '')}' based on review feedback.\n\n",
        ]

        if issues:
            parts.append("## Issues to Fix\n\n")
            for i, issue in enumerate(issues, 1):
                parts.append(
                    f"{i}. [{issue.get('severity', 'info')}] "
                    f"{issue.get('file', 'unknown')}: "
                    f"{issue.get('description', '')}\n"
                )
                if issue.get("suggestion"):
                    parts.append(f"   Suggestion: {issue['suggestion']}\n")
            parts.append("\n")

        parts.append(
            "## Instructions\n"
            "1. Read the files with issues\n"
            "2. Make targeted fixes for each issue\n"
            "3. Do NOT rewrite working code unnecessarily\n"
            "4. Output a JSON summary of changes made\n"
        )

        return "".join(parts)

    def parse_changes(self, output: str) -> dict[str, Any]:
        """Extract change summary from output."""
        raw: str | None = None
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", output, re.DOTALL)
        if match:
            raw = match.group(1).strip()
        else:
            start = output.find("{")
            end = output.rfind("}")
            if start != -1 and end != -1:
                raw = output[start : end + 1]

        if raw is not None:
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in gardener output: {e}")

        return {"changes_made": [], "summary": output[:500]}
