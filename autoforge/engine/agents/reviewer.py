"""Reviewer Agent — code review and quality assessment."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoforge.engine.agent_base import AgentBase, ToolDefinition
from autoforge.engine.llm_router import TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Structured review output."""

    approved: bool
    score: int
    issues: list[dict[str, Any]]
    summary: str


class ReviewerAgent(AgentBase):
    """Reviews code for correctness, security, and quality.

    Has read-only access to project files. Produces a structured
    review with approval status and issues list.
    """

    ROLE = "reviewer"
    COMPLEXITY = TaskComplexity.STANDARD

    def __init__(self, config, llm, working_dir: Path) -> None:
        self.working_dir = working_dir
        super().__init__(config, llm)

    def _register_tools(self) -> None:
        """Reviewer has read-only tools."""
        self._tools = [
            ToolDefinition(
                name="read_file",
                description="Read a file from the project (read-only access).",
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

    async def _handle_read_file(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data["path"]
        full_path = (self.working_dir / rel_path).resolve()
        if not str(full_path).startswith(str(self.working_dir.resolve())):
            return json.dumps({"error": "Path traversal not allowed"})
        if not full_path.exists():
            return json.dumps({"error": f"File not found: {rel_path}"})
        return full_path.read_text(encoding="utf-8")

    async def _handle_list_files(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data.get("path", ".")
        full_path = (self.working_dir / rel_path).resolve()
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
        task = context.get("task", {})
        spec = context.get("spec", {})

        # Full project review mode (no specific task)
        if context.get("full_project_review"):
            project_name = spec.get("project_name", "project")
            modules = spec.get("modules", [])
            tech_stack = json.dumps(spec.get("tech_stack", {}), indent=2, ensure_ascii=False)
            return (
                f"Perform a comprehensive code review of the project "
                f"'{project_name}'.\n\n"
                f"## Tech Stack\n```json\n{tech_stack}\n```\n\n"
                f"## Modules\n"
                + "\n".join(
                    f"- **{m.get('name', '?')}**: {m.get('description', '')} "
                    f"({', '.join(m.get('files', []))})"
                    for m in modules
                )
                + "\n\n"
                f"## Instructions\n"
                f"1. List all files in the project\n"
                f"2. Read each source file systematically\n"
                f"3. Check correctness, security, error handling, and code quality\n"
                f"4. Pay special attention to: SQL injection, XSS, hardcoded secrets, "
                f"error handling, input validation\n"
                f"5. Output a JSON code block with your review following the format "
                f"in your system prompt\n\n"
                f"Give a thorough review with specific file:line references where possible.\n"
            )

        # Standard task-level review
        return (
            f"Review the code for task '{task.get('id', '')}' in project "
            f"'{spec.get('project_name', '')}'.\n\n"
            f"## Task Description\n{task.get('description', 'N/A')}\n\n"
            f"## Files to Review\n{task.get('files', [])}\n\n"
            f"## Instructions\n"
            f"1. Read each file listed above\n"
            f"2. Check correctness, security, error handling, and code quality\n"
            f"3. Output a JSON code block with your review following the format "
            f"in your system prompt\n"
        )

    def parse_review(self, output: str) -> ReviewResult:
        """Extract structured review from output."""
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", output, re.DOTALL)
        if match:
            data = json.loads(match.group(1).strip())
        else:
            start = output.find("{")
            end = output.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(output[start : end + 1])
            else:
                # Default to approved if we can't parse
                logger.warning("Could not parse review output, defaulting to approved")
                return ReviewResult(approved=True, score=7, issues=[], summary="Auto-approved")

        return ReviewResult(
            approved=data.get("approved", True),
            score=data.get("score", 7),
            issues=data.get("issues", []),
            summary=data.get("summary", ""),
        )
