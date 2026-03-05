"""Reviewer Agent — code review and quality assessment."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoforge.engine.agent_base import AgentBase, FileToolsMixin, ToolDefinition
from autoforge.engine.llm_router import TaskComplexity
from autoforge.engine.sandbox import SandboxBase

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Structured review output."""

    approved: bool
    score: int
    issues: list[dict[str, Any]]
    summary: str


class ReviewerAgent(FileToolsMixin, AgentBase):
    """Reviews code for correctness, security, and quality.

    Has read-only access to project files plus the ability to run
    verification commands (syntax checks, linting, type checking).
    Produces a structured review with approval status and issues list.
    """

    ROLE = "reviewer"
    COMPLEXITY = TaskComplexity.STANDARD

    def __init__(
        self, config, llm, working_dir: Path,
        sandbox: "SandboxBase | None" = None,
    ) -> None:
        self.working_dir = working_dir
        self.sandbox = sandbox
        super().__init__(config, llm)

    def _register_tools(self) -> None:
        """Reviewer has read-only tools plus verification commands."""
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
            ToolDefinition(
                name="run_check",
                description=(
                    "Run a read-only verification command (syntax check, type check, lint). "
                    "Use to verify code actually compiles/parses correctly. "
                    "Examples: 'python -m py_compile src/app.py', 'npx tsc --noEmit', "
                    "'node --check src/index.js', 'python -m flake8 src/'. "
                    "Do NOT use for modifying files or running the application."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Verification command to run",
                        },
                    },
                    "required": ["command"],
                },
                handler=self._handle_run_check,
            ),
        ]

        # Add grep_search for finding code patterns during review
        from functools import partial

        from autoforge.engine.tools.search import (
            GREP_SEARCH_TOOL_SCHEMA,
            handle_grep_search,
        )

        self._tools.append(ToolDefinition(
            name="grep_search",
            description=(
                "Search project files for a pattern (regex). "
                "Use to find cross-cutting issues like hardcoded secrets, "
                "missing error handling patterns, or inconsistent naming."
            ),
            input_schema=GREP_SEARCH_TOOL_SCHEMA,
            handler=partial(handle_grep_search, working_dir=self.working_dir),
        ))

    async def _handle_run_check(self, input_data: dict) -> str:
        """Run a read-only verification command."""
        command = input_data.get("command", "")
        if not command:
            return json.dumps({"error": "No command provided"})

        # Block obviously dangerous commands
        dangerous = ["rm ", "mv ", "cp ", "write", "chmod", "chown", "kill", "sudo"]
        if any(d in command.lower() for d in dangerous):
            return json.dumps({"error": "Only read-only verification commands are allowed"})

        if self.sandbox:
            result = await self.sandbox.exec(command, timeout=30)
            return json.dumps({
                "exit_code": result.exit_code,
                "stdout": (result.stdout or "")[:4000],
                "stderr": (result.stderr or "")[:2000],
            })
        else:
            return json.dumps({"error": "No sandbox available for running commands"})

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
        architecture = context.get("architecture")
        arch_section = ""
        if architecture:
            arch_text = json.dumps(architecture, indent=2, ensure_ascii=False) if isinstance(architecture, dict) else str(architecture)
            arch_section = (
                f"## Architecture (validate code matches this design)\n"
                f"```json\n{arch_text[:3000]}\n```\n\n"
            )

        return (
            f"Review the code for task '{task.get('id', '')}' in project "
            f"'{spec.get('project_name', '')}'.\n\n"
            f"## Task Description\n{task.get('description', 'N/A')}\n\n"
            f"{arch_section}"
            f"## Files to Review\n{task.get('files', [])}\n\n"
            f"## Instructions\n"
            f"1. Read each file listed above\n"
            f"2. Check correctness, security, error handling, and code quality\n"
            f"3. Verify code matches the architecture design (data models, API patterns, conventions)\n"
            f"4. Output a JSON code block with your review following the format "
            f"in your system prompt\n"
        )

    def parse_review(self, output: str) -> ReviewResult:
        """Extract structured review from output."""
        _fail = ReviewResult(
            approved=False, score=0, issues=[],
            summary="Could not parse review output — manual review required",
        )

        from autoforge.engine.utils import extract_json_from_text
        schema = {
            "type": "object",
            "required": ["approved", "score", "issues"],
            "properties": {
                "approved": {"type": "boolean"},
                "score": {"type": "number"},
                "issues": {"type": "array", "items": {"type": "object"}},
                "summary": {"type": "string"},
            },
        }
        try:
            data = extract_json_from_text(output, schema=schema, strict=True)
        except ValueError as e:
            logger.warning("Could not find JSON in review output: %s", e)
            return _fail

        return ReviewResult(
            approved=data.get("approved", False),
            score=data.get("score", 0),
            issues=data.get("issues", []),
            summary=data.get("summary", ""),
        )
