"""Gardener Agent — code refactoring and quality improvement."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from autoforge.engine.agent_base import AgentBase, FileToolsMixin, ToolDefinition
from autoforge.engine.llm_router import TaskComplexity

logger = logging.getLogger(__name__)


class GardenerAgent(FileToolsMixin, AgentBase):
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
        from autoforge.engine.utils import extract_json_from_text
        try:
            return extract_json_from_text(output)
        except ValueError as e:
            logger.warning("Could not extract JSON from gardener output: %s", e)
            return {"changes_made": [], "summary": output[:500]}
