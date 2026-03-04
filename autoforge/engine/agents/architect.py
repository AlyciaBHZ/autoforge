"""Architect Agent — architecture design and task decomposition."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import autoforge
from autoforge.engine.agent_base import AgentBase, ToolDefinition
from autoforge.engine.llm_router import TaskComplexity

logger = logging.getLogger(__name__)


class ArchitectAgent(AgentBase):
    """Designs project architecture and creates the task DAG.

    Takes the spec from the Director and produces:
    1. Architecture documentation
    2. A list of tasks with dependencies for the build phase
    """

    ROLE = "architect"
    COMPLEXITY = TaskComplexity.HIGH  # Uses Opus for architectural decisions

    def __init__(self, config, llm, templates_dir: Path | None = None) -> None:
        self.templates_dir = templates_dir or autoforge.DATA_DIR / "templates"
        super().__init__(config, llm)

    def _register_tools(self) -> None:
        """Architect can read template files and search the web for reference."""
        self._tools = [
            ToolDefinition(
                name="read_template",
                description=(
                    "Read a file from the project templates directory. "
                    "Use this to examine existing template structures for reference."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path within the templates directory",
                        },
                    },
                    "required": ["path"],
                },
                handler=self._handle_read_template,
            ),
        ]

        # Add web tools if enabled
        if getattr(self.config, "web_tools_enabled", True):
            from functools import partial

            from autoforge.engine.tools.web import (
                FETCH_URL_TOOL_SCHEMA,
                SEARCH_WEB_TOOL_SCHEMA,
                handle_fetch_url,
                handle_search_web,
            )

            self._tools.append(ToolDefinition(
                name="search_web",
                description=(
                    "Search the web for library documentation, API specs, "
                    "and architectural patterns. Use to validate design decisions."
                ),
                input_schema=SEARCH_WEB_TOOL_SCHEMA,
                handler=partial(
                    handle_search_web,
                    backend=self.config.search_backend,
                    api_key=self.config.search_api_key,
                ),
            ))
            self._tools.append(ToolDefinition(
                name="fetch_url",
                description=(
                    "Fetch a web page and return its text content. "
                    "Use to read library docs, API references, or example code."
                ),
                input_schema=FETCH_URL_TOOL_SCHEMA,
                handler=handle_fetch_url,
            ))

    async def _handle_read_template(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data["path"]
        full_path = (self.templates_dir / rel_path).resolve()
        if not str(full_path).startswith(str(self.templates_dir.resolve())):
            return json.dumps({"error": "Path traversal not allowed"})
        if not full_path.exists():
            return json.dumps({"error": f"Template not found: {rel_path}"})
        return full_path.read_text(encoding="utf-8")

    def build_prompt(self, context: dict[str, Any]) -> str:
        spec = context.get("spec", {})
        return (
            f"Design the architecture and create a task breakdown for this project.\n\n"
            f"## Project Specification\n\n"
            f"```json\n{json.dumps(spec, indent=2, ensure_ascii=False)}\n```\n\n"
            f"## Instructions\n\n"
            f"1. Design the directory structure and file layout\n"
            f"2. Define key data models and API interfaces\n"
            f"3. Create a task list where each task:\n"
            f"   - Has a unique ID (TASK-001, TASK-002, etc.)\n"
            f"   - Owns specific files (no overlap between tasks)\n"
            f"   - Lists dependencies on other tasks\n"
            f"   - Has clear acceptance criteria\n"
            f"4. The first task should be project scaffolding (package.json, config)\n"
            f"5. Ensure tasks form a valid DAG (no cycles)\n\n"
            f"Output a single JSON code block following the format in your system prompt."
        )

    def parse_architecture(self, output: str) -> dict[str, Any]:
        """Extract architecture and tasks from the Architect's output."""
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", output, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())

        start = output.find("{")
        end = output.rfind("}")
        if start != -1 and end != -1:
            return json.loads(output[start : end + 1])

        raise ValueError("Could not extract architecture from Architect output")
