"""Director Agent — requirement analysis and task planning."""

from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from functools import partial
from typing import Any

from autoforge.engine.agent_base import AgentBase, ToolDefinition
from autoforge.engine.llm_router import TaskComplexity

logger = logging.getLogger(__name__)


class DirectorAgent(AgentBase):
    """Analyzes user requirements and produces structured project specifications.

    The Director is the first agent to run. It takes natural language input
    and outputs a JSON spec that other agents consume.
    """

    ROLE = "director"
    COMPLEXITY = TaskComplexity.HIGH  # Uses Opus for deep understanding

    _OUTPUT_SCHEMA_SPEC: dict[str, Any] = {
        "type": "object",
        "required": ["project_name", "modules"],
        "properties": {
            "project_name": {"type": "string"},
            "description": {"type": "string"},
            "modules": {"type": "array", "items": {"type": "object"}},
            "tech_stack": {"type": "object"},
            "build_contract": {"type": "object"},
        },
    }

    def _resolve_output_schema(self, context: dict[str, Any]) -> dict[str, Any] | None:
        return deepcopy(self._OUTPUT_SCHEMA_SPEC)

    def _register_tools(self) -> None:
        """Director has web tools and GitHub tools for researching frameworks."""
        self._tools = []

        # Add web tools if enabled
        if getattr(self.config, "web_tools_enabled", True):
            from autoforge.engine.tools.web import (
                FETCH_URL_TOOL_SCHEMA,
                SEARCH_WEB_TOOL_SCHEMA,
                handle_fetch_url,
                handle_search_web,
            )

            self._tools.append(ToolDefinition(
                name="search_web",
                description=(
                    "Search the web for information about frameworks, libraries, "
                    "and best practices. Use to research the latest tech options."
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
                    "Use to read documentation, API references, or framework guides."
                ),
                input_schema=FETCH_URL_TOOL_SCHEMA,
                handler=handle_fetch_url,
            ))

            # GitHub search tools for discovering open-source solutions
            from autoforge.engine.tools.github_search import (
                SEARCH_GITHUB_TOOL_SCHEMA,
                INSPECT_REPO_TOOL_SCHEMA,
                handle_search_github,
                handle_inspect_repo,
            )

            github_token = os.environ.get("GITHUB_TOKEN", "")

            self._tools.append(ToolDefinition(
                name="search_github",
                description=(
                    "Search GitHub for open-source repositories. Use to discover "
                    "existing libraries and frameworks that solve sub-problems. "
                    "Prefer well-maintained repos (high stars, recent updates)."
                ),
                input_schema=SEARCH_GITHUB_TOOL_SCHEMA,
                handler=partial(handle_search_github, github_token=github_token),
            ))
            self._tools.append(ToolDefinition(
                name="inspect_repo",
                description=(
                    "Inspect a GitHub repository's README, file structure, and "
                    "dependencies. Use after search_github to evaluate candidates."
                ),
                input_schema=INSPECT_REPO_TOOL_SCHEMA,
                handler=partial(handle_inspect_repo, github_token=github_token),
            ))

    def build_prompt(self, context: dict[str, Any]) -> str:
        description = context.get("project_description", "")
        return (
            f"Analyze the following project requirements and produce a detailed "
            f"specification.\n\n"
            f"## Project Description\n\n{description}\n\n"
            f"## Instructions\n\n"
            f"1. Understand what the user wants to build\n"
            f"2. Choose an appropriate technology stack\n"
            f"3. Break the project into independent modules\n"
            f"4. Define what files each module needs\n"
            f"5. Identify module dependencies\n"
            f"6. Explicitly state what is OUT of scope for MVP\n\n"
            f"Output a single JSON code block with the specification. "
            f"Follow the format defined in your system prompt exactly."
        )

    def parse_spec(self, output: str) -> dict[str, Any]:
        """Extract JSON spec from the Director's output text."""
        from autoforge.engine.utils import extract_json_from_text
        try:
            return extract_json_from_text(output, schema=self._OUTPUT_SCHEMA_SPEC, strict=True)
        except ValueError as e:
            # Lenient fallback to avoid a single malformed brace aborting the run.
            # We still enforce mandatory fields below and keep behavior deterministic.
            try:
                payload = extract_json_from_text(output, schema=self._OUTPUT_SCHEMA_SPEC, strict=False)
            except ValueError as e2:
                raise ValueError(
                    f"Could not extract JSON spec from Director output: {e}; fallback: {e2}"
                ) from e

            if not isinstance(payload, dict):
                raise ValueError(f"Could not extract JSON spec from Director output: {e}") from e

            if "project_name" not in payload or not payload["project_name"]:
                payload["project_name"] = "autoforge_project"
            else:
                payload["project_name"] = str(payload["project_name"]).strip() or "autoforge_project"
            if not isinstance(payload.get("modules"), list):
                payload["modules"] = []
            if not payload["modules"]:
                payload["modules"] = [{
                    "name": "core",
                    "description": f"Core module for {payload['project_name']}",
                    "files": [],
                    "dependencies": [],
                }]

            tech_stack = payload.get("tech_stack")
            if not isinstance(tech_stack, dict):
                payload["tech_stack"] = {}

            build_contract = payload.get("build_contract")
            if not isinstance(build_contract, dict):
                payload["build_contract"] = {}

            return payload


class DirectorFixAgent(AgentBase):
    """Creates fix tasks when tests fail.

    A specialized Director mode that analyzes test failures and produces
    targeted fix tasks for the Builder.
    """

    ROLE = "director_fix"
    COMPLEXITY = TaskComplexity.HIGH

    _OUTPUT_SCHEMA_FIX_TASK: dict[str, Any] = {
        "type": "object",
        "required": ["id", "description", "files"],
        "properties": {
            "id": {"type": "string"},
            "description": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "owner": {"type": "string"},
            "fix_strategy": {"type": "string"},
        },
    }

    def _resolve_output_schema(self, context: dict[str, Any]) -> dict[str, Any] | None:
        return deepcopy(self._OUTPUT_SCHEMA_FIX_TASK)

    def _register_tools(self) -> None:
        self._tools = []

    def _load_system_prompt(self) -> None:
        """Load system prompt, falling back to director.md if no dedicated file."""
        prompt_path = self.config.constitution_dir / "agents" / f"{self.ROLE}.md"
        if prompt_path.exists():
            self._system_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            # Fall back to base director prompt with fix-specific addendum
            base_path = self.config.constitution_dir / "agents" / "director.md"
            if base_path.exists():
                self._system_prompt = base_path.read_text(encoding="utf-8")
            else:
                self._system_prompt = f"You are the {self.ROLE} agent in the AutoForge system."
            self._system_prompt += (
                "\n\nYou are in FIX mode: analyze test failures and "
                "produce targeted fix tasks for the Builder agent."
            )

    def build_prompt(self, context: dict[str, Any]) -> str:
        failure = context.get("failure", {})
        spec = context.get("spec", {})
        return (
            f"A test has failed in the project '{spec.get('project_name', '')}'.\n\n"
            f"## Failure Details\n\n"
            f"Step: {failure.get('step', 'unknown')}\n"
            f"Command: {failure.get('command', 'unknown')}\n"
            f"Error:\n```\n{failure.get('stderr', failure.get('error', 'unknown'))}\n```\n\n"
            f"## Instructions\n\n"
            f"Analyze this failure and produce a fix task as a JSON code block:\n\n"
            f"```json\n"
            f'{{\n'
            f'  "id": "FIX-XXX",\n'
            f'  "description": "What needs to be fixed",\n'
            f'  "owner": "builder",\n'
            f'  "files": ["path/to/file/to/fix"],\n'
            f'  "fix_strategy": "Explanation of how to fix it"\n'
            f'}}\n'
            f"```"
        )

    def parse_fix_task(self, output: str) -> dict[str, Any]:
        """Extract fix task from output."""
        from autoforge.engine.utils import extract_json_from_text
        strict_schema = {
            "type": "object",
            "required": ["id", "description", "files"],
            "properties": {
                "id": {"type": "string"},
                "description": {"type": "string"},
                "files": {"type": "array", "items": {"type": "string"}},
                "owner": {"type": "string"},
            },
        }
        try:
            result = extract_json_from_text(output, schema=strict_schema, strict=True)
            result.setdefault("files", [])
            return result
        except ValueError as e:
            # Backward-compatible fallback: older prompts may omit `files`.
            fallback_schema = {
                "type": "object",
                "required": ["id", "description"],
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "owner": {"type": "string"},
                },
            }
            try:
                result = extract_json_from_text(output, schema=fallback_schema, strict=True)
                result.setdefault("files", [])
                return result
            except ValueError:
                # Last-chance fallback for soft JSON drift.
                payload = extract_json_from_text(output, schema=fallback_schema, strict=False)
                task_id = payload.get("id") or "FIX-RETRY"
                description = payload.get("description") or "Resolve reported test failure"
                files = payload.get("files", [])
                if not isinstance(files, list):
                    files = []
                if not isinstance(task_id, str) or not task_id:
                    task_id = "FIX-RETRY"
                return {
                    "id": task_id,
                    "description": str(description),
                    "owner": payload.get("owner", "builder")
                    if isinstance(payload.get("owner"), str)
                    else "builder",
                    "files": files,
                    "fix_strategy": payload.get("fix_strategy", ""),
                }
                # If extraction above succeeds, we always return a normalized task.
