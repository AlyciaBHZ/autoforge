"""Architect Agent — architecture design and task decomposition.

Enhanced with:
  - GitHub search for discovering open-source solutions
  - Multi-architecture generation with diversity filtering
  - Search tree integration for exploring multiple designs
"""

from __future__ import annotations

import json
import logging
import os
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

            # GitHub tools for discovering open-source components
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
                    "Search GitHub for open-source libraries and frameworks. "
                    "Use to find existing solutions for project components instead of "
                    "building from scratch. Check stars, activity, and license."
                ),
                input_schema=SEARCH_GITHUB_TOOL_SCHEMA,
                handler=partial(handle_search_github, github_token=github_token),
            ))
            self._tools.append(ToolDefinition(
                name="inspect_repo",
                description=(
                    "Inspect a GitHub repo's README and structure to evaluate "
                    "whether it's suitable for the project. Use after search_github."
                ),
                input_schema=INSPECT_REPO_TOOL_SCHEMA,
                handler=partial(handle_inspect_repo, github_token=github_token),
            ))

    async def _handle_read_template(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data["path"]
        try:
            full_path = self.validate_path(rel_path, self.templates_dir)
        except ValueError:
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

    async def generate_diverse_architectures(
        self,
        spec: dict[str, Any],
        num_candidates: int = 3,
    ) -> list[dict[str, str]]:
        """Generate multiple distinct architecture proposals for diversity.

        Uses the search tree module to create and evaluate candidates.
        This enables exploring fundamentally different approaches rather than
        committing to the first one the LLM generates.

        Inspired by SWE-Search (ICLR 2025) diversity mechanism:
        - Generate candidates with varied strategies
        - Filter out semantically similar proposals
        - Evaluate remaining candidates

        Returns list of {"description": ..., "strategy": ...} candidate dicts.
        """
        from autoforge.engine.search_tree import generate_candidates

        context = json.dumps(spec, indent=2, ensure_ascii=False)
        task_desc = (
            f"Design the architecture for project '{spec.get('project_name', '')}'.\n"
            f"Tech stack: {json.dumps(spec.get('tech_stack', {}))}\n"
            f"Modules: {[m.get('name', '') for m in spec.get('modules', [])]}"
        )

        candidates = await generate_candidates(
            llm=self.llm,
            task_description=task_desc,
            context=context,
            num_candidates=num_candidates,
            system_prompt=(
                "You are a software architect. Generate genuinely different "
                "architectural approaches — different libraries, patterns, or "
                "structural decisions. NOT just variations of the same idea."
            ),
        )

        # Diversity filter: remove candidates that are too similar
        if len(candidates) > 1:
            candidates = self._filter_similar(candidates)

        logger.info(f"[Architect] Generated {len(candidates)} diverse architecture candidates")
        return candidates

    @staticmethod
    def _filter_similar(
        candidates: list[dict[str, str]],
        threshold: float = 0.7,
    ) -> list[dict[str, str]]:
        """Remove candidates with high textual overlap (simple Jaccard similarity).

        A more sophisticated version would use embeddings, but word-level
        Jaccard is fast and sufficient for catching obviously duplicate proposals.
        """
        filtered = [candidates[0]]
        for cand in candidates[1:]:
            is_unique = True
            cand_words = set(cand.get("strategy", "").lower().split())
            for kept in filtered:
                kept_words = set(kept.get("strategy", "").lower().split())
                if not cand_words or not kept_words:
                    continue
                intersection = len(cand_words & kept_words)
                union = len(cand_words | kept_words)
                similarity = intersection / union if union > 0 else 0
                if similarity > threshold:
                    is_unique = False
                    logger.debug(f"[Architect] Filtered similar candidate "
                                 f"(similarity={similarity:.2f})")
                    break
            if is_unique:
                filtered.append(cand)
        return filtered
