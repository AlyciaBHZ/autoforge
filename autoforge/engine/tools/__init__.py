"""AutoForge agent tools — unified registry and context.

Tools are capabilities that agents can invoke during their agentic loop.
Each tool has a name, description, JSON schema for inputs, and an async handler.

Usage:
    from autoforge.engine.tools import ToolContext, ToolRegistry, get_default_registry

    ctx = ToolContext(working_dir=project_dir, config=config, sandbox=sandbox)
    registry = get_default_registry()
    tools = registry.get_tools_for("builder", ctx)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Shared context passed to all tool handlers.

    Provides a uniform interface so handlers don't need ad-hoc kwargs.
    """

    working_dir: Path
    config: Any = None  # ForgeConfig
    sandbox: Any = None  # SandboxBase or None
    agent_id: str = ""

    def validate_path(self, rel_path: str) -> Path:
        """Validate and resolve a path, preventing traversal attacks."""
        resolved = (self.working_dir / rel_path).resolve()
        base = self.working_dir.resolve()
        if not resolved.is_relative_to(base):
            raise ValueError(f"Path traversal detected: {rel_path}")
        return resolved


@dataclass
class ToolSpec:
    """Tool specification for the registry.

    Agents select tools by role. A tool can be available to multiple roles.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler_factory: Callable[[ToolContext], Callable[..., Coroutine[Any, Any, str]]]
    roles: set[str] = field(default_factory=lambda: {"builder"})
    requires_web: bool = False


class ToolRegistry:
    """Central registry for agent tools.

    Provides a single place to register tools and query them by agent role.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def get_tools_for(
        self,
        role: str,
        ctx: ToolContext,
        web_enabled: bool = True,
    ) -> list[dict[str, Any]]:
        """Get tool definitions with bound handlers for an agent role."""
        tools = []
        for spec in self._specs.values():
            if role not in spec.roles:
                continue
            if spec.requires_web and not web_enabled:
                continue
            handler = spec.handler_factory(ctx)
            tools.append({
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "handler": handler,
            })
        return tools

    def list_tools(self, role: str | None = None) -> list[str]:
        if role is None:
            return list(self._specs.keys())
        return [n for n, s in self._specs.items() if role in s.roles]


_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """Get or create the default global tool registry with built-in tools."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
        _register_builtin_tools(_default_registry)
    return _default_registry


def _register_builtin_tools(registry: ToolRegistry) -> None:
    """Register all built-in tools."""
    from autoforge.engine.tools.search import GREP_SEARCH_TOOL_SCHEMA, handle_grep_search
    from autoforge.engine.tools.web import FETCH_URL_TOOL_SCHEMA, handle_fetch_url

    all_read_roles = {"builder", "reviewer", "tester", "gardener", "scanner", "architect"}

    registry.register(ToolSpec(
        name="grep_search",
        description="Search project files for a regex pattern.",
        input_schema=GREP_SEARCH_TOOL_SCHEMA,
        handler_factory=lambda ctx: lambda input_data: handle_grep_search(
            input_data, working_dir=ctx.working_dir,
        ),
        roles=all_read_roles,
    ))

    registry.register(ToolSpec(
        name="fetch_url",
        description="Fetch a web page and return its text content.",
        input_schema=FETCH_URL_TOOL_SCHEMA,
        handler_factory=lambda ctx: handle_fetch_url,
        roles={"builder", "architect", "director"},
        requires_web=True,
    ))

    try:
        from autoforge.engine.tools.web import SEARCH_WEB_TOOL_SCHEMA, handle_search_web

        registry.register(ToolSpec(
            name="search_web",
            description="Search the web for information.",
            input_schema=SEARCH_WEB_TOOL_SCHEMA,
            handler_factory=lambda ctx: lambda input_data: handle_search_web(
                input_data,
                backend=getattr(ctx.config, "search_backend", "duckduckgo"),
                api_key=getattr(ctx.config, "search_api_key", ""),
            ),
            roles={"builder", "architect", "director"},
            requires_web=True,
        ))
    except ImportError:
        pass

    try:
        from autoforge.engine.tools.github_search import (
            SEARCH_GITHUB_TOOL_SCHEMA,
            handle_search_github,
        )

        registry.register(ToolSpec(
            name="search_github",
            description="Search GitHub for libraries and code.",
            input_schema=SEARCH_GITHUB_TOOL_SCHEMA,
            handler_factory=lambda ctx: lambda input_data: handle_search_github(
                input_data,
                github_token=getattr(ctx.config, "github_token", ""),
            ),
            roles={"builder", "architect"},
            requires_web=True,
        ))
    except ImportError:
        pass


__all__ = [
    "ToolContext",
    "ToolSpec",
    "ToolRegistry",
    "get_default_registry",
]
