"""Scanner Agent — codebase analysis and spec generation for existing projects."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoforge.engine.agent_base import AgentBase, ToolDefinition
from autoforge.engine.llm_router import TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Structured scan output."""

    spec: dict[str, Any] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    completeness: int = 0  # 0-100%
    summary: str = ""


class ScannerAgent(AgentBase):
    """Analyzes existing codebases and produces spec.json-compatible output.

    Examines project structure, detects tech stack, identifies modules,
    and finds gaps/TODOs. Uses Opus for deep analysis.
    """

    ROLE = "scanner"
    COMPLEXITY = TaskComplexity.HIGH  # Uses Opus for complex analysis

    def __init__(self, config, llm, working_dir: Path) -> None:
        self.working_dir = working_dir
        super().__init__(config, llm)

    def _register_tools(self) -> None:
        """Scanner has read-only tools plus safe inspection commands."""
        self._tools = [
            ToolDefinition(
                name="read_file",
                description="Read a file from the project (read-only).",
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
                description="List all files in a directory recursively.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative directory path (default: '.')",
                        },
                    },
                },
                handler=self._handle_list_files,
            ),
            ToolDefinition(
                name="run_command",
                description=(
                    "Run a safe read-only command (e.g., 'wc -l', 'git log --oneline -10', "
                    "'find . -name *.test.*'). Only inspection commands allowed."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute (read-only)",
                        },
                    },
                    "required": ["command"],
                },
                handler=self._handle_run_command,
            ),
        ]

        # Add grep_search for efficient codebase analysis
        from functools import partial

        from autoforge.engine.tools.search import (
            GREP_SEARCH_TOOL_SCHEMA,
            handle_grep_search,
        )

        self._tools.append(ToolDefinition(
            name="grep_search",
            description=(
                "Search project files for a pattern (regex). "
                "More efficient than reading files one by one. "
                "Use to find function definitions, imports, TODOs, and patterns."
            ),
            input_schema=GREP_SEARCH_TOOL_SCHEMA,
            handler=partial(handle_grep_search, working_dir=self.working_dir),
        ))

    async def _handle_read_file(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data["path"]
        try:
            full_path = self.validate_path(rel_path, self.working_dir)
        except ValueError:
            return json.dumps({"error": "Path traversal not allowed"})
        if not full_path.exists():
            return json.dumps({"error": f"File not found: {rel_path}"})
        try:
            content = full_path.read_text(encoding="utf-8")
            # Truncate very large files
            if len(content) > 50000:
                content = content[:50000] + "\n... (truncated)"
            return content
        except UnicodeDecodeError:
            return json.dumps({"error": f"Binary file: {rel_path}"})

    async def _handle_list_files(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data.get("path", ".")
        try:
            full_path = self.validate_path(rel_path, self.working_dir)
        except ValueError:
            return json.dumps({"error": "Path traversal not allowed"})
        if not full_path.is_dir():
            return json.dumps({"error": f"Not a directory: {rel_path}"})
        base_resolved = self.working_dir.resolve()
        files = []
        for p in sorted(full_path.rglob("*")):
            if p.is_file() and ".git" not in p.parts and "node_modules" not in p.parts:
                # Prevent symlinks from escaping the workspace
                if not p.resolve().is_relative_to(base_resolved):
                    continue
                try:
                    files.append(str(p.relative_to(self.working_dir)))
                except ValueError:
                    continue
        return json.dumps(files[:500])  # Limit to 500 files

    # Read-only commands the scanner is allowed to execute
    _ALLOWED_COMMANDS = frozenset({
        "ls", "cat", "head", "tail", "wc", "find", "tree", "file",
        "grep", "rg", "ag", "awk", "sed", "sort", "uniq", "diff",
        "stat", "du", "git", "python", "node", "go", "cargo",
    })

    async def _handle_run_command(self, input_data: dict[str, Any]) -> str:
        """Execute safe read-only commands for analysis.

        Uses an allowlist of safe commands instead of a bypassable blocklist.
        """
        import asyncio
        import shlex

        command = input_data["command"]

        # Allowlist: only permit known-safe read-only commands
        try:
            parts = shlex.split(command)
        except ValueError:
            return json.dumps({"error": "Invalid command syntax"})
        if not parts:
            return json.dumps({"error": "Empty command"})

        # Extract the base command name (strip path prefix)
        base_cmd = parts[0].rsplit("/", 1)[-1]
        if base_cmd not in self._ALLOWED_COMMANDS:
            return json.dumps({
                "error": f"Command '{base_cmd}' not in scanner allowlist. "
                f"Allowed: {', '.join(sorted(self._ALLOWED_COMMANDS))}"
            })

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.working_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            return json.dumps({
                "exit_code": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace")[:5000],
                "stderr": stderr.decode("utf-8", errors="replace")[:1000],
            })
        except asyncio.TimeoutError:
            return json.dumps({"error": "Command timed out (15s limit)"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def build_prompt(self, context: dict[str, Any]) -> str:
        project_path = context.get("project_path", str(self.working_dir))
        return (
            f"Analyze the existing project at: {project_path}\n\n"
            f"## Instructions\n"
            f"1. List all files in the project root\n"
            f"2. Read key files: package.json, requirements.txt, go.mod, Cargo.toml, "
            f"pyproject.toml, build.gradle, pubspec.yaml, etc.\n"
            f"3. Examine the directory structure to identify logical modules\n"
            f"4. Read representative source files from each module\n"
            f"5. Look for TODOs, FIXMEs, and incomplete features\n"
            f"6. Assess test coverage (presence of test files)\n\n"
            f"## Output\n"
            f"Output a single JSON code block with this structure:\n"
            f"```json\n"
            f'{{\n'
            f'  "project_name": "detected-name",\n'
            f'  "description": "What this project does",\n'
            f'  "tech_stack": {{\n'
            f'    "framework": "detected framework",\n'
            f'    "language": "primary language",\n'
            f'    "database": "database if any",\n'
            f'    "styling": "CSS framework if any",\n'
            f'    "runtime": "runtime environment"\n'
            f'  }},\n'
            f'  "project_type": "web-app|api-server|cli-tool|static-site|mobile-scaffold|desktop-scaffold|library",\n'
            f'  "modules": [\n'
            f'    {{\n'
            f'      "name": "module-name",\n'
            f'      "description": "What this module does",\n'
            f'      "files": ["src/path/to/file1.ts"],\n'
            f'      "dependencies": ["other-module"]\n'
            f'    }}\n'
            f'  ],\n'
            f'  "gaps": ["Missing: unit tests", "TODO: auth middleware", ...],\n'
            f'  "completeness": 75,\n'
            f'  "excluded": []\n'
            f'}}\n'
            f"```\n"
        )

    def parse_scan(self, output: str) -> ScanResult:
        """Extract structured scan result from agent output."""
        from autoforge.engine.utils import extract_json_from_text
        schema = {
            "type": "object",
            "required": ["spec", "gaps", "completeness"],
            "properties": {
                "spec": {"type": "object"},
                "gaps": {"type": "array", "items": {"type": "string"}},
                "completeness": {"type": "number"},
                "description": {"type": "string"},
                "excluded": {"type": "array", "items": {"type": "string"}},
            },
        }
        try:
            data = extract_json_from_text(output, schema=schema, strict=True)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("Could not parse scanner output: %s", e)
            return ScanResult(summary="Failed to parse scan results")

        return ScanResult(
            spec=data,
            gaps=data.get("gaps", []),
            completeness=data.get("completeness", 50),
            summary=data.get("description", ""),
        )
