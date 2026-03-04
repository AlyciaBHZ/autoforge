"""Code search tool — grep_search for agents.

Provides a cross-platform grep-like search over project files.
Uses system grep on POSIX, falls back to pure-Python on Windows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Limits to prevent runaway output
_MAX_RESULTS = 50
_MAX_LINE_LENGTH = 200
_CONTEXT_LINES = 2  # Lines of context above/below each match

# Directories to always skip
_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "coverage", ".eggs",
}

# Binary file extensions to skip
_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".pyo", ".class",
    ".pdf", ".doc", ".docx",
    ".db", ".sqlite", ".sqlite3",
}


async def handle_grep_search(
    input_data: dict[str, Any],
    working_dir: Path,
) -> str:
    """Search project files for a pattern.

    Input: {
        "pattern": "def.*test",
        "path": "src/",          # optional: subdirectory to search
        "file_glob": "*.py",     # optional: file pattern filter
        "max_results": 30        # optional: limit results (default 50)
    }

    Output: JSON array of matches with context.
    """
    pattern = input_data.get("pattern", "").strip()
    if not pattern:
        return json.dumps({"error": "Missing required parameter: pattern"})

    rel_path = input_data.get("path", ".")
    file_glob = input_data.get("file_glob", "")
    max_results = min(input_data.get("max_results", _MAX_RESULTS), _MAX_RESULTS)

    # Resolve and validate search path
    search_dir = (working_dir / rel_path).resolve()
    if not search_dir.is_relative_to(working_dir.resolve()):
        return json.dumps({"error": "Path traversal not allowed"})
    if not search_dir.is_dir():
        return json.dumps({"error": f"Not a directory: {rel_path}"})

    # Try system grep first (faster), fall back to Python
    if sys.platform != "win32":
        results = await _grep_system(pattern, search_dir, working_dir, file_glob, max_results)
    else:
        results = await _grep_python(pattern, search_dir, working_dir, file_glob, max_results)

    return json.dumps({
        "pattern": pattern,
        "total_matches": len(results),
        "matches": results,
    })


async def _grep_system(
    pattern: str,
    search_dir: Path,
    working_dir: Path,
    file_glob: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """Search using system grep (POSIX)."""
    cmd = [
        "grep", "-rn",
        "--color=never",
        f"--max-count={max_results}",
    ]

    # Add context lines
    cmd.append(f"-C{_CONTEXT_LINES}")

    # Add file glob filter
    if file_glob:
        cmd.extend(["--include", file_glob])

    # Exclude common directories
    for skip_dir in _SKIP_DIRS:
        cmd.extend(["--exclude-dir", skip_dir])

    # Exclude binary files
    cmd.append("--binary-files=without-match")

    cmd.extend([pattern, str(search_dir)])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    except FileNotFoundError:
        # grep not found — fall back to Python
        return await _grep_python(pattern, search_dir, working_dir, file_glob, max_results)
    except asyncio.TimeoutError:
        return [{"error": "Search timed out (15s limit)"}]

    if proc.returncode not in (0, 1):  # 1 = no matches
        err = stderr.decode("utf-8", errors="replace").strip()[:200]
        if err:
            return [{"error": f"grep error: {err}"}]

    output = stdout.decode("utf-8", errors="replace")
    return _parse_grep_output(output, working_dir, max_results)


def _parse_grep_output(
    output: str,
    working_dir: Path,
    max_results: int,
) -> list[dict[str, Any]]:
    """Parse grep -n output into structured results."""
    results: list[dict[str, Any]] = []
    for line in output.splitlines():
        if len(results) >= max_results:
            break

        # Match format: file:line_number:content
        match = re.match(r"^(.+?):(\d+)[:|-](.*)$", line)
        if match:
            filepath, line_no, content = match.groups()
            try:
                rel = str(Path(filepath).relative_to(working_dir))
            except ValueError:
                rel = filepath

            # Truncate long lines
            if len(content) > _MAX_LINE_LENGTH:
                content = content[:_MAX_LINE_LENGTH] + "..."

            results.append({
                "file": rel,
                "line": int(line_no),
                "content": content.strip(),
            })

    return results


async def _grep_python(
    pattern: str,
    search_dir: Path,
    working_dir: Path,
    file_glob: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """Pure-Python fallback search (for Windows or when grep is unavailable)."""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return [{"error": f"Invalid regex pattern: {e}"}]

    results: list[dict[str, Any]] = []

    # Determine file patterns to search
    if file_glob:
        file_iter = search_dir.rglob(file_glob)
    else:
        file_iter = search_dir.rglob("*")

    for filepath in file_iter:
        if len(results) >= max_results:
            break

        # Skip directories and binary files
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() in _BINARY_EXTENSIONS:
            continue
        if any(part in _SKIP_DIRS for part in filepath.parts):
            continue

        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            continue

        for i, line in enumerate(content.splitlines(), 1):
            if len(results) >= max_results:
                break

            if regex.search(line):
                try:
                    rel = str(filepath.relative_to(working_dir))
                except ValueError:
                    rel = str(filepath)

                # Truncate long lines
                display = line.strip()
                if len(display) > _MAX_LINE_LENGTH:
                    display = display[:_MAX_LINE_LENGTH] + "..."

                results.append({
                    "file": rel,
                    "line": i,
                    "content": display,
                })

    return results


# ── Tool definition (for agent registration) ──────────────────────

GREP_SEARCH_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": (
                "Regular expression pattern to search for in project files. "
                "Examples: 'def.*test', 'import.*React', 'TODO|FIXME', 'class User'."
            ),
        },
        "path": {
            "type": "string",
            "description": (
                "Subdirectory to search in, relative to project root. "
                "Default: search entire project."
            ),
        },
        "file_glob": {
            "type": "string",
            "description": (
                "File pattern filter. Examples: '*.py', '*.ts', '*.js'. "
                "Default: search all text files."
            ),
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of matching lines to return (default: 50).",
        },
    },
    "required": ["pattern"],
}
