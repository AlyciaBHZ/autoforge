"""GitHub search tools — repository and code discovery for agents.

Provides tools for searching GitHub repositories and inspecting their
contents, enabling agents to discover and evaluate open-source solutions
during project planning and implementation.

Dependencies: httpx (already required by web.py)
"""

from __future__ import annotations

import asyncio
import json
import logging
from base64 import b64decode
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

_GITHUB_API = "https://api.github.com"
_TIMEOUT = 15  # seconds
_USER_AGENT = "AutoForge/2.0 (AI Development Tool)"
_MAX_README_CHARS = 6000  # Truncate README to keep context manageable
_MAX_TREE_FILES = 200  # Max files to list from repo tree


# ── Handlers ───────────────────────────────────────────────────────


async def handle_search_github(
    input_data: dict[str, Any],
    github_token: str = "",
) -> str:
    """Search GitHub repositories with quality filtering.

    Input: {
        "query": "markdown to slides python",
        "language": "python",       # optional
        "sort": "stars",            # optional: stars, forks, updated
        "min_stars": 50,            # optional: quality threshold
        "per_page": 5               # optional: max 10
    }

    Output: JSON array of repositories with name, description, stars,
            language, last update, license, and URL.
    """
    query = input_data.get("query", "").strip()
    if not query:
        return json.dumps({"error": "Missing required parameter: query"})

    language = input_data.get("language", "")
    sort = input_data.get("sort", "stars")
    min_stars = input_data.get("min_stars", 50)
    per_page = min(input_data.get("per_page", 5), 10)

    # Build GitHub search query
    q_parts = [query]
    if language:
        q_parts.append(f"language:{language}")
    if min_stars > 0:
        q_parts.append(f"stars:>={min_stars}")

    q = " ".join(q_parts)

    try:
        import httpx
    except ImportError:
        return json.dumps({"error": "httpx not installed. Run: pip install httpx"})

    headers = _build_headers(github_token)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers) as client:
            resp = await client.get(
                f"{_GITHUB_API}/search/repositories",
                params={"q": q, "sort": sort, "order": "desc", "per_page": per_page},
            )
            resp.raise_for_status()
            data = resp.json()

        repos = []
        for item in data.get("items", []):
            license_info = item.get("license")
            repos.append({
                "full_name": item.get("full_name", ""),
                "description": (item.get("description") or "")[:200],
                "stars": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
                "language": item.get("language", ""),
                "updated_at": item.get("updated_at", ""),
                "license": license_info.get("spdx_id", "unknown") if license_info else "none",
                "url": item.get("html_url", ""),
                "topics": item.get("topics", [])[:5],
                "open_issues": item.get("open_issues_count", 0),
                "archived": item.get("archived", False),
            })

        return json.dumps({
            "query": q,
            "total_count": data.get("total_count", 0),
            "repositories": repos,
        })

    except Exception as e:
        logger.debug(f"search_github error for '{q}': {e}")
        return json.dumps({"error": str(e), "query": q})


async def handle_inspect_repo(
    input_data: dict[str, Any],
    github_token: str = "",
) -> str:
    """Inspect a GitHub repository: README, file tree, and dependencies.

    Input: {
        "owner": "expressjs",
        "repo": "express",
        "include_readme": true,    # optional, default true
        "include_tree": true       # optional, default true
    }

    Output: JSON with README content, file tree, and detected package info.
    """
    owner = input_data.get("owner", "").strip()
    repo = input_data.get("repo", "").strip()
    if not owner or not repo:
        return json.dumps({"error": "Missing required parameters: owner and repo"})

    include_readme = input_data.get("include_readme", True)
    include_tree = input_data.get("include_tree", True)

    try:
        import httpx
    except ImportError:
        return json.dumps({"error": "httpx not installed. Run: pip install httpx"})

    headers = _build_headers(github_token)
    result: dict[str, Any] = {"owner": owner, "repo": repo}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers) as client:
            # Fetch repo metadata
            meta_resp = await client.get(f"{_GITHUB_API}/repos/{owner}/{repo}")
            meta_resp.raise_for_status()
            meta = meta_resp.json()

            result["description"] = meta.get("description", "")
            result["stars"] = meta.get("stargazers_count", 0)
            result["language"] = meta.get("language", "")
            result["default_branch"] = meta.get("default_branch", "main")

            # Concurrent fetches
            tasks = []
            if include_readme:
                tasks.append(_fetch_readme(client, owner, repo))
            if include_tree:
                tasks.append(_fetch_tree(client, owner, repo, meta.get("default_branch", "main")))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            idx = 0
            if include_readme:
                readme = results[idx]
                result["readme"] = readme if isinstance(readme, str) else f"Error: {readme}"
                idx += 1
            if include_tree:
                tree = results[idx]
                result["file_tree"] = tree if isinstance(tree, list) else []

    except Exception as e:
        logger.debug(f"inspect_repo error for {owner}/{repo}: {e}")
        return json.dumps({"error": str(e), "owner": owner, "repo": repo})

    return json.dumps(result)


async def handle_search_github_code(
    input_data: dict[str, Any],
    github_token: str = "",
) -> str:
    """Search for code patterns across GitHub repositories.

    Input: {
        "query": "async def create_app",
        "language": "python",           # optional
        "repo": "owner/repo",           # optional: scope to specific repo
        "per_page": 5                   # optional: max 10
    }

    Output: JSON array of code matches with file path, repo, and snippet.
    Note: Requires authentication (GitHub token) for code search.
    """
    query = input_data.get("query", "").strip()
    if not query:
        return json.dumps({"error": "Missing required parameter: query"})

    if not github_token:
        return json.dumps({
            "error": "GitHub code search requires a personal access token. "
                     "Set GITHUB_TOKEN environment variable."
        })

    language = input_data.get("language", "")
    repo = input_data.get("repo", "")
    per_page = min(input_data.get("per_page", 5), 10)

    q_parts = [query]
    if language:
        q_parts.append(f"language:{language}")
    if repo:
        q_parts.append(f"repo:{repo}")
    q = " ".join(q_parts)

    try:
        import httpx
    except ImportError:
        return json.dumps({"error": "httpx not installed"})

    headers = _build_headers(github_token)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers) as client:
            resp = await client.get(
                f"{_GITHUB_API}/search/code",
                params={"q": q, "per_page": per_page},
            )
            resp.raise_for_status()
            data = resp.json()

        matches = []
        for item in data.get("items", []):
            repo_info = item.get("repository", {})
            matches.append({
                "file": item.get("name", ""),
                "path": item.get("path", ""),
                "repo": repo_info.get("full_name", ""),
                "repo_stars": repo_info.get("stargazers_count", 0),
                "url": item.get("html_url", ""),
            })

        return json.dumps({
            "query": q,
            "total_count": data.get("total_count", 0),
            "matches": matches,
        })

    except Exception as e:
        logger.debug(f"search_github_code error for '{q}': {e}")
        return json.dumps({"error": str(e), "query": q})


# ── Internal helpers ───────────────────────────────────────────────


def _build_headers(github_token: str) -> dict[str, str]:
    """Build request headers with optional authentication."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


async def _fetch_readme(client: Any, owner: str, repo: str) -> str:
    """Fetch and decode a repository's README."""
    try:
        resp = await client.get(f"{_GITHUB_API}/repos/{owner}/{repo}/readme")
        if resp.status_code == 404:
            return "(No README found)"
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", "")
        encoding = data.get("encoding", "base64")
        if encoding == "base64" and content:
            text = b64decode(content).decode("utf-8", errors="replace")
        else:
            text = content
        if len(text) > _MAX_README_CHARS:
            text = text[:_MAX_README_CHARS] + "\n\n... (truncated)"
        return text
    except Exception as e:
        return f"(Error fetching README: {e})"


async def _fetch_tree(client: Any, owner: str, repo: str, branch: str) -> list[str]:
    """Fetch the file tree (top-level + one level deep)."""
    try:
        resp = await client.get(
            f"{_GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}",
            params={"recursive": "1"},
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        tree = data.get("tree", [])
        # Return file paths only (skip blobs > 200 entries)
        paths = [
            item["path"] for item in tree
            if item.get("type") in ("blob", "tree")
        ]
        return paths[:_MAX_TREE_FILES]
    except Exception:
        return []


# ── Tool definitions (for agent registration) ──────────────────────

SEARCH_GITHUB_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Search query to find GitHub repositories. Be specific about "
                "functionality needed. Example: 'markdown to slides converter'"
            ),
        },
        "language": {
            "type": "string",
            "description": (
                "Filter by programming language. Example: 'python', 'typescript', 'rust'"
            ),
        },
        "sort": {
            "type": "string",
            "description": "Sort by: 'stars' (default), 'forks', or 'updated'",
        },
        "min_stars": {
            "type": "integer",
            "description": "Minimum star count to filter low-quality repos (default: 50)",
        },
        "per_page": {
            "type": "integer",
            "description": "Number of results to return (default: 5, max: 10)",
        },
    },
    "required": ["query"],
}

INSPECT_REPO_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "owner": {
            "type": "string",
            "description": "Repository owner (user or organization). Example: 'expressjs'",
        },
        "repo": {
            "type": "string",
            "description": "Repository name. Example: 'express'",
        },
        "include_readme": {
            "type": "boolean",
            "description": "Fetch and include README content (default: true)",
        },
        "include_tree": {
            "type": "boolean",
            "description": "Fetch and include file tree (default: true)",
        },
    },
    "required": ["owner", "repo"],
}

SEARCH_GITHUB_CODE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Code pattern to search for across GitHub. "
                "Example: 'async def create_app FastAPI'"
            ),
        },
        "language": {
            "type": "string",
            "description": "Filter by language. Example: 'python'",
        },
        "repo": {
            "type": "string",
            "description": "Scope search to a specific repo. Format: 'owner/repo'",
        },
        "per_page": {
            "type": "integer",
            "description": "Number of results (default: 5, max: 10)",
        },
    },
    "required": ["query"],
}
