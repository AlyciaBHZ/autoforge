"""File-system snapshot utilities for trace/replay and harness judging."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SKIP_DIRS: set[str] = {
    ".git",
    ".autoforge",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}


def compute_file_manifest(
    root: Path,
    *,
    skip_dirs: Iterable[str] = DEFAULT_SKIP_DIRS,
    max_bytes: int = 2 * 1024 * 1024,
) -> dict[str, Any]:
    """Compute a deterministic file manifest for a directory tree.

    Notes:
      - Only hashes files <= max_bytes (to avoid huge binary/lock artifacts).
      - Always returns JSON-serializable primitives.
    """
    root = root.resolve()
    skip = set(skip_dirs)
    files: list[dict[str, Any]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip for part in p.parts):
            continue
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        sha: str | None = None
        if int(st.st_size) <= int(max_bytes):
            try:
                sha = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                sha = None
        files.append(
            {
                "path": rel,
                "bytes": int(st.st_size),
                "sha256": sha,
            }
        )
    files.sort(key=lambda x: x["path"])
    return {
        "ts": float(time.time()),
        "root": str(root),
        "max_bytes": int(max_bytes),
        "files": files,
    }


def diff_manifests(before: dict[str, Any] | None, after: dict[str, Any]) -> dict[str, Any]:
    """Compute a stable diff between two manifests."""
    after_files = after.get("files", [])
    after_map = {
        str(f.get("path")): f
        for f in after_files
        if isinstance(f, dict) and isinstance(f.get("path"), str)
    }
    if before is None:
        return {
            "added": sorted(after_map.keys()),
            "removed": [],
            "changed": [],
            "counts": {"added": len(after_map), "removed": 0, "changed": 0},
        }

    before_files = before.get("files", [])
    before_map = {
        str(f.get("path")): f
        for f in before_files
        if isinstance(f, dict) and isinstance(f.get("path"), str)
    }

    before_keys = set(before_map.keys())
    after_keys = set(after_map.keys())
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    changed: list[str] = []
    for p in sorted(before_keys & after_keys):
        b = before_map[p]
        a = after_map[p]
        if (b.get("sha256") != a.get("sha256")) or (b.get("bytes") != a.get("bytes")):
            changed.append(p)
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "counts": {"added": len(added), "removed": len(removed), "changed": len(changed)},
    }

