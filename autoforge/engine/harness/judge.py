"""Harness judging primitives (tests, holdout, diffing)."""

from __future__ import annotations

import difflib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoforge.engine.runtime.fs import compute_file_manifest, diff_manifests
from autoforge.engine.runtime.telemetry import TelemetrySink
from autoforge.engine.sandbox import create_sandbox


def _safe_relpath(rel: str) -> str:
    rel = (rel or "").replace("\\", "/").lstrip("/")
    if ".." in Path(rel).parts:
        raise ValueError("Path traversal not allowed")
    return rel


@dataclass(frozen=True)
class TestResult:
    command: str
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float


async def run_test_command(
    config: Any,
    project_dir: Path,
    *,
    command: str,
    timeout_s: int = 900,
    telemetry: TelemetrySink | None = None,
) -> TestResult:
    sb = create_sandbox(config, project_dir, telemetry=telemetry)
    async with sb:
        res = await sb.exec(command, timeout=int(timeout_s))
    return TestResult(
        command=command,
        ok=(res.exit_code == 0 and not res.timed_out),
        exit_code=int(res.exit_code),
        stdout=res.stdout,
        stderr=res.stderr,
        timed_out=bool(res.timed_out),
        duration_seconds=float(res.duration_seconds),
    )


@dataclass
class DirDiff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    patch: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": self.added,
            "removed": self.removed,
            "changed": self.changed,
            "patch_chars": len(self.patch),
        }


def diff_directories(
    before_dir: Path,
    after_dir: Path,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_patch_chars: int = 200_000,
    exclude_rel_paths: list[str] | None = None,
) -> DirDiff:
    before = compute_file_manifest(before_dir, max_bytes=max_bytes)
    after = compute_file_manifest(after_dir, max_bytes=max_bytes)
    diff = diff_manifests(before, after)

    exclude = []
    for p in (exclude_rel_paths or []):
        try:
            exclude.append(_safe_relpath(p))
        except Exception:
            continue

    def _excluded(path: str) -> bool:
        if not exclude:
            return False
        raw = (path or "").replace("\\", "/").lstrip("/")
        for ex in exclude:
            exn = ex.rstrip("/")
            if not exn:
                continue
            if raw == exn or raw.startswith(exn + "/"):
                return True
        return False

    changed = [p for p in diff.get("changed", []) if not _excluded(str(p))]
    added = [p for p in diff.get("added", []) if not _excluded(str(p))]
    removed = [p for p in diff.get("removed", []) if not _excluded(str(p))]

    patch_parts: list[str] = []
    remaining = max(10_000, int(max_patch_chars))

    def _read_text(path: Path) -> list[str] | None:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        return raw.splitlines(keepends=True)

    for rel in changed + added + removed:
        if remaining <= 0:
            break
        rel = _safe_relpath(str(rel))
        a_path = after_dir / rel
        b_path = before_dir / rel

        a_lines = _read_text(a_path) if a_path.is_file() else []
        b_lines = _read_text(b_path) if b_path.is_file() else []
        if a_lines is None or b_lines is None:
            continue
        ud = difflib.unified_diff(
            b_lines,
            a_lines,
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            n=3,
        )
        chunk = "".join(list(ud))
        if not chunk:
            continue
        if len(chunk) > remaining:
            chunk = chunk[:remaining] + "\n... (truncated) ...\n"
        patch_parts.append(chunk)
        remaining -= len(chunk)

    return DirDiff(added=added, removed=removed, changed=changed, patch="".join(patch_parts))


def patch_similarity(generated_patch: str, golden_patch: str) -> float:
    if not generated_patch and not golden_patch:
        return 1.0
    if not generated_patch or not golden_patch:
        return 0.0
    return float(difflib.SequenceMatcher(a=golden_patch, b=generated_patch).ratio())


def hide_paths(project_dir: Path, rel_paths: list[str]) -> list[tuple[Path, Path]]:
    """Move holdout tests out of the visible tree before the agent runs."""
    moved: list[tuple[Path, Path]] = []
    hidden_root = project_dir / ".autoforge" / "harness_holdout_hidden"
    hidden_root.mkdir(parents=True, exist_ok=True)
    for rel in rel_paths:
        rel = _safe_relpath(rel)
        src = (project_dir / rel).resolve()
        try:
            if not src.is_relative_to(project_dir.resolve()):
                raise ValueError("Path escapes project_dir")
        except Exception:
            raise ValueError("Path traversal not allowed") from None
        if not src.exists():
            continue
        dst = (hidden_root / rel).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append((dst, src))
    return moved


def restore_paths(moves: list[tuple[Path, Path]]) -> None:
    for hidden, original in moves:
        if not hidden.exists():
            continue
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(hidden), str(original))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
