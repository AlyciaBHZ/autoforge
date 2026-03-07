from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoforge.engine.kernel.schema import read_kernel_json, write_kernel_json


@dataclass(frozen=True)
class KernelCheckpoint:
    path: Path
    run_id: str
    lineage_id: str
    parent_run_id: str
    project_id: str
    thread_id: str
    profile: str
    operation: str
    state_marker: str
    state_version: int
    state: dict[str, Any]
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "kernel_checkpoint",
            "run_id": self.run_id,
            "lineage_id": self.lineage_id,
            "parent_run_id": self.parent_run_id,
            "project_id": self.project_id,
            "thread_id": self.thread_id,
            "profile": self.profile,
            "operation": self.operation,
            "state_marker": self.state_marker,
            "state_version": int(self.state_version),
            "state": dict(self.state),
            "updated_at": float(self.updated_at),
        }


def write_kernel_checkpoint(
    path: Path,
    *,
    run_id: str,
    lineage_id: str,
    parent_run_id: str,
    project_id: str,
    thread_id: str,
    profile: str,
    operation: str,
    state_marker: str,
    state_version: int,
    state: dict[str, Any],
) -> KernelCheckpoint:
    checkpoint = KernelCheckpoint(
        path=path,
        run_id=str(run_id),
        lineage_id=str(lineage_id),
        parent_run_id=str(parent_run_id or ""),
        project_id=str(project_id or ""),
        thread_id=str(thread_id or ""),
        profile=str(profile or ""),
        operation=str(operation or ""),
        state_marker=str(state_marker or ""),
        state_version=int(state_version),
        state=dict(state),
        updated_at=time.time(),
    )
    write_kernel_json(path, checkpoint.to_dict(), artifact_type="kernel_checkpoint")
    return checkpoint


def read_kernel_checkpoint(path: Path) -> KernelCheckpoint | None:
    payload = read_kernel_json(path, artifact_type="kernel_checkpoint")
    state = payload.get("state", {})
    if not isinstance(state, dict) or not payload:
        return None
    return KernelCheckpoint(
        path=path,
        run_id=str(payload.get("run_id", "") or ""),
        lineage_id=str(payload.get("lineage_id", "") or ""),
        parent_run_id=str(payload.get("parent_run_id", "") or ""),
        project_id=str(payload.get("project_id", "") or ""),
        thread_id=str(payload.get("thread_id", "") or ""),
        profile=str(payload.get("profile", "") or ""),
        operation=str(payload.get("operation", "") or ""),
        state_marker=str(payload.get("state_marker", "") or ""),
        state_version=int(payload.get("state_version", 0) or 0),
        state=state,
        updated_at=float(payload.get("updated_at", 0.0) or 0.0),
    )


def find_latest_kernel_checkpoint(project_dir: Path, *, preferred_run_id: str | None = None) -> Path | None:
    run_root = project_dir / ".autoforge" / "kernel" / "runs"
    if not run_root.is_dir():
        return None
    if preferred_run_id:
        preferred = run_root / str(preferred_run_id) / "checkpoint.json"
        if preferred.is_file():
            return preferred
    candidates = sorted(
        (child for child in run_root.iterdir() if child.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for child in candidates:
        checkpoint = child / "checkpoint.json"
        if checkpoint.is_file():
            return checkpoint
    return None


def load_latest_kernel_checkpoint(
    project_dir: Path,
    *,
    preferred_run_id: str | None = None,
) -> KernelCheckpoint | None:
    checkpoint_path = find_latest_kernel_checkpoint(project_dir, preferred_run_id=preferred_run_id)
    if checkpoint_path is None:
        return None
    return read_kernel_checkpoint(checkpoint_path)
