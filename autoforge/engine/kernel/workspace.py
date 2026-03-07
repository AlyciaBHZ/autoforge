from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class WorkspaceLockRecord:
    holder: str
    run_id: str
    hostname: str
    pid: int
    acquired_at: float
    expires_at: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "holder": self.holder,
            "run_id": self.run_id,
            "hostname": self.hostname,
            "pid": int(self.pid),
            "acquired_at": float(self.acquired_at),
            "expires_at": float(self.expires_at),
            "metadata": dict(self.metadata),
        }


class WorkspaceLock:
    """Cross-process workspace lock with TTL-based stale recovery."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.lock_path = project_dir / ".autoforge" / "kernel" / "workspace.lock.json"

    def inspect(self) -> WorkspaceLockRecord | None:
        if not self.lock_path.exists():
            return None
        try:
            raw = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
            return WorkspaceLockRecord(
                holder=str(raw.get("holder", "")),
                run_id=str(raw.get("run_id", "")),
                hostname=str(raw.get("hostname", "")),
                pid=int(raw.get("pid", 0) or 0),
                acquired_at=float(raw.get("acquired_at", 0.0) or 0.0),
                expires_at=float(raw.get("expires_at", 0.0) or 0.0),
                metadata=dict(raw.get("metadata", {}) or {}),
            )
        except Exception:
            return None

    def acquire(
        self,
        *,
        holder: str,
        run_id: str,
        ttl_seconds: int = 900,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        metadata_dict = dict(metadata or {})
        now = time.time()
        record = WorkspaceLockRecord(
            holder=str(holder),
            run_id=str(run_id),
            hostname=socket.gethostname(),
            pid=os.getpid(),
            acquired_at=now,
            expires_at=now + max(30, int(ttl_seconds or 0)),
            metadata=metadata_dict,
        )
        payload = record.to_dict()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = self.inspect()
            if existing is not None and existing.expires_at > now:
                return False
            try:
                self.lock_path.unlink(missing_ok=True)
            except OSError:
                return False
            return self.acquire(
                holder=holder,
                run_id=run_id,
                ttl_seconds=ttl_seconds,
                metadata=metadata_dict,
            )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        except Exception:
            try:
                self.lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return True

    def heartbeat(self, *, holder: str, run_id: str, ttl_seconds: int = 900) -> bool:
        existing = self.inspect()
        if existing is None:
            return False
        if existing.holder != str(holder) or existing.run_id != str(run_id):
            return False
        updated = WorkspaceLockRecord(
            holder=existing.holder,
            run_id=existing.run_id,
            hostname=existing.hostname,
            pid=existing.pid,
            acquired_at=existing.acquired_at,
            expires_at=time.time() + max(30, int(ttl_seconds or 0)),
            metadata=existing.metadata,
        )
        _atomic_write_json(self.lock_path, updated.to_dict())
        return True

    def release(self, *, holder: str, run_id: str) -> bool:
        existing = self.inspect()
        if existing is None:
            return True
        if existing.holder != str(holder) or existing.run_id != str(run_id):
            return False
        try:
            self.lock_path.unlink(missing_ok=True)
            return True
        except OSError:
            return False
