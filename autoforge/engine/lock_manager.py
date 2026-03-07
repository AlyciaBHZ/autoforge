"""Lock Manager — atomic task claiming via filesystem.

Uses os.symlink() on POSIX (atomic) and file-write on Windows (with
FileExistsError-based race detection using os.open with O_CREAT|O_EXCL).
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from autoforge.engine.development_harness import (
    append_development_jsonl,
    write_development_json,
)

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"


class LockManager:
    """Atomic task claiming using filesystem locks.

    POSIX: symlink-based (atomic by OS guarantee).
    Windows: exclusive-create file (O_CREAT|O_EXCL is atomic on NTFS).

    A task is claimed by creating a lock file:
        locks/<task-id>.lock

    If the file already exists, the claim fails, guaranteeing no two
    agents can claim the same task.
    """

    # Characters allowed in task IDs to prevent path traversal
    _SAFE_ID_CHARS = frozenset(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "-_."
    )

    def __init__(self, lock_dir: Path, *, harness_root: Path | None = None) -> None:
        self.lock_dir = lock_dir
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        # In-memory cache: task_id -> agent_id for active locks.
        # Falls back to filesystem scanning when potentially stale.
        self._lock_cache: dict[str, str] = {}
        self._harness_root = harness_root
        if self._harness_root is not None:
            self._harness_root.mkdir(parents=True, exist_ok=True)
            self._write_lease_artifact()

    def _lease_events_path(self) -> Path | None:
        if self._harness_root is None:
            return None
        return self._harness_root / "lease_events.jsonl"

    def _append_lease_event(
        self,
        *,
        event_type: str,
        task_id: str,
        agent_id: str = "",
        detail: str = "",
    ) -> None:
        path = self._lease_events_path()
        if path is None:
            return
        append_development_jsonl(
            path,
            {
                "task_id": str(task_id or ""),
                "agent_id": str(agent_id or ""),
                "detail": str(detail or ""),
                "lock_dir": str(self.lock_dir),
            },
            event_type=event_type,
        )

    def _write_lease_artifact(self) -> None:
        if self._harness_root is None:
            return
        locks: list[dict[str, str]] = []
        for lock_file in sorted(self.lock_dir.glob("*.lock")):
            owner = self._read_owner(lock_file)
            if owner is None:
                continue
            locks.append({"task_id": lock_file.stem, "owner": owner})
        write_development_json(
            self._harness_root / "lease_artifact.json",
            {
                "lock_dir": str(self.lock_dir),
                "lock_count": len(locks),
                "locks": locks,
            },
            artifact_type="lease_artifact",
        )

    @classmethod
    def _sanitize_id(cls, task_id: str) -> str:
        """Sanitize task_id to prevent path traversal.

        Raises ValueError if the ID contains unsafe characters.
        """
        if not task_id:
            raise ValueError("Empty task_id")
        if not all(c in cls._SAFE_ID_CHARS for c in task_id):
            raise ValueError(
                f"Task ID contains unsafe characters: {task_id!r}"
            )
        return task_id

    def _lock_path(self, task_id: str) -> Path:
        """Return sanitized lock file path for a task ID."""
        safe_id = self._sanitize_id(task_id)
        return self.lock_dir / f"{safe_id}.lock"

    def try_claim(self, task_id: str, agent_id: str) -> bool:
        """Atomically claim a task. Returns True if successful."""
        lock_path = self._lock_path(task_id)
        try:
            if _IS_WINDOWS:
                # O_CREAT | O_EXCL is atomic on NTFS — fails if file exists
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, agent_id.encode())
                os.close(fd)
            else:
                os.symlink(agent_id, lock_path)
            self._lock_cache[task_id] = agent_id
            self._append_lease_event(event_type="lease_claimed", task_id=task_id, agent_id=agent_id)
            self._write_lease_artifact()
            logger.info(f"Task {task_id} claimed by {agent_id}")
            return True
        except FileExistsError:
            existing = self.get_owner(task_id)
            self._append_lease_event(
                event_type="lease_contention",
                task_id=task_id,
                agent_id=agent_id,
                detail=str(existing or ""),
            )
            logger.debug(f"Task {task_id} already claimed by {existing}")
            return False

    @contextlib.contextmanager
    def claim(self, task_id: str, agent_id: str) -> Iterator[bool]:
        """Context manager: claim a task and auto-release on exit."""
        claimed = self.try_claim(task_id, agent_id)
        try:
            yield claimed
        finally:
            if claimed:
                self.release(task_id, agent_id)

    def release(self, task_id: str, agent_id: str) -> bool:
        """Release a task lock. Only the owner can release.

        Uses atomic rename to avoid TOCTOU race: rename the lock to a
        temp name (atomic), verify ownership, then delete. If ownership
        doesn't match, rename back.
        """
        lock_path = self._lock_path(task_id)
        # Atomic rename to prevent concurrent release/claim races
        tmp_path = lock_path.with_suffix(".releasing")
        try:
            lock_path.rename(tmp_path)
        except FileNotFoundError:
            self._lock_cache.pop(task_id, None)
            return False
        except OSError:
            self._lock_cache.pop(task_id, None)
            return False

        try:
            owner = self._read_owner(tmp_path)
            if owner == agent_id:
                tmp_path.unlink(missing_ok=True)
                self._lock_cache.pop(task_id, None)
                self._append_lease_event(event_type="lease_released", task_id=task_id, agent_id=agent_id)
                self._write_lease_artifact()
                logger.info("Task %s released by %s", task_id, agent_id)
                return True
            else:
                # Not our lock — rename back
                try:
                    tmp_path.rename(lock_path)
                except OSError:
                    # If rename-back fails, clean up to avoid orphan
                    tmp_path.unlink(missing_ok=True)
                return False
        except Exception:
            # On any error, try to restore the lock file
            try:
                tmp_path.rename(lock_path)
            except OSError:
                tmp_path.unlink(missing_ok=True)
            self._lock_cache.pop(task_id, None)
            return False

    def get_owner(self, task_id: str) -> str | None:
        """Query who owns a task lock."""
        lock_path = self._lock_path(task_id)
        try:
            return self._read_owner(lock_path)
        except (FileNotFoundError, OSError):
            return None

    def _read_owner(self, lock_path: Path) -> str | None:
        """Read the owner from a lock file (symlink or regular file).

        On POSIX, only reads symlink targets (never follows symlinks to
        read file contents, preventing symlink-based path traversal).
        """
        if _IS_WINDOWS:
            # On Windows, refuse to read symlinks to prevent traversal
            if lock_path.is_symlink():
                logger.warning("Ignoring symlink lock file: %s", lock_path)
                return None
            if lock_path.exists():
                return lock_path.read_text(encoding="utf-8").strip()
        else:
            # Check symlink first (primary POSIX mechanism)
            if lock_path.is_symlink():
                return os.readlink(lock_path)
            # Regular files: only read if NOT a symlink (defense-in-depth)
            if lock_path.exists() and not lock_path.is_symlink():
                return lock_path.read_text(encoding="utf-8").strip()
        return None

    def _is_stale(self, lock_path: Path, max_age_seconds: int = 3600) -> bool:
        """Check if a lock file is stale (older than *max_age_seconds*).

        Returns False if the file doesn't exist or the mtime can't be read.
        """
        try:
            mtime = lock_path.stat().st_mtime
            age = time.time() - mtime
            return age > max_age_seconds
        except (FileNotFoundError, OSError):
            return False

    def agent_task_count(self, agent_id: str) -> int:
        """Count how many tasks an agent currently holds.

        Uses the in-memory cache for a fast first pass, then
        falls back to filesystem scanning to reconcile.
        """
        # Fast path: check cache first
        cached_count = sum(
            1 for owner in self._lock_cache.values() if owner == agent_id
        )
        # Reconcile with filesystem to catch external changes
        fs_count = 0
        refreshed_cache: dict[str, str] = {}
        for lock_file in self.lock_dir.glob("*.lock"):
            try:
                task_id = lock_file.stem
                owner = self._read_owner(lock_file)
                if owner is not None:
                    refreshed_cache[task_id] = owner
                    if owner == agent_id:
                        fs_count += 1
            except OSError:
                continue
        # Update cache from filesystem truth
        self._lock_cache = refreshed_cache
        return fs_count

    def enforce_single_task(
        self, agent_id: str, max_age_seconds: int = 3600
    ) -> bool:
        """Hard rule check: agent has no in-progress tasks.

        Stale locks (older than *max_age_seconds*) are automatically
        cleaned up and not counted against the agent.
        """
        count = 0
        for lock_file in self.lock_dir.glob("*.lock"):
            try:
                owner = self._read_owner(lock_file)
                if owner != agent_id:
                    continue
                # Skip (and remove) stale locks
                if self._is_stale(lock_file, max_age_seconds):
                    task_id = lock_file.stem
                    logger.warning(
                        f"Removing stale lock for task {task_id} "
                        f"(owner={agent_id})"
                    )
                    lock_file.unlink(missing_ok=True)
                    self._lock_cache.pop(task_id, None)
                    self._append_lease_event(event_type="lease_stale_cleared", task_id=task_id, agent_id=agent_id)
                    self._write_lease_artifact()
                    continue
                count += 1
            except OSError:
                continue
        return count == 0

    def clear_all(self) -> None:
        """Clear all locks (used on fresh start or cleanup)."""
        for lock_file in self.lock_dir.glob("*.lock"):
            try:
                lock_file.unlink()
            except OSError:
                continue
        self._lock_cache.clear()
        self._append_lease_event(event_type="lease_clear_all", task_id="*")
        self._write_lease_artifact()
        logger.info("All locks cleared")

    def clear_stale(self, max_age_seconds: int = 3600) -> None:
        """Clear stale locks older than *max_age_seconds*.

        This is used for best-effort recovery after interrupted runs without
        disrupting active locks belonging to other workers.
        """
        for lock_file in self.lock_dir.glob("*.lock"):
            try:
                if self._is_stale(lock_file, max_age_seconds):
                    lock_file.unlink(missing_ok=True)
                    self._lock_cache.pop(lock_file.stem, None)
                    self._append_lease_event(event_type="lease_stale_cleared", task_id=lock_file.stem)
            except OSError:
                continue
        self._write_lease_artifact()
