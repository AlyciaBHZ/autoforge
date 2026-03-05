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

    def __init__(self, lock_dir: Path) -> None:
        self.lock_dir = lock_dir
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        # In-memory cache: task_id -> agent_id for active locks.
        # Falls back to filesystem scanning when potentially stale.
        self._lock_cache: dict[str, str] = {}

    def try_claim(self, task_id: str, agent_id: str) -> bool:
        """Atomically claim a task. Returns True if successful."""
        lock_path = self.lock_dir / f"{task_id}.lock"
        try:
            if _IS_WINDOWS:
                # O_CREAT | O_EXCL is atomic on NTFS — fails if file exists
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, agent_id.encode())
                os.close(fd)
            else:
                os.symlink(agent_id, lock_path)
            self._lock_cache[task_id] = agent_id
            logger.info(f"Task {task_id} claimed by {agent_id}")
            return True
        except FileExistsError:
            existing = self.get_owner(task_id)
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
        """Release a task lock. Only the owner can release."""
        lock_path = self.lock_dir / f"{task_id}.lock"
        try:
            owner = self._read_owner(lock_path)
            if owner == agent_id:
                lock_path.unlink()
                self._lock_cache.pop(task_id, None)
                logger.info(f"Task {task_id} released by {agent_id}")
                return True
            return False
        except (FileNotFoundError, OSError):
            self._lock_cache.pop(task_id, None)
            return False

    def get_owner(self, task_id: str) -> str | None:
        """Query who owns a task lock."""
        lock_path = self.lock_dir / f"{task_id}.lock"
        try:
            return self._read_owner(lock_path)
        except (FileNotFoundError, OSError):
            return None

    def _read_owner(self, lock_path: Path) -> str | None:
        """Read the owner from a lock file (symlink or regular file)."""
        if _IS_WINDOWS:
            if lock_path.exists():
                return lock_path.read_text(encoding="utf-8").strip()
        else:
            # Check symlink first (primary POSIX mechanism)
            if lock_path.is_symlink():
                return os.readlink(lock_path)
            # Also handle regular files (e.g. created by a different
            # code-path, manual intervention, or cross-platform copies)
            if lock_path.exists():
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
        logger.info("All locks cleared")
