"""Lock Manager — atomic task claiming via filesystem.

Uses os.symlink() on POSIX (atomic) and file-write on Windows (with
FileExistsError-based race detection using os.open with O_CREAT|O_EXCL).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

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
            logger.info(f"Task {task_id} claimed by {agent_id}")
            return True
        except FileExistsError:
            existing = self.get_owner(task_id)
            logger.debug(f"Task {task_id} already claimed by {existing}")
            return False

    def release(self, task_id: str, agent_id: str) -> bool:
        """Release a task lock. Only the owner can release."""
        lock_path = self.lock_dir / f"{task_id}.lock"
        try:
            owner = self._read_owner(lock_path)
            if owner == agent_id:
                lock_path.unlink()
                logger.info(f"Task {task_id} released by {agent_id}")
                return True
            return False
        except (FileNotFoundError, OSError):
            return False

    def get_owner(self, task_id: str) -> Optional[str]:
        """Query who owns a task lock."""
        lock_path = self.lock_dir / f"{task_id}.lock"
        try:
            return self._read_owner(lock_path)
        except (FileNotFoundError, OSError):
            return None

    def _read_owner(self, lock_path: Path) -> Optional[str]:
        """Read the owner from a lock file (symlink or regular file)."""
        if _IS_WINDOWS:
            if lock_path.exists():
                return lock_path.read_text(encoding="utf-8").strip()
        else:
            if lock_path.is_symlink():
                return os.readlink(lock_path)
        return None

    def agent_task_count(self, agent_id: str) -> int:
        """Count how many tasks an agent currently holds."""
        count = 0
        for lock_file in self.lock_dir.glob("*.lock"):
            try:
                owner = self._read_owner(lock_file)
                if owner == agent_id:
                    count += 1
            except OSError:
                continue
        return count

    def enforce_single_task(self, agent_id: str) -> bool:
        """Hard rule check: agent has no in-progress tasks."""
        return self.agent_task_count(agent_id) == 0

    def clear_all(self) -> None:
        """Clear all locks (used on fresh start or cleanup)."""
        for lock_file in self.lock_dir.glob("*.lock"):
            try:
                lock_file.unlink()
            except OSError:
                continue
        logger.info("All locks cleared")
