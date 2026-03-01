"""Lock Manager — atomic task claiming via filesystem symlinks."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LockManager:
    """Atomic task claiming using filesystem symlinks.

    Uses os.symlink() which is atomic on POSIX systems.
    A task is claimed by creating a symlink:
        locks/<task-id>.lock -> <agent-instance-id>

    If the symlink already exists, the claim fails (FileExistsError),
    guaranteeing no two agents can claim the same task.
    """

    def __init__(self, lock_dir: Path) -> None:
        self.lock_dir = lock_dir
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    def try_claim(self, task_id: str, agent_id: str) -> bool:
        """Atomically claim a task. Returns True if successful."""
        lock_path = self.lock_dir / f"{task_id}.lock"
        try:
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
            if lock_path.is_symlink() and os.readlink(lock_path) == agent_id:
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
            if lock_path.is_symlink():
                return os.readlink(lock_path)
        except (FileNotFoundError, OSError):
            pass
        return None

    def agent_task_count(self, agent_id: str) -> int:
        """Count how many tasks an agent currently holds."""
        count = 0
        for lock_file in self.lock_dir.glob("*.lock"):
            try:
                if lock_file.is_symlink() and os.readlink(lock_file) == agent_id:
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
