"""Git Manager — worktree management for parallel agent isolation."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GitError(Exception):
    """Git operation failed."""


class GitManager:
    """Manages git repository and worktrees for agent isolation.

    Each builder agent gets its own worktree (branch), enabling
    parallel isolated development. Completed work merges back to main.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.worktrees_dir = project_dir.parent / "worktrees"

    async def _run_git(self, *args: str, cwd: Path | None = None) -> str:
        """Run a git command and return stdout."""
        cmd = ["git"] + list(args)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd or self.project_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            error_msg = stderr.decode(errors="replace").strip()
            raise GitError(f"git {' '.join(args)}: {error_msg}")
        return stdout.decode(errors="replace").strip()

    async def init_repo(self) -> None:
        """Initialize a git repo in the project directory."""
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Check if already a git repo
        git_dir = self.project_dir / ".git"
        if git_dir.exists():
            logger.debug(f"Git repo already exists at {self.project_dir}")
            return

        await self._run_git("init")
        await self._run_git("checkout", "-b", "main")

        # Create initial commit so worktrees can branch from it
        gitignore = self.project_dir / ".gitignore"
        gitignore.write_text("node_modules/\n.env\n__pycache__/\n.next/\n")
        await self._run_git("add", ".")
        await self._run_git("commit", "-m", "Initial project scaffold")
        logger.info(f"Git repo initialized at {self.project_dir}")

    async def create_worktree(self, branch_name: str) -> Path:
        """Create a git worktree for an agent to work in isolation."""
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        worktree_path = self.worktrees_dir / branch_name

        if worktree_path.exists():
            logger.debug(f"Worktree already exists: {worktree_path}")
            return worktree_path

        await self._run_git(
            "worktree", "add", "-b", branch_name, str(worktree_path), "main"
        )
        logger.info(f"Created worktree: {worktree_path} (branch: {branch_name})")
        return worktree_path

    async def commit_worktree(
        self, branch_name: str, message: str, task_id: str = ""
    ) -> str:
        """Commit all changes in a worktree. Returns commit hash."""
        worktree_path = self.worktrees_dir / branch_name

        # Stage all changes
        await self._run_git("add", ".", cwd=worktree_path)

        # Check if there's anything to commit
        try:
            await self._run_git("diff", "--cached", "--quiet", cwd=worktree_path)
            logger.debug(f"No changes to commit in {branch_name}")
            return ""
        except GitError:
            pass  # There are staged changes

        commit_msg = f"[{task_id}] {message}" if task_id else message
        await self._run_git("commit", "-m", commit_msg, cwd=worktree_path)

        # Get commit hash
        commit_hash = await self._run_git(
            "rev-parse", "--short", "HEAD", cwd=worktree_path
        )
        logger.info(f"Committed in {branch_name}: {commit_hash}")
        return commit_hash

    async def merge_branch(self, branch_name: str) -> None:
        """Merge a branch back into main."""
        await self._run_git("checkout", "main")
        try:
            await self._run_git(
                "merge", "--no-ff", branch_name, "-m", f"Merge {branch_name}"
            )
            logger.info(f"Merged {branch_name} into main")
        except GitError as e:
            logger.error(f"Merge conflict in {branch_name}: {e}")
            # Abort the merge on conflict
            await self._run_git("merge", "--abort")
            raise

    async def cleanup_worktree(self, branch_name: str) -> None:
        """Remove a worktree and its branch after merge."""
        worktree_path = self.worktrees_dir / branch_name
        try:
            await self._run_git("worktree", "remove", str(worktree_path))
        except GitError:
            logger.warning(f"Could not remove worktree {worktree_path}")
        try:
            await self._run_git("branch", "-d", branch_name)
        except GitError:
            logger.warning(f"Could not delete branch {branch_name}")

    @property
    def main_worktree(self) -> Path:
        return self.project_dir
