"""Git Manager — worktree management for parallel agent isolation."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from autoforge.engine.development_harness import (
    append_development_jsonl,
    write_development_json,
)
from autoforge.engine.runtime.commands import run_args

logger = logging.getLogger(__name__)

# Module-level cache for git availability
_git_available: bool | None = None


def is_git_available() -> bool:
    """Check if git is installed and accessible on the system.

    Result is cached after first check for performance.
    """
    global _git_available
    if _git_available is not None:
        return _git_available

    _git_available = shutil.which("git") is not None
    if _git_available:
        logger.debug("Git detected on system PATH")
    else:
        logger.info("Git not found on system PATH")
    return _git_available


async def get_git_version() -> str | None:
    """Get the installed git version string, or None if git is not available."""
    if not is_git_available():
        return None
    try:
        res = await run_args(
            ["git", "--version"],
            cwd=Path.cwd(),
            timeout_s=5.0,
            max_stdout_chars=2000,
            max_stderr_chars=2000,
            label="git.version",
        )
        if res.exit_code == 0:
            return (res.stdout or "").strip()
    except Exception:
        return None
    return None


class GitError(Exception):
    """Git operation failed."""


class GitManager:
    """Manages git repository and worktrees for agent isolation.

    Each builder agent gets its own worktree (branch), enabling
    parallel isolated development. Completed work merges back to main.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        # Worktrees are project-local to avoid cross-project collisions when multiple
        # projects share the same workspace directory.
        self.worktrees_dir = project_dir / ".autoforge" / "worktrees"
        self._harness_root = project_dir / ".autoforge" / "development_harness" / "execution_harness"

    def _worktree_manifest_path(self) -> Path:
        return self._harness_root / "worktree_manifest.json"

    def _merge_verdict_path(self) -> Path:
        return self._harness_root / "merge_verdict.json"

    def _write_worktree_manifest(self) -> None:
        branches: list[dict[str, str]] = []
        if self.worktrees_dir.exists():
            for path in sorted(self.worktrees_dir.iterdir()):
                if path.is_dir():
                    branches.append({"branch": path.name, "path": str(path)})
        write_development_json(
            self._worktree_manifest_path(),
            {
                "project_dir": str(self.project_dir),
                "main_worktree": str(self.project_dir),
                "worktrees_dir": str(self.worktrees_dir),
                "worktree_count": len(branches),
                "worktrees": branches,
            },
            artifact_type="worktree_manifest",
        )

    def _append_commit_receipt(self, *, branch_name: str, message: str, commit_hash: str) -> None:
        append_development_jsonl(
            self._harness_root / "commit_receipts.jsonl",
            {
                "project_dir": str(self.project_dir),
                "branch_name": str(branch_name or ""),
                "message": str(message or ""),
                "commit_hash": str(commit_hash or ""),
            },
            event_type="git_commit",
        )

    def _write_merge_verdict(self, *, branch_name: str, success: bool, error: str = "") -> None:
        write_development_json(
            self._merge_verdict_path(),
            {
                "project_dir": str(self.project_dir),
                "branch_name": str(branch_name or ""),
                "success": bool(success),
                "error": str(error or ""),
            },
            artifact_type="merge_verdict",
        )

    async def _run_git(self, *args: str, cwd: Path | None = None) -> str:
        """Run a git command and return stdout."""
        cmd = ["git", *list(args)]
        res = await run_args(
            cmd,
            cwd=cwd or self.project_dir,
            timeout_s=120.0,
            max_stdout_chars=12000,
            max_stderr_chars=8000,
            label="git",
        )
        if res.exit_code != 0:
            msg = (res.stderr or res.stdout or "").strip()
            raise GitError(f"git {' '.join(args)}: {msg}")
        return (res.stdout or "").strip()

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
        gitignore.write_text(
            "node_modules/\n"
            ".env\n"
            "__pycache__/\n"
            ".next/\n"
            ".autoforge/\n"
            ".forge_state.json\n"
            ".forge_task_transition_log.jsonl\n"
        )
        await self._run_git("add", ".")
        await self._run_git("commit", "-m", "Initial project scaffold")
        self._write_worktree_manifest()
        logger.info(f"Git repo initialized at {self.project_dir}")

    @staticmethod
    def _sanitize_branch_name(name: str) -> str:
        """Sanitize branch name to prevent command injection."""
        sanitized = re.sub(r"[^a-zA-Z0-9._/-]", "-", name)
        if sanitized.startswith("-"):
            sanitized = "b" + sanitized
        return sanitized

    async def create_worktree(self, branch_name: str) -> Path:
        """Create a git worktree for an agent to work in isolation."""
        branch_name = self._sanitize_branch_name(branch_name)
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        worktree_path = self.worktrees_dir / branch_name

        if worktree_path.exists():
            logger.debug(f"Worktree already exists: {worktree_path}")
            return worktree_path

        await self._run_git(
            "worktree", "add", "-b", branch_name, str(worktree_path), "main"
        )
        self._write_worktree_manifest()
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
        self._append_commit_receipt(branch_name=branch_name, message=commit_msg, commit_hash=commit_hash)
        self._write_worktree_manifest()
        logger.info(f"Committed in {branch_name}: {commit_hash}")
        return commit_hash

    async def merge_branch(self, branch_name: str) -> None:
        """Merge a branch back into main."""
        await self._run_git("checkout", "main")
        try:
            await self._run_git(
                "merge", "--no-ff", branch_name, "-m", f"Merge {branch_name}"
            )
            self._write_merge_verdict(branch_name=branch_name, success=True)
            self._write_worktree_manifest()
            logger.info(f"Merged {branch_name} into main")
        except GitError as e:
            logger.error(f"Merge conflict in {branch_name}: {e}")
            # Abort the merge on conflict
            await self._run_git("merge", "--abort")
            self._write_merge_verdict(branch_name=branch_name, success=False, error=str(e))
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
        self._write_worktree_manifest()

    @property
    def main_worktree(self) -> Path:
        return self.project_dir
