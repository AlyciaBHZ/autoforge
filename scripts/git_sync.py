#!/usr/bin/env python3
"""AutoForge Git Sync — automated version control workflow.

Provides automated merge, cherry-pick, and sync operations between branches.
Designed for multi-branch development where features are developed on feature
branches and need to be synced with main.

Usage:
    python scripts/git_sync.py sync              # Sync current branch with main
    python scripts/git_sync.py cherry-pick <sha>  # Cherry-pick a commit
    python scripts/git_sync.py merge-main         # Merge main into current branch
    python scripts/git_sync.py status             # Show branch sync status
    python scripts/git_sync.py changelog          # Show changes since last sync
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitResult:
    """Result of a git command."""
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def git(*args: str, check: bool = False) -> GitResult:
    """Run a git command and return the result."""
    cmd = ["git"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: git {' '.join(args)}")
        print(result.stderr.strip())
        sys.exit(1)
    return GitResult(result.returncode, result.stdout.strip(), result.stderr.strip())


def current_branch() -> str:
    """Get the current branch name."""
    return git("rev-parse", "--abbrev-ref", "HEAD", check=True).stdout


def has_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes."""
    return git("status", "--porcelain").stdout != ""


def get_merge_base(branch_a: str, branch_b: str) -> str:
    """Get the common ancestor of two branches."""
    return git("merge-base", branch_a, branch_b, check=True).stdout


def commits_behind_ahead(base: str, target: str) -> tuple[int, int]:
    """Get the number of commits behind and ahead between two refs."""
    result = git("rev-list", "--left-right", "--count", f"{base}...{target}", check=True)
    behind, ahead = result.stdout.split()
    return int(behind), int(ahead)


def list_commits(since_ref: str, until_ref: str = "HEAD") -> list[dict[str, str]]:
    """List commits between two refs."""
    result = git(
        "log", "--oneline", "--format=%H|%s|%an|%ad",
        "--date=short", f"{since_ref}..{until_ref}",
    )
    if not result.ok or not result.stdout:
        return []
    commits = []
    for line in result.stdout.splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            commits.append({
                "sha": parts[0],
                "message": parts[1],
                "author": parts[2],
                "date": parts[3],
            })
    return commits


def has_conflicts() -> bool:
    """Check if the current state has unresolved merge conflicts."""
    result = git("diff", "--name-only", "--diff-filter=U")
    return result.ok and result.stdout != ""


# ── Commands ──


def cmd_status(args: argparse.Namespace) -> int:
    """Show branch sync status relative to main."""
    branch = current_branch()
    main_branch = args.main_branch

    # Fetch latest
    print(f"Fetching origin/{main_branch}...")
    fetch_result = git("fetch", "origin", main_branch)
    if not fetch_result.ok:
        print(f"Warning: Could not fetch origin/{main_branch}: {fetch_result.stderr}")

    # Compare
    try:
        behind, ahead = commits_behind_ahead(f"origin/{main_branch}", "HEAD")
    except SystemExit:
        print(f"Error: Cannot compare with origin/{main_branch}")
        return 1

    print(f"\nBranch: {branch}")
    print(f"Main:   {main_branch}")
    print(f"Behind: {behind} commit(s)")
    print(f"Ahead:  {ahead} commit(s)")

    if behind == 0 and ahead == 0:
        print("\nBranches are in sync.")
    elif behind == 0:
        print(f"\nBranch is {ahead} commit(s) ahead of main. Consider creating a PR.")
    elif ahead == 0:
        print(f"\nBranch is {behind} commit(s) behind main. Run: python scripts/git_sync.py merge-main")
    else:
        print(f"\nBranch has diverged: {behind} behind, {ahead} ahead.")
        print("Run: python scripts/git_sync.py sync")

    # Show uncommitted changes
    if has_uncommitted_changes():
        print("\nWarning: You have uncommitted changes.")

    return 0


def cmd_changelog(args: argparse.Namespace) -> int:
    """Show changes since last sync with main."""
    main_branch = args.main_branch

    git("fetch", "origin", main_branch)

    # Commits on main that we don't have
    merge_base = get_merge_base(f"origin/{main_branch}", "HEAD")
    new_on_main = list_commits(merge_base, f"origin/{main_branch}")
    new_on_branch = list_commits(merge_base, "HEAD")

    if new_on_main:
        print(f"New commits on {main_branch} ({len(new_on_main)}):")
        for c in new_on_main:
            print(f"  {c['sha'][:8]} {c['date']} {c['message']}")
    else:
        print(f"No new commits on {main_branch}.")

    print()

    if new_on_branch:
        print(f"Your commits ({len(new_on_branch)}):")
        for c in new_on_branch:
            print(f"  {c['sha'][:8]} {c['date']} {c['message']}")
    else:
        print("No commits on this branch.")

    return 0


def cmd_merge_main(args: argparse.Namespace) -> int:
    """Merge main into the current branch."""
    branch = current_branch()
    main_branch = args.main_branch

    if branch in ("main", "master"):
        print("Error: Already on main branch. Nothing to merge.")
        return 1

    if has_uncommitted_changes():
        print("Error: Uncommitted changes detected. Commit or stash before merging.")
        return 1

    print(f"Fetching origin/{main_branch}...")
    git("fetch", "origin", main_branch, check=True)

    behind, _ = commits_behind_ahead(f"origin/{main_branch}", "HEAD")
    if behind == 0:
        print("Already up to date with main.")
        return 0

    print(f"Merging {behind} commit(s) from origin/{main_branch}...")
    result = git("merge", f"origin/{main_branch}", "--no-edit")

    if not result.ok:
        if has_conflicts():
            conflicted = git("diff", "--name-only", "--diff-filter=U").stdout
            print("\nMerge conflicts detected in:")
            for f in conflicted.splitlines():
                print(f"  - {f}")
            print("\nResolve conflicts manually, then run:")
            print("  git add <resolved files>")
            print("  git commit")
            print("\nOr abort the merge:")
            print("  git merge --abort")
            return 1
        print(f"Merge failed: {result.stderr}")
        return 1

    print("Merge successful.")
    return 0


def cmd_cherry_pick(args: argparse.Namespace) -> int:
    """Cherry-pick one or more commits into the current branch."""
    if has_uncommitted_changes():
        print("Error: Uncommitted changes detected. Commit or stash first.")
        return 1

    commits = args.commits
    print(f"Cherry-picking {len(commits)} commit(s)...")

    for sha in commits:
        # Verify the commit exists
        verify = git("rev-parse", "--verify", sha)
        if not verify.ok:
            print(f"Error: Commit {sha} not found. Fetch first: git fetch origin")
            return 1

        # Get commit info
        info = git("log", "--oneline", "-1", sha)
        print(f"\n  Picking: {info.stdout}")

        result = git("cherry-pick", sha)
        if not result.ok:
            if has_conflicts():
                conflicted = git("diff", "--name-only", "--diff-filter=U").stdout
                print("\n  Conflicts in:")
                for f in conflicted.splitlines():
                    print(f"    - {f}")
                print("\n  Resolve conflicts, then run:")
                print("    git add <resolved files>")
                print("    git cherry-pick --continue")
                print("\n  Or abort:")
                print("    git cherry-pick --abort")
                return 1
            print(f"  Cherry-pick failed: {result.stderr}")
            return 1

        print("  Done.")

    print(f"\nSuccessfully cherry-picked {len(commits)} commit(s).")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Full sync: merge main, handle conflicts, and push."""
    branch = current_branch()
    main_branch = args.main_branch

    if branch in ("main", "master"):
        print("Error: Cannot sync on main branch.")
        return 1

    if has_uncommitted_changes():
        print("Error: Uncommitted changes detected. Commit or stash first.")
        return 1

    # 1. Fetch
    print(f"Step 1/3: Fetching origin/{main_branch}...")
    git("fetch", "origin", main_branch, check=True)

    # 2. Check status
    behind, ahead = commits_behind_ahead(f"origin/{main_branch}", "HEAD")
    print(f"  Behind main: {behind}, Ahead: {ahead}")

    if behind == 0:
        print("  Already up to date.")
    else:
        # 3. Merge
        print(f"\nStep 2/3: Merging {behind} commit(s) from main...")
        result = git("merge", f"origin/{main_branch}", "--no-edit")

        if not result.ok:
            if has_conflicts():
                conflicted = git("diff", "--name-only", "--diff-filter=U").stdout
                print("\nConflicts detected. Resolve manually:")
                for f in conflicted.splitlines():
                    print(f"  - {f}")
                print("\nAfter resolving, run: git commit")
                print("Then re-run: python scripts/git_sync.py sync")
                return 1
            print(f"Merge failed: {result.stderr}")
            return 1
        print("  Merge successful.")

    # 4. Push
    print(f"\nStep 3/3: Pushing to origin/{branch}...")
    push_result = git("push", "-u", "origin", branch)
    if not push_result.ok:
        # Retry with exponential backoff
        import time
        delays = [2, 4, 8, 16]
        for delay in delays:
            print(f"  Push failed, retrying in {delay}s...")
            time.sleep(delay)
            push_result = git("push", "-u", "origin", branch)
            if push_result.ok:
                break
        if not push_result.ok:
            print(f"  Push failed after retries: {push_result.stderr}")
            print("  You can push manually: git push -u origin {branch}")
            return 1

    print("\nSync complete.")
    return 0


def cmd_pick_range(args: argparse.Namespace) -> int:
    """Cherry-pick a range of commits from main."""
    main_branch = args.main_branch

    if has_uncommitted_changes():
        print("Error: Uncommitted changes detected.")
        return 1

    git("fetch", "origin", main_branch, check=True)

    # List available commits on main
    merge_base = get_merge_base(f"origin/{main_branch}", "HEAD")
    available = list_commits(merge_base, f"origin/{main_branch}")

    if not available:
        print("No new commits on main to cherry-pick.")
        return 0

    print(f"Available commits on {main_branch} ({len(available)}):")
    for i, c in enumerate(available):
        print(f"  [{i}] {c['sha'][:8]} {c['date']} {c['message']}")

    # Parse range
    selection = args.range
    indices = set()
    for part in selection.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))

    selected = []
    for i in sorted(indices):
        if 0 <= i < len(available):
            selected.append(available[i]["sha"])
        else:
            print(f"Warning: Index {i} out of range, skipping.")

    if not selected:
        print("No valid commits selected.")
        return 1

    print(f"\nCherry-picking {len(selected)} commit(s)...")
    for sha in reversed(selected):  # Apply in chronological order
        info = git("log", "--oneline", "-1", sha)
        print(f"  Picking: {info.stdout}")
        result = git("cherry-pick", sha)
        if not result.ok:
            if has_conflicts():
                print("  Conflicts detected. Resolve and run: git cherry-pick --continue")
                return 1
            print(f"  Failed: {result.stderr}")
            return 1

    print("Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AutoForge Git Sync — automated version control workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--main-branch", default="main",
        help="Name of the main branch (default: main)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("status", help="Show branch sync status")
    subparsers.add_parser("changelog", help="Show changes since last sync")
    subparsers.add_parser("merge-main", help="Merge main into current branch")
    subparsers.add_parser("sync", help="Full sync: fetch + merge + push")

    cp = subparsers.add_parser("cherry-pick", help="Cherry-pick commits")
    cp.add_argument("commits", nargs="+", help="Commit SHA(s) to cherry-pick")

    pr = subparsers.add_parser("pick-range", help="Interactively cherry-pick from main")
    pr.add_argument("range", help="Commit indices to pick (e.g. '0,2-4')")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "status": cmd_status,
        "changelog": cmd_changelog,
        "merge-main": cmd_merge_main,
        "cherry-pick": cmd_cherry_pick,
        "sync": cmd_sync,
        "pick-range": cmd_pick_range,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
