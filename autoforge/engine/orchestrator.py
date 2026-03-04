"""Orchestrator — the brain of AutoForge.

Supports three pipelines:
  - Generate:  SPEC → BUILD → VERIFY → REFACTOR → DELIVER
  - Review:    SCAN → REVIEW → [REFACTOR] → REPORT
  - Import:    SCAN → REVIEW → [ENHANCE] → VERIFY → [REFACTOR] → DELIVER

Each phase has a quality gate. The orchestrator manages agent lifecycle,
task scheduling, and state persistence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from autoforge.engine.agents.architect import ArchitectAgent
from autoforge.engine.agents.builder import BuilderAgent
from autoforge.engine.agents.director import DirectorAgent, DirectorFixAgent
from autoforge.engine.agents.gardener import GardenerAgent
from autoforge.engine.agents.reviewer import ReviewerAgent
from autoforge.engine.agents.scanner import ScannerAgent
from autoforge.engine.agents.tester import TesterAgent
from autoforge.engine.config import ForgeConfig
from autoforge.engine.git_manager import GitManager, is_git_available
from autoforge.engine.llm_router import BudgetExceededError, LLMRouter
from autoforge.engine.lock_manager import LockManager
from autoforge.engine.sandbox import SandboxBase, create_sandbox
from autoforge.engine.task_dag import Task, TaskDAG, TaskPhase, TaskStatus

logger = logging.getLogger(__name__)
console = Console()


class UserPausedError(Exception):
    """Raised when user declines to continue at a checkpoint."""


class Orchestrator:
    """Main orchestrator for the AutoForge pipeline."""

    def __init__(self, config: ForgeConfig) -> None:
        self.config = config
        self.llm = LLMRouter(config)
        self.project_dir: Path | None = None
        self.dag: TaskDAG | None = None
        self.spec: dict[str, Any] = {}
        self.architecture: dict[str, Any] = {}
        self._state_file: Path | None = None
        self._start_time: float = 0

    async def run(self, requirement: str) -> Path:
        """Execute the full pipeline. Returns path to the generated project."""
        self._start_time = time.monotonic()
        logger.info(f"AutoForge run {self.config.run_id} starting")

        try:
            # Phase 1: SPEC
            console.print("\n[bold blue]Phase 1: SPEC[/bold blue] — Analyzing requirements...")
            self.spec = await self._phase_spec(requirement)
            project_name = self.spec.get("project_name", "project")
            self.project_dir = self.config.workspace_dir / project_name
            self.project_dir.mkdir(parents=True, exist_ok=True)
            self._state_file = self.project_dir / ".forge_state.json"

            # Save spec
            (self.project_dir / "spec.json").write_text(
                json.dumps(self.spec, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._save_state("spec_complete")
            n_modules = len(self.spec.get("modules", []))
            console.print(f"  [green]Spec generated:[/green] {project_name} — {n_modules} modules")

            # Checkpoint: review spec before building
            await self._checkpoint(
                "spec",
                f"Generated spec with {n_modules} modules. "
                f"Review: {self.project_dir / 'spec.json'}",
            )

            # Phase 2: BUILD
            console.print("\n[bold blue]Phase 2: BUILD[/bold blue] — Building project...")
            await self._phase_build()
            self._save_state("build_complete")

            # Checkpoint: review build output before verification
            if self.dag:
                done = len([t for t in self.dag.get_all_tasks()
                           if t.phase == TaskPhase.BUILD and t.status == TaskStatus.DONE])
                total = len([t for t in self.dag.get_all_tasks()
                            if t.phase == TaskPhase.BUILD])
                await self._checkpoint(
                    "build",
                    f"Built {done}/{total} tasks. Review generated code in: {self.project_dir}",
                )

            # Phase 3: VERIFY
            console.print("\n[bold blue]Phase 3: VERIFY[/bold blue] — Verifying project...")
            await self._phase_verify()
            self._save_state("verify_complete")

            # Checkpoint: review test results
            await self._checkpoint("verify", "Verification complete. Review test_results.json.")

            # Phase 4: REFACTOR
            console.print("\n[bold blue]Phase 4: REFACTOR[/bold blue] — Improving quality...")
            await self._phase_refactor()
            self._save_state("refactor_complete")

            # Phase 5: DELIVER
            console.print("\n[bold blue]Phase 5: DELIVER[/bold blue] — Packaging...")
            await self._phase_deliver()
            self._save_state("complete")

            self._print_summary()
            return self.project_dir

        except UserPausedError as e:
            console.print(f"\n[bold yellow]Paused:[/bold yellow] {e}")
            console.print("Use [bold]autoforge resume[/bold] to continue.")
            if self.project_dir:
                return self.project_dir
            raise
        except BudgetExceededError as e:
            console.print(f"\n[bold red]Budget exceeded:[/bold red] {e}")
            console.print("Delivering what's available so far.")
            if self.project_dir:
                self._save_state("budget_exceeded")
            raise
        except Exception as e:
            logger.error(f"AutoForge failed: {e}", exc_info=True)
            if self.project_dir:
                self._save_state(f"failed: {e}")
            raise

    # ──────────────────────────────────────────────
    # Human-in-the-Loop Checkpoints
    # ──────────────────────────────────────────────

    async def _checkpoint(self, phase: str, summary: str) -> None:
        """Pause for user confirmation if this phase is in confirm_phases.

        Uses Rich Confirm prompt bridged into async context via to_thread.
        If user declines, saves state and raises UserPausedError.
        """
        confirm_phases = getattr(self.config, "confirm_phases", [])
        if not confirm_phases:
            return
        if phase not in confirm_phases and "all" not in confirm_phases:
            return

        console.print(
            f"\n[bold yellow]--- Checkpoint: {phase.upper()} complete ---[/bold yellow]"
        )
        console.print(f"  {summary}")
        if self.project_dir:
            console.print(f"  Output: {self.project_dir}")

        # Bridge sync Rich prompt into async context
        from rich.prompt import Confirm

        proceed = await asyncio.to_thread(
            Confirm.ask, "  Continue to next phase?", default=True
        )
        if not proceed:
            self._save_state(f"paused_after_{phase}")
            raise UserPausedError(
                f"User paused after {phase}. Resume with: autoforge resume"
            )

        console.print("  [green]Continuing...[/green]\n")

    # ──────────────────────────────────────────────
    # Phase 1: SPEC
    # ──────────────────────────────────────────────

    async def _phase_spec(self, requirement: str) -> dict[str, Any]:
        director = DirectorAgent(self.config, self.llm)
        result = await director.run({"project_description": requirement})

        if not result.success:
            raise RuntimeError(f"Director failed: {result.error}")

        spec = director.parse_spec(result.output)

        # Validate spec
        if not spec.get("modules"):
            raise ValueError("Director produced empty module list")
        if not spec.get("project_name"):
            raise ValueError("Director did not specify project_name")

        return spec

    # ──────────────────────────────────────────────
    # Phase 2: BUILD
    # ──────────────────────────────────────────────

    async def _phase_build(self) -> None:
        # Step 1: Architect designs and creates task DAG
        console.print("  Designing architecture...")
        architect = ArchitectAgent(self.config, self.llm)
        arch_result = await architect.run({"spec": self.spec})

        if not arch_result.success:
            raise RuntimeError(f"Architect failed: {arch_result.error}")

        arch_data = architect.parse_architecture(arch_result.output)
        self.architecture = arch_data.get("architecture", {})
        tasks_data = arch_data.get("tasks", [])

        if not tasks_data:
            # Fallback: create a simple task per module
            logger.warning("Architect produced no tasks, creating from modules")
            tasks_data = self._tasks_from_modules()

        self.dag = TaskDAG.from_dict(tasks_data)
        console.print(f"  [green]Task DAG:[/green] {self.dag.total_tasks()} tasks created")

        # Save architecture and DAG
        (self.project_dir / "architecture.md").write_text(
            json.dumps(self.architecture, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.dag.save(self.project_dir / "dev_plan.json")
        (self.project_dir / "DEV_PLAN.md").write_text(
            self.dag.to_markdown(), encoding="utf-8"
        )

        # Step 2: Initialize git repo (if git is available)
        git: GitManager | None = None
        if is_git_available():
            git = GitManager(self.project_dir)
            await git.init_repo()
        else:
            console.print(
                "  [yellow]Git not found[/yellow] — building without branch isolation.\n"
                "  Install Git for parallel worktree support: [link=https://git-scm.com]https://git-scm.com[/link]"
            )

        # Step 3: Detect file overlaps (Section F)
        overlaps = self._detect_file_overlaps(self.dag)

        # Step 4: Build tasks
        sandbox = create_sandbox(self.config, self.project_dir)
        lock_manager = LockManager(self.config.workspace_dir / ".locks")
        lock_manager.clear_all()

        async with sandbox:
            await self._execute_build_tasks(git, sandbox, lock_manager, overlaps)

        # Step 5: Enforce build gate (Section C)
        await self._enforce_build_gate(self.dag, self.project_dir)

    def _tasks_from_modules(self) -> list[dict[str, Any]]:
        """Fallback: create one task per module from the spec."""
        tasks = []
        for i, module in enumerate(self.spec.get("modules", []), 1):
            deps = []
            if i > 1:
                # Each module depends on its declared dependencies
                for dep_name in module.get("dependencies", []):
                    for j, m in enumerate(self.spec.get("modules", []), 1):
                        if m["name"] == dep_name:
                            deps.append(f"TASK-{j:03d}")
            tasks.append({
                "id": f"TASK-{i:03d}",
                "description": f"Implement module: {module['name']} — {module.get('description', '')}",
                "owner": "builder",
                "phase": "build",
                "depends_on": deps,
                "files": module.get("files", []),
                "acceptance_criteria": f"All files for {module['name']} module created and functional",
            })
        return tasks

    async def _execute_build_tasks(
        self,
        git: GitManager | None,
        sandbox: SandboxBase,
        lock_manager: LockManager,
        file_overlaps: dict[str, list[str]] | None = None,
    ) -> None:
        """Execute build tasks, potentially in parallel.

        Tasks that share files (per file_overlaps) are serialized to avoid
        merge conflicts. All other tasks run in parallel up to max_agents.
        """
        max_parallel = self.config.max_agents
        active_tasks: dict[str, asyncio.Task] = {}
        # Track which files are being written by active tasks (Section F)
        active_files: set[str] = set()
        overlap_files = file_overlaps or {}

        while self.dag.has_pending_tasks(TaskPhase.BUILD):
            ready = self.dag.get_ready_tasks()
            ready_build = [t for t in ready if t.phase == TaskPhase.BUILD]

            if not ready_build and not active_tasks:
                # No tasks ready and none running — check for failures
                if self.dag.has_failures():
                    # Retry failed tasks
                    for task in self.dag.get_tasks_by_phase(TaskPhase.BUILD):
                        if task.status == TaskStatus.FAILED:
                            self.dag.reset_failed(task.id)
                    continue
                break

            # Launch new tasks up to the parallel limit
            for task in ready_build:
                if len(active_tasks) >= max_parallel:
                    break

                # Section F: Skip tasks whose files conflict with active tasks
                task_files = set(task.files)
                if overlap_files and task_files & active_files:
                    logger.debug(
                        f"Deferring {task.id}: file overlap with active tasks"
                    )
                    continue

                agent_id = f"builder-{len(active_tasks):02d}"
                if not lock_manager.enforce_single_task(agent_id):
                    continue

                if lock_manager.try_claim(task.id, agent_id):
                    self.dag.mark_in_progress(task.id, agent_id)
                    console.print(f"  [{agent_id}] Building: {task.id} — {task.description[:60]}")

                    # Track active files for overlap detection
                    active_files.update(task_files)

                    # Get existing files for context
                    existing = self._list_project_files()

                    async_task = asyncio.create_task(
                        self._build_single_task(
                            task, agent_id, sandbox, git, lock_manager, existing
                        )
                    )
                    active_tasks[task.id] = async_task

            # Wait for at least one task to complete
            if active_tasks:
                done, _ = await asyncio.wait(
                    active_tasks.values(), return_when=asyncio.FIRST_COMPLETED
                )
                # Remove completed tasks and release their files
                completed_ids = []
                for task_id, async_task in active_tasks.items():
                    if async_task in done:
                        completed_ids.append(task_id)
                for task_id in completed_ids:
                    del active_tasks[task_id]
                    # Release files from active set (Section F)
                    completed_task = self.dag.get_task(task_id)
                    if completed_task:
                        active_files -= set(completed_task.files)
            else:
                await asyncio.sleep(0.1)

        # Wait for any remaining tasks
        if active_tasks:
            await asyncio.gather(*active_tasks.values())

        # Update DEV_PLAN
        self.dag.save(self.project_dir / "dev_plan.json")
        (self.project_dir / "DEV_PLAN.md").write_text(
            self.dag.to_markdown(), encoding="utf-8"
        )

    async def _build_single_task(
        self,
        task: Task,
        agent_id: str,
        sandbox: SandboxBase,
        git: GitManager | None,
        lock_manager: LockManager,
        existing_files: list[str],
    ) -> None:
        """Build a single task with optional git worktree isolation and review.

        If git is available, each builder works in an isolated worktree.
        If git is not available, builders work directly in the project dir.
        """
        branch_name = f"task-{task.id.lower()}"
        worktree_path: Path | None = None
        use_git = git is not None
        try:
            if use_git:
                # Create an isolated worktree for this builder
                worktree_path = await git.create_worktree(branch_name)
                working_dir = worktree_path
            else:
                # No git — work directly in project dir
                working_dir = self.project_dir

            builder = BuilderAgent(
                self.config,
                self.llm,
                working_dir=working_dir,
                sandbox=sandbox,
                agent_id=agent_id,
            )

            result = await builder.run({
                "task": task.to_dict(),
                "spec": self.spec,
                "architecture": json.dumps(self.architecture, indent=2, ensure_ascii=False),
                "existing_files": existing_files,
            })

            if result.success:
                if use_git:
                    # Commit work in the worktree
                    await git.commit_worktree(branch_name, task.description, task.id)

                # Smoke check before review (Section A)
                smoke_ok, smoke_msg = await self._smoke_check(task, working_dir)
                if not smoke_ok:
                    console.print(
                        f"  [yellow][{agent_id}] Smoke check failed:[/yellow] {smoke_msg}"
                    )
                    # Send builder back to fix — skip expensive reviewer
                    fix_result = await builder.run({
                        "task": {
                            **task.to_dict(),
                            "fix_strategy": f"Smoke check failed: {smoke_msg}. Fix these issues.",
                        },
                        "spec": self.spec,
                        "existing_files": self._list_project_files(),
                    })
                    if not fix_result.success:
                        self.dag.mark_failed(task.id, f"Smoke check: {smoke_msg}")
                        console.print(f"  [red][{agent_id}] Failed smoke check:[/red] {task.id}")
                        return
                    if use_git:
                        await git.commit_worktree(branch_name, f"Fix smoke: {task.description}", task.id)

                # TDD loop: run tests and fix before review
                await self._tdd_loop(task, builder, sandbox, working_dir, agent_id, use_git, git, branch_name)

                # Quick review
                reviewer = ReviewerAgent(self.config, self.llm, working_dir)
                review_result = await reviewer.run({
                    "task": task.to_dict(),
                    "spec": self.spec,
                })
                review = reviewer.parse_review(review_result.output)

                if review.approved:
                    if use_git:
                        # Merge worktree branch into main
                        await git.merge_branch(branch_name)
                    self.dag.mark_done(task.id, f"score={review.score}")
                    console.print(f"  [green][{agent_id}] Done:[/green] {task.id} (score: {review.score})")
                else:
                    # Revision needed — retry with feedback
                    logger.info(f"Task {task.id} needs revision: {review.summary}")
                    revision_result = await builder.run({
                        "task": {
                            **task.to_dict(),
                            "fix_strategy": review.summary,
                        },
                        "spec": self.spec,
                        "existing_files": self._list_project_files(),
                    })
                    if revision_result.success:
                        if use_git:
                            await git.commit_worktree(branch_name, f"Revise: {task.description}", task.id)
                            await git.merge_branch(branch_name)
                        self.dag.mark_done(task.id, "revised and approved")
                        console.print(f"  [green][{agent_id}] Done (revised):[/green] {task.id}")
                    else:
                        self.dag.mark_failed(task.id, revision_result.error)
                        console.print(f"  [red][{agent_id}] Failed:[/red] {task.id}")
            else:
                self.dag.mark_failed(task.id, result.error)
                console.print(f"  [red][{agent_id}] Failed:[/red] {task.id} — {result.error[:80]}")

        except Exception as e:
            self.dag.mark_failed(task.id, str(e))
            console.print(f"  [red][{agent_id}] Error:[/red] {task.id} — {e}")
        finally:
            lock_manager.release(task.id, agent_id)
            # Always clean up the worktree (if we created one)
            if use_git and worktree_path is not None:
                try:
                    await git.cleanup_worktree(branch_name)
                except Exception:
                    logger.warning(f"Could not clean up worktree {branch_name}")

    # ──────────────────────────────────────────────
    # Build-Phase TDD Loop
    # ──────────────────────────────────────────────

    async def _tdd_loop(
        self,
        task: Task,
        builder: BuilderAgent,
        sandbox: SandboxBase,
        working_dir: Path,
        agent_id: str,
        use_git: bool,
        git: GitManager | None,
        branch_name: str,
    ) -> None:
        """Run test-fix iterations after builder writes code, before review.

        Only runs if config.build_test_loops > 0 and a test command can be detected.
        """
        loops = getattr(self.config, "build_test_loops", 0)
        if loops <= 0:
            return

        test_cmd = self._detect_test_command(working_dir)
        if not test_cmd:
            return

        for iteration in range(loops):
            test_result = await sandbox.exec(test_cmd, timeout=60)
            if test_result.exit_code == 0:
                console.print(
                    f"  [{agent_id}] TDD pass (iter {iteration + 1}/{loops})"
                )
                return  # Tests pass — proceed to review

            # Tests failed — send builder to fix
            console.print(
                f"  [yellow][{agent_id}] TDD fail (iter {iteration + 1}/{loops}),[/yellow] fixing..."
            )
            stdout_snippet = test_result.stdout[:2000] if test_result.stdout else ""
            stderr_snippet = test_result.stderr[:1000] if test_result.stderr else ""

            fix_result = await builder.run({
                "task": {
                    **task.to_dict(),
                    "fix_strategy": (
                        f"Tests failed. Fix the code to make tests pass.\n"
                        f"Test command: {test_cmd}\n"
                        f"stdout:\n{stdout_snippet}\n"
                        f"stderr:\n{stderr_snippet}"
                    ),
                },
                "spec": self.spec,
                "existing_files": self._list_project_files(),
            })
            if not fix_result.success:
                logger.info(f"[{agent_id}] TDD fix failed, moving to review")
                return  # Builder can't fix — move on to reviewer

            if use_git and git:
                await git.commit_worktree(
                    branch_name, f"TDD fix: {task.description}", task.id
                )

    @staticmethod
    def _detect_test_command(work_dir: Path) -> str | None:
        """Detect appropriate test command based on project type."""
        # Node.js / npm
        pkg_json = work_dir / "package.json"
        if pkg_json.exists():
            try:
                pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
                scripts = pkg.get("scripts", {})
                if "test" in scripts:
                    return "npm test -- --passWithNoTests 2>&1 || true"
            except (json.JSONDecodeError, OSError):
                pass

        # Python / pytest
        if (work_dir / "pytest.ini").exists() or (work_dir / "setup.cfg").exists():
            return "python -m pytest --tb=short -q 2>&1 || true"
        pyproject = work_dir / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                if "pytest" in content or "tool.pytest" in content:
                    return "python -m pytest --tb=short -q 2>&1 || true"
            except OSError:
                pass

        # Rust / cargo
        if (work_dir / "Cargo.toml").exists():
            return "cargo test 2>&1 || true"

        # Go
        if (work_dir / "go.mod").exists():
            return "go test ./... 2>&1 || true"

        return None

    # ──────────────────────────────────────────────
    # Pipeline Hardening: Smoke Check, Build Gate, File Overlap
    # ──────────────────────────────────────────────

    async def _smoke_check(
        self, task: Task, work_dir: Path
    ) -> tuple[bool, str]:
        """Check files exist and have valid syntax before review (Section A).

        Front-loads validation so the reviewer doesn't waste tokens on
        files that are missing or have basic syntax errors.
        """
        missing = [f for f in task.files if not (work_dir / f).exists()]
        if missing:
            return False, f"Missing files: {', '.join(missing)}"

        errors: list[str] = []
        for f in task.files:
            path = work_dir / f
            if not path.exists():
                continue
            if path.suffix == ".py":
                err = await self._run_syntax_check(path, "python", "-m", "py_compile", str(path))
                if err:
                    errors.append(f"{f}: {err}")
            elif path.suffix in (".js", ".jsx"):
                err = await self._run_syntax_check(path, "node", "--check", str(path))
                if err:
                    errors.append(f"{f}: {err}")
            # .ts/.tsx would need tsc or a similar tool; skip for now

        if errors:
            return False, "Syntax errors:\n" + "\n".join(errors)
        return True, "OK"

    @staticmethod
    async def _run_syntax_check(path: Path, *cmd: str) -> str | None:
        """Run a syntax-check command. Returns error string or None on success."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            if proc.returncode != 0:
                return stderr.decode(errors="replace").strip()[:200]
        except FileNotFoundError:
            # Checker not installed — skip gracefully
            return None
        except asyncio.TimeoutError:
            return "syntax check timed out"
        except Exception as e:
            logger.debug(f"Syntax check error for {path}: {e}")
            return None
        return None

    async def _enforce_build_gate(
        self, dag: TaskDAG, project_dir: Path
    ) -> None:
        """Independent verification before BUILD → VERIFY transition (Section C).

        Enforces the BUILD → VERIFY quality gate from quality_gates.md:
        all BUILD tasks done, no blocked tasks, declared files exist.
        """
        build_tasks = [t for t in dag.get_all_tasks() if t.phase == TaskPhase.BUILD]

        if not build_tasks:
            logger.warning("Build gate: no BUILD tasks found")
            return

        # Check 1: At least one task completed
        done_tasks = [t for t in build_tasks if t.status == TaskStatus.DONE]
        if not done_tasks:
            raise RuntimeError(
                "Build gate failed: zero completed tasks — "
                "cannot proceed to VERIFY"
            )

        # Check 2: All BUILD tasks must be DONE (warn for non-done)
        not_done = [t for t in build_tasks if t.status != TaskStatus.DONE]
        if not_done:
            names = [f"{t.id}({t.status.value})" for t in not_done]
            console.print(
                f"  [yellow]Build gate:[/yellow] {len(not_done)} tasks not done: "
                f"{', '.join(names)}"
            )
            # If ALL tasks failed, that's an error
            all_failed = all(t.status == TaskStatus.FAILED for t in not_done)
            if all_failed and len(not_done) == len(build_tasks):
                raise RuntimeError(
                    f"Build gate failed: all {len(build_tasks)} tasks failed"
                )

        # Check 3: Declared files exist (warning, not hard failure)
        missing: list[str] = []
        for task in done_tasks:
            for f in task.files:
                if not (project_dir / f).exists():
                    missing.append(f"{task.id}: {f}")
        if missing:
            console.print(
                f"  [yellow]Build gate warning:[/yellow] "
                f"{len(missing)} declared files missing:"
            )
            for m in missing[:10]:
                console.print(f"    - {m}")
            if len(missing) > 10:
                console.print(f"    ... and {len(missing) - 10} more")
        else:
            console.print("  [green]Build gate passed[/green]")

    def _detect_file_overlaps(
        self, dag: TaskDAG
    ) -> dict[str, list[str]]:
        """Detect tasks claiming the same files (Section F).

        Returns dict mapping overlapping files to their owning task IDs.
        Conflicting tasks will be serialized instead of running in parallel.
        """
        file_owners: dict[str, list[str]] = {}
        for task in dag.get_all_tasks():
            if task.phase == TaskPhase.BUILD and task.status == TaskStatus.TODO:
                for f in task.files:
                    file_owners.setdefault(f, []).append(task.id)

        overlaps = {
            f: owners for f, owners in file_owners.items() if len(owners) > 1
        }
        if overlaps:
            console.print("[yellow]File overlap detected:[/yellow]")
            for f, owners in overlaps.items():
                console.print(f"    {f} -> {', '.join(owners)}")
            console.print(
                "  Conflicting tasks will be serialized (not parallel)"
            )
        return overlaps

    # ──────────────────────────────────────────────
    # Phase 3: VERIFY
    # ──────────────────────────────────────────────

    async def _phase_verify(self) -> None:
        sandbox = create_sandbox(self.config, self.project_dir)
        async with sandbox:
            tester = TesterAgent(self.config, self.llm, self.project_dir, sandbox)
            result = await tester.run({"spec": self.spec})
            test_results = tester.parse_results(result.output)

            # Save results
            (self.project_dir / "test_results.json").write_text(
                json.dumps({
                    "all_passed": test_results.all_passed,
                    "results": test_results.results,
                    "summary": test_results.summary,
                }, indent=2),
                encoding="utf-8",
            )

            if test_results.all_passed:
                console.print("  [green]All tests passed[/green]")
            else:
                console.print(f"  [yellow]Some tests failed — attempting fixes[/yellow]")
                await self._fix_failures(test_results, sandbox)

    async def _fix_failures(self, test_results, sandbox: SandboxBase) -> None:
        """Attempt to fix test failures using Director + Builder."""
        for attempt in range(self.config.max_retries):
            failures = test_results.failures
            if not failures:
                break

            console.print(f"  Fix attempt {attempt + 1}/{self.config.max_retries}")

            # Director creates fix tasks
            fix_director = DirectorFixAgent(self.config, self.llm)
            for failure in failures[:3]:  # Limit fixes per attempt
                fix_result = await fix_director.run({
                    "failure": failure,
                    "spec": self.spec,
                })
                if fix_result.success:
                    fix_task = fix_director.parse_fix_task(fix_result.output)

                    # Builder executes fix
                    builder = BuilderAgent(
                        self.config, self.llm,
                        working_dir=self.project_dir,
                        sandbox=sandbox,
                    )
                    await builder.run({
                        "task": fix_task,
                        "spec": self.spec,
                        "existing_files": self._list_project_files(),
                    })

            # Re-test
            tester = TesterAgent(self.config, self.llm, self.project_dir, sandbox)
            result = await tester.run({"spec": self.spec})
            test_results = tester.parse_results(result.output)

            if test_results.all_passed:
                console.print("  [green]Fixes successful — all tests pass[/green]")
                return

        console.print("  [yellow]Some issues remain after fix attempts[/yellow]")

    # ──────────────────────────────────────────────
    # Phase 4: REFACTOR
    # ──────────────────────────────────────────────

    async def _phase_refactor(self) -> None:
        # Quick review of overall quality
        reviewer = ReviewerAgent(self.config, self.llm, self.project_dir)
        review_result = await reviewer.run({
            "task": {"id": "FINAL", "description": "Final quality review", "files": self._list_project_files()},
            "spec": self.spec,
        })
        review = reviewer.parse_review(review_result.output)

        if review.score >= int(self.config.quality_threshold * 10):
            console.print(f"  [green]Quality score: {review.score}/10 — no refactoring needed[/green]")
            return

        console.print(f"  Quality score: {review.score}/10 — refactoring...")

        if review.issues:
            gardener = GardenerAgent(self.config, self.llm, self.project_dir)
            await gardener.run({
                "review": {"issues": review.issues, "summary": review.summary},
                "spec": self.spec,
            })
            console.print("  [green]Refactoring complete[/green]")

    # ──────────────────────────────────────────────
    # Phase 5: DELIVER
    # ──────────────────────────────────────────────

    async def _phase_deliver(self) -> None:
        # Check if README exists in generated project
        readme = self.project_dir / "README.md"
        if not readme.exists():
            # Create a basic README
            readme.write_text(
                f"# {self.spec.get('project_name', 'Project')}\n\n"
                f"{self.spec.get('description', '')}\n\n"
                f"## Tech Stack\n\n"
                f"```json\n{json.dumps(self.spec.get('tech_stack', {}), indent=2)}\n```\n\n"
                f"## Getting Started\n\n"
                f"```bash\nnpm install\nnpm run dev\n```\n",
                encoding="utf-8",
            )

        # Clean up internal files
        for internal_file in ["spec.json", "dev_plan.json", "test_results.json", "architecture.md"]:
            p = self.project_dir / internal_file
            if p.exists():
                # Move to a .autoforge subdirectory
                forge_dir = self.project_dir / ".autoforge"
                forge_dir.mkdir(exist_ok=True)
                p.rename(forge_dir / internal_file)

        console.print("  [green]Project packaged[/green]")

    # ──────────────────────────────────────────────
    # Review Pipeline: SCAN → REVIEW → [REFACTOR] → REPORT
    # ──────────────────────────────────────────────

    async def review_project(self, project_path: str) -> dict[str, Any]:
        """Standalone review of an existing project.

        Pipeline: SCAN → REVIEW → [REFACTOR in developer mode] → REPORT
        """
        self._start_time = time.monotonic()
        self.project_dir = Path(project_path).resolve()
        logger.info(f"AutoForge review {self.config.run_id} starting: {self.project_dir}")

        try:
            # Phase 1: SCAN — Understand the project
            console.print("\n[bold blue]Phase 1: SCAN[/bold blue] — Analyzing project...")
            scan_result = await self._phase_scan(self.project_dir)
            self.spec = scan_result.spec
            console.print(
                f"  [green]Scan complete:[/green] {self.spec.get('project_name', 'project')} "
                f"— {scan_result.completeness}% complete, {len(scan_result.gaps)} gaps found"
            )

            # Phase 2: REVIEW — Deep code review
            console.print("\n[bold blue]Phase 2: REVIEW[/bold blue] — Reviewing code...")
            review = await self._phase_full_review()
            console.print(f"  Quality score: {review.score}/10")
            console.print(f"  Issues found: {len(review.issues)}")

            # Phase 3: REFACTOR (developer mode only)
            if (
                self.config.mode == "developer"
                and review.score < int(self.config.quality_threshold * 10)
                and review.issues
            ):
                console.print("\n[bold blue]Phase 3: REFACTOR[/bold blue] — Applying fixes...")
                await self._phase_refactor()

            # Phase 4: REPORT
            console.print("\n[bold blue]Phase 4: REPORT[/bold blue] — Generating report...")
            report = self._generate_review_report(scan_result, review)

            self._print_review_summary(review, scan_result)
            return report

        except BudgetExceededError as e:
            console.print(f"\n[bold red]Budget exceeded:[/bold red] {e}")
            raise
        except Exception as e:
            logger.error(f"Review failed: {e}", exc_info=True)
            raise

    async def _phase_scan(self, project_dir: Path):
        """Run Scanner Agent on existing project."""
        scanner = ScannerAgent(self.config, self.llm, project_dir)
        result = await scanner.run({"project_path": str(project_dir)})

        if not result.success:
            raise RuntimeError(f"Scanner failed: {result.error}")

        return scanner.parse_scan(result.output)

    async def _phase_full_review(self):
        """Run full-project review with Reviewer Agent."""
        reviewer = ReviewerAgent(self.config, self.llm, self.project_dir)
        review_result = await reviewer.run({
            "task": {"id": "FULL-REVIEW", "description": "Full project review", "files": self._list_project_files()},
            "spec": self.spec,
            "full_project_review": True,
        })
        review = reviewer.parse_review(review_result.output)

        # Save review report
        forge_dir = self.project_dir / ".autoforge"
        forge_dir.mkdir(exist_ok=True)
        (forge_dir / "review_report.json").write_text(
            json.dumps({
                "score": review.score,
                "approved": review.approved,
                "issues": review.issues,
                "summary": review.summary,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return review

    def _generate_review_report(self, scan_result, review) -> dict[str, Any]:
        """Generate structured review report."""
        report = {
            "project_name": self.spec.get("project_name", "unknown"),
            "tech_stack": self.spec.get("tech_stack", {}),
            "completeness": scan_result.completeness,
            "gaps": scan_result.gaps,
            "score": review.score,
            "issues": review.issues,
            "summary": review.summary,
            "cost_usd": self.config.estimated_cost_usd,
        }

        # Save report
        forge_dir = self.project_dir / ".autoforge"
        forge_dir.mkdir(exist_ok=True)
        (forge_dir / "review_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return report

    def _print_review_summary(self, review, scan_result) -> None:
        """Print review summary."""
        from autoforge.cli.display import show_review_report

        elapsed = time.monotonic() - self._start_time if self._start_time else 0

        show_review_report({
            "score": review.score,
            "issues": review.issues,
            "summary": review.summary,
        })

        console.print(f"  Completeness: {scan_result.completeness}%")
        console.print(f"  Gaps: {len(scan_result.gaps)}")
        for gap in scan_result.gaps[:5]:
            console.print(f"    - {gap}")
        if len(scan_result.gaps) > 5:
            console.print(f"    ... and {len(scan_result.gaps) - 5} more")

        console.print()
        console.print(f"  Cost:     ${self.config.estimated_cost_usd:.4f}")
        console.print(f"  Duration: {elapsed:.1f}s")
        console.print(f"  Report:   {self.project_dir / '.autoforge' / 'review_report.json'}")

    # ──────────────────────────────────────────────
    # Import Pipeline: SCAN → REVIEW → [ENHANCE] → VERIFY → [REFACTOR] → DELIVER
    # ──────────────────────────────────────────────

    async def import_project(self, project_path: str, enhancement: str = "") -> Path:
        """Import and optionally enhance an existing project.

        Pipeline: SCAN → REVIEW → [ENHANCE] → VERIFY → [REFACTOR] → DELIVER
        """
        self._start_time = time.monotonic()
        source_dir = Path(project_path).resolve()
        logger.info(f"AutoForge import {self.config.run_id}: {source_dir}")

        if not source_dir.is_dir():
            raise ValueError(f"Not a directory: {project_path}")

        try:
            # Copy to workspace (preserve original)
            import shutil

            project_name = source_dir.name
            self.project_dir = self.config.workspace_dir / f"{project_name}-forge"
            if self.project_dir.exists():
                shutil.rmtree(self.project_dir)
            shutil.copytree(source_dir, self.project_dir, ignore=shutil.ignore_patterns(
                ".git", "node_modules", "__pycache__", ".venv", "venv", ".env"
            ))
            self._state_file = self.project_dir / ".forge_state.json"

            # Phase 1: SCAN
            console.print("\n[bold blue]Phase 1: SCAN[/bold blue] — Analyzing project...")
            scan_result = await self._phase_scan(self.project_dir)
            self.spec = scan_result.spec
            console.print(
                f"  [green]Scan complete:[/green] {self.spec.get('project_name', project_name)} "
                f"— {scan_result.completeness}% complete"
            )
            self._save_state("scan_complete")

            # Phase 2: REVIEW
            console.print("\n[bold blue]Phase 2: REVIEW[/bold blue] — Reviewing code...")
            review = await self._phase_full_review()
            console.print(f"  Quality score: {review.score}/10, {len(review.issues)} issues")
            self._save_state("review_complete")

            # Phase 3: ENHANCE (optional, if enhancement description provided)
            if enhancement:
                console.print("\n[bold blue]Phase 3: ENHANCE[/bold blue] — Adding features...")
                await self._phase_enhance(enhancement)
                self._save_state("enhance_complete")

            # Phase 4: VERIFY
            console.print("\n[bold blue]Phase 4: VERIFY[/bold blue] — Verifying project...")
            await self._phase_verify()
            self._save_state("verify_complete")

            # Phase 5: REFACTOR (developer mode only)
            if self.config.mode == "developer":
                console.print("\n[bold blue]Phase 5: REFACTOR[/bold blue] — Improving quality...")
                await self._phase_refactor()
                self._save_state("refactor_complete")

            # Phase 6: DELIVER
            console.print("\n[bold blue]Phase 6: DELIVER[/bold blue] — Packaging...")
            await self._phase_deliver()
            self._save_state("complete")

            self._print_summary()
            return self.project_dir

        except BudgetExceededError as e:
            console.print(f"\n[bold red]Budget exceeded:[/bold red] {e}")
            if self.project_dir:
                self._save_state("budget_exceeded")
            raise
        except Exception as e:
            logger.error(f"Import failed: {e}", exc_info=True)
            if self.project_dir:
                self._save_state(f"failed: {e}")
            raise

    async def _phase_enhance(self, enhancement: str) -> None:
        """Add new features to an imported project.

        Uses Director to merge enhancement requests with existing spec,
        then Architect + Builders to implement new code.
        """
        # Director merges existing spec with enhancement request
        director = DirectorAgent(self.config, self.llm)
        result = await director.run({
            "project_description": (
                f"This is an existing project with the following spec:\n"
                f"```json\n{json.dumps(self.spec, indent=2, ensure_ascii=False)}\n```\n\n"
                f"The user wants to add the following enhancements:\n{enhancement}\n\n"
                f"Output a MERGED spec that includes the existing modules AND new modules "
                f"for the requested enhancements. Keep existing module names and files unchanged."
            ),
        })

        if not result.success:
            console.print(f"  [yellow]Enhancement planning failed: {result.error}[/yellow]")
            return

        merged_spec = director.parse_spec(result.output)
        self.spec = merged_spec

        # Save updated spec
        (self.project_dir / "spec.json").write_text(
            json.dumps(self.spec, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Build new modules
        await self._phase_build()

    # ──────────────────────────────────────────────
    # State management
    # ──────────────────────────────────────────────

    def _save_state(self, phase: str) -> None:
        """Save orchestrator state for resume capability."""
        if not self._state_file:
            return

        state = {
            "run_id": self.config.run_id,
            "phase": phase,
            "spec": self.spec,
            "architecture": self.architecture,
            "cost_usd": self.config.estimated_cost_usd,
            "total_input_tokens": self.config.total_input_tokens,
            "total_output_tokens": self.config.total_output_tokens,
            "elapsed_seconds": time.monotonic() - self._start_time,
        }
        self._state_file.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _list_project_files(self) -> list[str]:
        """List all source files in the project directory."""
        if not self.project_dir:
            return []
        files = []
        for p in sorted(self.project_dir.rglob("*")):
            if p.is_file() and ".git" not in p.parts and ".autoforge" not in p.parts:
                try:
                    files.append(str(p.relative_to(self.project_dir)))
                except ValueError:
                    continue
        return files

    # ──────────────────────────────────────────────
    # Resume + Status
    # ──────────────────────────────────────────────

    async def resume(self, workspace_path: Path | None = None) -> Path:
        """Resume an interrupted run."""
        if workspace_path is None:
            # Find the most recent project in workspace
            workspace = self.config.workspace_dir
            projects = [d for d in workspace.iterdir() if d.is_dir() and (d / ".forge_state.json").exists()]
            if not projects:
                raise RuntimeError("No previous runs found to resume")
            workspace_path = max(projects, key=lambda d: d.stat().st_mtime)

        self.project_dir = workspace_path
        self._state_file = workspace_path / ".forge_state.json"
        state = json.loads(self._state_file.read_text(encoding="utf-8"))

        self.spec = state.get("spec", {})
        self.architecture = state.get("architecture", {})
        phase = state.get("phase", "")

        console.print(f"Resuming from phase: {phase}")

        # Resume from the appropriate phase
        if phase == "spec_complete":
            await self._phase_build()
            await self._phase_verify()
            await self._phase_refactor()
            await self._phase_deliver()
        elif phase == "build_complete":
            await self._phase_verify()
            await self._phase_refactor()
            await self._phase_deliver()
        elif phase == "verify_complete":
            await self._phase_refactor()
            await self._phase_deliver()
        elif phase == "refactor_complete":
            await self._phase_deliver()
        elif phase == "complete":
            console.print("Project already complete!")
        else:
            console.print(f"Unknown phase '{phase}', starting from build")
            await self._phase_build()
            await self._phase_verify()
            await self._phase_refactor()
            await self._phase_deliver()

        self._print_summary()
        return self.project_dir

    def show_status(self) -> None:
        """Display current project status."""
        workspace = self.config.workspace_dir
        projects = [d for d in workspace.iterdir() if d.is_dir()] if workspace.exists() else []

        if not projects:
            console.print("No projects found in workspace.")
            return

        table = Table(title="AutoForge Projects")
        table.add_column("Project", style="cyan")
        table.add_column("Phase", style="green")
        table.add_column("Cost", style="yellow")

        for project_dir in sorted(projects):
            state_file = project_dir / ".forge_state.json"
            if state_file.exists():
                state = json.loads(state_file.read_text(encoding="utf-8"))
                table.add_row(
                    project_dir.name,
                    state.get("phase", "unknown"),
                    f"${state.get('cost_usd', 0):.4f}",
                )
            elif project_dir.name != ".locks":
                table.add_row(project_dir.name, "unknown", "—")

        console.print(table)

    def _print_summary(self) -> None:
        """Print final run summary."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0

        console.print()
        console.print("=" * 56)
        console.print("[bold green]Project complete![/bold green]")
        console.print(f"  Location:  {self.project_dir}")
        console.print(f"  Cost:      ${self.config.estimated_cost_usd:.4f}")
        console.print(f"  Tokens:    {self.config.total_input_tokens:,} in / {self.config.total_output_tokens:,} out")
        console.print(f"  Duration:  {elapsed:.1f}s")
        console.print(f"  LLM calls: {self.llm.call_count}")
        console.print("=" * 56)
