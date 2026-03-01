"""Orchestrator — the brain of AutoForge.

Drives the 5-phase pipeline: SPEC → BUILD → VERIFY → REFACTOR → DELIVER.
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

from engine.agents.architect import ArchitectAgent
from engine.agents.builder import BuilderAgent
from engine.agents.director import DirectorAgent, DirectorFixAgent
from engine.agents.gardener import GardenerAgent
from engine.agents.reviewer import ReviewerAgent
from engine.agents.tester import TesterAgent
from engine.config import ForgeConfig
from engine.git_manager import GitManager
from engine.llm_router import BudgetExceededError, LLMRouter
from engine.lock_manager import LockManager
from engine.sandbox import SandboxBase, create_sandbox
from engine.task_dag import Task, TaskDAG, TaskPhase, TaskStatus

logger = logging.getLogger(__name__)
console = Console()


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
            console.print(f"  [green]Spec generated:[/green] {project_name} — {len(self.spec.get('modules', []))} modules")

            # Phase 2: BUILD
            console.print("\n[bold blue]Phase 2: BUILD[/bold blue] — Building project...")
            await self._phase_build()
            self._save_state("build_complete")

            # Phase 3: VERIFY
            console.print("\n[bold blue]Phase 3: VERIFY[/bold blue] — Verifying project...")
            await self._phase_verify()
            self._save_state("verify_complete")

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

        # Step 2: Initialize git repo for the project
        git = GitManager(self.project_dir)
        await git.init_repo()

        # Step 3: Build tasks
        sandbox = create_sandbox(self.config, self.project_dir)
        lock_manager = LockManager(self.config.workspace_dir / ".locks")
        lock_manager.clear_all()

        async with sandbox:
            await self._execute_build_tasks(git, sandbox, lock_manager)

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
        git: GitManager,
        sandbox: SandboxBase,
        lock_manager: LockManager,
    ) -> None:
        """Execute build tasks, potentially in parallel."""
        max_parallel = self.config.max_agents
        active_tasks: dict[str, asyncio.Task] = {}

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

                agent_id = f"builder-{len(active_tasks):02d}"
                if not lock_manager.enforce_single_task(agent_id):
                    continue

                if lock_manager.try_claim(task.id, agent_id):
                    self.dag.mark_in_progress(task.id, agent_id)
                    console.print(f"  [{agent_id}] Building: {task.id} — {task.description[:60]}")

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
                # Remove completed tasks
                completed_ids = []
                for task_id, async_task in active_tasks.items():
                    if async_task in done:
                        completed_ids.append(task_id)
                for task_id in completed_ids:
                    del active_tasks[task_id]
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
        git: GitManager,
        lock_manager: LockManager,
        existing_files: list[str],
    ) -> None:
        """Build a single task with git worktree isolation and review."""
        branch_name = f"task-{task.id.lower()}"
        worktree_path: Path | None = None
        try:
            # Create an isolated worktree for this builder
            worktree_path = await git.create_worktree(branch_name)
            working_dir = worktree_path

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
                # Commit work in the worktree
                await git.commit_worktree(branch_name, task.description, task.id)

                # Quick review (reviewer reads from worktree)
                reviewer = ReviewerAgent(self.config, self.llm, working_dir)
                review_result = await reviewer.run({
                    "task": task.to_dict(),
                    "spec": self.spec,
                })
                review = reviewer.parse_review(review_result.output)

                if review.approved:
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
            # Always clean up the worktree
            if worktree_path is not None:
                try:
                    await git.cleanup_worktree(branch_name)
                except Exception:
                    logger.warning(f"Could not clean up worktree {branch_name}")

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
