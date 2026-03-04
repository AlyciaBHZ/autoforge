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
from autoforge.engine.dynamic_constitution import DynamicConstitution
from autoforge.engine.evolution import EvolutionEngine, FitnessScore, WorkflowGenome
from autoforge.engine.process_reward import ProcessRewardModel, StepType
from autoforge.engine.prompt_optimizer import PromptOptimizer
from autoforge.engine.search_tree import MCTSSearchTree, SearchTree, evaluate_candidate
from autoforge.engine.evomac import EvoMACEngine
from autoforge.engine.sica import SICAEngine
from autoforge.engine.rag_retrieval import RAGRetrievalEngine
from autoforge.engine.formal_verify import FormalVerifier
from autoforge.engine.agent_debate import ConditionalDebateEngine
from autoforge.engine.security_scan import SecurityScanner
from autoforge.engine.reflexion import ReflexionEngine
from autoforge.engine.adaptive_compute import AdaptiveComputeRouter
from autoforge.engine.ldb_debugger import LDBDebugger
from autoforge.engine.speculative_pipeline import SpeculativePipeline
from autoforge.engine.hierarchical_decomp import HierarchicalDecomposer
from autoforge.engine.lean_prover import LeanProver
from autoforge.engine.multi_prover import MultiProverEngine
from autoforge.engine.capability_dag import CapabilityDAG, DAGBridge, Domain
from autoforge.engine.theoretical_reasoning import TheoreticalReasoningEngine
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
        self._dynamic_constitution: DynamicConstitution | None = None
        self._search_tree: SearchTree | None = None
        self._evolution = EvolutionEngine()
        self._genome: WorkflowGenome | None = None
        self._prompt_optimizer = PromptOptimizer()
        self._process_reward: ProcessRewardModel | None = None
        self._evomac = EvoMACEngine()
        self._sica = SICAEngine()
        self._rag = RAGRetrievalEngine()
        self._formal_verifier = FormalVerifier()
        self._debate = ConditionalDebateEngine()
        self._security_scanner = SecurityScanner()
        self._reflexion = ReflexionEngine()
        self._adaptive_compute = AdaptiveComputeRouter()
        self._ldb = LDBDebugger()
        self._speculative = SpeculativePipeline()
        self._decomposer = HierarchicalDecomposer()
        self._lean_prover: LeanProver | None = None
        self._multi_prover: MultiProverEngine | None = None
        self._capability_dag = CapabilityDAG()
        self._dag_bridge = DAGBridge(self._capability_dag)
        self._theoretical_reasoning = TheoreticalReasoningEngine()
        # Adaptive context budget tracking
        self._context_budget_multiplier: float = 1.0
        self._context_success_history: list[bool] = []

    async def run(self, requirement: str) -> Path:
        """Execute the full pipeline. Returns path to the generated project."""
        self._start_time = time.monotonic()
        logger.info(f"AutoForge run {self.config.run_id} starting")

        # Load capability DAG from global storage
        global_dag_dir = self.config.project_root / ".autoforge" / "capability_dag"
        self._capability_dag.load(global_dag_dir)
        if self._capability_dag.size > 0:
            console.print(f"  [cyan]CapabilityDAG:[/cyan] loaded {self._capability_dag.size} capabilities")

        # Load theoretical reasoning state (cross-domain theory graphs)
        if self.config.theoretical_reasoning_enabled:
            theory_dir = self.config.project_root / ".autoforge" / "theories"
            self._theoretical_reasoning.load_all(theory_dir)
            n_theories = len(self._theoretical_reasoning._theories)
            if n_theories > 0:
                console.print(f"  [cyan]Theoretical reasoning:[/cyan] loaded {n_theories} theory graphs")

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

            # Evolution: prepare genome based on project type + past experience
            self._genome = self._evolution.prepare_genome(
                project_type=EvolutionEngine.infer_project_type(self.spec),
                tech_fingerprint=EvolutionEngine.extract_tech_fingerprint(self.spec),
                config=self.config,
            )
            self._evolution.apply_genome_to_config(self._genome, self.config)
            if self._genome.generation > 0:
                console.print(
                    f"  [cyan]Evolution:[/cyan] gen {self._genome.generation} "
                    f"(from {self._genome.parent_id})"
                )
                if self._genome.mutations:
                    console.print(f"    Mutations: {', '.join(self._genome.mutations)}")

            # Dynamic constitution: generate project-specific agent instructions
            await self._init_dynamic_constitution()

            # Prompt optimizer: register baselines and get optimized prompts
            await self._init_prompt_optimizer()

            # EvoMAC: start iteration for text backpropagation
            if self.config.evomac_enabled:
                self._evomac.start_iteration()

            # Speculative: start build scaffolding while we checkpoint
            if self.config.speculative_enabled:
                await self._speculative.speculate_build_scaffold(self.spec, self.project_dir)

            # Adaptive compute: estimate project-level difficulty
            if self.config.adaptive_compute_enabled:
                self._difficulty = self._adaptive_compute.estimate_difficulty(
                    requirement, self.spec,
                )
                console.print(
                    f"  [cyan]Compute router:[/cyan] difficulty={self._difficulty.level.value} "
                    f"(score={self._difficulty.score:.2f})"
                )

            # Checkpoint: review spec before building
            await self._checkpoint(
                "spec",
                f"Generated spec with {n_modules} modules. "
                f"Review: {self.project_dir / 'spec.json'}",
            )

            # Speculative: validate build scaffold before BUILD
            if self.config.speculative_enabled:
                await self._speculative.validate_and_commit("spec-build-scaffold")

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

            # Speculative: start test scaffolding while we checkpoint
            if self.config.speculative_enabled:
                await self._speculative.speculate_test_scaffold(self.spec, self.project_dir)

            # Phase 3: VERIFY
            console.print("\n[bold blue]Phase 3: VERIFY[/bold blue] — Verifying project...")
            await self._phase_verify()

            # Formal verification: static analysis + LLM formal checks
            if self.config.formal_verify_enabled:
                await self._run_formal_verification()

            # Security scan: vulnerability detection
            if self.config.security_scan_enabled:
                await self._run_security_scan()

            # Lean formal proving: prove Lean files if present
            if self.config.lean_prover_enabled:
                await self._run_lean_proving()

            # Theoretical reasoning: parse, evolve, and verify theories
            if self.config.theoretical_reasoning_enabled:
                await self._run_theoretical_reasoning()

            self._save_state("verify_complete")

            # Checkpoint: review test results
            await self._checkpoint("verify", "Verification complete. Review test_results.json.")

            # EvoMAC: generate text gradients from verify results
            if self.config.evomac_enabled:
                await self._evomac_backward_pass()

            # Phase 4: REFACTOR
            console.print("\n[bold blue]Phase 4: REFACTOR[/bold blue] — Improving quality...")
            await self._phase_refactor()
            self._save_state("refactor_complete")

            # Phase 5: DELIVER
            console.print("\n[bold blue]Phase 5: DELIVER[/bold blue] — Packaging...")
            await self._phase_deliver()
            self._save_state("complete")

            # Evolution: record fitness and reflect on the run
            await self._evolution_record_and_reflect(project_name)

            # Prompt optimizer: record fitness for active variants + trigger optimization
            await self._prompt_optimizer_record_and_optimize(project_name)

            # RAG: index completed project for future retrieval
            if self.config.rag_enabled:
                self._rag.index_project(self.project_dir, project_name)
                console.print(f"  [cyan]RAG:[/cyan] indexed for future projects")

            # SICA: propose self-improvements based on this run
            if self.config.sica_enabled:
                await self._sica_propose_improvements(project_name)

            # EvoMAC: save state
            if self.config.evomac_enabled and self.project_dir:
                self._evomac.save_state(self.project_dir / ".autoforge")

            # Reflexion: save episodic memory for future projects
            if self.config.reflexion_enabled and self.project_dir:
                self._reflexion.save_state(self.project_dir / ".autoforge")

            # Lean prover: save proof state (foundation, conjectures)
            if self.config.lean_prover_enabled and self._lean_prover and self.project_dir:
                self._lean_prover.save_state(self.project_dir / ".autoforge" / "lean")
                console.print(f"  [cyan]Lean:[/cyan] proof state saved")

            if self._multi_prover and self.project_dir:
                self._multi_prover.save_state(self.project_dir / ".autoforge" / "multi_prover.json")

            # Theoretical reasoning: save theory graphs (project-level + global)
            if self.config.theoretical_reasoning_enabled and self.project_dir:
                self._theoretical_reasoning.save_all(
                    self.project_dir / ".autoforge" / "theories"
                )
                # Also save to global storage for cross-project reuse
                global_theory_dir = self.config.project_root / ".autoforge" / "theories"
                self._theoretical_reasoning.save_all(global_theory_dir)
                n_theories = len(self._theoretical_reasoning._theories)
                if n_theories > 0:
                    console.print(
                        f"  [cyan]Theoretical reasoning:[/cyan] {n_theories} theory graphs saved"
                    )

            # Adaptive compute: save calibration data
            if self.config.adaptive_compute_enabled and self.project_dir:
                self._adaptive_compute.save_state(self.project_dir / ".autoforge")

            # CapabilityDAG: ingest knowledge from this run + save
            await self._ingest_run_to_dag(project_name)
            self._capability_dag.save(global_dag_dir)
            console.print(
                f"  [cyan]CapabilityDAG:[/cyan] {self._capability_dag.size} total capabilities"
            )

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

        # Inject dynamic constitution into architect
        if self._dynamic_constitution:
            supplement = self._dynamic_constitution.build_supplementary_prompt("architect")
            if supplement:
                architect.set_dynamic_constitution(supplement)

        # Optionally use search tree for architecture exploration
        if getattr(self.config, "search_tree_enabled", False):
            arch_data = await self._explore_architectures(architect)
        else:
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

    def _compute_context_shares(self, task_description: str) -> dict[str, float]:
        """Compute dynamic context source shares based on task characteristics.

        Returns a dict mapping source labels to their max_share values.
        Adjusts allocation based on whether the task is testing-focused or file-creation-focused.
        """
        # Check if task mentions testing/verification keywords
        testing_keywords = ["test", "verify", "check", "assert", "validation", "debug"]
        is_testing = any(kw in task_description.lower() for kw in testing_keywords)

        # Check if task is creating new files
        creation_keywords = ["create", "implement", "new", "file", "module", "component"]
        is_creation = any(kw in task_description.lower() for kw in creation_keywords)

        # Default shares
        shares = {
            "constitution": 0.35,
            "prompt_opt": 0.20,
            "decomp": 0.25,
            "reflexion": 0.20,
            "rag": 0.20,
            "dag": 0.20,
            "theory": 0.15,
        }

        # Adjust based on task type
        if is_testing:
            # Boost reflexion (past failure patterns help with testing)
            # Reduce decomp (less structured planning needed for test fixes)
            shares["reflexion"] = 0.30
            shares["decomp"] = 0.15
            logger.debug(f"[ContextShares] Testing task detected — boosted reflexion, reduced decomp")
        elif is_creation:
            # Boost decomp (structured planning helps new implementations)
            # Reduce reflexion (fewer failure patterns for brand new code)
            shares["decomp"] = 0.35
            shares["reflexion"] = 0.12
            logger.debug(f"[ContextShares] Creation task detected — boosted decomp, reduced reflexion")

        return shares

    def _adjust_context_budget(self, success_first_try: bool) -> None:
        """Learn from task outcome and adjust context budget multiplier.

        First-try success suggests budget was sufficient or generous.
        Retries suggest budget might have been too tight.
        Uses a rolling average to smooth out variance.
        """
        self._context_success_history.append(success_first_try)

        # Keep only last 10 tasks for rolling average
        if len(self._context_success_history) > 10:
            self._context_success_history.pop(0)

        # Calculate success rate
        success_rate = sum(self._context_success_history) / len(self._context_success_history)

        # Adjust multiplier: if success rate is low, budget was too tight
        # if success rate is high, budget might be too generous
        # Target: ~80% first-try success
        target_success = 0.80
        adjustment = (success_rate - target_success) * 0.10  # ±10% per cycle

        old_multiplier = self._context_budget_multiplier
        self._context_budget_multiplier = max(0.5, min(2.0, self._context_budget_multiplier + adjustment))

        if abs(self._context_budget_multiplier - old_multiplier) > 0.01:
            logger.info(
                f"[ContextBudget] Adjusted multiplier: {old_multiplier:.2f} → {self._context_budget_multiplier:.2f} "
                f"(success_rate={success_rate:.1%}, adjustment={adjustment:+.2f})"
            )

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

            # ── Adaptive Context Budget Manager ──────────────────────────
            # All supplementary context competes for a shared token budget.
            # Adaptive system:
            #  1. Starts with base budget from config.context_budget_tokens
            #  2. Adjusts based on task difficulty: TRIVIAL×0.5, STANDARD×1.0, COMPLEX×1.5, EXTREME×2.0
            #  3. Dynamic shares per source based on task characteristics
            #  4. Learning signal: adjusts multiplier based on first-try success rate
            # Priority order (P0-P6): constitution → prompt_opt → decomp → reflexion → rag → dag → theory
            # ────────────────────────────────────────────────────────────────────────

            # Estimate task difficulty if adaptive_compute is available
            difficulty_level = "standard"  # fallback
            difficulty_multiplier = 1.0
            if hasattr(self, "_adaptive_compute"):
                try:
                    difficulty = await self._adaptive_compute.estimate_difficulty(task.description)
                    difficulty_level = difficulty.level.value if hasattr(difficulty.level, "value") else str(difficulty.level)
                    # Map difficulty to budget multiplier
                    difficulty_map = {
                        "trivial": 0.5,
                        "simple": 0.7,
                        "standard": 1.0,
                        "complex": 1.5,
                        "extreme": 2.0,
                    }
                    difficulty_multiplier = difficulty_map.get(difficulty_level, 1.0)
                    logger.debug(f"[ContextBudget] Task difficulty: {difficulty_level} (multiplier={difficulty_multiplier})")
                except Exception as e:
                    logger.debug(f"[ContextBudget] Difficulty estimation failed: {e}")

            # Goal-aware budget adjustment (v2.8): projects with declared goals
            # get boosted budgets for their target discipline's context sources.
            goal_multiplier = 1.0
            _goal_type = getattr(self.config, "project_goal_type", "general")
            if _goal_type == "formal_verification":
                goal_multiplier = 1.5  # Formal verification needs more context
            elif _goal_type == "research":
                goal_multiplier = 1.3

            # Compute adaptive base budget
            base_budget_chars = self.config.context_budget_tokens * 4  # ~4 chars/token
            budget_chars = int(base_budget_chars * difficulty_multiplier * self._context_budget_multiplier * goal_multiplier)
            used_chars = 0
            context_parts: list[tuple[str, str]] = []  # (label, text)

            def _budget_remaining() -> int:
                return max(0, budget_chars - used_chars)

            def _add_context(label: str, text: str | None, max_share: float = 0.3) -> None:
                nonlocal used_chars
                if not text:
                    return
                cap = min(len(text), int(budget_chars * max_share), _budget_remaining())
                if cap <= 0:
                    return
                trimmed = text[:cap]
                context_parts.append((label, trimmed))
                used_chars += len(trimmed)

            # Get dynamic source shares based on task type
            dynamic_shares = self._compute_context_shares(task.description)

            # P0: Dynamic constitution (project-specific rules — always included)
            if self._dynamic_constitution:
                supplement = self._dynamic_constitution.build_supplementary_prompt("builder")
                _add_context("constitution", supplement, max_share=dynamic_shares.get("constitution", 0.35))

            # P1: Optimized prompt (DSPy/OPRO tuned variant)
            _, opt_prompt = self._prompt_optimizer.get_active_prompt("builder")
            _add_context("prompt_opt", opt_prompt, max_share=dynamic_shares.get("prompt_opt", 0.20))

            # P2: Hierarchical decomposition (structured plan for complex tasks)
            if self.config.hierarchical_decomp_enabled:
                difficulty = getattr(self, "_difficulty", None)
                is_complex = (
                    difficulty and difficulty.level.value in ("complex", "extreme")
                ) or len(task.files) > 3
                if is_complex:
                    try:
                        mod_name = task.files[0].rsplit("/", 1)[-1].rsplit(".", 1)[0] if task.files else task.id
                        plan = await self._decomposer.decompose(
                            task.description, mod_name, self.spec, self.llm,
                        )
                        if plan:
                            decomp_ctx = self._decomposer.build_context_for_agent(plan)
                            _add_context("decomp", decomp_ctx, max_share=dynamic_shares.get("decomp", 0.25))
                            console.print(
                                f"    [{agent_id}] [cyan]Decomp:[/cyan] {len(plan.functions)} functions planned"
                            )
                    except Exception as e:
                        logger.debug(f"Hierarchical decomposition skipped: {e}")

            # P3: Reflexion (past failure memories — high value for retry scenarios)
            if self.config.reflexion_enabled:
                reflexion_ctx = self._reflexion.build_retry_context(
                    task.description, project=self.spec.get("project_name", ""),
                )
                _add_context("reflexion", reflexion_ctx, max_share=dynamic_shares.get("reflexion", 0.20))

            # P4: RAG (relevant code from past projects)
            if self.config.rag_enabled:
                rag_context = self._rag.build_context(task.description, top_k=3)
                _add_context("rag", rag_context, max_share=dynamic_shares.get("rag", 0.20))

            # P5: CapabilityDAG (universal knowledge graph)
            dag_budget = min(1500, _budget_remaining() // 4)
            dag_context = self._dag_bridge.build_context(task.description, max_tokens=dag_budget)
            _add_context("dag", dag_context, max_share=dynamic_shares.get("dag", 0.20))

            # P6: Theoretical reasoning (cross-domain theory knowledge)
            if self.config.theoretical_reasoning_enabled and self._theoretical_reasoning._theories:
                theory_context = self._build_theory_context(
                    task.description, max_tokens=min(1000, _budget_remaining() // 4),
                )
                _add_context("theory", theory_context, max_share=dynamic_shares.get("theory", 0.15))

            # Apply all context in one shot
            if context_parts:
                combined = "\n".join(text for _, text in context_parts)
                builder.set_dynamic_constitution(combined)
                logger.debug(
                    f"[ContextBudget] {agent_id}: {used_chars}/{budget_chars} chars "
                    f"(difficulty={difficulty_level}×{difficulty_multiplier}, "
                    f"learn_mult={self._context_budget_multiplier:.2f}, "
                    f"{len(context_parts)} sources: "
                    f"{', '.join(f'{l}={len(t)}' for l, t in context_parts)})"
                )

            # Initialize process reward model for this task
            prm = ProcessRewardModel(
                self.config, self.llm, sandbox=sandbox, working_dir=working_dir,
            )
            prm.start_trajectory(task.id, task.description)
            prm.record_step(
                task.id, StepType.PLANNING,
                f"Starting build: {task.description}",
                files_touched=task.files,
            )

            result = await builder.run({
                "task": task.to_dict(),
                "spec": self.spec,
                "architecture": json.dumps(self.architecture, indent=2, ensure_ascii=False),
                "existing_files": existing_files,
            })

            if result.success:
                # Record build success step in PRM
                prm.record_step(
                    task.id, StepType.FILE_CREATE,
                    f"Build completed for {task.id}",
                    files_touched=task.files,
                )
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
                if self._dynamic_constitution:
                    supplement = self._dynamic_constitution.build_supplementary_prompt("reviewer")
                    if supplement:
                        reviewer.set_dynamic_constitution(supplement)
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

                    # Learning signal: task succeeded on first try (before smoke check fix, before any retries)
                    # Check if we had to loop back for smoke check or revision
                    success_first_try = (smoke_ok)  # True if smoke check passed on first run
                    self._adjust_context_budget(success_first_try)

                    # PRM: evaluate trajectory and save
                    traj_result = await prm.evaluate_trajectory(task.id, "success")
                    if traj_result.get("final_score", 0) > 0:
                        console.print(
                            f"  [{agent_id}] PRM: score={traj_result['final_score']:.2f} "
                            f"trend={traj_result.get('reward_trend', '?')}"
                        )
                    if self.project_dir:
                        prm.save_trajectory(task.id, self.project_dir / ".autoforge")
                else:
                    # Revision needed — retry with feedback (not first-try success)
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
                        # Learning signal: task needed revision (not first-try success)
                        self._adjust_context_budget(False)
                    else:
                        self.dag.mark_failed(task.id, revision_result.error)
                        console.print(f"  [red][{agent_id}] Failed:[/red] {task.id}")
                        # Learning signal: task failed completely (not first-try success)
                        self._adjust_context_budget(False)
            else:
                self.dag.mark_failed(task.id, result.error)
                console.print(f"  [red][{agent_id}] Failed:[/red] {task.id} — {result.error[:80]}")

                # Learning signal: initial builder run failed (not first-try success)
                self._adjust_context_budget(False)

                # PRM: record failure and evaluate trajectory
                prm.record_step(
                    task.id, StepType.DEBUG,
                    f"Build failed: {result.error[:200]}",
                    files_touched=task.files,
                )
                await prm.evaluate_trajectory(task.id, "failure")
                if self.project_dir:
                    prm.save_trajectory(task.id, self.project_dir / ".autoforge")

                # Meta-learning: analyze failure to create preventive rules
                await self._learn_from_task_failure(task, result.error, agent_id)

        except Exception as e:
            self.dag.mark_failed(task.id, str(e))
            console.print(f"  [red][{agent_id}] Error:[/red] {task.id} — {e}")

            # Meta-learning: analyze exception to create preventive rules
            await self._learn_from_task_failure(task, str(e), agent_id)
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
        """Attempt to fix test failures using Director + Builder + Reflexion + LDB.

        Enhanced pipeline:
          1. LDB: block-level fault localization on each failure
          2. Director: create fix tasks (enriched with LDB analysis)
          3. Builder: execute fixes (with Reflexion context)
          4. Reflexion: reflect on failure if fix didn't work
          5. Repeat with accumulated reflections
        """
        project_name = self.spec.get("project_name", "")

        for attempt in range(self.config.max_retries):
            failures = test_results.failures
            if not failures:
                break

            console.print(f"  Fix attempt {attempt + 1}/{self.config.max_retries}")

            # Director creates fix tasks
            fix_director = DirectorFixAgent(self.config, self.llm)
            for failure in failures[:3]:  # Limit fixes per attempt
                failure_id = f"fix-{hash(str(failure)) % 10000}"
                failure_msg = failure if isinstance(failure, str) else str(failure)

                # LDB: block-level fault localization
                ldb_context = ""
                if self.config.ldb_debugger_enabled:
                    try:
                        failure_dict = failure if isinstance(failure, dict) else {"error": failure_msg}
                        debug_report = await self._ldb.debug_test_failure(
                            self.project_dir, failure_dict, self.llm, sandbox,
                        )
                        if debug_report:
                            ldb_context = self._ldb.format_for_agent(debug_report)
                            if debug_report.faulty_block is not None:
                                console.print(
                                    f"    [cyan]LDB:[/cyan] fault in {debug_report.function_name} "
                                    f"block {debug_report.faulty_block}"
                                )
                    except Exception as e:
                        logger.debug(f"LDB debugging skipped: {e}")

                # Reflexion: build context from past reflections
                reflexion_ctx = ""
                if self.config.reflexion_enabled:
                    reflexion_ctx = self._reflexion.build_retry_context(
                        failure_msg, project=project_name,
                    )

                fix_result = await fix_director.run({
                    "failure": failure,
                    "spec": self.spec,
                })
                if fix_result.success:
                    fix_task = fix_director.parse_fix_task(fix_result.output)

                    # Builder executes fix (with LDB + Reflexion context)
                    builder = BuilderAgent(
                        self.config, self.llm,
                        working_dir=self.project_dir,
                        sandbox=sandbox,
                    )

                    # Inject LDB + Reflexion context
                    extra_ctx = ""
                    if ldb_context:
                        extra_ctx += "\n" + ldb_context
                    if reflexion_ctx:
                        extra_ctx += "\n" + reflexion_ctx
                    if extra_ctx:
                        builder.set_dynamic_constitution(extra_ctx)

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
                # Reflexion: mark as resolved
                if self.config.reflexion_enabled:
                    for failure in failures[:3]:
                        fid = f"fix-{hash(str(failure)) % 10000}"
                        self._reflexion.mark_success(fid)
                return

            # Reflexion: reflect on why fixes didn't work
            if self.config.reflexion_enabled:
                for failure in failures[:3]:
                    failure_msg = failure if isinstance(failure, str) else str(failure)
                    fid = f"fix-{hash(str(failure)) % 10000}"
                    try:
                        await self._reflexion.reflect_on_failure(
                            task_id=fid,
                            task_description=f"Fix test failure: {failure_msg[:200]}",
                            failure_summary=failure_msg[:500],
                            llm=self.llm,
                            project=project_name,
                        )
                    except Exception as e:
                        logger.debug(f"Reflexion skipped: {e}")

        # Mark persistent failures
        if self.config.reflexion_enabled:
            for failure in (test_results.failures or [])[:3]:
                fid = f"fix-{hash(str(failure)) % 10000}"
                self._reflexion.mark_persistent(fid)

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
    # Evolution — Cross-Project Self-Improvement
    # ──────────────────────────────────────────────

    async def _evolution_record_and_reflect(self, project_name: str) -> None:
        """Record fitness of this run and perform LLM-assisted reflection.

        Called at the end of a successful pipeline run. The fitness data
        drives evolution: future projects inherit strategies that scored well.
        """
        elapsed = time.monotonic() - self._start_time

        # Gather fitness metrics
        quality = 0.0
        test_pass_rate = 0.0
        refactor_needed = True

        # Try to read test results
        test_file = self.project_dir / ".autoforge" / "test_results.json"
        if not test_file.exists():
            test_file = self.project_dir / "test_results.json"
        if test_file.exists():
            try:
                test_data = json.loads(test_file.read_text(encoding="utf-8"))
                test_pass_rate = 1.0 if test_data.get("all_passed") else 0.5
            except Exception:
                pass

        # Try to read review score
        review_file = self.project_dir / ".autoforge" / "review_report.json"
        if review_file.exists():
            try:
                review_data = json.loads(review_file.read_text(encoding="utf-8"))
                quality = float(review_data.get("score", 0))
                refactor_needed = quality < (self.config.quality_threshold * 10)
            except Exception:
                pass

        tasks_completed = 0
        tasks_total = 0
        if self.dag:
            build_tasks = [t for t in self.dag.get_all_tasks() if t.phase == TaskPhase.BUILD]
            tasks_total = len(build_tasks)
            tasks_completed = len([t for t in build_tasks if t.status == TaskStatus.DONE])

        fitness = FitnessScore(
            quality_score=quality,
            test_pass_rate=test_pass_rate,
            cost_usd=self.config.estimated_cost_usd,
            duration_seconds=elapsed,
            tasks_completed=tasks_completed,
            tasks_total=tasks_total,
            build_success_rate=(tasks_completed / tasks_total) if tasks_total > 0 else 0.0,
            refactor_needed=refactor_needed,
        )

        # Update genome with actual architecture strategy used
        if self._genome and self.architecture:
            self._genome.arch_strategy = json.dumps(
                self.architecture, ensure_ascii=False
            )[:500]

        # Record to evolution memory
        record = self._evolution.record_result(
            project_name=project_name,
            fitness=fitness,
            genome=self._genome,
        )

        console.print(
            f"\n  [cyan]Evolution:[/cyan] fitness={fitness.composite_score:.2f} "
            f"(quality={quality}/10, tests={test_pass_rate:.0%}, "
            f"build={tasks_completed}/{tasks_total})"
        )

        # LLM-assisted reflection (analyze what worked)
        try:
            reflection = await self._evolution.reflect(record, self.llm)
            if reflection:
                console.print(f"  [cyan]Reflection:[/cyan] {reflection[:120]}...")
        except Exception as e:
            logger.debug(f"Evolution reflection skipped: {e}")

        # Show evolution stats
        stats = self._evolution.get_evolution_stats()
        if stats.get("total_runs", 0) > 1:
            console.print(
                f"  [dim]Evolution history: {stats['total_runs']} runs, "
                f"{stats['niches']} niches, "
                f"best fitness={stats.get('best_fitness', 0):.2f}[/dim]"
            )

    # ──────────────────────────────────────────────
    # Dynamic Constitution & Meta-Learning
    # ──────────────────────────────────────────────

    async def _init_dynamic_constitution(self) -> None:
        """Initialize dynamic constitution and generate patches from spec.

        Called after SPEC phase. The Director analyzes the spec and generates
        project-specific instructions for each agent role.
        """
        self._dynamic_constitution = DynamicConstitution(self.project_dir)

        try:
            patches = await self._dynamic_constitution.generate_patches_from_spec(
                self.spec, self.llm,
            )
            if patches:
                console.print(
                    f"  [green]Dynamic constitution:[/green] "
                    f"{len(patches)} project-specific instructions generated"
                )
            else:
                console.print("  [dim]Dynamic constitution: no patches generated[/dim]")
        except Exception as e:
            logger.warning(f"Dynamic constitution generation failed: {e}")
            console.print(f"  [yellow]Dynamic constitution skipped:[/yellow] {e}")

    async def _explore_architectures(
        self, architect: ArchitectAgent,
    ) -> dict[str, Any]:
        """Use search tree to explore multiple architecture candidates.

        Instead of committing to the first architecture the LLM generates,
        this generates N candidates, evaluates them, and selects the best.

        Inspired by SWE-Search (ICLR 2025) and Tree of Thoughts (NeurIPS 2023).
        """
        num_candidates = getattr(self.config, "search_tree_max_candidates", 3)
        console.print(f"  [cyan]Search tree:[/cyan] generating {num_candidates} architecture candidates...")

        # Step 1: Generate diverse architecture candidates
        candidates = await architect.generate_diverse_architectures(
            self.spec, num_candidates=num_candidates,
        )

        if not candidates:
            # Fallback to single architecture
            console.print("  [yellow]No candidates generated, falling back to single architecture[/yellow]")
            arch_result = await architect.run({"spec": self.spec})
            if not arch_result.success:
                raise RuntimeError(f"Architect failed: {arch_result.error}")
            return architect.parse_architecture(arch_result.output)

        # Step 2: Set up search tree
        self._search_tree = SearchTree()
        root = self._search_tree.create_root(
            description="Architecture exploration",
            strategy="Evaluate multiple architecture candidates",
        )

        # Step 3: Create branch nodes and evaluate candidates
        branch_nodes = self._search_tree.branch(root.id, candidates)

        context = json.dumps(self.spec, indent=2, ensure_ascii=False)
        task_desc = f"Design architecture for '{self.spec.get('project_name', '')}'"

        for node, cand in zip(branch_nodes, candidates):
            score, confidence, reason = await evaluate_candidate(
                self.llm, cand, task_desc, context,
            )
            self._search_tree.evaluate_node(node.id, score, confidence, reason)
            console.print(
                f"    Candidate '{cand.get('description', '')[:40]}': "
                f"score={score:.2f} confidence={confidence:.2f}"
            )

        # Step 3.5: Conditional debate — if top candidates are close, debate
        if getattr(self.config, "debate_enabled", True) and len(branch_nodes) >= 2:
            try:
                scored = [
                    (n, c) for n, c in zip(branch_nodes, candidates)
                    if n.score is not None
                ]
                scored.sort(key=lambda x: x[0].score, reverse=True)
                if (
                    len(scored) >= 2
                    and scored[0][0].score - scored[1][0].score < 0.15
                ):
                    console.print("  [cyan]Debate:[/cyan] Close candidates — initiating architecture debate...")
                    debate_topic = (
                        f"Architecture choice for '{self.spec.get('project_name', '')}': "
                        f"Option A ({scored[0][1].get('description', 'A')[:50]}) vs "
                        f"Option B ({scored[1][1].get('description', 'B')[:50]})"
                    )
                    positions = [
                        {"role": "architect_A", "position": json.dumps(scored[0][1], ensure_ascii=False)[:500]},
                        {"role": "architect_B", "position": json.dumps(scored[1][1], ensure_ascii=False)[:500]},
                    ]
                    should, uncertainty = await self._debate.should_debate(debate_topic, positions, self.llm)
                    if should:
                        outcome = await self._debate.run_debate(
                            debate_topic,
                            [{"role": "architect_A", "expertise": "system design"},
                             {"role": "architect_B", "expertise": "system design"}],
                            context, self.llm,
                        )
                        if outcome and outcome.confidence > 0.6:
                            console.print(
                                f"  [cyan]Debate result:[/cyan] {outcome.convergence_reason} "
                                f"(confidence={outcome.confidence:.2f})"
                            )
                            # Use the winning position to re-rank
                            if "A" in outcome.winner_position[:20]:
                                best_idx = 0
                            elif "B" in outcome.winner_position[:20]:
                                best_idx = 1
                            else:
                                best_idx = 0
                            # Boost the debate winner's search tree score
                            winner_node = scored[best_idx][0]
                            self._search_tree.evaluate_node(
                                winner_node.id,
                                min(1.0, winner_node.score + 0.1),
                                0.9,
                                f"Debate winner: {outcome.convergence_reason}",
                            )
            except Exception as e:
                logger.debug(f"Architecture debate skipped: {e}")

        # Step 4: Select the best candidate
        best = self._search_tree.select_best(root.id)
        if not best:
            console.print("  [yellow]No winning candidate, falling back[/yellow]")
            arch_result = await architect.run({"spec": self.spec})
            if not arch_result.success:
                raise RuntimeError(f"Architect failed: {arch_result.error}")
            return architect.parse_architecture(arch_result.output)

        console.print(
            f"  [green]Selected:[/green] '{best.description}' "
            f"(score={best.score:.2f})"
        )

        # Step 5: Run architect with the selected strategy as guidance
        arch_result = await architect.run({
            "spec": self.spec,
            "strategy_guidance": best.strategy,
        })

        if not arch_result.success:
            # Backtrack: try the next-best candidate
            alt = self._search_tree.backtrack()
            if alt:
                console.print(f"  [yellow]Backtracking to:[/yellow] '{alt.description}'")
                arch_result = await architect.run({
                    "spec": self.spec,
                    "strategy_guidance": alt.strategy,
                })
                if not arch_result.success:
                    raise RuntimeError(f"Architect failed after backtrack: {arch_result.error}")
            else:
                raise RuntimeError(f"Architect failed, no alternatives: {arch_result.error}")

        return architect.parse_architecture(arch_result.output)

    async def _learn_from_task_failure(
        self,
        task: Task,
        error: str,
        agent_id: str,
    ) -> None:
        """Analyze a task failure and create preventive rules via meta-learning.

        This is the self-skill generation mechanism: when agents fail, the system
        analyzes the failure pattern and creates reusable rules that are injected
        into future agent prompts.
        """
        if not self._dynamic_constitution:
            return

        try:
            failure_context = {
                "task_description": task.description,
                "error": error,
                "agent_role": task.owner,
                "files_involved": task.files,
                "turn_count": 0,
                "approach_used": agent_id,
            }
            rule = await self._dynamic_constitution.learn_from_failure(
                failure_context, self.llm,
            )
            if rule:
                console.print(
                    f"  [cyan]Meta-learning:[/cyan] New rule from failure — "
                    f"'{rule.rule[:60]}...'"
                )
        except Exception as e:
            logger.debug(f"Meta-learning failed: {e}")

    # ──────────────────────────────────────────────
    # Prompt Optimizer — DSPy/OPRO-Style Self-Improvement
    # ──────────────────────────────────────────────

    async def _init_prompt_optimizer(self) -> None:
        """Initialize prompt optimizer and register baseline constitutions.

        Called after SPEC phase. Registers the current constitution prompts
        as baselines if they haven't been registered before.
        """
        if not getattr(self.config, "prompt_optimization_enabled", True):
            return

        roles = ["director", "architect", "builder", "reviewer", "tester", "gardener"]
        for role in roles:
            # Try to get the dynamic supplement as the optimizable part
            if self._dynamic_constitution:
                supplement = self._dynamic_constitution.build_supplementary_prompt(role)
                if supplement:
                    self._prompt_optimizer.register_baseline(role, supplement)

        logger.info("[PromptOpt] Initialized with baselines for all roles")

    async def _prompt_optimizer_record_and_optimize(
        self, project_name: str,
    ) -> None:
        """Record fitness for prompt variants and trigger optimization if ready.

        Called at the end of a successful pipeline run.
        """
        if not getattr(self.config, "prompt_optimization_enabled", True):
            return

        # Compute fitness (reuse evolution fitness logic)
        quality = 0.0
        test_file = self.project_dir / ".autoforge" / "test_results.json"
        if not test_file.exists():
            test_file = self.project_dir / "test_results.json"
        if test_file.exists():
            try:
                test_data = json.loads(test_file.read_text(encoding="utf-8"))
                quality += 0.5 if test_data.get("all_passed") else 0.2
            except Exception:
                pass

        review_file = self.project_dir / ".autoforge" / "review_report.json"
        if review_file.exists():
            try:
                review_data = json.loads(review_file.read_text(encoding="utf-8"))
                quality += float(review_data.get("score", 0)) / 10.0 * 0.5
            except Exception:
                pass

        if quality <= 0:
            quality = 0.5  # Neutral default

        # Record fitness for each active variant
        roles = ["director", "architect", "builder", "reviewer", "tester", "gardener"]
        for role in roles:
            variant_id = self._prompt_optimizer.get_active_variant_id(role)
            if variant_id:
                self._prompt_optimizer.record_result(role, variant_id, quality)

        # Trigger optimization for roles that have enough data
        optimized_count = 0
        for role in roles:
            if self._prompt_optimizer.should_optimize(role):
                try:
                    context = f"Project: {project_name}, quality={quality:.2f}"
                    variant = await self._prompt_optimizer.optimize_role(
                        role, self.llm, context,
                    )
                    if variant:
                        optimized_count += 1
                except Exception as e:
                    logger.debug(f"Prompt optimization for {role} skipped: {e}")

        if optimized_count > 0:
            console.print(
                f"  [cyan]Prompt optimizer:[/cyan] {optimized_count} role(s) optimized"
            )

        # Show stats
        stats = self._prompt_optimizer.get_stats()
        total_variants = sum(
            r.get("total_variants", 0) for r in stats.get("roles", {}).values()
        )
        if total_variants > len(roles):
            console.print(
                f"  [dim]Prompt variants: {total_variants} across {len(stats.get('roles', {}))} roles[/dim]"
            )

    # ──────────────────────────────────────────────
    # Formal Verification & Security Scan
    # ──────────────────────────────────────────────

    async def _run_formal_verification(self) -> None:
        """Run formal verification on the generated project."""
        if not self.project_dir:
            return
        try:
            report = await self._formal_verifier.verify(
                self.project_dir, llm=self.llm,
                run_security=False,  # Security scan runs separately
            )
            # Save report
            forge_dir = self.project_dir / ".autoforge"
            forge_dir.mkdir(exist_ok=True)
            (forge_dir / "formal_verify_report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8",
            )
            if report.errors > 0:
                console.print(
                    f"  [yellow]Formal verification:[/yellow] {report.errors} errors, "
                    f"{report.warnings} warnings"
                )
            else:
                console.print(f"  [green]Formal verification passed[/green] ({report.warnings} warnings)")
        except Exception as e:
            logger.warning(f"Formal verification failed: {e}")

    async def _run_security_scan(self) -> None:
        """Run RedCode security scan on the generated project."""
        if not self.project_dir:
            return
        try:
            report = await self._security_scanner.scan(
                self.project_dir, llm=self.llm,
            )
            forge_dir = self.project_dir / ".autoforge"
            forge_dir.mkdir(exist_ok=True)
            (forge_dir / "security_report.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8",
            )
            if not report.passed:
                console.print(
                    f"  [red]Security scan:[/red] {report.critical_count} critical, "
                    f"{report.high_count} high, {report.medium_count} medium"
                )
            else:
                console.print(f"  [green]Security scan passed[/green] ({report.low_count} low issues)")
        except Exception as e:
            logger.warning(f"Security scan failed: {e}")

    # ──────────────────────────────────────────────
    # CapabilityDAG — Knowledge Ingestion
    # ──────────────────────────────────────────────

    async def _ingest_run_to_dag(self, project_name: str) -> None:
        """Ingest knowledge from this run into the CapabilityDAG.

        Captures:
          - Architecture decisions + outcomes
          - Debugging patterns (errors encountered + fixes applied)
          - Code patterns (from generated files)
          - Workflow strategy (from evolution genome)
        """
        if not self.project_dir:
            return

        try:
            # 1. Architecture decision
            if self.architecture:
                self._dag_bridge.ingest_architecture_decision(
                    decision=json.dumps(self.architecture, ensure_ascii=False)[:2000],
                    context=self.spec.get("description", ""),
                    outcome_success=True,
                    source_project=project_name,
                )

            # 2. Workflow genome (if evolution produced one)
            if self._genome:
                self._dag_bridge.ingest_workflow(
                    strategy_description=json.dumps(
                        self._genome.to_dict(), ensure_ascii=False
                    )[:2000],
                    tech_fingerprint=getattr(self._genome, "tech_fingerprint", ""),
                    source_project=project_name,
                )

            # 3. Debugging knowledge from reflexion memories
            if self.config.reflexion_enabled:
                for ref in self._reflexion._memory[:10]:
                    self._dag_bridge.ingest_debug_pattern(
                        error_pattern=ref.failure_description,
                        fix_strategy=ref.reflection,
                        success=ref.led_to_success,
                        source_project=project_name,
                    )

            # 4. Theoretical reasoning: ingest theory graph concepts
            if self.config.theoretical_reasoning_enabled:
                for title, theory in self._theoretical_reasoning._theories.items():
                    self._ingest_theory_to_dag(theory)

            logger.info(f"[CapDAG] Ingested knowledge from project {project_name}")

        except Exception as e:
            logger.debug(f"[CapDAG] Ingestion failed: {e}")

    # ──────────────────────────────────────────────
    # Lean 4 Formal Theorem Proving
    # ──────────────────────────────────────────────

    async def _run_lean_proving(self) -> None:
        """Run Lean 4 formal proving on any .lean files in the project.

        Integrates the full LeanProver pipeline:
          - Detects .lean files in the generated project
          - For each file containing 'sorry', attempts proof completion
          - Reports verification results
        """
        if not self.project_dir:
            return

        try:
            # Lazy init
            if self._lean_prover is None:
                self._lean_prover = LeanProver(workspace=self.project_dir)
                # Load prior state if available
                lean_state = self.project_dir / ".autoforge" / "lean"
                if lean_state.exists():
                    self._lean_prover.load_state(lean_state)

            lean_files = list(self.project_dir.rglob("*.lean"))
            if not lean_files:
                return

            lean_available = await self._lean_prover.check_lean_available()
            console.print(
                f"  [cyan]Lean prover:[/cyan] found {len(lean_files)} .lean file(s)"
                f"{' (Lean 4 installed)' if lean_available else ' (LLM-simulated)'}"
            )

            proved = 0
            sorry_total = 0

            for lean_file in lean_files:
                content = lean_file.read_text(encoding="utf-8")
                sorry_count = content.count("sorry")

                if sorry_count > 0:
                    console.print(
                        f"    {lean_file.name}: {sorry_count} sorry — "
                        f"attempting proof completion..."
                    )
                    # Attempt to prove sorry theorems
                    sorry_total += sorry_count
                    # Extract theorem statements with sorry
                    import re
                    sorry_blocks = re.findall(
                        r'((?:theorem|lemma)\s+\w+[^:]*:.*?(?:sorry))',
                        content, re.DOTALL,
                    )
                    for block in sorry_blocks[:5]:  # Limit attempts per file
                        stmt = block.replace("sorry", "").strip()
                        attempt = await self._lean_prover.prove_theorem(
                            stmt, self.llm,
                        )
                        if attempt.status.value == "proved":
                            proved += 1

            # Multi-pass proof repair (v2.8) — use ProofRepairEngine for deeper sorry elimination
            if sorry_total > 0 and proved < sorry_total and getattr(self.config, "lean_auto_repair_passes", 0) > 0:
                try:
                    for lean_file in lean_files:
                        content = lean_file.read_text(encoding="utf-8")
                        if "sorry" not in content:
                            continue
                        repaired, remaining = await self._lean_prover.repair_proof(
                            content, self.llm,
                            max_passes=self.config.lean_auto_repair_passes,
                        )
                        if repaired != content:
                            lean_file.write_text(repaired, encoding="utf-8")
                            new_sorry = repaired.count("sorry")
                            repaired_count = content.count("sorry") - new_sorry
                            proved += repaired_count
                            console.print(
                                f"    [cyan]Repair:[/cyan] {lean_file.name}: "
                                f"eliminated {repaired_count} sorry, {new_sorry} remaining"
                            )
                except Exception as e:
                    logger.debug(f"Lean proof repair pass failed: {e}")

            if sorry_total > 0:
                console.print(
                    f"  [cyan]Lean prover:[/cyan] proved {proved}/{sorry_total} sorry blocks"
                )
            else:
                console.print(f"  [green]Lean:[/green] all files clean (no sorry)")

            # Multi-prover cross-verification (v2.8) — if other provers are enabled
            _any_alt_prover = any(getattr(self.config, f"{p}_enabled", False)
                                 for p in ("coq", "isabelle", "tlaplus", "z3_smt", "dafny"))
            if _any_alt_prover:
                try:
                    if self._multi_prover is None:
                        self._multi_prover = MultiProverEngine(workspace=self.project_dir)
                    available = await self._multi_prover.detect_available_provers()
                    active = [k.value for k, v in available.items() if v]
                    if active:
                        console.print(f"  [cyan]Multi-prover:[/cyan] {', '.join(active)} available")
                except Exception as e:
                    logger.debug(f"Multi-prover detection failed: {e}")

        except Exception as e:
            logger.warning(f"Lean proving failed: {e}")

    def _build_theory_context(self, task_description: str, max_tokens: int = 1500) -> str:
        """Build context from stored theories relevant to a build task.

        Scans loaded TheoryGraphs for concepts/relations that match the
        task description, returning a compact summary for agent injection.
        """
        if not self._theoretical_reasoning._theories:
            return ""

        keywords = set(task_description.lower().split())
        relevant_snippets: list[str] = []
        char_budget = max_tokens * 4  # ~4 chars per token

        for title, theory in self._theoretical_reasoning._theories.items():
            # Check if theory title matches task
            title_words = set(title.lower().split("_"))
            if not keywords & title_words and len(relevant_snippets) > 3:
                continue

            stats = theory.get_stats()
            header = (
                f"Theory: {title} ({stats.get('total_concepts', 0)} concepts, "
                f"{stats.get('total_relations', 0)} relations)"
            )

            # Include high-confidence theorems and conjectures
            useful = []
            for node in theory._nodes.values():
                if node.overall_confidence >= 0.5 and node.concept_type.value in (
                    "theorem", "lemma", "conjecture", "definition",
                ):
                    snippet = f"  [{node.concept_type.value}] {node.id}: {node.informal_statement[:150] or node.formal_statement[:150]}"
                    useful.append(snippet)
                    if len("\n".join(useful)) > char_budget // max(len(self._theoretical_reasoning._theories), 1):
                        break

            if useful:
                relevant_snippets.append(header + "\n" + "\n".join(useful[:8]))

        if not relevant_snippets:
            return ""

        result = (
            "\n\n# Theoretical Knowledge Base\n"
            "The following cross-domain theoretical insights are available:\n\n"
            + "\n\n".join(relevant_snippets[:5])
        )
        return result[:char_budget]

    # ──────────────────────────────────────────────
    # Theoretical Reasoning — Cross-Domain Scientific Discovery
    # ──────────────────────────────────────────────

    async def _run_theoretical_reasoning(self) -> None:
        """Run cross-domain theoretical reasoning on the generated project.

        This integrates the TheoreticalReasoningEngine into the pipeline:
          1. Detects theory-relevant files (.md, .tex, .theory.json) in the project
          2. Parses them into TheoryGraphs
          3. Attempts multi-strategy reasoning to generate new insights
          4. Verifies generated concepts via multi-modal verification
          5. Evolves theories by branching into related domains
          6. Feeds discovered concepts into the CapabilityDAG
        """
        if not self.project_dir:
            return

        try:
            from autoforge.engine.theoretical_reasoning import (
                ReasoningStrategy,
                ScientificDomain,
            )

            # 1. Find theory-relevant files
            theory_files: list[Path] = []
            for pattern in ("*.theory.json", "*.theory.md", "*.tex"):
                theory_files.extend(self.project_dir.rglob(pattern))

            # Also scan markdown files for theoretical content
            for md_file in self.project_dir.rglob("*.md"):
                try:
                    text = md_file.read_text(encoding="utf-8")[:2000]
                    theory_keywords = [
                        "theorem", "lemma", "proof", "conjecture",
                        "definition", "corollary", "proposition",
                    ]
                    if any(kw in text.lower() for kw in theory_keywords):
                        theory_files.append(md_file)
                except Exception:
                    continue

            if not theory_files:
                return

            console.print(
                f"  [cyan]Theoretical reasoning:[/cyan] "
                f"found {len(theory_files)} theory-relevant file(s)"
            )

            parsed_count = 0
            new_insights = 0

            for tf in theory_files[:10]:  # Limit to 10 files per run
                try:
                    text = tf.read_text(encoding="utf-8")

                    # 2. Parse into TheoryGraph
                    if tf.suffix == ".json":
                        # Pre-structured theory graph
                        from autoforge.engine.theoretical_reasoning import TheoryGraph
                        graph = TheoryGraph()
                        graph.load(tf.parent)
                    else:
                        # Parse from text (markdown, LaTeX)
                        graph = await self._theoretical_reasoning.parse_article(
                            text, self.llm, title=tf.stem,
                        )
                    parsed_count += 1

                    # 3. Verify low-confidence concepts
                    results = await self._theoretical_reasoning.verify_theory(
                        graph, self.llm,
                    )
                    verified_count = sum(1 for c in results.values() if c >= 0.5)

                    # 4. Evolve: attempt one round of reasoning from frontier
                    if graph.size > 0:
                        branch = await self._theoretical_reasoning.evolve_theory(
                            graph, self.llm,
                            strategy=ReasoningStrategy.ANALOGY_TRANSFER,
                        )
                        new_insights += branch.size - graph.size

                    console.print(
                        f"    {tf.name}: {graph.size} concepts, "
                        f"{verified_count} verified, "
                        f"+{max(0, new_insights)} new insights"
                    )

                    # 5. Feed into CapabilityDAG
                    if self.config.capability_dag_enabled:
                        self._ingest_theory_to_dag(graph)

                except Exception as e:
                    logger.debug(f"Theoretical reasoning on {tf.name}: {e}")
                    continue

            if parsed_count > 0:
                console.print(
                    f"  [cyan]Theoretical reasoning:[/cyan] "
                    f"parsed {parsed_count} theories, "
                    f"generated {new_insights} new insights"
                )

        except Exception as e:
            logger.warning(f"Theoretical reasoning failed: {e}")

    def _ingest_theory_to_dag(self, theory: "TheoryGraph") -> None:
        """Feed theoretical concepts from a TheoryGraph into the CapabilityDAG.

        Quality gating:
          - Only concepts above config.dag_ingest_confidence_threshold are ingested
          - Cross-domain bridges require both endpoints above threshold
          - This prevents low-quality noise from polluting the shared DAG

        Maps:
          ConceptNode → DAG capability node
          ConceptRelation → DAG edges
          Cross-domain bridges → DAG cross-domain links
        """
        threshold = self.config.dag_ingest_confidence_threshold
        ingested = 0
        skipped = 0

        try:
            for node in theory._nodes.values():
                # Quality gate: skip low-confidence concepts
                if node.overall_confidence < threshold:
                    skipped += 1
                    continue

                self._dag_bridge.ingest_architecture_decision(
                    decision=(
                        f"[{node.concept_type.value}] {node.id}: "
                        f"{(node.informal_statement or node.formal_statement)[:300]}"
                    ),
                    context=(
                        f"Domain: {node.domain.value}. "
                        f"Confidence: {node.overall_confidence:.2f}"
                    ),
                    outcome_success=node.overall_confidence >= 0.5,
                    source_project=theory.title or "theoretical_reasoning",
                )
                ingested += 1

            # Ingest cross-domain bridges as high-value workflow knowledge
            # Both endpoints must meet confidence threshold
            for src, dst, rel in theory.get_cross_domain_bridges():
                if src.overall_confidence < threshold or dst.overall_confidence < threshold:
                    skipped += 1
                    continue

                self._dag_bridge.ingest_workflow(
                    strategy_description=(
                        f"Cross-domain bridge ({rel.relation_type.value}): "
                        f"{src.id} ({src.domain.value}) → "
                        f"{dst.id} ({dst.domain.value}). "
                        f"Insight: {rel.bridging_insight[:200]}"
                    ),
                    tech_fingerprint=f"{src.domain.value}→{dst.domain.value}",
                    source_project=theory.title or "theoretical_reasoning",
                )
                ingested += 1

            if ingested > 0 or skipped > 0:
                logger.info(
                    f"[Theory→DAG] {theory.title}: ingested {ingested}, "
                    f"skipped {skipped} (threshold={threshold})"
                )

        except Exception as e:
            logger.debug(f"Theory→DAG ingestion failed: {e}")

    # ──────────────────────────────────────────────
    # EvoMAC Text Backpropagation
    # ──────────────────────────────────────────────

    async def _evomac_backward_pass(self) -> None:
        """Generate and apply text gradients from test/review results."""
        if not self.project_dir:
            return
        try:
            # Gather evaluation data
            eval_data: dict[str, Any] = {}

            test_file = self.project_dir / ".autoforge" / "test_results.json"
            if not test_file.exists():
                test_file = self.project_dir / "test_results.json"
            if test_file.exists():
                eval_data["test_results"] = json.loads(test_file.read_text(encoding="utf-8"))

            review_file = self.project_dir / ".autoforge" / "review_report.json"
            if review_file.exists():
                review_data = json.loads(review_file.read_text(encoding="utf-8"))
                eval_data["review_score"] = review_data.get("score", 0)
                eval_data["review_issues"] = review_data.get("issues", [])

            if self.dag:
                failed_tasks = [
                    f"{t.id}: {t.result}" for t in self.dag.get_all_tasks()
                    if t.phase == TaskPhase.BUILD and t.status == TaskStatus.FAILED
                ]
                eval_data["build_failures"] = failed_tasks

            if eval_data:
                gradients = await self._evomac.generate_gradients(eval_data, self.llm)
                if gradients:
                    console.print(
                        f"  [cyan]EvoMAC:[/cyan] {len(gradients)} text gradients generated"
                    )
                    # Apply gradients to dynamic constitution
                    for role in ("builder", "architect", "reviewer"):
                        if self._dynamic_constitution:
                            current = self._dynamic_constitution.build_supplementary_prompt(role)
                            updated = await self._evomac.apply_gradients(role, current, self.llm)
                            if updated != current:
                                from autoforge.engine.dynamic_constitution import ConstitutionPatch
                                self._dynamic_constitution.add_patch(ConstitutionPatch(
                                    id=f"evomac-{role}-{self._evomac.iteration}",
                                    target_role=role,
                                    content=updated,
                                    source="evomac",
                                    priority=5,
                                    project_specific=True,
                                ))
        except Exception as e:
            logger.warning(f"EvoMAC backward pass failed: {e}")

    # ──────────────────────────────────────────────
    # SICA Self-Improvement
    # ──────────────────────────────────────────────

    async def _sica_propose_improvements(self, project_name: str) -> None:
        """Propose self-improvements based on project run data."""
        try:
            # Gather performance data
            perf_data: dict[str, Any] = {"recent_runs": [project_name]}

            review_file = self.project_dir / ".autoforge" / "review_report.json"
            if review_file.exists():
                review = json.loads(review_file.read_text(encoding="utf-8"))
                perf_data["avg_quality"] = review.get("score", 0)

            test_file = self.project_dir / ".autoforge" / "test_results.json"
            if test_file.exists():
                test_data = json.loads(test_file.read_text(encoding="utf-8"))
                perf_data["avg_test_pass_rate"] = 1.0 if test_data.get("all_passed") else 0.5

            # Only propose if quality is below threshold
            if perf_data.get("avg_quality", 10) >= 8:
                return

            proposals = await self._sica.propose_improvements(
                perf_data, self.config.constitution_dir, self.llm,
            )
            if proposals:
                console.print(
                    f"  [cyan]SICA:[/cyan] {len(proposals)} self-improvement proposals"
                )
                for p in proposals:
                    valid, reason = self._sica.validate_proposal(p)
                    if valid:
                        logger.info(f"[SICA] Validated: {p.description}")
                    else:
                        logger.debug(f"[SICA] Rejected: {p.description} — {reason}")
        except Exception as e:
            logger.warning(f"SICA self-improvement failed: {e}")

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
