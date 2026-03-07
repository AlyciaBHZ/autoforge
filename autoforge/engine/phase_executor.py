from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from autoforge.engine.kernel import resolve_profile


@dataclass(frozen=True)
class PhaseResult:
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    outcomes: tuple[str, ...] = ()
    checkpoint_summary: str = ""


@dataclass(frozen=True)
class PhaseInvocation:
    phase_id: str
    handler: Callable[[], Awaitable[dict[str, Any] | PhaseResult | None]]
    start_summary: str = ""
    checkpoint_phase: str = ""
    emit_start: bool = True


class PhaseExecutor:
    """Profile-aware phase scheduler backed by kernel phase graphs."""

    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator

    def _normalize_result(self, payload: dict[str, Any] | PhaseResult | None) -> PhaseResult:
        if isinstance(payload, PhaseResult):
            return payload
        if not isinstance(payload, dict):
            return PhaseResult()
        return PhaseResult(
            summary=str(payload.get("summary", "") or ""),
            metadata=dict(payload.get("metadata", {}) or {}),
            outcomes=tuple(str(item) for item in payload.get("outcomes", ()) if str(item).strip()),
            checkpoint_summary=str(payload.get("checkpoint_summary", "") or ""),
        )

    def _phase_state_for_exception(self, exc: Exception) -> str:
        name = exc.__class__.__name__
        if name == "BudgetExceededError":
            return "budget_exceeded"
        if name == "UserPausedError":
            return "paused"
        return "failed"

    def _ordered_invocations(
        self,
        *,
        profile_name: str,
        invocations: list[PhaseInvocation],
        start_phase: str | None = None,
    ) -> list[PhaseInvocation]:
        graph = resolve_profile(profile_name).phase_graph
        mapping = {item.phase_id: item for item in invocations}
        ordered = [mapping[phase_id] for phase_id in graph.phase_ids() if phase_id in mapping]
        if not start_phase:
            return ordered
        phase_ids = [item.phase_id for item in ordered]
        if start_phase not in phase_ids:
            return ordered
        return ordered[phase_ids.index(start_phase):]

    async def _execute_plan(
        self,
        *,
        profile_name: str,
        invocations: list[PhaseInvocation],
        start_phase: str | None = None,
    ) -> dict[str, PhaseResult]:
        results: dict[str, PhaseResult] = {}
        for invocation in self._ordered_invocations(
            profile_name=profile_name,
            invocations=invocations,
            start_phase=start_phase,
        ):
            if invocation.emit_start:
                self.orchestrator._kernel_phase(
                    invocation.phase_id,
                    "started",
                    summary=invocation.start_summary,
                )
            try:
                result = self._normalize_result(await invocation.handler())
            except Exception as exc:
                self.orchestrator._kernel_phase(
                    invocation.phase_id,
                    self._phase_state_for_exception(exc),
                    summary=str(exc),
                )
                raise
            self.orchestrator._kernel_phase(
                invocation.phase_id,
                "completed",
                summary=result.summary,
                metadata=result.metadata,
            )
            if result.outcomes:
                self.orchestrator._declare_kernel_outcomes(*result.outcomes)
            if result.checkpoint_summary:
                await self.orchestrator._checkpoint(
                    invocation.checkpoint_phase or invocation.phase_id,
                    result.checkpoint_summary,
                )
            results[invocation.phase_id] = result
        return results

    def _generate_invocations(self, runner: Any) -> list[PhaseInvocation]:
        return [
            PhaseInvocation("spec", runner.generate_spec, emit_start=False),
            PhaseInvocation("build", runner.generate_build, start_summary="Building project workspace", checkpoint_phase="build"),
            PhaseInvocation("test", runner.generate_test, start_summary="Running verification and test loops", checkpoint_phase="verify"),
            PhaseInvocation("refactor", runner.generate_refactor, start_summary="Applying hardening and cleanup", checkpoint_phase="refactor"),
            PhaseInvocation("deliver", runner.generate_deliver, start_summary="Packaging final handoff artifacts"),
        ]

    def _review_invocations(self, runner: Any) -> list[PhaseInvocation]:
        return [
            PhaseInvocation("intake", runner.review_intake, start_summary="Scanning project inputs"),
            PhaseInvocation("obligation_extraction", runner.review_obligation_extraction, start_summary="Extracting verification obligations from the codebase"),
            PhaseInvocation("check_prove", runner.review_judge, start_summary="Evaluating proof obligations and counterexamples"),
            PhaseInvocation("counterexample_report", runner.review_counterexample_report, start_summary="Generating verification report"),
        ]

    def _import_invocations(self, runner: Any) -> list[PhaseInvocation]:
        return [
            PhaseInvocation("spec", runner.import_spec, start_summary="Scanning imported project"),
            PhaseInvocation("build", runner.import_build, start_summary="Reviewing imported project for enhancement plan"),
            PhaseInvocation("test", runner.import_test, start_summary="Verifying imported project"),
            PhaseInvocation("refactor", runner.import_refactor, start_summary="Refactoring imported project"),
            PhaseInvocation("deliver", runner.import_deliver, start_summary="Packaging imported project"),
        ]

    def _research_invocations(self, runner: Any) -> list[PhaseInvocation]:
        return [
            PhaseInvocation("intake", runner.research_intake, start_summary="Framing research objective"),
            PhaseInvocation("claim_extraction", runner.research_claim_extraction, start_summary="Extracting research claims and targets"),
            PhaseInvocation("reproduction_plan", runner.research_reproduction_plan, start_summary="Planning reproducible experiment steps"),
            PhaseInvocation("build_experiment", runner.research_build_experiment, start_summary="Preparing experiment scaffold"),
            PhaseInvocation("run", runner.research_run, start_summary="Capturing run evidence"),
            PhaseInvocation("verify", runner.research_verify, start_summary="Evaluating research evidence"),
            PhaseInvocation("report", runner.research_report, start_summary="Compiling research report"),
        ]

    async def run_generate_pipeline(self, requirement: str) -> Path:
        from autoforge.engine.profile_runner import DevelopmentProfileRunner

        if not hasattr(self.orchestrator, "_executor_reset_context"):
            return await self.orchestrator._run_generate_pipeline(requirement)
        close_status = "failed"
        runner = DevelopmentProfileRunner(self.orchestrator)
        try:
            await runner.prepare_generate(requirement)
            await self._execute_plan(profile_name="development", invocations=self._generate_invocations(runner))
            self.orchestrator._print_summary()
            close_status = "completed"
            return self.orchestrator.project_dir
        except Exception as exc:
            name = exc.__class__.__name__
            if name == "UserPausedError":
                close_status = "paused"
                self.orchestrator._kernel_phase("checkpoint", "paused", summary=str(exc))
                if self.orchestrator.project_dir is not None:
                    return self.orchestrator.project_dir
                raise
            if name == "BudgetExceededError":
                close_status = "budget_exceeded"
                self.orchestrator._kernel_phase("deliver", "budget_exceeded", summary=str(exc))
                self.orchestrator._declare_kernel_outcomes("partial_delivery")
                if self.orchestrator.project_dir is not None:
                    self.orchestrator._save_state("budget_exceeded")
                raise
            self.orchestrator._kernel_phase("deliver", "failed", summary=str(exc))
            if self.orchestrator.project_dir is not None:
                self.orchestrator._save_state(f"failed: {exc}")
            raise
        finally:
            self.orchestrator._close_runtime(status=close_status)

    async def run_review_pipeline(self, project_path: str) -> dict[str, Any]:
        from autoforge.engine.profile_runner import VerificationProfileRunner

        if not hasattr(self.orchestrator, "_executor_reset_context"):
            return await self.orchestrator._run_review_pipeline(project_path)
        close_status = "failed"
        runner = VerificationProfileRunner(self.orchestrator)
        try:
            await runner.prepare_review(project_path)
            await self._execute_plan(profile_name="verification", invocations=self._review_invocations(runner))
            review = self.orchestrator._phase_context.get("review")
            scan_result = self.orchestrator._phase_context.get("scan_result")
            report = self.orchestrator._phase_context.get("review_report", {})
            self.orchestrator._print_review_summary(review, scan_result)
            close_status = "completed"
            return report
        except Exception as exc:
            name = exc.__class__.__name__
            if name == "BudgetExceededError":
                close_status = "budget_exceeded"
                self.orchestrator._kernel_phase("counterexample_report", "budget_exceeded", summary=str(exc))
                raise
            self.orchestrator._kernel_phase("counterexample_report", "failed", summary=str(exc))
            raise
        finally:
            self.orchestrator._close_runtime(status=close_status)

    async def run_import_pipeline(self, project_path: str, enhancement: str = "") -> Path:
        from autoforge.engine.profile_runner import DevelopmentProfileRunner

        if not hasattr(self.orchestrator, "_executor_reset_context"):
            return await self.orchestrator._run_import_pipeline(project_path, enhancement)
        close_status = "failed"
        runner = DevelopmentProfileRunner(self.orchestrator)
        try:
            await runner.prepare_import(project_path, enhancement)
            await self._execute_plan(profile_name="development", invocations=self._import_invocations(runner))
            self.orchestrator._print_summary()
            close_status = "completed"
            return self.orchestrator.project_dir
        except Exception as exc:
            name = exc.__class__.__name__
            if name == "BudgetExceededError":
                close_status = "budget_exceeded"
                self.orchestrator._kernel_phase("deliver", "budget_exceeded", summary=str(exc))
                self.orchestrator._declare_kernel_outcomes("partial_delivery")
                if self.orchestrator.project_dir is not None:
                    self.orchestrator._save_state("budget_exceeded")
                raise
            self.orchestrator._kernel_phase("deliver", "failed", summary=str(exc))
            if self.orchestrator.project_dir is not None:
                self.orchestrator._save_state(f"failed: {exc}")
            raise
        finally:
            self.orchestrator._close_runtime(status=close_status)

    async def run_research_pipeline(self, requirement: str) -> Path:
        from autoforge.engine.profile_runner import ResearchProfileRunner

        if not hasattr(self.orchestrator, "_executor_reset_context"):
            return await self.orchestrator._run_generate_pipeline(requirement)
        close_status = "failed"
        runner = ResearchProfileRunner(self.orchestrator)
        try:
            await runner.prepare_generate(requirement)
            await self._execute_plan(profile_name="research", invocations=self._research_invocations(runner))
            self.orchestrator._print_summary()
            close_status = "completed"
            return self.orchestrator.project_dir
        except Exception as exc:
            name = exc.__class__.__name__
            if name == "UserPausedError":
                close_status = "paused"
                self.orchestrator._kernel_phase("checkpoint", "paused", summary=str(exc))
                if self.orchestrator.project_dir is not None:
                    return self.orchestrator.project_dir
                raise
            if name == "BudgetExceededError":
                close_status = "budget_exceeded"
                self.orchestrator._kernel_phase("report", "budget_exceeded", summary=str(exc))
                self.orchestrator._declare_kernel_outcomes("failed_with_evidence")
                if self.orchestrator.project_dir is not None:
                    self.orchestrator._save_state("budget_exceeded")
                raise
            self.orchestrator._kernel_phase("report", "failed", summary=str(exc))
            if self.orchestrator.project_dir is not None:
                self.orchestrator._save_state(f"failed: {exc}")
            raise
        finally:
            self.orchestrator._close_runtime(status=close_status)

    async def run_resume_pipeline(self, workspace_path: Path | None = None) -> Path:
        if not hasattr(self.orchestrator, "_resolve_resume_workspace"):
            return await self.orchestrator._resume_pipeline(workspace_path)
        close_status = "failed"
        try:
            workspace = self.orchestrator._resolve_resume_workspace(workspace_path)
            self.orchestrator.project_dir = workspace
            self.orchestrator._state_file = workspace / ".forge_state.json"
            state, resume_metadata = self.orchestrator._load_resume_state_snapshot(workspace)
            profile_name = str(state.get("profile", "") or resolve_profile("development").name)
            self.orchestrator._executor_reset_context(operation="resume", profile=profile_name)
            phase_marker, prior_success_outcomes = self.orchestrator._restore_resumed_state(
                state,
                resume_metadata=resume_metadata,
            )
            operation = self.orchestrator._resume_operation_from_state(state)
            start_phase = self.orchestrator._resume_start_phase(
                profile_name=profile_name,
                marker=phase_marker,
            )

            if phase_marker == "complete":
                if prior_success_outcomes:
                    self.orchestrator._declare_kernel_outcomes(*prior_success_outcomes)
                close_status = "completed"
                self.orchestrator._print_summary()
                return self.orchestrator.project_dir

            if start_phase is not None:
                if operation == "review":
                    from autoforge.engine.profile_runner import VerificationProfileRunner

                    runner = VerificationProfileRunner(self.orchestrator)
                    await self._execute_plan(
                        profile_name="verification",
                        invocations=self._review_invocations(runner),
                        start_phase=start_phase,
                    )
                    review = self.orchestrator._phase_context.get("review")
                    scan_result = self.orchestrator._phase_context.get("scan_result")
                    self.orchestrator._print_review_summary(review, scan_result)
                elif operation == "import":
                    from autoforge.engine.profile_runner import DevelopmentProfileRunner

                    runner = DevelopmentProfileRunner(self.orchestrator)
                    await self._execute_plan(
                        profile_name="development",
                        invocations=self._import_invocations(runner),
                        start_phase=start_phase,
                    )
                    self.orchestrator._print_summary()
                elif operation == "research":
                    from autoforge.engine.profile_runner import ResearchProfileRunner

                    runner = ResearchProfileRunner(self.orchestrator)
                    await self._execute_plan(
                        profile_name="research",
                        invocations=self._research_invocations(runner),
                        start_phase=start_phase,
                    )
                    self.orchestrator._print_summary()
                else:
                    from autoforge.engine.profile_runner import DevelopmentProfileRunner

                    runner = DevelopmentProfileRunner(self.orchestrator)
                    await self._execute_plan(
                        profile_name="development",
                        invocations=self._generate_invocations(runner),
                        start_phase=start_phase,
                    )
                    self.orchestrator._print_summary()

            close_status = "completed"
            return self.orchestrator.project_dir
        except Exception as exc:
            operation = str(self.orchestrator._phase_context.get("resumed_operation", "") or "generate")
            if operation == "review":
                terminal_phase = "counterexample_report"
            elif operation == "research":
                terminal_phase = "report"
            else:
                terminal_phase = "deliver"
            name = exc.__class__.__name__
            if name == "BudgetExceededError":
                close_status = "budget_exceeded"
                self.orchestrator._kernel_phase(terminal_phase, "budget_exceeded", summary=str(exc))
                if terminal_phase == "deliver":
                    self.orchestrator._declare_kernel_outcomes("partial_delivery")
                elif terminal_phase == "report":
                    self.orchestrator._declare_kernel_outcomes("failed_with_evidence")
                if self.orchestrator.project_dir is not None:
                    self.orchestrator._save_state("budget_exceeded")
                raise
            self.orchestrator._kernel_phase(terminal_phase, "failed", summary=str(exc))
            if self.orchestrator.project_dir is not None:
                self.orchestrator._save_state(f"failed: {exc}")
            raise
        finally:
            self.orchestrator._close_runtime(status=close_status)
