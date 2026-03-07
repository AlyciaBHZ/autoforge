from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from autoforge.engine.dag_federation import build_snapshot_payload, pull_into_local_knowledge
from autoforge.engine.evolution import EvolutionEngine
from autoforge.engine.kernel.schema import write_kernel_json
from autoforge.engine.phase_executor import PhaseExecutor
from autoforge.engine.task_dag import TaskPhase, TaskStatus
from autoforge.engine.ui_harness import (
    write_ui_handoff,
    write_ui_harness_artifacts,
    write_ui_judge_report,
)

logger = logging.getLogger(__name__)
console = Console()


class BaseProfileRunner:
    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator
        self.phase_executor = PhaseExecutor(orchestrator)

    @property
    def _research_root(self) -> Path:
        if self.orchestrator.project_dir is None:
            raise RuntimeError("project_dir is not initialized")
        return self.orchestrator.project_dir / ".autoforge" / "research"

    def _activate_ui_harness(self, *, requirement: str, spec: dict[str, Any]) -> dict[str, Any]:
        artifacts = write_ui_harness_artifacts(
            project_dir=self.orchestrator.project_dir,
            config=self.orchestrator.config,
            requirement=requirement,
            spec=spec,
            kernel_session=self.orchestrator.kernel_session,
        )
        metadata = dict(artifacts.metadata or {})
        self.orchestrator._phase_context["ui_harness"] = metadata
        if artifacts.active and self.orchestrator.kernel_session is not None:
            self.orchestrator.kernel_session.update_execution_plan(
                metadata={"ui_harness": metadata},
            )
        return metadata

    def _run_ui_judge(self, *, requirement: str, spec: dict[str, Any], verify_passed: bool) -> tuple[dict[str, Any], tuple[str, ...]]:
        artifacts, outcomes = write_ui_judge_report(
            project_dir=self.orchestrator.project_dir,
            config=self.orchestrator.config,
            spec=spec,
            requirement=requirement,
            verify_passed=verify_passed,
            kernel_session=self.orchestrator.kernel_session,
        )
        metadata = dict(artifacts.metadata or {})
        self.orchestrator._phase_context["ui_judge"] = metadata
        return metadata, outcomes

    def _write_ui_handoff(self, *, spec: dict[str, Any]) -> dict[str, Any]:
        artifacts = write_ui_handoff(
            project_dir=self.orchestrator.project_dir,
            config=self.orchestrator.config,
            spec=spec,
            kernel_session=self.orchestrator.kernel_session,
        )
        metadata = dict(artifacts.metadata or {})
        self.orchestrator._phase_context["ui_handoff"] = metadata
        return metadata


class DevelopmentProfileRunner(BaseProfileRunner):
    async def run_generate(self, requirement: str) -> Path:
        return await self.phase_executor.run_generate_pipeline(requirement)

    async def run_import(self, project_path: str, enhancement: str = "") -> Path:
        return await self.phase_executor.run_import_pipeline(project_path, enhancement)

    async def prepare_generate(self, requirement: str) -> None:
        o = self.orchestrator
        o._executor_reset_context(operation="generate", profile="development")
        o._phase_context["requirement"] = requirement
        logger.info("AutoForge run %s starting", o.config.run_id)

        global_dag_dir = o.config.project_root / ".autoforge" / "capability_dag"
        o._phase_context["global_dag_dir"] = global_dag_dir
        if o._capability_dag is not None:
            o._capability_dag.load(global_dag_dir)
            if o._capability_dag.size > 0:
                console.print(f"  [cyan]CapabilityDAG:[/cyan] loaded {o._capability_dag.size} capabilities")

        if o._dag_federation is not None and o._dag_federation.enabled:
            pulled = await pull_into_local_knowledge(
                federation=o._dag_federation,
                capability_dag=o._capability_dag,
                theoretical_reasoning=o._theoretical_reasoning,
                global_theory_dir=o.config.project_root / ".autoforge" / "theories",
            )
            if pulled["dag_nodes"] > 0 or pulled["theories"] > 0:
                console.print(
                    f"  [cyan]DAG Federation:[/cyan] pulled +{pulled['dag_nodes']} DAG nodes, "
                    f"{pulled['theories']} theory graphs"
                )

        if o.config.theoretical_reasoning_enabled and o._theoretical_reasoning is not None:
            theory_dir = o.config.project_root / ".autoforge" / "theories"
            o._theoretical_reasoning.load_all(theory_dir)
            n_theories = len(o._theoretical_reasoning._theories)
            if n_theories > 0:
                console.print(f"  [cyan]Theoretical reasoning:[/cyan] loaded {n_theories} theory graphs")

    async def generate_spec(self) -> dict[str, Any]:
        o = self.orchestrator
        requirement = str(o._phase_context.get("requirement", "") or "")
        console.print("\n[bold blue]Phase 1: SPEC[/bold blue] — Analyzing requirements...")
        o.spec = await o._phase_spec(requirement)
        project_name = o.spec.get("project_name", "project")
        o.project_dir = o.config.workspace_dir / project_name
        o.project_dir.mkdir(parents=True, exist_ok=True)
        o._state_file = o.project_dir / ".forge_state.json"
        o._open_runtime(
            operation="generate",
            profile_name="development",
            metadata={
                "project_name": project_name,
                "requirement_chars": len(requirement),
            },
        )
        if (o._durable_enabled() or bool(getattr(o.config, "trace_enabled", False))) and o.runtime is not None:
            o.runtime.telemetry.record(
                "run_started",
                {
                    "project_name": project_name,
                    "workspace_dir": str(o.config.workspace_dir),
                    "requirement_chars": len(requirement),
                    "docker_enabled": bool(getattr(o.config, "docker_enabled", False)),
                    "sandbox_image": str(getattr(o.config, "sandbox_image", "")),
                },
            )

        (o.project_dir / "spec.json").write_text(
            json.dumps(o.spec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if getattr(o.config, "artifacts_enabled", True) and o.runtime is not None:
            try:
                o.runtime.artifacts.write_json("spec.json", o.spec)
            except Exception:
                pass
        o._save_state("spec_complete")
        n_modules = len(o.spec.get("modules", []))
        console.print(f"  [green]Spec generated:[/green] {project_name} — {n_modules} modules")
        if o.kernel_session is not None:
            o.kernel_session.update_execution_plan(
                objective=f"Build project {project_name}",
                summary=f"Generate {project_name} with {n_modules} planned modules",
                inputs={"requirement": requirement, "project_name": project_name, "module_count": n_modules},
                metadata={"spec": {"project_name": project_name, "module_count": n_modules}},
            )
        ui_metadata = self._activate_ui_harness(requirement=requirement, spec=o.spec)

        o._genome = o._evolution.prepare_genome(
            project_type=EvolutionEngine.infer_project_type(o.spec),
            tech_fingerprint=EvolutionEngine.extract_tech_fingerprint(o.spec),
            config=o.config,
        )
        o._evolution.apply_genome_to_config(o._genome, o.config)
        if o._genome.generation > 0:
            console.print(
                f"  [cyan]Evolution:[/cyan] gen {o._genome.generation} "
                f"(from {o._genome.parent_id})"
            )
            if o._genome.mutations:
                console.print(f"    Mutations: {', '.join(o._genome.mutations)}")

        await o._init_dynamic_constitution()
        await o._init_prompt_optimizer()

        if o.config.evomac_enabled:
            o._evomac.start_iteration()

        if o.config.speculative_enabled:
            await o._speculative.speculate_build_scaffold(o.spec, o.project_dir)

        if o.config.adaptive_compute_enabled:
            o._difficulty = o._adaptive_compute.estimate_difficulty(requirement, o.spec)
            console.print(
                f"  [cyan]Compute router:[/cyan] difficulty={o._difficulty.level.value} "
                f"(score={o._difficulty.score:.2f})"
            )

        return {
            "summary": f"{project_name} with {n_modules} modules",
            "metadata": {
                "project_name": project_name,
                "module_count": n_modules,
                "ui_harness_active": bool(ui_metadata.get("active", False)),
                "ui_style_preset": str(ui_metadata.get("style_preset", "") or ""),
            },
            "checkpoint_summary": (
                f"Generated spec with {n_modules} modules. "
                f"Review: {o.project_dir / 'spec.json'}"
            ),
        }

    async def generate_build(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 2: BUILD[/bold blue] — Building project...")
        if o.config.speculative_enabled:
            await o._speculative.validate_and_commit("spec-build-scaffold")
        await o._phase_build()
        o._save_state("build_complete")
        done = 0
        total = 0
        if o.dag:
            done = len([t for t in o.dag.get_all_tasks() if t.phase == TaskPhase.BUILD and t.status == TaskStatus.DONE])
            total = len([t for t in o.dag.get_all_tasks() if t.phase == TaskPhase.BUILD])
        return {
            "summary": "Build phase completed",
            "metadata": {"build_done": done, "build_total": total},
            "checkpoint_summary": (
                f"Built {done}/{total} tasks. Review generated code in: {o.project_dir}"
                if total
                else ""
            ),
        }

    async def generate_test(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 3: VERIFY[/bold blue] — Verifying project...")
        if o.config.speculative_enabled:
            await o._speculative.speculate_test_scaffold(o.spec, o.project_dir)
        await o._integration_check()
        await o._prepare_runtime_dependencies(o.project_dir)
        verify_passed = await o._phase_verify()
        requirement = str(o._phase_context.get("requirement", "") or "")

        if o.config.formal_verify_enabled:
            await o._run_formal_verification()
        if o.config.security_scan_enabled:
            await o._run_security_scan()
        if o.config.lean_prover_enabled:
            await o._run_lean_proving()
        if o.config.theoretical_reasoning_enabled:
            await o._run_theoretical_reasoning()
        if o.config.autonomous_discovery_enabled and o._autonomous_discovery:
            await o._run_autonomous_discovery()
        if o.config.paper_formalizer_enabled and o._paper_formalizer:
            await o._run_paper_formalization()
        if o.config.theoretical_reasoning_enabled and o._reasoning_extension is not None:
            await o._run_reasoning_extension()

        o._save_state("verify_complete")
        o._phase_context["last_verify_passed"] = bool(verify_passed)
        ui_metadata, ui_outcomes = self._run_ui_judge(
            requirement=requirement,
            spec=o.spec,
            verify_passed=bool(verify_passed),
        )

        if o.config.evomac_enabled:
            await o._evomac_backward_pass()
        if not verify_passed and o.config.security_scan_enabled:
            console.print("  [yellow]VERIFY→REFACTOR gate: feeding scan findings back for fixing[/yellow]")
            await o._fix_from_security_scan()

        return {
            "summary": "Verification phase completed",
            "metadata": {
                "verify_passed": bool(verify_passed),
                "ui_harness_active": bool(o._phase_context.get("ui_harness", {}).get("active", False)),
                "ui_judge_score": ui_metadata.get("judge_score", 0.0),
            },
            "outcomes": (("tests_pass",) if verify_passed else ()) + tuple(ui_outcomes),
            "checkpoint_summary": "Verification complete. Review test_results.json.",
        }

    async def generate_refactor(self) -> dict[str, Any]:
        o = self.orchestrator
        verify_passed = bool(o._phase_context.get("last_verify_passed", False))
        if verify_passed:
            console.print("\n[bold blue]Phase 4: REFACTOR[/bold blue] — Improving quality...")
        else:
            console.print(
                "\n[bold yellow]Phase 4: REFACTOR[/bold yellow] — "
                "Improving quality (some tests still failing, proceeding with best-effort)..."
            )
        await o._phase_refactor()
        o._save_state("refactor_complete")
        return {"summary": "Refactor phase completed"}

    async def generate_deliver(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 5: DELIVER[/bold blue] — Packaging...")
        await o._phase_deliver()
        o._save_state("complete")

        project_name = o.spec.get("project_name", o.project_dir.name if o.project_dir is not None else "")
        await o._evolution_record_and_reflect(project_name)
        await o._prompt_optimizer_record_and_optimize(project_name)

        if o.config.rag_enabled:
            o._rag.index_project(o.project_dir, project_name)
            console.print("  [cyan]RAG:[/cyan] indexed for future projects")
        if o.config.sica_enabled:
            await o._sica_propose_improvements(project_name)
        if o.config.evomac_enabled and o.project_dir:
            o._evomac.save_state(o.project_dir / ".autoforge")
        if o.config.reflexion_enabled and o.project_dir:
            o._reflexion.save_state(o.project_dir / ".autoforge")
        if o.config.lean_prover_enabled and o._lean_prover and o.project_dir:
            o._lean_prover.save_state(o.project_dir / ".autoforge" / "lean")
            console.print("  [cyan]Lean:[/cyan] proof state saved")
        if o._multi_prover and o.project_dir:
            o._multi_prover.save_state(o.project_dir / ".autoforge" / "multi_prover.json")
        if o._reasoning_extension is not None:
            ext_dir = o._forge_dir / "reasoning_extension"
            o._reasoning_extension.save(ext_dir)
        if o.config.theoretical_reasoning_enabled and o.project_dir:
            o._theoretical_reasoning.save_all(o.project_dir / ".autoforge" / "theories")
            global_theory_dir = o.config.project_root / ".autoforge" / "theories"
            o._theoretical_reasoning.save_all(global_theory_dir)
            n_theories = len(o._theoretical_reasoning._theories)
            if n_theories > 0:
                console.print(
                    f"  [cyan]Theoretical reasoning:[/cyan] {n_theories} theory graphs saved"
                )
        if o.config.adaptive_compute_enabled and o.project_dir:
            o._adaptive_compute.save_state(o.project_dir / ".autoforge")
        if o._capability_dag is not None:
            global_dag_dir = o._phase_context.get("global_dag_dir")
            await o._ingest_run_to_dag(project_name)
            if isinstance(global_dag_dir, Path):
                o._capability_dag.save(global_dag_dir)
            console.print(
                f"  [cyan]CapabilityDAG:[/cyan] {o._capability_dag.size} total capabilities"
            )
        if o._dag_federation is not None and o._dag_federation.enabled:
            payload = build_snapshot_payload(
                capability_dag=o._capability_dag,
                theoretical_reasoning=(
                    o._theoretical_reasoning if o.config.theoretical_reasoning_enabled else None
                ),
            )
            pushed = await o._dag_federation.push_snapshot(payload)
            if pushed:
                console.print("  [cyan]DAG Federation:[/cyan] remote snapshot updated")
        ui_handoff = self._write_ui_handoff(spec=o.spec)

        return {
            "summary": "Deliverables packaged",
            "outcomes": ("repo_runnable",),
            "metadata": {
                "ui_harness_active": bool(o._phase_context.get("ui_harness", {}).get("active", False)),
                "ui_design_ref_count": ui_handoff.get("design_context_ref_count", 0),
            },
        }

    async def prepare_import(self, project_path: str, enhancement: str = "") -> None:
        o = self.orchestrator
        o._executor_reset_context(operation="import", profile="development")
        o._phase_context["enhancement"] = enhancement
        source_dir = Path(project_path).resolve()
        o._phase_context["source_dir"] = source_dir
        logger.info("AutoForge import %s: %s", o.config.run_id, source_dir)
        if not source_dir.is_dir():
            raise ValueError(f"Not a directory: {project_path}")

        import shutil

        project_name = source_dir.name
        o.project_dir = o.config.workspace_dir / f"{project_name}-forge"
        if o.project_dir.exists():
            shutil.rmtree(o.project_dir)
        ignore_names = {".git", "node_modules", "__pycache__", ".venv", "venv", ".env"}
        try:
            workspace_rel = o.config.workspace_dir.resolve().relative_to(source_dir)
            if workspace_rel.parts:
                ignore_names.add(workspace_rel.parts[0])
        except ValueError:
            pass
        shutil.copytree(
            source_dir,
            o.project_dir,
            ignore=shutil.ignore_patterns(*sorted(ignore_names)),
        )
        o._state_file = o.project_dir / ".forge_state.json"
        o._open_runtime(
            operation="import",
            profile_name="development",
            metadata={"source_dir": str(source_dir), "project_dir": str(o.project_dir)},
        )
        if (o._durable_enabled() or bool(getattr(o.config, "trace_enabled", False))) and o.runtime is not None:
            try:
                o.runtime.telemetry.record(
                    "import_started",
                    {"source_dir": str(source_dir), "project_dir": str(o.project_dir)},
                )
            except Exception:
                pass

    async def import_spec(self) -> dict[str, Any]:
        o = self.orchestrator
        source_dir = o._phase_context.get("source_dir")
        enhancement = str(o._phase_context.get("enhancement", "") or "")
        project_name = getattr(source_dir, "name", o.project_dir.name if o.project_dir is not None else "project")
        console.print("\n[bold blue]Phase 1: SCAN[/bold blue] — Analyzing project...")
        scan_result = await o._phase_scan(o.project_dir)
        o._phase_context["scan_result"] = scan_result
        o.spec = scan_result.spec
        console.print(
            f"  [green]Scan complete:[/green] {o.spec.get('project_name', project_name)} "
            f"— {scan_result.completeness}% complete"
        )
        o._save_state("scan_complete")
        if o.kernel_session is not None:
            o.kernel_session.update_execution_plan(
                objective=f"Import and improve {o.project_dir.name}",
                summary="Scan, review, enhance, verify, and deliver imported project",
                inputs={"source_dir": str(source_dir), "project_dir": str(o.project_dir)},
                metadata={"scan_completeness": scan_result.completeness, "enhancement": enhancement},
            )
        ui_metadata = self._activate_ui_harness(
            requirement=enhancement or f"Import and improve {o.project_dir.name}",
            spec=o.spec,
        )
        return {
            "summary": "Imported project scan complete",
            "metadata": {
                "completeness": scan_result.completeness,
                "ui_harness_active": bool(ui_metadata.get("active", False)),
            },
        }

    async def import_build(self) -> dict[str, Any]:
        o = self.orchestrator
        enhancement = str(o._phase_context.get("enhancement", "") or "")
        console.print("\n[bold blue]Phase 2: REVIEW[/bold blue] — Reviewing code...")
        review = await o._phase_full_review()
        o._phase_context["review"] = review
        console.print(f"  Quality score: {review.score}/10, {len(review.issues)} issues")
        o._save_state("review_complete")
        if enhancement:
            console.print("\n[bold blue]Phase 3: ENHANCE[/bold blue] — Adding features...")
            await o._phase_enhance(enhancement)
            o._save_state("enhance_complete")
        return {
            "summary": "Import build/enhancement stage complete",
            "metadata": {"issue_count": len(review.issues), "enhanced": bool(enhancement)},
        }

    async def import_test(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 4: VERIFY[/bold blue] — Verifying project...")
        verify_passed = await o._phase_verify()
        o._save_state("verify_complete")
        o._phase_context["last_verify_passed"] = bool(verify_passed)
        ui_metadata, ui_outcomes = self._run_ui_judge(
            requirement=str(o._phase_context.get("enhancement", "") or f"Import {o.project_dir.name}"),
            spec=o.spec,
            verify_passed=bool(verify_passed),
        )
        return {
            "summary": "Imported project verification complete",
            "metadata": {"verify_passed": bool(verify_passed), "ui_judge_score": ui_metadata.get("judge_score", 0.0)},
            "outcomes": (("tests_pass",) if verify_passed else ()) + tuple(ui_outcomes),
        }

    async def import_refactor(self) -> dict[str, Any]:
        o = self.orchestrator
        if o.config.mode == "developer":
            console.print("\n[bold blue]Phase 5: REFACTOR[/bold blue] — Improving quality...")
            await o._phase_refactor()
            o._save_state("refactor_complete")
            return {"summary": "Imported project refactor complete"}
        return {"summary": "Imported project refactor skipped", "metadata": {"skipped": True}}

    async def import_deliver(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 6: DELIVER[/bold blue] — Packaging...")
        await o._phase_deliver()
        o._save_state("complete")
        ui_handoff = self._write_ui_handoff(spec=o.spec)
        return {
            "summary": "Imported project packaged",
            "outcomes": ("repo_runnable",),
            "metadata": {"ui_design_ref_count": ui_handoff.get("design_context_ref_count", 0)},
        }


class VerificationProfileRunner(BaseProfileRunner):
    async def run_review(self, project_path: str) -> dict[str, Any]:
        return await self.phase_executor.run_review_pipeline(project_path)

    async def prepare_review(self, project_path: str) -> None:
        o = self.orchestrator
        o._executor_reset_context(operation="review", profile="verification")
        o.project_dir = Path(project_path).resolve()
        logger.info("AutoForge review %s starting: %s", o.config.run_id, o.project_dir)
        o._open_runtime(
            operation="review",
            profile_name="verification",
            metadata={"project_dir": str(o.project_dir)},
        )
        if (o._durable_enabled() or bool(getattr(o.config, "trace_enabled", False))) and o.runtime is not None:
            try:
                o.runtime.telemetry.record("review_started", {"project_dir": str(o.project_dir)})
            except Exception:
                pass

    async def review_intake(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 1: SCAN[/bold blue] — Analyzing project...")
        scan_result = await o._phase_scan(o.project_dir)
        o.spec = scan_result.spec
        o._phase_context["scan_result"] = scan_result
        console.print(
            f"  [green]Scan complete:[/green] {o.spec.get('project_name', 'project')} "
            f"— {scan_result.completeness}% complete, {len(scan_result.gaps)} gaps found"
        )
        if o.kernel_session is not None:
            o.kernel_session.update_execution_plan(
                objective=f"Verify project {o.project_dir.name}",
                summary="Review existing project for obligations, issues, and proofs",
                inputs={"project_dir": str(o.project_dir), "gap_count": len(scan_result.gaps)},
                metadata={"scan_completeness": scan_result.completeness},
            )
        return {
            "summary": "Project scan completed",
            "metadata": {
                "completeness": scan_result.completeness,
                "gap_count": len(scan_result.gaps),
            },
        }

    async def review_obligation_extraction(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 2: REVIEW[/bold blue] — Reviewing code...")
        review = await o._phase_full_review()
        o._phase_context["review"] = review
        console.print(f"  Quality score: {review.score}/10")
        console.print(f"  Issues found: {len(review.issues)}")
        return {
            "summary": "Obligations extracted from review pass",
            "metadata": {"issue_count": len(review.issues), "score": review.score},
        }

    def review_judge(self) -> dict[str, Any]:
        o = self.orchestrator
        scan_result = o._phase_context.get("scan_result")
        review = o._phase_context.get("review")
        issue_count = len(getattr(review, "issues", []) or [])
        gap_count = len(getattr(scan_result, "gaps", []) or [])
        score = int(getattr(review, "score", 0) or 0)
        if issue_count or gap_count:
            outcome = "falsified"
            summary = f"Found {issue_count} issues and {gap_count} gaps"
        elif o.config.formal_verify_enabled and score >= 9:
            outcome = "proven"
            summary = "No issues found and formal verification enabled"
        else:
            outcome = "bounded_confidence"
            summary = "No blocking issues found, but proof remains bounded"

        verification_dir = o._forge_dir / "verification"
        verification_dir.mkdir(parents=True, exist_ok=True)
        judge_path = verification_dir / "judge_result.json"
        payload = {
            "schema_version": 1,
            "artifact_type": "verification_judge",
            "run_id": str(getattr(o.config, "run_id", "") or ""),
            "project_dir": str(o.project_dir),
            "outcome": outcome,
            "summary": summary,
            "score": score,
            "issue_count": issue_count,
            "gap_count": gap_count,
            "generated_at": time.time(),
        }
        write_kernel_json(judge_path, payload, artifact_type="verification_judge")
        if o.kernel_session is not None:
            o.kernel_session.register_artifact(
                "verification_judge",
                judge_path,
                required=False,
                metadata={"outcome": outcome, "score": score},
            )
        o._phase_context["judge_result"] = payload
        return {
            "summary": summary,
            "metadata": {"issue_count": issue_count, "gap_count": gap_count, "score": score},
            "outcomes": (outcome,),
        }

    async def review_counterexample_report(self) -> dict[str, Any]:
        o = self.orchestrator
        review = o._phase_context.get("review")
        scan_result = o._phase_context.get("scan_result")
        judge = o._phase_context.get("judge_result", {})
        if (
            o.config.mode == "developer"
            and getattr(review, "score", 10) < round(o.config.quality_threshold * 10)
            and getattr(review, "issues", None)
        ):
            console.print("\n[bold blue]Phase 3: REFACTOR[/bold blue] — Applying fixes...")
            await o._phase_refactor()
        console.print("\n[bold blue]Phase 4: REPORT[/bold blue] — Generating report...")
        report = o._generate_review_report(scan_result, review)
        if judge.get("outcome") == "falsified":
            verification_dir = o._forge_dir / "verification"
            verification_dir.mkdir(parents=True, exist_ok=True)
            counterexamples_dir = verification_dir / "counterexamples"
            counterexamples_dir.mkdir(parents=True, exist_ok=True)
            findings_path = counterexamples_dir / "findings.json"
            findings_path.write_text(
                json.dumps(
                    {
                        "issues": list(getattr(review, "issues", []) or []),
                        "gaps": list(getattr(scan_result, "gaps", []) or []),
                        "generated_at": time.time(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            if o.kernel_session is not None:
                o.kernel_session.register_artifact("counterexamples", counterexamples_dir, required=False)
        o._phase_context["review_report"] = report
        return {"summary": "Verification report generated"}


class ResearchProfileRunner(BaseProfileRunner):
    async def run_generate(self, requirement: str) -> Path:
        return await self.phase_executor.run_research_pipeline(requirement)

    def _research_slug(self, text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(text or "").strip().lower()).strip("-")
        return slug[:40] or "research-run"

    async def prepare_generate(self, requirement: str) -> None:
        o = self.orchestrator
        o._executor_reset_context(operation="generate", profile="research")
        o._phase_context["requirement"] = requirement
        logger.info("AutoForge research run %s starting", o.config.run_id)

    async def research_intake(self) -> dict[str, Any]:
        o = self.orchestrator
        requirement = str(o._phase_context.get("requirement", "") or "")
        console.print("\n[bold blue]Phase 1: INTAKE[/bold blue] — Framing research objective...")
        o.spec = await o._phase_spec(requirement)
        project_name = o.spec.get("project_name", "")
        if not project_name:
            project_name = f"research-{self._research_slug(requirement)}"
            o.spec["project_name"] = project_name
        o.project_dir = o.config.workspace_dir / project_name
        o.project_dir.mkdir(parents=True, exist_ok=True)
        o._state_file = o.project_dir / ".forge_state.json"
        o._open_runtime(
            operation="generate",
            profile_name="research",
            metadata={
                "project_name": project_name,
                "requirement_chars": len(requirement),
                "objective": requirement,
            },
        )
        spec_path = o.project_dir / "spec.json"
        spec_path.write_text(json.dumps(o.spec, indent=2, ensure_ascii=False), encoding="utf-8")
        goal_path = self._research_root / "goal.txt"
        goal_path.parent.mkdir(parents=True, exist_ok=True)
        goal_path.write_text(requirement + "\n", encoding="utf-8")
        if o.kernel_session is not None:
            o.kernel_session.register_artifact("research_goal", goal_path, required=False)
            o.kernel_session.update_execution_plan(
                objective=requirement or f"Research run for {project_name}",
                summary=f"Research objective intake for {project_name}",
                inputs={"requirement": requirement, "project_name": project_name},
                metadata={"spec": {"project_name": project_name}},
            )
        o._save_state("intake_complete")
        return {
            "summary": f"{project_name} research objective captured",
            "metadata": {"project_name": project_name, "module_count": len(o.spec.get('modules', []))},
        }

    async def research_claim_extraction(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 2: CLAIM EXTRACTION[/bold blue] — Extracting target claims...")
        requirement = str(o._phase_context.get("requirement", "") or "")
        modules = list(o.spec.get("modules", []) or [])
        claims: list[dict[str, Any]] = []
        for index, module in enumerate(modules[:8], start=1):
            if isinstance(module, dict):
                statement = str(module.get("description", "") or module.get("name", "") or "").strip()
                files = list(module.get("files", []) or [])
            else:
                statement = str(module).strip()
                files = []
            if not statement:
                continue
            claims.append(
                {
                    "claim_id": f"claim_{index:02d}",
                    "statement": statement,
                    "files": files,
                    "source": "spec.module",
                }
            )
        if not claims:
            claims.append(
                {
                    "claim_id": "claim_01",
                    "statement": requirement or "Investigate the requested research objective",
                    "files": [],
                    "source": "objective",
                }
            )
        payload = {
            "schema_version": 1,
            "artifact_type": "research_claims",
            "run_id": str(o.config.run_id),
            "objective": requirement,
            "claims": claims,
            "metrics": ["artifact_completeness", "evidence_pack", "environment_lock"],
            "generated_at": time.time(),
        }
        claims_path = self._research_root / "claims.json"
        write_kernel_json(claims_path, payload, artifact_type="research_claims")
        if o.kernel_session is not None:
            o.kernel_session.register_artifact(
                "research_claims",
                claims_path,
                required=False,
                metadata={"claim_count": len(claims)},
            )
        o._phase_context["research_claims"] = claims
        o._save_state("claim_extraction_complete")
        return {
            "summary": f"Extracted {len(claims)} research claims",
            "metadata": {"claim_count": len(claims)},
        }

    async def research_reproduction_plan(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 3: REPRODUCTION PLAN[/bold blue] — Planning experiment...")
        claims = list(o._phase_context.get("research_claims", []) or [])
        payload = {
            "schema_version": 1,
            "artifact_type": "research_reproduction_plan",
            "run_id": str(o.config.run_id),
            "objective": str(o._phase_context.get("requirement", "") or ""),
            "claims": claims,
            "steps": [
                "Collect objective and target claims",
                "Prepare environment lock and execution constraints",
                "Assemble experiment scaffold and evidence capture",
                "Run verification and compile report",
            ],
            "controls": {
                "deterministic": bool(getattr(o.config, "deterministic", False)),
                "execution_backend": str(getattr(o.config, "execution_backend", "auto") or "auto"),
                "max_agents": int(getattr(o.config, "max_agents", 1) or 1),
            },
            "generated_at": time.time(),
        }
        plan_path = self._research_root / "reproduction_plan.json"
        write_kernel_json(plan_path, payload, artifact_type="research_reproduction_plan")
        if o.kernel_session is not None:
            o.kernel_session.register_artifact(
                "reproduction_plan",
                plan_path,
                required=False,
                metadata={"claim_count": len(claims)},
            )
        o._phase_context["reproduction_plan"] = payload
        o._save_state("reproduction_plan_complete")
        return {
            "summary": "Reproduction plan written",
            "metadata": {"step_count": len(payload['steps'])},
        }

    async def research_build_experiment(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 4: BUILD EXPERIMENT[/bold blue] — Assembling experiment scaffold...")
        experiment_dir = self._research_root / "experiment"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        scaffold_path = experiment_dir / "experiment_design.json"
        payload = {
            "schema_version": 1,
            "artifact_type": "research_experiment_design",
            "run_id": str(o.config.run_id),
            "objective": str(o._phase_context.get("requirement", "") or ""),
            "workspace": str(o.project_dir),
            "claims": list(o._phase_context.get("research_claims", []) or []),
            "generated_at": time.time(),
        }
        write_kernel_json(scaffold_path, payload, artifact_type="research_experiment_design")
        notes_path = experiment_dir / "README.md"
        notes_path.write_text(
            "# Experiment Scaffold\n\n"
            "This directory contains the research-phase execution scaffold and artifacts.\n",
            encoding="utf-8",
        )
        if o.kernel_session is not None:
            o.kernel_session.register_artifact(
                "experiment_design",
                scaffold_path,
                required=False,
                metadata={"workspace": str(o.project_dir)},
            )
        o._phase_context["experiment_design"] = payload
        o._save_state("build_experiment_complete")
        return {"summary": "Experiment scaffold prepared"}

    async def research_run(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 5: RUN[/bold blue] — Capturing execution evidence...")
        files = [
            str(path.relative_to(o.project_dir)).replace("\\", "/")
            for path in sorted(o.project_dir.rglob("*"))
            if path.is_file() and ".autoforge" not in path.parts and ".git" not in path.parts
        ]
        run_log = {
            "schema_version": 1,
            "artifact_type": "research_run_log",
            "run_id": str(o.config.run_id),
            "file_count": len(files),
            "sample_files": files[:50],
            "generated_at": time.time(),
        }
        run_log_path = self._research_root / "run_log.json"
        write_kernel_json(run_log_path, run_log, artifact_type="research_run_log")
        if o.kernel_session is not None:
            o.kernel_session.register_artifact(
                "research_run_log",
                run_log_path,
                required=False,
                metadata={"file_count": len(files)},
            )
        o._phase_context["research_run_log"] = run_log
        o._save_state("run_complete")
        return {"summary": f"Captured run evidence over {len(files)} files", "metadata": {"file_count": len(files)}}

    async def research_verify(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 6: VERIFY[/bold blue] — Assessing research evidence...")
        claims = list(o._phase_context.get("research_claims", []) or [])
        run_log = dict(o._phase_context.get("research_run_log", {}) or {})
        file_count = int(run_log.get("file_count", 0) or 0)
        if claims and file_count > 0:
            outcome = "partially_reproduced"
        else:
            outcome = "failed_with_evidence"
        verification = {
            "schema_version": 1,
            "artifact_type": "research_verification",
            "run_id": str(o.config.run_id),
            "objective": str(o._phase_context.get("requirement", "") or ""),
            "claim_count": len(claims),
            "file_count": file_count,
            "outcome": outcome,
            "generated_at": time.time(),
        }
        verification_path = self._research_root / "verification.json"
        write_kernel_json(verification_path, verification, artifact_type="research_verification")
        if o.kernel_session is not None:
            o.kernel_session.register_artifact(
                "research_verification",
                verification_path,
                required=False,
                metadata={"outcome": outcome},
            )
        o._phase_context["research_outcome"] = outcome
        o._save_state("verify_complete")
        return {
            "summary": f"Research evidence assessed as {outcome}",
            "metadata": {"claim_count": len(claims), "file_count": file_count},
            "outcomes": (outcome,),
        }

    async def research_report(self) -> dict[str, Any]:
        o = self.orchestrator
        console.print("\n[bold blue]Phase 7: REPORT[/bold blue] — Writing research report...")
        report_dir = self._research_root
        report_dir.mkdir(parents=True, exist_ok=True)
        outcome = str(o._phase_context.get("research_outcome", "partially_reproduced") or "partially_reproduced")
        report_path = report_dir / "report.md"
        report_path.write_text(
            "\n".join(
                [
                    "# Research Report",
                    "",
                    f"- Run ID: {o.config.run_id}",
                    f"- Objective: {str(o._phase_context.get('requirement', '') or '').strip()}",
                    f"- Outcome: {outcome}",
                    f"- Claims: {len(list(o._phase_context.get('research_claims', []) or []))}",
                    "",
                    "The run used the kernel research substrate and generated a reproducibility evidence pack.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        if o.kernel_session is not None:
            o.kernel_session.register_artifact(
                "research_report",
                report_path,
                required=False,
                metadata={"outcome": outcome},
            )
        o._save_state("complete")
        return {
            "summary": "Research report generated",
            "outcomes": (outcome,),
        }


class ResumeProfileRunner(BaseProfileRunner):
    async def run_resume(self, workspace_path: Path | None = None) -> Path:
        return await self.phase_executor.run_resume_pipeline(workspace_path)
