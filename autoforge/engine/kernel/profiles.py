from __future__ import annotations

from autoforge.engine.kernel.contracts import (
    ArtifactContract,
    ArtifactSpec,
    KernelProfile,
    PhaseEdge,
    PhaseGraph,
    PhaseNode,
    SuccessContract,
)


def _phase(
    phase_id: str,
    description: str,
    *,
    handler: str = "",
    resume_markers: tuple[str, ...] = (),
) -> tuple[str, str, str, tuple[str, ...]]:
    return (phase_id, description, handler, resume_markers)


def _linear_graph(*phases: tuple[str, str, str, tuple[str, ...]]) -> PhaseGraph:
    nodes = tuple(
        PhaseNode(
            id=phase_id,
            description=description,
            terminal=index == len(phases) - 1,
            handler=handler or phase_id,
            resume_markers=tuple(resume_markers),
        )
        for index, (phase_id, description, handler, resume_markers) in enumerate(phases)
    )
    edges = tuple(
        PhaseEdge(source=phases[index][0], target=phases[index + 1][0], label="next")
        for index in range(len(phases) - 1)
    )
    return PhaseGraph(start=phases[0][0], phases=nodes, edges=edges)


DEVELOPMENT_PROFILE = KernelProfile(
    name="development",
    summary="Builds runnable repositories with deployable outputs and durable traces.",
    phase_graph=_linear_graph(
        _phase(
            "spec",
            "Requirement shaping, project decomposition, and execution planning.",
            handler="phase_spec",
            resume_markers=("scan_complete", "spec_complete", "spec"),
        ),
        _phase(
            "build",
            "Implementation, scaffolding, and workspace mutation.",
            handler="phase_build",
            resume_markers=("build_in_progress", "build_complete", "review_complete", "enhance_complete", "build"),
        ),
        _phase(
            "test",
            "Runtime checks, integration tests, and verification loops.",
            handler="phase_verify",
            resume_markers=("verify_in_progress", "verify_complete", "dependencies_prepared", "test", "verify"),
        ),
        _phase(
            "refactor",
            "Quality hardening and cleanup after feedback.",
            handler="phase_refactor",
            resume_markers=("refactor_in_progress", "refactor_complete", "refactor"),
        ),
        _phase(
            "deliver",
            "Packaging, handoff artifacts, and deploy guidance.",
            handler="phase_deliver",
            resume_markers=("complete", "budget_exceeded", "deliver"),
        ),
    ),
    success_contract=SuccessContract(
        description="The repository is runnable and its tests pass.",
        success_outcomes=("repo_runnable", "tests_pass"),
        allowed_outcomes=("repo_runnable", "tests_pass", "partial_delivery", "failed"),
        satisfaction_mode="all",
    ),
    artifact_contract=ArtifactContract(
        artifacts=(
            ArtifactSpec(
                kind="code",
                description="Runnable project workspace or patch set.",
                path_hint="project_root/",
                media_type="application/x-directory",
                tags=("workspace", "source"),
            ),
            ArtifactSpec(
                kind="deploy_guide",
                description="Deployment or operator handoff instructions.",
                path_hint="DEPLOY_GUIDE.md",
                media_type="text/markdown",
                tags=("handoff",),
            ),
            ArtifactSpec(
                kind="traces",
                description="Kernel/runtime traces for replay and post-mortem debugging.",
                required=False,
                path_hint=".autoforge/traces/",
                media_type="application/jsonl",
                tags=("observability", "replay"),
            ),
        )
    ),
    tags=("build", "delivery"),
    phase_aliases={"test": ("verify",)},
)


VERIFICATION_PROFILE = KernelProfile(
    name="verification",
    summary="Extracts obligations, checks them, and produces proofs or counterexamples.",
    phase_graph=_linear_graph(
        _phase(
            "intake",
            "Collect targets, scope, and verification surface.",
            handler="phase_scan",
            resume_markers=("scan", "intake"),
        ),
        _phase(
            "obligation_extraction",
            "Derive claims, invariants, and proof obligations.",
            handler="phase_review",
            resume_markers=("review", "obligation_extraction"),
        ),
        _phase(
            "check_prove",
            "Run proofs, checks, scans, or refutations.",
            handler="phase_verify",
            resume_markers=("verify", "formal_verify", "check_prove"),
        ),
        _phase(
            "counterexample_report",
            "Summarize failures, limits, or proved guarantees.",
            handler="phase_report",
            resume_markers=("report", "complete", "counterexample_report"),
        ),
    ),
    success_contract=SuccessContract(
        description="Every run ends in a proof, a falsification, or bounded confidence.",
        success_outcomes=("proven", "falsified", "bounded_confidence"),
        allowed_outcomes=("proven", "falsified", "bounded_confidence", "failed"),
        satisfaction_mode="any",
    ),
    artifact_contract=ArtifactContract(
        artifacts=(
            ArtifactSpec(
                kind="obligation_ledger",
                description="Structured list of extracted claims or obligations.",
                path_hint=".autoforge/verification/obligations.json",
                media_type="application/json",
                tags=("verification", "claims"),
            ),
            ArtifactSpec(
                kind="verification_report",
                description="Primary proof, refutation, or analysis report.",
                path_hint=".autoforge/review_report.json",
                media_type="application/json",
                tags=("report", "verification"),
            ),
            ArtifactSpec(
                kind="verification_judge",
                description="Structured judge verdict over obligations and evidence.",
                required=False,
                path_hint=".autoforge/verification/judge_result.json",
                media_type="application/json",
                tags=("judge", "verification"),
            ),
            ArtifactSpec(
                kind="counterexamples",
                description="Counterexamples, failing traces, or repro cases.",
                required=False,
                path_hint=".autoforge/verification/counterexamples/",
                media_type="application/json",
                tags=("debug", "evidence"),
            ),
            ArtifactSpec(
                kind="traces",
                description="Replayable kernel/runtime traces.",
                required=False,
                path_hint=".autoforge/traces/",
                media_type="application/jsonl",
                tags=("observability", "replay"),
            ),
        )
    ),
    tags=("audit", "verification"),
    phase_aliases={
        "intake": ("scan",),
        "obligation_extraction": ("review",),
        "check_prove": ("verify", "formal_verify"),
    },
)


RESEARCH_PROFILE = KernelProfile(
    name="research",
    summary="Runs reproduction-oriented research workflows with environment locks and evidence packs.",
    phase_graph=_linear_graph(
        _phase(
            "intake",
            "Collect claims, hypotheses, and source materials.",
            handler="phase_intake",
            resume_markers=("scan", "spec", "intake", "intake_complete"),
        ),
        _phase(
            "claim_extraction",
            "Distill target claims and measurable outcomes.",
            handler="phase_claim_extraction",
            resume_markers=("claim_extraction", "claim_extraction_complete"),
        ),
        _phase(
            "reproduction_plan",
            "Define experiment plan, controls, and environment.",
            handler="phase_reproduction_plan",
            resume_markers=("plan", "reproduction_plan", "reproduction_plan_complete"),
        ),
        _phase(
            "build_experiment",
            "Implement experiment code and execution harness.",
            handler="phase_build_experiment",
            resume_markers=("build", "enhance", "build_experiment", "build_experiment_complete"),
        ),
        _phase(
            "run",
            "Execute experiments or proofs.",
            handler="phase_run",
            resume_markers=("run", "run_complete"),
        ),
        _phase(
            "verify",
            "Check outcomes against claims and controls.",
            handler="phase_verify",
            resume_markers=("verify", "verify_complete"),
        ),
        _phase(
            "report",
            "Produce evidence-backed research report.",
            handler="phase_report",
            resume_markers=("report", "complete"),
        ),
    ),
    success_contract=SuccessContract(
        description="Research runs end in reproduced, partially reproduced, or failed with evidence.",
        success_outcomes=("reproduced", "partially_reproduced", "failed_with_evidence"),
        allowed_outcomes=(
            "reproduced",
            "partially_reproduced",
            "failed_with_evidence",
            "failed",
        ),
        satisfaction_mode="any",
    ),
    artifact_contract=ArtifactContract(
        artifacts=(
            ArtifactSpec(
                kind="brief",
                description="Research brief or reproduction objective summary.",
                path_hint=".autoforge/research/brief.md",
                media_type="text/markdown",
                tags=("planning",),
            ),
            ArtifactSpec(
                kind="env_lock",
                description="Pinned runtime environment and execution policy lock.",
                path_hint=".autoforge/kernel/runs/<run_id>/environment.lock.json",
                media_type="application/json",
                tags=("reproducibility", "environment"),
            ),
            ArtifactSpec(
                kind="experiment_code",
                description="Experiment harnesses, scripts, or notebooks.",
                path_hint="project_root/",
                media_type="application/x-directory",
                tags=("source", "experiment"),
            ),
            ArtifactSpec(
                kind="metrics",
                description="Metrics, measurements, and scored outputs.",
                path_hint=".autoforge/research/metrics.json",
                media_type="application/json",
                tags=("metrics",),
            ),
            ArtifactSpec(
                kind="evidence_pack",
                description="Collected traces, logs, outputs, and supporting evidence.",
                path_hint=".autoforge/research/evidence_pack/",
                media_type="application/x-directory",
                tags=("evidence", "replay"),
            ),
        )
    ),
    tags=("research", "reproducibility"),
)


PROFILE_REGISTRY: dict[str, KernelProfile] = {
    DEVELOPMENT_PROFILE.name: DEVELOPMENT_PROFILE,
    VERIFICATION_PROFILE.name: VERIFICATION_PROFILE,
    RESEARCH_PROFILE.name: RESEARCH_PROFILE,
}


def resolve_profile(name: str | None) -> KernelProfile:
    raw = str(name or "").strip().lower()
    if not raw:
        raw = "development"
    aliases = {
        "developer": "development",
        "dev": "development",
        "verify": "verification",
        "verification": "verification",
        "academic": "research",
        "research": "research",
    }
    canonical = aliases.get(raw, raw)
    if canonical not in PROFILE_REGISTRY:
        raise KeyError(f"Unknown kernel profile: {name}")
    return PROFILE_REGISTRY[canonical]


def default_profile_for_command(
    command: str | None,
    *,
    mode: str | None = None,
    explicit_profile: str | None = None,
) -> str:
    explicit = str(explicit_profile or "").strip().lower()
    if explicit:
        return resolve_profile(explicit).name
    cmd = str(command or "").strip().lower()
    normalized_mode = str(mode or "").strip().lower()
    if cmd in {"review"}:
        return "verification"
    if cmd in {"paper"} or normalized_mode == "research":
        return "research"
    return "development"
