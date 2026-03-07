from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from autoforge.cli.app import _build_config_overrides, build_parser
from autoforge.engine.config import ForgeConfig
from autoforge.engine.run_controller import RunController
from autoforge.engine.phase_executor import PhaseExecutor
from autoforge.engine.kernel import (
    KernelSession,
    WorkspaceLock,
    apply_harness_judge_overlay,
    create_runtime_from_config,
    default_profile_for_command,
    export_evidence_pack,
    export_replay_bundle,
    find_latest_kernel_checkpoint,
    inspect_kernel_run,
    load_kernel_event_stream,
    load_latest_kernel_checkpoint,
    render_kernel_inspection,
    render_kernel_event_stream,
    evaluate_contract_verdict,
    resolve_profile,
    resolve_kernel_run_dir,
)


def _make_local_tmp_dir() -> Path:
    path = Path.cwd() / ".tmp_kernel_tests" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def _create_research_run(
    tmp_path: Path,
    *,
    run_id: str = "run-kernel-1",
    project_dir: Path | None = None,
) -> tuple[Path, ForgeConfig]:
    workspace_dir = tmp_path / "workspace"
    if project_dir is None:
        project_dir = workspace_dir / "demo"
    else:
        workspace_dir = project_dir.parent
    project_dir.mkdir(parents=True)
    config = ForgeConfig(
        project_root=tmp_path,
        workspace_dir=workspace_dir,
        run_id=run_id,
        profile="research",
    )
    runtime = create_runtime_from_config(config, project_dir)
    try:
        session = KernelSession.open(
            config=config,
            project_dir=project_dir,
            runtime=runtime,
            profile_name="research",
            operation="generate",
            surface="cli",
            metadata={"purpose": "test", "objective": "reproduce a claim"},
        )
        session.record_phase("intake", state="completed", summary="intake done")
        session.write_checkpoint(
            state_marker="intake",
            state_version=1,
            state={
                "run_id": run_id,
                "operation": "generate",
                "profile": "research",
                "phase": "intake",
            },
        )
        turn = session.start_turn(kind="phase", phase="report", input_summary="write report")
        session.emit_item(
            turn=turn,
            item_type="report",
            summary="report emitted",
            payload={"path": "report.md"},
        )
        session.complete_turn(turn, output_summary="report complete")
        session.record_inbox_message(source="cli:test", kind="note", text="please include evidence")
        session.close(status="completed")
    finally:
        runtime.close()
    return project_dir, config


def test_profile_registry_contracts_are_stable():
    development = resolve_profile("dev")
    verification = resolve_profile("verification")
    research = resolve_profile("research")

    assert development.phase_graph.phase_ids() == (
        "spec",
        "build",
        "test",
        "refactor",
        "deliver",
    )
    assert development.success_contract.is_success("tests_pass") is True
    assert verification.phase_graph.phase_ids() == (
        "intake",
        "obligation_extraction",
        "check_prove",
        "counterexample_report",
    )
    assert research.artifact_contract.required_kinds() == (
        "brief",
        "env_lock",
        "experiment_code",
        "metrics",
        "evidence_pack",
    )
    assert default_profile_for_command("review", mode="developer") == "verification"
    assert development.phase_graph.phase_for_resume_marker("verify_complete") == "test"
    assert verification.phase_graph.phase_for_resume_marker("report") == "counterexample_report"
    assert research.phase_graph.phase_for_resume_marker("claim_extraction_complete") == "claim_extraction"
    assert research.phase_graph.phase_for_resume_marker("verify_complete") == "verify"


def test_workspace_lock_prevents_double_acquire():
    tmp_path = _make_local_tmp_dir()
    try:
        project_dir = tmp_path / "workspace" / "demo"
        project_dir.mkdir(parents=True)

        lock = WorkspaceLock(project_dir)
        assert lock.acquire(holder="cli:1", run_id="run-a", ttl_seconds=120) is True
        assert lock.acquire(holder="cli:2", run_id="run-b", ttl_seconds=120) is False
        assert lock.heartbeat(holder="cli:1", run_id="run-a", ttl_seconds=240) is True
        record = lock.inspect()
        assert record is not None
        assert record.holder == "cli:1"
        assert lock.release(holder="cli:1", run_id="run-a") is True
        assert lock.acquire(holder="cli:2", run_id="run-b", ttl_seconds=120) is True
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_kernel_session_writes_manifest_event_log_and_env_lock():
    tmp_path = _make_local_tmp_dir()
    try:
        project_dir, config = _create_research_run(tmp_path)
        run_dir = project_dir / ".autoforge" / "kernel" / "runs" / "run-kernel-1"
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        env_lock = json.loads((run_dir / "environment.lock.json").read_text(encoding="utf-8"))
        artifact_manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        verdict = json.loads((run_dir / "contract_verdict.json").read_text(encoding="utf-8"))
        plan = json.loads((run_dir / "execution_plan.json").read_text(encoding="utf-8"))
        events = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        inbox = json.loads((run_dir / "inbox.json").read_text(encoding="utf-8"))
        store = (project_dir / ".autoforge" / "kernel" / "run_store.sqlite3")

        assert manifest["status"] == "completed"
        assert manifest["profile"] == "research"
        assert manifest["schema_version"] == 1
        assert manifest["lineage_id"] == "run-kernel-1"
        assert env_lock["schema_version"] == 1
        assert env_lock["profile"] == "research"
        assert env_lock["runtime"]["backend"] == config.execution_backend
        assert plan["schema_version"] == 1
        assert plan["operation"] == "generate"
        assert plan["lineage_id"] == "run-kernel-1"
        assert plan["objective"] == "reproduce a claim"
        assert verdict["contract_satisfied"] is True
        assert artifact_manifest["schema_version"] == 1
        assert "partially_reproduced" in verdict["inferred_outcomes"]
        assert any(item["kind"] == "experiment_code" for item in artifact_manifest["artifacts"])
        assert any(item["kind"] == "env_lock" for item in artifact_manifest["artifacts"])
        assert any(item["kind"] == "brief" for item in artifact_manifest["artifacts"])
        assert any(item["kind"] == "metrics" for item in artifact_manifest["artifacts"])
        assert any(item["kind"] == "evidence_pack" for item in artifact_manifest["artifacts"])
        assert any(item["kind"] == "kernel_checkpoint" for item in artifact_manifest["artifacts"])
        assert any(item["kind"] == "contract_verdict" for item in artifact_manifest["artifacts"])
        assert any('"kind": "thread_started"' in line for line in events)
        assert any('"kind": "thread_completed"' in line for line in events)
        assert inbox["schema_version"] == 1
        assert inbox["messages"][0]["text"] == "please include evidence"
        assert store.exists()
        assert not (project_dir / ".autoforge" / "kernel" / "workspace.lock.json").exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_kernel_checkpoint_roundtrip_and_discovery():
    tmp_path = _make_local_tmp_dir()
    try:
        project_dir, _config = _create_research_run(tmp_path, run_id="run-kernel-ckpt")
        checkpoint_path = find_latest_kernel_checkpoint(project_dir)
        checkpoint = load_latest_kernel_checkpoint(project_dir)

        assert checkpoint_path is not None
        assert checkpoint_path.name == "checkpoint.json"
        assert checkpoint is not None
        assert checkpoint.run_id == "run-kernel-ckpt"
        assert checkpoint.state_marker == "intake"
        assert checkpoint.state["profile"] == "research"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_inspector_and_evidence_pack_export():
    tmp_path = _make_local_tmp_dir()
    try:
        project_dir, _config = _create_research_run(tmp_path, run_id="run-kernel-2")
        inspection = inspect_kernel_run(project_dir, tail=10)
        resolved = resolve_kernel_run_dir(
            project_dir / ".autoforge" / "kernel" / "runs" / "run-kernel-2" / "manifest.json"
        )
        rendered = render_kernel_inspection(inspection)

        assert resolved.name == "run-kernel-2"
        assert inspection.manifest["run_id"] == "run-kernel-2"
        assert inspection.plan["profile"] == "research"
        assert inspection.event_count >= 1
        assert "run_id: run-kernel-2" in rendered
        assert "objective: reproduce a claim" in rendered

        pack = export_evidence_pack(project_dir, run_id="run-kernel-2")
        assert pack.output_dir.exists()
        assert pack.manifest_path.exists()
        assert pack.summary_path.exists()
        summary = json.loads(pack.summary_path.read_text(encoding="utf-8"))
        assert summary["manifest"]["run_id"] == "run-kernel-2"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_inspector_migrates_legacy_unschemed_kernel_payloads():
    tmp_path = _make_local_tmp_dir()
    try:
        run_dir = tmp_path / "workspace" / "demo" / ".autoforge" / "kernel" / "runs" / "legacy-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"run_id": "legacy-run", "profile": "development", "status": "completed"}, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "profile.json").write_text(
            json.dumps({"name": "development"}, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "contracts.json").write_text(
            json.dumps({"success_contract": {}, "artifact_contract": {}}, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "artifact_manifest.json").write_text(
            json.dumps({"artifacts": []}, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "inbox.json").write_text(
            json.dumps({"messages": []}, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "events.jsonl").write_text("", encoding="utf-8")

        inspection = inspect_kernel_run(run_dir)
        assert inspection.manifest["schema_version"] == 1
        assert inspection.profile["schema_version"] == 1
        assert inspection.contracts["schema_version"] == 1
        assert inspection.artifact_manifest["schema_version"] == 1
        assert inspection.inbox["schema_version"] == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_event_stream_and_replay_bundle_export():
    tmp_path = _make_local_tmp_dir()
    try:
        project_dir, _config = _create_research_run(tmp_path, run_id="run-kernel-3")
        trajectory_path = project_dir / ".autoforge" / "trajectory_task-a.json"
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(
            json.dumps({"task_id": "task-a", "steps": [{"reward": 0.9}]}, ensure_ascii=False),
            encoding="utf-8",
        )

        stream = load_kernel_event_stream(project_dir, tail=5)
        rendered = render_kernel_event_stream(stream)

        assert stream.run_id == "run-kernel-3"
        assert stream.event_count >= 1
        assert stream.event_kind_counts["thread_started"] == 1
        assert "events_tail:" in rendered

        bundle = export_replay_bundle(project_dir, run_id="run-kernel-3")
        assert bundle.output_dir.exists()
        assert bundle.manifest_path.exists()
        assert bundle.summary_path.exists()
        assert bundle.rubric_path.exists()
        assert bundle.items_path.exists()
        assert bundle.schema_path.exists()

        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(bundle.summary_path.read_text(encoding="utf-8"))
        rubric = json.loads(bundle.rubric_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in bundle.items_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert manifest["bundle_type"] == "kernel_replay_judge_bundle"
        assert summary["event_stream"]["run_id"] == "run-kernel-3"
        assert summary["trajectory_count"] == 1
        assert rows[0]["item"]["run_id"] == "run-kernel-3"
        assert rows[0]["sample"]["expected_status"] == "completed"
        assert any(check["id"] == "status_completed" for check in rubric["checks"])
        assert any(path == "trajectories/trajectory_task-a.json" for path in bundle.copied_files)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_replay_bundle_embeds_harness_judge_results():
    tmp_path = _make_local_tmp_dir()
    try:
        harness_run_dir = tmp_path / "harness_run"
        project_dir = harness_run_dir / "cases" / "todo_case" / "workspace"
        project_dir, _config = _create_research_run(
            tmp_path,
            run_id="run-kernel-4",
            project_dir=project_dir,
        )

        (harness_run_dir / "dataset.jsonl").write_text(
            json.dumps(
                {
                    "id": "todo_case",
                    "mode": "generate",
                    "description": "Build a todo CLI",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (harness_run_dir / "report.json").write_text(
            json.dumps(
                {
                    "summary": {"run_id": "harness-demo", "total_cases": 1},
                    "cases": [
                        {
                            "case_id": "todo_case",
                            "ok": True,
                            "visible_tests_ok": True,
                            "hidden_tests_ok": False,
                            "golden_patch_similarity": 0.87,
                            "diff_counts": {"added": 2, "removed": 0, "changed": 1},
                            "error": "",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        bundle_dir = harness_run_dir / "openai_eval_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "bundle_manifest.json").write_text(
            json.dumps({"source_type": "harness_run", "case_count": 1}, ensure_ascii=False),
            encoding="utf-8",
        )
        case_root = harness_run_dir / "cases" / "todo_case"
        (case_root / "case_result.json").write_text("{}", encoding="utf-8")
        (case_root / "dir_diff.json").write_text("{}", encoding="utf-8")
        (case_root / "golden_patch_score.json").write_text(
            json.dumps({"similarity": 0.87}, ensure_ascii=False),
            encoding="utf-8",
        )

        bundle = export_replay_bundle(project_dir, run_id="run-kernel-4")
        summary = json.loads(bundle.summary_path.read_text(encoding="utf-8"))
        rubric = json.loads(bundle.rubric_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in bundle.items_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert summary["harness"]["case_id"] == "todo_case"
        assert summary["harness"]["judge_results"]["visible_tests_ok"] is True
        assert summary["harness"]["judge_results"]["hidden_tests_ok"] is False
        assert rows[0]["item"]["judge_results"]["golden_patch_similarity"] == 0.87
        assert rows[0]["sample"]["observed_harness_results"]["hidden_tests_ok"] is False
        assert any(check["id"] == "harness_judge_results_present" for check in rubric["checks"])
        assert any(path == "harness/report.json" for path in bundle.copied_files)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_harness_overlay_updates_contract_verdict():
    tmp_path = _make_local_tmp_dir()
    try:
        project_dir, _config = _create_research_run(tmp_path, run_id="run-kernel-5")
        overlay = apply_harness_judge_overlay(
            project_dir,
            run_id="run-kernel-5",
            payload={
                "case_id": "demo",
                "visible_tests_ok": True,
                "hidden_tests_ok": False,
                "golden_patch_similarity": 0.61,
                "ok": True,
            },
        )
        verdict = json.loads((overlay.verdict_path).read_text(encoding="utf-8"))
        judge = json.loads((overlay.judge_path).read_text(encoding="utf-8"))

        assert judge["hidden_tests_ok"] is False
        assert verdict["metadata"]["harness_judge"]["hidden_tests_ok"] is False
        assert overlay.contract_satisfied is True
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_development_contract_requires_all_success_outcomes():
    profile = resolve_profile("development")
    verdict = evaluate_contract_verdict(
        profile=profile,
        run_id="run-dev-1",
        operation="generate",
        surface="cli",
        status="budget_exceeded",
        phase_history=[],
        artifacts=[
            {"kind": "code", "exists": True},
            {"kind": "deploy_guide", "exists": True},
        ],
        declared_outcomes=["partial_delivery"],
        metadata={},
    )

    assert verdict.allowed_outcome_satisfied is True
    assert verdict.success_outcome_satisfied is False
    assert verdict.contract_satisfied is False


def test_cli_profile_override_and_config_normalization():
    parser = build_parser()
    args = parser.parse_args(["--profile", "research", "generate", "demo"])
    overrides = _build_config_overrides(args)

    assert overrides["profile"] == "research"
    assert overrides["client_surface"] == "cli"

    config = ForgeConfig(profile="verify", client_surface="")
    assert config.profile == "verification"
    assert config.client_surface == "cli"

    forked = config.fork()
    assert forked.profile == "verification"
    assert forked.client_surface == "cli"

    defaulted = ForgeConfig()
    assert defaulted.profile == ""
    assert default_profile_for_command("review", mode=defaulted.mode, explicit_profile=defaulted.profile) == "verification"

    with patch("autoforge.engine.config._load_global_config", return_value={}):
        env_defaulted = ForgeConfig.from_env()
    assert env_defaulted.profile == ""
    assert default_profile_for_command("review", mode=env_defaulted.mode, explicit_profile=env_defaulted.profile) == "verification"


def test_resume_operation_prefers_research_profile_over_generate_operation():
    from autoforge.engine.orchestrator import Orchestrator

    operation = Orchestrator._resume_operation_from_state(
        object(),
        {
            "profile": "research",
            "operation": "generate",
        },
    )

    assert operation == "research"


def test_cli_kernel_subcommands_parse():
    parser = build_parser()
    args = parser.parse_args(["kernel", "inspect", "workspace/demo", "--tail", "5"])
    assert args.command == "kernel"
    assert args.kernel_action == "inspect"
    assert args.tail == 5

    args = parser.parse_args(["kernel", "events", "workspace/demo", "--tail", "7", "--follow"])
    assert args.command == "kernel"
    assert args.kernel_action == "events"
    assert args.tail == 7
    assert args.follow is True

    args = parser.parse_args(["kernel", "evidence-pack", "workspace/demo", "--kernel-only"])
    assert args.command == "kernel"
    assert args.kernel_action == "evidence-pack"
    assert args.kernel_only is True

    args = parser.parse_args(["kernel", "replay-bundle", "workspace/demo", "--kernel-only"])
    assert args.command == "kernel"
    assert args.kernel_action == "replay-bundle"
    assert args.kernel_only is True


def test_run_controller_dispatches_to_profile_runners():
    development_orchestrator = SimpleNamespace(config=ForgeConfig())
    research_orchestrator = SimpleNamespace(config=ForgeConfig(profile="research"))
    review_orchestrator = SimpleNamespace(config=ForgeConfig())
    import_orchestrator = SimpleNamespace(config=ForgeConfig())
    resume_orchestrator = SimpleNamespace(config=ForgeConfig())

    with (
        patch(
            "autoforge.engine.run_controller.DevelopmentProfileRunner.run_generate",
            new_callable=AsyncMock,
            return_value=Path("development-demo"),
        ) as dev_generate,
        patch(
            "autoforge.engine.run_controller.ResearchProfileRunner.run_generate",
            new_callable=AsyncMock,
            return_value=Path("research-demo"),
        ) as research_generate,
        patch(
            "autoforge.engine.run_controller.VerificationProfileRunner.run_review",
            new_callable=AsyncMock,
            return_value={"path": "repo"},
        ) as run_review,
        patch(
            "autoforge.engine.run_controller.DevelopmentProfileRunner.run_import",
            new_callable=AsyncMock,
            return_value=Path("repo:enhance"),
        ) as run_import,
        patch(
            "autoforge.engine.run_controller.ResumeProfileRunner.run_resume",
            new_callable=AsyncMock,
            return_value=Path("workspace"),
        ) as run_resume,
    ):
        assert asyncio.run(RunController(development_orchestrator).run_generate("demo")) == Path("development-demo")
        assert asyncio.run(RunController(research_orchestrator).run_generate("demo")) == Path("research-demo")
        assert asyncio.run(RunController(review_orchestrator).run_review("repo")) == {"path": "repo"}
        assert asyncio.run(RunController(import_orchestrator).run_import("repo", "enhance")) == Path("repo:enhance")
        assert asyncio.run(RunController(resume_orchestrator).run_resume(Path("workspace"))) == Path("workspace")

    dev_generate.assert_awaited_once_with("demo")
    research_generate.assert_awaited_once_with("demo")
    run_review.assert_awaited_once_with("repo")
    run_import.assert_awaited_once_with("repo", "enhance")
    run_resume.assert_awaited_once_with(Path("workspace"))


def test_phase_executor_runs_graph_and_resume_from_successor():
    class _GraphOrchestrator:
        def __init__(self) -> None:
            self.events: list[tuple[Any, ...]] = []
            self.config = ForgeConfig()
            self.project_dir = Path("workspace")
            self._phase_context: dict[str, Any] = {}

        def _kernel_phase(self, phase: str, state: str, *, summary: str = "", metadata: dict[str, Any] | None = None) -> None:
            self.events.append(("phase", phase, state, summary))

        def _declare_kernel_outcomes(self, *outcomes: str) -> None:
            self.events.append(("outcomes",) + tuple(outcomes))

        async def _checkpoint(self, phase: str, summary: str) -> None:
            self.events.append(("checkpoint", phase, summary))

        def _print_summary(self) -> None:
            self.events.append(("summary",))

        def _close_runtime(self, *, status: str) -> None:
            self.events.append(("close", status))

        def _save_state(self, phase: str) -> None:
            self.events.append(("save_state", phase))

        def _resolve_resume_workspace(self, workspace_path: Path | None = None) -> Path:
            return workspace_path or Path("workspace")

        def _load_resume_state_snapshot(self, workspace_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
            return {
                "profile": "development",
                "operation": "generate",
                "phase": "spec_complete",
            }, {
                "loaded_from_kernel": True,
                "loaded_from_journal": False,
            }

        def _executor_reset_context(self, *, operation: str, profile: str) -> None:
            self.events.append(("reset", operation, profile))
            self._phase_context = {"operation": operation, "profile": profile}

        def _restore_resumed_state(self, state: dict[str, Any], *, resume_metadata: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
            self.events.append(("restore", state["phase"]))
            return str(state["phase"]), ()

        def _resume_operation_from_state(self, state: dict[str, Any]) -> str:
            return "generate"

        def _resume_start_phase(self, *, profile_name: str, marker: str) -> str | None:
            self.events.append(("resume_start", profile_name, marker))
            return "build"

    stub = _GraphOrchestrator()
    executor = PhaseExecutor(stub)

    async def _prepare_generate(self, requirement: str) -> None:
        stub.events.append(("prepare_generate", requirement))

    async def _generate_spec(self) -> dict[str, Any]:
        stub.events.append(("handler", "spec"))
        return {"summary": "spec done", "checkpoint_summary": "spec checkpoint"}

    async def _generate_build(self) -> dict[str, Any]:
        stub.events.append(("handler", "build"))
        return {"summary": "build done", "checkpoint_summary": "build checkpoint"}

    async def _generate_test(self) -> dict[str, Any]:
        stub.events.append(("handler", "test"))
        return {"summary": "test done", "outcomes": ("tests_pass",), "checkpoint_summary": "verify checkpoint"}

    async def _generate_refactor(self) -> dict[str, Any]:
        stub.events.append(("handler", "refactor"))
        return {"summary": "refactor done"}

    async def _generate_deliver(self) -> dict[str, Any]:
        stub.events.append(("handler", "deliver"))
        return {"summary": "deliver done", "outcomes": ("repo_runnable",)}

    with (
        patch("autoforge.engine.profile_runner.DevelopmentProfileRunner.prepare_generate", new=_prepare_generate),
        patch("autoforge.engine.profile_runner.DevelopmentProfileRunner.generate_spec", new=_generate_spec),
        patch("autoforge.engine.profile_runner.DevelopmentProfileRunner.generate_build", new=_generate_build),
        patch("autoforge.engine.profile_runner.DevelopmentProfileRunner.generate_test", new=_generate_test),
        patch("autoforge.engine.profile_runner.DevelopmentProfileRunner.generate_refactor", new=_generate_refactor),
        patch("autoforge.engine.profile_runner.DevelopmentProfileRunner.generate_deliver", new=_generate_deliver),
    ):
        result = asyncio.run(executor.run_generate_pipeline("demo"))
        assert result == Path("workspace")
        handler_order = [event[1] for event in stub.events if event[0] == "handler"]
        assert handler_order == ["spec", "build", "test", "refactor", "deliver"]
        checkpoint_phases = [event[1] for event in stub.events if event[0] == "checkpoint"]
        assert checkpoint_phases == ["spec", "build", "verify"]
        assert ("outcomes", "tests_pass") in stub.events
        assert ("outcomes", "repo_runnable") in stub.events
        assert stub.events[-1] == ("close", "completed")

        stub.events.clear()
        resumed = asyncio.run(executor.run_resume_pipeline(Path("workspace")))
        assert resumed == Path("workspace")
        resumed_handlers = [event[1] for event in stub.events if event[0] == "handler"]
        assert resumed_handlers == ["build", "test", "refactor", "deliver"]


def test_phase_executor_runs_research_graph_and_resume_from_successor():
    class _ResearchGraphOrchestrator:
        def __init__(self) -> None:
            self.events: list[tuple[Any, ...]] = []
            self.config = ForgeConfig(profile="research")
            self.project_dir = Path("workspace")
            self._phase_context: dict[str, Any] = {}

        def _kernel_phase(self, phase: str, state: str, *, summary: str = "", metadata: dict[str, Any] | None = None) -> None:
            self.events.append(("phase", phase, state, summary))

        def _declare_kernel_outcomes(self, *outcomes: str) -> None:
            self.events.append(("outcomes",) + tuple(outcomes))

        async def _checkpoint(self, phase: str, summary: str) -> None:
            self.events.append(("checkpoint", phase, summary))

        def _print_summary(self) -> None:
            self.events.append(("summary",))

        def _close_runtime(self, *, status: str) -> None:
            self.events.append(("close", status))

        def _save_state(self, phase: str) -> None:
            self.events.append(("save_state", phase))

        def _resolve_resume_workspace(self, workspace_path: Path | None = None) -> Path:
            return workspace_path or Path("workspace")

        def _load_resume_state_snapshot(self, workspace_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
            return {
                "profile": "research",
                "operation": "research",
                "phase": "claim_extraction_complete",
            }, {
                "loaded_from_kernel": True,
                "loaded_from_journal": False,
            }

        def _executor_reset_context(self, *, operation: str, profile: str) -> None:
            self.events.append(("reset", operation, profile))
            self._phase_context = {"operation": operation, "profile": profile}

        def _restore_resumed_state(self, state: dict[str, Any], *, resume_metadata: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
            self.events.append(("restore", state["phase"]))
            return str(state["phase"]), ()

        def _resume_operation_from_state(self, state: dict[str, Any]) -> str:
            return "research"

        def _resume_start_phase(self, *, profile_name: str, marker: str) -> str | None:
            self.events.append(("resume_start", profile_name, marker))
            return "reproduction_plan"

    stub = _ResearchGraphOrchestrator()
    executor = PhaseExecutor(stub)

    async def _prepare_generate(self, requirement: str) -> None:
        stub.events.append(("prepare_generate", requirement))

    async def _research_intake(self) -> dict[str, Any]:
        stub.events.append(("handler", "intake"))
        return {"summary": "intake done"}

    async def _research_claims(self) -> dict[str, Any]:
        stub.events.append(("handler", "claim_extraction"))
        return {"summary": "claims done"}

    async def _research_plan(self) -> dict[str, Any]:
        stub.events.append(("handler", "reproduction_plan"))
        return {"summary": "plan done", "checkpoint_summary": "plan checkpoint"}

    async def _research_build(self) -> dict[str, Any]:
        stub.events.append(("handler", "build_experiment"))
        return {"summary": "build done"}

    async def _research_run(self) -> dict[str, Any]:
        stub.events.append(("handler", "run"))
        return {"summary": "run done"}

    async def _research_verify(self) -> dict[str, Any]:
        stub.events.append(("handler", "verify"))
        return {"summary": "verify done", "outcomes": ("partially_reproduced",)}

    async def _research_report(self) -> dict[str, Any]:
        stub.events.append(("handler", "report"))
        return {"summary": "report done", "outcomes": ("partially_reproduced",)}

    with (
        patch("autoforge.engine.profile_runner.ResearchProfileRunner.prepare_generate", new=_prepare_generate),
        patch("autoforge.engine.profile_runner.ResearchProfileRunner.research_intake", new=_research_intake),
        patch("autoforge.engine.profile_runner.ResearchProfileRunner.research_claim_extraction", new=_research_claims),
        patch("autoforge.engine.profile_runner.ResearchProfileRunner.research_reproduction_plan", new=_research_plan),
        patch("autoforge.engine.profile_runner.ResearchProfileRunner.research_build_experiment", new=_research_build),
        patch("autoforge.engine.profile_runner.ResearchProfileRunner.research_run", new=_research_run),
        patch("autoforge.engine.profile_runner.ResearchProfileRunner.research_verify", new=_research_verify),
        patch("autoforge.engine.profile_runner.ResearchProfileRunner.research_report", new=_research_report),
    ):
        result = asyncio.run(executor.run_research_pipeline("demo research"))
        assert result == Path("workspace")
        handler_order = [event[1] for event in stub.events if event[0] == "handler"]
        assert handler_order == [
            "intake",
            "claim_extraction",
            "reproduction_plan",
            "build_experiment",
            "run",
            "verify",
            "report",
        ]
        assert ("outcomes", "partially_reproduced") in stub.events
        assert stub.events[-1] == ("close", "completed")

        stub.events.clear()
        resumed = asyncio.run(executor.run_resume_pipeline(Path("workspace")))
        assert resumed == Path("workspace")
        resumed_handlers = [event[1] for event in stub.events if event[0] == "handler"]
        assert resumed_handlers == ["reproduction_plan", "build_experiment", "run", "verify", "report"]
