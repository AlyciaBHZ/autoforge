"""Batch harness runner (JSONL dataset) for industrial evaluation."""

from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from autoforge.engine.config import ForgeConfig
from autoforge.engine.harness.dataset import HarnessCase, load_dataset
from autoforge.engine.harness.environment import (
    build_plan_from_dockerfile,
    docker_build_image,
    docker_image_exists,
)
from autoforge.engine.harness.openai_export import export_run_to_openai_bundle
from autoforge.engine.harness.judge import (
    diff_directories,
    hide_paths,
    patch_similarity,
    run_test_command,
    write_json,
)
from autoforge.engine.harness.report import CaseMetrics, HarnessReport, classify_error
from autoforge.engine.runtime.runtime import ForgeRuntime


@dataclass(frozen=True)
class HarnessRunConfig:
    concurrency: int = 1
    out_dir: Path | None = None
    docker_required: bool = True
    trace_enabled: bool = True


def _copytree_clean(src: Path, dst: Path) -> None:
    ignore_names = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".env",
        ".autoforge",
    }
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(*sorted(ignore_names)),
    )


def _safe_relpath(rel: str) -> str:
    rel = (rel or "").replace("\\", "/").lstrip("/")
    if ".." in Path(rel).parts:
        raise ValueError("Path traversal not allowed")
    return rel


def _restore_holdout_into_project(project_dir: Path, rel_paths: list[str]) -> None:
    """Restore holdout tests from .autoforge/harness_holdout_hidden into project root."""
    if not rel_paths:
        return
    hidden_root = project_dir / ".autoforge" / "harness_holdout_hidden"
    for rel in rel_paths:
        rel = _safe_relpath(rel)
        hidden = (hidden_root / rel).resolve()
        if not hidden.exists():
            continue
        dst = (project_dir / rel).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            try:
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            except Exception:
                pass
        shutil.move(str(hidden), str(dst))


async def prewarm_dataset_images(
    dataset_path: Path,
    *,
    out_dir: Path,
) -> dict[str, str]:
    """Build docker images referenced via env.dockerfile; returns mapping case_id->tag."""
    cases = load_dataset(dataset_path)
    mapping: dict[str, str] = {}
    base_dir = out_dir / "prewarm"
    base_dir.mkdir(parents=True, exist_ok=True)
    for c in cases:
        dockerfile = (c.env.dockerfile or "").strip()
        if not dockerfile:
            continue
        plan = build_plan_from_dockerfile(dockerfile, base_dir=base_dir)
        exists = await docker_image_exists(plan.tag, cwd=base_dir)
        if not exists:
            ok = await docker_build_image(plan)
            if not ok:
                continue
        mapping[c.id] = plan.tag
    return mapping


async def run_dataset(
    base_config: ForgeConfig,
    dataset_path: Path,
    *,
    cfg: HarnessRunConfig,
) -> HarnessReport:
    run_id = f"harness-{uuid.uuid4().hex[:12]}"
    out_dir = (
        cfg.out_dir
        or (base_config.project_root / ".autoforge" / "harness" / "runs" / run_id)
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)

    report = HarnessReport(run_id=run_id)
    cases = load_dataset(dataset_path)

    # Persist a copy of the dataset for provenance.
    try:
        shutil.copyfile(str(dataset_path), str(out_dir / "dataset.jsonl"))
    except Exception:
        pass

    sem = asyncio.Semaphore(max(1, int(cfg.concurrency)))

    async def _run_case(case: HarnessCase) -> CaseMetrics:
        started = time.monotonic()
        case_root = (out_dir / "cases" / case.id).resolve()
        case_root.mkdir(parents=True, exist_ok=True)

        baseline_dir = case_root / "baseline"
        project_src = Path(case.project_path).resolve() if case.project_path else None
        if project_src is not None and project_src.is_dir():
            _copytree_clean(project_src, baseline_dir)

        # Derive per-case config (isolated, no shared sub-config objects).
        workspace_dir = (case_root / "workspace").resolve()
        workspace_dir.mkdir(parents=True, exist_ok=True)
        cconf = base_config.fork(workspace_dir=workspace_dir)
        cconf.run_id = f"{run_id}-{case.id}-{uuid.uuid4().hex[:6]}"

        if case.budget_usd is not None:
            cconf.budget_limit_usd = float(case.budget_usd)
        if case.max_agents is not None:
            cconf.max_agents = max(1, int(case.max_agents))

        # Harness anti-cheat: no web tools.
        cconf.tools.web_tools_enabled = False

        # Sandbox isolation defaults.
        cconf.docker_enabled = True
        cconf.docker_required = bool(
            cfg.docker_required if case.env.docker_required is None else case.env.docker_required
        )
        cconf.docker_network_mode = (case.env.docker_network_mode or "none").strip() or "none"
        if case.env.docker_memory_limit:
            cconf.docker_memory_limit = case.env.docker_memory_limit
        if case.env.docker_cpu_limit:
            cconf.docker_cpu_limit = case.env.docker_cpu_limit
        if case.env.docker_pids_limit is not None:
            cconf.docker_pids_limit = int(case.env.docker_pids_limit)
        elif int(getattr(cconf, "docker_pids_limit", 0) or 0) <= 0:
            cconf.docker_pids_limit = 512
        if case.env.sandbox_image:
            cconf.sandbox_image = case.env.sandbox_image
        elif case.env.dockerfile:
            env_base = (out_dir / "env").resolve()
            env_base.mkdir(parents=True, exist_ok=True)
            plan = build_plan_from_dockerfile(case.env.dockerfile, base_dir=env_base)
            exists = await docker_image_exists(plan.tag, cwd=env_base)
            if not exists:
                built = await docker_build_image(plan)
                if not built:
                    raise RuntimeError(f"Failed to build docker image for case {case.id}")
            cconf.sandbox_image = plan.tag

        # Tracing controls.
        trace_on = bool(cfg.trace_enabled and case.trace.enabled)
        cconf.obs.trace_enabled = trace_on
        cconf.obs.trace_capture_llm_content = bool(case.trace.llm_content)
        cconf.obs.trace_capture_command_output = bool(case.trace.command_output)
        cconf.obs.trace_capture_fs_snapshots = bool(case.trace.fs_snapshots)

        project_dir: Path | None = None
        error = ""
        ok = False

        try:
            from autoforge.engine.orchestrator import Orchestrator

            orchestrator = Orchestrator(cconf)
            try:
                if case.mode == "generate":
                    project_dir = await orchestrator.run(case.description)
                elif case.mode == "review":
                    if not baseline_dir.is_dir():
                        raise RuntimeError("baseline_dir missing for review case")
                    work = (case_root / "project").resolve()
                    _copytree_clean(baseline_dir, work)
                    if case.judge.hide_paths:
                        try:
                            hide_paths(work, case.judge.hide_paths)
                        except Exception:
                            pass
                    _ = await orchestrator.review_project(str(work))
                    project_dir = work
                else:  # import
                    if project_src is None or not project_src.is_dir():
                        raise RuntimeError("project_path missing for import case")
                    if case.judge.hide_paths and baseline_dir.is_dir():
                        # Hide holdout tests inside the source copy so they end up in
                        # project_dir/.autoforge/... (not visible to the agent).
                        try:
                            hide_paths(baseline_dir, case.judge.hide_paths)
                        except Exception:
                            pass
                    project_dir = await orchestrator.import_project(str(baseline_dir), case.enhance)
                ok = True
            finally:
                try:
                    await orchestrator.llm.close()
                except Exception:
                    pass
        except Exception as e:
            ok = False
            error = str(e)

        # Post-run judging: tests + diffs + golden patch (if any).
        visible_ok: bool | None = None
        hidden_ok: bool | None = None
        golden_sim: float | None = None
        diff_counts: dict[str, int] = {}

        if project_dir is not None and project_dir.is_dir():
            telemetry = None
            runtime: ForgeRuntime | None = None
            if trace_on:
                runtime = ForgeRuntime.create(
                    project_dir,
                    cconf.run_id,
                    artifacts_enabled=bool(getattr(cconf, "artifacts_enabled", True)),
                    trace_enabled=True,
                    trace_write_header=False,
                    trace_capture_llm_content=bool(cconf.obs.trace_capture_llm_content),
                    trace_capture_command_output=bool(cconf.obs.trace_capture_command_output),
                    trace_capture_fs_snapshots=bool(cconf.obs.trace_capture_fs_snapshots),
                    trace_max_inline_chars=int(getattr(cconf.obs, "trace_max_inline_chars", 20000) or 20000),
                    trace_redact_secrets=bool(getattr(cconf.obs, "trace_redact_secrets", True)),
                    trace_secrets=[s for s in list(getattr(cconf, "api_keys", {}).values()) if isinstance(s, str) and s],
                )
                telemetry = runtime.telemetry

            try:
                if case.judge.visible_test_command:
                    tr = await run_test_command(
                        cconf,
                        project_dir,
                        command=case.judge.visible_test_command,
                        telemetry=telemetry,
                    )
                    visible_ok = tr.ok
                    if telemetry is not None:
                        try:
                            telemetry.record(
                                "harness_visible_test",
                                {"ok": tr.ok, "exit_code": tr.exit_code, "command": tr.command},
                            )
                        except Exception:
                            pass

                if case.judge.hidden_test_command:
                    if case.judge.hide_paths:
                        try:
                            _restore_holdout_into_project(project_dir, case.judge.hide_paths)
                        except Exception:
                            pass
                    tr = await run_test_command(
                        cconf,
                        project_dir,
                        command=case.judge.hidden_test_command,
                        telemetry=telemetry,
                    )
                    hidden_ok = tr.ok
                    if telemetry is not None:
                        try:
                            telemetry.record(
                                "harness_hidden_test",
                                {"ok": tr.ok, "exit_code": tr.exit_code, "command": tr.command},
                            )
                        except Exception:
                            pass

                if baseline_dir.is_dir():
                    dd = diff_directories(baseline_dir, project_dir, exclude_rel_paths=case.judge.hide_paths)
                    diff_counts = {
                        "added": len(dd.added),
                        "removed": len(dd.removed),
                        "changed": len(dd.changed),
                    }
                    write_json(
                        case_root / "dir_diff.json",
                        dd.to_dict() | {"patch_preview": dd.patch[:5000]},
                    )

                    if case.judge.golden_patch_path and Path(case.judge.golden_patch_path).is_file():
                        golden = Path(case.judge.golden_patch_path).read_text(
                            encoding="utf-8",
                            errors="replace",
                        )
                        golden_sim = patch_similarity(dd.patch, golden)
                        write_json(
                            case_root / "golden_patch_score.json",
                            {
                                "similarity": golden_sim,
                                "golden_patch_path": case.judge.golden_patch_path,
                            },
                        )
            finally:
                if runtime is not None:
                    try:
                        runtime.close()
                    except Exception:
                        pass

        duration = time.monotonic() - started
        metrics = CaseMetrics(
            case_id=case.id,
            mode=case.mode,
            ok=ok,
            duration_seconds=float(duration),
            cost_usd=float(getattr(cconf, "estimated_cost_usd", 0.0)),
            input_tokens=int(getattr(cconf, "total_input_tokens", 0)),
            output_tokens=int(getattr(cconf, "total_output_tokens", 0)),
            project_dir=str(project_dir) if project_dir else "",
            error=error,
            error_type=classify_error(error),
            visible_tests_ok=visible_ok,
            hidden_tests_ok=hidden_ok,
            golden_patch_similarity=golden_sim,
            diff_counts=diff_counts,
        )
        write_json(case_root / "case_result.json", metrics.to_dict())
        return metrics

    async def _worker(case: HarnessCase) -> None:
        async with sem:
            m = await _run_case(case)
            report.cases.append(m)

    await asyncio.gather(*[asyncio.create_task(_worker(c)) for c in cases])
    report.finalize()

    write_json(out_dir / "report.json", report.to_dict())
    try:
        export_run_to_openai_bundle(out_dir, out_dir=out_dir / "openai_eval_bundle")
    except Exception as exc:
        write_json(
            out_dir / "openai_eval_bundle_error.json",
            {"error": str(exc)},
        )
    return report
