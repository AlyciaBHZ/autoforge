from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoforge.engine.kernel.events import load_kernel_event_stream
from autoforge.engine.kernel.inspector import inspect_kernel_run


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _copy_if_exists(src: Path, dst: Path, *, exclude_roots: tuple[Path, ...] = ()) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        resolved_excludes = tuple(root.resolve() for root in exclude_roots)

        def _ignore(directory: str, names: list[str]) -> list[str]:
            if not resolved_excludes:
                return []
            current = Path(directory).resolve()
            ignored: list[str] = []
            for name in names:
                child = (current / name).resolve()
                for root in resolved_excludes:
                    try:
                        child.relative_to(root)
                        ignored.append(name)
                        break
                    except ValueError:
                        continue
            return ignored

        shutil.copytree(src, dst, ignore=_ignore)
    else:
        shutil.copy2(src, dst)
    return True


def _relative_to(base_dir: Path, target: Path) -> str:
    try:
        return str(target.resolve().relative_to(base_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(target)


def _locate_trajectories(project_dir: Path) -> list[Path]:
    root = project_dir / ".autoforge"
    if not root.is_dir():
        return []
    return sorted(root.glob("trajectory_*.json"), key=lambda item: item.name)


def _find_harness_run_dir(project_dir: Path) -> Path | None:
    for candidate in (project_dir, *project_dir.parents):
        if not (candidate / "dataset.jsonl").is_file() or not (candidate / "report.json").is_file():
            continue
        try:
            relative = project_dir.resolve().relative_to(candidate.resolve())
        except Exception:
            continue
        parts = relative.parts
        if len(parts) >= 3 and parts[0] == "cases" and parts[2] == "workspace":
            return candidate
    return None


def _infer_harness_case_id(project_dir: Path, harness_run_dir: Path) -> str:
    try:
        relative = project_dir.resolve().relative_to(harness_run_dir.resolve())
    except Exception:
        return ""
    parts = relative.parts
    if len(parts) >= 3 and parts[0] == "cases" and parts[2] == "workspace":
        return str(parts[1])
    return ""


def _load_harness_context(
    *,
    project_dir: Path,
    output_dir: Path,
    copied: list[str],
) -> dict[str, Any]:
    harness_run_dir = _find_harness_run_dir(project_dir)
    if harness_run_dir is None:
        return {}

    report = _read_json(harness_run_dir / "report.json")
    bundle_manifest = _read_json(harness_run_dir / "openai_eval_bundle" / "bundle_manifest.json")
    case_id = _infer_harness_case_id(project_dir, harness_run_dir)
    case_metrics: dict[str, Any] = {}
    if case_id:
        for entry in report.get("cases", []):
            if isinstance(entry, dict) and str(entry.get("case_id", "")).strip() == case_id:
                case_metrics = entry
                break

    harness_targets = [
        ("dataset.jsonl", harness_run_dir / "dataset.jsonl"),
        ("report.json", harness_run_dir / "report.json"),
    ]
    for rel_name, src in harness_targets:
        if _copy_if_exists(src, output_dir / "harness" / rel_name):
            copied.append(f"harness/{rel_name}")

    bundle_dir = harness_run_dir / "openai_eval_bundle"
    if _copy_if_exists(bundle_dir, output_dir / "harness" / "openai_eval_bundle"):
        copied.append("harness/openai_eval_bundle")

    case_root = harness_run_dir / "cases" / case_id if case_id else None
    if case_root is not None:
        for rel_name in ("case_result.json", "dir_diff.json", "golden_patch_score.json"):
            src = case_root / rel_name
            if _copy_if_exists(src, output_dir / "harness" / "cases" / case_id / rel_name):
                copied.append(f"harness/cases/{case_id}/{rel_name}")

    judge_results = {
        "visible_tests_ok": case_metrics.get("visible_tests_ok"),
        "hidden_tests_ok": case_metrics.get("hidden_tests_ok"),
        "golden_patch_similarity": case_metrics.get("golden_patch_similarity"),
        "diff_counts": case_metrics.get("diff_counts", {}),
        "ok": case_metrics.get("ok"),
        "error": case_metrics.get("error", ""),
    }
    return {
        "run_dir": str(harness_run_dir),
        "case_id": case_id,
        "report_summary": report.get("summary", {}),
        "case_metrics": case_metrics,
        "openai_eval_bundle_manifest": bundle_manifest,
        "judge_results": judge_results,
    }


def _item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "required": ["run_id", "profile", "operation", "status", "contracts", "artifacts"],
        "properties": {
            "run_id": {"type": "string"},
            "profile": {"type": "string"},
            "operation": {"type": "string"},
            "status": {"type": "string"},
            "objective": {"type": "string"},
            "phase_history": {"type": "array"},
            "contracts": {"type": "object"},
            "artifacts": {"type": "array"},
            "events": {"type": "object"},
            "judge_results": {"type": "object"},
            "harness": {"type": "object"},
            "source": {"type": "object"},
        },
    }


def _judge_rubric(
    inspection,
    event_stream,
    copied_files: list[str],
    harness_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    success_contract = inspection.contracts.get("success_contract", {})
    artifact_contract = inspection.contracts.get("artifact_contract", {})
    required_kinds = list(artifact_contract.get("required_kinds", []))
    rubric = {
        "schema_version": 1,
        "run_id": inspection.manifest.get("run_id", ""),
        "profile": inspection.manifest.get("profile", ""),
        "operation": inspection.plan.get("operation", ""),
        "expected_status": "completed",
        "success_contract": success_contract,
        "required_artifacts": required_kinds,
        "checks": [
            {
                "id": "status_completed",
                "description": "Run should terminate in completed status for replay-grade export.",
                "expected": "completed",
            },
            {
                "id": "success_contract_satisfied",
                "description": success_contract.get("description", ""),
                "success_outcomes": list(success_contract.get("success_outcomes", [])),
                "allowed_outcomes": list(success_contract.get("allowed_outcomes", [])),
            },
            {
                "id": "required_artifacts_present",
                "description": "All required artifact kinds from the kernel artifact contract should exist.",
                "required_kinds": required_kinds,
            },
            {
                "id": "event_stream_complete",
                "description": "Replay bundle should include kernel events and summarized event counts.",
                "event_count": int(event_stream.event_count),
                "copied_kernel_event_log": "kernel/events.jsonl" in copied_files,
            },
        ],
    }
    harness = dict(harness_context or {})
    judge_results = harness.get("judge_results", {}) if isinstance(harness, dict) else {}
    if judge_results:
        rubric["checks"].append(
            {
                "id": "harness_judge_results_present",
                "description": "Replay bundle carries local harness judge verdicts for comparison.",
                "judge_results": judge_results,
                "case_id": harness.get("case_id", ""),
            }
        )
    return rubric


@dataclass(frozen=True)
class ReplayJudgeBundle:
    run_dir: Path
    output_dir: Path
    manifest_path: Path
    summary_path: Path
    rubric_path: Path
    items_path: Path
    schema_path: Path
    copied_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "summary_path": str(self.summary_path),
            "rubric_path": str(self.rubric_path),
            "items_path": str(self.items_path),
            "schema_path": str(self.schema_path),
            "copied_files": list(self.copied_files),
        }


def export_replay_bundle(
    path: Path,
    *,
    run_id: str | None = None,
    out_dir: Path | None = None,
    include_workspace_artifacts: bool = True,
) -> ReplayJudgeBundle:
    inspection = inspect_kernel_run(path, run_id=run_id, tail=50)
    event_stream = load_kernel_event_stream(path, run_id=run_id, tail=0)

    run_dir = inspection.run_dir
    raw_project_dir = str(inspection.manifest.get("project_dir", "") or "").strip()
    project_dir = (
        Path(raw_project_dir).resolve()
        if raw_project_dir
        else run_dir.parent.parent.parent.parent.resolve()
    )
    default_root = run_dir.parent.parent / "replay_bundles" / run_dir.name
    output_dir = (out_dir or default_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    kernel_files = [
        "manifest.json",
        "execution_plan.json",
        "contract_verdict.json",
        "harness_judge.json",
        "profile.json",
        "contracts.json",
        "environment.lock.json",
        "artifact_manifest.json",
        "events.jsonl",
        "inbox.json",
    ]
    for name in kernel_files:
        if _copy_if_exists(run_dir / name, output_dir / "kernel" / name):
            copied.append(f"kernel/{name}")
    run_store = run_dir.parent.parent / "run_store.sqlite3"
    if _copy_if_exists(run_store, output_dir / "kernel" / "run_store.sqlite3"):
        copied.append("kernel/run_store.sqlite3")

    if include_workspace_artifacts:
        for artifact in inspection.artifact_manifest.get("artifacts", []):
            absolute = Path(str(artifact.get("absolute_path", "") or "")).resolve()
            kind = str(artifact.get("kind", "") or "artifact")
            label = absolute.name or kind
            dst = output_dir / "artifacts" / kind / label
            try:
                if _copy_if_exists(absolute, dst, exclude_roots=(output_dir,)):
                    copied.append(f"artifacts/{kind}/{label}")
            except Exception:
                continue

    for trajectory_path in _locate_trajectories(project_dir):
        dst = output_dir / "trajectories" / trajectory_path.name
        if _copy_if_exists(trajectory_path, dst):
            copied.append(f"trajectories/{trajectory_path.name}")

    harness_context = _load_harness_context(project_dir=project_dir, output_dir=output_dir, copied=copied)

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "inspection": inspection.to_dict(),
        "event_stream": event_stream.to_dict(),
        "harness": harness_context,
        "copied_files": copied,
        "trajectory_count": len([item for item in copied if item.startswith("trajectories/")]),
    }
    _write_json(summary_path, summary_payload)

    rubric_path = output_dir / "judge_rubric.json"
    rubric = _judge_rubric(inspection, event_stream, copied, harness_context=harness_context)
    _write_json(rubric_path, rubric)

    item = {
        "run_id": inspection.manifest.get("run_id", ""),
        "thread_id": inspection.manifest.get("thread_id", ""),
        "profile": inspection.manifest.get("profile", ""),
        "operation": inspection.plan.get("operation", ""),
        "status": inspection.manifest.get("status", ""),
        "objective": inspection.plan.get("objective", ""),
        "phase_history": list(inspection.manifest.get("phase_history", [])),
        "contracts": {
            "success": inspection.contracts.get("success_contract", {}),
            "artifacts": inspection.contracts.get("artifact_contract", {}),
        },
        "artifacts": list(inspection.artifact_manifest.get("artifacts", [])),
        "events": {
            "count": event_stream.event_count,
            "kind_counts": dict(event_stream.event_kind_counts),
        },
        "judge_results": dict(harness_context.get("judge_results", {})),
        "harness": {
            "run_dir": harness_context.get("run_dir", ""),
            "case_id": harness_context.get("case_id", ""),
            "report_summary": harness_context.get("report_summary", {}),
            "openai_eval_bundle_manifest": harness_context.get("openai_eval_bundle_manifest", {}),
        },
        "source": {
            "run_dir": str(run_dir),
            "project_dir": str(project_dir),
            "summary_path": "summary.json",
            "judge_rubric_path": "judge_rubric.json",
        },
    }
    sample = {
        "expected_status": "completed",
        "required_artifacts": list(
            inspection.contracts.get("artifact_contract", {}).get("required_kinds", [])
        ),
        "judge_rubric_path": "judge_rubric.json",
        "observed_harness_results": dict(harness_context.get("judge_results", {})),
    }

    items_path = output_dir / "items.jsonl"
    schema_path = output_dir / "item_schema.json"
    _write_jsonl(items_path, [{"item": item, "sample": sample}])
    _write_json(schema_path, _item_schema())

    manifest_path = output_dir / "bundle_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "bundle_type": "kernel_replay_judge_bundle",
            "run_id": inspection.manifest.get("run_id", ""),
            "thread_id": inspection.manifest.get("thread_id", ""),
            "profile": inspection.manifest.get("profile", ""),
            "operation": inspection.plan.get("operation", ""),
            "status": inspection.manifest.get("status", ""),
            "source_run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "copied_files": copied,
            "include_workspace_artifacts": bool(include_workspace_artifacts),
            "summary_path": _relative_to(output_dir, summary_path),
            "judge_rubric_path": _relative_to(output_dir, rubric_path),
            "items_path": _relative_to(output_dir, items_path),
            "item_schema_path": _relative_to(output_dir, schema_path),
        },
    )

    return ReplayJudgeBundle(
        run_dir=run_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        rubric_path=rubric_path,
        items_path=items_path,
        schema_path=schema_path,
        copied_files=tuple(copied),
    )
