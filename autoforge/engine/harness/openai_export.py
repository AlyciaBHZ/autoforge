"""Export local AutoForge harness artifacts into OpenAI-friendly eval bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoforge.engine.harness.dataset import HarnessCase, load_dataset


@dataclass(frozen=True)
class OpenAIEvalBundle:
    """Filesystem bundle that can be ingested by OpenAI-style eval tooling."""

    source_type: str
    source_path: Path
    bundle_dir: Path
    items_path: Path
    schema_path: Path
    manifest_path: Path
    case_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_path": str(self.source_path),
            "bundle_dir": str(self.bundle_dir),
            "items_path": str(self.items_path),
            "schema_path": str(self.schema_path),
            "manifest_path": str(self.manifest_path),
            "case_count": self.case_count,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _bundle_dir_for_source(source_path: Path) -> Path:
    if source_path.is_dir():
        return source_path / "openai_eval_bundle"
    return source_path.parent / f"{source_path.stem}.openai_eval_bundle"


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _relative_to(base_dir: Path, target: Path | None) -> str:
    if target is None:
        return ""
    try:
        return str(target.resolve().relative_to(base_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(target)


def _locate_trace_path(project_dir: str) -> Path | None:
    if not project_dir:
        return None
    trace_root = Path(project_dir).resolve() / ".autoforge" / "traces"
    if not trace_root.is_dir():
        return None
    candidates = sorted(
        trace_root.glob("*.jsonl"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    return candidates[-1] if candidates else None


def _expected_outcome(
    case: HarnessCase,
    *,
    golden_similarity_threshold: float,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "require_ok": True,
        "require_visible_tests": bool(case.judge.visible_test_command),
        "require_hidden_tests": bool(case.judge.hidden_test_command),
    }
    if case.judge.golden_patch_path:
        expected["golden_similarity_threshold"] = float(golden_similarity_threshold)
    return expected


def _sample_payload(
    case: HarnessCase,
    *,
    include_golden_patch: bool,
    golden_similarity_threshold: float,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "expected_outcome": _expected_outcome(
            case,
            golden_similarity_threshold=golden_similarity_threshold,
        )
    }
    if include_golden_patch and case.judge.golden_patch_path:
        patch_path = Path(case.judge.golden_patch_path)
        if patch_path.is_file():
            sample["golden_patch"] = _safe_read_text(patch_path)
            sample["golden_patch_path"] = str(patch_path)
    return sample


def _case_prompt(case: HarnessCase) -> dict[str, Any]:
    return {
        "description": case.description,
        "project_path": case.project_path,
        "enhance": case.enhance,
    }


def _case_env(case: HarnessCase) -> dict[str, Any]:
    return {
        "sandbox_image": case.env.sandbox_image,
        "docker_memory_limit": case.env.docker_memory_limit,
        "docker_cpu_limit": case.env.docker_cpu_limit,
        "docker_pids_limit": case.env.docker_pids_limit,
        "docker_network_mode": case.env.docker_network_mode,
        "docker_required": case.env.docker_required,
        "dockerfile": case.env.dockerfile,
    }


def _case_judge(case: HarnessCase) -> dict[str, Any]:
    return {
        "visible_test_command": case.judge.visible_test_command,
        "hidden_test_command": case.judge.hidden_test_command,
        "hide_paths": list(case.judge.hide_paths),
        "golden_patch_path": case.judge.golden_patch_path,
        "has_golden_patch": bool(case.judge.golden_patch_path),
    }


def _case_trace(case: HarnessCase) -> dict[str, Any]:
    return {
        "enabled": case.trace.enabled,
        "llm_content": case.trace.llm_content,
        "command_output": case.trace.command_output,
        "fs_snapshots": case.trace.fs_snapshots,
    }


def _run_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "required": ["case_id", "mode", "prompt", "env", "judge", "trace"],
        "properties": {
            "case_id": {"type": "string"},
            "mode": {"type": "string"},
            "prompt": {"type": "object"},
            "env": {"type": "object"},
            "judge": {"type": "object"},
            "trace": {"type": "object"},
            "raw_case": {"type": "object"},
            "metrics": {"type": "object"},
            "artifacts": {"type": "object"},
            "source": {"type": "object"},
        },
    }


def _sample_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "expected_outcome": {"type": "object"},
            "golden_patch": {"type": "string"},
            "golden_patch_path": {"type": "string"},
        },
    }


def _build_dataset_row(
    case: HarnessCase,
    *,
    include_golden_patch: bool,
    golden_similarity_threshold: float,
) -> dict[str, Any]:
    return {
        "item": {
            "case_id": case.id,
            "mode": case.mode,
            "prompt": _case_prompt(case),
            "budget_usd": case.budget_usd,
            "max_agents": case.max_agents,
            "env": _case_env(case),
            "judge": _case_judge(case),
            "trace": _case_trace(case),
            "raw_case": case.raw,
        },
        "sample": _sample_payload(
            case,
            include_golden_patch=include_golden_patch,
            golden_similarity_threshold=golden_similarity_threshold,
        ),
    }


def export_dataset_to_openai_bundle(
    dataset_path: Path,
    *,
    out_dir: Path | None = None,
    include_golden_patch: bool = False,
    golden_similarity_threshold: float = 0.8,
) -> OpenAIEvalBundle:
    """Export a harness dataset into an OpenAI-friendly JSONL eval bundle."""
    source_path = dataset_path.resolve()
    cases = load_dataset(source_path)
    bundle_dir = (out_dir or _bundle_dir_for_source(source_path)).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _build_dataset_row(
            case,
            include_golden_patch=include_golden_patch,
            golden_similarity_threshold=golden_similarity_threshold,
        )
        for case in cases
    ]
    items_path = bundle_dir / "items.jsonl"
    schema_path = bundle_dir / "item_schema.json"
    manifest_path = bundle_dir / "bundle_manifest.json"
    _write_jsonl(items_path, rows)
    _write_json(schema_path, _run_item_schema())
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "source_type": "dataset",
            "source_path": str(source_path),
            "created_at": _utc_now_iso(),
            "case_count": len(cases),
            "items_path": items_path.name,
            "item_schema_path": schema_path.name,
            "sample_schema": _sample_schema(),
            "openai_format": {
                "record_shape": {"item": "object", "sample": "object"},
                "data_source_config": {
                    "type": "custom",
                    "item_schema_path": schema_path.name,
                },
            },
            "notes": [
                "Bundle mirrors AutoForge dataset cases into OpenAI custom-eval JSONL records.",
                "Use local harness runs for deterministic build/test execution; use this bundle for external grading or trace analysis.",
            ],
        },
    )
    return OpenAIEvalBundle(
        source_type="dataset",
        source_path=source_path,
        bundle_dir=bundle_dir,
        items_path=items_path,
        schema_path=schema_path,
        manifest_path=manifest_path,
        case_count=len(cases),
    )


def _load_run_payload(run_dir: Path) -> tuple[list[HarnessCase], dict[str, Any], dict[str, dict[str, Any]]]:
    dataset_path = run_dir / "dataset.jsonl"
    report_path = run_dir / "report.json"
    if not dataset_path.is_file() or not report_path.is_file():
        raise ValueError(
            f"Harness run directory must contain dataset.jsonl and report.json: {run_dir}"
        )
    cases = load_dataset(dataset_path)
    report = json.loads(_safe_read_text(report_path))
    metrics_by_case: dict[str, dict[str, Any]] = {}
    for entry in report.get("cases", []):
        if isinstance(entry, dict):
            case_id = str(entry.get("case_id", "")).strip()
            if case_id:
                metrics_by_case[case_id] = entry
    return cases, report, metrics_by_case


def _artifact_paths(run_dir: Path, case_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
    case_root = run_dir / "cases" / case_id
    project_dir = str(metrics.get("project_dir", "") or "")
    trace_path = _locate_trace_path(project_dir)
    artifacts = {
        "project_dir": project_dir,
        "case_result_path": _relative_to(run_dir, case_root / "case_result.json"),
        "dir_diff_path": _relative_to(run_dir, case_root / "dir_diff.json"),
        "golden_patch_score_path": _relative_to(run_dir, case_root / "golden_patch_score.json"),
        "trace_path": _relative_to(run_dir, trace_path),
    }
    return artifacts


def _build_run_row(
    run_dir: Path,
    case: HarnessCase,
    metrics: dict[str, Any],
    *,
    include_golden_patch: bool,
    golden_similarity_threshold: float,
) -> dict[str, Any]:
    return {
        "item": {
            "case_id": case.id,
            "mode": case.mode,
            "prompt": _case_prompt(case),
            "budget_usd": case.budget_usd,
            "max_agents": case.max_agents,
            "env": _case_env(case),
            "judge": _case_judge(case),
            "trace": _case_trace(case),
            "raw_case": case.raw,
            "metrics": metrics,
            "artifacts": _artifact_paths(run_dir, case.id, metrics),
            "source": {
                "type": "autoforge_harness_run",
                "run_dir": str(run_dir),
            },
        },
        "sample": _sample_payload(
            case,
            include_golden_patch=include_golden_patch,
            golden_similarity_threshold=golden_similarity_threshold,
        ),
    }


def export_run_to_openai_bundle(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    include_golden_patch: bool = False,
    golden_similarity_threshold: float = 0.8,
) -> OpenAIEvalBundle:
    """Export a completed local harness run into an OpenAI-friendly eval bundle."""
    source_path = run_dir.resolve()
    cases, report, metrics_by_case = _load_run_payload(source_path)
    bundle_dir = (out_dir or _bundle_dir_for_source(source_path)).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _build_run_row(
            source_path,
            case,
            metrics_by_case.get(case.id, {"case_id": case.id, "error": "missing_case_result"}),
            include_golden_patch=include_golden_patch,
            golden_similarity_threshold=golden_similarity_threshold,
        )
        for case in cases
    ]
    items_path = bundle_dir / "items.jsonl"
    schema_path = bundle_dir / "item_schema.json"
    manifest_path = bundle_dir / "bundle_manifest.json"
    _write_jsonl(items_path, rows)
    _write_json(schema_path, _run_item_schema())
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "source_type": "harness_run",
            "source_path": str(source_path),
            "created_at": _utc_now_iso(),
            "case_count": len(cases),
            "items_path": items_path.name,
            "item_schema_path": schema_path.name,
            "sample_schema": _sample_schema(),
            "report_summary": report.get("summary", {}),
            "openai_format": {
                "record_shape": {"item": "object", "sample": "object"},
                "data_source_config": {
                    "type": "custom",
                    "item_schema_path": schema_path.name,
                },
            },
            "notes": [
                "Bundle captures deterministic local harness outcomes plus trace/diff artifact pointers.",
                "This keeps build and test execution local while producing eval-ready JSONL for OpenAI-style external graders.",
            ],
        },
    )
    return OpenAIEvalBundle(
        source_type="harness_run",
        source_path=source_path,
        bundle_dir=bundle_dir,
        items_path=items_path,
        schema_path=schema_path,
        manifest_path=manifest_path,
        case_count=len(cases),
    )


def export_to_openai_bundle(
    source_path: Path,
    *,
    out_dir: Path | None = None,
    include_golden_patch: bool = False,
    golden_similarity_threshold: float = 0.8,
) -> OpenAIEvalBundle:
    """Export a dataset or completed harness run into an OpenAI-friendly bundle."""
    resolved = source_path.resolve()
    if resolved.is_dir():
        return export_run_to_openai_bundle(
            resolved,
            out_dir=out_dir,
            include_golden_patch=include_golden_patch,
            golden_similarity_threshold=golden_similarity_threshold,
        )
    if resolved.is_file():
        return export_dataset_to_openai_bundle(
            resolved,
            out_dir=out_dir,
            include_golden_patch=include_golden_patch,
            golden_similarity_threshold=golden_similarity_threshold,
        )
    raise ValueError(f"Path not found: {resolved}")
