from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoforge.engine.kernel.inspector import resolve_kernel_run_dir
from autoforge.engine.kernel.profiles import resolve_profile
from autoforge.engine.kernel.run_store import KernelRunStore
from autoforge.engine.kernel.schema import read_kernel_json, write_kernel_json
from autoforge.engine.kernel.verdict import evaluate_contract_verdict, write_contract_verdict


def _read_json(path: Path) -> dict[str, Any]:
    return read_kernel_json(path, artifact_type=path.stem)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_kernel_json(path, payload, artifact_type=path.stem)


def _safe_relpath(base: Path, target: Path) -> str:
    try:
        return str(target.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(target)


def _load_declared_outcomes(verdict: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    outcomes = verdict.get("declared_outcomes", [])
    if isinstance(outcomes, list):
        return [str(item) for item in outcomes if str(item).strip()]
    meta = manifest.get("metadata", {})
    if isinstance(meta, dict):
        declared = meta.get("declared_outcomes", [])
        if isinstance(declared, list):
            return [str(item) for item in declared if str(item).strip()]
    return []


def _upsert_artifact(artifact_manifest: dict[str, Any], entry: dict[str, Any]) -> None:
    artifacts = artifact_manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []
    artifacts = [
        item for item in artifacts
        if not (
            isinstance(item, dict)
            and item.get("kind") == entry.get("kind")
            and item.get("absolute_path") == entry.get("absolute_path")
        )
    ]
    artifacts.append(entry)
    artifact_manifest["artifacts"] = artifacts
    artifact_manifest["updated_at"] = float(time.time())


@dataclass(frozen=True)
class HarnessJudgeOverlay:
    run_dir: Path
    judge_path: Path
    verdict_path: Path
    contract_satisfied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "judge_path": str(self.judge_path),
            "verdict_path": str(self.verdict_path),
            "contract_satisfied": bool(self.contract_satisfied),
        }


def apply_harness_judge_overlay(
    path: Path,
    *,
    run_id: str,
    payload: dict[str, Any],
) -> HarnessJudgeOverlay:
    run_dir = resolve_kernel_run_dir(path, run_id=run_id)
    project_dir = run_dir.parent.parent.parent.parent.resolve()
    manifest_path = run_dir / "manifest.json"
    verdict_path = run_dir / "contract_verdict.json"
    artifact_manifest_path = run_dir / "artifact_manifest.json"
    judge_path = run_dir / "harness_judge.json"

    manifest = _read_json(manifest_path)
    artifact_manifest = _read_json(artifact_manifest_path)
    existing_verdict = _read_json(verdict_path)

    judge_payload = dict(payload)
    judge_payload.setdefault("recorded_at", float(time.time()))
    judge_payload.setdefault("run_id", run_id)
    judge_payload.setdefault("project_dir", str(project_dir))
    _write_json(judge_path, judge_payload)

    artifact_entry = {
        "kind": "harness_judge",
        "path": _safe_relpath(project_dir, judge_path),
        "absolute_path": str(judge_path),
        "exists": judge_path.exists(),
        "required": False,
        "description": "Harness judge verdict for this kernel run.",
        "metadata": {
            "visible_tests_ok": judge_payload.get("visible_tests_ok"),
            "hidden_tests_ok": judge_payload.get("hidden_tests_ok"),
            "golden_patch_similarity": judge_payload.get("golden_patch_similarity"),
        },
        "recorded_at": float(time.time()),
    }
    _upsert_artifact(artifact_manifest, artifact_entry)
    _write_json(artifact_manifest_path, artifact_manifest)

    profile = resolve_profile(str(manifest.get("profile", "") or "development"))
    phase_history = manifest.get("phase_history", [])
    if not isinstance(phase_history, list):
        phase_history = []
    artifacts = artifact_manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []

    declared_outcomes = _load_declared_outcomes(existing_verdict, manifest)
    if judge_payload.get("visible_tests_ok") is True and judge_payload.get("hidden_tests_ok") in {True, None}:
        if "tests_pass" not in declared_outcomes:
            declared_outcomes.append("tests_pass")

    verdict = evaluate_contract_verdict(
        profile=profile,
        run_id=str(manifest.get("run_id", run_id) or run_id),
        operation=str(manifest.get("operation", "") or ""),
        surface=str(manifest.get("surface", "") or ""),
        status=str(manifest.get("status", "") or "completed"),
        phase_history=phase_history,
        artifacts=artifacts,
        declared_outcomes=declared_outcomes,
        metadata={
            "lineage_id": str(manifest.get("lineage_id", "") or ""),
            "parent_run_id": str(manifest.get("parent_run_id", "") or ""),
            "project_id": str(manifest.get("project_id", "") or ""),
            "harness_judge": judge_payload,
        },
    )
    write_contract_verdict(verdict_path, verdict)

    manifest["contract_verdict"] = str(verdict_path)
    manifest["contract_satisfied"] = bool(verdict.contract_satisfied)
    _write_json(manifest_path, manifest)

    store = KernelRunStore(project_dir / ".autoforge" / "kernel" / "run_store.sqlite3")
    try:
        existing = store.fetch_run(str(manifest.get("run_id", run_id) or run_id))
        metadata = dict(existing.metadata) if existing is not None else {}
        metadata["harness_judge"] = judge_payload
        store.sync_run(
            run_id=str(manifest.get("run_id", run_id) or run_id),
            lineage_id=str(manifest.get("lineage_id", "") or (existing.lineage_id if existing else run_id)),
            parent_run_id=str(manifest.get("parent_run_id", "") or (existing.parent_run_id if existing else "")),
            project_id=str(manifest.get("project_id", "") or (existing.project_id if existing else "")),
            thread_id=str(manifest.get("thread_id", "") or (existing.thread_id if existing else "")),
            profile=str(manifest.get("profile", "") or (existing.profile if existing else profile.name)),
            operation=str(manifest.get("operation", "") or (existing.operation if existing else "")),
            surface=str(manifest.get("surface", "") or (existing.surface if existing else "")),
            status=str(manifest.get("status", "") or (existing.status if existing else "completed")),
            project_dir=str(manifest.get("project_dir", "") or (existing.project_dir if existing else project_dir)),
            current_phase=(existing.current_phase if existing is not None else ""),
            objective=(existing.objective if existing is not None else ""),
            summary=(existing.summary if existing is not None else ""),
            metadata=metadata,
            verdict=verdict.to_dict(),
            completed_at=(existing.completed_at if existing is not None else time.time()),
        )
        store.record_artifact(run_id=str(manifest.get("run_id", run_id) or run_id), artifact=artifact_entry)
    finally:
        store.close()

    return HarnessJudgeOverlay(
        run_dir=run_dir,
        judge_path=judge_path,
        verdict_path=verdict_path,
        contract_satisfied=bool(verdict.contract_satisfied),
    )
