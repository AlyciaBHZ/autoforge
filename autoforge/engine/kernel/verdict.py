from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoforge.engine.kernel.contracts import KernelProfile


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class ContractVerdict:
    run_id: str
    profile: str
    operation: str
    surface: str
    status: str
    declared_outcomes: tuple[str, ...] = ()
    inferred_outcomes: tuple[str, ...] = ()
    success_outcomes: tuple[str, ...] = ()
    allowed_outcomes: tuple[str, ...] = ()
    present_required_artifacts: tuple[str, ...] = ()
    missing_required_artifacts: tuple[str, ...] = ()
    artifacts_satisfied: bool = False
    allowed_outcome_satisfied: bool = False
    success_outcome_satisfied: bool = False
    contract_satisfied: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "contract_verdict",
            "run_id": self.run_id,
            "profile": self.profile,
            "operation": self.operation,
            "surface": self.surface,
            "status": self.status,
            "declared_outcomes": list(self.declared_outcomes),
            "inferred_outcomes": list(self.inferred_outcomes),
            "success_outcomes": list(self.success_outcomes),
            "allowed_outcomes": list(self.allowed_outcomes),
            "present_required_artifacts": list(self.present_required_artifacts),
            "missing_required_artifacts": list(self.missing_required_artifacts),
            "artifacts_satisfied": bool(self.artifacts_satisfied),
            "allowed_outcome_satisfied": bool(self.allowed_outcome_satisfied),
            "success_outcome_satisfied": bool(self.success_outcome_satisfied),
            "contract_satisfied": bool(self.contract_satisfied),
            "metadata": dict(self.metadata),
            "generated_at": float(self.generated_at),
        }


def write_contract_verdict(path: Path, verdict: ContractVerdict) -> None:
    _atomic_write_json(path, verdict.to_dict())


def _completed_phase_ids(phase_history: list[dict[str, Any]]) -> set[str]:
    completed: set[str] = set()
    for entry in phase_history:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("state", "") or "") == "completed":
            completed.add(str(entry.get("phase", "") or ""))
    return completed


def _required_artifact_verdict(
    profile: KernelProfile,
    artifacts: list[dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required = set(profile.artifact_contract.required_kinds())
    present = {
        str(item.get("kind", "") or "")
        for item in artifacts
        if isinstance(item, dict) and bool(item.get("exists", False))
    }
    present_required = tuple(sorted(required & present))
    missing_required = tuple(sorted(required - present))
    return present_required, missing_required


def _dedupe(seq: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out)


def _infer_outcomes(
    profile: KernelProfile,
    *,
    status: str,
    phase_history: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    if not status:
        return ()
    completed = _completed_phase_ids(phase_history)
    artifact_kinds = {
        str(item.get("kind", "") or "")
        for item in artifacts
        if isinstance(item, dict) and bool(item.get("exists", False))
    }
    meta = dict(metadata or {})
    harness_judge = meta.get("harness_judge", {}) if isinstance(meta.get("harness_judge", {}), dict) else {}
    outcomes: list[str] = []

    if profile.name == "development":
        if "code" in artifact_kinds:
            outcomes.append("repo_runnable")
        visible_ok = harness_judge.get("visible_tests_ok")
        hidden_ok = harness_judge.get("hidden_tests_ok")
        tests_ok = "test" in completed
        if visible_ok is False or hidden_ok is False:
            tests_ok = False
        elif visible_ok is True and (hidden_ok is True or hidden_ok is None):
            tests_ok = True
        if tests_ok:
            outcomes.append("tests_pass")
        if status == "budget_exceeded":
            outcomes.append("partial_delivery")
    elif profile.name == "verification":
        judge_outcome = ""
        for item in artifacts:
            if not isinstance(item, dict):
                continue
            if str(item.get("kind", "") or "") != "verification_judge":
                continue
            metadata = item.get("metadata", {})
            if isinstance(metadata, dict):
                token = str(metadata.get("outcome", "") or "").strip()
                if token:
                    judge_outcome = token
                    break
        if judge_outcome:
            outcomes.append(judge_outcome)
        if status == "completed" and {"verification_report", "obligation_ledger"} <= artifact_kinds:
            if not judge_outcome:
                outcomes.append("bounded_confidence")
    elif profile.name == "research":
        if {"metrics", "evidence_pack"} <= artifact_kinds:
            if status == "completed":
                outcomes.append("partially_reproduced")
            elif status in {"failed", "budget_exceeded"}:
                outcomes.append("failed_with_evidence")

    return _dedupe(outcomes)


def evaluate_contract_verdict(
    *,
    profile: KernelProfile,
    run_id: str,
    operation: str,
    surface: str,
    status: str,
    phase_history: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    declared_outcomes: list[str] | tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ContractVerdict:
    declared = _dedupe(list(declared_outcomes or []))
    inferred = _infer_outcomes(
        profile,
        status=str(status or ""),
        phase_history=list(phase_history),
        artifacts=list(artifacts),
        metadata=dict(metadata or {}),
    )
    combined = _dedupe(list(declared) + list(inferred))
    allowed = tuple(profile.success_contract.allowed_outcomes)
    success = tuple(profile.success_contract.success_outcomes)
    present_required, missing_required = _required_artifact_verdict(profile, artifacts)
    artifacts_satisfied = len(missing_required) == 0
    allowed_outcome_satisfied = any(item in allowed for item in combined)
    success_outcome_satisfied = profile.success_contract.satisfies(combined)
    terminal = str(status or "") not in {"", "planned", "active", "resuming"}

    return ContractVerdict(
        run_id=str(run_id),
        profile=profile.name,
        operation=str(operation or ""),
        surface=str(surface or ""),
        status=str(status or ""),
        declared_outcomes=declared,
        inferred_outcomes=inferred,
        success_outcomes=success,
        allowed_outcomes=allowed,
        present_required_artifacts=present_required,
        missing_required_artifacts=missing_required,
        artifacts_satisfied=artifacts_satisfied,
        allowed_outcome_satisfied=allowed_outcome_satisfied,
        success_outcome_satisfied=success_outcome_satisfied,
        contract_satisfied=bool(
            terminal
            and artifacts_satisfied
            and allowed_outcome_satisfied
            and success_outcome_satisfied
        ),
        metadata=dict(metadata or {}),
    )
