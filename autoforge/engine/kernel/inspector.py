from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoforge.engine.kernel.schema import read_kernel_json


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolve_kernel_run_dir(path: Path, run_id: str | None = None) -> Path:
    target = path.resolve()
    if target.is_file():
        if target.name in {
            "manifest.json",
            "events.jsonl",
            "artifact_manifest.json",
            "profile.json",
            "contracts.json",
            "environment.lock.json",
            "execution_plan.json",
            "inbox.json",
        }:
            target = target.parent
        else:
            raise FileNotFoundError(f"Not a kernel run file: {target}")

    if (target / "manifest.json").is_file() and (target / "events.jsonl").is_file():
        return target

    run_root = target / ".autoforge" / "kernel" / "runs"
    if run_root.is_dir():
        candidates = sorted(
            [child for child in run_root.iterdir() if child.is_dir()],
            key=lambda item: item.stat().st_mtime,
        )
        if run_id:
            for child in candidates:
                if child.name == run_id:
                    return child
            raise FileNotFoundError(f"Run id not found under {run_root}: {run_id}")
        if candidates:
            return candidates[-1]

    if target.name == "runs" and target.is_dir():
        candidates = sorted(
            [child for child in target.iterdir() if child.is_dir()],
            key=lambda item: item.stat().st_mtime,
        )
        if run_id:
            for child in candidates:
                if child.name == run_id:
                    return child
            raise FileNotFoundError(f"Run id not found under {target}: {run_id}")
        if candidates:
            return candidates[-1]

    raise FileNotFoundError(f"Could not resolve kernel run dir from {path}")


@dataclass(frozen=True)
class KernelInspection:
    run_dir: Path
    manifest: dict[str, Any]
    plan: dict[str, Any]
    verdict: dict[str, Any]
    harness_judge: dict[str, Any]
    profile: dict[str, Any]
    contracts: dict[str, Any]
    artifact_manifest: dict[str, Any]
    inbox: dict[str, Any]
    event_count: int
    event_kind_counts: dict[str, int]
    last_events: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "manifest": self.manifest,
            "plan": self.plan,
            "verdict": self.verdict,
            "harness_judge": self.harness_judge,
            "profile": self.profile,
            "contracts": self.contracts,
            "artifact_manifest": self.artifact_manifest,
            "inbox": self.inbox,
            "event_count": self.event_count,
            "event_kind_counts": dict(self.event_kind_counts),
            "last_events": list(self.last_events),
        }


def inspect_kernel_run(path: Path, *, run_id: str | None = None, tail: int = 20) -> KernelInspection:
    run_dir = resolve_kernel_run_dir(path, run_id=run_id)
    manifest = read_kernel_json(run_dir / "manifest.json", artifact_type="kernel_manifest")
    plan = (
        read_kernel_json(run_dir / "execution_plan.json", artifact_type="execution_plan")
        if (run_dir / "execution_plan.json").is_file()
        else {}
    )
    verdict = (
        read_kernel_json(run_dir / "contract_verdict.json", artifact_type="contract_verdict")
        if (run_dir / "contract_verdict.json").is_file()
        else {}
    )
    harness_judge = (
        read_kernel_json(run_dir / "harness_judge.json", artifact_type="harness_judge")
        if (run_dir / "harness_judge.json").is_file()
        else {}
    )
    profile = read_kernel_json(run_dir / "profile.json", artifact_type="kernel_profile")
    contracts = read_kernel_json(run_dir / "contracts.json", artifact_type="kernel_contracts")
    artifact_manifest = read_kernel_json(run_dir / "artifact_manifest.json", artifact_type="artifact_manifest")
    inbox = read_kernel_json(run_dir / "inbox.json", artifact_type="kernel_inbox")

    event_kind_counts: dict[str, int] = {}
    last_events: list[dict[str, Any]] = []
    event_count = 0
    events_path = run_dir / "events.jsonl"
    if events_path.is_file():
        lines = events_path.read_text(encoding="utf-8", errors="replace").splitlines()
        event_count = len(lines)
        if tail > 0:
            lines = lines[-tail:]
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(event.get("kind", "unknown"))
            event_kind_counts[kind] = event_kind_counts.get(kind, 0) + 1
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                last_events.append(event)

    return KernelInspection(
        run_dir=run_dir,
        manifest=manifest,
        plan=plan,
        verdict=verdict,
        harness_judge=harness_judge,
        profile=profile,
        contracts=contracts,
        artifact_manifest=artifact_manifest,
        inbox=inbox,
        event_count=event_count,
        event_kind_counts=event_kind_counts,
        last_events=last_events,
    )


def render_kernel_inspection(inspection: KernelInspection) -> str:
    manifest = inspection.manifest
    plan = inspection.plan
    verdict = inspection.verdict
    phase_history = manifest.get("phase_history", [])
    artifact_entries = inspection.artifact_manifest.get("artifacts", [])
    lines = [
        f"run_id: {manifest.get('run_id', '')}",
        f"lineage_id: {manifest.get('lineage_id', '')}",
        f"project_id: {manifest.get('project_id', '')}",
        f"profile: {manifest.get('profile', '')}",
        f"operation: {plan.get('operation', '')}",
        f"status: {manifest.get('status', '')}",
        f"run_dir: {inspection.run_dir}",
        f"current_phase: {plan.get('current_phase', '')}",
        f"events: {inspection.event_count}",
        f"artifacts: {len(artifact_entries)}",
        f"inbox_messages: {len(inspection.inbox.get('messages', []))}",
    ]
    if plan.get("objective"):
        lines.append(f"objective: {plan.get('objective', '')}")
    if verdict:
        lines.append(f"contract_satisfied: {verdict.get('contract_satisfied', False)}")
        lines.append(f"declared_outcomes: {', '.join(verdict.get('declared_outcomes', []))}")
        lines.append(f"inferred_outcomes: {', '.join(verdict.get('inferred_outcomes', []))}")
    if harness_judge := inspection.harness_judge:
        lines.append(
            "harness_judge: "
            f"visible={harness_judge.get('visible_tests_ok')} "
            f"hidden={harness_judge.get('hidden_tests_ok')} "
            f"golden={harness_judge.get('golden_patch_similarity')}"
        )
    if phase_history:
        lines.append("phase_history:")
        for entry in phase_history[-10:]:
            lines.append(
                f"  - {entry.get('phase', '')}: {entry.get('state', '')} {str(entry.get('summary', '')).strip()}".rstrip()
            )
    if inspection.event_kind_counts:
        lines.append("event_kinds:")
        for kind, count in sorted(
            inspection.event_kind_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            lines.append(f"  - {kind}: {count}")
    if inspection.last_events:
        lines.append("last_events:")
        for event in inspection.last_events[-5:]:
            lines.append(
                f"  - #{event.get('seq', '')} {event.get('kind', '')}"
            )
    return "\n".join(lines)
