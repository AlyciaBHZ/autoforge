from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoforge.engine.kernel.contracts import KernelProfile
from autoforge.engine.kernel.schema import KERNEL_SCHEMA_VERSION, write_kernel_json


def _utc_ts() -> float:
    return time.time()


@dataclass
class ExecutionPlanArtifact:
    """Repository-local execution plan for one kernel run."""

    schema_version: int
    run_id: str
    lineage_id: str
    parent_run_id: str
    project_id: str
    thread_id: str
    profile: str
    operation: str
    surface: str
    objective: str
    summary: str
    phase_graph: dict[str, Any]
    success_contract: dict[str, Any]
    artifact_contract: dict[str, Any]
    constraints: dict[str, Any]
    inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[str] = field(default_factory=list)
    phase_states: dict[str, str] = field(default_factory=dict)
    current_phase: str = ""
    status: str = "planned"
    created_at: float = field(default_factory=_utc_ts)
    updated_at: float = field(default_factory=_utc_ts)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        lineage_id: str,
        parent_run_id: str,
        project_id: str,
        thread_id: str,
        profile: KernelProfile,
        operation: str,
        surface: str,
        objective: str,
        summary: str,
        constraints: dict[str, Any],
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        checkpoints: list[str] | None = None,
    ) -> "ExecutionPlanArtifact":
        phase_states = {phase: "pending" for phase in profile.phase_graph.phase_ids()}
        start_phase = profile.phase_graph.start
        phase_states[start_phase] = "ready"
        return cls(
            schema_version=KERNEL_SCHEMA_VERSION,
            run_id=run_id,
            lineage_id=str(lineage_id),
            parent_run_id=str(parent_run_id),
            project_id=str(project_id),
            thread_id=thread_id,
            profile=profile.name,
            operation=str(operation),
            surface=str(surface),
            objective=str(objective),
            summary=str(summary),
            phase_graph=profile.phase_graph.to_dict(),
            success_contract=profile.success_contract.to_dict(),
            artifact_contract=profile.artifact_contract.to_dict(),
            constraints=dict(constraints),
            inputs=dict(inputs or {}),
            metadata=dict(metadata or {}),
            checkpoints=list(checkpoints or []),
            phase_states=phase_states,
            current_phase=start_phase,
        )

    def update(
        self,
        *,
        objective: str | None = None,
        summary: str | None = None,
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
        current_phase: str | None = None,
    ) -> None:
        if objective is not None:
            self.objective = str(objective)
        if summary is not None:
            self.summary = str(summary)
        if inputs:
            self.inputs.update(dict(inputs))
        if metadata:
            self.metadata.update(dict(metadata))
        if status is not None:
            self.status = str(status)
        if current_phase is not None:
            self.current_phase = str(current_phase)
        self.updated_at = _utc_ts()

    def mark_phase(self, phase: str, state: str) -> None:
        self.phase_states[str(phase)] = str(state)
        self.current_phase = str(phase)
        self.updated_at = _utc_ts()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_type": "execution_plan",
            "run_id": self.run_id,
            "lineage_id": self.lineage_id,
            "parent_run_id": self.parent_run_id,
            "project_id": self.project_id,
            "thread_id": self.thread_id,
            "profile": self.profile,
            "operation": self.operation,
            "surface": self.surface,
            "objective": self.objective,
            "summary": self.summary,
            "phase_graph": dict(self.phase_graph),
            "success_contract": dict(self.success_contract),
            "artifact_contract": dict(self.artifact_contract),
            "constraints": dict(self.constraints),
            "inputs": dict(self.inputs),
            "metadata": dict(self.metadata),
            "checkpoints": list(self.checkpoints),
            "phase_states": dict(self.phase_states),
            "current_phase": self.current_phase,
            "status": self.status,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }


def write_execution_plan(path: Path, plan: ExecutionPlanArtifact) -> None:
    write_kernel_json(path, plan.to_dict(), artifact_type="execution_plan")
