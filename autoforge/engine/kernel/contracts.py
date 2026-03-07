from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PhaseNode:
    """A named node in a kernel phase graph."""

    id: str
    description: str = ""
    terminal: bool = False
    handler: str = ""
    resume_markers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "terminal": bool(self.terminal),
            "handler": self.handler,
            "resume_markers": list(self.resume_markers),
        }


@dataclass(frozen=True)
class PhaseEdge:
    """A directed edge between phases."""

    source: str
    target: str
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
        }


@dataclass(frozen=True)
class PhaseGraph:
    """Declarative phase graph for a kernel profile."""

    start: str
    phases: tuple[PhaseNode, ...]
    edges: tuple[PhaseEdge, ...]

    def __post_init__(self) -> None:
        phase_ids = [phase.id for phase in self.phases]
        if not phase_ids:
            raise ValueError("PhaseGraph requires at least one phase")
        if len(set(phase_ids)) != len(phase_ids):
            raise ValueError("PhaseGraph phases must be unique")
        if self.start not in phase_ids:
            raise ValueError(f"PhaseGraph start={self.start!r} is not a known phase")
        for edge in self.edges:
            if edge.source not in phase_ids or edge.target not in phase_ids:
                raise ValueError(
                    f"Invalid edge {edge.source!r}->{edge.target!r}; phase not found"
                )
        self._ensure_acyclic()

    def _ensure_acyclic(self) -> None:
        adjacency: dict[str, list[str]] = {phase.id: [] for phase in self.phases}
        for edge in self.edges:
            adjacency[edge.source].append(edge.target)

        visited: set[str] = set()
        active: set[str] = set()

        def _dfs(node: str) -> None:
            if node in active:
                raise ValueError("PhaseGraph must be acyclic")
            if node in visited:
                return
            active.add(node)
            for child in adjacency.get(node, []):
                _dfs(child)
            active.remove(node)
            visited.add(node)

        _dfs(self.start)

    def phase_ids(self) -> tuple[str, ...]:
        return tuple(phase.id for phase in self.phases)

    def successors(self, phase_id: str) -> tuple[str, ...]:
        return tuple(edge.target for edge in self.edges if edge.source == phase_id)

    def contains(self, phase_id: str) -> bool:
        return phase_id in self.phase_ids()

    def phase_for_resume_marker(self, marker: str) -> str | None:
        token = str(marker or "").strip().lower()
        if not token:
            return None
        for phase in self.phases:
            markers = {str(item).strip().lower() for item in phase.resume_markers if str(item).strip()}
            if token in markers:
                return phase.id
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "phases": [phase.to_dict() for phase in self.phases],
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass(frozen=True)
class SuccessContract:
    """Declares which outcomes satisfy a profile."""

    description: str
    success_outcomes: tuple[str, ...]
    allowed_outcomes: tuple[str, ...]
    satisfaction_mode: str = "any"

    def __post_init__(self) -> None:
        if not self.allowed_outcomes:
            raise ValueError("SuccessContract requires allowed outcomes")
        missing = set(self.success_outcomes) - set(self.allowed_outcomes)
        if missing:
            raise ValueError(
                f"Success outcomes must be included in allowed outcomes: {sorted(missing)}"
            )
        if self.satisfaction_mode not in {"any", "all"}:
            raise ValueError("SuccessContract satisfaction_mode must be 'any' or 'all'")

    def is_success(self, outcome: str) -> bool:
        return str(outcome).strip() in self.success_outcomes

    def satisfies(self, outcomes: tuple[str, ...]) -> bool:
        observed = {str(item).strip() for item in outcomes if str(item).strip()}
        if self.satisfaction_mode == "all":
            return set(self.success_outcomes).issubset(observed)
        return any(item in observed for item in self.success_outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "success_outcomes": list(self.success_outcomes),
            "allowed_outcomes": list(self.allowed_outcomes),
            "satisfaction_mode": self.satisfaction_mode,
        }


@dataclass(frozen=True)
class ArtifactSpec:
    """Contract entry for a single artifact kind."""

    kind: str
    description: str
    required: bool = True
    path_hint: str = ""
    media_type: str = "application/octet-stream"
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "required": bool(self.required),
            "path_hint": self.path_hint,
            "media_type": self.media_type,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ArtifactContract:
    """Declares which artifacts a profile is expected to produce."""

    artifacts: tuple[ArtifactSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        kinds = [artifact.kind for artifact in self.artifacts]
        if len(set(kinds)) != len(kinds):
            raise ValueError("ArtifactContract kinds must be unique")

    def required_kinds(self) -> tuple[str, ...]:
        return tuple(a.kind for a in self.artifacts if a.required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "required_kinds": list(self.required_kinds()),
        }


@dataclass(frozen=True)
class KernelProfile:
    """First-class kernel contract for one operating profile."""

    name: str
    summary: str
    phase_graph: PhaseGraph
    success_contract: SuccessContract
    artifact_contract: ArtifactContract
    tags: tuple[str, ...] = ()
    phase_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def normalize_phase(self, phase_id: str) -> str:
        phase = str(phase_id).strip()
        if self.phase_graph.contains(phase):
            return phase
        for canonical, aliases in self.phase_aliases.items():
            if phase == canonical or phase in aliases:
                return canonical
        return phase

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "tags": list(self.tags),
            "phase_graph": self.phase_graph.to_dict(),
            "success_contract": self.success_contract.to_dict(),
            "artifact_contract": self.artifact_contract.to_dict(),
            "phase_aliases": {k: list(v) for k, v in self.phase_aliases.items()},
        }
