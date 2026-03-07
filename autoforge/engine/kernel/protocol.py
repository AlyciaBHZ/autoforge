from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class KernelThread:
    """Durable top-level run container."""

    run_id: str
    lineage_id: str
    parent_run_id: str
    project_id: str
    profile: str
    surface: str
    created_at: float = field(default_factory=time.time)
    thread_id: str = field(default_factory=lambda: _new_id("thread"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "run_id": self.run_id,
            "lineage_id": self.lineage_id,
            "parent_run_id": self.parent_run_id,
            "project_id": self.project_id,
            "profile": self.profile,
            "surface": self.surface,
            "created_at": float(self.created_at),
            "metadata": dict(self.metadata),
        }


@dataclass
class KernelTurn:
    """One logical unit of kernel work."""

    kind: str
    phase: str
    input_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    turn_id: str = field(default_factory=lambda: _new_id("turn"))
    status: str = "started"
    completed_at: float | None = None
    output_summary: str = ""

    def complete(
        self,
        *,
        status: str = "completed",
        output_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.completed_at = time.time()
        if output_summary:
            self.output_summary = output_summary
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "kind": self.kind,
            "phase": self.phase,
            "status": self.status,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "created_at": float(self.created_at),
            "completed_at": float(self.completed_at) if self.completed_at is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass
class KernelItem:
    """Typed output produced within a turn."""

    turn_id: str
    item_type: str
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    item_id: str = field(default_factory=lambda: _new_id("item"))
    status: str = "completed"
    completed_at: float | None = None

    def complete(self, *, status: str = "completed", payload: dict[str, Any] | None = None) -> None:
        self.status = status
        self.completed_at = time.time()
        if payload:
            self.payload.update(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "turn_id": self.turn_id,
            "item_type": self.item_type,
            "summary": self.summary,
            "payload": dict(self.payload),
            "status": self.status,
            "created_at": float(self.created_at),
            "completed_at": float(self.completed_at) if self.completed_at is not None else None,
        }


@dataclass(frozen=True)
class KernelEvent:
    """Append-only JSONL event entry."""

    seq: int
    kind: str
    payload: dict[str, Any]
    run_id: str
    thread_id: str
    profile: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": int(self.seq),
            "ts": float(self.ts),
            "kind": self.kind,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "profile": self.profile,
            "payload": dict(self.payload),
        }
