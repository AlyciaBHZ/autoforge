"""Telemetry interfaces for the engine.

Telemetry is intentionally lightweight:
  - no external dependencies
  - safe to call from any module (best-effort, never crash the pipeline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class TelemetryRecord:
    ts: float
    run_id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class TelemetrySink(Protocol):
    """Best-effort telemetry sink used across LLM/sandbox/orchestrator."""

    def record(self, kind: str, payload: Mapping[str, Any]) -> None:
        ...

    def record_llm_call(self, payload: Mapping[str, Any]) -> None:
        ...

    def record_command(self, payload: Mapping[str, Any]) -> None:
        ...

    def record_snapshot(self, payload: Mapping[str, Any]) -> None:
        ...


class NoopTelemetry:
    def record(self, kind: str, payload: Mapping[str, Any]) -> None:
        return

    def record_llm_call(self, payload: Mapping[str, Any]) -> None:
        return

    def record_command(self, payload: Mapping[str, Any]) -> None:
        return

    def record_snapshot(self, payload: Mapping[str, Any]) -> None:
        return

