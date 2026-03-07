from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoforge.engine.kernel.inspector import resolve_kernel_run_dir


@dataclass(frozen=True)
class KernelEventStream:
    """Parsed kernel event stream with summary counters."""

    run_dir: Path
    run_id: str
    event_count: int
    invalid_line_count: int
    first_seq: int | None
    last_seq: int | None
    event_kind_counts: dict[str, int]
    phase_events: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "event_count": int(self.event_count),
            "invalid_line_count": int(self.invalid_line_count),
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "event_kind_counts": dict(self.event_kind_counts),
            "phase_events": list(self.phase_events),
            "events": list(self.events),
        }


def _read_events(events_path: Path) -> tuple[list[dict[str, Any]], int]:
    parsed: list[dict[str, Any]] = []
    invalid_line_count = 0
    for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            invalid_line_count += 1
            continue
        if isinstance(event, dict):
            parsed.append(event)
        else:
            invalid_line_count += 1
    return parsed, invalid_line_count


def load_kernel_event_stream(
    path: Path,
    *,
    run_id: str | None = None,
    tail: int = 20,
) -> KernelEventStream:
    run_dir = resolve_kernel_run_dir(path, run_id=run_id)
    events_path = run_dir / "events.jsonl"
    parsed, invalid_line_count = _read_events(events_path) if events_path.is_file() else ([], 0)

    event_kind_counts: dict[str, int] = {}
    phase_events: list[dict[str, Any]] = []
    for event in parsed:
        kind = str(event.get("kind", "unknown") or "unknown")
        event_kind_counts[kind] = event_kind_counts.get(kind, 0) + 1
        if kind == "phase":
            phase_events.append(event)

    selected = parsed if int(tail) <= 0 else parsed[-int(tail):]
    first_seq = None
    last_seq = None
    if parsed:
        first_seq = int(parsed[0].get("seq", 0) or 0)
        last_seq = int(parsed[-1].get("seq", 0) or 0)

    return KernelEventStream(
        run_dir=run_dir,
        run_id=str(run_dir.name),
        event_count=len(parsed),
        invalid_line_count=invalid_line_count,
        first_seq=first_seq,
        last_seq=last_seq,
        event_kind_counts=event_kind_counts,
        phase_events=tuple(phase_events),
        events=tuple(selected),
    )


def render_kernel_event(event: dict[str, Any]) -> str:
    seq = event.get("seq", "")
    kind = str(event.get("kind", "") or "")
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}

    if kind == "phase":
        phase = str(payload.get("phase", "") or "")
        state = str(payload.get("state", "") or "")
        summary = str(payload.get("summary", "") or "").strip()
        line = f"#{seq} phase {phase} -> {state}"
        return f"{line} ({summary})" if summary else line
    if kind == "checkpoint":
        phase = str(payload.get("phase", "") or "")
        proceed = payload.get("proceed")
        summary = str(payload.get("summary", "") or "").strip()
        return f"#{seq} checkpoint {phase} proceed={proceed} {summary}".rstrip()
    if kind == "artifact_registered":
        artifact_kind = str(payload.get("kind", "") or "")
        path = str(payload.get("path", "") or "")
        return f"#{seq} artifact {artifact_kind} {path}".rstrip()
    if kind in {"turn_started", "turn_completed"}:
        phase = str(payload.get("phase", "") or "")
        status = str(payload.get("status", "") or "")
        turn_id = str(payload.get("turn_id", "") or "")
        line = f"#{seq} {kind} {phase}".rstrip()
        if status:
            line += f" status={status}"
        if turn_id:
            line += f" turn={turn_id}"
        return line
    if kind == "inbox_message":
        text = str(payload.get("text", "") or "").strip()
        if len(text) > 80:
            text = text[:80] + "..."
        return f"#{seq} inbox {text}".rstrip()
    return f"#{seq} {kind}".rstrip()


def render_kernel_event_stream(stream: KernelEventStream) -> str:
    lines = [
        f"run_id: {stream.run_id}",
        f"run_dir: {stream.run_dir}",
        f"events: {stream.event_count}",
        f"invalid_lines: {stream.invalid_line_count}",
    ]
    if stream.first_seq is not None or stream.last_seq is not None:
        lines.append(f"seq_range: {stream.first_seq}..{stream.last_seq}")
    if stream.phase_events:
        lines.append(f"phase_events: {len(stream.phase_events)}")
    if stream.event_kind_counts:
        lines.append("event_kinds:")
        for kind, count in sorted(
            stream.event_kind_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            lines.append(f"  - {kind}: {count}")
    if stream.events:
        lines.append("events_tail:")
        for event in stream.events:
            lines.append(f"  - {render_kernel_event(event)}")
    return "\n".join(lines)
