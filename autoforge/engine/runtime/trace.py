"""Trace recorder (trajectory export) for harness-grade replay.

Writes JSONL events to:
  <project_dir>/.autoforge/traces/<run_id>.jsonl

Large payload fields are offloaded into the ArtifactStore and replaced by
artifact references in the trace event payload.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from autoforge.engine.runtime.artifacts import ArtifactStore, ArtifactRef, NullArtifactStore
from autoforge.engine.runtime.fs import compute_file_manifest, diff_manifests
from autoforge.engine.runtime.telemetry import TelemetrySink


def _artifact_ref_dict(ref: ArtifactRef) -> dict[str, Any]:
    return {
        "path": ref.path,
        "bytes": int(ref.bytes),
        "sha256": ref.sha256,
        "media_type": ref.media_type,
    }


def _safe_label(label: str) -> str:
    raw = (label or "").strip()
    raw = raw.replace("\\", "/")
    raw = re.sub(r"[^a-zA-Z0-9._/-]+", "-", raw)
    raw = raw.strip("-") or "snapshot"
    return raw[:120]


class TraceRecorder(TelemetrySink):
    """Telemetry sink that emits replayable JSONL traces + optional durable delegate."""

    schema_version = 1

    def __init__(
        self,
        run_id: str,
        trace_path: Path,
        *,
        delegate: TelemetrySink | None = None,
        artifacts: ArtifactStore | NullArtifactStore | None = None,
        write_header: bool = True,
        capture_llm_content: bool = False,
        capture_command_output: bool = False,
        capture_fs_snapshots: bool = False,
        max_inline_chars: int = 20000,
        redact_secrets: bool = True,
        secrets: list[str] | None = None,
    ) -> None:
        self.run_id = (run_id or "").strip()
        self.trace_path = trace_path
        self.capture_llm_content = bool(capture_llm_content)
        self.capture_command_output = bool(capture_command_output)
        self.capture_fs_snapshots = bool(capture_fs_snapshots)
        self.max_inline_chars = max(2000, int(max_inline_chars or 0))
        self.redact_secrets = bool(redact_secrets)
        self._secrets = [s for s in (secrets or []) if isinstance(s, str) and len(s) >= 8]
        self._delegate = delegate
        self._artifacts: ArtifactStore | NullArtifactStore = artifacts or NullArtifactStore()

        self._lock = threading.RLock()
        self._seq = 0
        self._fh = None

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.trace_path.open("a", encoding="utf-8")
        if write_header:
            self.record(
                "trace_started",
                {"schema_version": int(self.schema_version), "path": str(self.trace_path)},
            )

        self._last_manifest: dict[str, Any] | None = None
        self._last_manifest_label: str = ""

    def close(self) -> None:
        with self._lock:
            fh = self._fh
            self._fh = None
            if fh is None:
                return
            try:
                fh.flush()
            except Exception:
                pass
            try:
                fh.close()
            except Exception:
                pass

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _redact_str(self, s: str) -> str:
        if not self.redact_secrets or not s:
            return s
        out = s
        for secret in self._secrets:
            if secret and secret in out:
                out = out.replace(secret, "<redacted>")
        return out

    def _jsonable(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return self._redact_str(obj) if isinstance(obj, str) else obj
        if isinstance(obj, Path):
            return self._redact_str(str(obj))
        if is_dataclass(obj):
            return self._jsonable(asdict(obj))
        if isinstance(obj, dict):
            return {str(k): self._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._jsonable(v) for v in obj]
        if isinstance(obj, bytes):
            # Never inline arbitrary bytes; store length only.
            return {"bytes_len": int(len(obj))}
        return self._redact_str(str(obj))

    def _maybe_offload(self, key: str, value: Any, *, seq: int) -> Any:
        """Offload large values to artifacts and return a compact reference."""
        try:
            raw_len = len(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            raw_len = len(str(value))
        if raw_len <= self.max_inline_chars:
            return value

        hint = _safe_label(key or "payload").replace("/", "-")
        if isinstance(value, str):
            ref = self._artifacts.write_text(f"trace/offload/{seq:06d}-{hint}.txt", value)
        else:
            ref = self._artifacts.write_json(f"trace/offload/{seq:06d}-{hint}.json", value)
        preview = ""
        if isinstance(value, str):
            preview = value[: min(400, len(value))]
        return {
            "offloaded": True,
            "preview": preview,
            "artifact": _artifact_ref_dict(ref),
        }

    def _write_line(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            fh = self._fh
            if fh is None:
                return
            fh.write(line + "\n")
            fh.flush()

    def record(self, kind: str, payload: Mapping[str, Any]) -> None:
        seq = self._next_seq()
        normalized = self._jsonable(dict(payload))
        compact: dict[str, Any] = {}
        for k, v in normalized.items():
            compact[str(k)] = self._maybe_offload(str(k), v, seq=seq)

        event = {
            "ts": float(time.time()),
            "run_id": self.run_id,
            "seq": int(seq),
            "kind": str(kind),
            "payload": compact,
        }

        # Delegate first (best-effort): payload already compacted (artifact refs).
        if self._delegate is not None:
            try:
                self._delegate.record(str(kind), compact)
            except Exception:
                pass

        try:
            self._write_line(event)
        except Exception:
            return

    def record_llm_call(self, payload: Mapping[str, Any]) -> None:
        self.record("llm_call", payload)

    def record_command(self, payload: Mapping[str, Any]) -> None:
        self.record("command", payload)

    def record_snapshot(self, payload: Mapping[str, Any]) -> None:
        self.record("snapshot", payload)

    def record_fs_snapshot(
        self,
        project_dir: Path,
        *,
        label: str,
        max_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if not self.capture_fs_snapshots:
            return

        manifest = compute_file_manifest(project_dir, max_bytes=max_bytes)
        diff = diff_manifests(self._last_manifest, manifest)
        ref = self._artifacts.write_json(
            f"fs/manifests/{self._next_seq():06d}-{_safe_label(label)}.json",
            manifest,
        )

        self.record(
            "fs_snapshot",
            {
                "label": str(label),
                "project_dir": str(project_dir),
                "file_count": len(manifest.get("files", [])),
                "diff": diff,
                "manifest": {"artifact": _artifact_ref_dict(ref)},
                "prev_label": self._last_manifest_label,
            },
        )
        self._last_manifest = manifest
        self._last_manifest_label = str(label)
