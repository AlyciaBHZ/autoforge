"""Durable execution journal (SQLite) + snapshot persistence.

Design goals:
  - append-only events for post-mortem + replay
  - snapshots for fast resume
  - safe by default: best-effort, never crash the pipeline due to telemetry
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from autoforge.engine.runtime.telemetry import TelemetrySink


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


@dataclass(frozen=True)
class SnapshotRecord:
    run_id: str
    state_version: int
    ts: float
    phase: str
    payload: dict[str, Any]


class RunJournal(TelemetrySink):
    """SQLite-backed journal for a project directory."""

    def __init__(self, project_dir: Path, *, run_id: str | None = None) -> None:
        self.project_dir = project_dir
        self._forge_dir = project_dir / ".autoforge"
        self.db_path = self._forge_dir / "run_journal.sqlite3"
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._run_id: str = (run_id or "").strip()

    @property
    def run_id(self) -> str:
        return self._run_id

    def set_run_id(self, run_id: str) -> None:
        self._run_id = (run_id or "").strip()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def _open(self) -> sqlite3.Connection:
        self._forge_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              ts REAL NOT NULL,
              kind TEXT NOT NULL,
              payload_json TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, ts);")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              state_version INTEGER NOT NULL,
              ts REAL NOT NULL,
              phase TEXT NOT NULL,
              payload_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshots_run_version "
            "ON snapshots(run_id, state_version);"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_run_ts ON snapshots(run_id, ts);")
        conn.commit()
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                self._conn = self._open()
            return self._conn

    def append_event(self, kind: str, payload: Mapping[str, Any]) -> None:
        run_id = self._run_id
        if not run_id:
            # Do not create orphan events without a run_id; caller should set run_id
            return
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO events(run_id, ts, kind, payload_json) VALUES (?, ?, ?, ?);",
                (run_id, float(time.time()), str(kind), json.dumps(dict(payload), ensure_ascii=False)),
            )
            conn.commit()
        except Exception:
            # Best-effort; telemetry must never crash the pipeline.
            return

    def save_snapshot(
        self,
        payload: Mapping[str, Any],
        *,
        phase: str,
        state_version: int,
        json_path: Path | None = None,
    ) -> None:
        run_id = self._run_id
        if not run_id:
            return
        ts = float(time.time())
        payload_dict = dict(payload)
        payload_json = json.dumps(payload_dict, indent=2, ensure_ascii=False)
        try:
            if json_path is not None:
                _atomic_write_text(json_path, payload_json, encoding="utf-8")
        except Exception:
            # Snapshot file write is best-effort; durable store still matters.
            pass

        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO snapshots(run_id, state_version, ts, phase, payload_json) "
                "VALUES (?, ?, ?, ?, ?);",
                (run_id, int(state_version), ts, str(phase), payload_json),
            )
            conn.commit()
        except Exception:
            return

    def load_latest_snapshot(self, run_id: str | None = None) -> SnapshotRecord | None:
        """Load the latest snapshot for a run_id. If run_id is None, loads latest across runs."""
        try:
            conn = self._get_conn()
            if run_id:
                row = conn.execute(
                    "SELECT run_id, state_version, ts, phase, payload_json "
                    "FROM snapshots WHERE run_id=? ORDER BY state_version DESC LIMIT 1;",
                    (str(run_id),),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT run_id, state_version, ts, phase, payload_json "
                    "FROM snapshots ORDER BY ts DESC LIMIT 1;"
                ).fetchone()
            if not row:
                return None
            rid, version, ts, phase, payload_json = row
            payload = json.loads(payload_json) if payload_json else {}
            if not isinstance(payload, dict):
                payload = {}
            return SnapshotRecord(
                run_id=str(rid),
                state_version=int(version),
                ts=float(ts),
                phase=str(phase),
                payload=payload,
            )
        except Exception:
            return None

    # ── TelemetrySink ──────────────────────────────────────────────────────

    def record(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.append_event(kind, payload)

    def record_llm_call(self, payload: Mapping[str, Any]) -> None:
        self.append_event("llm_call", payload)

    def record_command(self, payload: Mapping[str, Any]) -> None:
        self.append_event("command", payload)

    def record_snapshot(self, payload: Mapping[str, Any]) -> None:
        self.append_event("snapshot", payload)

