from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


@dataclass(frozen=True)
class KernelRunRecord:
    run_id: str
    lineage_id: str
    parent_run_id: str
    project_id: str
    thread_id: str
    profile: str
    operation: str
    surface: str
    status: str
    project_dir: str
    current_phase: str
    objective: str
    summary: str
    metadata: dict[str, Any]
    verdict: dict[str, Any]
    started_at: float
    updated_at: float
    completed_at: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "lineage_id": self.lineage_id,
            "parent_run_id": self.parent_run_id,
            "project_id": self.project_id,
            "thread_id": self.thread_id,
            "profile": self.profile,
            "operation": self.operation,
            "surface": self.surface,
            "status": self.status,
            "project_dir": self.project_dir,
            "current_phase": self.current_phase,
            "objective": self.objective,
            "summary": self.summary,
            "metadata": dict(self.metadata),
            "verdict": dict(self.verdict),
            "started_at": float(self.started_at),
            "updated_at": float(self.updated_at),
            "completed_at": float(self.completed_at) if self.completed_at is not None else None,
        }


class KernelRunStore:
    """SQLite-backed index for repo-local kernel runs."""

    schema_version = 1

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def _open(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              lineage_id TEXT NOT NULL,
              parent_run_id TEXT NOT NULL DEFAULT '',
              project_id TEXT NOT NULL DEFAULT '',
              thread_id TEXT NOT NULL,
              profile TEXT NOT NULL,
              operation TEXT NOT NULL,
              surface TEXT NOT NULL,
              status TEXT NOT NULL,
              project_dir TEXT NOT NULL,
              current_phase TEXT NOT NULL DEFAULT '',
              objective TEXT NOT NULL DEFAULT '',
              summary TEXT NOT NULL DEFAULT '',
              metadata_json TEXT NOT NULL DEFAULT '{}',
              verdict_json TEXT NOT NULL DEFAULT '{}',
              started_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              completed_at REAL
            );

            CREATE TABLE IF NOT EXISTS phase_history (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              phase TEXT NOT NULL,
              state TEXT NOT NULL,
              summary TEXT NOT NULL DEFAULT '',
              metadata_json TEXT NOT NULL DEFAULT '{}',
              ts REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS artifacts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              kind TEXT NOT NULL,
              path TEXT NOT NULL,
              absolute_path TEXT NOT NULL,
              exists_flag INTEGER NOT NULL,
              required_flag INTEGER,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              recorded_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_runs_lineage ON runs(lineage_id, started_at);
            CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id, started_at);
            CREATE INDEX IF NOT EXISTS idx_phase_history_run ON phase_history(run_id, ts);
            CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id, recorded_at);
            """
        )
        conn.commit()
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                self._conn = self._open()
            return self._conn

    def sync_run(
        self,
        *,
        run_id: str,
        lineage_id: str,
        parent_run_id: str,
        project_id: str,
        thread_id: str,
        profile: str,
        operation: str,
        surface: str,
        status: str,
        project_dir: str,
        current_phase: str = "",
        objective: str = "",
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        verdict: dict[str, Any] | None = None,
        completed_at: float | None = None,
    ) -> None:
        conn = self._get_conn()
        now = float(time.time())
        row = conn.execute(
            "SELECT started_at FROM runs WHERE run_id = ?;",
            (str(run_id),),
        ).fetchone()
        started_at = float(row["started_at"]) if row is not None else now
        conn.execute(
            """
            INSERT INTO runs(
              run_id, lineage_id, parent_run_id, project_id, thread_id, profile,
              operation, surface, status, project_dir, current_phase, objective,
              summary, metadata_json, verdict_json, started_at, updated_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
              lineage_id=excluded.lineage_id,
              parent_run_id=excluded.parent_run_id,
              project_id=excluded.project_id,
              thread_id=excluded.thread_id,
              profile=excluded.profile,
              operation=excluded.operation,
              surface=excluded.surface,
              status=excluded.status,
              project_dir=excluded.project_dir,
              current_phase=excluded.current_phase,
              objective=excluded.objective,
              summary=excluded.summary,
              metadata_json=excluded.metadata_json,
              verdict_json=excluded.verdict_json,
              updated_at=excluded.updated_at,
              completed_at=excluded.completed_at;
            """,
            (
                str(run_id),
                str(lineage_id),
                str(parent_run_id or ""),
                str(project_id or ""),
                str(thread_id),
                str(profile),
                str(operation),
                str(surface),
                str(status),
                str(project_dir),
                str(current_phase or ""),
                str(objective or ""),
                str(summary or ""),
                _json_dumps(dict(metadata or {})),
                _json_dumps(dict(verdict or {})),
                started_at,
                now,
                float(completed_at) if completed_at is not None else None,
            ),
        )
        conn.commit()

    def record_phase(
        self,
        *,
        run_id: str,
        phase: str,
        state: str,
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO phase_history(run_id, phase, state, summary, metadata_json, ts)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                str(run_id),
                str(phase),
                str(state),
                str(summary or ""),
                _json_dumps(dict(metadata or {})),
                float(ts if ts is not None else time.time()),
            ),
        )
        conn.commit()

    def record_artifact(self, *, run_id: str, artifact: dict[str, Any]) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO artifacts(run_id, kind, path, absolute_path, exists_flag, required_flag, metadata_json, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                str(run_id),
                str(artifact.get("kind", "") or ""),
                str(artifact.get("path", "") or ""),
                str(artifact.get("absolute_path", "") or ""),
                1 if bool(artifact.get("exists", False)) else 0,
                None if artifact.get("required") is None else (1 if bool(artifact.get("required")) else 0),
                _json_dumps(dict(artifact.get("metadata", {}) or {})),
                float(artifact.get("recorded_at", time.time()) or time.time()),
            ),
        )
        conn.commit()

    def fetch_run(self, run_id: str) -> KernelRunRecord | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (str(run_id),)).fetchone()
        if row is None:
            return None
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        verdict = json.loads(row["verdict_json"]) if row["verdict_json"] else {}
        return KernelRunRecord(
            run_id=str(row["run_id"]),
            lineage_id=str(row["lineage_id"]),
            parent_run_id=str(row["parent_run_id"] or ""),
            project_id=str(row["project_id"] or ""),
            thread_id=str(row["thread_id"]),
            profile=str(row["profile"]),
            operation=str(row["operation"]),
            surface=str(row["surface"]),
            status=str(row["status"]),
            project_dir=str(row["project_dir"]),
            current_phase=str(row["current_phase"] or ""),
            objective=str(row["objective"] or ""),
            summary=str(row["summary"] or ""),
            metadata=metadata if isinstance(metadata, dict) else {},
            verdict=verdict if isinstance(verdict, dict) else {},
            started_at=float(row["started_at"]),
            updated_at=float(row["updated_at"]),
            completed_at=float(row["completed_at"]) if row["completed_at"] is not None else None,
        )
