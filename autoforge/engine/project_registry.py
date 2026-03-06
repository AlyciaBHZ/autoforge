"""Project Registry — SQLite-backed multi-project management.

Tracks all projects across daemon runs: queued, building, completed, failed.
Each project has its own budget, workspace, and status.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
import json
from enum import Enum
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class ProjectStatus(str, Enum):
    QUEUED = "queued"
    BUILDING = "building"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStage(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    ARTIFACT_GENERATED = "artifact_generated"
    COMPILED = "compiled"
    TESTED = "tested"
    RUNTIME_VERIFIED = "runtime_verified"
    FORMALIZED = "formalized"
    VERIFIED_WITH_SORRY = "verified_with_sorry"
    PROVED = "proved"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunRecord:
    run_id: str
    project_id: str
    stage: RunStage
    created_at: str


@dataclass
class TaskRecord:
    id: str
    run_id: str
    task_id: str
    status: str
    retry_count: int
    updated_at: str


@dataclass
class TaskLease:
    task_id: str
    holder: str
    lease_until: str


@dataclass
class ArtifactRecord:
    id: str
    run_id: str
    kind: str
    path: str
    created_at: str


@dataclass
class ExecutionEvent:
    id: str
    run_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: str


@dataclass
class VerificationRecord:
    id: str
    run_id: str
    verification_type: str
    status: str
    details: str
    created_at: str


@dataclass
class Project:
    """A registered project."""

    id: str
    name: str
    description: str
    status: ProjectStatus
    phase: str
    workspace_path: str
    requested_by: str  # "telegram:<user_id>" or "webhook:<ip>" or "cli"
    budget_usd: float
    cost_usd: float
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None
    idempotency_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "phase": self.phase,
            "workspace_path": self.workspace_path,
            "requested_by": self.requested_by,
            "budget_usd": self.budget_usd,
            "cost_usd": self.cost_usd,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "idempotency_key": self.idempotency_key,
        }


@dataclass
class ProjectMessage:
    """A per-project user message/instruction (async interference)."""

    id: int
    project_id: str
    ts: float
    kind: str
    source: str
    text: str
    handled_at: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "project_id": self.project_id,
            "ts": float(self.ts),
            "kind": self.kind,
            "source": self.source,
            "text": self.text,
            "handled_at": float(self.handled_at) if self.handled_at is not None else None,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    phase TEXT NOT NULL DEFAULT '',
    workspace_path TEXT NOT NULL DEFAULT '',
    requested_by TEXT NOT NULL DEFAULT 'cli',
    budget_usd REAL NOT NULL DEFAULT 10.0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error TEXT,
    idempotency_key TEXT
);

CREATE TABLE IF NOT EXISTS project_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    ts REAL NOT NULL,
    kind TEXT NOT NULL DEFAULT 'note',
    source TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    handled_at REAL
);

CREATE TABLE IF NOT EXISTS run_records (
    run_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_records (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verification_records (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    verification_type TEXT NOT NULL,
    status TEXT NOT NULL,
    details TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_leases (
    task_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    lease_until TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS request_nonces (
    requester TEXT NOT NULL,
    nonce TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (requester, nonce)
);
"""


class ProjectRegistry:
    """SQLite-backed project registry for multi-project management."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    def _ensure_db(self) -> aiosqlite.Connection:
        """Return db connection or raise RuntimeError."""
        if self._db is None:
            raise RuntimeError("ProjectRegistry not opened. Call open() or use 'async with'.")
        return self._db

    async def open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        try:
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.executescript(_SCHEMA)
            await self._migrate_schema()
            await self._ensure_indexes()
            await self._db.commit()
        except Exception:
            await self._db.close()
            self._db = None
            raise

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> ProjectRegistry:
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _row_to_project(self, row: aiosqlite.Row) -> Project:
        return Project(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            status=ProjectStatus(row["status"]),
            phase=row["phase"],
            workspace_path=row["workspace_path"],
            requested_by=row["requested_by"],
            budget_usd=row["budget_usd"],
            cost_usd=row["cost_usd"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error=row["error"],
            idempotency_key=row["idempotency_key"] if "idempotency_key" in row.keys() else None,
        )

    def _row_to_message(self, row: aiosqlite.Row) -> ProjectMessage:
        return ProjectMessage(
            id=int(row["id"]),
            project_id=str(row["project_id"]),
            ts=float(row["ts"]),
            kind=str(row["kind"]),
            source=str(row["source"]),
            text=str(row["text"]),
            handled_at=float(row["handled_at"]) if row["handled_at"] is not None else None,
        )

    async def _migrate_schema(self) -> None:
        """Apply additive migrations for older SQLite files."""
        db = self._ensure_db()
        cursor = await db.execute("PRAGMA table_info(projects)")
        rows = await cursor.fetchall()
        columns = {row["name"] for row in rows}
        if "idempotency_key" not in columns:
            await db.execute("ALTER TABLE projects ADD COLUMN idempotency_key TEXT")

    async def _ensure_indexes(self) -> None:
        db = self._ensure_db()
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_projects_status_created ON projects(status, created_at)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_projects_requester_created ON projects(requested_by, created_at DESC)"
        )
        await db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_requester_idempotency "
            "ON projects(requested_by, idempotency_key)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_project_messages_project_id "
            "ON project_messages(project_id, id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_project_messages_unhandled "
            "ON project_messages(project_id, handled_at, id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run_created ON execution_events(run_id, created_at)"
        )
        
    async def enqueue_run(
        self,
        *,
        description: str,
        requested_by: str = "cli",
        budget_usd: float = 10.0,
        idempotency_key: str | None = None,
    ) -> Project:
        """Unified durable entrypoint for all channels."""
        project = await self.enqueue(
            description=description,
            requested_by=requested_by,
            budget_usd=budget_usd,
            idempotency_key=idempotency_key,
        )
        run_id = f"run_{project.id}"
        now = datetime.now(timezone.utc).isoformat()
        db = self._ensure_db()
        await db.execute(
            "INSERT OR IGNORE INTO run_records (run_id, project_id, stage, created_at) VALUES (?, ?, ?, ?)",
            (run_id, project.id, RunStage.PLANNED.value, now),
        )
        await self.append_event(run_id, "RunEnqueued", {"project_id": project.id, "requested_by": requested_by})
        return project

    async def set_run_stage(self, run_id: str, stage: RunStage) -> None:
        db = self._ensure_db()
        await db.execute("UPDATE run_records SET stage = ? WHERE run_id = ?", (stage.value, run_id))
        await db.commit()

    async def append_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        db = self._ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        event_id = uuid.uuid4().hex
        await db.execute(
            "INSERT INTO execution_events (id, run_id, event_type, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (event_id, run_id, event_type, json.dumps(payload, ensure_ascii=False), now),
        )
        await db.commit()

    async def acquire_task_lease(self, task_id: str, holder: str, ttl_seconds: int = 60) -> bool:
        db = self._ensure_db()
        now_dt = datetime.now(timezone.utc)
        lease_until = (now_dt.timestamp() + ttl_seconds)
        now = now_dt.isoformat()
        until_iso = datetime.fromtimestamp(lease_until, timezone.utc).isoformat()
        await db.execute(
            "DELETE FROM task_leases WHERE task_id = ? AND lease_until < ?",
            (task_id, now),
        )
        cur = await db.execute(
            "INSERT OR IGNORE INTO task_leases (task_id, holder, lease_until, heartbeat_at) VALUES (?, ?, ?, ?)",
            (task_id, holder, until_iso, now),
        )
        await db.commit()
        return cur.rowcount > 0

    async def heartbeat_task_lease(self, task_id: str, holder: str, ttl_seconds: int = 60) -> bool:
        db = self._ensure_db()
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        until_iso = datetime.fromtimestamp(now_dt.timestamp() + ttl_seconds, timezone.utc).isoformat()
        cur = await db.execute(
            "UPDATE task_leases SET lease_until = ?, heartbeat_at = ? WHERE task_id = ? AND holder = ?",
            (until_iso, now, task_id, holder),
        )
        await db.commit()
        return cur.rowcount > 0

    async def release_task_lease(self, task_id: str, holder: str) -> None:
        db = self._ensure_db()
        await db.execute("DELETE FROM task_leases WHERE task_id = ? AND holder = ?", (task_id, holder))
        await db.commit()

    async def requeue_stale_builds(self, max_age_seconds: int = 600) -> int:
        """Move stale BUILDING projects back to QUEUED for crash recovery."""
        db = self._ensure_db()
        threshold = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - max_age_seconds, timezone.utc).isoformat()
        cur = await db.execute(
            "UPDATE projects SET status = ?, phase = 'requeued' WHERE status = ? AND started_at IS NOT NULL AND started_at < ?",
            (ProjectStatus.QUEUED.value, ProjectStatus.BUILDING.value, threshold),
        )
        await db.commit()
        return cur.rowcount


    async def register_request_nonce(self, requester: str, nonce: str) -> bool:
        """Register nonce for replay protection; False if already seen."""
        db = self._ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        cur = await db.execute(
            "INSERT OR IGNORE INTO request_nonces (requester, nonce, created_at) VALUES (?, ?, ?)",
            (requester, nonce, now),
        )
        await db.commit()
        return cur.rowcount > 0

    async def purge_old_nonces(self, older_than_seconds: int = 3600) -> int:
        db = self._ensure_db()
        threshold = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - older_than_seconds, timezone.utc).isoformat()
        cur = await db.execute("DELETE FROM request_nonces WHERE created_at < ?", (threshold,))
        await db.commit()
        return cur.rowcount

    async def enqueue(
        self,
        description: str,
        requested_by: str = "cli",
        budget_usd: float = 10.0,
        idempotency_key: str | None = None,
    ) -> Project:
        """Add a new project to the build queue."""
        # Validate inputs
        description = description.strip()
        if not description:
            raise ValueError("Project description cannot be empty")
        if len(description) > 10000:
            raise ValueError("Project description too long (max 10,000 characters)")
        if budget_usd <= 0:
            raise ValueError("Budget must be positive")

        project_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        db = self._ensure_db()

        await db.execute(
            """INSERT INTO projects (id, description, status, requested_by, budget_usd, created_at, idempotency_key)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                project_id,
                description,
                ProjectStatus.QUEUED.value,
                requested_by,
                budget_usd,
                now,
                idempotency_key,
            ),
        )
        await db.commit()
        return await self.get(project_id)

    async def add_message(
        self,
        project_id: str,
        *,
        text: str,
        kind: str = "note",
        source: str = "",
    ) -> ProjectMessage:
        """Append a message to a project inbox."""
        project_id = str(project_id).strip()
        if not project_id:
            raise ValueError("project_id required")
        msg_text = str(text or "").strip()
        if not msg_text:
            raise ValueError("Message text cannot be empty")
        if len(msg_text) > 10000:
            raise ValueError("Message too long (max 10,000 characters)")
        db = self._ensure_db()
        ts = datetime.now(timezone.utc).timestamp()
        cursor = await db.execute(
            "INSERT INTO project_messages(project_id, ts, kind, source, text, handled_at) "
            "VALUES (?, ?, ?, ?, ?, NULL) RETURNING *;",
            (project_id, float(ts), str(kind or "note"), str(source or ""), msg_text),
        )
        row = await cursor.fetchone()
        await db.commit()
        if row is None:
            raise RuntimeError("Failed to insert message")
        return self._row_to_message(row)

    async def list_unhandled_messages(
        self,
        project_id: str,
        *,
        after_id: int = 0,
        limit: int = 100,
    ) -> list[ProjectMessage]:
        """List unhandled messages for a project (newest first is not guaranteed)."""
        db = self._ensure_db()
        safe_limit = max(1, min(int(limit), 500))
        cursor = await db.execute(
            "SELECT * FROM project_messages "
            "WHERE project_id = ? AND handled_at IS NULL AND id > ? "
            "ORDER BY id ASC LIMIT ?;",
            (str(project_id), int(after_id), int(safe_limit)),
        )
        rows = await cursor.fetchall()
        return [self._row_to_message(r) for r in rows]

    async def mark_message_handled(self, message_id: int) -> None:
        """Mark a message as handled."""
        db = self._ensure_db()
        ts = datetime.now(timezone.utc).timestamp()
        await db.execute(
            "UPDATE project_messages SET handled_at = ? WHERE id = ?;",
            (float(ts), int(message_id)),
        )
        await db.commit()

    async def get(self, project_id: str) -> Project:
        """Get a project by ID."""
        db = self._ensure_db()
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Project not found: {project_id}")
        return self._row_to_project(row)

    async def get_for_requester(self, project_id: str, requested_by: str) -> Project:
        """Get a project by id scoped to the requesting owner."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM projects WHERE id = ? AND requested_by = ?",
            (project_id, requested_by),
        )
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Project not found: {project_id}")
        return self._row_to_project(row)

    async def get_by_idempotency(
        self,
        requested_by: str,
        idempotency_key: str,
    ) -> Project | None:
        """Lookup an existing request by requester + idempotency key."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM projects WHERE requested_by = ? AND idempotency_key = ?",
            (requested_by, idempotency_key),
        )
        row = await cursor.fetchone()
        return self._row_to_project(row) if row is not None else None

    async def list_all(self, limit: int = 50) -> list[Project]:
        """List all projects, newest first."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM projects ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_project(r) for r in rows]

    async def list_for_requester(self, requested_by: str, limit: int = 50) -> list[Project]:
        """List projects belonging to one requester, newest first."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM projects WHERE requested_by = ? ORDER BY created_at DESC LIMIT ?",
            (requested_by, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_project(r) for r in rows]

    async def list_by_status(self, status: ProjectStatus) -> list[Project]:
        """List projects with a given status."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM projects WHERE status = ? ORDER BY created_at ASC",
            (status.value,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_project(r) for r in rows]

    async def list_by_status_for_requester(
        self,
        status: ProjectStatus,
        requested_by: str,
    ) -> list[Project]:
        """List projects for a requester with a given status."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM projects WHERE status = ? AND requested_by = ? ORDER BY created_at ASC",
            (status.value, requested_by),
        )
        rows = await cursor.fetchall()
        return [self._row_to_project(r) for r in rows]

    async def dequeue(self) -> Project | None:
        """Get the oldest queued project and atomically mark it as building."""
        db = self._ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        cursor = await db.execute(
            """UPDATE projects
               SET status = ?, started_at = ?
               WHERE id = (
                   SELECT id FROM projects
                   WHERE status = ?
                   ORDER BY created_at ASC
                   LIMIT 1
               )
               RETURNING *""",
            (ProjectStatus.BUILDING.value, now, ProjectStatus.QUEUED.value),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        await db.commit()
        return self._row_to_project(row)

    async def update_phase(self, project_id: str, phase: str) -> None:
        """Update the current build phase."""
        db = self._ensure_db()
        await db.execute(
            "UPDATE projects SET phase = ? WHERE id = ?", (phase, project_id)
        )
        await db.commit()

    async def update_name(self, project_id: str, name: str, workspace_path: str) -> None:
        """Update project name and workspace path once SPEC completes."""
        db = self._ensure_db()
        await db.execute(
            "UPDATE projects SET name = ?, workspace_path = ? WHERE id = ?",
            (name, workspace_path, project_id),
        )
        await db.commit()

    async def mark_completed(self, project_id: str, cost_usd: float) -> None:
        """Mark a project as completed."""
        db = self._ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "UPDATE projects SET status = ?, cost_usd = ?, completed_at = ?, phase = 'complete' WHERE id = ?",
            (ProjectStatus.COMPLETED.value, cost_usd, now, project_id),
        )
        await db.commit()

    async def mark_failed(self, project_id: str, error: str, cost_usd: float = 0.0) -> None:
        """Mark a project as failed."""
        db = self._ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "UPDATE projects SET status = ?, error = ?, cost_usd = ?, completed_at = ? WHERE id = ?",
            (ProjectStatus.FAILED.value, error, cost_usd, now, project_id),
        )
        await db.commit()

    async def mark_paused(self, project_id: str, *, error: str = "", cost_usd: float = 0.0) -> None:
        """Mark a project as paused (HIL checkpoint declined)."""
        db = self._ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "UPDATE projects SET status = ?, error = ?, cost_usd = ?, completed_at = ? WHERE id = ?",
            (ProjectStatus.PAUSED.value, error, cost_usd, now, project_id),
        )
        await db.commit()

    async def cancel(self, project_id: str) -> bool:
        """Cancel a queued project. Returns False if not queued."""
        db = self._ensure_db()
        cursor = await db.execute(
            "UPDATE projects SET status = ? WHERE id = ? AND status = ?",
            (ProjectStatus.CANCELLED.value, project_id, ProjectStatus.QUEUED.value),
        )
        await db.commit()
        return cursor.rowcount > 0

    async def cancel_for_requester(self, project_id: str, requested_by: str) -> bool:
        """Cancel a queued project owned by requester. Returns False if unavailable."""
        db = self._ensure_db()
        cursor = await db.execute(
            "UPDATE projects SET status = ? WHERE id = ? AND requested_by = ? AND status = ?",
            (
                ProjectStatus.CANCELLED.value,
                project_id,
                requested_by,
                ProjectStatus.QUEUED.value,
            ),
        )
        await db.commit()
        return cursor.rowcount > 0

    async def unpause_for_requester(self, project_id: str, requested_by: str) -> bool:
        """Re-queue a paused project owned by requester. Returns False if unavailable."""
        db = self._ensure_db()
        cursor = await db.execute(
            "UPDATE projects SET status = ?, error = NULL, completed_at = NULL "
            "WHERE id = ? AND requested_by = ? AND status = ?",
            (
                ProjectStatus.QUEUED.value,
                project_id,
                requested_by,
                ProjectStatus.PAUSED.value,
            ),
        )
        await db.commit()
        return cursor.rowcount > 0

    async def unpause(self, project_id: str) -> bool:
        """Re-queue a paused project (admin/local use)."""
        db = self._ensure_db()
        cursor = await db.execute(
            "UPDATE projects SET status = ?, error = NULL, completed_at = NULL "
            "WHERE id = ? AND status = ?",
            (
                ProjectStatus.QUEUED.value,
                project_id,
                ProjectStatus.PAUSED.value,
            ),
        )
        await db.commit()
        return cursor.rowcount > 0

    async def queue_size(self) -> int:
        """Number of projects waiting in queue."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT COUNT(*) FROM projects WHERE status = ?",
            (ProjectStatus.QUEUED.value,),
        )
        row = await cursor.fetchone()
        return row[0]

    async def queue_size_for_requester(self, requested_by: str) -> int:
        """Number of queued projects for one requester."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT COUNT(*) FROM projects WHERE status = ? AND requested_by = ?",
            (ProjectStatus.QUEUED.value, requested_by),
        )
        row = await cursor.fetchone()
        return row[0]

    async def total_cost(self) -> float:
        """Total cost across all projects."""
        db = self._ensure_db()
        cursor = await db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM projects")
        row = await cursor.fetchone()
        return row[0]

    async def total_cost_for_requester(self, requested_by: str) -> float:
        """Total cost for a specific requester."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM projects WHERE requested_by = ?",
            (requested_by,),
        )
        row = await cursor.fetchone()
        return row[0]

    async def count_created_since(self, requested_by: str, since_iso: str) -> int:
        """Count requests created by requester since timestamp."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT COUNT(*) FROM projects WHERE requested_by = ? AND created_at >= ?",
            (requested_by, since_iso),
        )
        row = await cursor.fetchone()
        return row[0]

    async def count_by_status_for_requester(
        self,
        requested_by: str,
        *,
        statuses: list[ProjectStatus],
    ) -> int:
        """Count projects for requester in a list of statuses."""
        if not statuses:
            return 0
        placeholders = ",".join("?" for _ in statuses)
        args = [requested_by, *[s.value for s in statuses]]
        db = self._ensure_db()
        cursor = await db.execute(
            f"SELECT COUNT(*) FROM projects WHERE requested_by = ? AND status IN ({placeholders})",
            args,
        )
        row = await cursor.fetchone()
        return row[0]
