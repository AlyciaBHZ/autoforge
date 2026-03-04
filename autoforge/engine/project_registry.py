"""Project Registry — SQLite-backed multi-project management.

Tracks all projects across daemon runs: queued, building, completed, failed.
Each project has its own budget, workspace, and status.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class ProjectStatus(str, Enum):
    QUEUED = "queued"
    BUILDING = "building"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
    error TEXT
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
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

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
        )

    async def enqueue(
        self,
        description: str,
        requested_by: str = "cli",
        budget_usd: float = 10.0,
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
            """INSERT INTO projects (id, description, status, requested_by, budget_usd, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (project_id, description, ProjectStatus.QUEUED.value, requested_by, budget_usd, now),
        )
        await db.commit()
        return await self.get(project_id)

    async def get(self, project_id: str) -> Project:
        """Get a project by ID."""
        db = self._ensure_db()
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Project not found: {project_id}")
        return self._row_to_project(row)

    async def list_all(self, limit: int = 50) -> list[Project]:
        """List all projects, newest first."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM projects ORDER BY created_at DESC LIMIT ?", (limit,)
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

    async def dequeue(self) -> Project | None:
        """Get the oldest queued project and mark it as building."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM projects WHERE status = ? ORDER BY created_at ASC LIMIT 1",
            (ProjectStatus.QUEUED.value,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        project = self._row_to_project(row)
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "UPDATE projects SET status = ?, started_at = ? WHERE id = ?",
            (ProjectStatus.BUILDING.value, now, project.id),
        )
        await db.commit()
        project.status = ProjectStatus.BUILDING
        project.started_at = now
        return project

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

    async def cancel(self, project_id: str) -> bool:
        """Cancel a queued project. Returns False if not queued."""
        db = self._ensure_db()
        cursor = await db.execute(
            "UPDATE projects SET status = ? WHERE id = ? AND status = ?",
            (ProjectStatus.CANCELLED.value, project_id, ProjectStatus.QUEUED.value),
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

    async def total_cost(self) -> float:
        """Total cost across all projects."""
        db = self._ensure_db()
        cursor = await db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM projects")
        row = await cursor.fetchone()
        return row[0]
