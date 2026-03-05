"""Task DAG — directed acyclic graph for task scheduling."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPhase(Enum):
    SPEC = "spec"
    BUILD = "build"
    VERIFY = "verify"
    REFACTOR = "refactor"
    DELIVER = "deliver"


@dataclass
class Task:
    """A single task in the DAG."""

    id: str
    description: str
    owner: str = "builder"
    phase: TaskPhase = TaskPhase.BUILD
    status: TaskStatus = TaskStatus.TODO
    depends_on: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    claimed_by: str | None = None
    result: str | None = None
    acceptance_criteria: str = ""
    exports: str = ""  # Interface contract: what this task provides to downstream tasks
    retry_count: int = 0

    def is_ready(self, completed_ids: set[str]) -> bool:
        """A task is ready if all its dependencies are completed."""
        return all(dep in completed_ids for dep in self.depends_on)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "owner": self.owner,
            "phase": self.phase.value,
            "status": self.status.value,
            "depends_on": self.depends_on,
            "files": self.files,
            "claimed_by": self.claimed_by,
            "result": self.result,
            "acceptance_criteria": self.acceptance_criteria,
            "exports": self.exports,
            "retry_count": self.retry_count,
        }


class TaskDAG:
    """Manages the directed acyclic graph of tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        self._tasks[task.id] = task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def total_tasks(self) -> int:
        return len(self._tasks)

    def get_ready_tasks(self) -> list[Task]:
        """Return tasks whose dependencies are all completed and status is TODO."""
        completed = {
            tid for tid, t in self._tasks.items() if t.status == TaskStatus.DONE
        }
        return [
            t
            for t in self._tasks.values()
            if t.status == TaskStatus.TODO and t.is_ready(completed)
        ]

    def get_tasks_by_phase(self, phase: TaskPhase) -> list[Task]:
        return [t for t in self._tasks.values() if t.phase == phase]

    def has_pending_tasks(self, phase: TaskPhase | None = None) -> bool:
        """Check if there are tasks not yet done."""
        tasks = self.get_tasks_by_phase(phase) if phase else self.get_all_tasks()
        return any(
            t.status in (TaskStatus.TODO, TaskStatus.IN_PROGRESS) for t in tasks
        )

    def is_finished(self) -> bool:
        """All tasks are done, blocked, or failed (no work remaining)."""
        return all(
            t.status in (TaskStatus.DONE, TaskStatus.BLOCKED, TaskStatus.FAILED)
            for t in self._tasks.values()
        )

    def is_all_done(self) -> bool:
        """All tasks completed successfully (none blocked or failed)."""
        return all(
            t.status == TaskStatus.DONE for t in self._tasks.values()
        )

    def is_complete(self) -> bool:
        """Backward-compatible alias: finished with no failures."""
        return self.is_finished() and not self.has_failures()

    def has_failures(self) -> bool:
        return any(
            t.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)
            for t in self._tasks.values()
        )

    def mark_in_progress(self, task_id: str, agent_id: str) -> None:
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task: {task_id}")
        task = self._tasks[task_id]
        task.status = TaskStatus.IN_PROGRESS
        task.claimed_by = agent_id

    def mark_done(self, task_id: str, result: str = "") -> None:
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task: {task_id}")
        task = self._tasks[task_id]
        task.status = TaskStatus.DONE
        task.result = result
        task.claimed_by = None

    def mark_failed(self, task_id: str, error: str = "") -> None:
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task: {task_id}")
        task = self._tasks[task_id]
        task.retry_count += 1
        if task.retry_count >= 3:
            task.status = TaskStatus.BLOCKED
            task.result = f"Blocked after 3 failures: {error}"
        else:
            task.status = TaskStatus.FAILED
            task.result = error

    def reset_failed(self, task_id: str) -> None:
        """Reset a failed task to TODO for retry."""
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task: {task_id}")
        task = self._tasks[task_id]
        if task.status == TaskStatus.FAILED:
            task.status = TaskStatus.TODO
            task.retry_count = 0
            task.claimed_by = None

    def validate_acyclic(self) -> None:
        """Check for cycles in the DAG. Raises ValueError if a cycle exists.

        Uses iterative DFS to avoid stack overflow on deep DAGs.
        """
        visited: set[str] = set()
        path: set[str] = set()

        for start_tid in self._tasks:
            if start_tid in visited:
                continue
            # Iterative DFS using explicit stack: (task_id, is_backtrack)
            stack: list[tuple[str, bool]] = [(start_tid, False)]
            while stack:
                task_id, is_backtrack = stack.pop()
                if is_backtrack:
                    path.discard(task_id)
                    visited.add(task_id)
                    continue
                if task_id in path:
                    raise ValueError(f"Cycle detected in task DAG involving task: {task_id}")
                if task_id in visited:
                    continue
                path.add(task_id)
                # Push backtrack marker first (processed after children)
                stack.append((task_id, True))
                task = self._tasks.get(task_id)
                if task:
                    for dep in task.depends_on:
                        stack.append((dep, False))

    def validate(self) -> list[str]:
        """Full DAG validation: cycles + dangling dependencies.

        Returns a list of warning messages (empty if clean).
        Raises ValueError on cycle detection.
        """
        # Check cycles first (raises on failure)
        self.validate_acyclic()

        # Check for dangling dependency references
        warnings: list[str] = []
        known_ids = set(self._tasks.keys())
        for task in self._tasks.values():
            for dep_id in task.depends_on:
                if dep_id not in known_ids:
                    msg = (
                        f"Task '{task.id}' depends on '{dep_id}' "
                        f"which does not exist in the DAG"
                    )
                    warnings.append(msg)
                    logger.warning(msg)
        return warnings

    @classmethod
    def from_dict(cls, tasks_data: list[dict]) -> TaskDAG:
        """Build DAG from a list of task dictionaries (e.g. from Architect output)."""
        dag = cls()
        skipped = 0
        for item in tasks_data:
            if "id" not in item or "description" not in item:
                logger.error(
                    f"Skipping malformed task entry (missing required "
                    f"'id' or 'description' field): {item}"
                )
                skipped += 1
                continue
            try:
                task = Task(
                    id=item["id"],
                    description=item["description"],
                    owner=item.get("owner", "builder"),
                    phase=TaskPhase(item.get("phase", "build")),
                    status=TaskStatus(item.get("status", "todo")),
                    depends_on=item.get("depends_on", []),
                    files=item.get("files", []),
                    claimed_by=item.get("claimed_by"),
                    result=item.get("result"),
                    acceptance_criteria=item.get("acceptance_criteria", ""),
                    exports=item.get("exports", ""),
                    retry_count=item.get("retry_count", 0),
                )
                dag.add_task(task)
            except (ValueError, KeyError) as e:
                logger.error(f"Skipping invalid task {item.get('id', '?')}: {e}")
                skipped += 1

        if skipped:
            logger.error(f"Total tasks skipped due to errors: {skipped}/{len(tasks_data)}")

        # Full validation: cycles + dangling dependencies
        dag.validate()
        return dag

    def save(self, path: Path) -> None:
        """Persist DAG state to JSON for resume capability."""
        data = [t.to_dict() for t in self._tasks.values()]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug(f"DAG saved to {path}")

    @classmethod
    def load(cls, path: Path) -> TaskDAG:
        """Load DAG state from JSON for resume."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in task DAG file {path}: {e}") from e
        if not isinstance(data, list):
            raise ValueError(f"Task DAG file must contain a JSON array, got {type(data).__name__}")
        return cls.from_dict(data)

    def to_markdown(self) -> str:
        """Render the DAG as a markdown task list (DEV_PLAN format)."""
        lines = ["# DEV_PLAN\n"]

        for phase in TaskPhase:
            tasks = self.get_tasks_by_phase(phase)
            if not tasks:
                continue

            lines.append(f"## Phase: {phase.value.upper()}\n")
            for t in tasks:
                status_icon = {
                    TaskStatus.TODO: "[ ]",
                    TaskStatus.IN_PROGRESS: "[~]",
                    TaskStatus.DONE: "[x]",
                    TaskStatus.FAILED: "[!]",
                    TaskStatus.BLOCKED: "[B]",
                }[t.status]

                owner_str = f"owner: {t.claimed_by or t.owner}"
                line = f"- {status_icon} {t.id}: {t.description} | {owner_str}"
                if t.depends_on:
                    line += f" | depends: {', '.join(t.depends_on)}"
                lines.append(line)
            lines.append("")

        return "\n".join(lines)
