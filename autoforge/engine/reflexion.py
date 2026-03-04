"""Reflexion — Verbal Reinforcement Learning with Episodic Memory.

Inspired by Reflexion (NeurIPS 2023, Shinn et al.).

Key insight: agents that verbally reflect on failures and maintain those
reflections as episodic memory achieve +11% on HumanEval over baseline.

Unlike SICA (which edits constitution files permanently), Reflexion
maintains structured *episodic* memory — short reflections on what went
wrong and why — that is injected into the next retry attempt.  This is
the "verbal RL" loop: fail → reflect → retry with reflection context.

Integration points:
  - _fix_failures: after each failed fix attempt, generate a reflection
  - _build_single_task: inject relevant past reflections as context
  - Cross-project: persist reflections for recurring failure patterns
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoforge.engine.llm_router import TaskComplexity

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────


@dataclass
class Reflection:
    """A single verbal reflection on a failure."""

    id: str
    task_description: str
    failure_summary: str
    reflection: str  # The verbal reflection — what went wrong + what to try next
    timestamp: float = field(default_factory=time.time)
    attempt_number: int = 1
    outcome: str = "pending"  # "pending" | "resolved" | "persistent"
    project: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_description": self.task_description,
            "failure_summary": self.failure_summary,
            "reflection": self.reflection,
            "timestamp": self.timestamp,
            "attempt_number": self.attempt_number,
            "outcome": self.outcome,
            "project": self.project,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Reflection:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ReflectionMemory:
    """Episodic memory buffer for reflections."""

    reflections: list[Reflection] = field(default_factory=list)
    max_per_task: int = 5  # Max reflections per task chain
    max_total: int = 200  # Total memory budget
    max_context_reflections: int = 3  # Max reflections injected into prompt

    def add(self, reflection: Reflection) -> None:
        self.reflections.append(reflection)
        if len(self.reflections) > self.max_total:
            # Evict oldest resolved reflections first
            resolved = [r for r in self.reflections if r.outcome == "resolved"]
            if resolved:
                self.reflections.remove(resolved[0])
            else:
                self.reflections.pop(0)

    def get_relevant(
        self, task_description: str, project: str = "", top_k: int = 0,
    ) -> list[Reflection]:
        """Retrieve reflections relevant to a task, most recent first."""
        if top_k <= 0:
            top_k = self.max_context_reflections

        candidates: list[Reflection] = []
        task_words = set(task_description.lower().split())

        for r in reversed(self.reflections):
            # Relevance: word overlap between task descriptions
            ref_words = set(r.task_description.lower().split())
            overlap = len(task_words & ref_words) / max(len(task_words | ref_words), 1)

            # Boost same-project reflections
            same_project = r.project == project if project else False

            if overlap > 0.15 or same_project:
                candidates.append(r)

        # Prioritize: unresolved > recent > high overlap
        candidates.sort(
            key=lambda r: (
                r.outcome != "resolved",  # Unresolved first
                r.timestamp,  # Then most recent
            ),
            reverse=True,
        )
        return candidates[:top_k]

    def mark_resolved(self, reflection_id: str) -> None:
        for r in self.reflections:
            if r.id == reflection_id:
                r.outcome = "resolved"
                return

    def get_chain(self, task_id_prefix: str) -> list[Reflection]:
        """Get the reflection chain for a task (all attempts)."""
        return [
            r for r in self.reflections
            if r.id.startswith(task_id_prefix)
        ]


# ──────────────────────────────────────────────
# Reflexion Engine
# ──────────────────────────────────────────────

# Template for generating reflections from failure
REFLECT_PROMPT = """\
You are a senior software engineer reflecting on a failed code attempt.

## Task
{task_description}

## What happened
{failure_summary}

## Attempt #{attempt_number}
{previous_reflections}

## Instructions
Write a concise reflection (2-4 sentences) that:
1. Identifies the ROOT CAUSE of the failure (not just symptoms)
2. Proposes a SPECIFIC different strategy for the next attempt
3. Notes any patterns you've seen across attempts

Format your reflection as a single paragraph. Be specific and actionable.
Do NOT repeat the failure description — focus on WHY it failed and WHAT to change."""

# Template for building retry context from reflections
CONTEXT_TEMPLATE = """\
## Previous Reflections (Episodic Memory)
You have attempted similar tasks before. Learn from these reflections:

{reflections}

IMPORTANT: Do NOT repeat the same approach that failed. Use these reflections
to inform a DIFFERENT strategy."""


class ReflexionEngine:
    """Verbal reinforcement learning engine.

    Maintains episodic memory of reflections and generates new reflections
    from failures.  Reflections are injected into retry prompts so agents
    learn from their mistakes within a single pipeline run *and* across
    projects.
    """

    def __init__(self) -> None:
        self._memory = ReflectionMemory()
        self._attempt_counters: dict[str, int] = {}  # task_id → attempt count
        self._persistence_path: Path | None = None

    # ── Core API ─────────────────────────────────

    async def reflect_on_failure(
        self,
        task_id: str,
        task_description: str,
        failure_summary: str,
        llm: Any,
        project: str = "",
    ) -> Reflection:
        """Generate a verbal reflection on a failure.

        Args:
            task_id: Unique task identifier.
            task_description: What the task was trying to do.
            failure_summary: What went wrong (test output, error message, etc.)
            llm: LLMRouter instance for generating reflections.
            project: Project name for cross-project memory.

        Returns:
            A Reflection object containing the verbal reflection.
        """
        # Track attempts
        self._attempt_counters[task_id] = self._attempt_counters.get(task_id, 0) + 1
        attempt = self._attempt_counters[task_id]

        # Build previous reflections context
        chain = self._memory.get_chain(task_id)
        prev_text = ""
        if chain:
            prev_text = "\n".join(
                f"- Attempt {r.attempt_number}: {r.reflection}" for r in chain
            )

        prompt = REFLECT_PROMPT.format(
            task_description=task_description[:500],
            failure_summary=failure_summary[:1000],
            attempt_number=attempt,
            previous_reflections=prev_text or "(first attempt)",
        )

        try:
            response = await llm.call(
                system="You are a reflective coding agent. Be concise and specific.",
                messages=[{"role": "user", "content": prompt}],
                complexity=TaskComplexity.STANDARD,
            )

            reflection_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    reflection_text += block.text

            reflection = Reflection(
                id=f"{task_id}-attempt{attempt}",
                task_description=task_description[:200],
                failure_summary=failure_summary[:300],
                reflection=reflection_text.strip()[:500],
                attempt_number=attempt,
                project=project,
                tags=self._extract_tags(failure_summary),
            )
            self._memory.add(reflection)
            logger.info(f"[Reflexion] Generated reflection for {task_id} (attempt {attempt})")
            return reflection

        except Exception as e:
            logger.warning(f"[Reflexion] Failed to generate reflection: {e}")
            # Fallback: create a minimal reflection
            reflection = Reflection(
                id=f"{task_id}-attempt{attempt}",
                task_description=task_description[:200],
                failure_summary=failure_summary[:300],
                reflection=f"Previous attempt failed: {failure_summary[:200]}. Try a different approach.",
                attempt_number=attempt,
                project=project,
            )
            self._memory.add(reflection)
            return reflection

    def build_retry_context(
        self,
        task_description: str,
        project: str = "",
    ) -> str:
        """Build context string from relevant past reflections.

        This is injected into the agent's prompt on retry attempts.
        """
        relevant = self._memory.get_relevant(task_description, project)
        if not relevant:
            return ""

        reflection_strs = []
        for r in relevant:
            prefix = f"[Attempt {r.attempt_number}"
            if r.project:
                prefix += f", project={r.project}"
            prefix += "]"
            reflection_strs.append(f"{prefix} {r.reflection}")

        return CONTEXT_TEMPLATE.format(
            reflections="\n\n".join(reflection_strs),
        )

    def mark_success(self, task_id: str) -> None:
        """Mark all reflections for a task as resolved (the fix worked)."""
        chain = self._memory.get_chain(task_id)
        for r in chain:
            r.outcome = "resolved"
        # Reset attempt counter
        self._attempt_counters.pop(task_id, None)

    def mark_persistent(self, task_id: str) -> None:
        """Mark failure as persistent (all retries exhausted)."""
        chain = self._memory.get_chain(task_id)
        for r in chain:
            if r.outcome == "pending":
                r.outcome = "persistent"

    def get_recent_memories(self, n: int) -> list[Reflection]:
        """Return the *n* most recent reflections from episodic memory."""
        return self._memory.reflections[-n:] if n > 0 else []

    # ── Persistence ──────────────────────────────

    def save_state(self, output_dir: Path) -> None:
        """Persist reflection memory to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "reflections": [r.to_dict() for r in self._memory.reflections],
            "attempt_counters": self._attempt_counters,
        }
        (output_dir / "reflexion_memory.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8",
        )

    def load_state(self, state_dir: Path) -> None:
        """Load reflection memory from disk."""
        path = state_dir / "reflexion_memory.json"
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            self._memory.reflections = [
                Reflection.from_dict(r) for r in state.get("reflections", [])
            ]
            self._attempt_counters = state.get("attempt_counters", {})
            logger.info(
                f"[Reflexion] Loaded {len(self._memory.reflections)} reflections"
            )
        except Exception as e:
            logger.warning(f"[Reflexion] Failed to load state: {e}")

    # ── Stats ────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        total = len(self._memory.reflections)
        resolved = sum(1 for r in self._memory.reflections if r.outcome == "resolved")
        persistent = sum(1 for r in self._memory.reflections if r.outcome == "persistent")
        return {
            "total_reflections": total,
            "resolved": resolved,
            "persistent": persistent,
            "pending": total - resolved - persistent,
            "resolution_rate": resolved / max(total, 1),
        }

    # ── Internal ─────────────────────────────────

    @staticmethod
    def _extract_tags(failure_summary: str) -> list[str]:
        """Extract rough failure category tags from error text."""
        tags: list[str] = []
        lower = failure_summary.lower()
        tag_keywords = {
            "import": "import_error",
            "syntax": "syntax_error",
            "type": "type_error",
            "attribute": "attribute_error",
            "timeout": "timeout",
            "assert": "assertion_failure",
            "connection": "network_error",
            "permission": "permission_error",
            "not found": "not_found",
            "undefined": "undefined_reference",
        }
        for keyword, tag in tag_keywords.items():
            if keyword in lower:
                tags.append(tag)
        return tags[:5]
