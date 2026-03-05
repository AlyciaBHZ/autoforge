"""Mid-task checkpoints — lightweight review during agent execution.

Instead of waiting until a Builder finishes all 25-30 turns before review,
this module enables mid-execution direction checks. A lightweight LLM call
(using the fast model) evaluates whether the agent is on the right track.

Inspired by Process Reward Models (PRMs): instead of only rewarding the
final output, we give feedback at each step of the reasoning process.

Usage in agent_base.py:
    checkpoint_mgr = CheckpointManager(config, llm)
    ...
    # At turn 8, 15, etc.:
    verdict = await checkpoint_mgr.check_direction(
        task_description=task_desc,
        messages_so_far=messages,
        files_written=files_list,
        turn=current_turn,
    )
    if verdict.should_reset:
        # Rollback to checkpoint and try different approach
        messages = checkpoint_mgr.get_checkpoint_messages(checkpoint_id)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


_CHECKPOINT_DIRECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["on_track", "score", "suggested_action"],
    "properties": {
        "on_track": {"type": "boolean"},
        "score": {"type": "number"},
        "feedback": {"type": "string"},
        "suggested_action": {
            "type": "string",
            "enum": ["continue", "adjust", "reset"],
        },
    },
}


@dataclass
class CheckpointVerdict:
    """Result of a mid-task direction check."""
    on_track: bool                     # Is the agent heading in the right direction?
    score: float = 0.0                 # 0.0 to 1.0 direction quality
    should_reset: bool = False         # Should we rollback to last checkpoint?
    feedback: str = ""                 # Specific guidance for course correction
    suggested_action: str = "continue" # "continue", "adjust", or "reset"


@dataclass
class Checkpoint:
    """A snapshot of agent state at a particular turn."""
    id: str
    turn: int
    timestamp: float
    messages_snapshot: list[dict[str, Any]]
    files_at_checkpoint: list[str]
    score: float = 0.0
    verdict: CheckpointVerdict | None = None


class CheckpointManager:
    """Manages mid-task checkpoints for agent direction monitoring.

    Configuration:
        checkpoint_interval: Check every N turns (default: 8)
        min_score_threshold: Below this score, suggest course correction (default: 0.4)
        reset_threshold: Below this, force a reset to last good checkpoint (default: 0.2)
    """

    def __init__(
        self,
        config: Any,
        llm: Any,
        checkpoint_interval: int = 8,
        min_score_threshold: float = 0.4,
        reset_threshold: float = 0.2,
    ) -> None:
        self.config = config
        self.llm = llm
        self.checkpoint_interval = checkpoint_interval
        self.min_score_threshold = min_score_threshold
        self.reset_threshold = reset_threshold
        self._checkpoints: dict[str, Checkpoint] = {}
        self._checkpoint_order: list[str] = []

    def should_check(self, turn: int) -> bool:
        """Determine if a checkpoint review should happen at this turn."""
        return turn > 0 and turn % self.checkpoint_interval == 0

    def save_checkpoint(
        self,
        turn: int,
        messages: list[dict[str, Any]],
        files_written: list[str],
    ) -> str:
        """Save a checkpoint snapshot of current agent state.

        Returns the checkpoint ID.
        """
        checkpoint_id = f"ckpt-{turn:03d}-{int(time.time()) % 10000}"
        # Deep-copy messages (they contain mutable lists)
        import copy
        checkpoint = Checkpoint(
            id=checkpoint_id,
            turn=turn,
            timestamp=time.time(),
            messages_snapshot=copy.deepcopy(messages),
            files_at_checkpoint=list(files_written),
        )
        self._checkpoints[checkpoint_id] = checkpoint
        self._checkpoint_order.append(checkpoint_id)

        logger.info(f"[Checkpoint] Saved {checkpoint_id} at turn {turn} "
                     f"({len(files_written)} files, {len(messages)} messages)")
        return checkpoint_id

    async def check_direction(
        self,
        task_description: str,
        messages_so_far: list[dict[str, Any]],
        files_written: list[str],
        turn: int,
        context: str = "",
    ) -> CheckpointVerdict:
        """Perform a lightweight direction check.

        Uses a fast model to evaluate whether the agent's recent actions
        are moving toward task completion effectively.

        Args:
            task_description: What the agent should be doing.
            messages_so_far: Full message history.
            files_written: List of files written so far.
            turn: Current turn number.
            context: Additional context (spec, architecture).

        Returns:
            CheckpointVerdict with direction assessment.
        """
        from autoforge.engine.llm_router import TaskComplexity

        # Build a condensed summary of recent activity
        recent_summary = self._summarize_recent_turns(messages_so_far, last_n=5)

        prompt = (
            f"You are a technical project supervisor doing a mid-task review.\n\n"
            f"## Task Goal\n{task_description}\n\n"
            f"## Progress So Far (Turn {turn})\n"
            f"Files written: {files_written}\n\n"
            f"## Recent Activity (last 5 turns)\n{recent_summary}\n\n"
        )
        if context:
            prompt += f"## Project Context\n{context[:1500]}\n\n"

        prompt += (
            f"## Evaluate\n"
            f"Is this agent on the right track to complete the task?\n"
            f"Look for:\n"
            f"- Wrong direction: using wrong framework/library/approach\n"
            f"- Spinning: repeating similar actions without progress\n"
            f"- Over-engineering: doing more than needed\n"
            f"- Missing the point: working on the wrong aspect\n\n"
            f"Output JSON:\n"
            f'{{"on_track": true/false, "score": 0.0-1.0, '
            f'"feedback": "specific guidance", '
            f'"suggested_action": "continue|adjust|reset"}}'
        )

        try:
            response = await self.llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You are a senior tech lead reviewing a junior developer's progress. "
                       "Be constructive but honest. Respond with JSON only.",
                messages=[{"role": "user", "content": prompt}],
                response_json_schema=_CHECKPOINT_DIRECTION_SCHEMA,
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            from autoforge.engine.utils import extract_json_from_text
            try:
                data = extract_json_from_text(
                    text,
                    schema=_CHECKPOINT_DIRECTION_SCHEMA,
                    strict=True,
                )
                score = float(data.get("score", 0.5))

                verdict = CheckpointVerdict(
                    on_track=data.get("on_track", True),
                    score=score,
                    should_reset=score < self.reset_threshold,
                    feedback=data.get("feedback", ""),
                    suggested_action=data.get("suggested_action", "continue"),
                )
            except (ValueError, json.JSONDecodeError):
                verdict = CheckpointVerdict(
                    on_track=False, score=0.3,
                    feedback="Could not parse evaluation — treating as low-confidence.",
                )

        except Exception as e:
            logger.warning(f"[Checkpoint] Direction check failed: {e}")
            verdict = CheckpointVerdict(
                on_track=True, score=0.5,
                feedback=f"Check failed ({e}) — continuing.",
            )

        # Save verdict with checkpoint
        checkpoint_id = self.save_checkpoint(turn, messages_so_far, files_written)
        self._checkpoints[checkpoint_id].verdict = verdict
        self._checkpoints[checkpoint_id].score = verdict.score

        logger.info(
            f"[Checkpoint] Turn {turn}: score={verdict.score:.2f} "
            f"action={verdict.suggested_action} on_track={verdict.on_track}"
        )

        return verdict

    def get_last_good_checkpoint(self) -> Checkpoint | None:
        """Find the most recent checkpoint with a passing score."""
        for cid in reversed(self._checkpoint_order):
            cp = self._checkpoints.get(cid)
            if cp and cp.score >= self.min_score_threshold:
                return cp
        return None

    def get_checkpoint_messages(self, checkpoint_id: str) -> list[dict[str, Any]]:
        """Retrieve the message snapshot from a checkpoint for rollback."""
        cp = self._checkpoints.get(checkpoint_id)
        if cp:
            return cp.messages_snapshot
        return []

    def get_all_checkpoints(self) -> list[dict[str, Any]]:
        """Return all checkpoints as serializable dicts."""
        return [
            {
                "id": cp.id,
                "turn": cp.turn,
                "score": cp.score,
                "files": cp.files_at_checkpoint,
                "verdict": {
                    "on_track": cp.verdict.on_track,
                    "score": cp.verdict.score,
                    "feedback": cp.verdict.feedback,
                    "action": cp.verdict.suggested_action,
                } if cp.verdict else None,
            }
            for cp in (self._checkpoints[cid] for cid in self._checkpoint_order)
        ]

    def _summarize_recent_turns(
        self,
        messages: list[dict[str, Any]],
        last_n: int = 5,
    ) -> str:
        """Create a condensed summary of the last N message pairs."""
        parts = []
        # Take the last 2*last_n messages (assistant + user pairs)
        recent = messages[-(last_n * 2):]
        for msg in recent:
            role = msg.get("role", "?")
            content = msg.get("content", "")

            if isinstance(content, str):
                parts.append(f"[{role}] {content[:300]}")
            elif isinstance(content, list):
                # Summarize tool calls/results
                for item in content:
                    if hasattr(item, "type"):
                        if item.type == "tool_use":
                            parts.append(f"[{role}] Tool: {item.name}({json.dumps(item.input)[:100]})")
                        elif item.type == "text":
                            parts.append(f"[{role}] {item.text[:200]}")
                    elif isinstance(item, dict):
                        if item.get("type") == "tool_result":
                            result_preview = str(item.get("content", ""))[:150]
                            parts.append(f"[tool_result] {result_preview}")

        return "\n".join(parts[-15:])  # Cap at 15 lines
