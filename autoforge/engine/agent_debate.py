"""Conditional multi-agent debate — reward-guided directed debate.

ICLR 2025 research shows that simple Multi-Agent Debate (MAD) does not
always outperform single-agent Chain-of-Thought (CoT). The key insight
is that debate must be CONDITIONAL and DIRECTED:

  1. Debate is only triggered when there's genuine uncertainty or
     disagreement between agents
  2. A reward signal guides which arguments to pursue vs. abandon
  3. The debate has a convergence criterion, not just fixed rounds

This module implements:
  - **Conditional trigger**: Debate only when uncertainty > threshold
  - **Reward-directed**: Process reward model scores guide argument selection
  - **Convergence check**: Stop when agents reach consensus or max rounds
  - **Synthesis**: Combine best arguments into final decision

Use cases in AutoForge:
  - Architecture design debates (Architect vs. alternatives)
  - Code review disputes (Reviewer vs. Builder on approach)
  - Technology choice conflicts (multiple valid stacks)

References:
  - On the Limits of Multi-Agent Debate (ICLR 2025)
  - Reward-guided debate frameworks (2025)
  - Process reward models for debate evaluation (2024)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Algorithmic Argument Ranking
# ──────────────────────────────────────────────


class EloArgumentRanker:
    """Elo-based ranking system for debate arguments.

    Tracks argument quality using the Elo rating system, allowing
    arguments to be objectively ranked and compared across debates.
    """
    K_FACTOR = 32

    def __init__(self) -> None:
        self._ratings: dict[str, float] = {}  # argument_id → Elo rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for rating_a vs rating_b."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, winner_id: str, loser_id: str) -> None:
        """Update ratings after a debate outcome."""
        ra = self._ratings.get(winner_id, 1500)
        rb = self._ratings.get(loser_id, 1500)
        ea = self.expected_score(ra, rb)
        self._ratings[winner_id] = ra + self.K_FACTOR * (1 - ea)
        self._ratings[loser_id] = rb + self.K_FACTOR * (0 - (1 - ea))

    def get_rating(self, arg_id: str) -> float:
        """Get current Elo rating for an argument."""
        return self._ratings.get(arg_id, 1500)


class ArgumentQualityScorer:
    """Score argument quality using text metrics (non-LLM).

    Evaluates arguments based on:
    - Evidence density: presence of evidence indicators
    - Specificity: unique terminology
    - Length adequacy: not too short or verbose
    """

    def score(self, argument: str, context: str = "") -> dict[str, float]:
        """Score argument quality metrics.

        Returns a dict with individual metrics and composite score [0, 1].
        """
        words = argument.lower().split()

        # Evidence density: ratio of evidence-indicating words
        evidence_words = {
            'because', 'therefore', 'since', 'given', 'evidence', 'data',
            'shows', 'demonstrates', 'proves', 'according', 'research',
            'study', 'result', 'evidence', 'test', 'validation'
        }
        evidence_count = sum(1 for w in words if w in evidence_words)
        evidence_density = evidence_count / max(len(words), 1)

        # Specificity: unique terms / total terms
        specificity = len(set(words)) / max(len(words), 1)

        # Length adequacy: penalize too short or too long
        # Optimal around 50 words; acceptable 30-200
        length_score = min(1.0, len(words) / 50) * min(1.0, 200 / max(len(words), 1))

        # Composite: weighted combination
        composite = (
            evidence_density * 0.4 +
            specificity * 0.3 +
            length_score * 0.3
        )

        return {
            "evidence_density": evidence_density,
            "specificity": specificity,
            "length_adequacy": length_score,
            "composite": composite,
        }


class LogicalConsistencyChecker:
    """Check logical consistency of arguments.

    Uses propositional logic patterns and term analysis to detect
    contradictions and inconsistencies.
    """

    def check_consistency(self, arguments: list[str]) -> dict[str, Any]:
        """Check consistency across a set of arguments.

        Returns dict with:
        - consistent: bool
        - conflicts: list of (arg_i, arg_j, reason)
        - confidence: float [0, 1]
        """
        if len(arguments) < 2:
            return {"consistent": True, "conflicts": [], "confidence": 1.0}

        conflicts = []

        # Check each pair for potential contradiction
        for i, arg_a in enumerate(arguments):
            for j, arg_b in enumerate(arguments[i+1:], i+1):
                if self._may_contradict(arg_a, arg_b):
                    conflicts.append((i, j, "Potential contradiction detected"))

        consistent = len(conflicts) == 0
        # Confidence: 1.0 if consistent, decreases with conflict severity
        confidence = max(0.1, 1.0 - len(conflicts) * 0.2)

        return {
            "consistent": consistent,
            "conflicts": conflicts,
            "confidence": confidence,
        }

    def _may_contradict(self, a: str, b: str) -> bool:
        """Check if two arguments may contradict."""
        a_lower, b_lower = a.lower(), b.lower()

        # Simple pattern: negation indicators
        negators = ['never', 'always', 'must not', 'should not', 'do not', 'cannot']
        a_has_neg = any(neg in a_lower for neg in negators)
        b_has_neg = any(neg in b_lower for neg in negators)

        # If one has negation and the other doesn't, check for shared key terms
        if a_has_neg != b_has_neg:
            a_terms = set(a_lower.split()) - {'the', 'a', 'an', 'is', 'to', 'and', 'or'}
            b_terms = set(b_lower.split()) - {'the', 'a', 'an', 'is', 'to', 'and', 'or'}
            shared = a_terms & b_terms
            if len(shared) > 3:
                return True

        return False


def _extract_json(text: str) -> dict[str, Any] | None:
    """Robustly extract JSON object from LLM response text."""
    from autoforge.engine.utils import extract_json_from_text
    try:
        return extract_json_from_text(
            text,
            schema={
                "type": "object",
                "required": ["consistent", "conflicts", "confidence"],
                "properties": {
                    "consistent": {"type": "boolean"},
                    "conflicts": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "action": {"type": "string"},
                },
            },
        )
    except ValueError:
        return None


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────


@dataclass
class DebateArgument:
    """A single argument in a debate."""
    agent_role: str          # Who made this argument
    position: str            # The stance/approach advocated
    reasoning: str           # Supporting reasoning
    evidence: str = ""       # Code examples, benchmarks, etc.
    reward_score: float = 0.0  # Process reward model score
    round_number: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_role": self.agent_role,
            "position": self.position,
            "reasoning": self.reasoning[:500],
            "reward_score": self.reward_score,
            "round_number": self.round_number,
        }


@dataclass
class DebateOutcome:
    """Result of a multi-agent debate."""
    topic: str
    winner_position: str
    synthesis: str               # Combined best arguments
    confidence: float = 0.0      # How confident we are in the outcome
    rounds: int = 0
    arguments: list[DebateArgument] = field(default_factory=list)
    convergence_reason: str = ""  # "consensus", "max_rounds", "dominant_reward"
    triggered: bool = True        # Was the debate actually triggered?

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "winner_position": self.winner_position,
            "synthesis": self.synthesis[:1000],
            "confidence": self.confidence,
            "rounds": self.rounds,
            "convergence_reason": self.convergence_reason,
            "triggered": self.triggered,
            "num_arguments": len(self.arguments),
        }


# ──────────────────────────────────────────────
# Debate Engine
# ──────────────────────────────────────────────


class ConditionalDebateEngine:
    """Reward-guided conditional multi-agent debate.

    Key principles:
    1. Don't debate everything — only trigger when there's genuine
       uncertainty or significant disagreement
    2. Use reward signals to prune bad arguments early
    3. Stop when consensus is reached, not after fixed rounds
    4. Synthesise the best parts of each position

    The debate follows this protocol:
    1. Each participant states their initial position
    2. Each participant critiques the other positions
    3. A reward model scores each argument
    4. Low-scoring arguments are pruned
    5. Participants revise their positions
    6. Check for convergence → repeat or synthesise
    """

    # Minimum uncertainty to trigger debate (0-1 scale)
    UNCERTAINTY_THRESHOLD = 0.4
    # Maximum debate rounds
    MAX_ROUNDS = 3
    # Minimum reward score to keep an argument alive
    MIN_REWARD_THRESHOLD = 0.3
    # Consensus threshold: if top argument leads by this margin, stop
    CONSENSUS_MARGIN = 0.25

    def __init__(self) -> None:
        self._debate_history: list[DebateOutcome] = []
        self._elo_ranker = EloArgumentRanker()
        self._quality_scorer = ArgumentQualityScorer()
        self._consistency_checker = LogicalConsistencyChecker()
        self._algorithm_ratio = 0.0  # Track fraction of decisions using algorithms vs LLM

    # ──────── Conditional Trigger ────────

    async def should_debate(
        self,
        topic: str,
        initial_positions: list[dict[str, str]],
        llm: Any,
    ) -> tuple[bool, float]:
        """Determine if a debate should be triggered.

        Returns (should_trigger, uncertainty_score).

        Debate is triggered when:
        1. There are multiple genuinely different positions
        2. The uncertainty/disagreement is above threshold
        3. The topic is important enough (not trivial decisions)
        """
        from autoforge.engine.llm_router import TaskComplexity

        if len(initial_positions) < 2:
            return False, 0.0

        positions_text = ""
        for i, pos in enumerate(initial_positions, 1):
            positions_text += (
                f"Position {i} ({pos.get('agent', 'agent')}):\n"
                f"{pos.get('position', '')}\n\n"
            )

        prompt = (
            "Evaluate whether these positions represent genuine disagreement "
            "that warrants a structured debate, or if one position is clearly "
            "superior.\n\n"
            f"## Topic: {topic}\n\n"
            f"## Positions\n{positions_text}\n"
            "## Task\n"
            "Output JSON:\n"
            '{"uncertainty": 0.0-1.0, "reason": "...", '
            '"positions_differ_substantially": true/false}\n\n'
            "uncertainty = 0 means one position is clearly best; "
            "uncertainty = 1 means they're equally valid."
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You evaluate whether multiple viewpoints warrant formal debate.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            data = _extract_json(text)
            if data:
                uncertainty = float(data.get("uncertainty", 0.5))
                should_trigger = (
                    uncertainty >= self.UNCERTAINTY_THRESHOLD
                    and data.get("positions_differ_substantially", True)
                )
                logger.info(
                    f"[Debate] Uncertainty={uncertainty:.2f}, "
                    f"trigger={should_trigger}: {data.get('reason', '')[:100]}"
                )
                return should_trigger, uncertainty

        except Exception as e:
            logger.warning(f"[Debate] Trigger evaluation failed: {e}")

        # Default: debate if we have multiple positions
        return True, 0.5

    # ──────── Run Debate ────────

    async def run_debate(
        self,
        topic: str,
        participants: list[dict[str, str]],
        context: dict[str, Any],
        llm: Any,
        reward_fn: Any | None = None,
    ) -> DebateOutcome:
        """Run a full conditional debate.

        Args:
            topic: What is being debated
            participants: List of {"agent": role, "position": initial stance}
            context: Additional context (spec, requirements, etc.)
            llm: LLM router
            reward_fn: Optional process reward function for scoring arguments

        Returns:
            DebateOutcome with the synthesised result.
        """
        from autoforge.engine.llm_router import TaskComplexity

        # Step 0: Check if debate is warranted
        should_trigger, uncertainty = await self.should_debate(
            topic, participants, llm,
        )

        if not should_trigger:
            # No debate needed — return the best initial position.
            # Use score if available (from reward-scored candidates), otherwise
            # pick the first participant which is the top candidate from the
            # search tree.
            best = max(participants, key=lambda p: p.get("score", 0.0)) if any(
                "score" in p for p in participants
            ) else participants[0]
            outcome = DebateOutcome(
                topic=topic,
                winner_position=best.get("position", ""),
                synthesis=best.get("position", ""),
                confidence=1.0 - uncertainty,
                rounds=0,
                convergence_reason="no_debate_needed",
                triggered=False,
            )
            self._debate_history.append(outcome)
            return outcome

        # Step 1: Collect initial arguments
        arguments: list[DebateArgument] = []
        for p in participants:
            arg = DebateArgument(
                agent_role=p.get("agent", "unknown"),
                position=p.get("position", ""),
                reasoning=p.get("reasoning", p.get("position", "")),
                round_number=0,
            )
            arguments.append(arg)

        # Step 2: Debate rounds
        all_arguments = list(arguments)

        for round_num in range(1, self.MAX_ROUNDS + 1):
            # Each participant critiques others and refines position
            new_arguments = await self._debate_round(
                topic, arguments, context, round_num, llm,
            )

            # Score arguments: try algorithms first, LLM as fallback
            for arg in new_arguments:
                # Primary: algorithmic scoring (deterministic, fast)
                quality_metrics = self._quality_scorer.score(arg.position, arg.reasoning)
                algo_score = quality_metrics["composite"]
                used_algo = True

                # Fallback: reward function or LLM if available
                if reward_fn:
                    try:
                        reward_score = await reward_fn(arg.position, arg.reasoning)
                        # Blend algorithmic + reward scores
                        arg.reward_score = 0.6 * algo_score + 0.4 * reward_score
                        used_algo = False
                    except Exception:
                        arg.reward_score = algo_score
                else:
                    # Last resort: LLM evaluator
                    try:
                        llm_score = await self._evaluate_argument(
                            arg, topic, context, llm,
                        )
                        arg.reward_score = 0.6 * algo_score + 0.4 * llm_score
                        used_algo = False
                    except Exception:
                        arg.reward_score = algo_score

                # Track algorithm usage ratio
                if used_algo:
                    self._algorithm_ratio = (self._algorithm_ratio * 0.9) + 0.1
                else:
                    self._algorithm_ratio = self._algorithm_ratio * 0.9

            # Prune low-scoring arguments
            surviving = [
                a for a in new_arguments
                if a.reward_score >= self.MIN_REWARD_THRESHOLD
            ]
            if not surviving:
                surviving = new_arguments[:2]  # Keep at least 2

            all_arguments.extend(surviving)
            arguments = surviving

            # Check convergence
            converged, reason = self._check_convergence(arguments)
            if converged:
                outcome = await self._synthesise(
                    topic, all_arguments, context, reason, round_num, llm,
                )
                self._debate_history.append(outcome)
                return outcome

        # Max rounds reached — synthesise best arguments
        outcome = await self._synthesise(
            topic, all_arguments, context, "max_rounds",
            self.MAX_ROUNDS, llm,
        )
        self._debate_history.append(outcome)
        return outcome

    async def _debate_round(
        self,
        topic: str,
        current_arguments: list[DebateArgument],
        context: dict[str, Any],
        round_num: int,
        llm: Any,
    ) -> list[DebateArgument]:
        """Run one round of debate: each participant responds to others."""
        from autoforge.engine.llm_router import TaskComplexity

        new_arguments = []

        for arg in current_arguments:
            # Build opponent positions
            opponents = [
                a for a in current_arguments
                if a.agent_role != arg.agent_role
            ]
            opponent_text = "\n".join(
                f"- {a.agent_role}: {a.position} (score: {a.reward_score:.2f})"
                for a in opponents
            )

            prompt = (
                f"You are the {arg.agent_role} agent in a structured debate.\n\n"
                f"## Topic: {topic}\n"
                f"## Your Current Position\n{arg.position}\n\n"
                f"## Opponent Arguments\n{opponent_text}\n\n"
                f"## Round {round_num}\n"
                "Consider the opposing arguments. Either:\n"
                "1. Strengthen your position with new evidence/reasoning\n"
                "2. Concede valid points and refine your position\n"
                "3. Synthesise the best of both approaches\n\n"
                "Output JSON:\n"
                '{"position": "your refined position", '
                '"reasoning": "why this is better", '
                '"concessions": "what you concede from opponents"}'
            )

            try:
                response = await llm.call(
                    complexity=TaskComplexity.STANDARD,
                    system=f"You are the {arg.agent_role}. Engage constructively.",
                    messages=[{"role": "user", "content": prompt}],
                )

                text = ""
                for block in response.content:
                    if block.type == "text":
                        text += block.text

                data = _extract_json(text)
                if data:
                    new_arg = DebateArgument(
                        agent_role=arg.agent_role,
                        position=data.get("position", arg.position),
                        reasoning=data.get("reasoning", ""),
                        round_number=round_num,
                    )
                    new_arguments.append(new_arg)
                else:
                    new_arguments.append(arg)

            except Exception as e:
                logger.debug(f"[Debate] Round {round_num} failed for {arg.agent_role}: {e}")
                new_arguments.append(arg)

        return new_arguments

    async def _evaluate_argument(
        self,
        argument: DebateArgument,
        topic: str,
        context: dict[str, Any],
        llm: Any,
    ) -> float:
        """Use LLM to score an argument (0-1)."""
        from autoforge.engine.llm_router import TaskComplexity

        prompt = (
            f"Score this debate argument on a scale of 0.0 to 1.0:\n\n"
            f"Topic: {topic}\n"
            f"Position: {argument.position}\n"
            f"Reasoning: {argument.reasoning}\n\n"
            f"Evaluate: correctness, feasibility, completeness, clarity.\n"
            f"Output ONLY a JSON: {{\"score\": 0.X, \"reason\": \"...\"}}"
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You are an objective judge evaluating debate arguments.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            data = _extract_json(text)
            if data:
                return float(data.get("score", 0.5))

        except Exception:
            pass

        return 0.5

    def _check_convergence(
        self, arguments: list[DebateArgument],
    ) -> tuple[bool, str]:
        """Check if the debate has converged.

        Returns (converged, reason).

        Uses both score-based and logical consistency checks.
        """
        if len(arguments) <= 1:
            return True, "single_position"

        # Check if top argument has dominant reward score
        sorted_args = sorted(arguments, key=lambda a: a.reward_score, reverse=True)
        if len(sorted_args) >= 2:
            margin = sorted_args[0].reward_score - sorted_args[1].reward_score
            if margin >= self.CONSENSUS_MARGIN:
                return True, "dominant_reward"

        # Check if all positions are now similar (textual convergence)
        positions = [a.position for a in arguments]
        if len(set(positions)) == 1:
            return True, "consensus"

        # Check logical consistency of remaining arguments
        consistency = self._consistency_checker.check_consistency(positions)
        if consistency["consistent"] and consistency["confidence"] > 0.8:
            return True, "logical_consistency"

        return False, ""

    async def _synthesise(
        self,
        topic: str,
        all_arguments: list[DebateArgument],
        context: dict[str, Any],
        convergence_reason: str,
        rounds: int,
        llm: Any,
    ) -> DebateOutcome:
        """Synthesise the best arguments into a final position."""
        from autoforge.engine.llm_router import TaskComplexity

        # Sort by reward score
        sorted_args = sorted(all_arguments, key=lambda a: a.reward_score, reverse=True)
        top_args = sorted_args[:5]

        args_text = "\n".join(
            f"- [{a.agent_role}, score={a.reward_score:.2f}] {a.position}"
            for a in top_args
        )

        prompt = (
            f"Synthesise the best debate arguments into a final decision.\n\n"
            f"## Topic: {topic}\n"
            f"## Top Arguments (ranked by quality)\n{args_text}\n\n"
            f"## Debate ended: {convergence_reason} (after {rounds} rounds)\n\n"
            f"Create a synthesis that combines the strongest elements. "
            f"Output JSON:\n"
            f'{{"winner_position": "...", "synthesis": "...", "confidence": 0.X}}'
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You synthesise debate outcomes into actionable decisions.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            data = _extract_json(text)
            if data:
                return DebateOutcome(
                    topic=topic,
                    winner_position=data.get("winner_position", top_args[0].position if top_args else ""),
                    synthesis=data.get("synthesis", ""),
                    confidence=float(data.get("confidence", 0.7)),
                    rounds=rounds,
                    arguments=all_arguments,
                    convergence_reason=convergence_reason,
                )

        except Exception as e:
            logger.warning(f"[Debate] Synthesis failed: {e}")

        # Fallback: use the highest-scored argument
        best = top_args[0] if top_args else DebateArgument(agent_role="", position="")
        return DebateOutcome(
            topic=topic,
            winner_position=best.position,
            synthesis=best.reasoning,
            confidence=0.5,
            rounds=rounds,
            arguments=all_arguments,
            convergence_reason=convergence_reason,
        )

    # ──────── Statistics ────────

    def get_debate_stats(self) -> dict[str, Any]:
        """Get statistics about past debates."""
        if not self._debate_history:
            return {"total_debates": 0}

        triggered = [d for d in self._debate_history if d.triggered]
        return {
            "total_debates": len(self._debate_history),
            "triggered": len(triggered),
            "skipped": len(self._debate_history) - len(triggered),
            "avg_rounds": (
                sum(d.rounds for d in triggered) / len(triggered)
                if triggered else 0
            ),
            "avg_confidence": (
                sum(d.confidence for d in self._debate_history)
                / len(self._debate_history)
            ),
            "convergence_reasons": {
                reason: sum(1 for d in self._debate_history if d.convergence_reason == reason)
                for reason in set(d.convergence_reason for d in self._debate_history)
            },
            "algorithm_ratio": self._algorithm_ratio,
            "elo_ranker_ratings": self._elo_ranker._ratings,
        }
