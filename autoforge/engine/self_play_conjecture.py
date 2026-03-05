"""
Self-Play Conjecturing Engine

A dual-agent system where a Conjecturer generates statements of calibrated difficulty
and a Prover attempts to prove/disprove them. The two agents improve each other iteratively.

Reference: STP (ICML 2025) — 28.5% on LeanWorkbook (2x previous SOTA).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

from autoforge.engine.theoretical_reasoning import (
    ConceptNode,
    ConceptType,
    ScientificDomain,
    TheoryGraph,
)

logger = logging.getLogger(__name__)


class DifficultyLevel(str, Enum):
    """Enumeration of difficulty levels for conjectures."""

    TRIVIAL = "trivial"        # Direct corollary, one step
    EASY = "easy"              # Short proof, 2-3 steps
    MEDIUM = "medium"          # Non-trivial, needs a key insight
    HARD = "hard"              # Multi-step, combines techniques
    RESEARCH = "research"      # Open problem level


@dataclass
class ConjectureAttempt:
    """Records a single conjecture and proof attempt."""

    conjecture: ConceptNode
    difficulty_target: DifficultyLevel
    difficulty_actual: DifficultyLevel = DifficultyLevel.MEDIUM
    proved: bool = False
    proof_sketch: str = ""
    attempts: int = 0
    prover_feedback: str = ""
    round_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conjecture": {
                "id": self.conjecture.id,
                "label": self.conjecture.label,
                "description": self.conjecture.description,
            },
            "difficulty_target": self.difficulty_target.value,
            "difficulty_actual": self.difficulty_actual.value,
            "proved": self.proved,
            "proof_sketch": self.proof_sketch,
            "attempts": self.attempts,
            "prover_feedback": self.prover_feedback,
            "round_number": self.round_number,
        }


class DifficultyCalibrator:
    """Calibrates difficulty levels to maintain ~50% success rate."""

    def __init__(self) -> None:
        """Initialize the difficulty calibrator."""
        self._history: list[ConjectureAttempt] = []
        self._sweet_spot_rate: dict[str, float] = {
            level.value: 0.5 for level in DifficultyLevel
        }

    def record(self, attempt: ConjectureAttempt) -> None:
        """
        Record a proof attempt and update success rates.

        Args:
            attempt: The conjecture attempt to record.
        """
        self._history.append(attempt)
        self._update_rates()

    def _update_rates(self) -> None:
        """Update success rates using Bayesian Beta distribution."""
        for level in DifficultyLevel:
            attempts_at_level = [
                a for a in self._history if a.difficulty_target == level
            ]
            if not attempts_at_level:
                continue

            successes = sum(1 for a in attempts_at_level if a.proved)
            failures = len(attempts_at_level) - successes

            # Beta distribution: Beta(successes+1, failures+1)
            # Expected value = (successes + 1) / (successes + failures + 2)
            expected_value = (successes + 1) / (len(attempts_at_level) + 2)
            self._sweet_spot_rate[level.value] = expected_value

    def get_target_difficulty(self) -> DifficultyLevel:
        """
        Get the difficulty level closest to 50% success rate.

        Returns:
            The difficulty level with expected value closest to 0.5.
        """
        closest_level = DifficultyLevel.MEDIUM
        closest_distance = 1.0

        for level in DifficultyLevel:
            rate = self._sweet_spot_rate[level.value]
            distance = abs(rate - 0.5)
            if distance < closest_distance:
                closest_distance = distance
                closest_level = level

        logger.debug(
            f"Selected difficulty {closest_level.value} "
            f"(rate: {self._sweet_spot_rate[closest_level.value]:.2%})"
        )
        return closest_level

    def get_statistics(self) -> dict[str, Any]:
        """
        Get calibrator statistics.

        Returns:
            Dictionary containing success rates and history.
        """
        return {
            "sweet_spot_rates": dict(self._sweet_spot_rate),
            "total_attempts": len(self._history),
            "proved_count": sum(1 for a in self._history if a.proved),
            "failed_count": sum(1 for a in self._history if not a.proved),
            "success_rate": sum(1 for a in self._history if a.proved)
            / len(self._history)
            if self._history
            else 0.0,
        }


class SelfPlayEngine:
    """Self-play conjecturing engine with difficulty calibration."""

    def __init__(self, max_rounds: int = 10, conjectures_per_round: int = 3) -> None:
        """
        Initialize the self-play engine.

        Args:
            max_rounds: Maximum number of self-play rounds.
            conjectures_per_round: Number of conjectures per round.
        """
        self._max_rounds = max_rounds
        self._conjectures_per_round = conjectures_per_round
        self._calibrator = DifficultyCalibrator()
        self._proved_conjectures: list[ConjectureAttempt] = []
        self._failed_conjectures: list[ConjectureAttempt] = []
        self._round_log: list[dict[str, Any]] = []

    async def run(
        self,
        graph: TheoryGraph,
        llm: Any,
        *,
        output_dir: Path | None = None,
    ) -> list[ConjectureAttempt]:
        """
        Run the self-play conjecturing loop.

        Args:
            graph: The theory graph to extend.
            llm: The LLM instance for generation and proving.
            output_dir: Optional directory to save results.

        Returns:
            List of all successfully proved conjectures.
        """
        logger.info(
            f"Starting self-play conjecturing for {self._max_rounds} rounds, "
            f"{self._conjectures_per_round} conjectures/round"
        )

        for round_num in range(self._max_rounds):
            logger.info(f"=== Round {round_num + 1}/{self._max_rounds} ===")

            # Get target difficulty from calibrator
            target_difficulty = self._calibrator.get_target_difficulty()
            logger.info(f"Target difficulty: {target_difficulty.value}")

            # Generate conjectures
            try:
                conjectures = await self._generate_conjectures(
                    graph, llm, target_difficulty
                )
                logger.info(f"Generated {len(conjectures)} conjectures")
            except Exception as e:
                logger.error(f"Error generating conjectures: {e}", exc_info=True)
                continue

            # Attempt to prove each conjecture
            round_proved = 0
            for conjecture in conjectures:
                try:
                    result = await self._attempt_proof(conjecture, llm, graph)
                    result.round_number = round_num + 1

                    # Record with calibrator
                    self._calibrator.record(result)

                    if result.proved:
                        self._proved_conjectures.append(result)
                        round_proved += 1
                        logger.info(
                            f"✓ Proved: {conjecture.label} "
                            f"(actual difficulty: {result.difficulty_actual.value})"
                        )

                        # Add to graph as new theorem
                        await self._add_to_graph(result, graph)
                    else:
                        self._failed_conjectures.append(result)
                        logger.info(
                            f"✗ Failed: {conjecture.label} "
                            f"(feedback: {result.prover_feedback[:100]}...)"
                        )

                except Exception as e:
                    logger.error(
                        f"Error attempting proof for {conjecture.label}: {e}",
                        exc_info=True,
                    )

            # Log round statistics
            stats = self._calibrator.get_statistics()
            round_stats = {
                "round": round_num + 1,
                "target_difficulty": target_difficulty.value,
                "conjectures_generated": len(conjectures),
                "proved_this_round": round_proved,
                "calibrator_stats": stats,
                "timestamp": time.time(),
            }
            self._round_log.append(round_stats)
            logger.info(
                f"Round {round_num + 1} summary: {round_proved}/{len(conjectures)} "
                f"proved, overall success rate: {stats['success_rate']:.2%}"
            )

        # Save results if output directory provided
        if output_dir:
            await self._save_results(output_dir)

        logger.info(
            f"Self-play complete: {len(self._proved_conjectures)} proved, "
            f"{len(self._failed_conjectures)} failed"
        )
        return self._proved_conjectures

    async def _generate_conjectures(
        self, graph: TheoryGraph, llm: Any, difficulty: DifficultyLevel
    ) -> list[ConceptNode]:
        """
        Generate conjectures at target difficulty.

        Args:
            graph: The theory graph.
            llm: The LLM instance.
            difficulty: Target difficulty level.

        Returns:
            List of generated conjecture nodes.
        """
        time_estimate = self._difficulty_to_time(difficulty)

        # Build context from graph
        frontier_concepts = self._get_frontier_concepts(graph)
        frontier_text = "\n".join(
            [
                f"- {c.label}: {c.description}"
                for c in frontier_concepts[:10]  # Limit to 10
            ]
        )

        prompt = f"""You are a mathematical conjecture generator. Generate {self._conjectures_per_round} novel conjectures
at the following difficulty level:

DIFFICULTY: {difficulty.value}
TIME ESTIMATE: A strong graduate student could prove this in approximately {time_estimate}.

Current theory frontier (key concepts to build upon):
{frontier_text}

Requirements:
1. Each conjecture should be mathematically sound and interesting.
2. Each should be provable in approximately {time_estimate} for a strong mathematician.
3. Include diverse areas - don't repeat the same technique.
4. Build on the frontier concepts provided.

Output format: Return a JSON array with objects containing:
- "title": Brief title of the conjecture
- "statement": Formal mathematical statement
- "reason": Why this is interesting given the frontier
- "hint": A key insight that might help prove it

Example output format:
[
  {{"title": "...", "statement": "...", "reason": "...", "hint": "..."}},
  ...
]

Generate the conjectures now:"""

        try:
            # Call LLM (simplified - in real implementation would use proper routing)
            response = await llm.generate(prompt, max_tokens=2000)

            # Parse JSON response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in conjecture generation response")
                return []

            conjectures_json = json.loads(json_match.group())
            conjectures = []

            for i, conj in enumerate(conjectures_json):
                node = ConceptNode(
                    id=f"conj_{int(time.time())}_{i}",
                    formal_statement=conj.get("statement", ""),
                    informal_statement=conj.get("reason", ""),
                    concept_type=ConceptType.CONJECTURE,
                    domain=ScientificDomain.PURE_MATHEMATICS,
                    metadata={
                        "legacy_label": conj.get("title", f"Conjecture {i}"),
                        "reason": conj.get("reason", ""),
                        "hint": conj.get("hint", ""),
                        "difficulty_target": difficulty.value,
                    },
                )
                conjectures.append(node)

            logger.info(f"Generated {len(conjectures)} conjectures")
            return conjectures

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse conjecture JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error generating conjectures: {e}", exc_info=True)
            return []

    async def _attempt_proof(
        self, conjecture: ConceptNode, llm: Any, graph: TheoryGraph
    ) -> ConjectureAttempt:
        """
        Attempt to prove a conjecture using the Prover agent.

        Args:
            conjecture: The conjecture to prove.
            llm: The LLM instance.
            graph: The theory graph.

        Returns:
            ConjectureAttempt with proof status and details.
        """
        attempt = ConjectureAttempt(
            conjecture=conjecture,
            difficulty_target=DifficultyLevel(
                conjecture.metadata.get("difficulty_target", DifficultyLevel.MEDIUM.value)
            ),
        )

        # Attempt 1: Direct proof
        prompt_attempt_1 = f"""You are a mathematician tasked with proving the following conjecture:

STATEMENT: {conjecture.description}

CONTEXT: {conjecture.metadata.get("reason", "")}
KEY INSIGHT: {conjecture.metadata.get("hint", "")}

Available theorems and definitions from the theory:
{self._get_available_theorems(graph, limit=20)}

Attempt to prove or disprove this conjecture. Provide:
1. Your proof strategy
2. Step-by-step reasoning
3. Key lemmas or theorems used
4. Final conclusion (PROVED, DISPROVED, or UNKNOWN)

Be rigorous but concise."""

        try:
            response_1 = await llm.generate(prompt_attempt_1, max_tokens=1500)
            attempt.attempts = 1
            attempt.proof_sketch = response_1

            # Parse result
            if self._parse_proof_result(response_1, attempt):
                attempt.difficulty_actual = self._assess_difficulty(response_1)
                return attempt

            # Attempt 2: Alternative approach after failure
            logger.info(
                f"First proof attempt failed for {conjecture.label}, trying alternative"
            )

            prompt_attempt_2 = f"""Your first proof attempt for the following conjecture was unsuccessful:

STATEMENT: {conjecture.description}

Your previous attempt: {response_1[:500]}...

Please try a different approach. Consider:
1. Working backwards from the conclusion
2. Using proof by contradiction
3. Breaking into cases
4. Using algebraic or combinatorial manipulation

Provide your alternative proof attempt:"""

            response_2 = await llm.generate(prompt_attempt_2, max_tokens=1500)
            attempt.attempts = 2
            attempt.prover_feedback = response_1[:200]
            attempt.proof_sketch = response_2

            if self._parse_proof_result(response_2, attempt):
                attempt.difficulty_actual = self._assess_difficulty(response_2)
                return attempt

            # Mark as failed if both attempts unsuccessful
            attempt.proved = False
            attempt.difficulty_actual = self._assess_difficulty(response_1 + response_2)
            attempt.prover_feedback = "Both proof attempts unsuccessful"

        except Exception as e:
            logger.error(f"Error in proof attempt: {e}", exc_info=True)
            attempt.proved = False
            attempt.prover_feedback = f"Error during proof: {str(e)}"

        return attempt

    def _parse_proof_result(
        self, response: str, attempt: ConjectureAttempt
    ) -> bool:
        """
        Parse LLM response to determine if proof was successful.

        Args:
            response: The LLM response text.
            attempt: The ConjectureAttempt to update.

        Returns:
            True if proof appears successful.
        """
        response_lower = response.lower()

        # Check for explicit proof status
        if "proved" in response_lower and "disproved" not in response_lower:
            attempt.proved = True
            return True
        elif "disproved" in response_lower or "contradiction" in response_lower:
            attempt.proved = False
            attempt.prover_feedback = "Conjecture disproved"
            return True
        elif "unknown" in response_lower:
            attempt.proved = False
            attempt.prover_feedback = "Proof status unknown"
            return True

        # Check for proof structure indicators
        proof_indicators = [
            "therefore",
            "thus",
            "hence",
            "by lemma",
            "by theorem",
            "it follows",
            "q.e.d",
        ]
        if any(indicator in response_lower for indicator in proof_indicators):
            attempt.proved = True
            return True

        return False

    def _assess_difficulty(self, proof: str) -> DifficultyLevel:
        """
        Assess actual difficulty based on proof characteristics.

        Args:
            proof: The proof text.

        Returns:
            Estimated difficulty level.
        """
        # Count key indicators
        step_count = len(re.findall(r'step\s+\d+|line\s+\d+|thus|therefore', proof, re.IGNORECASE))
        technique_count = len(
            re.findall(
                r'induction|contradiction|case analysis|lemma|theorem|substitution|'
                r'algebraic|combinatorial|calculus|differential',
                proof,
                re.IGNORECASE,
            )
        )

        total_complexity = step_count + (technique_count * 2)

        if total_complexity < 3:
            return DifficultyLevel.TRIVIAL
        elif total_complexity < 7:
            return DifficultyLevel.EASY
        elif total_complexity < 15:
            return DifficultyLevel.MEDIUM
        elif total_complexity < 30:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.RESEARCH

    def _difficulty_to_time(self, level: DifficultyLevel) -> str:
        """
        Map difficulty level to human-readable time estimate.

        Args:
            level: The difficulty level.

        Returns:
            Human-readable time estimate.
        """
        time_map = {
            DifficultyLevel.TRIVIAL: "5 minutes",
            DifficultyLevel.EASY: "30 minutes",
            DifficultyLevel.MEDIUM: "2-3 hours",
            DifficultyLevel.HARD: "1-2 days",
            DifficultyLevel.RESEARCH: "weeks to months",
        }
        return time_map.get(level, "unknown")

    def _get_frontier_concepts(self, graph: TheoryGraph, limit: int = 10) -> list[ConceptNode]:
        """
        Get frontier concepts from the theory graph.

        Args:
            graph: The theory graph.
            limit: Maximum number of concepts to return.

        Returns:
            List of frontier concepts.
        """
        try:
            # In a real implementation, this would traverse the graph
            # to find the most recent/advanced concepts
            frontier = list(graph.nodes.values())[:limit]
            return frontier
        except Exception as e:
            logger.warning(f"Error getting frontier concepts: {e}")
            return []

    def _get_available_theorems(self, graph: TheoryGraph, limit: int = 20) -> str:
        """
        Get available theorems from the theory graph as text.

        Args:
            graph: The theory graph.
            limit: Maximum number of theorems to include.

        Returns:
            Text representation of available theorems.
        """
        try:
            theorems = [
                node
                for node in graph.nodes.values()
                if node.concept_type == ConceptType.THEOREM
            ]
            theorem_text = "\n".join(
                [f"- {t.label}: {t.description}" for t in theorems[:limit]]
            )
            return theorem_text if theorem_text else "No theorems available"
        except Exception as e:
            logger.warning(f"Error getting available theorems: {e}")
            return "Theorems unavailable"

    async def _add_to_graph(
        self, attempt: ConjectureAttempt, graph: TheoryGraph
    ) -> None:
        """
        Add a proved conjecture to the theory graph.

        Args:
            attempt: The successful proof attempt.
            graph: The theory graph to update.
        """
        try:
            # Create new theorem node
            theorem = ConceptNode(
                id=f"thm_{attempt.conjecture.id}",
                formal_statement=attempt.conjecture.formal_statement,
                informal_statement=attempt.conjecture.informal_statement,
                concept_type=ConceptType.THEOREM,
                domain=attempt.conjecture.domain,
                metadata={
                    "legacy_label": f"Theorem: {attempt.conjecture.label}",
                    "proof_sketch": attempt.proof_sketch,
                    "derived_from_conjecture": attempt.conjecture.id,
                    "difficulty": attempt.difficulty_actual.value,
                },
            )

            # Add to graph (implementation depends on TheoryGraph API)
            if hasattr(graph, "add_node"):
                graph.add_node(theorem)
            elif hasattr(graph, "nodes"):
                graph.nodes[theorem.id] = theorem

            logger.info(f"Added theorem to graph: {theorem.label}")

        except Exception as e:
            logger.error(f"Error adding theorem to graph: {e}", exc_info=True)

    async def _save_results(self, output_dir: Path) -> None:
        """
        Save results to disk.

        Args:
            output_dir: Directory to save results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save proved conjectures
            proved_data = [c.to_dict() for c in self._proved_conjectures]
            with open(output_dir / "proved_conjectures.json", "w") as f:
                json.dump(proved_data, f, indent=2)

            # Save failed conjectures
            failed_data = [c.to_dict() for c in self._failed_conjectures]
            with open(output_dir / "failed_conjectures.json", "w") as f:
                json.dump(failed_data, f, indent=2)

            # Save round log
            with open(output_dir / "round_log.json", "w") as f:
                json.dump(self._round_log, f, indent=2)

            # Save calibrator statistics
            stats = self._calibrator.get_statistics()
            with open(output_dir / "calibrator_stats.json", "w") as f:
                json.dump(stats, f, indent=2)

            logger.info(f"Results saved to {output_dir}")

        except Exception as e:
            logger.error(f"Error saving results: {e}", exc_info=True)

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary containing summary statistics.
        """
        stats = self._calibrator.get_statistics()
        return {
            "total_rounds": len(self._round_log),
            "proved_conjectures": len(self._proved_conjectures),
            "failed_conjectures": len(self._failed_conjectures),
            "overall_success_rate": stats["success_rate"],
            "calibrator_stats": stats,
            "round_log": self._round_log,
        }
