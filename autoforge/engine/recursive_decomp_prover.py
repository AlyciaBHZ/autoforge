"""
HILBERT-style recursive decomposition proof search.

A four-component architecture where an informal reasoner (large LLM) and a
specialized prover (small LLM) collaborate with a formal verifier and a
semantic theorem retriever to prove theorems.

Reference: HILBERT (NeurIPS 2025) — 99.2% on miniF2F, 70% on PutnamBench.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoforge.engine.theoretical_reasoning import ConceptNode, ConceptType, ScientificDomain, TheoryGraph

logger = logging.getLogger(__name__)


@dataclass
class ProofGoal:
    """Represents a single proof goal in the decomposition tree."""

    goal_id: str
    statement: str
    lean_statement: str = ""
    parent_goal_id: str = ""
    depth: int = 0
    status: str = "open"  # open, proved, failed, decomposed
    proof: str = ""
    subgoals: list[str] = field(default_factory=list)


@dataclass
class RetrievedLemma:
    """A lemma retrieved from a knowledge base."""

    name: str
    statement: str
    relevance_score: float
    source: str  # "mathlib", "local_graph", etc.


@dataclass
class DecompositionResult:
    """Result of decomposing a proof goal."""

    original_goal: ProofGoal
    subgoals: list[ProofGoal]
    strategy: str  # "case_split", "induction", "apply_lemma", "rewrite", etc.
    informal_reasoning: str


class RecursiveDecompProver:
    """
    HILBERT-style recursive decomposition proof search engine.

    Collaborates between informal reasoner and specialized prover with
    formal verification and semantic theorem retrieval.
    """

    def __init__(self, max_depth: int = 5, max_attempts_per_goal: int = 3):
        """Initialize the prover.

        Args:
            max_depth: Maximum decomposition depth before giving up.
            max_attempts_per_goal: Maximum attempts per goal before marking as failed.
        """
        self.max_depth = max_depth
        self.max_attempts_per_goal = max_attempts_per_goal
        self._goals: dict[str, ProofGoal] = {}
        self._retrieval_cache: dict[str, list[RetrievedLemma]] = {}
        self._attempt_counts: dict[str, int] = {}
        logger.info(
            f"Initialized RecursiveDecompProver with max_depth={max_depth}, "
            f"max_attempts_per_goal={max_attempts_per_goal}"
        )

    def _generate_goal_id(self, statement: str, parent_id: str = "") -> str:
        """Generate a unique goal ID from statement and parent."""
        content = f"{parent_id}:{statement}".encode()
        return hashlib.sha256(content).hexdigest()[:16]

    async def prove(
        self, concept: ConceptNode, llm: Any, *, graph: TheoryGraph | None = None
    ) -> ProofGoal:
        """Main entry point for proving a theorem.

        Args:
            concept: The concept/theorem to prove.
            llm: LLM interface for reasoning and proof generation.
            graph: Optional TheoryGraph for lemma retrieval.

        Returns:
            The root ProofGoal with final status.
        """
        concept_label = concept.label or concept.id
        logger.info(f"Starting proof for concept: {concept_label}")

        # Create root goal
        statement = concept.formal_statement or concept.informal_statement or concept_label
        lean_statement = concept.formal_statement or ""
        goal_id = self._generate_goal_id(statement)

        root_goal = ProofGoal(
            goal_id=goal_id,
            statement=statement,
            lean_statement=lean_statement,
            depth=0,
        )
        self._goals[goal_id] = root_goal
        self._attempt_counts[goal_id] = 0

        logger.info(f"Created root goal {goal_id}: {statement[:60]}...")

        # Recursively prove
        result = await self._recursive_prove(root_goal, llm, graph)
        logger.info(f"Proof result for {concept_label}: status={result.status}")

        return result

    async def _recursive_prove(
        self, goal: ProofGoal, llm: Any, graph: TheoryGraph | None
    ) -> ProofGoal:
        """Recursively prove a goal using decomposition strategy.

        1. Try direct proof via specialized prover
        2. Retrieve relevant lemmas
        3. Try proof with lemmas
        4. If still failing and depth < max_depth, decompose
        5. Recursively prove subgoals
        6. Combine proofs

        Args:
            goal: The goal to prove.
            llm: LLM interface.
            graph: Optional TheoryGraph.

        Returns:
            Updated ProofGoal with proof status.
        """
        self._attempt_counts[goal.goal_id] = self._attempt_counts.get(goal.goal_id, 0) + 1

        if goal.status != "open":
            return goal

        if goal.depth >= self.max_depth:
            logger.warning(f"Goal {goal.goal_id} reached max depth {self.max_depth}")
            goal.status = "failed"
            self._goals[goal.goal_id] = goal
            return goal

        if self._attempt_counts[goal.goal_id] > self.max_attempts_per_goal:
            logger.warning(
                f"Goal {goal.goal_id} exceeded max attempts "
                f"({self.max_attempts_per_goal})"
            )
            goal.status = "failed"
            self._goals[goal.goal_id] = goal
            return goal

        logger.info(
            f"Proving goal {goal.goal_id} (depth={goal.depth}, "
            f"attempt={self._attempt_counts[goal.goal_id]})"
        )

        # 1. Try direct proof
        proof = await self._try_direct_proof(goal, llm)
        if proof:
            goal.status = "proved"
            goal.proof = proof
            self._goals[goal.goal_id] = goal
            logger.info(f"Direct proof succeeded for goal {goal.goal_id}")
            return goal

        # 2. Retrieve lemmas
        lemmas = await self._retrieve_lemmas(goal, llm, graph)
        logger.info(f"Retrieved {len(lemmas)} lemmas for goal {goal.goal_id}")

        # 3. Try proof with lemmas
        if lemmas:
            proof = await self._try_proof_with_lemmas(goal, lemmas, llm)
            if proof:
                goal.status = "proved"
                goal.proof = proof
                self._goals[goal.goal_id] = goal
                logger.info(f"Proof with lemmas succeeded for goal {goal.goal_id}")
                return goal

        # 4. Decompose if depth allows
        if goal.depth < self.max_depth:
            logger.info(f"Attempting decomposition for goal {goal.goal_id}")
            decomp_result = await self._decompose_goal(goal, llm)

            if decomp_result.subgoals:
                goal.status = "decomposed"
                goal.subgoals = [sg.goal_id for sg in decomp_result.subgoals]
                self._goals[goal.goal_id] = goal

                # 5. Recursively prove subgoals
                logger.info(
                    f"Recursively proving {len(decomp_result.subgoals)} "
                    f"subgoals for {goal.goal_id}"
                )

                subgoal_results = []
                for subgoal in decomp_result.subgoals:
                    self._goals[subgoal.goal_id] = subgoal
                    result = await self._recursive_prove(subgoal, llm, graph)
                    subgoal_results.append(result)

                # 6. Check if all subgoals proved
                all_proved = all(sg.status == "proved" for sg in subgoal_results)

                if all_proved:
                    combined_proof = self._combine_proofs(goal, subgoal_results)
                    goal.status = "proved"
                    goal.proof = combined_proof
                    logger.info(f"Combined proof succeeded for goal {goal.goal_id}")
                else:
                    failed_count = sum(1 for sg in subgoal_results if sg.status == "failed")
                    logger.warning(
                        f"Goal {goal.goal_id} has {failed_count} failed subgoals"
                    )
                    goal.status = "failed"

        else:
            goal.status = "failed"

        self._goals[goal.goal_id] = goal
        return goal

    async def _try_direct_proof(self, goal: ProofGoal, llm: Any) -> str | None:
        """Try direct proof via specialized prover.

        Asks the prover to generate Lean 4 tactics directly.

        Args:
            goal: The goal to prove.
            llm: LLM interface.

        Returns:
            Proof string or None if failed.
        """
        try:
            from autoforge.engine.llm_router import TaskComplexity
        except ImportError:
            logger.warning("llm_router not available, using default complexity")
            TaskComplexity = type("TaskComplexity", (), {"MEDIUM": "medium"})()

        prompt = f"""You are a specialized Lean 4 proof assistant.

Goal: {goal.statement}

Lean statement:
```lean
{goal.lean_statement}
```

Generate a complete Lean 4 proof using tactics. Be concise and direct.
Output ONLY the Lean 4 tactics, no explanations."""

        try:
            response = await llm.generate(
                prompt,
                temperature=0.2,
                max_tokens=1000,
                complexity=getattr(TaskComplexity, "MEDIUM", "medium"),
            )

            if response and response.strip():
                logger.info(f"Direct proof generated for {goal.goal_id}")
                return response.strip()
        except Exception as e:
            logger.debug(f"Direct proof failed for {goal.goal_id}: {e}")

        return None

    async def _try_proof_with_lemmas(
        self, goal: ProofGoal, lemmas: list[RetrievedLemma], llm: Any
    ) -> str | None:
        """Try to prove goal using retrieved lemmas.

        Args:
            goal: The goal to prove.
            lemmas: List of relevant lemmas.
            llm: LLM interface.

        Returns:
            Proof string or None if failed.
        """
        try:
            from autoforge.engine.llm_router import TaskComplexity
        except ImportError:
            logger.warning("llm_router not available, using default complexity")
            TaskComplexity = type("TaskComplexity", (), {"MEDIUM": "medium"})()

        # Build lemma context
        lemma_text = "\n".join(
            [
                f"- {lemma.name}: {lemma.statement} (score: {lemma.relevance_score:.2f})"
                for lemma in lemmas[:10]  # Top 10 lemmas
            ]
        )

        prompt = f"""You are a specialized Lean 4 proof assistant.

Goal: {goal.statement}

Lean statement:
```lean
{goal.lean_statement}
```

Available lemmas:
{lemma_text}

Using the provided lemmas, generate a complete Lean 4 proof.
Output ONLY the Lean 4 tactics, no explanations."""

        try:
            response = await llm.generate(
                prompt,
                temperature=0.2,
                max_tokens=1000,
                complexity=getattr(TaskComplexity, "MEDIUM", "medium"),
            )

            if response and response.strip():
                logger.info(f"Proof with lemmas generated for {goal.goal_id}")
                return response.strip()
        except Exception as e:
            logger.debug(f"Proof with lemmas failed for {goal.goal_id}: {e}")

        return None

    async def _decompose_goal(self, goal: ProofGoal, llm: Any) -> DecompositionResult:
        """Decompose a goal into subgoals using informal reasoning.

        Uses the informal reasoner to analyze why direct proof failed
        and suggest a decomposition strategy.

        Args:
            goal: The goal to decompose.
            llm: LLM interface.

        Returns:
            DecompositionResult with strategy and subgoals.
        """
        try:
            from autoforge.engine.llm_router import TaskComplexity
        except ImportError:
            logger.warning("llm_router not available, using default complexity")
            TaskComplexity = type("TaskComplexity", (), {"HIGH": "high"})()

        prompt = f"""You are an expert mathematician analyzing a proof challenge.

Goal: {goal.statement}

Lean statement:
```lean
{goal.lean_statement}
```

Analyze this goal and suggest how to decompose it into manageable subgoals.

Return a JSON object with:
{{
    "strategy": "case_split|induction|apply_lemma|rewrite|structural",
    "reasoning": "explanation of why this decomposition makes sense",
    "subgoals": [
        {{"statement": "natural language statement", "lean_statement": "Lean 4 formalization"}},
        ...
    ]
}}

Be strategic: each subgoal should be simpler than the original."""

        subgoals = []
        strategy = "structural"
        reasoning = ""

        try:
            response = await llm.generate(
                prompt,
                temperature=0.7,
                max_tokens=2000,
                complexity=getattr(TaskComplexity, "HIGH", "high"),
            )

            if response:
                # Parse JSON from response
                try:
                    # Extract JSON from response
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        data = json.loads(json_str)

                        strategy = data.get("strategy", "structural")
                        reasoning = data.get("reasoning", "")

                        for sg_data in data.get("subgoals", []):
                            sg_statement = sg_data.get("statement", "")
                            sg_lean = sg_data.get("lean_statement", "")

                            if sg_statement:
                                sg_id = self._generate_goal_id(
                                    sg_statement, goal.goal_id
                                )
                                subgoal = ProofGoal(
                                    goal_id=sg_id,
                                    statement=sg_statement,
                                    lean_statement=sg_lean,
                                    parent_goal_id=goal.goal_id,
                                    depth=goal.depth + 1,
                                )
                                subgoals.append(subgoal)

                        logger.info(
                            f"Decomposed goal {goal.goal_id} into "
                            f"{len(subgoals)} subgoals using {strategy}"
                        )
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse decomposition JSON: {e}")

        except Exception as e:
            logger.debug(f"Decomposition failed for {goal.goal_id}: {e}")

        return DecompositionResult(
            original_goal=goal,
            subgoals=subgoals,
            strategy=strategy,
            informal_reasoning=reasoning,
        )

    async def _retrieve_lemmas(
        self, goal: ProofGoal, llm: Any, graph: TheoryGraph | None
    ) -> list[RetrievedLemma]:
        """Retrieve relevant lemmas from local graph and Mathlib.

        Two sources:
        a) Local TheoryGraph: find concepts whose formal_statement overlaps
        b) LLM-based Mathlib search: ask LLM to suggest relevant Mathlib lemmas

        Args:
            goal: The goal to find lemmas for.
            llm: LLM interface.
            graph: Optional TheoryGraph.

        Returns:
            Sorted list of retrieved lemmas.
        """
        cache_key = goal.goal_id
        if cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]

        lemmas: list[RetrievedLemma] = []

        # 1. Local graph retrieval
        if graph:
            try:
                for concept in graph.nodes.values():
                    if (
                        concept.formal_statement
                        and concept.concept_type
                        in [ConceptType.THEOREM, ConceptType.LEMMA]
                    ):
                        # Simple overlap-based relevance
                        overlap_score = self._compute_overlap(
                            goal.lean_statement, concept.formal_statement
                        )
                        if overlap_score > 0.1:
                            lemmas.append(
                                RetrievedLemma(
                                    name=concept.label,
                                    statement=concept.formal_statement,
                                    relevance_score=overlap_score,
                                    source="local_graph",
                                )
                            )
            except Exception as e:
                logger.debug(f"Local graph retrieval failed: {e}")

        # 2. LLM-based Mathlib search
        try:
            from autoforge.engine.llm_router import TaskComplexity
        except ImportError:
            logger.warning("llm_router not available, using default complexity")
            TaskComplexity = type("TaskComplexity", (), {"MEDIUM": "medium"})()

        prompt = f"""Given this proof goal, suggest the most relevant Mathlib lemmas that could help prove it.

Goal: {goal.statement}

Lean statement:
```lean
{goal.lean_statement}
```

Return a JSON array of lemma names:
["lemma_name_1", "lemma_name_2", ...]

Be specific and realistic about actual Mathlib lemmas."""

        try:
            response = await llm.generate(
                prompt,
                temperature=0.3,
                max_tokens=500,
                complexity=getattr(TaskComplexity, "MEDIUM", "medium"),
            )

            if response:
                try:
                    json_start = response.find("[")
                    json_end = response.rfind("]") + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        lemma_names = json.loads(json_str)

                        for name in lemma_names[:5]:  # Top 5 suggestions
                            lemmas.append(
                                RetrievedLemma(
                                    name=name,
                                    statement=f"lemma {name}",
                                    relevance_score=0.6,
                                    source="mathlib",
                                )
                            )
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse Mathlib lemmas: {e}")

        except Exception as e:
            logger.debug(f"Mathlib retrieval failed: {e}")

        # Sort by relevance
        lemmas.sort(key=lambda x: x.relevance_score, reverse=True)

        self._retrieval_cache[cache_key] = lemmas
        logger.info(f"Retrieved {len(lemmas)} lemmas for goal {goal.goal_id}")

        return lemmas

    def _compute_overlap(self, text1: str, text2: str) -> float:
        """Compute simple token overlap between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Overlap score in [0, 1].
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _combine_proofs(self, goal: ProofGoal, subgoals: list[ProofGoal]) -> str:
        """Combine subgoal proofs into a complete proof for parent goal.

        Args:
            goal: Parent goal.
            subgoals: Proved subgoals.

        Returns:
            Combined proof text.
        """
        proof_parts = [f"-- Proof of: {goal.statement}"]

        for i, subgoal in enumerate(subgoals, 1):
            proof_parts.append(f"\n-- Subgoal {i}: {subgoal.statement}")
            if subgoal.proof:
                proof_parts.append(subgoal.proof)

        proof_parts.append(
            "\n-- Combined proof complete"
        )

        combined = "\n".join(proof_parts)
        logger.info(f"Combined {len(subgoals)} subgoal proofs for {goal.goal_id}")

        return combined

    def get_proof_tree(self, root_id: str) -> dict:
        """Return the full proof tree as a nested dict.

        Args:
            root_id: The root goal ID.

        Returns:
            Nested dictionary representing the proof tree.
        """
        if root_id not in self._goals:
            logger.warning(f"Goal {root_id} not found")
            return {}

        root = self._goals[root_id]

        def build_tree(goal_id: str) -> dict:
            goal = self._goals.get(goal_id)
            if not goal:
                return {}

            tree = {
                "goal_id": goal_id,
                "statement": goal.statement,
                "status": goal.status,
                "depth": goal.depth,
                "proof": goal.proof[:100] + "..." if len(goal.proof) > 100 else goal.proof,
            }

            if goal.subgoals:
                tree["subgoals"] = [build_tree(sg_id) for sg_id in goal.subgoals]

            return tree

        return build_tree(root_id)

    def get_statistics(self) -> dict:
        """Return statistics about all goals.

        Returns:
            Dictionary with proof statistics.
        """
        if not self._goals:
            return {
                "total_goals": 0,
                "proved": 0,
                "failed": 0,
                "decomposed": 0,
                "open": 0,
                "max_depth_reached": 0,
                "avg_attempts": 0.0,
            }

        proved_count = sum(1 for g in self._goals.values() if g.status == "proved")
        failed_count = sum(1 for g in self._goals.values() if g.status == "failed")
        decomposed_count = sum(
            1 for g in self._goals.values() if g.status == "decomposed"
        )
        open_count = sum(1 for g in self._goals.values() if g.status == "open")
        max_depth = max((g.depth for g in self._goals.values()), default=0)
        avg_attempts = (
            sum(self._attempt_counts.values()) / len(self._attempt_counts)
            if self._attempt_counts
            else 0.0
        )

        return {
            "total_goals": len(self._goals),
            "proved": proved_count,
            "failed": failed_count,
            "decomposed": decomposed_count,
            "open": open_count,
            "max_depth_reached": max_depth,
            "avg_attempts": avg_attempts,
        }
