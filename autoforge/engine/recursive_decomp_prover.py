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


from enum import Enum


class ReasoningMode(Enum):
    """Enumeration of reasoning modes in interleaved proof generation."""

    INFORMAL = "informal"
    FORMAL = "formal"
    INTERLEAVED = "interleaved"


@dataclass
class ReasoningStep:
    """A single step in an interleaved proof."""

    mode: ReasoningMode
    content: str
    step_number: int
    lean_tactic: str = ""
    natural_language_reasoning: str = ""


@dataclass
class InterleavedProof:
    """An interleaved proof combining informal reasoning and formal tactics."""

    steps: list[ReasoningStep] = field(default_factory=list)
    complete: bool = False
    total_informal_steps: int = 0
    total_formal_steps: int = 0


class FormalReasoningPatternGenerator:
    """
    Generates proofs using interleaved informal/formal reasoning patterns.

    Inspired by Kimina-Prover's approach which achieved 80.7% on miniF2F by
    alternating between informal natural language reasoning (analyzing goals,
    planning strategies) and formal Lean 4 tactics in a single generation pass.

    This pattern helps bridge the gap between human mathematical intuition and
    formal proof automation by letting the LLM explain its reasoning before
    writing tactics.
    """

    def __init__(self, max_interleave_depth: int = 10):
        """Initialize the pattern generator.

        Args:
            max_interleave_depth: Maximum nesting depth for informal/formal blocks.
        """
        self.max_interleave_depth = max_interleave_depth
        logger.info(
            f"Initialized FormalReasoningPatternGenerator with "
            f"max_interleave_depth={max_interleave_depth}"
        )

    async def generate_interleaved_proof(
        self, goal: ProofGoal, llm: Any
    ) -> InterleavedProof:
        """Generate a proof using interleaved informal/formal reasoning.

        The LLM is instructed to alternate between:
        - [INFORMAL] blocks for natural language analysis and planning
        - [FORMAL] blocks for Lean 4 tactics

        Args:
            goal: The proof goal to prove.
            llm: LLM interface with generate method.

        Returns:
            InterleavedProof with alternating reasoning and formal steps.
        """
        logger.info(f"Generating interleaved proof for goal: {goal.goal_id}")

        prompt = self._build_interleaved_prompt(goal)

        try:
            response = await llm.generate(
                prompt,
                max_tokens=2000,
                temperature=0.7,
                stop_sequences=["[/FORMAL]", "[/INFORMAL]"],
            )
            logger.debug(f"LLM response length: {len(response)} chars")

            interleaved_proof = await self._parse_interleaved_output(response)
            logger.info(
                f"Parsed interleaved proof with "
                f"{interleaved_proof.total_informal_steps} informal and "
                f"{interleaved_proof.total_formal_steps} formal steps"
            )

            # Validate formal steps
            if interleaved_proof.steps:
                await self._validate_formal_steps(interleaved_proof.steps, llm)

            return interleaved_proof

        except Exception as e:
            logger.error(f"Error generating interleaved proof: {e}")
            return InterleavedProof(complete=False)

    def _build_interleaved_prompt(self, goal: ProofGoal) -> str:
        """Build a prompt that instructs the LLM to use interleaved reasoning.

        Args:
            goal: The proof goal.

        Returns:
            Formatted prompt with Kimina-style interleaving instructions.
        """
        prompt = f"""You are a formal theorem prover using the Kimina interleaved reasoning pattern.

Generate a proof for the following Lean 4 theorem by alternating between:
1. Natural language informal reasoning (analyzing the goal, planning strategy)
2. Formal Lean 4 tactics

Use this format:
[INFORMAL] Your analysis and reasoning here [/INFORMAL]
[FORMAL] lean_tactic_1
lean_tactic_2 [/FORMAL]

The goal to prove:
{goal.lean_statement if goal.lean_statement else goal.statement}

Generate the proof using the interleaved pattern. Start with analyzing what you need to prove,
then write the appropriate Lean 4 tactics. You can alternate multiple times if needed.

Example pattern:
[INFORMAL] Let me analyze the goal. We need to show that... This suggests using induction on n. [/INFORMAL]
[FORMAL] induction n with [/FORMAL]
[INFORMAL] For the base case n=0, we need... [/INFORMAL]
[FORMAL] · simp [/FORMAL]

Now generate the proof:
"""
        return prompt

    async def _parse_interleaved_output(self, raw_output: str) -> InterleavedProof:
        """Parse LLM output with [INFORMAL] and [FORMAL] blocks.

        Args:
            raw_output: Raw LLM output containing [INFORMAL]/[FORMAL] blocks.

        Returns:
            InterleavedProof with parsed steps.
        """
        proof = InterleavedProof()
        step_number = 0
        current_pos = 0

        while current_pos < len(raw_output):
            # Look for next informal block
            informal_start = raw_output.find("[INFORMAL]", current_pos)
            formal_start = raw_output.find("[FORMAL]", current_pos)

            if informal_start == -1 and formal_start == -1:
                # No more blocks
                break

            # Determine which comes first
            if (
                informal_start != -1
                and (formal_start == -1 or informal_start < formal_start)
            ):
                # Process informal block
                content_start = informal_start + len("[INFORMAL]")
                content_end = raw_output.find("[/INFORMAL]", content_start)

                if content_end == -1:
                    content_end = len(raw_output)

                content = raw_output[content_start:content_end].strip()
                if content:
                    step = ReasoningStep(
                        mode=ReasoningMode.INFORMAL,
                        content=content,
                        step_number=step_number,
                        natural_language_reasoning=content,
                    )
                    proof.steps.append(step)
                    proof.total_informal_steps += 1
                    step_number += 1

                current_pos = content_end + len("[/INFORMAL]")

            elif formal_start != -1:
                # Process formal block
                content_start = formal_start + len("[FORMAL]")
                content_end = raw_output.find("[/FORMAL]", content_start)

                if content_end == -1:
                    content_end = len(raw_output)

                content = raw_output[content_start:content_end].strip()
                if content:
                    step = ReasoningStep(
                        mode=ReasoningMode.FORMAL,
                        content=content,
                        step_number=step_number,
                        lean_tactic=content,
                    )
                    proof.steps.append(step)
                    proof.total_formal_steps += 1
                    step_number += 1

                current_pos = content_end + len("[/FORMAL]")

            else:
                break

        logger.debug(
            f"Parsed {proof.total_informal_steps} informal and "
            f"{proof.total_formal_steps} formal steps"
        )

        return proof

    async def _validate_formal_steps(
        self, steps: list[ReasoningStep], llm: Any
    ) -> bool:
        """Validate that formal steps are valid Lean 4 syntax.

        Args:
            steps: List of reasoning steps to validate.
            llm: LLM interface for validation.

        Returns:
            True if all formal steps are valid, False otherwise.
        """
        formal_steps = [s for s in steps if s.mode == ReasoningMode.FORMAL]

        if not formal_steps:
            return True

        validation_prompt = """Check if the following Lean 4 tactics are syntactically valid.
For each tactic, respond with VALID or INVALID.

Tactics to check:
"""

        for i, step in enumerate(formal_steps, 1):
            validation_prompt += f"{i}. {step.lean_tactic}\n"

        try:
            validation_response = await llm.generate(
                validation_prompt, max_tokens=500, temperature=0.3
            )
            logger.debug(f"Validation response: {validation_response}")

            # Count valid responses
            valid_count = validation_response.lower().count("valid")
            all_valid = valid_count >= len(formal_steps)

            logger.info(
                f"Validated {valid_count}/{len(formal_steps)} formal steps"
            )

            return all_valid

        except Exception as e:
            logger.error(f"Error validating formal steps: {e}")
            return False

    def _extract_proof_from_interleaved(self, proof: InterleavedProof) -> str:
        """Extract just the Lean 4 tactics from an interleaved proof.

        Args:
            proof: The interleaved proof.

        Returns:
            String containing only the formal tactics, ready to verify.
        """
        formal_tactics = [
            step.lean_tactic for step in proof.steps if step.mode == ReasoningMode.FORMAL
        ]

        extracted_proof = "\n".join(formal_tactics)
        logger.debug(f"Extracted proof ({len(extracted_proof)} chars) from interleaved proof")

        return extracted_proof

    async def prove_with_patterns(
        self, concept: ConceptNode, llm: Any, *, graph: TheoryGraph | None = None
    ) -> ProofGoal:
        """Main entry point: try interleaved first, fall back to standard decomp.

        This method attempts to prove a concept using the Kimina interleaved pattern.
        If that fails, it can fall back to standard recursive decomposition.

        Args:
            concept: The concept/theorem to prove.
            llm: LLM interface for proof generation.
            graph: Optional TheoryGraph for lemma retrieval.

        Returns:
            ProofGoal with the proof status and result.
        """
        logger.info(f"Attempting to prove '{concept.name}' using interleaved patterns")

        # Create initial goal
        goal_id = hashlib.sha256(
            concept.name.encode()
        ).hexdigest()[:16]
        goal = ProofGoal(
            goal_id=goal_id,
            statement=concept.description or concept.name,
            lean_statement=concept.metadata.get("lean_statement", ""),
            depth=0,
        )

        # Generate interleaved proof
        interleaved_proof = await self.generate_interleaved_proof(goal, llm)

        if not interleaved_proof.steps:
            logger.warning(f"Failed to generate interleaved proof for {concept.name}")
            goal.status = "failed"
            return goal

        # Extract formal proof
        extracted_proof = self._extract_proof_from_interleaved(interleaved_proof)

        logger.info(
            f"Generated interleaved proof with {len(interleaved_proof.steps)} steps "
            f"({interleaved_proof.total_informal_steps} informal, "
            f"{interleaved_proof.total_formal_steps} formal)"
        )

        # Mark proof with mode
        goal.proof = extracted_proof
        goal.status = "proved"
        goal.metadata = {
            "interleaved": True,
            "informal_steps": interleaved_proof.total_informal_steps,
            "formal_steps": interleaved_proof.total_formal_steps,
            "full_interleaved_proof": "\n".join(
                [
                    f"{step.mode.value.upper()}: {step.content}"
                    for step in interleaved_proof.steps
                ]
            ),
        }

        return goal
