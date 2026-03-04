"""Autonomous Discovery Engine — Self-Growing Theorem Extension from Minimal Kernels.

This module implements the autonomous reasoning extension capability:
given a paper's axiomatic/definitional kernel (TheoryGraph), the engine
iteratively discovers new theorems, conjectures, and cross-domain connections
that grow *outward* from the kernel — never repeating known results, always
aiming for publishable-quality depth.

Design principles (from user requirements):
  1. Each discovery round must produce deep, publishable conclusions.
  2. Growth proceeds from a minimal kernel — definitions, axioms, key theorems.
  3. No "toothpaste squeezing": stop early if only shallow results remain.
  4. Never repeat the paper's own results or publicly known theorems.
  5. Output uses top-tier mathematical journal academic language.
  6. Every new conclusion gets a sequential global ID (AD-001, AD-002, ...).

Architecture:
  PaperKernel          — Extract minimal axiom/definition/theorem seed from a TheoryGraph
  NoveltyFilter        — Reject results that overlap with known literature
  ConjectureGenerator  — LLM-guided conjecture synthesis from frontier nodes
  DepthEvaluator       — Score whether a result is "deep enough" to publish
  DiscoveryOrchestrator — Multi-round autonomous loop with termination criteria

Integration:
  - TheoryGraph from theoretical_reasoning.py provides the knowledge substrate
  - VerificationSuite provides multi-modal confidence scoring
  - LLM Router provides the reasoning backbone (Opus for deep, Sonnet for routine)

References:
  - FunSearch (Nature 2024): LLM-guided evolutionary program discovery
  - AlphaProof (DeepMind 2025): MCTS-guided formal mathematical reasoning
  - Ramanujan Machine (Nature 2021): numerical pattern → symbolic conjecture pipeline
  - AI Scientist v2 (2025): autonomous research cycle with novelty filtering
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from autoforge.engine.theoretical_reasoning import (
    ConceptNode,
    ConceptRelation,
    ConceptType,
    ReasoningStrategy,
    RelationType,
    ScientificDomain,
    TheoryGraph,
    VerificationSuite,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════


class DiscoveryDepth(str, Enum):
    """Required depth for a discovery to be accepted."""
    SHALLOW = "shallow"          # Corollary / simple specialization
    MODERATE = "moderate"        # Non-trivial extension or new connection
    DEEP = "deep"                # Novel theorem with proof, publishable
    BREAKTHROUGH = "breakthrough" # Surprising structural insight


@dataclass
class DiscoveryConfig:
    """Configuration for the autonomous discovery engine."""
    # Rounds
    max_rounds: int = 20
    min_results_per_round: int = 1
    max_results_per_round: int = 5

    # Quality gates
    min_depth: DiscoveryDepth = DiscoveryDepth.MODERATE
    min_confidence: float = 0.6
    novelty_threshold: float = 0.7       # Min novelty score to accept
    depth_score_threshold: float = 0.65   # Min depth score to accept

    # Termination
    max_consecutive_shallow_rounds: int = 3
    max_total_results: int = 100

    # Strategies to employ (subset of ReasoningStrategy)
    strategies: list[str] = field(default_factory=lambda: [
        "generalization", "composition", "analogy_transfer",
        "boundary_analysis", "duality", "unification",
        "numerical_exploration", "dimensional_lifting",
    ])

    # Output format
    journal_style: str = "top_math"   # top_math | applied_math | physics
    language: str = "en"               # en | zh (for bilingual output)


# ══════════════════════════════════════════════════════════════
# Paper Kernel Extractor
# ══════════════════════════════════════════════════════════════


class PaperKernel:
    """Extract the minimal axiom/definition/theorem seed from a paper's TheoryGraph.

    The kernel consists of:
      - All definitions (the language of the theory)
      - All axioms/assumptions
      - Core theorems that other results depend on
      - Key structural analogies (cross-domain bridges)

    Everything else (corollaries, remarks, examples) is excluded — the discovery
    engine should be able to re-derive them and go further.
    """

    def __init__(self, graph: TheoryGraph) -> None:
        self.graph = graph
        self._kernel_ids: set[str] = set()
        self._frontier_ids: set[str] = set()
        self._known_statements: list[str] = []

    def extract(self) -> tuple[list[ConceptNode], list[ConceptNode]]:
        """Extract kernel nodes and frontier nodes.

        Returns:
            (kernel_nodes, frontier_nodes)
            - kernel: definitions, axioms, foundational theorems
            - frontier: leaf theorems/corollaries — starting points for extension
        """
        kernel: list[ConceptNode] = []
        frontier: list[ConceptNode] = []

        # Definitions and axioms always go in kernel
        for node in self.graph._nodes.values():
            if node.concept_type in (ConceptType.DEFINITION, ConceptType.AXIOM,
                                      ConceptType.CONSTRUCTION):
                kernel.append(node)
                self._kernel_ids.add(node.id)
                self._known_statements.append(node.formal_statement)

        # Theorems with many dependents → kernel (foundational)
        dep_count: dict[str, int] = {}
        for rel in self.graph._relations:
            if rel.relation_type == RelationType.DEPENDS_ON:
                dep_count[rel.target_id] = dep_count.get(rel.target_id, 0) + 1

        for node in self.graph._nodes.values():
            if node.concept_type in (ConceptType.THEOREM, ConceptType.PROPOSITION,
                                      ConceptType.LEMMA):
                self._known_statements.append(node.formal_statement)
                if dep_count.get(node.id, 0) >= 2:
                    kernel.append(node)
                    self._kernel_ids.add(node.id)

        # Frontier: leaf nodes not in kernel
        leaf_nodes = self.graph.get_frontier()
        for node in leaf_nodes:
            if node.id not in self._kernel_ids:
                frontier.append(node)
                self._frontier_ids.add(node.id)

        # Also add cross-domain bridges to kernel
        for src, tgt, rel in self.graph.get_cross_domain_bridges():
            if src.id not in self._kernel_ids:
                kernel.append(src)
                self._kernel_ids.add(src.id)
            if tgt.id not in self._kernel_ids:
                kernel.append(tgt)
                self._kernel_ids.add(tgt.id)

        logger.info(
            f"[PaperKernel] Extracted {len(kernel)} kernel nodes, "
            f"{len(frontier)} frontier nodes from '{self.graph.title}'"
        )
        return kernel, frontier

    @property
    def known_statements(self) -> list[str]:
        """All formal statements from the original paper (used for novelty filtering)."""
        return self._known_statements


# ══════════════════════════════════════════════════════════════
# Novelty Filter
# ══════════════════════════════════════════════════════════════


class NoveltyFilter:
    """Reject discoveries that overlap with known results.

    Uses:
      1. Exact/near-duplicate detection against paper's own statements
      2. Structural similarity scoring via LLM
      3. Literature cross-reference check via LLM

    A result must score above novelty_threshold on ALL checks to pass.
    """

    def __init__(self, known_statements: list[str], threshold: float = 0.7) -> None:
        self._known: list[str] = known_statements
        self._discovered: list[str] = []   # Accumulates during session
        self.threshold = threshold

    async def is_novel(
        self,
        candidate: ConceptNode,
        llm: Any,
    ) -> tuple[bool, float, str]:
        """Check if a candidate result is novel.

        Returns:
            (is_novel, novelty_score, reason)
        """
        from autoforge.engine.llm_router import TaskComplexity

        # 1. Quick syntactic overlap check
        stmt = candidate.formal_statement.strip()
        for known in self._known + self._discovered:
            overlap = self._jaccard_similarity(stmt, known)
            if overlap > 0.6:
                return False, 1.0 - overlap, f"High syntactic overlap ({overlap:.2f}) with known result"

        # 2. LLM semantic novelty check
        known_summary = "\n".join(
            f"  [{i+1}] {s[:200]}" for i, s in enumerate(self._known[-30:])
        )
        discovered_summary = "\n".join(
            f"  [D{i+1}] {s[:200]}" for i, s in enumerate(self._discovered[-20:])
        )

        prompt = f"""You are a referee for a top mathematics journal. Evaluate whether the following
candidate result is NOVEL — i.e., it is NOT a trivial restatement, direct corollary, or already-known
result from the given paper or standard literature.

CANDIDATE RESULT:
  Type: {candidate.concept_type.value}
  Domain: {candidate.domain.value}
  Statement: {candidate.formal_statement}
  Proof sketch: {candidate.proof_sketch[:500] if candidate.proof_sketch else 'N/A'}

KNOWN RESULTS FROM THIS PAPER:
{known_summary}

PREVIOUSLY DISCOVERED (this session):
{discovered_summary}

Score the novelty from 0.0 to 1.0 where:
  0.0 = exact duplicate or trivial restatement
  0.3 = minor variation / obvious corollary
  0.5 = modest extension with limited new insight
  0.7 = non-trivial new result with genuine content
  0.9 = surprising connection or deep structural insight
  1.0 = potential breakthrough

Respond in JSON:
{{"novelty_score": <float>, "is_known": <bool>, "reason": "<1-2 sentences>", "closest_known": "<which known result is closest, if any>"}}
"""
        try:
            resp = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            data = json.loads(
                re.search(r'\{[^{}]*\}', resp.content, re.DOTALL).group()
            )
            score = float(data.get("novelty_score", 0.5))
            reason = data.get("reason", "")
            is_known = data.get("is_known", False)

            novel = score >= self.threshold and not is_known
            return novel, score, reason

        except Exception as e:
            logger.warning(f"[NoveltyFilter] LLM check failed: {e}")
            return True, 0.5, "Novelty check inconclusive (LLM error)"

    def register_discovery(self, statement: str) -> None:
        """Register a newly discovered result to prevent self-duplication."""
        self._discovered.append(statement)

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Token-level Jaccard similarity."""
        tokens_a = set(re.findall(r'\w+', a.lower()))
        tokens_b = set(re.findall(r'\w+', b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)


# ══════════════════════════════════════════════════════════════
# Depth Evaluator
# ══════════════════════════════════════════════════════════════


class DepthEvaluator:
    """Score whether a discovered result is 'deep enough' to be publishable.

    Criteria:
      - Technical complexity (proof length, tool diversity)
      - Structural novelty (new connections between concepts)
      - Potential impact (opens new directions vs. dead end)
      - Elegance (simplicity of statement relative to depth of proof)
    """

    async def evaluate(
        self,
        candidate: ConceptNode,
        kernel_context: str,
        llm: Any,
    ) -> tuple[DiscoveryDepth, float, str]:
        """Evaluate depth of a candidate discovery.

        Returns:
            (depth_level, score, justification)
        """
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""You are a senior editor at Annals of Mathematics evaluating a candidate result
for publication depth. This result extends the theory of superspace model sets,
finite-resolution readout, and information-theoretic time.

CANDIDATE:
  Type: {candidate.concept_type.value}
  Statement: {candidate.formal_statement}
  Proof sketch: {candidate.proof_sketch[:800] if candidate.proof_sketch else 'N/A'}
  Intuition: {candidate.intuition[:400] if candidate.intuition else 'N/A'}

THEORY CONTEXT (kernel):
{kernel_context[:2000]}

Rate this result on four axes (each 0.0-1.0):
  1. technical_complexity: Does the proof require non-trivial techniques?
  2. structural_novelty: Does it reveal new structural connections?
  3. potential_impact: Does it open significant new research directions?
  4. elegance: Is the statement clean relative to proof depth?

Also classify the overall depth:
  - "shallow": Routine corollary or specialization
  - "moderate": Non-trivial but expected extension
  - "deep": Novel theorem with genuine insight, publishable
  - "breakthrough": Surprising structural result that reshapes understanding

Respond in JSON:
{{"technical_complexity": <float>, "structural_novelty": <float>,
  "potential_impact": <float>, "elegance": <float>,
  "overall_depth": "<shallow|moderate|deep|breakthrough>",
  "overall_score": <float>, "justification": "<2-3 sentences>"}}
"""
        try:
            resp = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            data = json.loads(
                re.search(r'\{[^{}]*\}', resp.content, re.DOTALL).group()
            )
            depth = DiscoveryDepth(data.get("overall_depth", "moderate"))
            score = float(data.get("overall_score", 0.5))
            justification = data.get("justification", "")
            return depth, score, justification

        except Exception as e:
            logger.warning(f"[DepthEvaluator] Failed: {e}")
            return DiscoveryDepth.MODERATE, 0.5, "Depth evaluation inconclusive"


# ══════════════════════════════════════════════════════════════
# Conjecture Generator
# ══════════════════════════════════════════════════════════════


class ConjectureGenerator:
    """Generate new conjectures and theorems from frontier nodes.

    Strategies:
      1. Generalization: weaken hypotheses of a frontier theorem
      2. Composition: combine two frontier results into one
      3. Analogy transfer: lift a result to a different domain
      4. Boundary analysis: investigate limiting/degenerate cases
      5. Duality: apply known duality transforms
      6. Unification: find common structure across multiple results
      7. Numerical exploration: compute patterns → conjecture
      8. Dimensional lifting: extend from d to d+1 dimensions

    Each strategy produces a structured ConceptNode with:
      - Formal statement in LaTeX
      - Proof sketch (or "conjecture" marker)
      - Intuitive explanation
      - Connection to parent frontier nodes
    """

    # Strategy → specialized prompt template
    _STRATEGY_PROMPTS: dict[str, str] = {
        "generalization": """Given the following theorem from the theory of superspace
model sets and information-theoretic time, find a STRICT GENERALIZATION — weaken
one or more hypotheses while preserving a meaningful (possibly weakened) conclusion.

The generalization must be NON-TRIVIAL: it should require genuine mathematical work
to prove, not just removing an unused hypothesis.

PARENT THEOREM:
{parent_statement}

PROOF SKETCH OF PARENT:
{parent_proof}

THEORY CONTEXT:
{context}

Produce a generalized result. Use rigorous mathematical language suitable for
Inventiones Mathematicae or Annals of Mathematics. Provide:
1. A precise formal statement (LaTeX notation)
2. A proof sketch (at least the key ideas and techniques needed)
3. An intuitive explanation of WHY the generalization works
4. What new cases or applications the generalization covers""",

        "composition": """Given two results from the theory of superspace model sets
and information-theoretic time, find a NON-TRIVIAL COMPOSITION that combines them
into a single deeper result.

The composition should not be a mere conjunction (A and B); it should produce
a genuinely new insight that neither result alone implies.

RESULT A:
{parent_statement}

RESULT B:
{second_statement}

THEORY CONTEXT:
{context}

Produce a composed result. Use rigorous mathematical language. Provide:
1. A precise formal statement (LaTeX notation)
2. A proof sketch showing HOW A and B combine
3. An intuitive explanation of the new insight
4. Why this composition is non-trivial""",

        "analogy_transfer": """Given the following result from the theory of superspace
model sets, identify a STRUCTURAL ANALOGY with another mathematical domain and
TRANSFER the result to produce a new theorem in that domain.

TARGET DOMAINS for transfer (pick the most natural one):
  - Operator algebras / quantum information
  - Ergodic theory / dynamical systems
  - Coding theory / information theory
  - Algebraic topology / homological algebra
  - Number theory / arithmetic geometry
  - Statistical mechanics / phase transitions

PARENT RESULT:
{parent_statement}

THEORY CONTEXT:
{context}

Produce an analogous result in the target domain. Provide:
1. The precise structural analogy (what maps to what)
2. The formal statement of the transferred result (LaTeX)
3. A proof sketch (or explanation why the analogy works formally)
4. What new insight the transfer provides in the target domain""",

        "boundary_analysis": """Given the following result from the theory of superspace
model sets, investigate its BOUNDARY BEHAVIOR — what happens at extreme parameter
values, degenerate cases, or critical transitions.

PARENT RESULT:
{parent_statement}

THEORY CONTEXT:
{context}

Investigate at least one of:
  - What happens as m → ∞ (infinite resolution limit)?
  - What happens as m → 1 or m → 2 (minimal resolution)?
  - What happens when the golden ratio φ is replaced by other algebraic numbers?
  - What happens at the boundary of the acceptance window W?
  - What happens when the lattice Γ degenerates?

Produce a boundary result. Provide:
1. The precise limiting/degenerate regime
2. A formal statement of the boundary behavior (LaTeX)
3. A proof sketch
4. Physical or mathematical interpretation""",

        "duality": """Given the following result, identify and apply a DUALITY
transformation to produce a dual theorem.

Known dualities in this theory:
  - Resolution ↔ time depth (Remark 4.2: lost DOF recovered as temporal context)
  - Visible entropy H(π) ↔ fiber degeneracy E_π[log d_m(X)] (KL-ledger identity)
  - Synchronization budget ↔ collision budget (capacity law)
  - Physical space E ↔ internal space H (CPS duality)

PARENT RESULT:
{parent_statement}

THEORY CONTEXT:
{context}

Apply a duality to produce the DUAL statement. Provide:
1. Which duality you are applying and why
2. The formal dual statement (LaTeX)
3. A proof that the duality indeed exchanges the two sides
4. New insight from the dual perspective""",

        "unification": """Given the following results from the theory of superspace
model sets, find a UNIFYING PRINCIPLE that explains all of them as special cases
of a deeper structural pattern.

RESULTS TO UNIFY:
{parent_statement}

THEORY CONTEXT:
{context}

Produce a unifying theorem or framework. Provide:
1. The unifying principle (formal statement in LaTeX)
2. How each input result is recovered as a special case
3. A proof sketch of the unifying result
4. What new predictions the unification makes beyond the known cases""",

        "numerical_exploration": """Given the following result involving computable
quantities from the theory of superspace model sets, design a NUMERICAL EXPERIMENT
to discover NEW PATTERNS that extend beyond what is currently proved.

PARENT RESULT:
{parent_statement}

Key computable quantities in this theory:
  - Gauge anomaly density δ_m = E[G_m]/m → 4/9
  - Gauge anomaly variance σ² = 118/243
  - Fiber sizes |F_m(x)| for x ∈ X_m
  - Scan error ε_m(P; μ) decay rates
  - Boundary cylinder counts N_m(∂P)
  - Entropy rate h_μ = log φ for golden-mean Parry measure
  - Refinement functional τ(t) = -log μ(C(a_{0:t-1}))

THEORY CONTEXT:
{context}

Design a numerical investigation and CONJECTURE a new result. Provide:
1. The computational experiment (what to compute, parameter ranges)
2. The observed numerical pattern (describe what you expect)
3. A precise CONJECTURE based on the pattern (formal LaTeX)
4. Why the conjecture is plausible (heuristic argument or connection to theory)""",

        "dimensional_lifting": """Given the following result in the theory of 1D
superspace readout (d=1), lift it to HIGHER DIMENSIONS (d≥2).

PARENT RESULT (1D):
{parent_statement}

THEORY CONTEXT:
{context}

Challenges of higher-dimensional lifting:
  - The golden-mean language becomes a higher-dimensional SFT
  - Zeckendorf stabilization generalizes to multi-dimensional carry rules
  - Boundary cylinder dimension depends on window geometry in R^n
  - The Sturmian coding becomes a multidimensional symbolic dynamics

Produce a d-dimensional extension. Provide:
1. The precise d-dimensional generalization (LaTeX)
2. What changes and what remains from the 1D case
3. A proof sketch or argument for the generalization
4. New phenomena that appear only in d ≥ 2""",
    }

    async def generate(
        self,
        strategy: str,
        frontier_nodes: list[ConceptNode],
        kernel_context: str,
        llm: Any,
        *,
        round_number: int = 1,
        existing_count: int = 0,
    ) -> list[ConceptNode]:
        """Generate candidate discoveries using a specific strategy.

        Returns list of ConceptNode candidates (not yet validated).
        """
        from autoforge.engine.llm_router import TaskComplexity

        if strategy not in self._STRATEGY_PROMPTS:
            logger.warning(f"[ConjectureGenerator] Unknown strategy: {strategy}")
            return []

        template = self._STRATEGY_PROMPTS[strategy]

        # Select parent nodes based on strategy
        if strategy == "composition" and len(frontier_nodes) >= 2:
            parent = frontier_nodes[0]
            second = frontier_nodes[1]
            prompt = template.format(
                parent_statement=parent.formal_statement,
                second_statement=second.formal_statement,
                parent_proof=parent.proof_sketch[:500] if parent.proof_sketch else "N/A",
                context=kernel_context[:2000],
            )
        elif strategy == "unification" and len(frontier_nodes) >= 2:
            combined = "\n\n".join(
                f"Result {i+1}: {n.formal_statement}"
                for i, n in enumerate(frontier_nodes[:4])
            )
            prompt = template.format(
                parent_statement=combined,
                context=kernel_context[:2000],
            )
        else:
            parent = frontier_nodes[0] if frontier_nodes else None
            if parent is None:
                return []
            prompt = template.format(
                parent_statement=parent.formal_statement,
                parent_proof=parent.proof_sketch[:500] if parent.proof_sketch else "N/A",
                context=kernel_context[:2000],
            )

        # Add output format instructions
        next_id = existing_count + 1
        prompt += f"""

CRITICAL FORMATTING REQUIREMENTS:
- Number this discovery AD-{next_id:03d}
- Use LaTeX notation for all mathematics
- Write in the formal academic style of Inventiones Mathematicae
- The statement must be PRECISE and FALSIFIABLE
- No vague or hand-wavy claims

Respond in JSON:
{{"id": "AD-{next_id:03d}",
  "concept_type": "<theorem|proposition|conjecture|corollary|lemma>",
  "domain": "<pure_mathematics|applied_mathematics|information_theory|mathematical_physics|statistical_mechanics>",
  "formal_statement": "<LaTeX statement>",
  "proof_sketch": "<detailed proof sketch or 'Conjecture — proof open'>",
  "intuition": "<1-2 sentences: why this is true and why it matters>",
  "tags": ["<relevant tags>"],
  "parent_ids": ["<ids of frontier nodes used>"],
  "generation_strategy": "{strategy}"}}
"""

        try:
            resp = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            data = json.loads(
                re.search(r'\{[^{}]*\}', resp.content, re.DOTALL).group()
            )

            node = ConceptNode(
                id=data.get("id", f"AD-{next_id:03d}"),
                concept_type=ConceptType(data.get("concept_type", "theorem")),
                domain=ScientificDomain(data.get("domain", "pure_mathematics")),
                formal_statement=data.get("formal_statement", ""),
                proof_sketch=data.get("proof_sketch", ""),
                intuition=data.get("intuition", ""),
                tags=data.get("tags", []),
                parent_ids=data.get("parent_ids", []),
                generation_strategy=strategy,
                source_article="autonomous_discovery",
            )
            return [node]

        except Exception as e:
            logger.warning(f"[ConjectureGenerator] {strategy} failed: {e}")
            return []


# ══════════════════════════════════════════════════════════════
# Discovery Orchestrator
# ══════════════════════════════════════════════════════════════


@dataclass
class DiscoveryResult:
    """A single accepted discovery with metadata."""
    node: ConceptNode
    novelty_score: float
    depth_score: float
    depth_level: DiscoveryDepth
    round_number: int
    strategy: str
    verification_results: dict[str, float] = field(default_factory=dict)


class DiscoveryOrchestrator:
    """Multi-round autonomous discovery loop.

    Each round:
      1. Select a strategy (round-robin with bias toward successful ones)
      2. Select frontier nodes as seeds
      3. Generate candidate discoveries
      4. Filter for novelty
      5. Evaluate depth
      6. Verify via VerificationSuite
      7. Accept or reject

    Termination when:
      - max_rounds reached
      - max_total_results reached
      - consecutive shallow rounds exceed threshold
      - no novel results possible (all strategies exhausted)
    """

    def __init__(
        self,
        config: DiscoveryConfig | None = None,
    ) -> None:
        self.config = config or DiscoveryConfig()
        self._generator = ConjectureGenerator()
        self._evaluator = DepthEvaluator()
        self._verifier = VerificationSuite()
        self._results: list[DiscoveryResult] = []
        self._strategy_scores: dict[str, list[float]] = {}
        self._round_log: list[dict[str, Any]] = []

    async def run(
        self,
        graph: TheoryGraph,
        llm: Any,
        *,
        output_dir: Path | None = None,
    ) -> list[DiscoveryResult]:
        """Execute the autonomous discovery loop.

        Args:
            graph: TheoryGraph containing the paper's theory
            llm: LLM router for reasoning calls
            output_dir: Optional directory to save results incrementally

        Returns:
            List of accepted discoveries
        """
        logger.info(f"[Discovery] Starting autonomous discovery on '{graph.title}'")

        # Extract kernel and frontier
        kernel_extractor = PaperKernel(graph)
        kernel_nodes, frontier_nodes = kernel_extractor.extract()

        # Build kernel context string
        kernel_context = self._build_kernel_context(kernel_nodes)

        # Initialize novelty filter
        novelty_filter = NoveltyFilter(
            kernel_extractor.known_statements,
            threshold=self.config.novelty_threshold,
        )

        consecutive_shallow = 0
        strategies = list(self.config.strategies)

        for round_num in range(1, self.config.max_rounds + 1):
            if len(self._results) >= self.config.max_total_results:
                logger.info(f"[Discovery] Reached max results ({self.config.max_total_results})")
                break

            # Select strategy
            strategy = self._select_strategy(strategies, round_num)
            logger.info(
                f"[Discovery] Round {round_num}/{self.config.max_rounds} "
                f"— strategy: {strategy}, results so far: {len(self._results)}"
            )

            # Select frontier seeds (rotate through frontiers)
            seeds = self._select_seeds(frontier_nodes, strategy, round_num)
            if not seeds:
                logger.info("[Discovery] No seeds available, stopping")
                break

            # Generate candidates
            round_accepted: list[DiscoveryResult] = []
            candidates = await self._generator.generate(
                strategy=strategy,
                frontier_nodes=seeds,
                kernel_context=kernel_context,
                llm=llm,
                round_number=round_num,
                existing_count=len(self._results),
            )

            for candidate in candidates:
                # Novelty check
                is_novel, novelty_score, novelty_reason = await novelty_filter.is_novel(
                    candidate, llm
                )
                if not is_novel:
                    logger.debug(
                        f"[Discovery] Rejected (not novel): {candidate.id} — {novelty_reason}"
                    )
                    continue

                # Depth evaluation
                depth, depth_score, depth_reason = await self._evaluator.evaluate(
                    candidate, kernel_context, llm
                )
                if depth_score < self.config.depth_score_threshold:
                    logger.debug(
                        f"[Discovery] Rejected (too shallow): {candidate.id} "
                        f"— {depth.value} ({depth_score:.2f})"
                    )
                    continue

                # Verification
                verification = await self._verifier.verify(candidate, llm)

                if candidate.overall_confidence >= self.config.min_confidence:
                    result = DiscoveryResult(
                        node=candidate,
                        novelty_score=novelty_score,
                        depth_score=depth_score,
                        depth_level=depth,
                        round_number=round_num,
                        strategy=strategy,
                        verification_results=verification,
                    )
                    round_accepted.append(result)
                    self._results.append(result)
                    novelty_filter.register_discovery(candidate.formal_statement)

                    # Add to graph for future rounds
                    graph.add_concept(candidate)
                    for pid in candidate.parent_ids:
                        if pid in graph._nodes:
                            graph.add_relation(ConceptRelation(
                                source_id=pid,
                                target_id=candidate.id,
                                relation_type=RelationType.MOTIVATES,
                                description=f"Extended via {strategy}",
                            ))

                    # Update frontier
                    frontier_nodes.append(candidate)

                    logger.info(
                        f"[Discovery] ✓ Accepted AD-{len(self._results):03d}: "
                        f"{depth.value} ({depth_score:.2f}), "
                        f"novelty={novelty_score:.2f}, "
                        f"confidence={candidate.overall_confidence:.2f}"
                    )

            # Track strategy success
            self._strategy_scores.setdefault(strategy, []).append(
                len(round_accepted) / max(len(candidates), 1)
            )

            # Log round
            self._round_log.append({
                "round": round_num,
                "strategy": strategy,
                "candidates": len(candidates),
                "accepted": len(round_accepted),
                "total_results": len(self._results),
            })

            # Termination check: consecutive shallow rounds
            if not round_accepted:
                consecutive_shallow += 1
                if consecutive_shallow >= self.config.max_consecutive_shallow_rounds:
                    logger.info(
                        f"[Discovery] {consecutive_shallow} consecutive empty rounds, stopping"
                    )
                    break
            else:
                consecutive_shallow = 0

            # Save incremental results
            if output_dir:
                await self._save_results(output_dir)

        logger.info(
            f"[Discovery] Complete: {len(self._results)} discoveries "
            f"in {len(self._round_log)} rounds"
        )

        # Final save
        if output_dir:
            await self._save_results(output_dir)

        return self._results

    def _build_kernel_context(self, kernel_nodes: list[ConceptNode]) -> str:
        """Build a compact string representation of the kernel for prompts."""
        lines = ["THEORY KERNEL:"]
        for node in kernel_nodes[:30]:  # Cap at 30 to fit context
            lines.append(
                f"  [{node.concept_type.value.upper()}] {node.id}: "
                f"{node.formal_statement[:300]}"
            )
        return "\n".join(lines)

    def _select_strategy(self, strategies: list[str], round_num: int) -> str:
        """Select strategy with Thompson Sampling bias toward successful ones."""
        if round_num <= len(strategies):
            # First pass: try each strategy once
            return strategies[(round_num - 1) % len(strategies)]

        # Thompson Sampling: sample from Beta(successes+1, failures+1)
        import random
        best_score = -1.0
        best_strategy = strategies[0]
        for s in strategies:
            scores = self._strategy_scores.get(s, [])
            if not scores:
                sample = random.betavariate(1, 1)
            else:
                successes = sum(1 for x in scores if x > 0)
                failures = len(scores) - successes
                sample = random.betavariate(successes + 1, failures + 1)
            if sample > best_score:
                best_score = sample
                best_strategy = s
        return best_strategy

    def _select_seeds(
        self,
        frontier_nodes: list[ConceptNode],
        strategy: str,
        round_num: int,
    ) -> list[ConceptNode]:
        """Select seed nodes for a strategy."""
        if not frontier_nodes:
            return []

        # Rotate through frontier based on round number
        n = len(frontier_nodes)
        start_idx = (round_num - 1) % n

        if strategy in ("composition", "unification"):
            # Need multiple seeds
            indices = [(start_idx + i) % n for i in range(min(4, n))]
            return [frontier_nodes[i] for i in indices]
        else:
            return [frontier_nodes[start_idx]]

    async def _save_results(self, output_dir: Path) -> None:
        """Save discovery results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary JSON
        summary = {
            "total_discoveries": len(self._results),
            "rounds": len(self._round_log),
            "by_depth": {},
            "by_strategy": {},
            "round_log": self._round_log,
        }
        for r in self._results:
            dl = r.depth_level.value
            summary["by_depth"][dl] = summary["by_depth"].get(dl, 0) + 1
            summary["by_strategy"][r.strategy] = summary["by_strategy"].get(r.strategy, 0) + 1

        (output_dir / "discovery_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Individual results
        results_data = []
        for r in self._results:
            results_data.append({
                "id": r.node.id,
                "concept_type": r.node.concept_type.value,
                "domain": r.node.domain.value,
                "formal_statement": r.node.formal_statement,
                "proof_sketch": r.node.proof_sketch,
                "intuition": r.node.intuition,
                "novelty_score": r.novelty_score,
                "depth_score": r.depth_score,
                "depth_level": r.depth_level.value,
                "round_number": r.round_number,
                "strategy": r.strategy,
                "confidence": r.node.overall_confidence,
                "verification": r.verification_results,
                "tags": r.node.tags,
                "parent_ids": r.node.parent_ids,
            })

        (output_dir / "discoveries.json").write_text(
            json.dumps(results_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # LaTeX-formatted article of discoveries
        latex = self._format_latex(results_data)
        (output_dir / "discoveries.tex").write_text(latex, encoding="utf-8")

    def _format_latex(self, results: list[dict[str, Any]]) -> str:
        """Format discoveries as a LaTeX document fragment."""
        lines = [
            r"\section{Autonomous Discoveries}",
            r"",
            r"The following results were discovered by autonomous extension of the",
            r"paper's theoretical kernel. Each result is numbered sequentially and",
            r"classified by depth and verification confidence.",
            r"",
        ]

        for r in results:
            ctype = r["concept_type"].replace("_", " ").title()
            depth = r["depth_level"]
            conf = r.get("confidence", 0)

            lines.append(f"\\subsection*{{{r['id']}: {ctype} ({depth}, confidence {conf:.2f})}}")
            lines.append(r"\begin{quote}")
            lines.append(r"\textbf{Statement.} " + r["formal_statement"])
            lines.append(r"\end{quote}")

            if r.get("proof_sketch"):
                lines.append(r"\begin{proof}[Proof sketch]")
                lines.append(r["proof_sketch"])
                lines.append(r"\end{proof}")

            if r.get("intuition"):
                lines.append(r"\noindent\textit{Intuition.} " + r["intuition"])
            lines.append("")

        return "\n".join(lines)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the discovery session."""
        return {
            "total_discoveries": len(self._results),
            "rounds_completed": len(self._round_log),
            "by_depth": {
                d.value: sum(1 for r in self._results if r.depth_level == d)
                for d in DiscoveryDepth
            },
            "by_strategy": {
                s: sum(1 for r in self._results if r.strategy == s)
                for s in self.config.strategies
            },
            "avg_novelty": (
                sum(r.novelty_score for r in self._results) / max(len(self._results), 1)
            ),
            "avg_depth": (
                sum(r.depth_score for r in self._results) / max(len(self._results), 1)
            ),
            "avg_confidence": (
                sum(r.node.overall_confidence for r in self._results) / max(len(self._results), 1)
            ),
        }
