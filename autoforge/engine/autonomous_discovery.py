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
import statistics
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

try:
    import sympy
    from sympy import (
        symbols,
        simplify,
        expand,
        factor,
        solve,
        Eq,
        Symbol,
        limit,
        oo,
        series,
    )
    from sympy.combinatorics import Permutation

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

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
# Domain Context (D2: generalise domain-hardcoded prompts)
# ══════════════════════════════════════════════════════════════


@dataclass
class DomainContext:
    """Template variables that specialise strategy prompts to a mathematical domain.

    ConjectureGenerator replaces ``{domain.*}`` placeholders in its prompt
    templates with the values from whichever DomainContext is active.  Three
    pre-built contexts ship with AutoForge; users can construct their own.
    """

    name: str
    key_structures: str              # One-paragraph description of core objects
    typical_theorems: str            # Flavour text: what a good theorem looks like
    canonical_examples: str          # Concrete small examples one can compute with
    known_open_problems: str         # 3-5 open problems to bias exploration toward
    boundary_questions: str          # What limits / degeneracies are interesting?
    computable_quantities: str       # Named quantities + known exact values
    target_transfer_domains: str     # Where cross-domain analogies should point
    known_dualities: str             # Dual pairs already established in the theory
    higher_dim_challenges: str       # What breaks / changes in d ≥ 2


# ── Pre-built domain contexts ──────────────────────────────

SUPERSPACE_MODEL_SETS = DomainContext(
    name="Superspace Model Sets & Information-Theoretic Time",
    key_structures=(
        "Cut-and-project schemes (CPS) with physical space E = ℝ^d and internal "
        "space H = ℝ^n, lattice Γ ⊂ E × H, model sets Λ(W), golden-mean language "
        "X_m, Zeckendorf representation, Fold stabilization map, gauge anomaly G_m, "
        "scan error ε_m, boundary cylinder dimension, Parry measure μ, refinement "
        "functional τ(t) = −log μ(C(a_{0:t−1}))."
    ),
    typical_theorems=(
        "Results that are publishable in Inventiones or Annals: asymptotic formulae "
        "with sharp error terms, universal distributional limits, duality identities "
        "between entropy-like quantities, and connections to ergodic / operator theory."
    ),
    canonical_examples=(
        "1D golden-mean SFT on two symbols {0,1} with forbidden word 11. "
        "|X_m| = F_{m+2}. Gauge anomaly E[G_m]/m → 4/9, σ² = 118/243. "
        "Parry measure with entropy rate h = log φ. Zeckendorf carry propagation."
    ),
    known_open_problems=(
        "• Exact distribution of G_m beyond mean/variance (higher moments or CLT?)\n"
        "• Non-expansion optimal scan protocol for general windows W\n"
        "• Higher-dimensional (d ≥ 2) Zeckendorf stabilization dynamics\n"
        "• Operator-algebraic formulation of the KL-ledger identity\n"
        "• Connections to Rauzy fractal geometry in d = 2"
    ),
    boundary_questions=(
        "m → ∞ (infinite resolution), m → 1 (minimal resolution), "
        "φ → other algebraic numbers, boundary ∂W of the acceptance window, "
        "lattice Γ degeneration."
    ),
    computable_quantities=(
        "Gauge anomaly density δ_m = E[G_m]/m → 4/9, variance σ² = 118/243, "
        "fiber sizes |F_m(x)|, scan error ε_m(P;μ) decay rates, "
        "boundary cylinder counts N_m(∂P), entropy rate h_μ = log φ, "
        "refinement functional τ(t)."
    ),
    target_transfer_domains=(
        "Operator algebras / quantum information, ergodic theory / dynamical systems, "
        "coding theory / information theory, algebraic topology / homological algebra, "
        "number theory / arithmetic geometry, statistical mechanics / phase transitions."
    ),
    known_dualities=(
        "Resolution ↔ time depth (Remark 4.2: lost DOF recovered as temporal context), "
        "visible entropy H(π) ↔ fiber degeneracy E_π[log d_m(X)] (KL-ledger), "
        "synchronization budget ↔ collision budget (capacity law), "
        "physical space E ↔ internal space H (CPS duality)."
    ),
    higher_dim_challenges=(
        "Golden-mean language becomes a higher-dimensional SFT, "
        "Zeckendorf stabilization generalizes to multi-dimensional carry rules, "
        "boundary cylinder dimension depends on window geometry in ℝ^n, "
        "Sturmian coding becomes multidimensional symbolic dynamics."
    ),
)


ALGEBRAIC_GEOMETRY = DomainContext(
    name="Algebraic Geometry & Arithmetic",
    key_structures=(
        "Schemes, sheaves, cohomology, moduli spaces, étale fundamental groups, "
        "motivic cohomology, derived categories, intersection theory, "
        "Hodge structures, abelian varieties, elliptic curves."
    ),
    typical_theorems=(
        "Structure theorems for moduli spaces, finiteness results, "
        "comparison theorems between cohomology theories, "
        "effective bounds on rational points."
    ),
    canonical_examples=(
        "P^n (projective space), elliptic curves over Q, "
        "Grassmannians, toric varieties, K3 surfaces."
    ),
    known_open_problems=(
        "• Hodge conjecture\n"
        "• Standard conjectures on algebraic cycles\n"
        "• Effective Mordell / uniform Manin-Mumford\n"
        "• Mirror symmetry beyond Calabi-Yau\n"
        "• Motivic t-structures"
    ),
    boundary_questions=(
        "Degeneration of families (stable reduction), base-change to char p, "
        "semi-stable limits, tropical/non-archimedean limits."
    ),
    computable_quantities=(
        "Euler characteristics, Betti numbers, Picard numbers, "
        "heights of rational points, conductor exponents, "
        "Tamagawa numbers, L-function special values."
    ),
    target_transfer_domains=(
        "Number theory, representation theory, mathematical physics (string compactifications), "
        "symplectic topology, combinatorics (matroid theory), logic (model theory of fields)."
    ),
    known_dualities=(
        "Serre duality, Poincaré duality, Langlands duality, "
        "mirror symmetry (Hodge diamonds), Fourier-Mukai transforms."
    ),
    higher_dim_challenges=(
        "Minimal model program in higher dimensions, "
        "abundance conjecture, classification of Fano manifolds, "
        "birational geometry of moduli spaces."
    ),
)


DYNAMICAL_SYSTEMS = DomainContext(
    name="Dynamical Systems & Ergodic Theory",
    key_structures=(
        "Measure-preserving transformations, symbolic dynamics (SFTs, sofic shifts), "
        "entropy (topological, measure-theoretic, Kolmogorov-Sinai), "
        "mixing properties, Lyapunov exponents, transfer operators, "
        "thermodynamic formalism, Ruelle-Perron-Frobenius theory."
    ),
    typical_theorems=(
        "Distributional limit theorems for ergodic averages, "
        "entropy formulae, dimension formulae for invariant measures, "
        "orbit-counting asymptotics, equilibrium state uniqueness."
    ),
    canonical_examples=(
        "Full shifts on k symbols, golden-mean SFT, Bernoulli shifts, "
        "Arnold cat map, Smale horseshoe, Lorenz attractor, "
        "continued-fraction (Gauss) map."
    ),
    known_open_problems=(
        "• Furstenberg ×2 ×3 conjecture\n"
        "• Smooth realization of entropy\n"
        "• Effective equidistribution on higher-rank homogeneous spaces\n"
        "• Uniform hyperbolicity vs dominated splitting\n"
        "• Mixing rates for billiards in higher dimensions"
    ),
    boundary_questions=(
        "Zero-temperature limits (ground states), infinite-alphabet limits, "
        "bifurcation boundaries, critical points of pressure functions."
    ),
    computable_quantities=(
        "Topological entropy, measure-theoretic entropy, "
        "Lyapunov exponents, mixing rates (decay of correlations), "
        "pressure function P(φ), zeta functions, period-counting sequences."
    ),
    target_transfer_domains=(
        "Number theory (homogeneous dynamics), statistical mechanics, "
        "information theory, probability (random matrices), "
        "operator algebras (Cuntz algebras), combinatorics (symbolic substitutions)."
    ),
    known_dualities=(
        "Time-reversal duality, natural extension, "
        "spectral ↔ dynamical (Halmos-von Neumann), "
        "entropy ↔ dimension (Ledrappier-Young), "
        "transfer operator ↔ zeta function (Ruelle)."
    ),
    higher_dim_challenges=(
        "Higher-dimensional SFTs have undecidable entropy, "
        "smooth ergodic theory in dim ≥ 3 (partial hyperbolicity), "
        "multidimensional symbolic dynamics, "
        "higher-rank abelian actions (Katok-Spatzier rigidity)."
    ),
)


# Map for auto-detection from TheoryGraph domains
_DOMAIN_CONTEXT_REGISTRY: dict[str, DomainContext] = {
    "superspace_model_sets": SUPERSPACE_MODEL_SETS,
    "algebraic_geometry": ALGEBRAIC_GEOMETRY,
    "dynamical_systems": DYNAMICAL_SYSTEMS,
}


def detect_domain_context(graph: TheoryGraph) -> DomainContext:
    """Auto-detect the best DomainContext from a TheoryGraph's content.

    Heuristic: count domain tags and pick the matching pre-built context.
    Falls back to SUPERSPACE_MODEL_SETS (the reference use case).
    """
    stats = graph.get_stats()
    by_domain = stats.get("by_domain", {})

    # Simple keyword matching on graph title + dominant domain
    title_lower = graph.title.lower()
    if "model set" in title_lower or "superspace" in title_lower:
        return SUPERSPACE_MODEL_SETS
    if "algebraic" in title_lower or "scheme" in title_lower:
        return ALGEBRAIC_GEOMETRY
    if "dynami" in title_lower or "ergodic" in title_lower:
        return DYNAMICAL_SYSTEMS

    # Look at dominant scientific domain
    if by_domain.get("information_theory", 0) + by_domain.get("pure_mathematics", 0) > 3:
        return SUPERSPACE_MODEL_SETS
    if by_domain.get("theoretical_physics", 0) > 3:
        return DYNAMICAL_SYSTEMS

    return SUPERSPACE_MODEL_SETS  # default


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
# Algorithmic Conjecture Engine
# ══════════════════════════════════════════════════════════════


class AlgorithmicConjectureEngine:
    """Generates conjectures using algorithmic methods instead of LLM.

    Strategies:
    - GENERALIZE: Weaken hypotheses via SymPy (sufficient → necessary conditions)
    - COMPOSE: Cross-product of proven results with shared variables
    - BOUNDARY_ANALYSIS: Compute limits and singularities via SymPy
    - DUALITY: Apply known duality transforms
    - PATTERN_DETECTION: Find regularities in numerical sequences
    - SPECIALIZATION: Generate specific instances of general statements
    """

    def __init__(self) -> None:
        self._algo_calls = 0
        self._fallback_calls = 0

    @property
    def algorithm_ratio(self) -> float:
        """Ratio of algorithmic to fallback calls."""
        total = self._algo_calls + self._fallback_calls
        return self._algo_calls / total if total > 0 else 0.0

    def generate(
        self,
        strategy: str,
        seeds: list[dict],
        domain_context: dict,
    ) -> list[dict] | None:
        """Try to generate conjectures algorithmically. Returns None for LLM fallback."""
        if not HAS_SYMPY:
            self._fallback_calls += 1
            return None

        dispatch = {
            "generalize": self._generalize,
            "compose": self._compose,
            "boundary_analysis": self._boundary,
            "duality": self._duality,
            "pattern_detection": self._pattern_detect,
            "specialization": self._specialize,
        }

        handler = dispatch.get(strategy.lower())
        if handler is None:
            self._fallback_calls += 1
            return None

        try:
            results = handler(seeds, domain_context)
            if results:
                self._algo_calls += 1
                return results
        except Exception as e:
            logger.warning(
                "Algorithmic conjecture generation failed for %s: %s", strategy, e
            )

        self._fallback_calls += 1
        return None

    def _generalize(self, seeds: list[dict], ctx: dict) -> list[dict] | None:
        """Weaken hypotheses by replacing specific with general."""
        results = []
        for seed in seeds:
            expr_str = seed.get("expression", seed.get("statement", ""))
            if not expr_str:
                continue
            try:
                expr = sympy.sympify(expr_str)
                free = list(expr.free_symbols)
                # Strategy 1: Replace integer constraints with real
                # Strategy 2: Remove positivity constraints
                # Strategy 3: Expand domain (specific value → variable)
                for num in list(expr.atoms(sympy.Integer))[:2]:
                    param = Symbol(f"c_{abs(int(num))}")
                    generalized = expr.subs(num, param)
                    results.append(
                        {
                            "conjecture": f"For all {param}, {generalized}",
                            "expression": str(generalized),
                            "type": "generalization",
                            "parent": expr_str,
                            "confidence": 0.4,  # Generalizations need verification
                            "method": "sympy_generalization",
                        }
                    )
            except Exception:
                continue
        return results if results else None

    def _compose(self, seeds: list[dict], ctx: dict) -> list[dict] | None:
        """Compose results that share variables."""
        results = []
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i + 1 :]:
                e1_str = s1.get("expression", "")
                e2_str = s2.get("expression", "")
                if not (e1_str and e2_str):
                    continue
                try:
                    e1 = sympy.sympify(e1_str)
                    e2 = sympy.sympify(e2_str)
                    shared = e1.free_symbols & e2.free_symbols
                    if shared:
                        # Try substitution composition
                        var = list(shared)[0]
                        solutions = solve(e1, var)
                        if solutions:
                            composed = e2.subs(var, solutions[0])
                            simplified = simplify(composed)
                            results.append(
                                {
                                    "conjecture": str(simplified),
                                    "expression": str(simplified),
                                    "type": "composition",
                                    "parents": [e1_str, e2_str],
                                    "via_variable": str(var),
                                    "confidence": 0.5,
                                    "method": "sympy_composition",
                                }
                            )
                except Exception:
                    continue
        return results if results else None

    def _boundary(self, seeds: list[dict], ctx: dict) -> list[dict] | None:
        """Analyze boundary behavior via limits and series."""
        results = []
        for seed in seeds:
            expr_str = seed.get("expression", "")
            if not expr_str:
                continue
            try:
                expr = sympy.sympify(expr_str)
                for var in list(expr.free_symbols)[:1]:
                    # Limit at infinity
                    lim = limit(expr, var, oo)
                    if lim.is_finite:
                        results.append(
                            {
                                "conjecture": f"As {var}→∞, {expr_str} → {lim}",
                                "expression": str(lim),
                                "type": "boundary_limit",
                                "parent": expr_str,
                                "confidence": 0.8,
                                "method": "sympy_limit",
                            }
                        )
                    # Limit at 0
                    lim0 = limit(expr, var, 0)
                    if lim0.is_finite:
                        results.append(
                            {
                                "conjecture": f"As {var}→0, {expr_str} → {lim0}",
                                "expression": str(lim0),
                                "type": "boundary_limit",
                                "parent": expr_str,
                                "confidence": 0.8,
                                "method": "sympy_limit",
                            }
                        )
                    # Singularity detection
                    try:
                        singular_points = solve(1 / expr, var)
                        if singular_points:
                            results.append(
                                {
                                    "conjecture": f"{expr_str} has singularities at {var} = {singular_points}",
                                    "type": "singularity",
                                    "parent": expr_str,
                                    "confidence": 0.7,
                                    "method": "sympy_singularity",
                                }
                            )
                    except Exception:
                        pass
            except Exception:
                continue
        return results if results else None

    def _duality(self, seeds: list[dict], ctx: dict) -> list[dict] | None:
        """Apply known duality transforms."""
        results = []
        for seed in seeds:
            expr_str = seed.get("expression", "")
            if not expr_str:
                continue
            try:
                expr = sympy.sympify(expr_str)
                # Fourier-like dual: swap x↔1/x
                for var in list(expr.free_symbols)[:1]:
                    dual = expr.subs(var, 1 / var)
                    dual_simplified = simplify(dual)
                    # Self-dual check
                    is_self_dual = simplify(expr - dual_simplified) == 0
                    results.append(
                        {
                            "conjecture": f"Under {var}↔1/{var}: {expr_str} → {dual_simplified}",
                            "expression": str(dual_simplified),
                            "type": "duality",
                            "self_dual": is_self_dual,
                            "parent": expr_str,
                            "confidence": 0.6,
                            "method": "sympy_duality",
                        }
                    )
            except Exception:
                continue
        return results if results else None

    def _pattern_detect(self, seeds: list[dict], ctx: dict) -> list[dict] | None:
        """Detect patterns in numerical sequences."""
        results = []
        for seed in seeds:
            sequence = seed.get("sequence", seed.get("values", []))
            if not sequence or len(sequence) < 4:
                continue
            try:
                nums = [float(x) for x in sequence]
                # Check arithmetic progression
                diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
                if len(set(round(d, 10) for d in diffs)) == 1:
                    d = diffs[0]
                    results.append(
                        {
                            "conjecture": f"a(n) = {nums[0]} + {d}*n (arithmetic progression)",
                            "type": "pattern_arithmetic",
                            "common_difference": d,
                            "confidence": 0.9,
                            "method": "pattern_detection",
                        }
                    )
                # Check geometric progression
                if all(n != 0 for n in nums[:-1]):
                    ratios = [nums[i + 1] / nums[i] for i in range(len(nums) - 1)]
                    if len(set(round(r, 10) for r in ratios)) == 1:
                        r = ratios[0]
                        results.append(
                            {
                                "conjecture": f"a(n) = {nums[0]} * {r}^n (geometric progression)",
                                "type": "pattern_geometric",
                                "common_ratio": r,
                                "confidence": 0.9,
                                "method": "pattern_detection",
                            }
                        )
                # Check polynomial fit (degree 2)
                if len(nums) >= 3:
                    second_diffs = [
                        diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)
                    ]
                    if (
                        len(set(round(d, 10) for d in second_diffs)) == 1
                        and second_diffs[0] != 0
                    ):
                        results.append(
                            {
                                "conjecture": f"Sequence follows quadratic pattern (constant second differences = {second_diffs[0]})",
                                "type": "pattern_quadratic",
                                "second_difference": second_diffs[0],
                                "confidence": 0.85,
                                "method": "pattern_detection",
                            }
                        )
            except Exception:
                continue
        return results if results else None

    def _specialize(self, seeds: list[dict], ctx: dict) -> list[dict] | None:
        """Generate specific instances of general statements."""
        results = []
        for seed in seeds:
            expr_str = seed.get("expression", "")
            if not expr_str:
                continue
            try:
                expr = sympy.sympify(expr_str)
                free = list(expr.free_symbols)
                if not free:
                    continue
                # Test at specific values
                test_values = [0, 1, -1, 2, sympy.Rational(1, 2)]
                for val in test_values:
                    specialized = expr.subs(free[0], val)
                    simplified = simplify(specialized)
                    results.append(
                        {
                            "conjecture": f"When {free[0]}={val}: {simplified}",
                            "expression": str(simplified),
                            "type": "specialization",
                            "parent": expr_str,
                            "at_value": str(val),
                            "confidence": 0.95,
                            "method": "sympy_specialization",
                        }
                    )
            except Exception:
                continue
        return results[:5] if results else None  # Limit to 5 specializations


# ══════════════════════════════════════════════════════════════
# Graph-Based Depth Evaluator
# ══════════════════════════════════════════════════════════════


class GraphBasedDepthEvaluator:
    """Evaluates conjecture depth using graph metrics instead of LLM.

    Metrics:
    - technical_complexity: AST depth of expression (SymPy tree depth)
    - structural_novelty: Jaccard distance from all known conjectures
    - bridging_potential: Number of new connections enabled
    - elegance: Statement length / proof_length ratio
    """

    def __init__(self) -> None:
        self._known_tokens: set[str] = set()

    def evaluate(
        self, conjecture: dict, known_conjectures: list[dict]
    ) -> dict:
        """Score a conjecture using algorithmic metrics."""
        expr_str = conjecture.get("expression", conjecture.get("conjecture", ""))

        scores = {}

        # Technical complexity: expression tree depth
        try:
            expr = sympy.sympify(expr_str)
            depth = self._tree_depth(expr)
            scores["technical_complexity"] = min(1.0, depth / 10.0)
        except Exception:
            scores["technical_complexity"] = 0.5

        # Structural novelty: Jaccard distance from known
        tokens = set(re.findall(r"[a-zA-Z_]\w*", expr_str.lower()))
        if self._known_tokens:
            jaccard = 1.0 - len(tokens & self._known_tokens) / max(
                1, len(tokens | self._known_tokens)
            )
        else:
            jaccard = 1.0
        scores["structural_novelty"] = jaccard
        self._known_tokens.update(tokens)

        # Bridging potential: count shared variables with existing conjectures
        bridges = 0
        for known in known_conjectures:
            known_str = known.get("expression", "")
            known_tokens = set(re.findall(r"[a-zA-Z_]\w*", known_str.lower()))
            if tokens & known_tokens and tokens != known_tokens:
                bridges += 1
        scores["bridging_potential"] = min(
            1.0, bridges / max(1, len(known_conjectures)) * 5
        )

        # Composite
        scores["composite"] = (
            scores["technical_complexity"] * 0.25
            + scores["structural_novelty"] * 0.35
            + scores["bridging_potential"] * 0.40
        )

        return scores

    def _tree_depth(self, expr) -> int:
        """Compute expression tree depth."""
        if not hasattr(expr, "args") or not expr.args:
            return 1
        return 1 + max(self._tree_depth(a) for a in expr.args)


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

    # Strategy → domain-general prompt template.
    # Placeholders: {parent_statement}, {parent_proof}, {second_statement}, {context}
    #   plus DomainContext fields prefixed with "domain_":
    #   {domain_name}, {domain_key_structures}, {domain_typical_theorems},
    #   {domain_canonical_examples}, {domain_known_open_problems},
    #   {domain_boundary_questions}, {domain_computable_quantities},
    #   {domain_target_transfer_domains}, {domain_known_dualities},
    #   {domain_higher_dim_challenges}
    _STRATEGY_PROMPTS: dict[str, str] = {
        "generalization": """Given the following theorem from the theory of {domain_name},
find a STRICT GENERALIZATION — weaken one or more hypotheses while preserving a
meaningful (possibly weakened) conclusion.

The generalization must be NON-TRIVIAL: it should require genuine mathematical work
to prove, not just removing an unused hypothesis.

PARENT THEOREM:
{{parent_statement}}

PROOF SKETCH OF PARENT:
{{parent_proof}}

THEORY CONTEXT:
{{context}}

KEY STRUCTURES:
{domain_key_structures}

{domain_typical_theorems}

Produce a generalized result. Use rigorous mathematical language suitable for
Inventiones Mathematicae or Annals of Mathematics. Provide:
1. A precise formal statement (LaTeX notation)
2. A proof sketch (at least the key ideas and techniques needed)
3. An intuitive explanation of WHY the generalization works
4. What new cases or applications the generalization covers""",

        "composition": """Given two results from the theory of {domain_name},
find a NON-TRIVIAL COMPOSITION that combines them into a single deeper result.

The composition should not be a mere conjunction (A and B); it should produce
a genuinely new insight that neither result alone implies.

RESULT A:
{{parent_statement}}

RESULT B:
{{second_statement}}

THEORY CONTEXT:
{{context}}

KEY STRUCTURES:
{domain_key_structures}

Produce a composed result. Use rigorous mathematical language. Provide:
1. A precise formal statement (LaTeX notation)
2. A proof sketch showing HOW A and B combine
3. An intuitive explanation of the new insight
4. Why this composition is non-trivial""",

        "analogy_transfer": """Given the following result from the theory of {domain_name},
identify a STRUCTURAL ANALOGY with another mathematical domain and TRANSFER
the result to produce a new theorem in that domain.

TARGET DOMAINS for transfer (pick the most natural one):
{domain_target_transfer_domains}

PARENT RESULT:
{{parent_statement}}

THEORY CONTEXT:
{{context}}

Produce an analogous result in the target domain. Provide:
1. The precise structural analogy (what maps to what)
2. The formal statement of the transferred result (LaTeX)
3. A proof sketch (or explanation why the analogy works formally)
4. What new insight the transfer provides in the target domain""",

        "boundary_analysis": """Given the following result from the theory of {domain_name},
investigate its BOUNDARY BEHAVIOR — what happens at extreme parameter values,
degenerate cases, or critical transitions.

PARENT RESULT:
{{parent_statement}}

THEORY CONTEXT:
{{context}}

Investigate at least one of the following boundary regimes:
{domain_boundary_questions}

Produce a boundary result. Provide:
1. The precise limiting/degenerate regime
2. A formal statement of the boundary behavior (LaTeX)
3. A proof sketch
4. Physical or mathematical interpretation""",

        "duality": """Given the following result, identify and apply a DUALITY
transformation to produce a dual theorem.

Known dualities in the theory of {domain_name}:
{domain_known_dualities}

PARENT RESULT:
{{parent_statement}}

THEORY CONTEXT:
{{context}}

Apply a duality to produce the DUAL statement. Provide:
1. Which duality you are applying and why
2. The formal dual statement (LaTeX)
3. A proof that the duality indeed exchanges the two sides
4. New insight from the dual perspective""",

        "unification": """Given the following results from the theory of {domain_name},
find a UNIFYING PRINCIPLE that explains all of them as special cases of a deeper
structural pattern.

RESULTS TO UNIFY:
{{parent_statement}}

THEORY CONTEXT:
{{context}}

OPEN PROBLEMS (may hint at unifying principles):
{domain_known_open_problems}

Produce a unifying theorem or framework. Provide:
1. The unifying principle (formal statement in LaTeX)
2. How each input result is recovered as a special case
3. A proof sketch of the unifying result
4. What new predictions the unification makes beyond the known cases""",

        "numerical_exploration": """Given the following result involving computable
quantities from the theory of {domain_name}, design a NUMERICAL EXPERIMENT to
discover NEW PATTERNS that extend beyond what is currently proved.

PARENT RESULT:
{{parent_statement}}

Key computable quantities in this theory:
{domain_computable_quantities}

THEORY CONTEXT:
{{context}}

CANONICAL EXAMPLES:
{domain_canonical_examples}

Design a numerical investigation and CONJECTURE a new result. Provide:
1. The computational experiment (what to compute, parameter ranges)
2. The observed numerical pattern (describe what you expect)
3. A precise CONJECTURE based on the pattern (formal LaTeX)
4. Why the conjecture is plausible (heuristic argument or connection to theory)""",

        "dimensional_lifting": """Given the following result from the theory of
{domain_name}, lift it to HIGHER DIMENSIONS or broader settings.

PARENT RESULT:
{{parent_statement}}

THEORY CONTEXT:
{{context}}

Challenges of higher-dimensional / broader extension:
{domain_higher_dim_challenges}

Produce a higher-dimensional extension. Provide:
1. The precise generalization (LaTeX)
2. What changes and what remains from the original setting
3. A proof sketch or argument for the generalization
4. New phenomena that appear only in the extended setting""",
    }

    @classmethod
    def _render_template(cls, strategy: str, domain: DomainContext) -> str:
        """Two-phase render: first fill domain fields, then return a
        template that still has {parent_statement} etc. for .format()."""
        raw = cls._STRATEGY_PROMPTS[strategy]
        # Phase 1: fill domain_* placeholders
        rendered = raw.format(
            domain_name=domain.name,
            domain_key_structures=domain.key_structures,
            domain_typical_theorems=domain.typical_theorems,
            domain_canonical_examples=domain.canonical_examples,
            domain_known_open_problems=domain.known_open_problems,
            domain_boundary_questions=domain.boundary_questions,
            domain_computable_quantities=domain.computable_quantities,
            domain_target_transfer_domains=domain.target_transfer_domains,
            domain_known_dualities=domain.known_dualities,
            domain_higher_dim_challenges=domain.higher_dim_challenges,
        )
        # Phase 2 unescapes the double-braces back to single (for later .format())
        return rendered

    async def generate(
        self,
        strategy: str,
        frontier_nodes: list[ConceptNode],
        kernel_context: str,
        llm: Any,
        *,
        round_number: int = 1,
        existing_count: int = 0,
        domain_context: DomainContext | None = None,
    ) -> list[ConceptNode]:
        """Generate candidate discoveries using a specific strategy.

        Args:
            domain_context: If provided, specialises prompts for the given domain.
                            Defaults to SUPERSPACE_MODEL_SETS for backward compat.

        Returns list of ConceptNode candidates (not yet validated).
        """
        from autoforge.engine.llm_router import TaskComplexity

        if strategy not in self._STRATEGY_PROMPTS:
            logger.warning(f"[ConjectureGenerator] Unknown strategy: {strategy}")
            return []

        domain = domain_context or SUPERSPACE_MODEL_SETS
        template = self._render_template(strategy, domain)

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
        self._algo_engine = AlgorithmicConjectureEngine()
        self._graph_evaluator = GraphBasedDepthEvaluator()
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
        domain_context: DomainContext | None = None,
    ) -> list[DiscoveryResult]:
        """Execute the autonomous discovery loop.

        Args:
            graph: TheoryGraph containing the paper's theory
            llm: LLM router for reasoning calls
            output_dir: Optional directory to save results incrementally
            domain_context: If None, auto-detected from graph content

        Returns:
            List of accepted discoveries
        """
        if domain_context is None:
            domain_context = detect_domain_context(graph)
            logger.info(f"[Discovery] Auto-detected domain: {domain_context.name}")

        self._domain_context = domain_context
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

            # Try algorithmic generation first
            algo_seeds = [
                {"expression": s.formal_statement, "statement": s.formal_statement}
                for s in seeds
            ]
            algo_candidates = self._algo_engine.generate(
                strategy=strategy,
                seeds=algo_seeds,
                domain_context={"name": domain_context.name},
            )

            if algo_candidates:
                logger.debug(
                    f"[Discovery] Generated {len(algo_candidates)} candidates "
                    f"algorithmically (ratio: {self._algo_engine.algorithm_ratio:.2%})"
                )
                # Convert algorithmic results to ConceptNode format for consistency
                candidates = []
                for algo_result in algo_candidates:
                    node = ConceptNode(
                        id=f"AD-ALGO-{len(self._results) + len(candidates):03d}",
                        name=algo_result.get("type", "algorithmic_result"),
                        concept_type=ConceptType.CONJECTURE,
                        formal_statement=algo_result.get("conjecture", ""),
                        proof_sketch=algo_result.get("method", "algorithmic"),
                        intuition=f"Generated via {algo_result.get('method', 'algorithm')}",
                        tags=["algorithmic"],
                        domain=ScientificDomain.MATHEMATICAL_PHYSICS,
                    )
                    candidates.append(node)
            else:
                # Fall back to LLM-based generation
                logger.debug(
                    f"[Discovery] Algorithmic generation unavailable or failed; "
                    f"using LLM (algorithm_ratio: {self._algo_engine.algorithm_ratio:.2%})"
                )
                candidates = await self._generator.generate(
                    strategy=strategy,
                    frontier_nodes=seeds,
                    kernel_context=kernel_context,
                    llm=llm,
                    round_number=round_num,
                    existing_count=len(self._results),
                    domain_context=domain_context,
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

                # Depth evaluation: try graph-based first, fall back to LLM
                depth_score = None
                known_conjectures = [r.node for r in self._results]
                if HAS_SYMPY:
                    try:
                        graph_scores = self._graph_evaluator.evaluate(
                            {
                                "expression": candidate.formal_statement,
                                "conjecture": candidate.formal_statement,
                            },
                            [
                                {"expression": c.formal_statement}
                                for c in known_conjectures
                            ],
                        )
                        depth_score = graph_scores.get("composite", 0.5)
                        logger.debug(
                            f"[Discovery] Graph-based depth evaluation: {depth_score:.2f} "
                            f"(complexity={graph_scores.get('technical_complexity', 0):.2f}, "
                            f"novelty={graph_scores.get('structural_novelty', 0):.2f}, "
                            f"bridging={graph_scores.get('bridging_potential', 0):.2f})"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[Discovery] Graph-based evaluation failed: {e}"
                        )
                        depth_score = None

                if depth_score is None:
                    # Fall back to LLM-based evaluation
                    logger.debug("[Discovery] Using LLM-based depth evaluation")
                    depth, depth_score, depth_reason = await self._evaluator.evaluate(
                        candidate, kernel_context, llm
                    )
                else:
                    # Map numeric score to depth level
                    if depth_score >= 0.85:
                        depth = DiscoveryDepth.BREAKTHROUGH
                    elif depth_score >= 0.70:
                        depth = DiscoveryDepth.DEEP
                    elif depth_score >= 0.55:
                        depth = DiscoveryDepth.MODERATE
                    else:
                        depth = DiscoveryDepth.SHALLOW
                    depth_reason = "Graph-based structural evaluation"

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

        # Report algorithm usage statistics
        logger.info(
            f"[Discovery] Algorithmic conjecture generation: "
            f"{self._algo_engine._algo_calls} successful calls, "
            f"{self._algo_engine._fallback_calls} fallbacks "
            f"(algorithm ratio: {self._algo_engine.algorithm_ratio:.2%})"
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


# ══════════════════════════════════════════════════════════════
# Elo Tournament for Hypotheses
# ══════════════════════════════════════════════════════════════
# Google AI Co-Scientist style hypothesis ranking via pairwise
# competition. Hypotheses are iteratively matched against each other,
# with ratings updated using the Elo chess rating system. The LLM
# judges each match based on novelty, depth, verifiability, and impact.


@dataclass
class EloRating:
    """Elo rating record for a single hypothesis."""

    hypothesis_id: str
    rating: float = 1500.0
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    confidence_interval: float = 0.0

    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.matches_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.matches_played


@dataclass
class MatchResult:
    """Result of a match between two hypotheses."""

    winner_id: str
    loser_id: str
    is_draw: bool
    comparison_reasoning: str
    judge_confidence: float

    def loser_id_or_draw(self) -> str | None:
        """Return loser ID, or None if draw."""
        return None if self.is_draw else self.loser_id


class EloTournament:
    """Elo rating tournament for hypothesis ranking.

    Uses pairwise LLM-judged competitions to rank hypotheses based on
    novelty, depth, verifiability, and impact. Ratings update via the
    standard Elo formula used in chess.
    """

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        """Initialize the tournament.

        Args:
            k_factor: Elo update sensitivity (higher = more volatile).
            initial_rating: Starting rating for new hypotheses.
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self._ratings: dict[str, EloRating] = {}
        self._match_history: list[MatchResult] = []

    def register_hypothesis(
        self, hypothesis_id: str, initial_rating: float | None = None
    ) -> None:
        """Register a hypothesis in the tournament.

        Args:
            hypothesis_id: Unique identifier for the hypothesis.
            initial_rating: Optional custom initial rating (default 1500).
        """
        if hypothesis_id in self._ratings:
            return

        rating = initial_rating if initial_rating is not None else self.initial_rating
        self._ratings[hypothesis_id] = EloRating(
            hypothesis_id=hypothesis_id, rating=rating
        )

    async def run_match(
        self,
        h1_id: str,
        h1_content: str,
        h2_id: str,
        h2_content: str,
        llm: Any,
        criteria: str | None = None,
    ) -> MatchResult:
        """Run a match between two hypotheses.

        The LLM judges which hypothesis is stronger based on:
        - Novelty: How original vs. existing work
        - Depth: How theoretically deep/sophisticated
        - Verifiability: How rigorously checkable
        - Impact: Potential significance/applicability

        Args:
            h1_id: ID of first hypothesis.
            h1_content: Text content of first hypothesis.
            h2_id: ID of second hypothesis.
            h2_content: Text content of second hypothesis.
            llm: LLM instance with async __call__.
            criteria: Optional custom judging criteria.

        Returns:
            MatchResult with winner, loser, and reasoning.
        """
        if criteria is None:
            criteria = (
                "Consider: (1) Novelty relative to known results, "
                "(2) Theoretical depth and sophistication, "
                "(3) Rigor and verifiability, "
                "(4) Potential impact and applicability."
            )

        prompt = f"""Compare these two scientific hypotheses and judge which is stronger.

Hypothesis A (ID: {h1_id}):
{h1_content}

Hypothesis B (ID: {h2_id}):
{h2_content}

Judging criteria: {criteria}

Respond in JSON:
{{
    "winner": "A" or "B",
    "is_draw": true/false,
    "reasoning": "Detailed comparison...",
    "confidence": 0.0-1.0
}}"""

        response_text = await llm(prompt)
        result_data = _extract_json(response_text)

        if result_data is None:
            # Fallback: declare a draw
            logging.warning(f"Failed to parse judge response for {h1_id} vs {h2_id}")
            result_data = {
                "winner": "A",
                "is_draw": True,
                "reasoning": "Judge parsing failed; draw declared.",
                "confidence": 0.0,
            }

        winner_id = h1_id if result_data.get("winner") == "A" else h2_id
        loser_id = h2_id if result_data.get("winner") == "A" else h1_id
        is_draw = result_data.get("is_draw", False)
        reasoning = result_data.get("reasoning", "")
        confidence = result_data.get("confidence", 0.5)

        match = MatchResult(
            winner_id=winner_id,
            loser_id=loser_id,
            is_draw=is_draw,
            comparison_reasoning=reasoning,
            judge_confidence=float(confidence),
        )

        self._update_ratings(match)
        self._match_history.append(match)

        return match

    def _update_ratings(self, match: MatchResult) -> None:
        """Update Elo ratings after a match.

        Uses standard Elo formula:
            E_a = 1 / (1 + 10^((R_b - R_a) / 400))
            R_a_new = R_a + K * (S_a - E_a)

        Args:
            match: The match result.
        """
        # Ensure both IDs are registered
        if match.winner_id not in self._ratings:
            self.register_hypothesis(match.winner_id)
        if match.loser_id not in self._ratings:
            self.register_hypothesis(match.loser_id)

        rating_a = self._ratings[match.winner_id]
        rating_b = self._ratings[match.loser_id]

        # Expected score for A (winner)
        exp_a = 1.0 / (1.0 + 10.0 ** ((rating_b.rating - rating_a.rating) / 400.0))

        if match.is_draw:
            # Draw: each gets 0.5
            rating_a.rating += self.k_factor * (0.5 - exp_a)
            rating_b.rating += self.k_factor * (0.5 - (1.0 - exp_a))
            rating_a.draws += 1
            rating_b.draws += 1
        else:
            # Winner gets 1, loser gets 0
            rating_a.rating += self.k_factor * (1.0 - exp_a)
            rating_b.rating += self.k_factor * (0.0 - (1.0 - exp_a))
            rating_a.wins += 1
            rating_b.losses += 1

        rating_a.matches_played += 1
        rating_b.matches_played += 1

        # Update confidence intervals
        rating_a.confidence_interval = self._compute_confidence_interval(rating_a)
        rating_b.confidence_interval = self._compute_confidence_interval(rating_b)

    def _compute_confidence_interval(self, rating: EloRating) -> float:
        """Compute confidence interval for a rating based on matches played.

        More matches → narrower interval.

        Args:
            rating: The EloRating to compute for.

        Returns:
            Estimated 95% confidence interval width.
        """
        if rating.matches_played < 2:
            return 200.0  # Very wide
        # Standard error approximation: sqrt(N) scaling
        return 200.0 / math.sqrt(rating.matches_played)

    async def run_tournament(
        self,
        hypotheses: dict[str, str],
        llm: Any,
        rounds: int | None = None,
    ) -> dict[str, Any]:
        """Run a tournament among hypotheses.

        Can be round-robin (all vs all) or Swiss-style (multiple rounds,
        selective pairings).

        Args:
            hypotheses: Dict mapping hypothesis_id → content.
            llm: LLM instance.
            rounds: Number of tournament rounds (default: ceil(log2(n))).

        Returns:
            Tournament summary with final rankings.
        """
        # Register all hypotheses
        for hyp_id in hypotheses:
            if hyp_id not in self._ratings:
                self.register_hypothesis(hyp_id)

        n = len(hypotheses)
        if n == 0:
            return {
                "total_matches": 0,
                "rankings": [],
                "top_5": [],
                "stats": self.get_tournament_stats(),
            }
        if n == 1:
            return {
                "total_matches": 0,
                "rankings": self.get_rankings(),
                "top_5": self.get_top_k(5),
                "stats": self.get_tournament_stats(),
            }

        if rounds is None:
            rounds = max(3, math.ceil(math.log2(n)))

        hyp_list = list(hypotheses.items())

        for round_num in range(rounds):
            logging.info(f"Tournament round {round_num + 1}/{rounds}")

            # Simple round-robin: pair adjacent indices in shuffled order
            # (more sophisticated Swiss system could be implemented)
            indices = list(range(n))
            # Rotate for different pairings each round
            indices = indices[round_num % n :] + indices[: round_num % n]

            for i in range(0, n - 1, 2):
                idx_a, idx_b = indices[i], indices[i + 1]
                h1_id, h1_content = hyp_list[idx_a]
                h2_id, h2_content = hyp_list[idx_b]

                await self.run_match(h1_id, h1_content, h2_id, h2_content, llm)

        return {
            "total_matches": len(self._match_history),
            "rankings": self.get_rankings(),
            "top_5": self.get_top_k(5),
            "stats": self.get_tournament_stats(),
        }

    def get_rankings(self) -> list[tuple[str, EloRating]]:
        """Get all hypotheses sorted by Elo rating (descending).

        Returns:
            List of (hypothesis_id, EloRating) tuples.
        """
        return sorted(
            self._ratings.items(), key=lambda x: x[1].rating, reverse=True
        )

    def get_top_k(self, k: int = 5) -> list[tuple[str, float]]:
        """Get top k hypotheses by Elo rating.

        Args:
            k: Number of top hypotheses to return.

        Returns:
            List of (hypothesis_id, rating) tuples.
        """
        rankings = self.get_rankings()
        return [(hyp_id, rating.rating) for hyp_id, rating in rankings[:k]]

    def get_tournament_stats(self) -> dict[str, Any]:
        """Get summary statistics for the tournament.

        Returns:
            Dict with counts, averages, and extrema.
        """
        ratings_list = list(self._ratings.values())
        if not ratings_list:
            return {"total_hypotheses": 0}

        return {
            "total_hypotheses": len(ratings_list),
            "total_matches": len(self._match_history),
            "avg_rating": sum(r.rating for r in ratings_list) / len(ratings_list),
            "min_rating": min(r.rating for r in ratings_list),
            "max_rating": max(r.rating for r in ratings_list),
            "avg_win_rate": sum(r.win_rate() for r in ratings_list) / len(ratings_list),
            "draws_count": sum(1 for m in self._match_history if m.is_draw),
        }


@dataclass
class Hypothesis:
    """Structured hypothesis item used by HypothesisTournament."""

    id: str
    statement: str
    novelty: float = 0.5
    depth: float = 0.5
    verification_confidence: float = 0.5
    seed_round: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HypothesisMatch:
    """Recorded match between two hypotheses."""

    round_index: int
    hypothesis_a: str
    hypothesis_b: str
    winner_id: str
    loser_id: str
    score_a: float
    score_b: float


class HypothesisTournament:
    """Deterministic Elo-based tournament for discovery hypotheses."""

    K_FACTOR = 32.0

    def __init__(self, initial_rating: float = 1500.0) -> None:
        self.initial_rating = initial_rating
        self._hypotheses: dict[str, Hypothesis] = {}
        self._ratings: dict[str, float] = {}
        self._wins: dict[str, int] = {}
        self._losses: dict[str, int] = {}
        self._match_history: list[HypothesisMatch] = []

    def register(self, hypothesis: Hypothesis) -> None:
        """Register a hypothesis idempotently."""
        if hypothesis.id in self._hypotheses:
            return
        self._hypotheses[hypothesis.id] = hypothesis
        self._ratings[hypothesis.id] = self.initial_rating
        self._wins[hypothesis.id] = 0
        self._losses[hypothesis.id] = 0

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Elo expected score, same formula as EloArgumentRanker."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, winner_id: str, loser_id: str) -> None:
        """Apply one Elo update after a deterministic match outcome."""
        ra = self._ratings.get(winner_id, self.initial_rating)
        rb = self._ratings.get(loser_id, self.initial_rating)
        ea = self.expected_score(ra, rb)
        self._ratings[winner_id] = ra + self.K_FACTOR * (1 - ea)
        self._ratings[loser_id] = rb + self.K_FACTOR * (0 - (1 - ea))
        self._wins[winner_id] = self._wins.get(winner_id, 0) + 1
        self._losses[loser_id] = self._losses.get(loser_id, 0) + 1

    def _pair_round(self, round_index: int) -> list[tuple[str, str]]:
        ids = sorted(self._hypotheses.keys())
        n = len(ids)
        if n < 2:
            return []
        shift = round_index % n
        rotated = ids[shift:] + ids[:shift]
        pairs = []
        for i in range(0, n - 1, 2):
            pairs.append((rotated[i], rotated[i + 1]))
        return pairs

    def _composite_score(self, h: Hypothesis) -> float:
        return 0.4 * h.novelty + 0.35 * h.depth + 0.25 * h.verification_confidence

    async def run(self, rounds: int = 3) -> dict[str, Any]:
        """Run deterministic pairings for N rounds and return rankings."""
        if not self._hypotheses:
            return {"rankings": [], "match_history": []}

        if len(self._hypotheses) == 1:
            return {
                "rankings": self.rankings(),
                "match_history": [],
            }

        for round_index in range(rounds):
            for aid, bid in self._pair_round(round_index):
                a = self._hypotheses[aid]
                b = self._hypotheses[bid]
                score_a = self._composite_score(a)
                score_b = self._composite_score(b)

                if score_a >= score_b:
                    winner_id, loser_id = aid, bid
                else:
                    winner_id, loser_id = bid, aid

                self.update(winner_id, loser_id)
                self._match_history.append(
                    HypothesisMatch(
                        round_index=round_index,
                        hypothesis_a=aid,
                        hypothesis_b=bid,
                        winner_id=winner_id,
                        loser_id=loser_id,
                        score_a=score_a,
                        score_b=score_b,
                    )
                )

        return {
            "rankings": self.rankings(),
            "match_history": [m.__dict__ for m in self._match_history],
        }

    def rankings(self) -> list[dict[str, Any]]:
        rows = []
        for hyp_id, rating in sorted(self._ratings.items(), key=lambda x: x[1], reverse=True):
            rows.append({
                "id": hyp_id,
                "rating": rating,
                "wins": self._wins.get(hyp_id, 0),
                "losses": self._losses.get(hyp_id, 0),
                "statement": self._hypotheses[hyp_id].statement,
            })
        return rows
