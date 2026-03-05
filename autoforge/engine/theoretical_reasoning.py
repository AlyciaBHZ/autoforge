"""Theoretical Reasoning Engine — Cross-Domain Scientific Discovery & Evolution.

This module is the intellectual core of AutoForge's scientific capability.
Unlike lean_prover.py (which handles formal verification in one syntax),
this engine operates at the level of *theoretical reasoning itself* —
the kind that produces 1600-page cross-domain research articles connecting
number theory, dynamical systems, operator theory, and statistical mechanics.

Design philosophy:
  1. **Lean is one syntax, not the goal.** The goal is theoretical insight.
     Formal verification (Lean, Z3, Isabelle) is one of many verification
     backends. Numerical experiments, dimensional analysis, consistency
     checks, and symmetry preservation are equally valid.

  2. **Cross-domain reasoning is the core capability.** The most valuable
     scientific insights come from structural isomorphisms across fields
     (e.g., golden ratio dynamics ↔ Zeckendorf representations ↔ ζ-functions
     ↔ Chebotarev density — as in the reference article).

  3. **Theory evolution, not just verification.** The system should generate
     new theoretical branches from existing work: relax assumptions,
     transfer to new domains, compose results, find boundary cases.

  4. **Article-level output.** The deliverable is not a proof but a structured
     theoretical argument: definitions → lemmas → theorems → cross-domain
     connections → conjectures → verification.

Architecture:

    ConceptNode         — Atomic theoretical concept (definition, theorem, conjecture, ...)
        ↓ connected by
    ConceptRelation     — Typed relation (depends_on, analogous_to, generalizes, ...)
        ↓ organized in
    TheoryGraph         — The knowledge structure of a theory/article
        ↓ reasoned about by
    ReasoningEngine     — Multi-strategy reasoning (analogy, composition, generalization, ...)
        ↓ verified by
    VerificationSuite   — Multi-modal verification (formal, numerical, consistency, physical)
        ↓ evolved by
    TheoryEvolver       — Branch, extend, compose theories
        ↓ output by
    ArticleGenerator    — Produce structured research articles

Integration:
  - CapabilityDAG provides persistent cross-project knowledge
  - LeanProver handles formal math verification (one backend among many)
  - LLM Router handles AI reasoning calls

References:
  - FunSearch (Nature 2024): LLM-guided evolutionary discovery
  - AlphaEvolve/AlphaProof (DeepMind 2025): Discovery → reasoning → proof pipeline
  - HERMES (2025): Hybrid neuro-symbolic step-level verification
  - SciAgent (2025): Multi-agent scientific reasoning with tool augmentation
  - Graph of Thought (2024): DAG-structured reasoning with cross-branch flow
  - LacMaterial (2024): Explicit analogical reasoning across domains
  - Ramanujan Machine (Nature 2021): Numerical pattern → symbolic conjecture
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

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Enumerations
# ══════════════════════════════════════════════════════════════


class ScientificDomain(str, Enum):
    """Scientific domains — not exhaustive, extensible via GENERAL."""
    PURE_MATHEMATICS = "pure_mathematics"
    MATHEMATICS = "pure_mathematics"  # Legacy alias
    APPLIED_MATHEMATICS = "applied_mathematics"
    THEORETICAL_PHYSICS = "theoretical_physics"
    MATHEMATICAL_PHYSICS = "mathematical_physics"
    THEORETICAL_CHEMISTRY = "theoretical_chemistry"
    THEORETICAL_BIOLOGY = "theoretical_biology"
    COMPUTER_SCIENCE = "computer_science"
    INFORMATION_THEORY = "information_theory"
    STATISTICAL_MECHANICS = "statistical_mechanics"
    GENERAL = "general"


class ConceptType(str, Enum):
    """Types of theoretical concepts."""
    DEFINITION = "definition"
    AXIOM = "axiom"
    LEMMA = "lemma"
    PROPOSITION = "proposition"
    THEOREM = "theorem"
    COROLLARY = "corollary"
    CONJECTURE = "conjecture"
    CONSTRUCTION = "construction"       # Explicit mathematical construction
    OBSERVATION = "observation"         # Empirical/numerical observation
    PRINCIPLE = "principle"             # Guiding principle (e.g., conservation law)
    STRUCTURAL_ANALOGY = "analogy"      # Cross-domain structural isomorphism
    OPEN_PROBLEM = "open_problem"
    ALGORITHM = "algorithm"
    EXPERIMENT = "experiment"           # Numerical/computational experiment


class RelationType(str, Enum):
    """Types of relations between concepts."""
    DEPENDS_ON = "depends_on"           # B requires A
    GENERALIZES = "generalizes"         # B generalizes A (weaker assumptions)
    SPECIALIZES = "specializes"         # B is a special case of A
    ANALOGOUS_TO = "analogous_to"       # Structural similarity across domains
    CONTRADICTS = "contradicts"         # A and B cannot both hold
    REFINES = "refines"                 # B improves/tightens A
    COMPOSES_WITH = "composes_with"     # A + B → C
    MOTIVATES = "motivates"             # A suggests investigating B
    VERIFIED_BY = "verified_by"         # A is verified by experiment/computation B
    DUAL_TO = "dual_to"                # A and B are dual (in some formal sense)
    LIFTS_TO = "lifts_to"             # A in domain X lifts to B in domain Y
    REDUCES_TO = "reduces_to"          # A reduces to B in some limit


class VerificationMode(str, Enum):
    """How a theoretical claim can be verified."""
    FORMAL_PROOF = "formal_proof"       # Lean, Isabelle, Coq
    NUMERICAL = "numerical"             # High-precision computation
    SYMBOLIC_CAS = "symbolic_cas"       # Computer algebra (SymPy, Mathematica)
    DIMENSIONAL = "dimensional"         # Dimensional analysis (physics)
    CONSISTENCY = "consistency"         # Internal consistency check
    SYMMETRY = "symmetry"              # Symmetry/conservation law check
    LIMITING_CASE = "limiting_case"     # Reduce to known result in special case
    STATISTICAL = "statistical"         # Statistical test / Monte Carlo
    PEER_REVIEW = "peer_review"         # Human expert evaluation
    LLM_EVALUATION = "llm_evaluation"   # AI evaluation (lowest confidence)
    VLM_VISUAL = "vlm_visual"           # Visual/diagram verification via VLM (AI Scientist v2)


class ReasoningStrategy(str, Enum):
    """Strategies for generating new theoretical insights."""
    ANALOGY_TRANSFER = "analogy_transfer"       # If X works in A, does it work in B?
    GENERALIZATION = "generalization"             # Weaken assumptions → broader result
    SPECIALIZATION = "specialization"             # Strengthen assumptions → sharper result
    COMPOSITION = "composition"                   # Combine two results → new result
    CONTRAPOSITIVE = "contrapositive"             # Negate and reverse
    BOUNDARY_ANALYSIS = "boundary_analysis"       # What happens at the edges?
    DIMENSIONAL_LIFTING = "dimensional_lifting"   # Lift from n to n+1 dimensions
    DUALITY = "duality"                           # Apply a known duality transform
    DECOMPOSITION = "decomposition"               # Break into independent sub-problems
    UNIFICATION = "unification"                   # Find common structure in disparate results
    NUMERICAL_EXPLORATION = "numerical_exploration"  # Compute and discover patterns
    STRUCTURAL_INDUCTION = "structural_induction"   # Induct over structure, not just numbers


# ══════════════════════════════════════════════════════════════
# Core Data Structures
# ══════════════════════════════════════════════════════════════


@dataclass
class ConceptNode:
    """An atomic theoretical concept.

    This is richer than CapabilityNode — it's designed specifically for
    the structure of theoretical reasoning. A concept has:
      - A formal statement (LaTeX or natural language)
      - An informal explanation (intuition, motivation)
      - A domain and type (theorem in pure math, principle in physics, ...)
      - Verification status across multiple modes
      - Cross-references to dependencies and related concepts
    """
    id: str
    concept_type: ConceptType
    domain: ScientificDomain
    formal_statement: str                          # LaTeX or formal notation
    informal_statement: str = ""                   # Natural language explanation
    intuition: str = ""                            # Why this is true / what it means
    proof_sketch: str = ""                         # Informal proof idea
    formal_proof: str = ""                         # Formal proof (Lean, etc.)
    tags: list[str] = field(default_factory=list)
    sub_domain: str = ""                           # More specific (e.g., "analytic number theory")

    # Verification
    verification_status: dict[str, float] = field(default_factory=dict)
    # Maps VerificationMode.value → confidence (0-1)
    # e.g., {"formal_proof": 1.0, "numerical": 0.95, "consistency": 0.8}
    overall_confidence: float = 0.0

    # Provenance
    source_article: str = ""                       # Which article/paper this comes from
    source_section: str = ""
    generation_strategy: str = ""                  # How this was discovered
    parent_ids: list[str] = field(default_factory=list)

    # Metadata
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_confidence(self) -> float:
        """Recompute overall confidence from verification modes.

        Weights:
          formal_proof: 1.0 (if verified formally, it's true)
          numerical: 0.85
          symbolic_cas: 0.80
          consistency: 0.70
          limiting_case: 0.75
          dimensional: 0.65
          symmetry: 0.65
          statistical: 0.60
          peer_review: 0.50
          llm_evaluation: 0.35
        """
        weights = {
            "formal_proof": 1.0, "numerical": 0.85, "symbolic_cas": 0.80,
            "limiting_case": 0.75, "consistency": 0.70, "dimensional": 0.65,
            "symmetry": 0.65, "statistical": 0.60, "peer_review": 0.50,
            "vlm_visual": 0.45, "llm_evaluation": 0.35,
        }
        if not self.verification_status:
            self.overall_confidence = 0.0
            return 0.0

        # If formally proved, confidence = 1.0
        if self.verification_status.get("formal_proof", 0) >= 0.99:
            self.overall_confidence = 1.0
            return 1.0

        # Otherwise: weighted combination (diminishing returns for multiple modes)
        total_weight = 0.0
        weighted_sum = 0.0
        for mode, conf in self.verification_status.items():
            w = weights.get(mode, 0.3)
            total_weight += w
            weighted_sum += w * conf

        raw = weighted_sum / max(total_weight, 0.01)
        # Bonus for multiple verification modes (cross-validation)
        mode_count = sum(1 for c in self.verification_status.values() if c > 0.5)
        bonus = min(0.15, mode_count * 0.03)
        self.overall_confidence = min(0.99, raw + bonus)
        return self.overall_confidence

    # Backward-compatible aliases for legacy modules.
    @property
    def statement(self) -> str:
        return self.formal_statement

    @statement.setter
    def statement(self, value: str) -> None:
        self.formal_statement = value

    @property
    def description(self) -> str:
        return self.formal_statement or self.informal_statement

    @description.setter
    def description(self, value: str) -> None:
        self.formal_statement = value

    @property
    def label(self) -> str:
        legacy = str(self.metadata.get("legacy_label", "")).strip()
        if legacy:
            return legacy
        text = (self.informal_statement or self.formal_statement).strip()
        if not text:
            return self.id
        first_line = text.splitlines()[0].strip()
        return first_line[:120] if first_line else self.id

    @label.setter
    def label(self, value: str) -> None:
        self.metadata["legacy_label"] = value

    @property
    def name(self) -> str:
        return self.label

    @name.setter
    def name(self, value: str) -> None:
        self.label = value

    @property
    def type(self) -> ConceptType:
        return self.concept_type

    @type.setter
    def type(self, value: ConceptType | str) -> None:
        if isinstance(value, ConceptType):
            self.concept_type = value
            return
        try:
            self.concept_type = ConceptType(value)
        except ValueError:
            logger.debug("[ConceptNode] Unknown legacy type '%s', keeping current", value)

    @property
    def confidence(self) -> float:
        return self.overall_confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        self.overall_confidence = max(0.0, min(1.0, float(value)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "concept_type": self.concept_type.value,
            "domain": self.domain.value,
            "formal_statement": self.formal_statement,
            "informal_statement": self.informal_statement,
            "intuition": self.intuition,
            "proof_sketch": self.proof_sketch,
            "formal_proof": self.formal_proof,
            "tags": self.tags,
            "sub_domain": self.sub_domain,
            "verification_status": self.verification_status,
            "overall_confidence": self.overall_confidence,
            "source_article": self.source_article,
            "source_section": self.source_section,
            "generation_strategy": self.generation_strategy,
            "parent_ids": self.parent_ids,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConceptNode:
        return cls(
            id=data["id"],
            concept_type=ConceptType(data.get("concept_type", "theorem")),
            domain=ScientificDomain(data.get("domain", "general")),
            formal_statement=data.get("formal_statement", ""),
            informal_statement=data.get("informal_statement", ""),
            intuition=data.get("intuition", ""),
            proof_sketch=data.get("proof_sketch", ""),
            formal_proof=data.get("formal_proof", ""),
            tags=data.get("tags", []),
            sub_domain=data.get("sub_domain", ""),
            verification_status=data.get("verification_status", {}),
            overall_confidence=data.get("overall_confidence", 0.0),
            source_article=data.get("source_article", ""),
            source_section=data.get("source_section", ""),
            generation_strategy=data.get("generation_strategy", ""),
            parent_ids=data.get("parent_ids", []),
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConceptRelation:
    """A directed relation between two concepts."""
    source_id: str
    target_id: str
    relation_type: RelationType
    description: str = ""               # Why this relation holds
    strength: float = 1.0               # How strong the analogy/connection is
    bridging_insight: str = ""          # For ANALOGOUS_TO: what's the bridge?

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "description": self.description,
            "strength": self.strength,
            "bridging_insight": self.bridging_insight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConceptRelation:
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data.get("relation_type", "depends_on")),
            description=data.get("description", ""),
            strength=data.get("strength", 1.0),
            bridging_insight=data.get("bridging_insight", ""),
        )

    @property
    def id(self) -> str:
        return f"{self.source_id}:{self.relation_type.value}:{self.target_id}"

    @property
    def type(self) -> RelationType:
        return self.relation_type

    @type.setter
    def type(self, value: RelationType | str) -> None:
        if isinstance(value, RelationType):
            self.relation_type = value
            return
        try:
            self.relation_type = RelationType(value)
        except ValueError:
            logger.debug("[ConceptRelation] Unknown legacy type '%s', keeping current", value)


# ══════════════════════════════════════════════════════════════
# Theory Graph
# ══════════════════════════════════════════════════════════════


class TheoryGraph:
    """The knowledge structure of a theory or research article.

    A directed graph where nodes are theoretical concepts and edges
    are relations (dependency, analogy, generalization, ...).

    This represents the structure of a paper like the reference article:
    definitions → lemmas → theorems → cross-domain connections → conjectures.
    """

    def __init__(self, title: str = "", source: str = "") -> None:
        self.title = title
        self.source = source
        self._nodes: dict[str, ConceptNode] = {}
        self._relations: list[ConceptRelation] = []
        self._forward: dict[str, list[str]] = {}    # id → [target_ids]
        self._backward: dict[str, list[str]] = {}   # id → [source_ids]
        self._domain_index: dict[str, set[str]] = {}
        self._type_index: dict[str, set[str]] = {}

    @property
    def nodes(self) -> dict[str, ConceptNode]:
        """Legacy alias for concept dictionary."""
        return self._nodes

    @property
    def concepts(self) -> dict[str, ConceptNode]:
        """Legacy alias for concept dictionary."""
        return self._nodes

    @property
    def relations(self) -> list[ConceptRelation]:
        """Legacy alias for relation list."""
        return self._relations

    def add_concept(self, node: ConceptNode) -> None:
        """Add a concept to the theory."""
        previous = self._nodes.get(node.id)
        if previous is not None:
            old_domain = previous.domain.value
            old_type = previous.concept_type.value
            if old_domain in self._domain_index:
                self._domain_index[old_domain].discard(previous.id)
                if not self._domain_index[old_domain]:
                    self._domain_index.pop(old_domain, None)
            if old_type in self._type_index:
                self._type_index[old_type].discard(previous.id)
                if not self._type_index[old_type]:
                    self._type_index.pop(old_type, None)

        self._nodes[node.id] = node
        # Index by domain
        dk = node.domain.value
        if dk not in self._domain_index:
            self._domain_index[dk] = set()
        self._domain_index[dk].add(node.id)
        # Index by type
        tk = node.concept_type.value
        if tk not in self._type_index:
            self._type_index[tk] = set()
        self._type_index[tk].add(node.id)

    def add_relation(self, rel: ConceptRelation) -> None:
        """Add a relation between concepts."""
        if rel.source_id not in self._nodes or rel.target_id not in self._nodes:
            logger.debug(f"[TheoryGraph] Skipping relation: missing node(s)")
            return
        self._relations.append(rel)
        self._forward.setdefault(rel.source_id, []).append(rel.target_id)
        self._backward.setdefault(rel.target_id, []).append(rel.source_id)

    def add_node(self, node: ConceptNode) -> None:
        """Legacy alias for add_concept."""
        self.add_concept(node)

    def get_concept(self, node_id: str) -> ConceptNode | None:
        return self._nodes.get(node_id)

    def get_node(self, node_id: str) -> ConceptNode | None:
        """Legacy alias for get_concept."""
        return self.get_concept(node_id)

    def get_concepts_by_type(self, ctype: ConceptType) -> list[ConceptNode]:
        ids = self._type_index.get(ctype.value, set())
        return [self._nodes[i] for i in ids if i in self._nodes]

    def get_concepts_by_domain(self, domain: ScientificDomain) -> list[ConceptNode]:
        ids = self._domain_index.get(domain.value, set())
        return [self._nodes[i] for i in ids if i in self._nodes]

    def get_dependencies(self, node_id: str) -> list[ConceptNode]:
        """Get all concepts this one depends on (transitive)."""
        visited: set[str] = set()
        result: list[ConceptNode] = []

        def dfs(nid: str) -> None:
            if nid in visited:
                return
            visited.add(nid)
            for dep_id in self._backward.get(nid, []):
                if dep_id in self._nodes:
                    result.append(self._nodes[dep_id])
                    dfs(dep_id)
        dfs(node_id)
        return result

    def get_analogies(self) -> list[ConceptRelation]:
        """Get all cross-domain analogies."""
        return [
            r for r in self._relations
            if r.relation_type == RelationType.ANALOGOUS_TO
        ]

    def get_conjectures(self) -> list[ConceptNode]:
        """Get all unproved conjectures."""
        return [
            n for n in self._nodes.values()
            if n.concept_type in (ConceptType.CONJECTURE, ConceptType.OPEN_PROBLEM)
            and n.overall_confidence < 0.9
        ]

    def get_frontier(self) -> list[ConceptNode]:
        """Get leaf nodes — theorems/results with no dependents.

        These are the current frontier of the theory,
        and the natural starting points for evolution.
        """
        non_leaf = set()
        for rel in self._relations:
            non_leaf.add(rel.source_id)
        return [
            n for nid, n in self._nodes.items()
            if nid not in non_leaf
            and n.concept_type in (ConceptType.THEOREM, ConceptType.PROPOSITION,
                                    ConceptType.CONJECTURE, ConceptType.COROLLARY)
        ]

    def get_cross_domain_bridges(self) -> list[tuple[ConceptNode, ConceptNode, ConceptRelation]]:
        """Find all pairs of concepts from different domains connected by analogy."""
        bridges = []
        for rel in self._relations:
            if rel.relation_type in (RelationType.ANALOGOUS_TO, RelationType.LIFTS_TO,
                                      RelationType.DUAL_TO):
                src = self._nodes.get(rel.source_id)
                tgt = self._nodes.get(rel.target_id)
                if src and tgt and src.domain != tgt.domain:
                    bridges.append((src, tgt, rel))
        return bridges

    @property
    def size(self) -> int:
        return len(self._nodes)

    def get_stats(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "total_concepts": len(self._nodes),
            "total_relations": len(self._relations),
            "by_type": {
                t: len(ids) for t, ids in self._type_index.items()
            },
            "by_domain": {
                d: len(ids) for d, ids in self._domain_index.items()
            },
            "conjectures": len(self.get_conjectures()),
            "cross_domain_bridges": len(self.get_cross_domain_bridges()),
        }

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "title": self.title,
            "source": self.source,
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "relations": [r.to_dict() for r in self._relations],
        }
        (path / "theory_graph.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info(f"[TheoryGraph] Saved '{self.title}': "
                     f"{len(self._nodes)} concepts, {len(self._relations)} relations")

    def load(self, path: Path) -> None:
        fp = path / "theory_graph.json"
        if not fp.exists():
            return
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            self.title = data.get("title", "")
            self.source = data.get("source", "")
            for nid, ndata in data.get("nodes", {}).items():
                self.add_concept(ConceptNode.from_dict(ndata))
            for rdata in data.get("relations", []):
                self.add_relation(ConceptRelation.from_dict(rdata))
            logger.info(f"[TheoryGraph] Loaded '{self.title}': "
                         f"{len(self._nodes)} concepts")
        except Exception as e:
            logger.warning(f"[TheoryGraph] Failed to load: {e}")


# ══════════════════════════════════════════════════════════════
# Multi-Modal Verification Suite
# ══════════════════════════════════════════════════════════════


class VerificationSuite:
    """Multi-modal verification for theoretical claims.

    Different domains require different verification:
      - Pure math: formal proof (Lean) + numerical check
      - Physics: dimensional analysis + limiting cases + conservation laws
      - Biology: consistency with known mechanisms + statistical evidence
      - All: LLM evaluation as universal fallback (lowest confidence)

    The suite runs all applicable modes and aggregates confidence.
    """

    async def verify(
        self,
        concept: ConceptNode,
        llm: Any,
        *,
        modes: list[VerificationMode] | None = None,
    ) -> dict[str, float]:
        """Verify a concept through multiple modes.

        Returns dict of {mode: confidence} for each attempted mode.
        """
        if modes is None:
            modes = self._select_modes(concept)

        results: dict[str, float] = {}
        for mode in modes:
            try:
                confidence = await self._verify_one(concept, mode, llm)
                results[mode.value] = confidence
            except Exception as e:
                logger.debug(f"[Verify] {mode.value} failed for {concept.id[:8]}: {e}")

        concept.verification_status.update(results)
        concept.update_confidence()
        return results

    def _select_modes(self, concept: ConceptNode) -> list[VerificationMode]:
        """Select appropriate verification modes based on domain and type."""
        modes = [VerificationMode.LLM_EVALUATION]  # Always available

        if concept.domain in (ScientificDomain.PURE_MATHEMATICS,
                               ScientificDomain.APPLIED_MATHEMATICS):
            modes.extend([
                VerificationMode.CONSISTENCY,
                VerificationMode.NUMERICAL,
                VerificationMode.LIMITING_CASE,
            ])
            if concept.concept_type in (ConceptType.THEOREM, ConceptType.LEMMA):
                modes.append(VerificationMode.FORMAL_PROOF)

        elif concept.domain in (ScientificDomain.THEORETICAL_PHYSICS,
                                 ScientificDomain.MATHEMATICAL_PHYSICS):
            modes.extend([
                VerificationMode.DIMENSIONAL,
                VerificationMode.SYMMETRY,
                VerificationMode.LIMITING_CASE,
                VerificationMode.NUMERICAL,
                VerificationMode.CONSISTENCY,
            ])

        elif concept.domain == ScientificDomain.STATISTICAL_MECHANICS:
            modes.extend([
                VerificationMode.STATISTICAL,
                VerificationMode.NUMERICAL,
                VerificationMode.LIMITING_CASE,
                VerificationMode.CONSISTENCY,
            ])

        else:
            modes.extend([VerificationMode.CONSISTENCY, VerificationMode.NUMERICAL])

        # VLM visual verification: available for any concept with formal statements
        # containing equations, diagrams, or formulas (AI Scientist v2 idea)
        if any(kw in concept.formal_statement.lower() for kw in [
            "\\frac", "\\sum", "\\int", "\\lim", "equation", "diagram",
            "figure", "graph", "plot", "matrix", "\\begin{",
        ]):
            modes.append(VerificationMode.VLM_VISUAL)

        return modes

    async def _verify_one(
        self,
        concept: ConceptNode,
        mode: VerificationMode,
        llm: Any,
    ) -> float:
        """Run a single verification mode. Returns confidence 0-1."""
        from autoforge.engine.llm_router import TaskComplexity

        prompts = {
            VerificationMode.LLM_EVALUATION: self._prompt_llm_eval,
            VerificationMode.CONSISTENCY: self._prompt_consistency,
            VerificationMode.NUMERICAL: self._prompt_numerical,
            VerificationMode.DIMENSIONAL: self._prompt_dimensional,
            VerificationMode.SYMMETRY: self._prompt_symmetry,
            VerificationMode.LIMITING_CASE: self._prompt_limiting_case,
            VerificationMode.STATISTICAL: self._prompt_statistical,
            VerificationMode.VLM_VISUAL: self._prompt_vlm_visual,
        }

        prompt_fn = prompts.get(mode)
        if not prompt_fn:
            return 0.0

        prompt = prompt_fn(concept)
        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system=f"You are a verification expert. Mode: {mode.value}. "
                       f"Return ONLY JSON: {{\"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text
            data = _extract_json(text)
            if data and "confidence" in data:
                return min(1.0, max(0.0, float(data["confidence"])))
        except Exception as e:
            logger.debug(f"[Verify] {mode.value} error: {e}")
        return 0.0

    @staticmethod
    def _prompt_llm_eval(concept: ConceptNode) -> str:
        return f"""Evaluate the correctness and plausibility of this {concept.domain.value} {concept.concept_type.value}.

## Formal Statement
{concept.formal_statement[:3000]}

## Informal Description
{concept.informal_statement[:1000]}

{f"## Proof Sketch: {concept.proof_sketch[:1000]}" if concept.proof_sketch else ""}

Check for:
1. Logical consistency — does the statement make sense?
2. Known results — does this contradict any established results?
3. Plausibility — is this the kind of result that could be true?
4. Completeness — are all necessary conditions stated?

Return JSON: {{"confidence": 0.0-1.0, "reasoning": "your analysis"}}"""

    @staticmethod
    def _prompt_consistency(concept: ConceptNode) -> str:
        return f"""Check the internal consistency of this {concept.concept_type.value}.

## Statement
{concept.formal_statement[:3000]}

Check:
1. Are the hypotheses compatible with each other?
2. Does the conclusion follow from the hypotheses in principle?
3. Are there obvious counterexamples?
4. Does the notation/terminology self-consistent?
5. Are edge cases handled (empty sets, zero, infinity)?

Return JSON: {{"confidence": 0.0-1.0, "reasoning": "your analysis"}}"""

    @staticmethod
    def _prompt_numerical(concept: ConceptNode) -> str:
        return f"""Design a numerical check for this {concept.concept_type.value}.

## Statement
{concept.formal_statement[:3000]}

Generate a Python snippet that numerically tests this claim for small/concrete cases.
Then evaluate whether the numerical evidence supports the claim.

Return JSON: {{
  "confidence": 0.0-1.0,
  "python_check": "code that tests this",
  "test_results": "what the code would show",
  "reasoning": "analysis"
}}"""

    @staticmethod
    def _prompt_dimensional(concept: ConceptNode) -> str:
        return f"""Perform dimensional analysis on this {concept.concept_type.value}.

## Statement
{concept.formal_statement[:3000]}

Check:
1. Do both sides of equations have the same dimensions/units?
2. Are dimensionless quantities correctly identified?
3. Do scaling arguments support the claimed relationships?

Return JSON: {{"confidence": 0.0-1.0, "reasoning": "your analysis"}}"""

    @staticmethod
    def _prompt_symmetry(concept: ConceptNode) -> str:
        return f"""Check symmetry and conservation law consistency of this {concept.concept_type.value}.

## Statement
{concept.formal_statement[:3000]}

Check:
1. Does the result respect expected symmetries (time reversal, parity, gauge, etc.)?
2. Are conservation laws satisfied?
3. Is covariance preserved under relevant transformations?
4. Do boundary conditions respect the symmetries?

Return JSON: {{"confidence": 0.0-1.0, "reasoning": "your analysis"}}"""

    @staticmethod
    def _prompt_limiting_case(concept: ConceptNode) -> str:
        return f"""Check limiting cases for this {concept.concept_type.value}.

## Statement
{concept.formal_statement[:3000]}

Check:
1. What happens in trivial/degenerate cases (n=0, n=1, dimension=1)?
2. Does the result reduce to known results in special limits?
3. What happens as parameters → 0 or → ∞?
4. Are there known special cases that can be directly checked?

Return JSON: {{"confidence": 0.0-1.0, "reasoning": "your analysis"}}"""

    @staticmethod
    def _prompt_statistical(concept: ConceptNode) -> str:
        return f"""Design a statistical test for this {concept.concept_type.value}.

## Statement
{concept.formal_statement[:3000]}

Design a Monte Carlo or statistical test:
1. What random variables/distributions are relevant?
2. What sample size would be needed?
3. What test statistic captures the claim?
4. What p-value threshold would support or reject it?

Return JSON: {{"confidence": 0.0-1.0, "reasoning": "your analysis"}}"""

    @staticmethod
    def _prompt_vlm_visual(concept: ConceptNode) -> str:
        """VLM visual verification prompt (AI Scientist v2, Sakana ICLR 2025).

        Asks the LLM to "render" the mathematical structure mentally and
        check for visual/structural correctness. When used with a multimodal
        model, this can check LaTeX rendering, graph structures, and diagram
        consistency without needing an actual renderer.
        """
        return f"""Visually verify this {concept.concept_type.value} from {concept.domain.value}.

## Formal Statement
{concept.formal_statement[:3000]}

## Informal Description
{concept.informal_statement[:1000]}

Perform a "visual inspection" of the mathematical/scientific structure:
1. **Notation consistency**: Are symbols used consistently? Any overloaded notation?
2. **Structural correctness**: If this involves a diagram, graph, or commutative diagram,
   does the structure make sense? Do arrows compose correctly?
3. **Dimensional/type correctness**: Do the types on both sides of equations match?
4. **Boundary visualization**: At extreme values (0, ∞, ε→0), does the expression
   behave as expected visually?
5. **Pattern recognition**: Does this look similar to known results? If so, which?

Return JSON: {{"confidence": 0.0-1.0, "reasoning": "your visual analysis"}}"""


# ══════════════════════════════════════════════════════════════
# Reasoning Engine — The Core
# ══════════════════════════════════════════════════════════════


class ReasoningEngine:
    """Multi-strategy theoretical reasoning engine.

    Given a set of known concepts, generates new theoretical insights
    through structured reasoning strategies. This is the intellectual
    core — the part that "thinks like a theorist."

    Each strategy mirrors how real theorists work:
      - ANALOGY_TRANSFER: "This looks like X in domain A. Does it work in B?"
      - GENERALIZATION: "Theorem holds for Z. Does it hold for all rings?"
      - COMPOSITION: "Combining lemma A and theorem B gives us..."
      - BOUNDARY_ANALYSIS: "What happens as ε→0? As n→∞?"
      - UNIFICATION: "Results X, Y, Z all share structure S. Can we unify?"

    Inspired by:
      - Graph of Thought (DAG-structured reasoning)
      - SciAgent (multi-agent tool-augmented reasoning)
      - LacMaterial (explicit analogical reasoning)
    """

    def __init__(self, verification: VerificationSuite | None = None) -> None:
        self._verification = verification or VerificationSuite()
        self._reasoning_history: list[dict[str, Any]] = []

    async def reason(
        self,
        strategy: ReasoningStrategy,
        source_concepts: list[ConceptNode],
        llm: Any,
        *,
        target_domain: ScientificDomain | None = None,
        context: str = "",
        num_candidates: int = 3,
    ) -> list[ConceptNode]:
        """Apply a reasoning strategy to generate new concepts.

        Args:
            strategy: Which reasoning strategy to use
            source_concepts: Input concepts to reason from
            llm: LLM interface
            target_domain: For cross-domain strategies, which domain to target
            context: Additional context (e.g., from the theory graph)
            num_candidates: How many candidate concepts to generate

        Returns:
            List of new ConceptNodes (verified, with confidence scores)
        """
        from autoforge.engine.llm_router import TaskComplexity

        prompt = self._build_reasoning_prompt(
            strategy, source_concepts, target_domain, context, num_candidates,
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system=self._system_prompt(strategy),
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            # Parse generated concepts
            candidates = self._parse_concepts(text, strategy, source_concepts)

            # Verify each candidate
            verified: list[ConceptNode] = []
            for candidate in candidates:
                await self._verification.verify(candidate, llm)
                if candidate.overall_confidence > 0.15:  # Minimal threshold
                    verified.append(candidate)

            # Record reasoning history
            self._reasoning_history.append({
                "strategy": strategy.value,
                "source_ids": [c.id for c in source_concepts],
                "generated": len(candidates),
                "verified": len(verified),
                "timestamp": time.time(),
            })

            logger.info(f"[Reasoning] {strategy.value}: "
                        f"{len(verified)}/{len(candidates)} candidates verified")
            return verified

        except Exception as e:
            logger.warning(f"[Reasoning] {strategy.value} failed: {e}")
            return []

    def _system_prompt(self, strategy: ReasoningStrategy) -> str:
        base = ("You are a theoretical scientist with expertise across mathematics, "
                "physics, chemistry, and biology. You reason at the level of a "
                "research mathematician — rigorous, creative, and cross-disciplinary.")

        strategy_prompts = {
            ReasoningStrategy.ANALOGY_TRANSFER:
                f"{base} You specialize in finding structural isomorphisms across "
                "scientific domains — the kind of deep analogies that connect "
                "number theory to statistical mechanics, or topology to quantum field theory.",

            ReasoningStrategy.GENERALIZATION:
                f"{base} You specialize in finding the most general setting in which "
                "a result holds — weakening assumptions while preserving the core insight.",

            ReasoningStrategy.COMPOSITION:
                f"{base} You specialize in combining existing results to derive new ones — "
                "seeing how theorem A and lemma B together imply something unexpected.",

            ReasoningStrategy.UNIFICATION:
                f"{base} You specialize in finding hidden common structure across "
                "seemingly unrelated results — the kind of insight that leads to "
                "new mathematical frameworks.",

            ReasoningStrategy.BOUNDARY_ANALYSIS:
                f"{base} You specialize in edge cases, limiting behaviors, and "
                "phase transitions — the boundary is where the interesting physics lives.",

            ReasoningStrategy.NUMERICAL_EXPLORATION:
                f"{base} You specialize in computational experiments that reveal "
                "unexpected patterns — the modern Ramanujan approach to discovery.",
        }
        return strategy_prompts.get(strategy, base)

    def _build_reasoning_prompt(
        self,
        strategy: ReasoningStrategy,
        sources: list[ConceptNode],
        target_domain: ScientificDomain | None,
        context: str,
        num_candidates: int,
    ) -> str:
        source_text = ""
        for i, c in enumerate(sources[:8]):
            source_text += (
                f"\n### Concept {i+1} [{c.domain.value}, {c.concept_type.value}]\n"
                f"**Statement**: {c.formal_statement[:800]}\n"
                f"**Intuition**: {c.intuition[:300]}\n"
                f"**Confidence**: {c.overall_confidence:.2f}\n"
            )

        strategy_instructions = {
            ReasoningStrategy.ANALOGY_TRANSFER: f"""
Find structural analogies between the given concepts and {target_domain.value if target_domain else 'other scientific domains'}.

Look for:
- Shared mathematical structures (groups, algebras, topological spaces)
- Parallel logical patterns (induction ↔ recursion, duality ↔ Fourier transform)
- Conceptual isomorphisms (entropy ↔ information, energy ↔ norm, symmetry ↔ conservation)

For each analogy, explicitly state:
1. What structure in domain A corresponds to what in domain B
2. Why the correspondence holds (the bridging insight)
3. What new results this suggests in the target domain""",

            ReasoningStrategy.GENERALIZATION: """
Generalize the given results by weakening assumptions.

For each generalization:
1. Identify which assumption is being relaxed
2. State the new, more general result
3. Explain why the proof technique still works (or what modifications are needed)
4. Note what new examples are covered""",

            ReasoningStrategy.COMPOSITION: """
Combine the given concepts to derive new results.

Look for:
- Theorem A provides input X; Theorem B consumes exactly X
- Lemma A gives an upper bound; Lemma B gives a matching lower bound → exact result
- Construction A + Property B → Application C
- Technique from proof A applied to statement B""",

            ReasoningStrategy.UNIFICATION: """
Find the common mathematical structure underlying the given concepts.

Look for:
- Shared algebraic structure (all instances of a general category)
- Common proof patterns (all provable by the same technique)
- Parallel definitions (all defined by the same universal property)
- Hidden symmetry that explains why all these results hold""",

            ReasoningStrategy.BOUNDARY_ANALYSIS: """
Analyze boundary/limiting behavior of the given concepts.

Check:
- What happens as key parameters → 0 or → ∞?
- Are there phase transitions at critical values?
- What are the degenerate cases and what do they tell us?
- Are the stated bounds tight? Can we find matching examples?""",

            ReasoningStrategy.NUMERICAL_EXPLORATION: """
Design computational experiments to explore the given concepts.

For each experiment:
1. What quantity to compute
2. Expected behavior (if the concept is correct)
3. What surprising patterns might emerge
4. How to interpret the results""",

            ReasoningStrategy.DIMENSIONAL_LIFTING: """
Lift the given results to higher dimensions or more general settings.

For each lifting:
1. What is the natural higher-dimensional analogue?
2. Which properties survive the lifting and which don't?
3. Are there new phenomena in higher dimensions (e.g., new invariants)?""",

            ReasoningStrategy.DUALITY: """
Apply duality transforms to the given concepts.

Look for:
1. Fourier duality, Pontryagin duality, Poincaré duality
2. Langlands-type correspondences
3. Mirror symmetry, electric-magnetic duality
4. What does the dual picture reveal that the primal doesn't?""",
        }

        instruction = strategy_instructions.get(
            strategy,
            "Generate new theoretical insights from the given concepts.",
        )

        return f"""## Source Concepts
{source_text}

{f"## Additional Context: {context[:2000]}" if context else ""}

## Task: {strategy.value}
{instruction}

## Output Format
Generate {num_candidates} new concepts. Return JSON array:
[
  {{
    "concept_type": "theorem|conjecture|observation|analogy|...",
    "domain": "{target_domain.value if target_domain else 'infer from content'}",
    "formal_statement": "precise mathematical/scientific statement",
    "informal_statement": "natural language explanation",
    "intuition": "why this should be true / what insight it captures",
    "proof_sketch": "how one might prove this (if applicable)",
    "tags": ["tag1", "tag2"],
    "sub_domain": "specific sub-field",
    "parent_indices": [0, 1],
    "bridging_insight": "for analogies: what's the structural bridge"
  }}
]"""

    def _parse_concepts(
        self,
        text: str,
        strategy: ReasoningStrategy,
        sources: list[ConceptNode],
    ) -> list[ConceptNode]:
        """Parse LLM output into ConceptNodes."""
        concepts: list[ConceptNode] = []
        try:
            if "[" not in text:
                return []
            json_str = text[text.index("["):text.rindex("]") + 1]
            items = json.loads(json_str)
            for item in items:
                if not isinstance(item, dict) or "formal_statement" not in item:
                    continue
                # Resolve parent IDs
                parent_ids = []
                for idx in item.get("parent_indices", []):
                    if isinstance(idx, int) and 0 <= idx < len(sources):
                        parent_ids.append(sources[idx].id)

                cid = hashlib.sha256(
                    item["formal_statement"].encode()
                ).hexdigest()[:12]

                try:
                    ctype = ConceptType(item.get("concept_type", "conjecture"))
                except ValueError:
                    ctype = ConceptType.CONJECTURE
                try:
                    domain = ScientificDomain(item.get("domain", "general"))
                except ValueError:
                    domain = ScientificDomain.GENERAL

                node = ConceptNode(
                    id=cid,
                    concept_type=ctype,
                    domain=domain,
                    formal_statement=item["formal_statement"],
                    informal_statement=item.get("informal_statement", ""),
                    intuition=item.get("intuition", ""),
                    proof_sketch=item.get("proof_sketch", ""),
                    tags=item.get("tags", []),
                    sub_domain=item.get("sub_domain", ""),
                    generation_strategy=strategy.value,
                    parent_ids=parent_ids,
                    metadata={"bridging_insight": item.get("bridging_insight", "")},
                )
                concepts.append(node)
        except Exception as e:
            logger.debug(f"[Reasoning] Parse failed: {e}")
        return concepts


# ══════════════════════════════════════════════════════════════
# Article Parser — Extract Theory Graph from Papers
# ══════════════════════════════════════════════════════════════


class ArticleParser:
    """Extract a TheoryGraph from a research article.

    This is how existing knowledge enters the system: you feed in a paper
    (or LaTeX source), and the parser extracts:
      - All definitions, theorems, lemmas, conjectures
      - Their dependency structure
      - Cross-domain connections
      - Open problems

    The output TheoryGraph becomes input for the ReasoningEngine and TheoryEvolver.
    """

    async def parse_article(
        self,
        text: str,
        llm: Any,
        *,
        title: str = "",
        max_sections: int = 50,
    ) -> TheoryGraph:
        """Parse an article into a TheoryGraph."""
        from autoforge.engine.llm_router import TaskComplexity

        graph = TheoryGraph(title=title, source="article_parser")

        # Phase 1: Extract high-level structure
        sections = await self._extract_sections(text, llm, max_sections)
        logger.info(f"[ArticleParser] Extracted {len(sections)} sections")

        # Phase 2: Extract concepts from each section
        all_concepts: list[dict[str, Any]] = []
        for section in sections:
            concepts = await self._extract_concepts(section, llm)
            all_concepts.extend(concepts)
        logger.info(f"[ArticleParser] Extracted {len(all_concepts)} concepts")

        # Phase 3: Build nodes
        idx_to_id: dict[int, str] = {}
        for i, cdata in enumerate(all_concepts):
            cid = hashlib.sha256(
                cdata.get("formal_statement", str(i)).encode()
            ).hexdigest()[:12]
            idx_to_id[i] = cid

            try:
                ctype = ConceptType(cdata.get("type", "theorem"))
            except ValueError:
                ctype = ConceptType.THEOREM
            try:
                domain = ScientificDomain(cdata.get("domain", "general"))
            except ValueError:
                domain = ScientificDomain.GENERAL

            node = ConceptNode(
                id=cid,
                concept_type=ctype,
                domain=domain,
                formal_statement=cdata.get("formal_statement", ""),
                informal_statement=cdata.get("informal_statement", ""),
                intuition=cdata.get("intuition", ""),
                proof_sketch=cdata.get("proof_sketch", ""),
                tags=cdata.get("tags", []),
                sub_domain=cdata.get("sub_domain", ""),
                source_article=title,
                source_section=cdata.get("section", ""),
            )
            graph.add_concept(node)

        # Phase 4: Extract relations
        relations = await self._extract_relations(all_concepts, llm)
        for rdata in relations:
            src_idx = rdata.get("source_index", -1)
            tgt_idx = rdata.get("target_index", -1)
            if src_idx in idx_to_id and tgt_idx in idx_to_id:
                try:
                    rtype = RelationType(rdata.get("relation_type", "depends_on"))
                except ValueError:
                    rtype = RelationType.DEPENDS_ON
                rel = ConceptRelation(
                    source_id=idx_to_id[src_idx],
                    target_id=idx_to_id[tgt_idx],
                    relation_type=rtype,
                    description=rdata.get("description", ""),
                    bridging_insight=rdata.get("bridging_insight", ""),
                )
                graph.add_relation(rel)

        logger.info(f"[ArticleParser] Built theory graph: "
                     f"{graph.size} concepts, {len(graph._relations)} relations")
        return graph

    async def _extract_sections(
        self, text: str, llm: Any, max_sections: int,
    ) -> list[dict[str, str]]:
        """Extract section structure from article text."""
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""Extract the section structure of this research article.
For each section, provide the title and a brief content summary.

## Article Text (first 10000 chars)
{text[:10000]}

Return JSON array of sections:
[
  {{"title": "section title", "content_summary": "what this section contains", "start_indicator": "first few words"}}
]
Limit to {max_sections} most important sections."""

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You extract the structure of scientific articles.",
                messages=[{"role": "user", "content": prompt}],
            )
            result_text = ""
            for block in response.content:
                if block.type == "text":
                    result_text += block.text
            if "[" in result_text:
                json_str = result_text[result_text.index("["):result_text.rindex("]") + 1]
                return json.loads(json_str)
        except Exception as e:
            logger.debug(f"[ArticleParser] Section extraction failed: {e}")
        return [{"title": "Full Article", "content_summary": text[:5000]}]

    async def _extract_concepts(
        self, section: dict[str, str], llm: Any,
    ) -> list[dict[str, Any]]:
        """Extract mathematical/scientific concepts from a section."""
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""Extract all formal mathematical/scientific concepts from this section.

## Section: {section.get('title', '')}
{section.get('content_summary', '')[:5000]}

For each concept (definition, theorem, lemma, conjecture, observation, etc.), extract:
1. type: definition, axiom, lemma, proposition, theorem, corollary, conjecture, observation, principle, analogy, algorithm
2. domain: pure_mathematics, theoretical_physics, statistical_mechanics, etc.
3. formal_statement: the precise statement
4. informal_statement: natural language explanation
5. intuition: why this is true or interesting
6. proof_sketch: brief proof idea (if given)
7. tags: relevant keywords
8. sub_domain: specific sub-field
9. dependencies: list of concept names this depends on

Return JSON array. Only include concepts that make specific, verifiable claims."""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You extract formal concepts from scientific articles with precision.",
                messages=[{"role": "user", "content": prompt}],
            )
            result_text = ""
            for block in response.content:
                if block.type == "text":
                    result_text += block.text
            if "[" in result_text:
                json_str = result_text[result_text.index("["):result_text.rindex("]") + 1]
                items = json.loads(json_str)
                for item in items:
                    item["section"] = section.get("title", "")
                return items
        except Exception as e:
            logger.debug(f"[ArticleParser] Concept extraction failed: {e}")
        return []

    async def _extract_relations(
        self, concepts: list[dict[str, Any]], llm: Any,
    ) -> list[dict[str, Any]]:
        """Extract relations between concepts."""
        from autoforge.engine.llm_router import TaskComplexity

        concept_list = ""
        for i, c in enumerate(concepts[:60]):  # Limit for context window
            concept_list += f"\n{i}. [{c.get('type', '?')}] {c.get('formal_statement', '')[:150]}"

        prompt = f"""Given these extracted concepts, identify the relations between them.

## Concepts
{concept_list}

## Relation Types
- depends_on: B requires A
- generalizes: B generalizes A
- specializes: B is a special case of A
- analogous_to: structural similarity across domains
- refines: B improves/tightens A
- composes_with: A + B → C
- motivates: A suggests investigating B
- dual_to: A and B are dual
- lifts_to: A in domain X lifts to B in domain Y
- reduces_to: A reduces to B in some limit

Return JSON array:
[
  {{"source_index": 0, "target_index": 3, "relation_type": "depends_on", "description": "why", "bridging_insight": ""}}
]
Focus on the most important structural relations, especially cross-domain analogies."""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You identify structural relations between scientific concepts.",
                messages=[{"role": "user", "content": prompt}],
            )
            result_text = ""
            for block in response.content:
                if block.type == "text":
                    result_text += block.text
            if "[" in result_text:
                json_str = result_text[result_text.index("["):result_text.rindex("]") + 1]
                return json.loads(json_str)
        except Exception as e:
            logger.debug(f"[ArticleParser] Relation extraction failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════
# Theory Evolver — Branch, Extend, and Compose Theories
# ══════════════════════════════════════════════════════════════


class TheoryEvolver:
    """Evolve existing theories by generating new branches and extensions.

    This is the engine that takes an existing TheoryGraph (like the user's
    reference article) and generates new research directions:
      - What if we apply this framework to a different domain?
      - What happens if we relax assumption X?
      - Can we combine this with theory Y?
      - What are the open boundary cases?

    Each evolution step produces a new TheoryGraph that branches from the original.
    Over time, this creates a tree of evolving theories.

    Inspired by:
      - FunSearch: evolutionary search for novel mathematical objects
      - AlphaEvolve: LLM-guided evolutionary coding
      - STP: self-play for autonomous theorem discovery
    """

    def __init__(self) -> None:
        self._reasoning = ReasoningEngine()
        self._parser = ArticleParser()
        self._verification = VerificationSuite()
        self._evolution_history: list[dict[str, Any]] = []
        # Bayesian Surprise tracking (AutoDiscovery, Ai2 2026)
        # Prior belief: uniform distribution over strategy-domain success
        self._strategy_stats: dict[str, dict[str, float]] = {}  # strategy → {attempts, successes, surprise_sum}

    def _bayesian_surprise(self, branch: "TheoryGraph", source: "TheoryGraph") -> float:
        """Compute Bayesian Surprise — how much a branch updates our beliefs.

        Surprise = KL divergence between posterior and prior, approximated as:
          - Novel concepts (new types/domains vs source)
          - Unexpected cross-domain bridges
          - Confidence above prior expectation

        Inspired by AutoDiscovery (Ai2, 2026.02): use surprise as MCTS reward
        to guide hypothesis generation toward genuinely novel directions.
        """
        if branch.size <= 0:
            return 0.0

        source_domains = {n.domain for n in source._nodes.values()}
        source_types = {n.concept_type for n in source._nodes.values()}

        surprise = 0.0
        new_count = 0

        for node in branch._nodes.values():
            if node.id not in source._nodes:
                new_count += 1
                # Domain novelty: higher surprise for concepts in new domains
                if node.domain not in source_domains:
                    surprise += 2.0
                # Type novelty: conjectures and analogies are more surprising
                if node.concept_type not in source_types:
                    surprise += 1.0
                if node.concept_type in (ConceptType.CONJECTURE, ConceptType.ANALOGY):
                    surprise += 1.5
                # Confidence surprise: high confidence on new concepts is unexpected
                if node.overall_confidence > 0.6:
                    surprise += node.overall_confidence

        # Cross-domain bridges are high surprise
        bridges = branch.get_cross_domain_bridges()
        source_bridges = source.get_cross_domain_bridges()
        new_bridges = len(bridges) - len(source_bridges)
        surprise += new_bridges * 3.0

        # Normalize by new concept count (avoid division by zero)
        return surprise / max(new_count, 1)

    def _select_strategy_by_surprise(
        self,
        strategies: list["ReasoningStrategy"],
        target_domains: list["ScientificDomain"],
    ) -> tuple["ReasoningStrategy", "ScientificDomain"]:
        """Select strategy and domain using Thompson Sampling on surprise scores.

        Strategies that have historically produced high-surprise branches
        are sampled more often, with exploration via Beta distribution.
        """
        import random

        best_score = -1.0
        best_strategy = strategies[0]
        best_domain = target_domains[0]

        for s in strategies:
            for d in target_domains:
                key = f"{s.value}:{d.value}"
                stats = self._strategy_stats.get(key, {"attempts": 0, "successes": 0, "surprise_sum": 0.0})
                alpha = max(1.0, stats.get("successes", 0) + 1)
                beta_val = max(1.0, stats.get("attempts", 0) - stats.get("successes", 0) + 1)
                # Thompson sample: higher surprise history → higher sample
                sample = random.betavariate(alpha, beta_val)
                # Boost by average surprise
                avg_surprise = stats.get("surprise_sum", 0) / max(stats.get("attempts", 0), 1)
                score = sample * (1.0 + avg_surprise)

                if score > best_score:
                    best_score = score
                    best_strategy = s
                    best_domain = d

        return best_strategy, best_domain

    def _record_surprise(self, strategy: "ReasoningStrategy", domain: "ScientificDomain", surprise: float) -> None:
        """Update Bayesian surprise statistics for a strategy-domain pair."""
        key = f"{strategy.value}:{domain.value}"
        if key not in self._strategy_stats:
            self._strategy_stats[key] = {"attempts": 0, "successes": 0, "surprise_sum": 0.0}
        self._strategy_stats[key]["attempts"] += 1
        self._strategy_stats[key]["surprise_sum"] += surprise
        if surprise > 1.0:  # Threshold for "successful" surprise
            self._strategy_stats[key]["successes"] += 1

    async def evolve(
        self,
        source_theory: TheoryGraph,
        strategy: ReasoningStrategy,
        llm: Any,
        *,
        target_domain: ScientificDomain | None = None,
        focus_concepts: list[str] | None = None,
        num_new_concepts: int = 5,
    ) -> TheoryGraph:
        """Generate a new theory branch from an existing theory.

        Args:
            source_theory: The theory to evolve from
            strategy: Which reasoning strategy to apply
            llm: LLM interface
            target_domain: For cross-domain evolution
            focus_concepts: Specific concept IDs to focus on (default: frontier)
            num_new_concepts: How many new concepts to generate

        Returns:
            A new TheoryGraph containing the evolved theory branch
        """
        # Select source concepts
        if focus_concepts:
            sources = [
                source_theory.get_concept(cid) for cid in focus_concepts
                if source_theory.get_concept(cid) is not None
            ]
        else:
            # Default: use the frontier (leaf theorems/conjectures)
            sources = source_theory.get_frontier()[:10]
            if not sources:
                # Fallback: highest-confidence concepts
                all_concepts = list(source_theory._nodes.values())
                all_concepts.sort(key=lambda c: c.overall_confidence, reverse=True)
                sources = all_concepts[:10]

        if not sources:
            logger.warning("[Evolver] No source concepts for evolution")
            return TheoryGraph(title=f"Empty evolution of {source_theory.title}")

        # Build context from the theory's cross-domain bridges
        bridges = source_theory.get_cross_domain_bridges()
        bridge_context = ""
        for src, tgt, rel in bridges[:5]:
            bridge_context += (
                f"\n- [{src.domain.value}] {src.informal_statement[:100]} "
                f"↔ [{tgt.domain.value}] {tgt.informal_statement[:100]}"
                f"\n  Bridge: {rel.bridging_insight[:200]}"
            )

        context = f"Cross-domain bridges in source theory:{bridge_context}" if bridge_context else ""

        # Generate new concepts
        new_concepts = await self._reasoning.reason(
            strategy=strategy,
            source_concepts=sources,
            llm=llm,
            target_domain=target_domain,
            context=context,
            num_candidates=num_new_concepts,
        )

        # Build the evolved theory graph
        branch_title = (
            f"{source_theory.title} → {strategy.value}"
            f"{f' → {target_domain.value}' if target_domain else ''}"
        )
        branch = TheoryGraph(title=branch_title, source="theory_evolver")

        # Copy relevant source concepts
        for src in sources:
            branch.add_concept(src)

        # Add new concepts
        for nc in new_concepts:
            branch.add_concept(nc)
            # Add DERIVED_FROM relations
            for pid in nc.parent_ids:
                if branch.get_concept(pid):
                    branch.add_relation(ConceptRelation(
                        source_id=pid,
                        target_id=nc.id,
                        relation_type=RelationType.MOTIVATES,
                        description=f"Generated via {strategy.value}",
                    ))

        # Record evolution
        self._evolution_history.append({
            "source_title": source_theory.title,
            "strategy": strategy.value,
            "target_domain": target_domain.value if target_domain else None,
            "new_concepts": len(new_concepts),
            "branch_title": branch_title,
            "timestamp": time.time(),
        })

        logger.info(f"[Evolver] '{branch_title}': "
                     f"{len(new_concepts)} new concepts from {len(sources)} sources")
        return branch

    async def autonomous_exploration(
        self,
        source_theory: TheoryGraph,
        llm: Any,
        *,
        max_rounds: int = 5,
        strategies: list[ReasoningStrategy] | None = None,
        target_domains: list[ScientificDomain] | None = None,
    ) -> list[TheoryGraph]:
        """Run multiple evolution rounds autonomously with surprise-guided selection.

        Instead of cycling through strategies uniformly, uses Bayesian Surprise
        (AutoDiscovery, Ai2 2026) to guide exploration toward the most novel
        and informative research directions. Strategies that historically produce
        high-surprise branches are selected more often via Thompson Sampling.

        This is the "self-emergent discovery" mode: the system explores the
        theory space by trying different strategies and domains, keeping the
        most promising branches measured by surprise score.
        """
        if strategies is None:
            strategies = [
                ReasoningStrategy.ANALOGY_TRANSFER,
                ReasoningStrategy.GENERALIZATION,
                ReasoningStrategy.COMPOSITION,
                ReasoningStrategy.BOUNDARY_ANALYSIS,
                ReasoningStrategy.UNIFICATION,
                ReasoningStrategy.NUMERICAL_EXPLORATION,
                ReasoningStrategy.DUALITY,
            ]
        if target_domains is None:
            target_domains = [
                ScientificDomain.THEORETICAL_PHYSICS,
                ScientificDomain.STATISTICAL_MECHANICS,
                ScientificDomain.INFORMATION_THEORY,
                ScientificDomain.PURE_MATHEMATICS,
                ScientificDomain.MATHEMATICAL_PHYSICS,
            ]

        branches: list[TheoryGraph] = []
        branch_scores: list[tuple[TheoryGraph, float]] = []  # (branch, surprise)
        current_theory = source_theory

        for round_num in range(max_rounds):
            # Surprise-guided selection (replaces uniform cycling)
            strategy, target = self._select_strategy_by_surprise(strategies, target_domains)

            logger.info(f"[Evolver] Autonomous round {round_num + 1}/{max_rounds}: "
                        f"{strategy.value} → {target.value} (surprise-guided)")

            branch = await self.evolve(
                current_theory,
                strategy=strategy,
                llm=llm,
                target_domain=target,
                num_new_concepts=3,
            )

            # Compute Bayesian surprise for this branch
            surprise = self._bayesian_surprise(branch, current_theory)
            self._record_surprise(strategy, target, surprise)

            if branch.size > len(list(current_theory._nodes.values())):
                branches.append(branch)
                branch_scores.append((branch, surprise))

                # Follow high-surprise branches (not just any branch with conjectures)
                if surprise > 1.5:
                    current_theory = branch
                    logger.info(
                        f"[Evolver] High surprise ({surprise:.2f}), following this branch"
                    )
                elif branch.get_conjectures():
                    current_theory = branch

            logger.info(
                f"[Evolver] Round {round_num + 1}: surprise={surprise:.2f}, "
                f"branch_size={branch.size}"
            )

        # Sort branches by surprise for downstream consumers
        branch_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_branches = [b for b, _ in branch_scores]

        logger.info(
            f"[Evolver] Autonomous exploration complete: {len(branches)} branches, "
            f"top surprise={branch_scores[0][1]:.2f}" if branch_scores else
            f"[Evolver] Autonomous exploration complete: 0 branches"
        )
        return sorted_branches if sorted_branches else branches


# ══════════════════════════════════════════════════════════════
# Article Generator — Produce Structured Research Output
# ══════════════════════════════════════════════════════════════


class ArticleGenerator:
    """Generate structured research articles from a TheoryGraph.

    Given a theory graph (either parsed from an existing paper or generated
    by the TheoryEvolver), produces a structured research article with:
      - Title and abstract
      - Definitions and notation
      - Main results (ordered by dependency)
      - Cross-domain connections
      - Proofs or proof sketches
      - Open problems and conjectures
      - References to source material

    Output formats: LaTeX, Markdown, structured JSON.
    """

    async def generate(
        self,
        theory: TheoryGraph,
        llm: Any,
        *,
        format: str = "latex",      # "latex", "markdown", "json"
        depth: str = "full",         # "abstract", "summary", "full"
        audience: str = "research",  # "research", "survey", "pedagogical"
    ) -> str:
        """Generate a research article from a theory graph."""
        from autoforge.engine.llm_router import TaskComplexity

        # Build ordered concept list (topologically sorted by dependencies)
        ordered = self._topological_order(theory)

        # Group by section
        sections = self._organize_sections(ordered, theory)

        # Generate article
        stats = theory.get_stats()
        prompt = self._build_generation_prompt(
            theory, sections, stats, format, depth, audience,
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system=self._generation_system_prompt(format, audience),
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text
            return text
        except Exception as e:
            logger.warning(f"[ArticleGen] Generation failed: {e}")
            return f"% Article generation failed: {e}"

    def _topological_order(self, theory: TheoryGraph) -> list[ConceptNode]:
        """Topologically sort concepts by dependency."""
        # Kahn's algorithm
        in_degree: dict[str, int] = {nid: 0 for nid in theory._nodes}
        for rel in theory._relations:
            if rel.relation_type == RelationType.DEPENDS_ON:
                if rel.target_id in in_degree:
                    in_degree[rel.target_id] += 1

        queue = [nid for nid, d in in_degree.items() if d == 0]
        ordered: list[str] = []

        while queue:
            nid = queue.pop(0)
            ordered.append(nid)
            for target in theory._forward.get(nid, []):
                if target in in_degree:
                    in_degree[target] -= 1
                    if in_degree[target] == 0:
                        queue.append(target)

        # Add any remaining (cycles or unconnected)
        for nid in theory._nodes:
            if nid not in ordered:
                ordered.append(nid)

        return [theory._nodes[nid] for nid in ordered if nid in theory._nodes]

    def _organize_sections(
        self,
        ordered: list[ConceptNode],
        theory: TheoryGraph,
    ) -> list[dict[str, Any]]:
        """Organize concepts into article sections."""
        sections: list[dict[str, Any]] = []

        # Group by: definitions → lemmas → main theorems → cross-domain → conjectures
        groups = {
            "Definitions and Notation": [ConceptType.DEFINITION, ConceptType.AXIOM],
            "Preliminary Results": [ConceptType.LEMMA],
            "Main Results": [ConceptType.THEOREM, ConceptType.PROPOSITION, ConceptType.COROLLARY],
            "Constructions and Algorithms": [ConceptType.CONSTRUCTION, ConceptType.ALGORITHM],
            "Cross-Domain Connections": [ConceptType.STRUCTURAL_ANALOGY],
            "Observations and Experiments": [ConceptType.OBSERVATION, ConceptType.EXPERIMENT],
            "Conjectures and Open Problems": [ConceptType.CONJECTURE, ConceptType.OPEN_PROBLEM],
            "Guiding Principles": [ConceptType.PRINCIPLE],
        }

        for section_title, types in groups.items():
            concepts = [c for c in ordered if c.concept_type in types]
            if concepts:
                sections.append({
                    "title": section_title,
                    "concepts": concepts,
                })

        return sections

    def _build_generation_prompt(
        self,
        theory: TheoryGraph,
        sections: list[dict[str, Any]],
        stats: dict[str, Any],
        format: str,
        depth: str,
        audience: str,
    ) -> str:
        sections_text = ""
        for sec in sections:
            sections_text += f"\n\n## {sec['title']}\n"
            for c in sec["concepts"][:15]:
                sections_text += (
                    f"\n### [{c.concept_type.value}] {c.formal_statement[:500]}\n"
                    f"Informal: {c.informal_statement[:200]}\n"
                    f"Intuition: {c.intuition[:200]}\n"
                    f"Confidence: {c.overall_confidence:.2f}\n"
                )
                if c.proof_sketch:
                    sections_text += f"Proof sketch: {c.proof_sketch[:300]}\n"

        bridges = theory.get_cross_domain_bridges()
        bridge_text = ""
        for src, tgt, rel in bridges[:10]:
            bridge_text += (
                f"\n- {src.domain.value} ↔ {tgt.domain.value}: "
                f"{rel.bridging_insight[:200]}"
            )

        return f"""Write a {audience} article based on this theory graph.

## Title: {theory.title}

## Statistics
{json.dumps(stats, indent=2)}

## Organized Content
{sections_text}

## Cross-Domain Bridges
{bridge_text}

## Format: {format}
## Depth: {depth}
## Audience: {audience}

Write a complete, well-structured article. Include:
1. Abstract summarizing the main contributions
2. Introduction with motivation and context
3. All definitions, stated precisely
4. All results, with proof sketches where available
5. Cross-domain connections section highlighting structural analogies
6. Discussion of open problems and future directions
7. Conclusion

Use {"LaTeX" if format == "latex" else "Markdown"} formatting."""

    @staticmethod
    def _generation_system_prompt(format: str, audience: str) -> str:
        return (
            f"You are a research article writer producing {audience}-level "
            f"scientific articles in {format} format. You write at the level "
            f"of top-tier journal publications: precise, rigorous, but with "
            f"clear motivation and intuition. You excel at connecting ideas "
            f"across mathematical and scientific domains."
        )


# ══════════════════════════════════════════════════════════════
# Main Facade — TheoreticalReasoningEngine
# ══════════════════════════════════════════════════════════════


class TheoreticalReasoningEngine:
    """Main entry point for theoretical reasoning capabilities.

    Orchestrates:
      - ArticleParser: Extract knowledge from existing papers
      - ReasoningEngine: Generate new insights
      - TheoryEvolver: Branch and extend theories
      - VerificationSuite: Multi-modal verification
      - ArticleGenerator: Produce research articles

    Usage (from orchestrator or directly):
      engine = TheoreticalReasoningEngine()
      graph = await engine.parse_article(text, llm, title="...")
      branches = await engine.evolve_theory(graph, llm)
      article = await engine.generate_article(branches[0], llm)
    """

    def __init__(self) -> None:
        self._parser = ArticleParser()
        self._reasoning = ReasoningEngine()
        self._evolver = TheoryEvolver()
        self._verification = VerificationSuite()
        self._generator = ArticleGenerator()
        self._theories: dict[str, TheoryGraph] = {}

    async def parse_article(
        self,
        text: str,
        llm: Any,
        *,
        title: str = "",
    ) -> TheoryGraph:
        """Parse a research article into a theory graph."""
        graph = await self._parser.parse_article(text, llm, title=title)
        self._theories[graph.title] = graph
        return graph

    async def reason_from(
        self,
        theory: TheoryGraph,
        strategy: ReasoningStrategy,
        llm: Any,
        *,
        target_domain: ScientificDomain | None = None,
    ) -> list[ConceptNode]:
        """Generate new insights from a theory using a specific strategy."""
        sources = theory.get_frontier()[:10]
        if not sources:
            sources = list(theory._nodes.values())[:10]
        return await self._reasoning.reason(
            strategy=strategy,
            source_concepts=sources,
            llm=llm,
            target_domain=target_domain,
        )

    async def evolve_theory(
        self,
        theory: TheoryGraph,
        llm: Any,
        *,
        strategy: ReasoningStrategy | None = None,
        target_domain: ScientificDomain | None = None,
    ) -> TheoryGraph:
        """Generate a new theory branch."""
        if strategy is None:
            strategy = ReasoningStrategy.ANALOGY_TRANSFER
        branch = await self._evolver.evolve(
            theory, strategy=strategy, llm=llm, target_domain=target_domain,
        )
        self._theories[branch.title] = branch
        return branch

    async def autonomous_explore(
        self,
        theory: TheoryGraph,
        llm: Any,
        *,
        max_rounds: int = 5,
    ) -> list[TheoryGraph]:
        """Run autonomous exploration from a theory."""
        branches = await self._evolver.autonomous_exploration(
            theory, llm, max_rounds=max_rounds,
        )
        for branch in branches:
            self._theories[branch.title] = branch
        return branches

    async def verify_theory(
        self,
        theory: TheoryGraph,
        llm: Any,
    ) -> dict[str, float]:
        """Verify all concepts in a theory graph."""
        results: dict[str, float] = {}
        for node in theory._nodes.values():
            if node.overall_confidence < 0.5:
                await self._verification.verify(node, llm)
            results[node.id] = node.overall_confidence
        return results

    async def generate_article(
        self,
        theory: TheoryGraph,
        llm: Any,
        *,
        format: str = "latex",
    ) -> str:
        """Generate a research article from a theory graph."""
        return await self._generator.generate(theory, llm, format=format)

    def save_all(self, base_dir: Path) -> None:
        """Save all theories to disk."""
        base_dir.mkdir(parents=True, exist_ok=True)
        for title, theory in self._theories.items():
            safe_name = re.sub(r'[^\w\-]', '_', title)[:50]
            theory.save(base_dir / safe_name)

    def load_all(self, base_dir: Path) -> None:
        """Load all theories from disk."""
        if not base_dir.exists():
            return
        for subdir in base_dir.iterdir():
            if subdir.is_dir() and (subdir / "theory_graph.json").exists():
                theory = TheoryGraph()
                theory.load(subdir)
                if theory.title:
                    self._theories[theory.title] = theory
        logger.info(f"[TheoreticalReasoning] Loaded {len(self._theories)} theories")

    # ── Autonomous Reasoning Extension ──

    async def run_reasoning_extension(
        self,
        llm: Any,
        *,
        max_rounds: int = 10,
        formalize: bool = False,
        target_conclusions: int = 0,
    ) -> dict[str, Any]:
        """Run the autonomous reasoning extension engine.

        Derives structure from the minimal kernel via growth operators,
        producing numbered, publication-worthy conclusions in academic
        mathematical language. Each round extends the kernel autonomously.

        Args:
            llm: LLM router
            max_rounds: Maximum reasoning rounds
            formalize: Whether to formalize conclusions in Lean 4
            target_conclusions: Stop when this many conclusions reached (0 = no limit)

        Returns:
            Dict with rounds, conclusions, stats, and formatted report.
        """
        from autoforge.engine.reasoning_extension import ReasoningExtensionEngine

        engine = ReasoningExtensionEngine()

        # Load existing state if available
        state_dir = self._get_extension_state_dir()
        if state_dir:
            engine.load(state_dir)

        # Run continuous reasoning
        rounds = await engine.run_continuous(
            llm,
            max_rounds=max_rounds,
            formalize=formalize,
            target_conclusions=target_conclusions,
        )

        # Save state
        if state_dir:
            engine.save(state_dir)

        return {
            "rounds": [r.to_dict() for r in rounds],
            "total_conclusions": engine.conclusion_count,
            "stats": engine.get_stats(),
            "report_markdown": engine.generate_report(latex=False),
            "report_latex": engine.generate_report(latex=True),
        }

    async def verify_article_claims(
        self,
        article_text: str,
        llm: Any,
        *,
        title: str = "Untitled",
        cross_verify: bool = False,
    ) -> dict[str, Any]:
        """Verify an existing article's mathematical claims.

        Parses the article, extracts all mathematical claims, formalizes
        them in Lean 4, verifies with compiler, and optionally cross-verifies
        with multiple provers.

        Args:
            article_text: Article text (LaTeX, Markdown, or plain)
            llm: LLM router
            title: Article title
            cross_verify: Whether to use multi-prover cross-verification

        Returns:
            Dict with verification report.
        """
        from autoforge.engine.article_verifier import ArticleVerifier

        verifier = ArticleVerifier()
        report = await verifier.verify_article(
            article_text, llm,
            title=title,
            cross_verify=cross_verify,
        )

        return {
            "report": report.to_dict(),
            "summary": report.format_summary(),
            "confidence": report.overall_confidence,
            "assessment": report.assessment,
        }

    def _get_extension_state_dir(self) -> Path | None:
        """Get the directory for reasoning extension state."""
        try:
            base = Path(".autoforge") / "reasoning_extension"
            base.mkdir(parents=True, exist_ok=True)
            return base
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════


def _extract_json(text: str) -> dict[str, Any] | None:
    """Robustly extract JSON object from LLM output."""
    if "{" not in text:
        return None
    try:
        json_str = text[text.index("{"):text.rindex("}") + 1]
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ══════════════════════════════════════════════════════════════
# HyperEdge & HyperGraph — N-ary Relationships for TheoryGraph
# ══════════════════════════════════════════════════════════════
# SciAgents-style hyperedges to model n-ary relationships (≥2 nodes)
# in addition to the standard binary ConceptRelations. Enables discovery
# of complex multi-concept interactions and structures.


@dataclass
class HyperEdge:
    """N-ary relationship connecting 2+ nodes in the theory graph.

    Unlike binary ConceptRelation, HyperEdge can connect any number of
    nodes with a single unified relationship. Example: a mathematical
    structure (e.g., "commutative monoid") connects the concepts
    commutativity, associativity, identity, and closure.
    """

    edge_id: str
    node_ids: list[str]  # ≥2 node IDs
    relation_type: str  # E.g., "structural_analogy", "mutual_constraint"
    weight: float = 1.0  # Importance / strength
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str = ""


class HyperGraph:
    """Hypergraph extension for TheoryGraph.

    Maintains n-ary relationships alongside the binary graph structure.
    Useful for discovering complex multi-concept patterns, cliques, and
    structural analogies across the theory.
    """

    def __init__(self, theory_graph: TheoryGraph):
        """Initialize HyperGraph on top of a TheoryGraph.

        Args:
            theory_graph: The underlying TheoryGraph to extend.
        """
        self.theory_graph = theory_graph
        self._hyperedges: dict[str, HyperEdge] = {}
        self._node_to_edges: dict[str, list[str]] = {}  # node_id → edge_ids

    def add_hyperedge(
        self,
        node_ids: list[str],
        relation_type: str,
        weight: float = 1.0,
        description: str = "",
    ) -> str:
        """Add a new hyperedge to the graph.

        Args:
            node_ids: List of ≥2 node IDs to connect.
            relation_type: Type of n-ary relationship.
            weight: Importance of this relationship.
            description: Human-readable description.

        Returns:
            The generated edge_id.

        Raises:
            ValueError: If fewer than 2 nodes specified.
        """
        if len(node_ids) < 2:
            raise ValueError(f"HyperEdge requires ≥2 nodes; got {len(node_ids)}")

        # Verify all nodes exist in the theory graph
        for node_id in node_ids:
            if node_id not in self.theory_graph.nodes:
                logging.warning(f"Node {node_id} not in theory graph; proceeding anyway")

        # Generate edge ID
        edge_id = hashlib.md5(
            f"{relation_type}:{','.join(sorted(node_ids))}".encode()
        ).hexdigest()[:16]

        edge = HyperEdge(
            edge_id=edge_id,
            node_ids=sorted(node_ids),  # Canonical order
            relation_type=relation_type,
            weight=weight,
            description=description,
        )

        self._hyperedges[edge_id] = edge

        # Update reverse index
        for node_id in node_ids:
            if node_id not in self._node_to_edges:
                self._node_to_edges[node_id] = []
            if edge_id not in self._node_to_edges[node_id]:
                self._node_to_edges[node_id].append(edge_id)

        return edge_id

    def get_edges_for_node(self, node_id: str) -> list[HyperEdge]:
        """Get all hyperedges involving a given node.

        Args:
            node_id: The node ID.

        Returns:
            List of HyperEdge objects.
        """
        edge_ids = self._node_to_edges.get(node_id, [])
        return [self._hyperedges[eid] for eid in edge_ids if eid in self._hyperedges]

    def get_shared_edges(self, node_ids: list[str]) -> list[HyperEdge]:
        """Find hyperedges connecting all specified nodes.

        Args:
            node_ids: Nodes that must all be in the edge.

        Returns:
            List of HyperEdges containing all specified nodes.
        """
        if not node_ids:
            return []

        # Start with edges of first node
        candidate_ids = set(self._node_to_edges.get(node_ids[0], []))

        # Intersect with edges of remaining nodes
        for node_id in node_ids[1:]:
            candidate_ids &= set(self._node_to_edges.get(node_id, []))

        return [
            edge
            for edge in [self._hyperedges.get(eid) for eid in candidate_ids]
            if edge is not None
        ]

    def find_cliques(self, min_size: int = 3) -> list[list[str]]:
        """Find groups of nodes densely connected by hyperedges.

        A clique is a subset of nodes where every pair is connected by
        at least one hyperedge (including edges connecting both).

        Args:
            min_size: Minimum clique size.

        Returns:
            List of node_id lists forming cliques.
        """
        cliques = []

        # Simple greedy approach: for each hyperedge, check if it forms
        # or extends a clique
        for edge in self._hyperedges.values():
            if len(edge.node_ids) >= min_size:
                cliques.append(edge.node_ids)

        # Optionally find larger cliques by merging edges
        # (more sophisticated algorithm could use Bron-Kerbosch)
        return cliques

    async def discover_hyperedges(self, llm: Any) -> list[str]:
        """Use LLM to discover n-ary relationships in the graph.

        Samples connected components of the graph and asks the LLM to
        identify structural, conceptual, or logical relationships spanning
        3+ concepts.

        Args:
            llm: LLM instance with async __call__.

        Returns:
            List of newly discovered edge_ids.
        """
        new_edges = []

        # Sample a few nodes and their neighborhoods
        node_ids = list(self.theory_graph.nodes.keys())
        if not node_ids:
            return []

        sample_size = min(5, len(node_ids))
        sampled_nodes = node_ids[:sample_size]

        for node_id in sampled_nodes:
            node = self.theory_graph.nodes[node_id]

            # Get neighbors (nodes connected by binary relations)
            neighbor_ids = set()
            for rel in self.theory_graph.relations:
                if rel.source == node_id:
                    neighbor_ids.add(rel.target)
                elif rel.target == node_id:
                    neighbor_ids.add(rel.source)

            if len(neighbor_ids) < 2:
                continue

            neighbor_ids = list(neighbor_ids)[:4]  # Limit to 4 neighbors

            # Build context
            neighbor_docs = [
                f"- {nid}: {self.theory_graph.nodes[nid].description}"
                for nid in neighbor_ids
            ]

            prompt = f"""Analyze these interconnected concepts and identify 
n-ary relationships (structures, patterns, or logical constraints connecting 
3+ concepts):

Central concept: {node_id}
{node.description}

Connected concepts:
{chr(10).join(neighbor_docs)}

List 1-2 n-ary relationships as JSON:
[{{"relation_type": "...", "node_ids": [...], "description": "..."}}]"""

            try:
                response = await llm(prompt)
                # Extract JSON list
                if "[" in response and "{" in response:
                    json_str = response[
                        response.index("[") : response.rindex("]") + 1
                    ]
                    discovered = json.loads(json_str)
                    for item in discovered:
                        edge_id = self.add_hyperedge(
                            node_ids=item.get("node_ids", [neighbor_ids[:2]]),
                            relation_type=item.get("relation_type", "unknown"),
                            description=item.get("description", ""),
                        )
                        new_edges.append(edge_id)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logging.debug(f"Failed to parse hyperedge discovery: {e}")
                continue

        return new_edges

    def to_ontology_dict(self) -> dict[str, Any]:
        """Export hypergraph as an ontology dictionary.

        Suitable for serialization, knowledge graph conversion, or
        external tools.

        Returns:
            Dict representation with nodes and hyperedges.
        """
        return {
            "nodes": [
                {
                    "id": node_id,
                    "concept_type": node.concept_type.value,
                    "description": node.description,
                }
                for node_id, node in self.theory_graph.nodes.items()
            ],
            "hyperedges": [
                {
                    "id": edge.edge_id,
                    "node_ids": edge.node_ids,
                    "relation_type": edge.relation_type,
                    "weight": edge.weight,
                    "description": edge.description,
                }
                for edge in self._hyperedges.values()
            ],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the hypergraph.

        Returns:
            Dict with counts and metrics.
        """
        edges_by_type: dict[str, int] = {}
        size_distribution: dict[int, int] = {}

        for edge in self._hyperedges.values():
            edges_by_type[edge.relation_type] = (
                edges_by_type.get(edge.relation_type, 0) + 1
            )
            size = len(edge.node_ids)
            size_distribution[size] = size_distribution.get(size, 0) + 1

        return {
            "total_hyperedges": len(self._hyperedges),
            "edges_by_type": edges_by_type,
            "size_distribution": size_distribution,
            "avg_edge_size": (
                sum(len(e.node_ids) for e in self._hyperedges.values())
                / max(len(self._hyperedges), 1)
            ),
            "avg_weight": (
                sum(e.weight for e in self._hyperedges.values())
                / max(len(self._hyperedges), 1)
            ),
            "nodes_with_hyperedges": len(self._node_to_edges),
        }
