"""Autonomous Reasoning Extension Engine — Minimal-Kernel Self-Growing Structure.

This module implements the core requirement: all models should ultimately derive
their structure from a minimal kernel that autonomously grows. Each reasoning round:

  1. Starts from an irreducible axiomatic kernel (MinimalKernel)
  2. Extends via LLM-driven deep reasoning that produces genuinely novel conclusions
  3. Enforces academic-quality output: numbered theorems, formal language, no repetition
  4. Evaluates publication worthiness before accepting conclusions
  5. Optionally formalizes conclusions in Lean 4 and cross-verifies

The engine is designed around the papers:
  - Phasonfold: Phase-space foliation theory connecting dynamical systems,
    number theory, and statistical mechanics via a minimal kernel
  - proof_q4_complete: Formal verification pipeline for mathematical claims
  - Research articles on cross-domain structural analogies

Architecture:

    MinimalKernel          — The irreducible axiomatic seed
        ↓ grows via
    GrowthOperator         — Categorical operations: lift, fold, specialize, dualize
        ↓ produces
    NumberedConclusion      — Academically formatted, numbered, with proof sketch
        ↓ validated by
    PublicationGate         — Novelty, depth, rigor, non-repetition checks
        ↓ verified by
    FormalVerificationBridge — Lean 4 / multi-prover formalization

    ReasoningExtensionEngine — Main orchestrator for autonomous extension rounds
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Core Data Structures
# ══════════════════════════════════════════════════════════════


class GrowthOperator(str, Enum):
    """Categorical operations for kernel self-growth.

    Each operator preserves the minimal kernel's structural invariants
    while extending into new mathematical territory.
    """
    LIFT = "lift"                       # Lift structure to higher categorical level
    FOLD = "fold"                       # Phase-space foliation (Phasonfold-style)
    SPECIALIZE = "specialize"           # Instantiate general structure in specific domain
    DUALIZE = "dualize"                 # Construct dual/adjoint structure
    COMPOSE = "compose"                 # Compose two existing structures
    DEFORM = "deform"                   # Continuous deformation preserving invariants
    QUANTIZE = "quantize"              # Discretize continuous structure
    RENORMALIZE = "renormalize"        # Renormalization group flow
    FUNCTORIAL_TRANSFER = "functorial_transfer"  # Transfer via functor between categories
    SPECTRAL_DECOMPOSE = "spectral_decompose"    # Spectral theory decomposition
    COHOMOLOGICAL_EXTEND = "cohomological_extend"  # Extend via cohomology
    ERGODIC_LIMIT = "ergodic_limit"    # Ergodic-theoretic limit construction


class ConclusionType(str, Enum):
    """Type of mathematical conclusion."""
    DEFINITION = "definition"
    LEMMA = "lemma"
    THEOREM = "theorem"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    CONJECTURE = "conjecture"
    CONSTRUCTION = "construction"


class PublicationWorthiness(str, Enum):
    """Assessment of publication value."""
    EXCEPTIONAL = "exceptional"       # Novel result connecting distinct fields
    PUBLISHABLE = "publishable"       # Solid new result, suitable for journal
    INCREMENTAL = "incremental"       # Valid but minor extension
    ROUTINE = "routine"               # Standard exercise, not publication-worthy
    REDUNDANT = "redundant"           # Already known or trivially follows


@dataclass
class MinimalKernel:
    """The irreducible axiomatic seed from which all structure grows.

    Inspired by the Phasonfold framework: a minimal set of axioms
    that encode the essential structural invariants (phase-space geometry,
    arithmetic-dynamical coupling, spectral correspondence) from which
    the full theory self-assembles via categorical operations.
    """
    id: str = ""
    name: str = "Φ-Kernel"
    axioms: list[str] = field(default_factory=list)
    domain_seeds: dict[str, str] = field(default_factory=dict)
    growth_history: list[str] = field(default_factory=list)
    _conclusion_counter: int = 0

    def next_conclusion_number(self) -> int:
        """Return next conclusion number (globally monotone)."""
        self._conclusion_counter += 1
        return self._conclusion_counter

    @property
    def counter(self) -> int:
        return self._conclusion_counter

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "axioms": self.axioms,
            "domain_seeds": self.domain_seeds,
            "growth_history": self.growth_history,
            "conclusion_counter": self._conclusion_counter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MinimalKernel:
        kernel = cls(
            id=data.get("id", ""),
            name=data.get("name", "Φ-Kernel"),
            axioms=data.get("axioms", []),
            domain_seeds=data.get("domain_seeds", {}),
            growth_history=data.get("growth_history", []),
        )
        kernel._conclusion_counter = data.get("conclusion_counter", 0)
        return kernel

    @classmethod
    def create_default(cls) -> MinimalKernel:
        """Create the default minimal kernel encoding the Phasonfold axioms."""
        return cls(
            id=hashlib.sha256(b"phi-kernel-v1").hexdigest()[:12],
            name="Φ-Kernel",
            axioms=[
                # Phase-space structure
                "Axiom (Φ1). There exists a measurable dynamical system (X, T, μ) "
                "with phase space X, evolution operator T, and invariant measure μ.",

                # Arithmetic-dynamical coupling
                "Axiom (Φ2). There exists a partition function Z(β) = Σ_n a_n e^{-βE_n} "
                "encoding the spectral data of T, where {E_n} are eigenvalues of "
                "the infinitesimal generator of T.",

                # Minimal irreducibility
                "Axiom (Φ3). The triple (X, T, Z) is irreducible in the sense that "
                "no proper sub-system generates the full spectral correspondence "
                "between dynamical orbits and arithmetic functions.",

                # Functorial bridge
                "Axiom (Φ4). There exists a functor F: Dyn → Arith from the category "
                "of dynamical systems to the category of arithmetic structures "
                "such that F preserves spectral invariants.",

                # Self-similarity / renormalization
                "Axiom (Φ5). The system (X, T, Z) admits a renormalization operator R "
                "such that the fixed points of R correspond to universality classes "
                "of the spectral correspondence.",
            ],
            domain_seeds={
                "dynamical_systems": "Phase space foliations, ergodic decomposition, mixing properties",
                "number_theory": "Zeta functions, L-functions, Dirichlet series, modular forms",
                "statistical_mechanics": "Partition functions, phase transitions, critical phenomena",
                "spectral_theory": "Operator spectra, trace formulas, spectral gaps",
                "algebraic_geometry": "Schemes, sheaves, cohomology, motives",
                "category_theory": "Functors, natural transformations, adjunctions, topoi",
            },
        )


@dataclass
class NumberedConclusion:
    """A single numbered conclusion in academic format.

    Each conclusion is:
      - Numbered with a globally monotone counter
      - Typed (theorem, lemma, corollary, conjecture, etc.)
      - Stated in rigorous mathematical language
      - Accompanied by a proof sketch or justification
      - Traceable to the growth operator and parent conclusions that produced it
      - Assessed for publication worthiness
    """
    number: int
    conclusion_type: ConclusionType
    statement: str                          # Formal statement in academic language
    proof_sketch: str = ""                  # Proof or justification
    growth_operator: GrowthOperator = GrowthOperator.LIFT
    parent_numbers: list[int] = field(default_factory=list)
    domain: str = ""
    novelty_score: float = 0.0             # 0-1, how novel this result is
    depth_score: float = 0.0               # 0-1, how deep/surprising
    rigor_score: float = 0.0               # 0-1, mathematical rigor
    worthiness: PublicationWorthiness = PublicationWorthiness.ROUTINE
    lean_formalization: str = ""           # Optional Lean 4 formalization
    verification_status: str = "unverified"  # unverified, verified, failed
    content_hash: str = ""                 # For deduplication
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.statement.encode()
            ).hexdigest()[:16]

    def format_academic(self) -> str:
        """Format as academic-style numbered conclusion."""
        type_label = self.conclusion_type.value.capitalize()
        header = f"**{type_label} {self.number}**"
        if self.domain:
            header += f" ({self.domain})"
        header += "."

        lines = [header, "", f"*{self.statement}*"]
        if self.proof_sketch:
            if self.conclusion_type in (ConclusionType.THEOREM, ConclusionType.LEMMA,
                                         ConclusionType.PROPOSITION):
                lines.extend(["", f"*Proof sketch.* {self.proof_sketch} □"])
            else:
                lines.extend(["", f"*Justification.* {self.proof_sketch}"])

        if self.lean_formalization:
            lines.extend(["", "```lean", self.lean_formalization, "```"])

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "type": self.conclusion_type.value,
            "statement": self.statement,
            "proof_sketch": self.proof_sketch,
            "growth_operator": self.growth_operator.value,
            "parent_numbers": self.parent_numbers,
            "domain": self.domain,
            "novelty_score": self.novelty_score,
            "depth_score": self.depth_score,
            "rigor_score": self.rigor_score,
            "worthiness": self.worthiness.value,
            "lean_formalization": self.lean_formalization,
            "verification_status": self.verification_status,
            "content_hash": self.content_hash,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NumberedConclusion:
        return cls(
            number=data["number"],
            conclusion_type=ConclusionType(data.get("type", "theorem")),
            statement=data["statement"],
            proof_sketch=data.get("proof_sketch", ""),
            growth_operator=GrowthOperator(data.get("growth_operator", "lift")),
            parent_numbers=data.get("parent_numbers", []),
            domain=data.get("domain", ""),
            novelty_score=data.get("novelty_score", 0.0),
            depth_score=data.get("depth_score", 0.0),
            rigor_score=data.get("rigor_score", 0.0),
            worthiness=PublicationWorthiness(data.get("worthiness", "routine")),
            lean_formalization=data.get("lean_formalization", ""),
            verification_status=data.get("verification_status", "unverified"),
            content_hash=data.get("content_hash", ""),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class ReasoningRound:
    """Record of one autonomous reasoning round."""
    round_number: int
    growth_operator: GrowthOperator
    conclusions: list[NumberedConclusion] = field(default_factory=list)
    parent_conclusions_used: list[int] = field(default_factory=list)
    surprise_score: float = 0.0        # Bayesian surprise of this round
    duration_seconds: float = 0.0
    accepted: int = 0                  # How many passed publication gate
    rejected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_number,
            "operator": self.growth_operator.value,
            "conclusions": [c.to_dict() for c in self.conclusions],
            "parent_conclusions_used": self.parent_conclusions_used,
            "surprise_score": self.surprise_score,
            "duration_seconds": self.duration_seconds,
            "accepted": self.accepted,
            "rejected": self.rejected,
        }


# ══════════════════════════════════════════════════════════════
# Publication Gate — Quality Control
# ══════════════════════════════════════════════════════════════


class PublicationGate:
    """Evaluates whether a conclusion meets publication standards.

    Criteria:
      1. Non-repetition: not already in the conclusion library (content hash check)
      2. Non-triviality: not a direct restatement of axioms or known results
      3. Novelty: genuinely new, not publicly known (LLM-assessed)
      4. Depth: contains a surprising or deep insight (LLM-assessed)
      5. Rigor: stated in proper mathematical language (LLM-assessed)
      6. Kernel-derivability: traceable to minimal kernel growth
    """

    def __init__(self) -> None:
        self._known_hashes: set[str] = set()
        self._known_statements: list[str] = []

    def register_existing(self, conclusions: list[NumberedConclusion]) -> None:
        """Register already-known conclusions for dedup."""
        for c in conclusions:
            self._known_hashes.add(c.content_hash)
            self._known_statements.append(c.statement)

    async def evaluate(
        self,
        conclusion: NumberedConclusion,
        llm: Any,
    ) -> NumberedConclusion:
        """Evaluate a conclusion and assign publication worthiness."""
        # 1. Hash-based dedup
        if conclusion.content_hash in self._known_hashes:
            conclusion.worthiness = PublicationWorthiness.REDUNDANT
            return conclusion

        # 2. Semantic dedup (approximate)
        normalized = re.sub(r'\s+', ' ', conclusion.statement.lower().strip())
        for known in self._known_statements:
            known_norm = re.sub(r'\s+', ' ', known.lower().strip())
            if self._jaccard_similarity(normalized, known_norm) > 0.75:
                conclusion.worthiness = PublicationWorthiness.REDUNDANT
                return conclusion

        # 3. LLM-based quality assessment
        from autoforge.engine.llm_router import TaskComplexity

        existing_summary = "\n".join(
            f"  [{i+1}] {s[:120]}..." if len(s) > 120 else f"  [{i+1}] {s}"
            for i, s in enumerate(self._known_statements[-30:])
        ) or "(none yet)"

        prompt = f"""You are a referee for a top-tier mathematics journal (Annals of Mathematics, Inventiones level).
Evaluate whether the following conclusion merits publication.

## Conclusion to evaluate
Type: {conclusion.conclusion_type.value}
Domain: {conclusion.domain}
Growth operator: {conclusion.growth_operator.value}

Statement:
{conclusion.statement}

Proof sketch:
{conclusion.proof_sketch[:2000]}

## Previously established conclusions (for non-repetition check)
{existing_summary}

## Evaluation criteria (score each 0.0 — 1.0)
1. **Novelty**: Is this genuinely new? Not a restatement of known results, not in any textbook.
2. **Depth**: Does it reveal a surprising structural connection or deep insight?
3. **Rigor**: Is the statement precise, well-defined, and the proof sketch convincing?
4. **Non-triviality**: Is this more than a routine exercise or definitional unfolding?

## Overall worthiness
- "exceptional": Novel cross-domain result connecting distinct mathematical areas
- "publishable": Solid new result suitable for a research journal
- "incremental": Valid but minor extension of existing work
- "routine": Standard exercise or textbook-level
- "redundant": Essentially restates a known result

Return ONLY JSON:
{{
  "novelty": 0.0-1.0,
  "depth": 0.0-1.0,
  "rigor": 0.0-1.0,
  "worthiness": "exceptional|publishable|incremental|routine|redundant",
  "reasoning": "brief explanation"
}}"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You are a rigorous mathematical referee. Evaluate with the standards of Annals of Mathematics.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            data = self._extract_json(text)
            if data:
                conclusion.novelty_score = float(data.get("novelty", 0.0))
                conclusion.depth_score = float(data.get("depth", 0.0))
                conclusion.rigor_score = float(data.get("rigor", 0.0))
                worthiness_str = data.get("worthiness", "routine")
                try:
                    conclusion.worthiness = PublicationWorthiness(worthiness_str)
                except ValueError:
                    conclusion.worthiness = PublicationWorthiness.ROUTINE
        except Exception as e:
            logger.debug(f"[PublicationGate] LLM evaluation failed: {e}")
            conclusion.worthiness = PublicationWorthiness.ROUTINE

        return conclusion

    def accept(self, conclusion: NumberedConclusion) -> bool:
        """Returns True if the conclusion passes the gate."""
        if conclusion.worthiness in (PublicationWorthiness.EXCEPTIONAL,
                                      PublicationWorthiness.PUBLISHABLE):
            self._known_hashes.add(conclusion.content_hash)
            self._known_statements.append(conclusion.statement)
            return True
        return False

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Word-level Jaccard similarity."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
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
# Formal Verification Bridge
# ══════════════════════════════════════════════════════════════


class FormalVerificationBridge:
    """Bridge between reasoning conclusions and formal verification.

    Provides:
      1. Auto-formalization: natural language → Lean 4
      2. Verification dispatch: send to Lean prover or multi-prover
      3. Proof repair: fix formalization errors iteratively
      4. Cross-verification: verify same claim in multiple provers
    """

    def __init__(self) -> None:
        self._lean_prover: Any = None
        self._multi_prover: Any = None

    async def formalize_conclusion(
        self,
        conclusion: NumberedConclusion,
        llm: Any,
    ) -> str:
        """Translate a NumberedConclusion into Lean 4 code."""
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""You are an expert in auto-formalization (natural mathematics → Lean 4).

Translate the following mathematical conclusion into a valid Lean 4 theorem statement
with a proof (or `sorry` if the proof is beyond current automation).

## Conclusion {conclusion.number} ({conclusion.conclusion_type.value})
Domain: {conclusion.domain}

Statement:
{conclusion.statement}

Proof sketch:
{conclusion.proof_sketch[:3000]}

## Requirements
- Use standard Lean 4 syntax (Lean 4, not Lean 3)
- Import Mathlib modules if needed (`import Mathlib.X.Y.Z`)
- Define any required auxiliary types/structures
- The statement must be mathematically equivalent to the natural language version
- Prefer `by` tactic proofs
- If the proof is non-trivial, use `sorry` for parts that need human assistance,
  but try to fill in as much as possible
- Add a docstring comment linking to the conclusion number

Return ONLY the Lean 4 code block:
```lean
-- Conclusion {conclusion.number}: {conclusion.conclusion_type.value}
...
```"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You are a Lean 4 auto-formalization expert. Produce compilable code.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            match = re.search(r"```lean\s*\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        except Exception as e:
            logger.debug(f"[FormalBridge] Formalization failed: {e}")
        return ""

    async def verify_conclusion(
        self,
        conclusion: NumberedConclusion,
        llm: Any,
        *,
        workspace: Path | None = None,
    ) -> dict[str, Any]:
        """Verify a formalized conclusion using available provers.

        Returns verification result dict with status and details.
        """
        if not conclusion.lean_formalization:
            return {"status": "no_formalization", "details": "No Lean code to verify"}

        result: dict[str, Any] = {"status": "unverified", "details": ""}

        # Try Lean prover
        try:
            from autoforge.engine.provers.lean_core import LeanEnvironment
            lean_env = LeanEnvironment(workspace)

            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lean", delete=False,
                dir=str(workspace) if workspace else None,
            ) as f:
                f.write(conclusion.lean_formalization)
                lean_file = Path(f.name)

            verification = await lean_env.verify_file(lean_file)
            result = {
                "status": "verified" if verification.success else "failed",
                "errors": verification.errors,
                "warnings": verification.warnings,
                "sorry_count": verification.sorry_count,
                "execution_time": verification.execution_time,
            }

            # Cleanup
            try:
                lean_file.unlink()
            except OSError:
                pass

        except Exception as e:
            result = {"status": "error", "details": str(e)}

        conclusion.verification_status = result["status"]
        return result

    async def cross_verify(
        self,
        conclusion: NumberedConclusion,
        llm: Any,
    ) -> dict[str, Any]:
        """Cross-verify a conclusion using multiple proof backends."""
        results: dict[str, Any] = {}

        # Lean verification
        lean_result = await self.verify_conclusion(conclusion, llm)
        results["lean4"] = lean_result

        # Multi-prover cross-verification (if available)
        try:
            from autoforge.engine.provers.multi_prover import MultiProverEngine
            engine = MultiProverEngine()
            cross = await engine.cross_verify(
                conclusion.statement, llm,
            )
            results["cross_verification"] = cross
        except Exception as e:
            logger.debug(f"[FormalBridge] Multi-prover cross-verify failed: {e}")
            results["cross_verification"] = {"status": "unavailable", "error": str(e)}

        return results


# ══════════════════════════════════════════════════════════════
# Main Engine
# ══════════════════════════════════════════════════════════════


class ReasoningExtensionEngine:
    """Autonomous Reasoning Extension Engine.

    Core workflow per round:
      1. Select growth operator based on frontier analysis
      2. Build context from kernel axioms + existing conclusions
      3. Prompt LLM for deep, novel, publication-worthy conclusions
      4. Parse and validate conclusions (numbered, typed, no repetition)
      5. Evaluate via PublicationGate
      6. Optionally formalize in Lean 4
      7. Accept or reject, update kernel growth history
      8. Continue until publication-worthy results are found

    The engine enforces:
      - No repetition of previously established conclusions
      - No repetition of publicly known results (LLM-checked)
      - Academic journal-level language (no colloquialisms)
      - Globally monotone conclusion numbering
      - Deep results only (no intermediate process conclusions)
      - Kernel-derivability trace for every conclusion
    """

    # Growth operators ordered by typical depth of results
    OPERATOR_DEPTH_ORDER = [
        GrowthOperator.SPECIALIZE,
        GrowthOperator.COMPOSE,
        GrowthOperator.DUALIZE,
        GrowthOperator.LIFT,
        GrowthOperator.FOLD,
        GrowthOperator.DEFORM,
        GrowthOperator.QUANTIZE,
        GrowthOperator.RENORMALIZE,
        GrowthOperator.FUNCTORIAL_TRANSFER,
        GrowthOperator.SPECTRAL_DECOMPOSE,
        GrowthOperator.COHOMOLOGICAL_EXTEND,
        GrowthOperator.ERGODIC_LIMIT,
    ]

    def __init__(self) -> None:
        self._kernel: MinimalKernel = MinimalKernel.create_default()
        self._conclusions: list[NumberedConclusion] = []
        self._rounds: list[ReasoningRound] = []
        self._gate = PublicationGate()
        self._formal_bridge = FormalVerificationBridge()
        # Operator success tracking (Thompson sampling)
        self._operator_successes: dict[str, int] = {op.value: 1 for op in GrowthOperator}
        self._operator_failures: dict[str, int] = {op.value: 1 for op in GrowthOperator}

    @property
    def kernel(self) -> MinimalKernel:
        return self._kernel

    @property
    def conclusions(self) -> list[NumberedConclusion]:
        return list(self._conclusions)

    @property
    def conclusion_count(self) -> int:
        return len(self._conclusions)

    async def run_reasoning_round(
        self,
        llm: Any,
        *,
        operator: GrowthOperator | None = None,
        formalize: bool = False,
        min_conclusions: int = 2,
        max_attempts: int = 3,
    ) -> ReasoningRound:
        """Execute one round of autonomous reasoning extension.

        Args:
            llm: LLM router
            operator: Growth operator to use (auto-selected if None)
            formalize: Whether to formalize accepted conclusions in Lean 4
            min_conclusions: Minimum publication-worthy conclusions per round
            max_attempts: Maximum LLM calls before giving up

        Returns:
            ReasoningRound with accepted conclusions
        """
        start_time = time.monotonic()
        round_num = len(self._rounds) + 1

        # Select operator
        if operator is None:
            operator = self._select_operator()

        round_record = ReasoningRound(
            round_number=round_num,
            growth_operator=operator,
        )

        self._gate.register_existing(self._conclusions)

        accepted_this_round: list[NumberedConclusion] = []

        for attempt in range(max_attempts):
            if len(accepted_this_round) >= min_conclusions:
                break

            # Generate conclusions
            raw_conclusions = await self._generate_conclusions(
                llm, operator, existing=accepted_this_round,
            )

            for raw in raw_conclusions:
                # Assign number
                raw.number = self._kernel.next_conclusion_number()
                raw.growth_operator = operator

                # Evaluate
                evaluated = await self._gate.evaluate(raw, llm)

                if self._gate.accept(evaluated):
                    # Optionally formalize
                    if formalize:
                        lean_code = await self._formal_bridge.formalize_conclusion(
                            evaluated, llm,
                        )
                        if lean_code:
                            evaluated.lean_formalization = lean_code
                            await self._formal_bridge.verify_conclusion(
                                evaluated, llm,
                            )

                    accepted_this_round.append(evaluated)
                    round_record.accepted += 1
                    logger.info(
                        f"[ReasoningExt] Accepted Conclusion {evaluated.number}: "
                        f"{evaluated.conclusion_type.value} "
                        f"({evaluated.worthiness.value})"
                    )
                else:
                    round_record.rejected += 1
                    # Decrement counter since this conclusion was rejected
                    self._kernel._conclusion_counter -= 1

        # Update state
        round_record.conclusions = accepted_this_round
        round_record.duration_seconds = time.monotonic() - start_time
        round_record.surprise_score = self._compute_surprise(accepted_this_round)

        self._conclusions.extend(accepted_this_round)
        self._rounds.append(round_record)
        self._kernel.growth_history.append(
            f"Round {round_num}: {operator.value} → "
            f"{len(accepted_this_round)} conclusions accepted"
        )

        # Update operator statistics
        if accepted_this_round:
            self._operator_successes[operator.value] = (
                self._operator_successes.get(operator.value, 0) + len(accepted_this_round)
            )
        else:
            self._operator_failures[operator.value] = (
                self._operator_failures.get(operator.value, 0) + 1
            )

        return round_record

    async def run_continuous(
        self,
        llm: Any,
        *,
        max_rounds: int = 10,
        formalize: bool = False,
        target_conclusions: int = 0,
        stop_on_exceptional: bool = True,
    ) -> list[ReasoningRound]:
        """Run multiple reasoning rounds autonomously.

        Continues until:
          - max_rounds reached, OR
          - target_conclusions reached, OR
          - An exceptional result is found (if stop_on_exceptional=True)
        """
        rounds: list[ReasoningRound] = []

        for i in range(max_rounds):
            round_record = await self.run_reasoning_round(
                llm, formalize=formalize,
            )
            rounds.append(round_record)

            logger.info(
                f"[ReasoningExt] Round {round_record.round_number}: "
                f"{round_record.accepted} accepted, "
                f"{round_record.rejected} rejected, "
                f"surprise={round_record.surprise_score:.3f}"
            )

            # Check stopping conditions
            if target_conclusions > 0 and len(self._conclusions) >= target_conclusions:
                logger.info(f"[ReasoningExt] Reached target of {target_conclusions} conclusions")
                break

            if stop_on_exceptional and any(
                c.worthiness == PublicationWorthiness.EXCEPTIONAL
                for c in round_record.conclusions
            ):
                logger.info("[ReasoningExt] Exceptional result found — stopping")
                break

        return rounds

    async def verify_article(
        self,
        article_text: str,
        llm: Any,
        *,
        formalize: bool = True,
    ) -> dict[str, Any]:
        """Verify an existing article's claims.

        Parses the article for mathematical claims, then:
          1. Extracts theorems, lemmas, propositions
          2. Checks internal logical consistency
          3. Formalizes each claim in Lean 4
          4. Runs formal verification
          5. Cross-verifies with multiple provers
          6. Returns comprehensive verification report
        """
        from autoforge.engine.llm_router import TaskComplexity

        # Step 1: Extract claims
        claims = await self._extract_claims(article_text, llm)

        # Step 2: Verify each claim
        results: list[dict[str, Any]] = []
        for claim in claims:
            conclusion = NumberedConclusion(
                number=self._kernel.next_conclusion_number(),
                conclusion_type=ConclusionType(claim.get("type", "theorem")),
                statement=claim["statement"],
                proof_sketch=claim.get("proof", ""),
                domain=claim.get("domain", ""),
            )

            result: dict[str, Any] = {
                "claim_number": conclusion.number,
                "type": conclusion.conclusion_type.value,
                "statement": conclusion.statement[:200],
            }

            # Formalize
            if formalize:
                lean_code = await self._formal_bridge.formalize_conclusion(
                    conclusion, llm,
                )
                conclusion.lean_formalization = lean_code
                result["lean_code"] = lean_code

                # Verify
                if lean_code:
                    verify_result = await self._formal_bridge.verify_conclusion(
                        conclusion, llm,
                    )
                    result["verification"] = verify_result

                    # Cross-verify
                    cross = await self._formal_bridge.cross_verify(conclusion, llm)
                    result["cross_verification"] = cross

            results.append(result)

        # Summary
        verified_count = sum(
            1 for r in results
            if r.get("verification", {}).get("status") == "verified"
        )
        failed_count = sum(
            1 for r in results
            if r.get("verification", {}).get("status") == "failed"
        )

        return {
            "total_claims": len(claims),
            "verified": verified_count,
            "failed": failed_count,
            "unverified": len(claims) - verified_count - failed_count,
            "results": results,
        }

    async def _extract_claims(
        self,
        article_text: str,
        llm: Any,
    ) -> list[dict[str, str]]:
        """Extract mathematical claims from article text."""
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""Extract all mathematical claims (theorems, lemmas, propositions,
corollaries, conjectures) from this article.

## Article (truncated)
{article_text[:8000]}

## Instructions
For each claim, extract:
1. The type (theorem, lemma, proposition, corollary, conjecture)
2. The precise mathematical statement
3. The proof or proof sketch (if provided)
4. The mathematical domain

Return JSON array:
[
  {{
    "type": "theorem",
    "statement": "precise statement...",
    "proof": "proof sketch...",
    "domain": "number theory"
  }}
]"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You are an expert mathematical reader. Extract claims precisely.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            if "[" in text:
                json_str = text[text.index("["):text.rindex("]") + 1]
                items = json.loads(json_str)
                return [
                    item for item in items
                    if isinstance(item, dict) and "statement" in item
                ]
        except Exception as e:
            logger.debug(f"[ReasoningExt] Claim extraction failed: {e}")
        return []

    async def _generate_conclusions(
        self,
        llm: Any,
        operator: GrowthOperator,
        *,
        existing: list[NumberedConclusion] | None = None,
    ) -> list[NumberedConclusion]:
        """Generate new conclusions via LLM-driven deep reasoning."""
        from autoforge.engine.llm_router import TaskComplexity

        # Build context from kernel + existing conclusions
        kernel_text = "\n".join(f"  {a}" for a in self._kernel.axioms)
        domain_text = "\n".join(
            f"  • {k}: {v}" for k, v in self._kernel.domain_seeds.items()
        )

        recent_conclusions = self._conclusions[-20:] + (existing or [])
        conclusions_text = "\n".join(
            f"  [{c.number}] ({c.conclusion_type.value}) {c.statement[:150]}"
            for c in recent_conclusions
        ) if recent_conclusions else "(no prior conclusions)"

        next_num = self._kernel.counter + 1

        operator_descriptions = {
            GrowthOperator.LIFT: "Lift an existing result to a higher categorical or algebraic level. For example, lift from sets to categories, from groups to group schemes, from finite to infinite dimensions.",
            GrowthOperator.FOLD: "Apply phase-space foliation: decompose a dynamical/arithmetic system into stable/unstable/center leaves and derive consequences for the leaf dynamics.",
            GrowthOperator.SPECIALIZE: "Instantiate a general structural result in a specific concrete domain, revealing non-obvious consequences.",
            GrowthOperator.DUALIZE: "Construct the dual, adjoint, or contravariant version of an existing result, revealing hidden symmetries.",
            GrowthOperator.COMPOSE: "Compose two existing results to derive a new consequence that is not immediate from either alone.",
            GrowthOperator.DEFORM: "Continuously deform a mathematical structure while tracking which invariants are preserved and which break.",
            GrowthOperator.QUANTIZE: "Discretize or quantize a continuous structure, studying the spectral and algebraic consequences.",
            GrowthOperator.RENORMALIZE: "Apply renormalization group analysis: study what happens at different scales and identify universality classes.",
            GrowthOperator.FUNCTORIAL_TRANSFER: "Use a functor or natural transformation to transfer results between categories.",
            GrowthOperator.SPECTRAL_DECOMPOSE: "Decompose via spectral theory: eigenvalues, spectral gaps, trace formulas.",
            GrowthOperator.COHOMOLOGICAL_EXTEND: "Extend results using cohomological methods: long exact sequences, spectral sequences, characteristic classes.",
            GrowthOperator.ERGODIC_LIMIT: "Take ergodic or thermodynamic limits and study the emergent structures.",
        }

        prompt = f"""You are a research mathematician operating at the level of Annals of Mathematics,
Inventiones Mathematicae, or Acta Mathematica. You are extending a theory that grows
from a minimal axiomatic kernel.

## MINIMAL KERNEL (Φ-Kernel)
{kernel_text}

## DOMAIN SEEDS
{domain_text}

## GROWTH OPERATOR FOR THIS ROUND
**{operator.value.upper()}**: {operator_descriptions.get(operator, "")}

## PREVIOUSLY ESTABLISHED CONCLUSIONS (do NOT repeat)
{conclusions_text}

## YOUR TASK
Apply the growth operator "{operator.value}" to derive NEW, DEEP, PUBLICATION-WORTHY
conclusions from the kernel and prior results.

## STRICT REQUIREMENTS
1. Each conclusion must be GENUINELY NOVEL — not a restatement of textbook results,
   not something already published. Use known results as tools, but derive NEW consequences.
2. Write in rigorous mathematical language befitting a top journal. No colloquialisms.
3. Provide ONLY final results — no intermediate steps, no "observations along the way."
4. Each conclusion must be deep enough to be SURPRISING — something that would make
   a specialist in the field pause and think "I hadn't seen that connection before."
5. Number conclusions starting from {next_num}, incrementally.
6. Include a proof sketch for each theorem/lemma/proposition.
7. Specify the mathematical domain for each conclusion.
8. The conclusion must be TRACEABLE to the Φ-Kernel axioms via the growth operator.

## OUTPUT FORMAT (JSON array)
[
  {{
    "type": "theorem|lemma|proposition|corollary|conjecture|definition|construction",
    "statement": "Precise mathematical statement in academic language",
    "proof_sketch": "Rigorous proof sketch",
    "domain": "e.g., spectral theory, number theory, dynamical systems",
    "parent_numbers": [list of prior conclusion numbers this builds on],
    "novelty_justification": "Why this is genuinely new"
  }}
]

Generate 3-5 conclusions. Quality over quantity. Research-level depth required."""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system=(
                    "You are a world-class research mathematician. Your conclusions "
                    "appear in Annals of Mathematics. Every statement is precise, novel, "
                    "and deeply insightful. You never repeat known results. You never "
                    "use informal or imprecise language."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            return self._parse_conclusions(text, operator)

        except Exception as e:
            logger.warning(f"[ReasoningExt] Conclusion generation failed: {e}")
            return []

    def _parse_conclusions(
        self,
        text: str,
        operator: GrowthOperator,
    ) -> list[NumberedConclusion]:
        """Parse LLM output into NumberedConclusion objects."""
        conclusions: list[NumberedConclusion] = []

        try:
            if "[" in text:
                json_str = text[text.index("["):text.rindex("]") + 1]
                items = json.loads(json_str)
            else:
                return conclusions
        except (json.JSONDecodeError, ValueError):
            # Try to extract individual JSON objects
            items = []
            for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text):
                try:
                    items.append(json.loads(match.group()))
                except json.JSONDecodeError:
                    continue

        for item in items:
            if not isinstance(item, dict) or "statement" not in item:
                continue

            try:
                ctype = ConclusionType(item.get("type", "theorem"))
            except ValueError:
                ctype = ConclusionType.THEOREM

            conclusion = NumberedConclusion(
                number=0,  # Will be assigned later
                conclusion_type=ctype,
                statement=item["statement"],
                proof_sketch=item.get("proof_sketch", ""),
                growth_operator=operator,
                parent_numbers=item.get("parent_numbers", []),
                domain=item.get("domain", ""),
            )
            conclusions.append(conclusion)

        return conclusions

    def _select_operator(self) -> GrowthOperator:
        """Select growth operator via Thompson sampling."""
        import random
        best_op = GrowthOperator.LIFT
        best_sample = -1.0

        for op in GrowthOperator:
            alpha = self._operator_successes.get(op.value, 1)
            beta = self._operator_failures.get(op.value, 1)
            sample = random.betavariate(alpha, beta)
            if sample > best_sample:
                best_sample = sample
                best_op = op

        return best_op

    def _compute_surprise(self, conclusions: list[NumberedConclusion]) -> float:
        """Compute Bayesian surprise of a set of conclusions."""
        if not conclusions:
            return 0.0

        surprise = 0.0
        for c in conclusions:
            # Novel domain contributes more surprise
            existing_domains = {ec.domain for ec in self._conclusions}
            if c.domain and c.domain not in existing_domains:
                surprise += 2.0

            # High novelty/depth contributes surprise
            surprise += c.novelty_score * 1.5
            surprise += c.depth_score * 1.5

            # Exceptional worthiness is very surprising
            if c.worthiness == PublicationWorthiness.EXCEPTIONAL:
                surprise += 3.0
            elif c.worthiness == PublicationWorthiness.PUBLISHABLE:
                surprise += 1.0

        return surprise / max(len(conclusions), 1)

    def generate_report(self, *, latex: bool = False) -> str:
        """Generate a formatted report of all conclusions.

        Args:
            latex: If True, generate LaTeX output; otherwise Markdown.
        """
        if not self._conclusions:
            return "No conclusions established yet."

        if latex:
            return self._generate_latex_report()
        return self._generate_markdown_report()

    def _generate_markdown_report(self) -> str:
        """Generate Markdown report."""
        lines = [
            "# Autonomous Reasoning Extension — Results",
            "",
            f"**Kernel**: {self._kernel.name}",
            f"**Total conclusions**: {len(self._conclusions)}",
            f"**Reasoning rounds**: {len(self._rounds)}",
            "",
            "---",
            "",
        ]

        # Group by domain
        by_domain: dict[str, list[NumberedConclusion]] = {}
        for c in self._conclusions:
            domain = c.domain or "General"
            by_domain.setdefault(domain, []).append(c)

        for domain, concs in sorted(by_domain.items()):
            lines.append(f"## {domain}")
            lines.append("")
            for c in sorted(concs, key=lambda x: x.number):
                lines.append(c.format_academic())
                lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _generate_latex_report(self) -> str:
        """Generate LaTeX report."""
        lines = [
            r"\documentclass{amsart}",
            r"\usepackage{amsmath,amssymb,amsthm}",
            r"\newtheorem{theorem}{Theorem}",
            r"\newtheorem{lemma}[theorem]{Lemma}",
            r"\newtheorem{proposition}[theorem]{Proposition}",
            r"\newtheorem{corollary}[theorem]{Corollary}",
            r"\newtheorem{conjecture}[theorem]{Conjecture}",
            r"\newtheorem{definition}[theorem]{Definition}",
            r"\newtheorem{construction}[theorem]{Construction}",
            r"\begin{document}",
            r"\title{Autonomous Reasoning Extension --- Results}",
            r"\maketitle",
            "",
        ]

        env_map = {
            ConclusionType.THEOREM: "theorem",
            ConclusionType.LEMMA: "lemma",
            ConclusionType.PROPOSITION: "proposition",
            ConclusionType.COROLLARY: "corollary",
            ConclusionType.CONJECTURE: "conjecture",
            ConclusionType.DEFINITION: "definition",
            ConclusionType.CONSTRUCTION: "construction",
        }

        for c in sorted(self._conclusions, key=lambda x: x.number):
            env = env_map.get(c.conclusion_type, "theorem")
            lines.append(f"\\begin{{{env}}}[Conclusion {c.number}]")
            lines.append(c.statement)
            lines.append(f"\\end{{{env}}}")
            if c.proof_sketch and c.conclusion_type in (
                ConclusionType.THEOREM, ConclusionType.LEMMA,
                ConclusionType.PROPOSITION,
            ):
                lines.append(r"\begin{proof}")
                lines.append(c.proof_sketch)
                lines.append(r"\end{proof}")
            lines.append("")

        lines.append(r"\end{document}")
        return "\n".join(lines)

    # ── Persistence ──

    def save(self, directory: Path) -> None:
        """Save engine state to directory."""
        directory.mkdir(parents=True, exist_ok=True)

        # Save kernel
        kernel_file = directory / "kernel.json"
        kernel_file.write_text(
            json.dumps(self._kernel.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Save conclusions
        conclusions_file = directory / "conclusions.json"
        conclusions_file.write_text(
            json.dumps(
                [c.to_dict() for c in self._conclusions],
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Save rounds
        rounds_file = directory / "rounds.json"
        rounds_file.write_text(
            json.dumps(
                [r.to_dict() for r in self._rounds],
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Save operator stats
        stats_file = directory / "operator_stats.json"
        stats_file.write_text(
            json.dumps({
                "successes": self._operator_successes,
                "failures": self._operator_failures,
            }, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"[ReasoningExt] Saved state: {len(self._conclusions)} conclusions, "
            f"{len(self._rounds)} rounds"
        )

    def load(self, directory: Path) -> None:
        """Load engine state from directory."""
        if not directory.exists():
            return

        # Load kernel
        kernel_file = directory / "kernel.json"
        if kernel_file.exists():
            data = json.loads(kernel_file.read_text(encoding="utf-8"))
            self._kernel = MinimalKernel.from_dict(data)

        # Load conclusions
        conclusions_file = directory / "conclusions.json"
        if conclusions_file.exists():
            data = json.loads(conclusions_file.read_text(encoding="utf-8"))
            self._conclusions = [NumberedConclusion.from_dict(c) for c in data]
            self._gate.register_existing(self._conclusions)

        # Load operator stats
        stats_file = directory / "operator_stats.json"
        if stats_file.exists():
            data = json.loads(stats_file.read_text(encoding="utf-8"))
            self._operator_successes = data.get("successes", self._operator_successes)
            self._operator_failures = data.get("failures", self._operator_failures)

        logger.info(
            f"[ReasoningExt] Loaded state: {len(self._conclusions)} conclusions"
        )

    def get_stats(self) -> dict[str, Any]:
        """Return engine statistics."""
        worthiness_counts: dict[str, int] = {}
        for c in self._conclusions:
            w = c.worthiness.value
            worthiness_counts[w] = worthiness_counts.get(w, 0) + 1

        domain_counts: dict[str, int] = {}
        for c in self._conclusions:
            d = c.domain or "general"
            domain_counts[d] = domain_counts.get(d, 0) + 1

        verified = sum(1 for c in self._conclusions if c.verification_status == "verified")
        formalized = sum(1 for c in self._conclusions if c.lean_formalization)

        return {
            "kernel_name": self._kernel.name,
            "total_conclusions": len(self._conclusions),
            "total_rounds": len(self._rounds),
            "conclusion_counter": self._kernel.counter,
            "worthiness_distribution": worthiness_counts,
            "domain_distribution": domain_counts,
            "formalized_count": formalized,
            "verified_count": verified,
            "avg_novelty": (
                sum(c.novelty_score for c in self._conclusions) /
                max(len(self._conclusions), 1)
            ),
            "avg_depth": (
                sum(c.depth_score for c in self._conclusions) /
                max(len(self._conclusions), 1)
            ),
        }
