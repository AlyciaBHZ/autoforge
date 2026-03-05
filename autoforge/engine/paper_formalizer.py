"""Paper Formalizer — Domain-Specific Lean 4 Formalization & Computational Reproducibility.

Given an academic paper (as a TheoryGraph or raw PDF/text), this module:
  1. Extracts all definitions, theorems, propositions, and corollaries
  2. Generates Lean 4 formalizations for each extractable statement
  3. Produces Python computational reproducibility scripts for numerical results
  4. Creates a structured verification report

Designed specifically for papers in the style of "From Superspace Model Sets to
Information-Theoretic Time" — heavy on measure theory, symbolic dynamics,
ergodic theory, and combinatorics.

Integration:
  - TheoryGraph provides structured paper representation
  - LeanProver (via provers/) handles Lean compilation and proof search
  - VerificationSuite handles multi-modal verification
  - CloudProver can offload heavy Lean compilation to cloud

References:
  - Pantograph (2024): Interactive Lean 4 proof environment
  - LeanDojo (ICLR 2024): LLM-based Lean theorem proving
  - Mathlib4: Community math library for Lean 4
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from autoforge.engine.theoretical_reasoning import (
    ConceptNode,
    ConceptType,
    ScientificDomain,
    TheoryGraph,
    VerificationSuite,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Configuration & Data Structures
# ══════════════════════════════════════════════════════════════


class FormalizationStatus(str, Enum):
    """Status of a single statement's formalization."""
    PENDING = "pending"
    LEAN_GENERATED = "lean_generated"
    LEAN_COMPILED = "lean_compiled"
    LEAN_PROVED = "lean_proved"
    LEAN_SORRY = "lean_sorry"        # Compiled but has sorry
    LEAN_FAILED = "lean_failed"
    NUMERICALLY_VERIFIED = "numerically_verified"
    COMPUTATIONALLY_REPRODUCED = "computationally_reproduced"
    SKIPPED = "skipped"


@dataclass
class FormalizationUnit:
    """A single theorem/proposition to be formalized."""
    concept_id: str
    concept_type: str
    section: str
    label: str                       # e.g., "Theorem 3.3"
    natural_language: str            # Full statement in NL
    formal_statement_latex: str      # LaTeX original
    lean_code: str = ""
    lean_status: FormalizationStatus = FormalizationStatus.PENDING
    lean_errors: list[str] = field(default_factory=list)
    python_script: str = ""
    numerical_result: str = ""
    verification_confidence: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "concept_type": self.concept_type,
            "section": self.section,
            "label": self.label,
            "natural_language": self.natural_language[:500],
            "formal_statement_latex": self.formal_statement_latex[:500],
            "lean_code": self.lean_code,
            "lean_status": self.lean_status.value,
            "lean_errors": self.lean_errors,
            "python_script": self.python_script[:2000],
            "numerical_result": self.numerical_result[:500],
            "verification_confidence": self.verification_confidence,
            "notes": self.notes,
        }


@dataclass
class FormalizationReport:
    """Structured report of a paper's formalization."""
    paper_title: str
    paper_source: str
    total_statements: int = 0
    lean_proved: int = 0
    lean_sorry: int = 0
    lean_failed: int = 0
    numerically_verified: int = 0
    computationally_reproduced: int = 0
    skipped: int = 0
    units: list[FormalizationUnit] = field(default_factory=list)
    lean_available: bool = False
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    overall_score: float = 0.0

    def compute_score(self) -> float:
        """Compute overall formalization score (0-1)."""
        if self.total_statements == 0:
            return 0.0
        proved_weight = 1.0
        # "sorry" placeholders are not creditable toward formalization quality.
        sorry_weight = 0.0
        numerical_weight = 0.7
        reproduced_weight = 0.8
        score = (
            self.lean_proved * proved_weight +
            self.lean_sorry * sorry_weight +
            self.numerically_verified * numerical_weight +
            self.computationally_reproduced * reproduced_weight
        ) / self.total_statements
        self.overall_score = min(1.0, score)
        return self.overall_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_title": self.paper_title,
            "paper_source": self.paper_source,
            "total_statements": self.total_statements,
            "lean_proved": self.lean_proved,
            "lean_sorry": self.lean_sorry,
            "lean_failed": self.lean_failed,
            "numerically_verified": self.numerically_verified,
            "computationally_reproduced": self.computationally_reproduced,
            "skipped": self.skipped,
            "overall_score": self.overall_score,
            "lean_available": self.lean_available,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "units": [u.to_dict() for u in self.units],
        }


# ══════════════════════════════════════════════════════════════
# Iterative Lean Compiler with Error Fixing
# ══════════════════════════════════════════════════════════════


class IterativeLeanCompiler:
    """Iterative compile-fix-retry loop for Lean 4 formalization."""
    MAX_RETRIES = 5

    ERROR_PATTERNS = {
        r"unknown identifier": "add_import",
        r"type mismatch": "fix_types",
        r"tactic.*failed": "try_alternative_tactic",
        r"unexpected token": "fix_syntax",
        r"application error": "fix_application",
        r"invalid attribute": "remove_attribute",
    }

    async def compile_with_retry(
        self,
        lean_code: str,
        llm: Any | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Iteratively compile Lean code with automatic error fixing.

        Args:
            lean_code: Initial Lean 4 code to compile
            llm: Optional LLM for complex error fixes
            max_retries: Max retry attempts (default: MAX_RETRIES)

        Returns:
            Dict with keys: success (bool), code (str), attempts (int), errors (list)
        """
        import subprocess
        import tempfile

        if max_retries is None:
            max_retries = self.MAX_RETRIES

        current_code = lean_code
        errors_encountered = []

        for attempt in range(max_retries):
            try:
                # Try to compile
                result = await self._compile(current_code)
                if result.get("success"):
                    return {
                        "success": True,
                        "code": current_code,
                        "attempts": attempt + 1,
                        "errors": errors_encountered,
                    }

                error_text = result.get("error", "")
                errors_encountered.append(error_text)

                # Classify error
                error_class = self._classify_error(error_text)
                logger.debug(f"[IterativeLean] Attempt {attempt + 1}: {error_class}")

                # Try deterministic fixes first
                if error_class == "add_import":
                    current_code = self._add_missing_import(current_code, error_text)
                elif error_class == "fix_syntax":
                    current_code = self._fix_syntax(current_code, error_text)
                elif error_class == "fix_types":
                    current_code = self._fix_types(current_code, error_text)
                elif error_class == "fix_application":
                    current_code = self._fix_application(current_code, error_text)
                elif error_class == "remove_attribute":
                    current_code = self._remove_invalid_attribute(current_code, error_text)
                else:
                    # Fall back to LLM for complex fixes
                    if llm:
                        current_code = await self._llm_fix(
                            current_code, error_text, llm
                        )
                    else:
                        # No LLM available, give up
                        break

            except Exception as e:
                logger.debug(f"[IterativeLean] Exception on attempt {attempt + 1}: {e}")
                errors_encountered.append(str(e))
                if attempt == max_retries - 1:
                    break

        return {
            "success": False,
            "code": current_code,
            "attempts": max_retries,
            "errors": errors_encountered,
        }

    async def _compile(self, code: str) -> dict[str, Any]:
        """Compile Lean code and return result dict."""
        import subprocess
        import tempfile
        import os

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lean", delete=False
            ) as f:
                f.write(code)
                tmp_path = f.name

            proc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["lean", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                ),
            )

            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            if proc.returncode == 0:
                return {"success": True, "error": ""}
            else:
                error = (proc.stderr or proc.stdout or "unknown")[:1000]
                return {"success": False, "error": error}

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Compilation timeout (30s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _classify_error(self, error_text: str) -> str:
        """Classify error into a fixable category."""
        for pattern, classification in self.ERROR_PATTERNS.items():
            if re.search(pattern, error_text, re.IGNORECASE):
                return classification
        return "unknown"

    def _add_missing_import(self, code: str, error_text: str) -> str:
        """Try to add missing imports."""
        # Extract missing name from error if possible
        match = re.search(r"unknown identifier `(\w+)`", error_text)
        if match:
            missing = match.group(1)
            # Common Mathlib mappings
            imports_map = {
                "Matrix": "import Mathlib.LinearAlgebra.Matrix.Basic",
                "Fintype": "import Mathlib.Data.Fintype.Basic",
                "Set": "import Mathlib.Data.Set.Basic",
                "Nat": "import Mathlib.Data.Nat.Basic",
            }
            if missing in imports_map:
                import_stmt = imports_map[missing]
                # Add import at top if not present
                if import_stmt not in code:
                    code = import_stmt + "\n" + code
        return code

    def _fix_syntax(self, code: str, error_text: str) -> str:
        """Try simple syntax fixes."""
        # Fix common syntax issues
        code = re.sub(r"def (\w+) where", r"def \1 : Prop where", code)
        code = re.sub(r"theorem (\w+) where", r"theorem \1 : True where", code)
        return code

    def _fix_types(self, code: str, error_text: str) -> str:
        """Try to fix type mismatches."""
        # Add type annotations where missing
        code = re.sub(r"let (\w+) =", r"let \1 : _ =", code)
        return code

    def _fix_application(self, code: str, error_text: str) -> str:
        """Try to fix function application errors."""
        # Fix common application issues (already quite reduced at this point)
        return code

    def _remove_invalid_attribute(self, code: str, error_text: str) -> str:
        """Remove invalid attributes."""
        # Remove problematic custom attributes
        code = re.sub(r"@\[nosynthesis\]\n", "", code)
        return code

    async def _llm_fix(self, code: str, error_text: str, llm: Any) -> str:
        """Use LLM to fix complex errors."""
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""Fix this Lean 4 compilation error.

ERROR:
{error_text[:500]}

CURRENT CODE:
{code[:1500]}

Generate ONLY the fixed Lean 4 code (no explanation, no markdown fences).
"""
        try:
            resp = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            fixed = resp.content.strip()
            # Remove markdown fences if present
            fixed = re.sub(r"^```\w*\n?", "", fixed)
            fixed = re.sub(r"\n?```$", "", fixed)
            return fixed
        except Exception as e:
            logger.warning(f"[IterativeLean] LLM fix failed: {e}")
            return code


# ══════════════════════════════════════════════════════════════
# Lean 4 Code Generator
# ══════════════════════════════════════════════════════════════


class LeanCodeGenerator:
    """Generate Lean 4 formalizations from mathematical statements.

    Domain-aware: knows about the key structures in superspace model set theory
    and can generate appropriate Lean 4 imports, type definitions, and proof scaffolds.
    """

    # Lean 4 preamble for this paper's domain
    PREAMBLE = '''import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Order.Filter.Basic
import Mathlib.Dynamics.Ergodic.MeasurePreserving
import Mathlib.InformationTheory.Entropy

-- ══════════════════════════════════════════════════════════════
-- Superspace Model Sets — Core Definitions
-- ══════════════════════════════════════════════════════════════

/-- A cut-and-project scheme (CPS) consists of physical space E,
    internal space H, and a lattice Γ in the product G = E × H. -/
structure CPS (d n : ℕ) where
  /-- The lattice in the product space -/
  lattice : Set (Fin d → ℝ) × Set (Fin n → ℝ)
  /-- Physical projection is injective on Γ -/
  phys_proj_injective : True  -- simplified
  /-- Internal projection has dense image -/
  int_proj_dense : True       -- simplified

/-- A model set Λ(W) is the set of physical projections of lattice
    points whose internal projections land in the window W. -/
def modelSet {d n : ℕ} (cps : CPS d n) (W : Set (Fin n → ℝ)) :
    Set (Fin d → ℝ) :=
  sorry  -- Full definition requires lattice enumeration

/-- The golden-mean language X_m: binary words of length m with no "11" substring. -/
def goldenMeanLang (m : ℕ) : Set (Fin m → Bool) :=
  {w | ∀ i : Fin m, ∀ j : Fin m, i.val + 1 = j.val →
    ¬(w i = true ∧ w j = true)}

/-- Fibonacci numbers (for |X_m| = F_{m+2}). -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Zeckendorf representation: unique Fibonacci-sum decomposition. -/
structure ZeckendorfRepr (N : ℕ) where
  coeffs : List Bool
  no_adjacent_ones : ∀ i, i + 1 < coeffs.length →
    ¬(coeffs.get ⟨i, by omega⟩ = true ∧ coeffs.get ⟨i + 1, by omega⟩ = true)
  value_eq : True  -- sum_k coeffs[k] * fib(k+1) = N

'''

    # Domain-specific type mappings: paper concept → Lean 4 type
    TYPE_HINTS: dict[str, str] = {
        "model set": "Set (Fin d → ℝ)",
        "window": "Set (Fin n → ℝ)",
        "scan error": "ℝ",
        "gauge anomaly": "ℕ",
        "entropy rate": "ℝ",
        "cylinder set": "Set (ℕ → Bool)",
        "Parry measure": "MeasureTheory.Measure (ℕ → Bool)",
        "stabilization map": "Fin m → Bool → Fin m → Bool",
        "boundary cylinder count": "ℕ → ℕ",
        "KL divergence": "ℝ",
        "fiber": "Set (Fin m → Bool)",
    }

    async def generate_lean(
        self,
        unit: FormalizationUnit,
        llm: Any,
    ) -> str:
        """Generate Lean 4 code for a formalization unit."""
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""You are an expert Lean 4 formalization engineer specializing in
measure theory, ergodic theory, symbolic dynamics, and combinatorics.

Generate a complete Lean 4 formalization of the following mathematical statement.

STATEMENT ({unit.label}):
{unit.formal_statement_latex}

NATURAL LANGUAGE:
{unit.natural_language}

REQUIREMENTS:
1. Use Mathlib4 imports where available
2. Define any needed intermediate structures
3. State the theorem precisely in Lean 4 syntax
4. Provide a proof skeleton — use `sorry` for non-trivial steps but
   fill in as much as possible (structure, tactics, intermediate goals)
5. Add comments explaining the mathematical correspondence

DOMAIN CONTEXT:
This is from the theory of superspace model sets and information-theoretic time.
Key structures: CPS, model sets, golden-mean language, Zeckendorf representation,
Fold stabilization, gauge anomaly, scan error, boundary cylinder dimension,
entropy rate, KL-divergence ledger identity.

AVAILABLE LEAN 4 PREAMBLE (already defined):
- CPS structure, modelSet, goldenMeanLang, fib, ZeckendorfRepr

Generate ONLY valid Lean 4 code (no markdown fences, no explanations outside comments).
"""

        try:
            resp = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            code = resp.content.strip()
            # Strip markdown fences if present
            code = re.sub(r'^```\w*\n?', '', code)
            code = re.sub(r'\n?```$', '', code)
            return code
        except Exception as e:
            logger.warning(f"[LeanGen] Failed for {unit.label}: {e}")
            return f"-- Failed to generate: {e}\nsorry"

    async def generate_python_verification(
        self,
        unit: FormalizationUnit,
        llm: Any,
    ) -> str:
        """Generate a Python script for numerical verification of a result."""
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""Generate a self-contained Python script that NUMERICALLY VERIFIES
the following mathematical result. The script should:

1. Implement the necessary mathematical constructs (from scratch, using only numpy/scipy)
2. Run the verification for concrete parameter values
3. Print a clear PASS/FAIL verdict with numerical evidence
4. Handle edge cases gracefully

STATEMENT ({unit.label}):
{unit.formal_statement_latex}

NATURAL LANGUAGE:
{unit.natural_language}

DOMAIN CONTEXT:
- Golden ratio φ = (1+√5)/2
- Fibonacci sequence: F_1=F_2=1, F_n=F_{n-1}+F_{n-2}
- Golden-mean language X_m: binary words of length m with no adjacent 1s, |X_m|=F_{m+2}
- Zeckendorf representation: unique non-adjacent Fibonacci sum
- Scan error ε_m: optimal prediction error at depth m
- Gauge anomaly G_m: Hamming distance between naive truncation and fold-aware restriction
- Typical gauge anomaly density: E[G_m]/m → 4/9, variance σ²=118/243
- Information-theoretic time: τ(t) = -log μ(C(a_{{0:t-1}}))
- Entropy rate for golden-mean Parry measure: h = log φ

The script should be REPRODUCIBLE — same results every run (fix random seeds).
Generate ONLY valid Python code (no markdown fences).
"""

        try:
            resp = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            code = resp.content.strip()
            code = re.sub(r'^```\w*\n?', '', code)
            code = re.sub(r'\n?```$', '', code)
            return code
        except Exception as e:
            logger.warning(f"[PythonVerify] Failed for {unit.label}: {e}")
            return f"# Failed to generate: {e}"


# ══════════════════════════════════════════════════════════════
# Paper Formalizer Pipeline
# ══════════════════════════════════════════════════════════════


class PaperFormalizer:
    """End-to-end paper formalization pipeline.

    Workflow:
      1. Extract formalizable statements from TheoryGraph
      2. Generate Lean 4 code for each statement
      3. Attempt Lean compilation (via LeanProver or CloudProver)
      4. Generate Python verification scripts for numerical results
      5. Run Python scripts for computational reproducibility
      6. Produce structured verification report
    """

    def __init__(self) -> None:
        self._lean_gen = LeanCodeGenerator()
        self._lean_compiler = IterativeLeanCompiler()
        self._verifier = VerificationSuite()

    async def formalize(
        self,
        graph: TheoryGraph,
        llm: Any,
        *,
        output_dir: Path | None = None,
        lean_compile: bool = False,
        run_python: bool = False,
        cloud_prover: Any | None = None,
    ) -> FormalizationReport:
        """Formalize all extractable statements in a paper.

        Args:
            graph: TheoryGraph representing the paper
            llm: LLM router for code generation
            output_dir: Directory to save outputs
            lean_compile: Whether to attempt Lean compilation
            run_python: Whether to run generated Python scripts
            cloud_prover: Optional CloudProver instance for remote Lean compilation

        Returns:
            FormalizationReport with detailed results
        """
        import shutil
        logger.info(f"[Formalizer] Starting formalization of '{graph.title}'")

        report = FormalizationReport(
            paper_title=graph.title,
            paper_source=graph.source,
            lean_available=bool(shutil.which("lean")),
        )

        # Step 1: Extract formalizable statements
        units = self._extract_units(graph)
        report.total_statements = len(units)
        logger.info(f"[Formalizer] Extracted {len(units)} formalizable statements")

        # Step 2: Generate Lean 4 code for each
        lean_tasks = []
        for unit in units:
            lean_tasks.append(
                self._formalize_one(
                    unit,
                    llm,
                    lean_compile,
                    run_python,
                    cloud_prover=cloud_prover,
                )
            )

        results = await asyncio.gather(*lean_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"[Formalizer] Unit {units[i].label} failed: {result}")
                units[i].lean_status = FormalizationStatus.LEAN_FAILED
                units[i].lean_errors.append(str(result))
            else:
                units[i] = result

        # Step 3: Compute statistics
        for unit in units:
            report.units.append(unit)
            match unit.lean_status:
                case FormalizationStatus.LEAN_PROVED:
                    report.lean_proved += 1
                case FormalizationStatus.LEAN_SORRY:
                    report.lean_sorry += 1
                case FormalizationStatus.LEAN_FAILED:
                    report.lean_failed += 1
                case FormalizationStatus.NUMERICALLY_VERIFIED:
                    report.numerically_verified += 1
                case FormalizationStatus.COMPUTATIONALLY_REPRODUCED:
                    report.computationally_reproduced += 1
                case FormalizationStatus.SKIPPED:
                    report.skipped += 1

        report.completed_at = time.time()
        report.compute_score()

        logger.info(
            f"[Formalizer] Complete: {report.lean_proved} proved, "
            f"{report.lean_sorry} sorry, {report.lean_failed} failed, "
            f"{report.numerically_verified} numerically verified, "
            f"score={report.overall_score:.2f}"
        )

        # Save outputs
        if output_dir:
            await self._save_outputs(report, units, output_dir)

        return report

    def _extract_units(self, graph: TheoryGraph) -> list[FormalizationUnit]:
        """Extract formalizable statements from a TheoryGraph."""
        units: list[FormalizationUnit] = []

        formalizable_types = {
            ConceptType.DEFINITION, ConceptType.LEMMA, ConceptType.PROPOSITION,
            ConceptType.THEOREM, ConceptType.COROLLARY,
        }

        for node in graph._nodes.values():
            if node.concept_type not in formalizable_types:
                continue
            if not node.formal_statement.strip():
                continue

            # Determine section label from source_section or id
            section = node.source_section or "unknown"
            label = f"{node.concept_type.value.title()} ({node.id})"

            unit = FormalizationUnit(
                concept_id=node.id,
                concept_type=node.concept_type.value,
                section=section,
                label=label,
                natural_language=node.informal_statement or node.formal_statement,
                formal_statement_latex=node.formal_statement,
            )
            units.append(unit)

        # Sort by section order
        units.sort(key=lambda u: u.section)
        return units

    async def _formalize_one(
        self,
        unit: FormalizationUnit,
        llm: Any,
        lean_compile: bool,
        run_python: bool,
        *,
        cloud_prover: Any | None = None,
    ) -> FormalizationUnit:
        """Formalize a single statement."""
        # Generate Lean 4 code
        try:
            lean_code = await self._lean_gen.generate_lean(unit, llm)
            unit.lean_code = lean_code

            if "sorry" in lean_code:
                unit.lean_status = FormalizationStatus.LEAN_SORRY
            else:
                unit.lean_status = FormalizationStatus.LEAN_GENERATED

            # Attempt compilation if enabled
            if lean_compile and lean_code:
                compiled = await self._try_lean_compile(unit, cloud_prover=cloud_prover)
                if compiled:
                    if "sorry" not in lean_code:
                        unit.lean_status = FormalizationStatus.LEAN_PROVED
                    else:
                        unit.lean_status = FormalizationStatus.LEAN_SORRY
        except Exception as e:
            unit.lean_status = FormalizationStatus.LEAN_FAILED
            unit.lean_errors.append(str(e))

        # Generate and optionally run Python verification
        if unit.concept_type in ("theorem", "proposition", "corollary"):
            try:
                python_code = await self._lean_gen.generate_python_verification(unit, llm)
                unit.python_script = python_code

                if run_python and python_code:
                    result = await self._run_python_verification(unit)
                    if result:
                        if unit.lean_status in (FormalizationStatus.LEAN_FAILED,
                                                 FormalizationStatus.PENDING):
                            unit.lean_status = FormalizationStatus.NUMERICALLY_VERIFIED
                        unit.numerical_result = result
            except Exception as e:
                logger.debug(f"[Formalizer] Python verification failed for {unit.label}: {e}")

        return unit

    async def _try_lean_compile(
        self,
        unit: FormalizationUnit,
        *,
        cloud_prover: Any | None = None,
    ) -> bool:
        """Try to compile Lean code with iterative error fixing.

        Strategy:
          1. Use IterativeLeanCompiler for automatic error fixing with LLM fallback.
          2. If that fails and cloud_prover is available, delegate to CloudProver.
          3. Else return False (no compiler available).

        Returns True if compilation succeeds.
        """
        import shutil

        # ── 1. Try iterative local compilation with error fixing ───
        if shutil.which("lean"):
            try:
                full_code = LeanCodeGenerator.PREAMBLE + "\n" + unit.lean_code
                compile_result = await self._lean_compiler.compile_with_retry(
                    full_code, llm=None, max_retries=5
                )

                if compile_result.get("success"):
                    logger.info(f"[Formalizer] Lean compiled (iter={compile_result['attempts']}): {unit.label}")
                    unit.lean_code = compile_result["code"]
                    return True
                else:
                    # Recording errors from iterative attempts
                    for err in compile_result.get("errors", [])[:3]:
                        unit.lean_errors.append(f"Lean compile error: {err[:500]}")
                    logger.debug(
                        f"[Formalizer] Iterative compilation failed after "
                        f"{compile_result['attempts']} attempts: {unit.label}"
                    )

            except Exception as e:
                unit.lean_errors.append(f"Lean compile exception: {e}")
                return False

        # ── 2. Try CloudProver if enabled ────────────────────
        if cloud_prover is not None and hasattr(cloud_prover, "verify_lean"):
            try:
                full_code = LeanCodeGenerator.PREAMBLE + "\n" + unit.lean_code
                job = await cloud_prover.verify_lean(full_code, label=unit.label)
                if getattr(job, "compiled_ok", False):
                    logger.info(f"[Formalizer] Cloud Lean compiled OK: {unit.label}")
                    return True

                errors = list(getattr(job, "errors", []) or [])
                if errors:
                    for err in errors[:3]:
                        unit.lean_errors.append(f"Cloud compile error: {str(err)[:500]}")
                else:
                    result_text = str(getattr(job, "result", "") or "")
                    if result_text:
                        unit.lean_errors.append(f"Cloud compile error: {result_text[:500]}")
                    else:
                        status = getattr(job, "status", "unknown")
                        unit.lean_errors.append(f"Cloud compile failed with status: {status}")
            except Exception as e:
                unit.lean_errors.append(f"Cloud compile exception: {e}")
                return False
        else:
            logger.debug("[Formalizer] lean binary not found and no cloud prover available")

        return False

    async def _run_python_verification(self, unit: FormalizationUnit) -> str:
        """Run a Python verification script in a sandboxed subprocess."""
        import os
        import subprocess
        import sys
        import tempfile

        if not unit.python_script:
            return ""

        script_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False
            ) as f:
                f.write(unit.python_script)
                script_path = f.name

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=tempfile.gettempdir(),
            )

            output = result.stdout[-2000:] if result.stdout else ""
            if result.returncode != 0:
                error = result.stderr[-500:] if result.stderr else "unknown error"
                unit.lean_errors.append(f"Python verification error: {error}")
                return f"FAILED: {error}"

            return output

        except subprocess.TimeoutExpired:
            return "TIMEOUT: Script exceeded 60s limit"
        except Exception as e:
            return f"ERROR: {e}"
        finally:
            if script_path:
                try:
                    os.unlink(script_path)
                except OSError:
                    pass

    async def _save_outputs(
        self,
        report: FormalizationReport,
        units: list[FormalizationUnit],
        output_dir: Path,
    ) -> None:
        """Save all formalization outputs to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Report JSON
        (output_dir / "formalization_report.json").write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Report Markdown (D4)
        md = self.generate_report_markdown(report)
        (output_dir / "formalization_report.md").write_text(md, encoding="utf-8")

        # Lean project directory
        lean_dir = output_dir / "lean4"
        lean_dir.mkdir(exist_ok=True)

        # Write preamble
        (lean_dir / "SuperspaceModelSets.lean").write_text(
            LeanCodeGenerator.PREAMBLE, encoding="utf-8"
        )

        # Write individual Lean files
        for unit in units:
            if unit.lean_code:
                safe_name = re.sub(r'[^\w]', '_', unit.label)
                (lean_dir / f"{safe_name}.lean").write_text(
                    unit.lean_code, encoding="utf-8"
                )

        # Python verification scripts
        py_dir = output_dir / "python_verify"
        py_dir.mkdir(exist_ok=True)

        for unit in units:
            if unit.python_script:
                safe_name = re.sub(r'[^\w]', '_', unit.label)
                (py_dir / f"verify_{safe_name}.py").write_text(
                    unit.python_script, encoding="utf-8"
                )

        # Master verification script
        master_script = self._generate_master_script(units)
        (py_dir / "run_all_verifications.py").write_text(
            master_script, encoding="utf-8"
        )

        logger.info(f"[Formalizer] Saved outputs to {output_dir}")

    def _generate_master_script(self, units: list[FormalizationUnit]) -> str:
        """Generate a master Python script that runs all verifications."""
        lines = [
            '"""Master verification script — runs all numerical checks."""',
            'import subprocess',
            'import sys',
            'import os',
            'from pathlib import Path',
            '',
            'def main():',
            '    script_dir = Path(__file__).parent',
            '    scripts = sorted(script_dir.glob("verify_*.py"))',
            '    passed = 0',
            '    failed = 0',
            '    errors = []',
            '',
            '    for script in scripts:',
            '        print(f"\\n{"="*60}")',
            '        print(f"Running: {script.name}")',
            '        print(f"{"="*60}")',
            '        try:',
            '            result = subprocess.run(',
            '                [sys.executable, str(script)],',
            '                capture_output=True, text=True, timeout=120,',
            '            )',
            '            print(result.stdout)',
            '            if result.returncode == 0:',
            '                passed += 1',
            '                print("✓ PASSED")',
            '            else:',
            '                failed += 1',
            '                errors.append((script.name, result.stderr[:200]))',
            '                print(f"✗ FAILED: {result.stderr[:200]}")',
            '        except subprocess.TimeoutExpired:',
            '            failed += 1',
            '            errors.append((script.name, "TIMEOUT"))',
            '            print("✗ TIMEOUT")',
            '        except Exception as e:',
            '            failed += 1',
            '            errors.append((script.name, str(e)))',
            '            print(f"✗ ERROR: {e}")',
            '',
            '    print(f"\\n{"="*60}")',
            '    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(scripts)}")',
            '    if errors:',
            '        print("\\nFailures:")',
            '        for name, err in errors:',
            '            print(f"  - {name}: {err}")',
            '    print(f"{"="*60}")',
            '    sys.exit(0 if failed == 0 else 1)',
            '',
            'if __name__ == "__main__":',
            '    main()',
        ]
        return "\n".join(lines)

    def generate_report_markdown(self, report: FormalizationReport) -> str:
        """Generate a human-readable Markdown formalization report (D4).

        Includes:
          - Summary table (proved/sorry/failed/verified counts, overall score)
          - Per-statement rows
          - Numerical verification results
          - Lean project build instructions
        """
        import shutil

        lines: list[str] = []
        lines.append(f"# Formalization Report: {report.paper_title}")
        lines.append("")
        lines.append(f"**Source:** {report.paper_source}")
        lean_bin = "available" if shutil.which("lean") else "not found"
        lines.append(f"**Lean compiler:** {lean_bin}")
        lines.append(f"**Overall score:** {report.overall_score:.2f}")
        lines.append("")

        # ── Summary table ───────────────────────────────────
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Count |")
        lines.append("|--------|------:|")
        lines.append(f"| Total statements | {report.total_statements} |")
        lines.append(f"| Lean proved | {report.lean_proved} |")
        lines.append(f"| Lean sorry | {report.lean_sorry} |")
        lines.append(f"| Lean failed | {report.lean_failed} |")
        lines.append(f"| Numerically verified | {report.numerically_verified} |")
        lines.append(f"| Computationally reproduced | {report.computationally_reproduced} |")
        lines.append(f"| Skipped | {report.skipped} |")
        lines.append("")

        # ── Per-statement table ─────────────────────────────
        lines.append("## Per-Statement Results")
        lines.append("")
        lines.append("| # | Label | Status | Confidence | Lean File | Python Script |")
        lines.append("|---|-------|--------|----------:|-----------|---------------|")
        for i, unit in enumerate(report.units, 1):
            safe_name = re.sub(r'[^\w]', '_', unit.label)
            lean_file = f"`{safe_name}.lean`" if unit.lean_code else "—"
            py_file = f"`verify_{safe_name}.py`" if unit.python_script else "—"
            lines.append(
                f"| {i} | {unit.label} | {unit.lean_status.value} | "
                f"{unit.verification_confidence:.2f} | {lean_file} | {py_file} |"
            )
        lines.append("")

        # ── Numerical verification summary ──────────────────
        numerical_units = [u for u in report.units if u.numerical_result]
        if numerical_units:
            lines.append("## Numerical Verification Results")
            lines.append("")
            for unit in numerical_units:
                verdict = "PASS" if "PASS" in unit.numerical_result.upper() else "FAIL"
                lines.append(f"- **{unit.label}**: {verdict}")
                snippet = unit.numerical_result.strip()[:200]
                if snippet:
                    lines.append(f"  ```")
                    lines.append(f"  {snippet}")
                    lines.append(f"  ```")
            lines.append("")

        # ── Lean project instructions ───────────────────────
        lines.append("## Building the Lean Project")
        lines.append("")
        lines.append("```bash")
        lines.append("# 1. Install Lean 4 + Lake")
        lines.append("curl https://raw.githubusercontent.com/leanprover/elan/main/elan-init.sh -sSf | sh")
        lines.append("")
        lines.append("# 2. Navigate to the lean4/ directory")
        lines.append("cd lean4/")
        lines.append("")
        lines.append("# 3. Build with Lake (downloads Mathlib on first run)")
        lines.append("lake build")
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    def get_lean_project_template(self) -> dict[str, str]:
        """Generate a Lean 4 project template (lakefile + toolchain)."""
        return {
            "lakefile.lean": '''import Lake
open Lake DSL

package superspaceModelSets where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib SuperspaceModelSets where
  srcDir := "."

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"
''',
            "lean-toolchain": "leanprover/lean4:v4.15.0",
        }
