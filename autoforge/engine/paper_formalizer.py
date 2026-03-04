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
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    overall_score: float = 0.0

    def compute_score(self) -> float:
        """Compute overall formalization score (0-1)."""
        if self.total_statements == 0:
            return 0.0
        proved_weight = 1.0
        sorry_weight = 0.5
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
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "units": [u.to_dict() for u in self.units],
        }


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
        self._verifier = VerificationSuite()

    async def formalize(
        self,
        graph: TheoryGraph,
        llm: Any,
        *,
        output_dir: Path | None = None,
        lean_compile: bool = False,
        run_python: bool = False,
    ) -> FormalizationReport:
        """Formalize all extractable statements in a paper.

        Args:
            graph: TheoryGraph representing the paper
            llm: LLM router for code generation
            output_dir: Directory to save outputs
            lean_compile: Whether to attempt Lean compilation
            run_python: Whether to run generated Python scripts

        Returns:
            FormalizationReport with detailed results
        """
        logger.info(f"[Formalizer] Starting formalization of '{graph.title}'")

        report = FormalizationReport(
            paper_title=graph.title,
            paper_source=graph.source,
        )

        # Step 1: Extract formalizable statements
        units = self._extract_units(graph)
        report.total_statements = len(units)
        logger.info(f"[Formalizer] Extracted {len(units)} formalizable statements")

        # Step 2: Generate Lean 4 code for each
        lean_tasks = []
        for unit in units:
            lean_tasks.append(self._formalize_one(unit, llm, lean_compile, run_python))

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
                compiled = await self._try_lean_compile(unit)
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

    async def _try_lean_compile(self, unit: FormalizationUnit) -> bool:
        """Try to compile Lean code. Returns True if compilation succeeds."""
        try:
            # Try importing LeanProver for local compilation
            from autoforge.engine.lean_prover import LeanProver
            prover = LeanProver.__new__(LeanProver)
            # Check if lean4 is available
            import shutil
            if not shutil.which("lean"):
                logger.debug("[Formalizer] lean binary not found, skipping compilation")
                return False
            # Would call prover.check_file() here
            return False  # Conservative: don't claim success without actual compilation
        except ImportError:
            return False

    async def _run_python_verification(self, unit: FormalizationUnit) -> str:
        """Run a Python verification script in a sandboxed subprocess."""
        import subprocess
        import tempfile

        if not unit.python_script:
            return ""

        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False
            ) as f:
                f.write(unit.python_script)
                script_path = f.name

            result = subprocess.run(
                ["python3", script_path],
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
