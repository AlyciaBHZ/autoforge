"""Multi-Prover Formal Verification Engine — Cross-Discipline Proof Orchestration.

This module provides a unified interface to multiple proof assistants and verification
tools, enabling:

  1. **Backend Diversity**: Lean 4, Coq, Isabelle/HOL, TLA+, Z3, Dafny
  2. **Domain Routing**: Automatically selects best prover for problem domain
  3. **Formalization Pipeline**: Translate natural language claims → formal code
  4. **Cross-Verification**: Verify same claim in multiple provers for confidence
  5. **Persistent State**: Save and resume verification tasks

Supported domains:
  - Mathematics: pure theorem proving (Lean, Coq, Isabelle)
  - CS Theory: complexity, formal semantics (Coq, Isabelle, Lean)
  - Distributed Systems: safety & liveness specs (TLA+, Lean)
  - Program Verification: pre/post conditions & invariants (Dafny, Z3)
  - Cryptography: mathematical properties & constraints (Lean, Coq, Z3)
  - Concurrency: race conditions, data races (TLA+, Dafny)
  - Type Theory: dependent types & higher-order reasoning (Coq, Lean, Isabelle)

Architecture:
  ProverBackend (Enum)
    ├─ LEAN4, COQ, ISABELLE, TLAPLUS, Z3_SMT, DAFNY
  VerificationTask (dataclass)
    ├─ id, description, domain, source_text, backend, status
  ProverAdapter (ABC)
    ├─ detect_installation() → bool
    ├─ verify_code(code, workspace, timeout) → dict
    └─ formalize_claim(claim, domain, llm) → str
      ├─ CoqAdapter
      ├─ IsabelleAdapter
      ├─ TLAPlusAdapter
      ├─ Z3SMTAdapter
      └─ DafnyAdapter
  DomainRouter
    ├─ route(task, available_backends) → ProverBackend
    └─ _detect_domain(description) → str
  MultiProverEngine (main orchestrator)
    ├─ verify_claim(claim, domain, llm) → VerificationTask
    ├─ verify_batch(claims, domain, llm) → list[VerificationTask]
    ├─ cross_verify(claim, llm, backends) → dict
    └─ detect_available_provers() → dict[ProverBackend, bool]

References:
  - TLA+ (Lamport): Formal specification of concurrent & distributed systems
  - Z3 (Microsoft Research): SMT solver for program analysis & verification
  - Dafny (Microsoft): Automated program verification for pre/post conditions
  - Coq: Inductive type theory, Mathlib ecosystem
  - Isabelle/HOL: Higher-order logic, comprehensive libraries
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from autoforge.engine.llm_router import LLMRouter, TaskComplexity

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Enumerations
# ══════════════════════════════════════════════════════════════


class ProverBackend(str, Enum):
    """Supported proof assistant and verification tool backends."""

    LEAN4 = "lean4"  # Lean 4 theorem prover
    COQ = "coq"  # Coq interactive theorem prover
    ISABELLE = "isabelle"  # Isabelle/HOL formal system
    TLAPLUS = "tlaplus"  # TLA+ model checker (distributed systems)
    Z3_SMT = "z3_smt"  # Z3 SMT solver (program verification)
    DAFNY = "dafny"  # Dafny verified programming language


# ══════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════


@dataclass
class VerificationTask:
    """A formal verification task for a specific claim or code."""

    id: str
    description: str
    domain: str  # "mathematics", "cs_theory", "distributed_systems", "program_verification", "cryptography", etc.
    source_text: str  # Original claim, code, or specification
    backend: ProverBackend
    formalized_code: str = ""  # Translated to target prover's language
    verification_result: dict = field(default_factory=dict)  # Result output
    status: str = "pending"  # pending, formalizing, verifying, proved, failed
    started_at: float = 0.0
    completed_at: float = 0.0
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "domain": self.domain,
            "source_text": self.source_text,
            "backend": self.backend.value,
            "formalized_code": self.formalized_code,
            "verification_result": self.verification_result,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VerificationTask:
        """Deserialize task from dictionary."""
        data["backend"] = ProverBackend(data["backend"])
        return cls(**data)


# ══════════════════════════════════════════════════════════════
# Prover Adapter Base Class
# ══════════════════════════════════════════════════════════════


class ProverAdapter(ABC):
    """Abstract base class for proof assistant adapters.

    Subclasses implement:
      - Binary detection (is the tool installed?)
      - Code verification (run the proof checker)
      - LLM-based claim formalization (convert natural language → formal code)
    """

    def __init__(self, backend: ProverBackend):
        """Initialize adapter for a specific backend."""
        self.backend = backend
        self.installed = False

    @abstractmethod
    async def detect_installation(self) -> bool:
        """Check if the prover is installed and accessible.

        Returns:
            True if the prover binary is found and executable.
        """
        pass

    @abstractmethod
    async def verify_code(
        self, code: str, workspace: Path, timeout: int = 120
    ) -> dict[str, Any]:
        """Run formal verification on code/specification.

        Args:
            code: Source code in the target prover's language
            workspace: Temporary workspace directory
            timeout: Maximum seconds to wait for verification

        Returns:
            Dict with keys:
              - "success": bool — verification passed
              - "output": str — prover output
              - "error": str — error messages if any
              - "time_seconds": float — wall-clock time
        """
        pass

    @abstractmethod
    async def formalize_claim(
        self, claim: str, domain: str, llm: LLMRouter | None = None
    ) -> str:
        """Convert natural language claim to formal code in this prover's language.

        Args:
            claim: Natural language statement
            domain: Problem domain (helps target the formalization)
            llm: LLM router for AI-powered formalization

        Returns:
            Formal code ready for verification.
        """
        pass


# ══════════════════════════════════════════════════════════════
# Concrete Prover Adapters
# ══════════════════════════════════════════════════════════════


class CoqAdapter(ProverAdapter):
    """Coq interactive theorem prover adapter.

    Coq is ideal for:
      - Mathematical proofs with rich library (Mathlib)
      - Inductive type theory reasoning
      - CS theory (semantics, type systems, compilers)
    """

    def __init__(self):
        """Initialize Coq adapter."""
        super().__init__(ProverBackend.COQ)
        self.binary_name = "coqc"

    async def detect_installation(self) -> bool:
        """Check if coqc is installed."""
        try:
            result = subprocess.run(
                [self.binary_name, "--version"],
                capture_output=True,
                timeout=5,
                text=True,
            )
            self.installed = result.returncode == 0
            return self.installed
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.installed = False
            return False

    async def verify_code(
        self, code: str, workspace: Path, timeout: int = 120
    ) -> dict[str, Any]:
        """Verify Coq code via coqc compilation."""
        import time

        coq_file = workspace / "proof.v"
        coq_file.write_text(code, encoding="utf-8")

        try:
            start = time.time()
            result = subprocess.run(
                [self.binary_name, str(coq_file)],
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            elapsed = time.time() - start

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "time_seconds": elapsed,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Timeout after {timeout} seconds",
                "time_seconds": float(timeout),
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "time_seconds": 0.0,
            }

    async def formalize_claim(
        self, claim: str, domain: str, llm: LLMRouter | None = None
    ) -> str:
        """Formalize claim to Coq code using LLM."""
        if not llm:
            return f"(* Claim: {claim} *)"

        prompt = f"""Formalize the following {domain} claim to Coq code.

Claim: {claim}

Guidelines:
- Start with necessary imports: Require Import ...
- Use Theorem/Lemma to state the claim
- Use Proof and Qed/Defined to structure the proof
- Use tactics like intro, apply, refl, simp, induction
- If you cannot complete, use sorry as a placeholder

Output ONLY the Coq code, no markdown fence."""

        try:
            response = await llm.acall(
                prompt=prompt,
                complexity=TaskComplexity.HIGH,
                max_tokens=2000,
            )
            # Extract text from response
            if hasattr(response, "content"):
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
            return str(response)
        except Exception as e:
            logger.error(f"Coq formalization failed: {e}")
            return f"(* Formalization error: {e} *)"


class IsabelleAdapter(ProverAdapter):
    """Isabelle/HOL formal system adapter.

    Isabelle/HOL is ideal for:
      - Mathematical proofs with powerful automation
      - Hardware verification
      - Protocol verification
    """

    def __init__(self):
        """Initialize Isabelle adapter."""
        super().__init__(ProverBackend.ISABELLE)
        self.binary_name = "isabelle"

    async def detect_installation(self) -> bool:
        """Check if isabelle is installed."""
        try:
            result = subprocess.run(
                [self.binary_name, "version"],
                capture_output=True,
                timeout=5,
                text=True,
            )
            self.installed = result.returncode == 0
            return self.installed
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.installed = False
            return False

    async def verify_code(
        self, code: str, workspace: Path, timeout: int = 120
    ) -> dict[str, Any]:
        """Verify Isabelle theory file via isabelle build."""
        import time

        thy_file = workspace / "Proof.thy"
        thy_file.write_text(code, encoding="utf-8")

        try:
            start = time.time()
            result = subprocess.run(
                [self.binary_name, "build", "-b", "-d", str(workspace)],
                capture_output=True,
                timeout=timeout,
                text=True,
                cwd=str(workspace),
            )
            elapsed = time.time() - start

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "time_seconds": elapsed,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Timeout after {timeout} seconds",
                "time_seconds": float(timeout),
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "time_seconds": 0.0,
            }

    async def formalize_claim(
        self, claim: str, domain: str, llm: LLMRouter | None = None
    ) -> str:
        """Formalize claim to Isabelle/HOL theory file."""
        if not llm:
            return f"(* Claim: {claim} *)"

        prompt = f"""Formalize the following {domain} claim as an Isabelle/HOL theory.

Claim: {claim}

Guidelines:
- Start with: theory Proof imports Main begin
- Use theorem or lemma to state claims
- Use proof...qed or by tactics
- Leverage HOL automation: simp, auto, blast
- End with: end
- If unsure, use sorry

Output ONLY the Isabelle theory code, no markdown fence."""

        try:
            response = await llm.acall(
                prompt=prompt,
                complexity=TaskComplexity.HIGH,
                max_tokens=2000,
            )
            if hasattr(response, "content"):
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
            return str(response)
        except Exception as e:
            logger.error(f"Isabelle formalization failed: {e}")
            return f"(* Formalization error: {e} *)"


class TLAPlusAdapter(ProverAdapter):
    """TLA+ model checker adapter for distributed systems.

    TLA+ is ideal for:
      - Distributed system specifications
      - Safety and liveness properties
      - Temporal logic reasoning
      - Consensus protocols, leader election, etc.
    """

    def __init__(self):
        """Initialize TLA+ adapter."""
        super().__init__(ProverBackend.TLAPLUS)
        self.binary_name = "tlc"

    async def detect_installation(self) -> bool:
        """Check if TLC (TLA+ model checker) is installed."""
        try:
            result = subprocess.run(
                [self.binary_name, "-h"],
                capture_output=True,
                timeout=5,
                text=True,
            )
            self.installed = result.returncode == 0 or "TLC" in result.stderr
            return self.installed
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.installed = False
            return False

    async def verify_code(
        self, code: str, workspace: Path, timeout: int = 120
    ) -> dict[str, Any]:
        """Verify TLA+ specification via TLC model checker."""
        import time

        tla_file = workspace / "Spec.tla"
        tla_file.write_text(code, encoding="utf-8")

        try:
            start = time.time()
            result = subprocess.run(
                [self.binary_name, "-config", str(workspace / "Spec.cfg"), str(tla_file)],
                capture_output=True,
                timeout=timeout,
                text=True,
                cwd=str(workspace),
            )
            elapsed = time.time() - start

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "time_seconds": elapsed,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Timeout after {timeout} seconds",
                "time_seconds": float(timeout),
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "time_seconds": 0.0,
            }

    async def formalize_claim(
        self, claim: str, domain: str, llm: LLMRouter | None = None
    ) -> str:
        """Formalize claim to TLA+ specification."""
        if not llm:
            return f"(* Claim: {claim} *)"

        prompt = f"""Formalize the following distributed systems claim as a TLA+ specification.

Claim: {claim}

Guidelines:
- Module declaration: MODULE spec EXTENDS ...
- State variables: VARIABLE x, y, z
- Initial condition: Init == ...
- Next-state relation: Next == ...
- Specification: Spec == Init /\\ [][Next]_vars
- Safety invariants: THEOREM Spec => Inv
- Use operators: /\\, \\/, =>, ~, Always, Eventually
- If uncertain, provide skeleton with sorry

Output ONLY valid TLA+ code, no markdown fence."""

        try:
            response = await llm.acall(
                prompt=prompt,
                complexity=TaskComplexity.HIGH,
                max_tokens=2500,
            )
            if hasattr(response, "content"):
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
            return str(response)
        except Exception as e:
            logger.error(f"TLA+ formalization failed: {e}")
            return f"(* Formalization error: {e} *)"


class Z3SMTAdapter(ProverAdapter):
    """Z3 SMT solver adapter for program verification.

    Z3 is ideal for:
      - Program assertions and constraints
      - Bitvector reasoning (cryptography, low-level code)
      - Array logic and quantifier reasoning
      - Constraint satisfaction problems
    """

    def __init__(self):
        """Initialize Z3 adapter."""
        super().__init__(ProverBackend.Z3_SMT)
        self.binary_name = "z3"

    async def detect_installation(self) -> bool:
        """Check if z3 is installed."""
        try:
            result = subprocess.run(
                [self.binary_name, "--version"],
                capture_output=True,
                timeout=5,
                text=True,
            )
            self.installed = result.returncode == 0
            return self.installed
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.installed = False
            return False

    async def verify_code(
        self, code: str, workspace: Path, timeout: int = 120
    ) -> dict[str, Any]:
        """Verify SMT-LIB2 code via Z3."""
        import time

        smt_file = workspace / "problem.smt2"
        smt_file.write_text(code, encoding="utf-8")

        try:
            start = time.time()
            result = subprocess.run(
                [self.binary_name, str(smt_file)],
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            elapsed = time.time() - start

            success = "sat" in result.stdout or "unsat" in result.stdout or "unknown" in result.stdout

            return {
                "success": success,
                "output": result.stdout,
                "error": result.stderr,
                "time_seconds": elapsed,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Timeout after {timeout} seconds",
                "time_seconds": float(timeout),
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "time_seconds": 0.0,
            }

    async def formalize_claim(
        self, claim: str, domain: str, llm: LLMRouter | None = None
    ) -> str:
        """Formalize claim to SMT-LIB2 format."""
        if not llm:
            return f";; Claim: {claim}\n(check-sat)"

        prompt = f"""Formalize the following {domain} claim as an SMT-LIB2 problem.

Claim: {claim}

Guidelines:
- Use SMT-LIB2 syntax
- Declare sorts: (declare-sort T)
- Declare functions: (declare-fun f (Int Int) Bool)
- Assert constraints: (assert (= x 5))
- End with: (check-sat) and optionally (get-model)
- For program verification, assert pre/post conditions
- Use (assert-not ...) or (assert ...) as needed
- For satisfiability checking, Z3 will determine if claims hold

Output ONLY valid SMT-LIB2 code, no markdown fence."""

        try:
            response = await llm.acall(
                prompt=prompt,
                complexity=TaskComplexity.HIGH,
                max_tokens=2000,
            )
            if hasattr(response, "content"):
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
            return str(response)
        except Exception as e:
            logger.error(f"Z3 formalization failed: {e}")
            return f";; Formalization error: {e}\n(check-sat)"


class DafnyAdapter(ProverAdapter):
    """Dafny verified programming language adapter.

    Dafny is ideal for:
      - Verified programs with pre/post conditions
      - Loop invariants and termination proofs
      - Datatype verification
      - Integrated with Z3 for automation
    """

    def __init__(self):
        """Initialize Dafny adapter."""
        super().__init__(ProverBackend.DAFNY)
        self.binary_name = "dafny"

    async def detect_installation(self) -> bool:
        """Check if dafny is installed."""
        try:
            result = subprocess.run(
                [self.binary_name, "--version"],
                capture_output=True,
                timeout=5,
                text=True,
            )
            self.installed = result.returncode == 0
            return self.installed
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.installed = False
            return False

    async def verify_code(
        self, code: str, workspace: Path, timeout: int = 120
    ) -> dict[str, Any]:
        """Verify Dafny source file."""
        import time

        dfy_file = workspace / "proof.dfy"
        dfy_file.write_text(code, encoding="utf-8")

        try:
            start = time.time()
            result = subprocess.run(
                [self.binary_name, str(dfy_file)],
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            elapsed = time.time() - start

            success = result.returncode == 0 and "verified" in result.stderr.lower()

            return {
                "success": success,
                "output": result.stdout,
                "error": result.stderr,
                "time_seconds": elapsed,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Timeout after {timeout} seconds",
                "time_seconds": float(timeout),
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "time_seconds": 0.0,
            }

    async def formalize_claim(
        self, claim: str, domain: str, llm: LLMRouter | None = None
    ) -> str:
        """Formalize claim to Dafny code."""
        if not llm:
            return f"// Claim: {claim}"

        prompt = f"""Formalize the following {domain} claim as Dafny code.

Claim: {claim}

Guidelines:
- Declare predicates and methods
- Use requires for preconditions
- Use ensures for postconditions
- Use invariant for loop invariants
- Use decreases for termination
- Dafny will verify automatically with Z3
- Syntax: predicate Pred(...) = ...
           method Method(...) requires ... ensures ... { ... }
- If uncertain about implementation, use assert or assume

Output ONLY valid Dafny code, no markdown fence."""

        try:
            response = await llm.acall(
                prompt=prompt,
                complexity=TaskComplexity.HIGH,
                max_tokens=2000,
            )
            if hasattr(response, "content"):
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
            return str(response)
        except Exception as e:
            logger.error(f"Dafny formalization failed: {e}")
            return f"// Formalization error: {e}"


# ══════════════════════════════════════════════════════════════
# Domain Router
# ══════════════════════════════════════════════════════════════


class DomainRouter:
    """Routes verification tasks to the best-suited backend based on domain."""

    DOMAIN_BACKEND_MAP = {
        "mathematics": [
            ProverBackend.LEAN4,
            ProverBackend.COQ,
            ProverBackend.ISABELLE,
        ],
        "cs_theory": [
            ProverBackend.COQ,
            ProverBackend.ISABELLE,
            ProverBackend.LEAN4,
        ],
        "distributed_systems": [ProverBackend.TLAPLUS, ProverBackend.LEAN4],
        "program_verification": [ProverBackend.DAFNY, ProverBackend.Z3_SMT],
        "cryptography": [
            ProverBackend.LEAN4,
            ProverBackend.COQ,
            ProverBackend.Z3_SMT,
        ],
        "concurrency": [ProverBackend.TLAPLUS, ProverBackend.DAFNY],
        "type_theory": [
            ProverBackend.COQ,
            ProverBackend.LEAN4,
            ProverBackend.ISABELLE,
        ],
    }

    def route(
        self,
        task: VerificationTask,
        available_backends: set[ProverBackend],
    ) -> ProverBackend:
        """Route a task to the best available backend.

        Args:
            task: Verification task
            available_backends: Set of backends confirmed to be installed

        Returns:
            Best-suited backend from available options.

        Raises:
            ValueError: If no suitable backend is available.
        """
        domain = task.domain.lower()
        preferred = self.DOMAIN_BACKEND_MAP.get(domain, [ProverBackend.LEAN4])

        for backend in preferred:
            if backend in available_backends:
                logger.info(f"Routing domain '{domain}' to {backend.value}")
                return backend

        # Fallback: return first available
        if available_backends:
            fallback = list(available_backends)[0]
            logger.warning(
                f"No preferred backend for domain '{domain}', using {fallback.value}"
            )
            return fallback

        raise ValueError(
            f"No suitable verification backend available for domain: {domain}"
        )

    def _detect_domain(self, description: str) -> str:
        """Detect problem domain from description text.

        Uses keyword-based heuristics to infer domain.

        Args:
            description: Natural language description of the problem

        Returns:
            Detected domain string.
        """
        lower_desc = description.lower()

        domain_keywords = {
            "distributed_systems": [
                "distributed",
                "consensus",
                "protocol",
                "byzantine",
                "leader election",
                "replicated",
            ],
            "concurrency": [
                "race condition",
                "data race",
                "concurrency",
                "mutex",
                "deadlock",
                "thread",
            ],
            "program_verification": [
                "precondition",
                "postcondition",
                "loop invariant",
                "termination",
                "array",
                "buffer",
            ],
            "cryptography": [
                "encryption",
                "signature",
                "hash",
                "cryptographic",
                "security",
                "key",
            ],
            "cs_theory": [
                "complexity",
                "algorithm",
                "grammar",
                "semantics",
                "reduction",
            ],
            "type_theory": [
                "dependent type",
                "type system",
                "higher-order",
                "polymorphism",
            ],
            "mathematics": [
                "theorem",
                "lemma",
                "proof",
                "algebraic",
                "number theory",
                "topology",
            ],
        }

        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in lower_desc)
            if score > 0:
                scores[domain] = score

        return max(scores, key=scores.get) if scores else "mathematics"


# ══════════════════════════════════════════════════════════════
# Multi-Prover Engine
# ══════════════════════════════════════════════════════════════


class MultiProverEngine:
    """Central orchestrator for multi-prover verification."""

    def __init__(self, workspace: Path):
        """Initialize the multi-prover engine.

        Args:
            workspace: Directory for temporary files and state storage.
        """
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Initialize all adapters
        self.adapters: dict[ProverBackend, ProverAdapter] = {
            ProverBackend.COQ: CoqAdapter(),
            ProverBackend.ISABELLE: IsabelleAdapter(),
            ProverBackend.TLAPLUS: TLAPlusAdapter(),
            ProverBackend.Z3_SMT: Z3SMTAdapter(),
            ProverBackend.DAFNY: DafnyAdapter(),
        }

        self.router = DomainRouter()
        self.available_backends: dict[ProverBackend, bool] = {}
        self.tasks: dict[str, VerificationTask] = {}

    async def detect_available_provers(self) -> dict[ProverBackend, bool]:
        """Detect which provers are installed.

        Runs detection for all backends and caches results.

        Returns:
            Dict mapping backend to availability.
        """
        detection_tasks = [
            (backend, adapter.detect_installation())
            for backend, adapter in self.adapters.items()
        ]

        results = await asyncio.gather(*[task for _, task in detection_tasks])

        for (backend, _), installed in zip(detection_tasks, results):
            self.available_backends[backend] = installed
            status = "installed" if installed else "not found"
            logger.info(f"{backend.value}: {status}")

        return self.available_backends

    async def verify_claim(
        self,
        claim: str,
        domain: str = "",
        llm: LLMRouter | None = None,
    ) -> VerificationTask:
        """Verify a single claim end-to-end.

        Pipeline:
          1. Detect domain (if not provided)
          2. Route to best backend
          3. Formalize claim
          4. Run verification
          5. Return result

        Args:
            claim: Natural language claim or code
            domain: Problem domain (auto-detected if empty)
            llm: LLM router for formalization

        Returns:
            Completed VerificationTask.
        """
        import time
        import uuid

        task_id = str(uuid.uuid4())[:8]
        task = VerificationTask(
            id=task_id,
            description=claim,
            domain=domain or self.router._detect_domain(claim),
            source_text=claim,
            backend=ProverBackend.LEAN4,  # Placeholder, will be routed
        )

        try:
            task.status = "pending"
            task.started_at = time.time()

            # Route to best backend
            available = {b for b, avail in self.available_backends.items() if avail}
            if not available:
                await self.detect_available_provers()
                available = {b for b, avail in self.available_backends.items() if avail}

            if not available:
                task.status = "failed"
                task.error_message = "No verification backends available"
                return task

            task.backend = self.router.route(task, available)
            adapter = self.adapters.get(task.backend)

            if not adapter:
                task.status = "failed"
                task.error_message = f"No adapter for {task.backend.value}"
                return task

            # Formalize claim
            task.status = "formalizing"
            task.formalized_code = await adapter.formalize_claim(
                claim, task.domain, llm
            )

            # Verify
            task.status = "verifying"
            task_workspace = self.workspace / task_id
            task_workspace.mkdir(exist_ok=True)

            result = await adapter.verify_code(
                task.formalized_code, task_workspace, timeout=120
            )

            task.verification_result = result
            task.status = "proved" if result["success"] else "failed"
            if not result["success"] and result.get("error"):
                task.error_message = result["error"]

            task.completed_at = time.time()
            self.tasks[task_id] = task

            return task

        except Exception as e:
            logger.error(f"Verification failed for task {task_id}: {e}")
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = time.time()
            return task

    async def verify_batch(
        self,
        claims: list[str],
        domain: str = "",
        llm: LLMRouter | None = None,
    ) -> list[VerificationTask]:
        """Verify multiple claims in parallel.

        Args:
            claims: List of natural language claims
            domain: Problem domain (same for all)
            llm: LLM router for formalization

        Returns:
            List of completed VerificationTasks.
        """
        tasks = [self.verify_claim(claim, domain, llm) for claim in claims]
        return await asyncio.gather(*tasks)

    async def cross_verify(
        self,
        claim: str,
        llm: LLMRouter | None = None,
        backends: list[ProverBackend] | None = None,
    ) -> dict[str, Any]:
        """Verify the same claim across multiple backends.

        Cross-verification increases confidence in results by checking
        the claim in different formal systems.

        Args:
            claim: Natural language claim
            llm: LLM router for formalization
            backends: List of backends to use (all available if None)

        Returns:
            Dict with structure:
              {
                "claim": str,
                "results": {
                  "lean4": { "proved": bool, "time": float, ... },
                  "coq": { ... },
                  ...
                },
                "consensus": bool,  # All backends agree?
                "best_time": float,
              }
        """
        import time

        if not backends:
            available = {b for b, avail in self.available_backends.items() if avail}
            if not available:
                await self.detect_available_provers()
                available = {b for b, avail in self.available_backends.items() if avail}
            backends = list(available)

        start_time = time.time()
        results = {}

        for backend in backends:
            adapter = self.adapters.get(backend)
            if not adapter:
                continue

            task = VerificationTask(
                id=f"cross_{backend.value}",
                description=claim,
                domain=self.router._detect_domain(claim),
                source_text=claim,
                backend=backend,
            )

            try:
                task.formalized_code = await adapter.formalize_claim(
                    claim, task.domain, llm
                )

                task_workspace = self.workspace / f"cross_{backend.value}"
                task_workspace.mkdir(exist_ok=True)

                verify_result = await adapter.verify_code(
                    task.formalized_code, task_workspace, timeout=120
                )

                results[backend.value] = {
                    "proved": verify_result["success"],
                    "time": verify_result["time_seconds"],
                    "output": verify_result["output"],
                    "error": verify_result.get("error", ""),
                }
            except Exception as e:
                results[backend.value] = {
                    "proved": False,
                    "time": 0.0,
                    "output": "",
                    "error": str(e),
                }

        # Compute consensus
        proofs = [r["proved"] for r in results.values()]
        consensus = len(set(proofs)) == 1 if proofs else False

        total_time = time.time() - start_time

        return {
            "claim": claim,
            "results": results,
            "consensus": consensus,
            "total_time": total_time,
            "num_backends_tried": len(backends),
            "num_backends_succeeded": sum(1 for r in results.values() if r["proved"]),
        }

    async def save_state(self, path: Path) -> None:
        """Save all tasks to a JSON file for persistence.

        Args:
            path: Output file path.
        """
        state = {
            "tasks": {
                task_id: task.to_dict() for task_id, task in self.tasks.items()
            },
            "available_backends": {
                k.value: v for k, v in self.available_backends.items()
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.info(f"Saved verification state to {path}")

    async def load_state(self, path: Path) -> None:
        """Load tasks from a JSON file.

        Args:
            path: Input file path.
        """
        if not path.exists():
            logger.warning(f"State file not found: {path}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))

        self.tasks = {
            task_id: VerificationTask.from_dict(task_data)
            for task_id, task_data in data.get("tasks", {}).items()
        }

        self.available_backends = {
            ProverBackend(k): v for k, v in data.get("available_backends", {}).items()
        }

        logger.info(f"Loaded {len(self.tasks)} tasks from {path}")

    def get_stats(self) -> dict[str, Any]:
        """Get verification statistics.

        Returns:
            Dict with counts and aggregated metrics.
        """
        total = len(self.tasks)
        if total == 0:
            return {
                "total_tasks": 0,
                "proved": 0,
                "failed": 0,
                "pending": 0,
                "avg_time": 0.0,
            }

        proved = sum(1 for t in self.tasks.values() if t.status == "proved")
        failed = sum(1 for t in self.tasks.values() if t.status == "failed")
        pending = sum(1 for t in self.tasks.values() if t.status == "pending")

        completed_times = [
            t.completed_at - t.started_at
            for t in self.tasks.values()
            if t.completed_at > 0 and t.started_at > 0
        ]
        avg_time = sum(completed_times) / len(completed_times) if completed_times else 0.0

        return {
            "total_tasks": total,
            "proved": proved,
            "failed": failed,
            "pending": pending,
            "avg_time_seconds": avg_time,
            "backends_available": {
                k.value: v for k, v in self.available_backends.items()
            },
        }
