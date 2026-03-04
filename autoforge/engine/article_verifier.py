"""Article Verification & Formal Verification Pipeline.

This module provides comprehensive verification of mathematical articles:

  1. **Claim Extraction**: Parse articles (PDF text, LaTeX, Markdown) to extract
     all mathematical claims (theorems, lemmas, propositions, conjectures).

  2. **Logical Consistency Check**: Verify internal logical dependencies —
     does each theorem follow from its stated prerequisites?

  3. **Auto-Formalization**: Translate natural-language claims into Lean 4
     formal statements, using LLM-based auto-formalization with iterative
     compiler-feedback refinement (inspired by PDA/FormL4, ICLR 2025).

  4. **Formal Verification**: Verify formalized claims using Lean 4 (via
     LeanEnvironment) and optionally cross-verify with other provers
     (Coq, Z3, Isabelle) via MultiProverEngine.

  5. **Proof Repair**: Iteratively fix formalization errors using
     ProofRepairEngine (multi-pass sorry elimination).

  6. **Verification Report**: Comprehensive report with per-claim status,
     confidence scores, and cross-verification results.

Architecture:

    ArticleParser
        ↓ extracts
    VerifiableClaim (list)
        ↓ each goes through
    FormalizationPipeline
        ├─ Auto-formalize (LLM → Lean 4)
        ├─ Compile check (Lean 4 verification)
        ├─ Iterative repair (compiler feedback → LLM → retry)
        └─ Cross-verify (multi-prover)
        ↓ results in
    ClaimVerificationResult
        ↓ aggregated into
    ArticleVerificationReport

References:
  - PDA (Process-Driven Autoformalization), ICLR 2025
  - Herald Translator, ICLR 2025 — 96.7% Pass@128 on miniF2F
  - DeepSeek-Prover-V2, arXiv 2504.21801 — recursive subgoal decomposition
  - Pantograph (TACAS 2025) — machine-to-machine Lean 4 interaction
  - LeanInteract, LeanDojo v2 — Python-Lean4 integration
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
# Data Structures
# ══════════════════════════════════════════════════════════════


class ClaimType(str, Enum):
    """Type of mathematical claim."""
    AXIOM = "axiom"
    DEFINITION = "definition"
    LEMMA = "lemma"
    THEOREM = "theorem"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    CONJECTURE = "conjecture"
    REMARK = "remark"


class VerificationStatus(str, Enum):
    """Verification status of a claim."""
    PENDING = "pending"
    FORMALIZED = "formalized"           # Lean 4 code generated
    COMPILES = "compiles"               # Lean 4 compiles (no errors)
    VERIFIED = "verified"               # Lean 4 proof complete (no sorry)
    VERIFIED_WITH_SORRY = "verified_with_sorry"  # Compiles but has sorry
    FAILED = "failed"                   # Lean 4 verification failed
    CROSS_VERIFIED = "cross_verified"   # Verified by multiple provers
    SKIPPED = "skipped"                 # Skipped (e.g., informal remark)
    ERROR = "error"                     # Processing error


@dataclass
class VerifiableClaim:
    """A single mathematical claim extracted from an article."""
    id: str
    claim_type: ClaimType
    label: str = ""                     # e.g., "Theorem 3.2"
    statement: str = ""                 # Natural language statement
    proof_text: str = ""                # Proof text from article
    dependencies: list[str] = field(default_factory=list)  # IDs of claims this depends on
    domain: str = ""                    # Mathematical domain
    section: str = ""                   # Section of the article
    lean_formalization: str = ""        # Lean 4 code
    verification_status: VerificationStatus = VerificationStatus.PENDING
    verification_errors: list[str] = field(default_factory=list)
    sorry_count: int = 0
    repair_attempts: int = 0
    cross_verification: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0            # Overall verification confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.claim_type.value,
            "label": self.label,
            "statement": self.statement,
            "proof_text": self.proof_text[:500],
            "dependencies": self.dependencies,
            "domain": self.domain,
            "section": self.section,
            "lean_formalization": self.lean_formalization,
            "status": self.verification_status.value,
            "errors": self.verification_errors,
            "sorry_count": self.sorry_count,
            "repair_attempts": self.repair_attempts,
            "cross_verification": self.cross_verification,
            "confidence": self.confidence,
        }


@dataclass
class ArticleVerificationReport:
    """Comprehensive verification report for an article."""
    title: str = ""
    total_claims: int = 0
    claims: list[VerifiableClaim] = field(default_factory=list)

    # Summary statistics
    verified: int = 0
    verified_with_sorry: int = 0
    failed: int = 0
    skipped: int = 0
    cross_verified: int = 0
    formalized: int = 0

    # Timing
    total_time: float = 0.0
    formalization_time: float = 0.0
    verification_time: float = 0.0

    # Dependency graph consistency
    dependency_consistent: bool = True
    dependency_issues: list[str] = field(default_factory=list)

    # Overall assessment
    overall_confidence: float = 0.0
    assessment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "total_claims": self.total_claims,
            "verified": self.verified,
            "verified_with_sorry": self.verified_with_sorry,
            "failed": self.failed,
            "skipped": self.skipped,
            "cross_verified": self.cross_verified,
            "formalized": self.formalized,
            "total_time": self.total_time,
            "formalization_time": self.formalization_time,
            "verification_time": self.verification_time,
            "dependency_consistent": self.dependency_consistent,
            "dependency_issues": self.dependency_issues,
            "overall_confidence": self.overall_confidence,
            "assessment": self.assessment,
            "claims": [c.to_dict() for c in self.claims],
        }

    def format_summary(self) -> str:
        """Format human-readable summary."""
        lines = [
            f"# Article Verification Report: {self.title}",
            "",
            f"**Total claims**: {self.total_claims}",
            f"**Verified (complete)**: {self.verified}",
            f"**Verified (with sorry)**: {self.verified_with_sorry}",
            f"**Failed**: {self.failed}",
            f"**Skipped**: {self.skipped}",
            f"**Cross-verified**: {self.cross_verified}",
            f"**Formalized**: {self.formalized}",
            "",
            f"**Overall confidence**: {self.overall_confidence:.1%}",
            f"**Assessment**: {self.assessment}",
            "",
            f"**Total time**: {self.total_time:.1f}s",
        ]

        if self.dependency_issues:
            lines.extend([
                "",
                "## Dependency Issues",
                *[f"  - {issue}" for issue in self.dependency_issues],
            ])

        lines.extend(["", "## Per-Claim Results"])
        for claim in self.claims:
            status_icon = {
                VerificationStatus.VERIFIED: "✓",
                VerificationStatus.CROSS_VERIFIED: "✓✓",
                VerificationStatus.VERIFIED_WITH_SORRY: "~",
                VerificationStatus.FAILED: "✗",
                VerificationStatus.SKIPPED: "—",
                VerificationStatus.FORMALIZED: "○",
            }.get(claim.verification_status, "?")

            label = claim.label or f"{claim.claim_type.value} {claim.id[:6]}"
            lines.append(f"  [{status_icon}] {label}: {claim.verification_status.value}")
            if claim.verification_errors:
                for err in claim.verification_errors[:2]:
                    lines.append(f"      Error: {err[:100]}")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Article Parser
# ══════════════════════════════════════════════════════════════


class ArticleParser:
    """Extract mathematical claims from article text.

    Handles multiple formats:
      - LaTeX (\\begin{theorem}...\\end{theorem} etc.)
      - Markdown (## Theorem 1, **Lemma 2**, etc.)
      - Plain text (heuristic keyword-based extraction)
    """

    # LaTeX environment patterns
    LATEX_ENVS = [
        "theorem", "lemma", "proposition", "corollary",
        "conjecture", "definition", "axiom", "remark",
    ]

    # Markdown patterns
    MD_PATTERNS = [
        r"(?:#+\s*)?(?:\*\*)?(?:Theorem|Lemma|Proposition|Corollary|Conjecture|Definition|Axiom)\s*[\d.]*(?:\*\*)?[.\s:]",
    ]

    async def parse(
        self,
        text: str,
        llm: Any,
        *,
        title: str = "Untitled",
    ) -> list[VerifiableClaim]:
        """Parse article text and extract mathematical claims.

        Uses a hybrid approach:
          1. Regex-based extraction for structured formats (LaTeX, Markdown)
          2. LLM-based extraction for unstructured text
          3. Dependency inference between claims
        """
        claims: list[VerifiableClaim] = []

        # Try LaTeX extraction first
        latex_claims = self._extract_latex(text)
        if latex_claims:
            claims.extend(latex_claims)
        else:
            # Try markdown extraction
            md_claims = self._extract_markdown(text)
            if md_claims:
                claims.extend(md_claims)

        # If structural extraction found few claims, supplement with LLM
        if len(claims) < 3:
            llm_claims = await self._extract_with_llm(text, llm, existing=claims)
            # Merge, avoiding duplicates
            existing_hashes = {c.id for c in claims}
            for lc in llm_claims:
                if lc.id not in existing_hashes:
                    claims.append(lc)

        # Infer dependencies
        claims = await self._infer_dependencies(claims, llm)

        return claims

    def _extract_latex(self, text: str) -> list[VerifiableClaim]:
        """Extract claims from LaTeX environments."""
        claims: list[VerifiableClaim] = []

        for env in self.LATEX_ENVS:
            # Match \begin{theorem}[label]...\end{theorem}
            pattern = (
                rf"\\begin\{{{env}\}}"
                rf"(?:\[([^\]]*)\])?"
                rf"(.*?)"
                rf"\\end\{{{env}\}}"
            )
            for match in re.finditer(pattern, text, re.DOTALL):
                label = match.group(1) or ""
                statement = match.group(2).strip()

                # Look for proof following the environment
                proof_text = ""
                end_pos = match.end()
                proof_match = re.search(
                    r"\\begin\{proof\}(.*?)\\end\{proof\}",
                    text[end_pos:end_pos + 5000],
                    re.DOTALL,
                )
                if proof_match:
                    proof_text = proof_match.group(1).strip()

                claim_type = self._env_to_type(env)
                claim_id = hashlib.sha256(
                    f"{env}:{label}:{statement[:100]}".encode()
                ).hexdigest()[:12]

                claims.append(VerifiableClaim(
                    id=claim_id,
                    claim_type=claim_type,
                    label=f"{env.capitalize()} {label}".strip() if label else "",
                    statement=statement,
                    proof_text=proof_text,
                ))

        return claims

    def _extract_markdown(self, text: str) -> list[VerifiableClaim]:
        """Extract claims from Markdown format."""
        claims: list[VerifiableClaim] = []
        lines = text.split("\n")
        current_claim: dict[str, str] | None = None
        current_body: list[str] = []

        for i, line in enumerate(lines):
            # Check for claim header
            header_match = re.match(
                r"(?:#+\s*)?(?:\*\*)?("
                r"Theorem|Lemma|Proposition|Corollary|Conjecture|Definition|Axiom"
                r")\s*([\d.]*)\s*(?:\*\*)?[.:\s]*(.*)",
                line, re.IGNORECASE,
            )

            if header_match:
                # Save previous claim
                if current_claim:
                    self._finalize_md_claim(current_claim, current_body, claims)

                ctype = header_match.group(1).lower()
                number = header_match.group(2)
                rest = header_match.group(3)
                current_claim = {
                    "type": ctype,
                    "number": number,
                    "first_line": rest,
                }
                current_body = []
            elif current_claim is not None:
                # Check for proof marker
                if re.match(r"(?:\*\*)?(?:Proof|Démonstration)[.:\s]", line, re.IGNORECASE):
                    current_claim["proof_start"] = str(len(current_body))
                current_body.append(line)

        # Save last claim
        if current_claim:
            self._finalize_md_claim(current_claim, current_body, claims)

        return claims

    def _finalize_md_claim(
        self,
        claim_data: dict[str, str],
        body_lines: list[str],
        claims: list[VerifiableClaim],
    ) -> None:
        """Finalize a Markdown claim."""
        body = "\n".join(body_lines).strip()
        ctype = self._env_to_type(claim_data["type"])
        number = claim_data.get("number", "")
        first_line = claim_data.get("first_line", "")

        # Split body into statement and proof
        proof_start = claim_data.get("proof_start")
        if proof_start:
            idx = int(proof_start)
            statement = first_line + "\n" + "\n".join(body_lines[:idx]).strip()
            proof_text = "\n".join(body_lines[idx:]).strip()
        else:
            statement = first_line + "\n" + body if first_line else body
            proof_text = ""

        label = f"{claim_data['type'].capitalize()} {number}".strip()
        claim_id = hashlib.sha256(
            f"{label}:{statement[:100]}".encode()
        ).hexdigest()[:12]

        claims.append(VerifiableClaim(
            id=claim_id,
            claim_type=ctype,
            label=label,
            statement=statement.strip(),
            proof_text=proof_text,
        ))

    async def _extract_with_llm(
        self,
        text: str,
        llm: Any,
        *,
        existing: list[VerifiableClaim] | None = None,
    ) -> list[VerifiableClaim]:
        """Use LLM to extract claims from unstructured text."""
        from autoforge.engine.llm_router import TaskComplexity

        existing_text = ""
        if existing:
            existing_text = "\n".join(
                f"  Already found: {c.label} — {c.statement[:80]}"
                for c in existing
            )

        prompt = f"""Extract ALL mathematical claims from this article text.

## Article (truncated)
{text[:10000]}

## Already extracted (skip these)
{existing_text or "(none)"}

## Instructions
For each claim, extract:
1. type: theorem, lemma, proposition, corollary, conjecture, definition, axiom
2. label: e.g., "Theorem 3.2", "Lemma A.1"
3. statement: the precise mathematical statement
4. proof: the proof text (if provided)
5. domain: the mathematical domain
6. section: which section of the article

Return JSON array:
[
  {{
    "type": "theorem",
    "label": "Theorem 1",
    "statement": "...",
    "proof": "...",
    "domain": "number theory",
    "section": "Section 3"
  }}
]"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You are an expert mathematical article parser. Extract claims precisely.",
                messages=[{"role": "user", "content": prompt}],
            )
            text_out = ""
            for block in response.content:
                if block.type == "text":
                    text_out += block.text

            claims: list[VerifiableClaim] = []
            if "[" in text_out:
                json_str = text_out[text_out.index("["):text_out.rindex("]") + 1]
                items = json.loads(json_str)
                for item in items:
                    if not isinstance(item, dict) or "statement" not in item:
                        continue

                    try:
                        ctype = ClaimType(item.get("type", "theorem"))
                    except ValueError:
                        ctype = ClaimType.THEOREM

                    claim_id = hashlib.sha256(
                        item["statement"][:100].encode()
                    ).hexdigest()[:12]

                    claims.append(VerifiableClaim(
                        id=claim_id,
                        claim_type=ctype,
                        label=item.get("label", ""),
                        statement=item["statement"],
                        proof_text=item.get("proof", ""),
                        domain=item.get("domain", ""),
                        section=item.get("section", ""),
                    ))
            return claims

        except Exception as e:
            logger.debug(f"[ArticleParser] LLM extraction failed: {e}")
            return []

    async def _infer_dependencies(
        self,
        claims: list[VerifiableClaim],
        llm: Any,
    ) -> list[VerifiableClaim]:
        """Infer logical dependencies between claims."""
        if len(claims) < 2:
            return claims

        from autoforge.engine.llm_router import TaskComplexity

        claims_summary = "\n".join(
            f"  [{c.id}] {c.label}: {c.statement[:100]}"
            for c in claims
        )

        prompt = f"""Given these mathematical claims from an article, determine the
logical dependency graph (which claims depend on which).

## Claims
{claims_summary}

## Instructions
For each claim, list the IDs of claims it directly depends on (uses in its proof).
Axioms and definitions have no dependencies.
Return JSON object mapping claim ID → list of dependency IDs:
{{
  "claim_id_1": [],
  "claim_id_2": ["claim_id_1"],
  ...
}}"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You analyze mathematical dependency structures.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            if "{" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]
                deps = json.loads(json_str)
                claim_map = {c.id: c for c in claims}
                for cid, dep_ids in deps.items():
                    if cid in claim_map and isinstance(dep_ids, list):
                        claim_map[cid].dependencies = [
                            d for d in dep_ids if d in claim_map
                        ]
        except Exception as e:
            logger.debug(f"[ArticleParser] Dependency inference failed: {e}")

        return claims

    @staticmethod
    def _env_to_type(env: str) -> ClaimType:
        """Map LaTeX environment name to ClaimType."""
        mapping = {
            "theorem": ClaimType.THEOREM,
            "lemma": ClaimType.LEMMA,
            "proposition": ClaimType.PROPOSITION,
            "corollary": ClaimType.COROLLARY,
            "conjecture": ClaimType.CONJECTURE,
            "definition": ClaimType.DEFINITION,
            "axiom": ClaimType.AXIOM,
            "remark": ClaimType.REMARK,
        }
        return mapping.get(env.lower(), ClaimType.THEOREM)


# ══════════════════════════════════════════════════════════════
# Formalization Pipeline
# ══════════════════════════════════════════════════════════════


class FormalizationPipeline:
    """Auto-formalize mathematical claims into Lean 4.

    Implements iterative compiler-feedback refinement (PDA-style):
      1. Initial formalization via LLM
      2. Compile with Lean 4
      3. If errors → feed errors back to LLM → regenerate
      4. Repeat up to max_iterations
      5. Attempt proof repair for remaining sorries

    Also supports cross-verification via multi-prover engine.
    """

    MAX_REPAIR_ITERATIONS = 3

    def __init__(self) -> None:
        self._formalization_cache: dict[str, str] = {}

    async def formalize_claim(
        self,
        claim: VerifiableClaim,
        llm: Any,
        *,
        context_claims: list[VerifiableClaim] | None = None,
    ) -> VerifiableClaim:
        """Auto-formalize a single claim into Lean 4.

        Uses iterative compiler-feedback refinement:
          1. Generate initial Lean 4 code
          2. Verify with Lean compiler
          3. If errors, feed errors to LLM and regenerate
          4. Repeat until success or max iterations
        """
        if claim.claim_type == ClaimType.REMARK:
            claim.verification_status = VerificationStatus.SKIPPED
            return claim

        from autoforge.engine.llm_router import TaskComplexity

        # Build context from dependent claims
        context = ""
        if context_claims:
            verified_deps = [
                c for c in context_claims
                if c.id in claim.dependencies and c.lean_formalization
            ]
            if verified_deps:
                context = "\n\n".join(
                    f"-- {c.label}\n{c.lean_formalization}"
                    for c in verified_deps
                )

        # Initial formalization
        lean_code = await self._auto_formalize(
            claim, llm, context=context,
        )

        if not lean_code:
            claim.verification_status = VerificationStatus.FAILED
            claim.verification_errors.append("Auto-formalization produced no output")
            return claim

        claim.lean_formalization = lean_code
        claim.verification_status = VerificationStatus.FORMALIZED

        # Iterative compiler-feedback refinement
        for iteration in range(self.MAX_REPAIR_ITERATIONS):
            verify_result = await self._verify_lean(lean_code)

            if verify_result["success"]:
                if verify_result["sorry_count"] > 0:
                    claim.verification_status = VerificationStatus.VERIFIED_WITH_SORRY
                    claim.sorry_count = verify_result["sorry_count"]
                else:
                    claim.verification_status = VerificationStatus.VERIFIED
                    claim.sorry_count = 0
                claim.verification_errors = verify_result.get("warnings", [])
                claim.confidence = 1.0 if claim.sorry_count == 0 else 0.7
                break
            else:
                # Feed errors back and retry
                errors = verify_result.get("errors", [])
                claim.verification_errors = errors
                claim.repair_attempts += 1

                lean_code = await self._repair_formalization(
                    claim, lean_code, errors, llm, context=context,
                )
                claim.lean_formalization = lean_code

                if not lean_code:
                    claim.verification_status = VerificationStatus.FAILED
                    break
        else:
            if claim.verification_status not in (
                VerificationStatus.VERIFIED,
                VerificationStatus.VERIFIED_WITH_SORRY,
            ):
                claim.verification_status = VerificationStatus.FAILED

        return claim

    async def cross_verify_claim(
        self,
        claim: VerifiableClaim,
        llm: Any,
    ) -> VerifiableClaim:
        """Cross-verify a claim using multiple proof backends."""
        if not claim.lean_formalization:
            return claim

        try:
            from autoforge.engine.provers.multi_prover import MultiProverEngine
            engine = MultiProverEngine()
            result = await engine.cross_verify(claim.statement, llm)
            claim.cross_verification = result

            # If cross-verification succeeds, upgrade status
            if isinstance(result, dict):
                successes = sum(
                    1 for v in result.values()
                    if isinstance(v, dict) and v.get("status") == "verified"
                )
                if successes >= 2:
                    claim.verification_status = VerificationStatus.CROSS_VERIFIED
                    claim.confidence = min(1.0, claim.confidence + 0.2)

        except Exception as e:
            logger.debug(f"[Formalization] Cross-verification failed: {e}")
            claim.cross_verification = {"error": str(e)}

        return claim

    async def _auto_formalize(
        self,
        claim: VerifiableClaim,
        llm: Any,
        *,
        context: str = "",
    ) -> str:
        """Generate Lean 4 formalization of a claim."""
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""You are an expert in auto-formalization (natural mathematics → Lean 4).
Formalize the following mathematical claim as a valid Lean 4 theorem with proof.

## Claim ({claim.claim_type.value})
{claim.label}

Statement:
{claim.statement}

{f"Proof from article:{chr(10)}{claim.proof_text[:3000]}" if claim.proof_text else ""}

{f"## Context (previously formalized claims){chr(10)}{context}" if context else ""}

## Requirements
1. Use Lean 4 syntax (not Lean 3)
2. Import Mathlib modules if needed
3. The Lean statement must be mathematically equivalent to the natural language
4. Provide a proof using `by` tactic mode
5. Use `sorry` only for genuinely hard steps, fill in as much as possible
6. Use standard Lean 4 tactics: intro, apply, exact, rw, simp, omega, ring, etc.
7. Include necessary type definitions or auxiliary lemmas

Return ONLY Lean 4 code:
```lean
import Mathlib
-- {claim.label}: {claim.claim_type.value}
...
```"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You are a Lean 4 auto-formalization expert. Write correct, compilable code.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            match = re.search(r"```lean\s*\n?(.*?)\n?```", text, re.DOTALL)
            return match.group(1).strip() if match else ""

        except Exception as e:
            logger.debug(f"[Formalization] Auto-formalize failed: {e}")
            return ""

    async def _repair_formalization(
        self,
        claim: VerifiableClaim,
        lean_code: str,
        errors: list[str],
        llm: Any,
        *,
        context: str = "",
    ) -> str:
        """Repair a failed Lean 4 formalization using compiler feedback."""
        from autoforge.engine.llm_router import TaskComplexity

        errors_text = "\n".join(f"  - {e}" for e in errors[:10])

        prompt = f"""The following Lean 4 formalization has compiler errors. Fix them.

## Original Claim
{claim.statement[:500]}

## Current Lean 4 Code
```lean
{lean_code}
```

## Compiler Errors
{errors_text}

## Instructions
1. Fix ALL compiler errors
2. Do not change the mathematical meaning
3. If a tactic fails, try alternative tactics
4. Use `sorry` only as a last resort for genuinely hard steps
5. Ensure all imports are correct

Return ONLY the corrected Lean 4 code:
```lean
-- corrected version
...
```"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You are a Lean 4 proof repair expert. Fix compiler errors precisely.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            match = re.search(r"```lean\s*\n?(.*?)\n?```", text, re.DOTALL)
            return match.group(1).strip() if match else ""

        except Exception as e:
            logger.debug(f"[Formalization] Repair failed: {e}")
            return ""

    async def _verify_lean(self, lean_code: str) -> dict[str, Any]:
        """Verify Lean 4 code using LeanEnvironment."""
        try:
            from autoforge.engine.provers.lean_core import LeanEnvironment

            import tempfile
            lean_env = LeanEnvironment()

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lean", delete=False,
            ) as f:
                f.write(lean_code)
                lean_file = Path(f.name)

            result = await lean_env.verify_file(lean_file)

            # Cleanup
            try:
                lean_file.unlink()
            except OSError:
                pass

            return {
                "success": result.success,
                "errors": result.errors,
                "warnings": result.warnings,
                "sorry_count": result.sorry_count,
                "execution_time": result.execution_time,
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "warnings": [],
                "sorry_count": 0,
                "execution_time": 0.0,
            }


# ══════════════════════════════════════════════════════════════
# Main Verification Engine
# ══════════════════════════════════════════════════════════════


class ArticleVerifier:
    """Main article verification engine.

    Orchestrates the full pipeline:
      1. Parse article → extract claims
      2. Topologically sort claims by dependencies
      3. Formalize each claim (in dependency order)
      4. Verify with Lean 4
      5. Optionally cross-verify with multiple provers
      6. Check dependency graph consistency
      7. Generate comprehensive report
    """

    def __init__(self) -> None:
        self._parser = ArticleParser()
        self._pipeline = FormalizationPipeline()

    async def verify_article(
        self,
        text: str,
        llm: Any,
        *,
        title: str = "Untitled",
        cross_verify: bool = False,
        max_claims: int = 50,
    ) -> ArticleVerificationReport:
        """Verify all mathematical claims in an article.

        Args:
            text: Article text (LaTeX, Markdown, or plain text)
            llm: LLM router
            title: Article title
            cross_verify: Whether to cross-verify with multiple provers
            max_claims: Maximum claims to process

        Returns:
            Comprehensive ArticleVerificationReport
        """
        start_time = time.monotonic()
        report = ArticleVerificationReport(title=title)

        # 1. Parse claims
        claims = await self._parser.parse(text, llm, title=title)
        claims = claims[:max_claims]
        report.total_claims = len(claims)

        if not claims:
            report.assessment = "No mathematical claims found in article."
            return report

        # 2. Topological sort by dependencies
        claims = self._topological_sort(claims)

        # 3. Formalize and verify each claim
        formalization_start = time.monotonic()
        claim_map: dict[str, VerifiableClaim] = {}

        for claim in claims:
            # Get context from verified dependencies
            context_claims = [
                claim_map[dep_id]
                for dep_id in claim.dependencies
                if dep_id in claim_map
            ]

            # Formalize
            claim = await self._pipeline.formalize_claim(
                claim, llm, context_claims=context_claims,
            )

            if claim.lean_formalization:
                report.formalized += 1

            # Cross-verify if requested
            if cross_verify and claim.verification_status in (
                VerificationStatus.VERIFIED,
                VerificationStatus.VERIFIED_WITH_SORRY,
            ):
                claim = await self._pipeline.cross_verify_claim(claim, llm)

            claim_map[claim.id] = claim

        report.formalization_time = time.monotonic() - formalization_start

        # 4. Check dependency consistency
        dep_issues = self._check_dependency_consistency(claims)
        report.dependency_consistent = len(dep_issues) == 0
        report.dependency_issues = dep_issues

        # 5. Compute statistics
        report.claims = claims
        for claim in claims:
            if claim.verification_status == VerificationStatus.VERIFIED:
                report.verified += 1
            elif claim.verification_status == VerificationStatus.VERIFIED_WITH_SORRY:
                report.verified_with_sorry += 1
            elif claim.verification_status == VerificationStatus.FAILED:
                report.failed += 1
            elif claim.verification_status == VerificationStatus.SKIPPED:
                report.skipped += 1
            elif claim.verification_status == VerificationStatus.CROSS_VERIFIED:
                report.cross_verified += 1

        # 6. Overall assessment
        report.total_time = time.monotonic() - start_time
        report.overall_confidence = self._compute_confidence(claims)
        report.assessment = self._generate_assessment(report)

        return report

    async def verify_single_claim(
        self,
        statement: str,
        llm: Any,
        *,
        proof: str = "",
        claim_type: str = "theorem",
        cross_verify: bool = False,
    ) -> VerifiableClaim:
        """Verify a single mathematical claim.

        Convenience method for verifying individual claims without
        parsing a full article.
        """
        try:
            ctype = ClaimType(claim_type)
        except ValueError:
            ctype = ClaimType.THEOREM

        claim = VerifiableClaim(
            id=hashlib.sha256(statement[:100].encode()).hexdigest()[:12],
            claim_type=ctype,
            statement=statement,
            proof_text=proof,
        )

        claim = await self._pipeline.formalize_claim(claim, llm)

        if cross_verify:
            claim = await self._pipeline.cross_verify_claim(claim, llm)

        return claim

    def _topological_sort(self, claims: list[VerifiableClaim]) -> list[VerifiableClaim]:
        """Topologically sort claims by dependency order."""
        claim_map = {c.id: c for c in claims}
        visited: set[str] = set()
        result: list[VerifiableClaim] = []

        def visit(cid: str) -> None:
            if cid in visited or cid not in claim_map:
                return
            visited.add(cid)
            claim = claim_map[cid]
            for dep in claim.dependencies:
                visit(dep)
            result.append(claim)

        for claim in claims:
            visit(claim.id)

        return result

    def _check_dependency_consistency(
        self,
        claims: list[VerifiableClaim],
    ) -> list[str]:
        """Check that the dependency graph is consistent.

        A dependency is inconsistent if:
          - A verified claim depends on a failed claim
          - There are circular dependencies
          - A claim references a non-existent dependency
        """
        issues: list[str] = []
        claim_map = {c.id: c for c in claims}

        for claim in claims:
            for dep_id in claim.dependencies:
                if dep_id not in claim_map:
                    issues.append(
                        f"{claim.label}: depends on unknown claim {dep_id}"
                    )
                    continue

                dep = claim_map[dep_id]
                if dep.verification_status == VerificationStatus.FAILED:
                    if claim.verification_status in (
                        VerificationStatus.VERIFIED,
                        VerificationStatus.CROSS_VERIFIED,
                    ):
                        issues.append(
                            f"{claim.label}: verified but depends on failed "
                            f"{dep.label}"
                        )

        # Check for cycles
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(cid: str) -> bool:
            visited.add(cid)
            rec_stack.add(cid)
            if cid in claim_map:
                for dep in claim_map[cid].dependencies:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        issues.append(f"Circular dependency detected involving {cid}")
                        return True
            rec_stack.discard(cid)
            return False

        for claim in claims:
            if claim.id not in visited:
                has_cycle(claim.id)

        return issues

    @staticmethod
    def _compute_confidence(claims: list[VerifiableClaim]) -> float:
        """Compute overall verification confidence."""
        if not claims:
            return 0.0

        scorable = [c for c in claims if c.verification_status != VerificationStatus.SKIPPED]
        if not scorable:
            return 0.0

        total = 0.0
        for c in scorable:
            if c.verification_status in (
                VerificationStatus.VERIFIED,
                VerificationStatus.CROSS_VERIFIED,
            ):
                total += 1.0
            elif c.verification_status == VerificationStatus.VERIFIED_WITH_SORRY:
                total += 0.5
            elif c.verification_status == VerificationStatus.FORMALIZED:
                total += 0.2

        return total / len(scorable)

    @staticmethod
    def _generate_assessment(report: ArticleVerificationReport) -> str:
        """Generate overall assessment text."""
        conf = report.overall_confidence

        if conf >= 0.9:
            return ("Excellent: Nearly all claims formally verified. "
                    "High confidence in mathematical correctness.")
        elif conf >= 0.7:
            return ("Good: Majority of claims verified, some with sorry placeholders. "
                    "Overall structure appears sound.")
        elif conf >= 0.5:
            return ("Mixed: Some claims verified, but significant portions remain "
                    "unverified or have sorry placeholders.")
        elif conf >= 0.3:
            return ("Weak: Most claims could not be fully verified. "
                    "Consider reviewing the proofs manually.")
        else:
            return ("Poor: Very few claims could be verified. The article may "
                    "contain errors or use techniques beyond current formalization capability.")
