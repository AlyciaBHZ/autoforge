"""Article Reasoning Orchestrator — End-to-End Academic Pipeline.

This module unifies all existing reasoning, verification, discovery, and formalization
components into a single end-to-end pipeline for comprehensive article analysis.

Design philosophy:
  1. **Single entry point**: One orchestrator handles the entire flow
  2. **Phase-based pipeline**: Preprocess → Parse → Verify → Reconcile → Formalize → Discover → Extend → Generate
  3. **Graceful degradation**: Each phase can be disabled; pipeline continues if optional phases fail
  4. **Comprehensive reporting**: Full JSON/Markdown summary with timing, confidence, and artifacts
  5. **Output artifacts**: Generated Lean 4 files, markdown summary, JSON report, discovery results

Architecture:

    ArticleInput (raw text + metadata)
        ↓ Preprocess
    normalized text
        ↓ Parse (ArticleParser from article_verifier)
    TheoryGraph
        ↓ Verify (ArticleVerifier from article_verifier)
    ArticleVerificationReport
        ↓ Reconcile (TheoryGraphReconciler)
    TheoryGraph (verified & reconciled)
        ↓ Formalize (PaperFormalizer from paper_formalizer)
    FormalizationReport + Lean files
        ↓ Discover (DiscoveryOrchestrator from autonomous_discovery)
    DiscoveryResult
        ↓ Extend (ReasoningExtensionEngine from reasoning_extension)
    ReasoningRound (new conclusions)
        ↓ Generate (ArticleGenerator from theoretical_reasoning)
    output_article (LaTeX/Markdown/JSON)
        ↓ Aggregate
    ArticleReasoningResult (complete report)

References:
  - article_verifier: ArticleVerifier, ArticleVerificationReport, VerifiableClaim
  - theoretical_reasoning: TheoryGraph, ArticleParser, ArticleGenerator
  - paper_formalizer: PaperFormalizer, FormalizationReport
  - autonomous_discovery: DiscoveryOrchestrator, DiscoveryResult
  - reasoning_extension: ReasoningExtensionEngine, MinimalKernel, NumberedConclusion
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Configuration & Data Structures
# ══════════════════════════════════════════════════════════════


class TaskComplexity(str, Enum):
    """Task complexity levels (from config.py)."""
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class ArticleReasoningConfig:
    """Configuration for the article reasoning pipeline."""
    # Phase toggles
    verify_claims: bool = True
    discover_new: bool = True
    formalize_lean: bool = True
    cross_verify: bool = False
    extend_reasoning: bool = True
    generate_output: bool = True

    # Discovery parameters
    max_discovery_rounds: int = 10
    min_discovery_depth: str = "MODERATE"  # SHALLOW, MODERATE, DEEP

    # Output format and audience
    output_format: str = "latex"  # latex, markdown, json
    output_audience: str = "research"  # research, survey, pedagogical

    # Advanced options
    lean_compile: bool = False
    run_python_verification: bool = False
    save_intermediate: bool = False
    verbose_logging: bool = True


@dataclass
class ArticleInput:
    """Input article with metadata."""
    text: str
    title: str = ""
    source_format: str = "text"  # text, latex, markdown, pdf_extracted
    domain_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArticleReasoningResult:
    """Complete article reasoning result with all outputs."""
    input_article: ArticleInput
    theory_graph: Any | None = None  # TheoryGraph
    verification_report: Any | None = None  # ArticleVerificationReport
    discovery_results: list[Any] = field(default_factory=list)  # DiscoveryResult list
    formalization_report: Any | None = None  # FormalizationReport
    reasoning_rounds: list[Any] = field(default_factory=list)  # ReasoningRound list
    output_article: str = ""  # Generated article in target format
    output_lean_files: dict[str, str] = field(default_factory=dict)  # {filename: code}
    overall_confidence: float = 0.0
    pipeline_log: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "input": {
                "title": self.input_article.title,
                "source_format": self.input_article.source_format,
                "domain_hint": self.input_article.domain_hint,
                "text_length": len(self.input_article.text),
            },
            "theory_graph": self.theory_graph.to_dict() if self.theory_graph else None,
            "verification_report": self.verification_report.to_dict() if self.verification_report else None,
            "discovery_results": [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in self.discovery_results],
            "formalization_report": self.formalization_report.to_dict() if self.formalization_report else None,
            "reasoning_rounds": [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in self.reasoning_rounds],
            "overall_confidence": self.overall_confidence,
            "timing": self.timing,
            "errors": self.errors,
            "pipeline_log": self.pipeline_log[-20:],  # Last 20 log entries
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def to_summary(self) -> str:
        """Generate markdown summary."""
        lines = [
            "# Article Reasoning Report",
            "",
            f"**Title**: {self.input_article.title or 'Untitled'}",
            f"**Format**: {self.input_article.source_format}",
            f"**Overall Confidence**: {self.overall_confidence:.1%}",
            "",
        ]

        if self.timing:
            lines.extend(["## Timing", ""])
            for phase, duration in sorted(self.timing.items()):
                lines.append(f"- **{phase}**: {duration:.2f}s")
            lines.append("")

        if self.verification_report:
            lines.extend([
                "## Verification Results",
                "",
                f"- **Total Claims**: {self.verification_report.total_claims}",
                f"- **Verified**: {self.verification_report.verified}",
                f"- **Verified with sorry**: {self.verification_report.verified_with_sorry}",
                f"- **Failed**: {self.verification_report.failed}",
                f"- **Assessment**: {self.verification_report.assessment}",
                "",
            ])

        if self.formalization_report:
            lines.extend([
                "## Formalization Results",
                "",
                f"- **Total Statements**: {self.formalization_report.total_statements}",
                f"- **Lean Proved**: {self.formalization_report.lean_proved}",
                f"- **Lean Sorry**: {self.formalization_report.lean_sorry}",
                f"- **Score**: {self.formalization_report.overall_score:.2f}",
                "",
            ])

        if self.discovery_results:
            lines.extend([
                "## Discovery Results",
                "",
                f"- **Rounds**: {len(self.discovery_results)}",
            ])
            for i, result in enumerate(self.discovery_results, 1):
                if hasattr(result, 'conjectures_generated'):
                    lines.append(f"  - Round {i}: {result.conjectures_generated} conjectures")
            lines.append("")

        if self.reasoning_rounds:
            lines.extend([
                "## Reasoning Extension",
                "",
                f"- **Rounds**: {len(self.reasoning_rounds)}",
            ])
            for i, rnd in enumerate(self.reasoning_rounds, 1):
                if hasattr(rnd, 'accepted'):
                    lines.append(f"  - Round {i}: {rnd.accepted} conclusions accepted")
            lines.append("")

        if self.errors:
            lines.extend([
                "## Errors & Warnings",
                "",
            ])
            for err in self.errors[:10]:
                lines.append(f"- {err}")
            lines.append("")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Article Preprocessor
# ══════════════════════════════════════════════════════════════


class ArticlePreprocessor:
    """Normalize article text from various formats."""

    async def preprocess(self, input: ArticleInput) -> str:
        """Normalize text from various source formats.

        Handles:
          - LaTeX: strips preamble, normalizes commands
          - Markdown: basic normalization
          - PDF extracted text: strip extra whitespace, fix hyphenation
          - Plain text: minimal processing
        """
        text = input.text

        if input.source_format == "latex":
            text = self._normalize_latex(text)
        elif input.source_format == "markdown":
            text = self._normalize_markdown(text)
        elif input.source_format == "pdf_extracted":
            text = await self._normalize_pdf_extracted(text)

        # Extract main body (skip preamble/abstract/bibliography for discovery)
        text = self._extract_body(text)

        return text.strip()

    def _normalize_latex(self, text: str) -> str:
        """Normalize LaTeX text.

        Removes:
          - \\documentclass, \\usepackage, preamble
          - Comments
          Preserves: \\begin{theorem}, \\cite, \\ref
        """
        import re

        # Remove preamble (before \begin{document})
        if "\\begin{document}" in text:
            text = text[text.index("\\begin{document}") + len("\\begin{document}"):]

        # Remove \end{document} and beyond
        if "\\end{document}" in text:
            text = text[:text.index("\\end{document}")]

        # Remove comments (% to end of line, except in code blocks)
        lines = []
        for line in text.split("\n"):
            if "%" in line and not line.strip().startswith("%"):
                # Simple heuristic: keep % if it's in an environment
                if "\\begin" not in line and "\\end" not in line:
                    line = line[:line.index("%")]
            lines.append(line)

        text = "\n".join(lines)

        # Normalize common LaTeX commands
        text = text.replace("\\emph{", "_")
        text = text.replace("\\textit{", "_")
        text = text.replace("\\textbf{", "**")
        text = re.sub(r"\}", "", text)  # Close braces (rough)

        return text

    def _normalize_markdown(self, text: str) -> str:
        """Normalize Markdown."""
        # Minimal processing: ensure consistent spacing
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines)

    async def _normalize_pdf_extracted(self, text: str) -> str:
        """Normalize PDF-extracted text.

        Handles:
          - Hyphenation at line boundaries
          - Extra whitespace
          - Mangled unicode
        """
        import re

        # Fix hyphenation: word- followed by newline becomes word
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Collapse excessive whitespace
        text = re.sub(r"\n\n+", "\n\n", text)
        text = re.sub(r"[ ]{2,}", " ", text)

        return text

    def _extract_body(self, text: str) -> str:
        """Extract main article body.

        Skips: abstract, acknowledgments, bibliography, appendix (markers only).
        For discovery phase, we want the core content.
        """
        import re

        # Define section markers
        skip_markers = [
            r"(?i)(acknowledgments?|thanks|references|bibliography|appendix|appendices)",
        ]

        # Find the earliest skip marker
        earliest_skip = len(text)
        for marker in skip_markers:
            match = re.search(marker, text)
            if match:
                earliest_skip = min(earliest_skip, match.start())

        body = text[:earliest_skip]

        # Optionally skip abstract (for some pipelines)
        # Uncomment if needed:
        # abstract_match = re.search(r"(?i)abstract\s*\n(.*?)(?=\n\s*\n)", body)
        # if abstract_match:
        #     body = body[:abstract_match.start()] + body[abstract_match.end():]

        return body


# ══════════════════════════════════════════════════════════════
# Theory Graph Reconciler
# ══════════════════════════════════════════════════════════════


class TheoryGraphReconciler:
    """Reconcile verified claims with TheoryGraph concepts.

    Maps extracted claims to concept nodes and updates verification status.
    """

    async def reconcile(
        self,
        theory_graph: Any,
        claims: list[Any],
        llm: Any,
    ) -> Any:
        """Reconcile claims from ArticleVerifier with TheoryGraph concepts.

        Args:
            theory_graph: TheoryGraph object
            claims: List of VerifiableClaim from ArticleVerifier
            llm: LLM router

        Returns:
            Updated TheoryGraph with verification status added to concepts
        """
        if not theory_graph or not claims:
            return theory_graph

        logger.info(f"[Reconciler] Mapping {len(claims)} claims to {len(theory_graph._nodes)} concepts")

        # Build a mapping of claim to concept
        claim_to_concept: dict[str, str] = {}

        for claim in claims:
            concept_id = await self._match_claim_to_concept(
                claim, theory_graph._nodes, llm
            )
            if concept_id:
                claim_to_concept[claim.id] = concept_id

        # Update verification status in theory graph
        for claim_id, concept_id in claim_to_concept.items():
            claim = next((c for c in claims if c.id == claim_id), None)
            if claim and concept_id in theory_graph._nodes:
                concept = theory_graph._nodes[concept_id]
                # Add verification metadata
                if not hasattr(concept, 'verification_status'):
                    concept.verification_status = claim.verification_status.value
                if not hasattr(concept, 'verification_confidence'):
                    concept.verification_confidence = claim.confidence

        logger.info(f"[Reconciler] Reconciled {len(claim_to_concept)} claims with concepts")
        return theory_graph

    async def _match_claim_to_concept(
        self,
        claim: Any,
        concepts: dict[str, Any],
        llm: Any,
    ) -> str | None:
        """Match a claim to the closest concept node.

        Uses LLM to find semantic overlap.
        """
        # Heuristic: find concepts with similar label or statement
        best_concept = None
        best_score = 0.0

        for concept_id, concept in concepts.items():
            score = 0.0

            # Simple text similarity
            concept_text = (concept.informal_statement or concept.formal_statement or "").lower()
            claim_text = claim.statement.lower()

            # Check for keyword overlap
            concept_words = set(concept_text.split())
            claim_words = set(claim_text.split())
            overlap = len(concept_words & claim_words)
            if overlap > 0:
                score = overlap / max(len(concept_words), len(claim_words), 1)

            if score > best_score:
                best_score = score
                best_concept = concept_id

        return best_concept if best_score > 0.1 else None


# ══════════════════════════════════════════════════════════════
# Main Orchestrator
# ══════════════════════════════════════════════════════════════


class ArticleReasoningOrchestrator:
    """End-to-end article reasoning pipeline.

    Orchestrates all phases:
      1. Preprocess: normalize text from various formats
      2. Parse: extract theory graph using ArticleParser
      3. Verify: verify claims using ArticleVerifier
      4. Reconcile: map claims to theory graph concepts
      5. Formalize: generate Lean 4 code using PaperFormalizer
      6. Discover: find new results using DiscoveryOrchestrator
      7. Extend: grow reasoning using ReasoningExtensionEngine
      8. Generate: create output article using ArticleGenerator
    """

    def __init__(self, config: ArticleReasoningConfig | None = None) -> None:
        """Initialize orchestrator.

        Args:
            config: ArticleReasoningConfig with phase toggles and parameters
        """
        self.config = config or ArticleReasoningConfig()
        self._preprocessor = ArticlePreprocessor()
        self._reconciler = TheoryGraphReconciler()

    async def reason(
        self,
        article: ArticleInput,
        llm: Any,
        output_dir: Path | None = None,
    ) -> ArticleReasoningResult:
        """Full end-to-end pipeline.

        Args:
            article: ArticleInput with text and metadata
            llm: LLM router
            output_dir: Optional output directory for artifacts

        Returns:
            ArticleReasoningResult with all outputs and metadata
        """
        result = ArticleReasoningResult(input_article=article)
        start_time = time.monotonic()

        try:
            # Phase 1: Preprocess
            result.pipeline_log.append("Phase 1: Preprocessing")
            phase_start = time.monotonic()
            normalized_text = await self._preprocess_phase(article)
            result.timing["preprocess"] = time.monotonic() - phase_start

            # Phase 2: Parse to TheoryGraph
            result.pipeline_log.append("Phase 2: Parsing to TheoryGraph")
            phase_start = time.monotonic()
            theory_graph = await self._parse_phase(normalized_text, article, llm)
            result.theory_graph = theory_graph
            result.timing["parse"] = time.monotonic() - phase_start

            # Phase 3: Verify Claims
            if self.config.verify_claims:
                result.pipeline_log.append("Phase 3: Verifying Claims")
                phase_start = time.monotonic()
                verification_report = await self._verify_phase(
                    normalized_text, article, llm
                )
                result.verification_report = verification_report
                result.timing["verify"] = time.monotonic() - phase_start

            # Phase 4: Reconcile with TheoryGraph
            if self.config.verify_claims and result.theory_graph and result.verification_report:
                result.pipeline_log.append("Phase 4: Reconciling Claims with TheoryGraph")
                phase_start = time.monotonic()
                result.theory_graph = await self._reconcile_phase(
                    result.theory_graph,
                    result.verification_report.claims,
                    llm,
                )
                result.timing["reconcile"] = time.monotonic() - phase_start

            # Phase 5: Formalize to Lean 4
            if self.config.formalize_lean and result.theory_graph:
                result.pipeline_log.append("Phase 5: Formalizing to Lean 4")
                phase_start = time.monotonic()
                formalization_report, lean_files = await self._formalize_phase(
                    result.theory_graph, llm
                )
                result.formalization_report = formalization_report
                result.output_lean_files = lean_files
                result.timing["formalize"] = time.monotonic() - phase_start

            # Phase 6: Autonomous Discovery
            if self.config.discover_new and result.theory_graph:
                result.pipeline_log.append("Phase 6: Autonomous Discovery")
                phase_start = time.monotonic()
                discovery_results = await self._discover_phase(
                    result.theory_graph, llm
                )
                result.discovery_results = discovery_results
                result.timing["discover"] = time.monotonic() - phase_start

            # Phase 7: Reasoning Extension
            if self.config.extend_reasoning and result.theory_graph:
                result.pipeline_log.append("Phase 7: Reasoning Extension")
                phase_start = time.monotonic()
                reasoning_rounds = await self._extend_phase(
                    result.theory_graph, llm
                )
                result.reasoning_rounds = reasoning_rounds
                result.timing["extend"] = time.monotonic() - phase_start

            # Phase 8: Generate Output Article
            if self.config.generate_output and result.theory_graph:
                result.pipeline_log.append("Phase 8: Generating Output Article")
                phase_start = time.monotonic()
                output_article = await self._generate_phase(
                    result.theory_graph,
                    article,
                    result.discovery_results,
                    result.reasoning_rounds,
                    llm,
                )
                result.output_article = output_article
                result.timing["generate"] = time.monotonic() - phase_start

            # Phase 9: Compute overall confidence
            result.overall_confidence = self._compute_confidence(result)
            result.timing["total"] = time.monotonic() - start_time

            result.pipeline_log.append(
                f"Pipeline complete in {result.timing['total']:.2f}s"
            )

            # Save artifacts if output_dir provided
            if output_dir:
                await self._save_outputs(result, output_dir)

            return result

        except Exception as e:
            logger.exception(f"[Orchestrator] Pipeline failed: {e}")
            result.errors.append(f"Pipeline failure: {str(e)}")
            result.timing["total"] = time.monotonic() - start_time
            return result

    async def quick_reason(
        self,
        article_text: str,
        llm: Any,
    ) -> ArticleReasoningResult:
        """Lightweight pipeline: parse + verify + discover, no formalization."""
        config = ArticleReasoningConfig(
            formalize_lean=False,
            extend_reasoning=False,
            generate_output=False,
        )
        orchestrator = ArticleReasoningOrchestrator(config)
        article = ArticleInput(text=article_text, title="Untitled")
        return await orchestrator.reason(article, llm)

    async def verify_only(
        self,
        article_text: str,
        llm: Any,
    ) -> ArticleReasoningResult:
        """Just parse and verify claims."""
        config = ArticleReasoningConfig(
            discover_new=False,
            formalize_lean=False,
            extend_reasoning=False,
            generate_output=False,
        )
        orchestrator = ArticleReasoningOrchestrator(config)
        article = ArticleInput(text=article_text, title="Untitled")
        return await orchestrator.reason(article, llm)

    async def discover_only(
        self,
        article_text: str,
        llm: Any,
        theory_graph: Any = None,
    ) -> ArticleReasoningResult:
        """Parse + discover new results."""
        config = ArticleReasoningConfig(
            verify_claims=False,
            formalize_lean=False,
            extend_reasoning=False,
            generate_output=False,
        )
        orchestrator = ArticleReasoningOrchestrator(config)
        article = ArticleInput(text=article_text, title="Untitled")
        result = await orchestrator.reason(article, llm)

        if theory_graph:
            result.theory_graph = theory_graph

        return result

    # ──────────────────────────────────────────────────────────────
    # Phase implementations
    # ──────────────────────────────────────────────────────────────

    async def _preprocess_phase(self, article: ArticleInput) -> str:
        """Preprocess article text."""
        try:
            text = await self._preprocessor.preprocess(article)
            logger.info(f"[Preprocess] Normalized {len(text)} chars")
            return text
        except Exception as e:
            logger.warning(f"[Preprocess] Failed: {e}")
            return article.text

    async def _parse_phase(
        self,
        text: str,
        article: ArticleInput,
        llm: Any,
    ) -> Any:
        """Parse to TheoryGraph using ArticleParser."""
        try:
            from autoforge.engine.theoretical_reasoning import ArticleParser

            parser = ArticleParser()
            theory_graph = await parser.parse_to_graph(
                text,
                llm,
                title=article.title,
                domain_hint=article.domain_hint,
            )
            logger.info(f"[Parse] Created TheoryGraph with {len(theory_graph._nodes)} concepts")
            return theory_graph

        except Exception as e:
            logger.warning(f"[Parse] Failed: {e}")
            return None

    async def _verify_phase(
        self,
        text: str,
        article: ArticleInput,
        llm: Any,
    ) -> Any:
        """Verify claims using ArticleVerifier."""
        try:
            from autoforge.engine.article_verifier import ArticleVerifier

            verifier = ArticleVerifier()
            report = await verifier.verify_article(
                text,
                llm,
                title=article.title,
                cross_verify=self.config.cross_verify,
            )
            logger.info(
                f"[Verify] Verified {report.verified} / {report.total_claims} claims"
            )
            return report

        except Exception as e:
            logger.warning(f"[Verify] Failed: {e}")
            return None

    async def _reconcile_phase(
        self,
        theory_graph: Any,
        claims: list[Any],
        llm: Any,
    ) -> Any:
        """Reconcile claims with TheoryGraph."""
        try:
            reconciled = await self._reconciler.reconcile(theory_graph, claims, llm)
            logger.info(f"[Reconcile] Updated TheoryGraph with verification data")
            return reconciled

        except Exception as e:
            logger.warning(f"[Reconcile] Failed: {e}")
            return theory_graph

    async def _formalize_phase(
        self,
        theory_graph: Any,
        llm: Any,
    ) -> tuple[Any, dict[str, str]]:
        """Formalize to Lean 4 using PaperFormalizer."""
        try:
            from autoforge.engine.paper_formalizer import PaperFormalizer

            formalizer = PaperFormalizer()
            report = await formalizer.formalize(
                theory_graph,
                llm,
                lean_compile=self.config.lean_compile,
                run_python=self.config.run_python_verification,
            )

            # Extract Lean files
            lean_files = {unit.label: unit.lean_code for unit in report.units if unit.lean_code}
            logger.info(f"[Formalize] Generated {len(lean_files)} Lean files")
            return report, lean_files

        except Exception as e:
            logger.warning(f"[Formalize] Failed: {e}")
            return None, {}

    async def _discover_phase(
        self,
        theory_graph: Any,
        llm: Any,
    ) -> list[Any]:
        """Autonomous discovery using DiscoveryOrchestrator."""
        try:
            from autoforge.engine.autonomous_discovery import DiscoveryOrchestrator

            orchestrator = DiscoveryOrchestrator()
            results = []

            for round_num in range(1, self.config.max_discovery_rounds + 1):
                result = await orchestrator.discover(
                    theory_graph,
                    llm,
                    max_depth=self.config.min_discovery_depth,
                )
                results.append(result)

                # Check termination
                if hasattr(result, 'conjectures_generated') and result.conjectures_generated == 0:
                    logger.info(f"[Discover] Terminating at round {round_num} (no new conjectures)")
                    break

            logger.info(f"[Discover] Completed {len(results)} discovery rounds")
            return results

        except Exception as e:
            logger.warning(f"[Discover] Failed: {e}")
            return []

    async def _extend_phase(
        self,
        theory_graph: Any,
        llm: Any,
    ) -> list[Any]:
        """Reasoning extension using ReasoningExtensionEngine."""
        try:
            from autoforge.engine.reasoning_extension import ReasoningExtensionEngine

            engine = ReasoningExtensionEngine()
            rounds = await engine.extend(
                theory_graph,
                llm,
                max_rounds=self.config.max_discovery_rounds,
            )

            logger.info(f"[Extend] Generated {len(rounds)} reasoning rounds")
            return rounds

        except Exception as e:
            logger.warning(f"[Extend] Failed: {e}")
            return []

    async def _generate_phase(
        self,
        theory_graph: Any,
        article: ArticleInput,
        discovery_results: list[Any],
        reasoning_rounds: list[Any],
        llm: Any,
    ) -> str:
        """Generate output article using ArticleGenerator."""
        try:
            from autoforge.engine.theoretical_reasoning import ArticleGenerator

            generator = ArticleGenerator()
            output = await generator.generate(
                theory_graph,
                llm,
                title=article.title,
                format=self.config.output_format,
                audience=self.config.output_audience,
                additional_results=discovery_results + reasoning_rounds,
            )

            logger.info(f"[Generate] Generated {len(output)} char output article")
            return output

        except Exception as e:
            logger.warning(f"[Generate] Failed: {e}")
            return ""

    # ──────────────────────────────────────────────────────────────
    # Utility methods
    # ──────────────────────────────────────────────────────────────

    def _compute_confidence(self, result: ArticleReasoningResult) -> float:
        """Compute overall confidence from sub-results."""
        components = []

        if result.verification_report:
            components.append(result.verification_report.overall_confidence)

        if result.formalization_report:
            components.append(result.formalization_report.overall_score)

        if result.reasoning_rounds:
            # Average rigor of reasoning rounds
            rigor_scores = []
            for rnd in result.reasoning_rounds:
                if hasattr(rnd, 'conclusions'):
                    for conc in rnd.conclusions:
                        if hasattr(conc, 'rigor_score'):
                            rigor_scores.append(conc.rigor_score)
            if rigor_scores:
                components.append(sum(rigor_scores) / len(rigor_scores))

        if not components:
            return 0.0

        return sum(components) / len(components)

    async def _save_outputs(
        self,
        result: ArticleReasoningResult,
        output_dir: Path,
    ) -> None:
        """Save all outputs to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON report
        json_path = output_dir / "reasoning_report.json"
        json_path.write_text(result.to_json(), encoding="utf-8")
        logger.info(f"[Save] Wrote {json_path}")

        # Markdown summary
        summary_path = output_dir / "summary.md"
        summary_path.write_text(result.to_summary(), encoding="utf-8")
        logger.info(f"[Save] Wrote {summary_path}")

        # Output article
        if result.output_article:
            article_ext = {
                "latex": ".tex",
                "markdown": ".md",
                "json": ".json",
            }.get(self.config.output_format, ".txt")
            article_path = output_dir / f"output_article{article_ext}"
            article_path.write_text(result.output_article, encoding="utf-8")
            logger.info(f"[Save] Wrote {article_path}")

        # Lean files
        if result.output_lean_files:
            lean_dir = output_dir / "lean"
            lean_dir.mkdir(exist_ok=True)
            for filename, code in result.output_lean_files.items():
                safe_name = filename.replace(" ", "_").replace("/", "_")
                lean_path = lean_dir / f"{safe_name}.lean"
                lean_path.write_text(code, encoding="utf-8")
            logger.info(f"[Save] Wrote {len(result.output_lean_files)} Lean files to {lean_dir}")


# ══════════════════════════════════════════════════════════════
# Convenience functions
# ══════════════════════════════════════════════════════════════


async def reason_about_article(
    text: str,
    llm: Any,
    title: str = "",
    output_dir: Path | None = None,
    **kwargs: Any,
) -> ArticleReasoningResult:
    """Convenience function: reason about an article with default config.

    Args:
        text: Article text
        llm: LLM router
        title: Article title
        output_dir: Optional output directory
        **kwargs: Additional config options (verify_claims, discover_new, etc.)

    Returns:
        ArticleReasoningResult
    """
    config_dict = {
        "verify_claims": kwargs.get("verify_claims", True),
        "discover_new": kwargs.get("discover_new", True),
        "formalize_lean": kwargs.get("formalize_lean", True),
        "cross_verify": kwargs.get("cross_verify", False),
        "extend_reasoning": kwargs.get("extend_reasoning", True),
        "generate_output": kwargs.get("generate_output", True),
        "max_discovery_rounds": kwargs.get("max_discovery_rounds", 10),
    }

    config = ArticleReasoningConfig(**config_dict)
    orchestrator = ArticleReasoningOrchestrator(config)
    article = ArticleInput(
        text=text,
        title=title,
        source_format=kwargs.get("source_format", "text"),
    )

    return await orchestrator.reason(article, llm, output_dir=output_dir)


async def reason_about_pdf(
    pdf_path: Path,
    llm: Any,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> ArticleReasoningResult:
    """Convenience function: reason about a PDF file.

    Attempts to extract text from PDF using pdfplumber or pypdf.

    Args:
        pdf_path: Path to PDF file
        llm: LLM router
        output_dir: Optional output directory
        **kwargs: Additional config options

    Returns:
        ArticleReasoningResult
    """
    try:
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            try:
                from pypdf import PdfReader

                reader = PdfReader(pdf_path)
                text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except ImportError:
                logger.warning("[PDF] Neither pdfplumber nor pypdf available, using raw text")
                text = pdf_path.read_text(encoding="utf-8", errors="ignore")

        title = pdf_path.stem
        return await reason_about_article(
            text,
            llm,
            title=title,
            source_format="pdf_extracted",
            output_dir=output_dir,
            **kwargs,
        )

    except Exception as e:
        logger.error(f"[PDF] Failed to extract text from {pdf_path}: {e}")
        raise
