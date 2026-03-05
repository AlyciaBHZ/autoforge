# Article Reasoning Orchestrator — Usage Guide

## Overview

The `ArticleReasoningOrchestrator` (in `autoforge/engine/article_reasoning.py`) provides a unified, end-to-end pipeline for comprehensive academic article analysis. It orchestrates eight phases of reasoning, verification, discovery, and formalization:

1. **Preprocess**: Normalize text from various formats (LaTeX, Markdown, PDF)
2. **Parse**: Extract theory graph using `ArticleParser`
3. **Verify**: Verify mathematical claims using `ArticleVerifier`
4. **Reconcile**: Map verified claims to theory graph concepts
5. **Formalize**: Generate Lean 4 code using `PaperFormalizer`
6. **Discover**: Find new results using `DiscoveryOrchestrator`
7. **Extend**: Grow reasoning using `ReasoningExtensionEngine`
8. **Generate**: Create output article using `ArticleGenerator`

Each phase is wrapped in try/except, logs activity, and can be disabled via config. The pipeline gracefully degrades: if a phase fails, the pipeline continues with available results.

## Quick Start

### Basic Usage

```python
from autoforge.engine.article_reasoning import (
    ArticleReasoningOrchestrator,
    ArticleInput,
)
from autoforge.engine.llm_router import LLMRouter

# Initialize
llm = LLMRouter()  # Your LLM router
orchestrator = ArticleReasoningOrchestrator()

# Create input
article = ArticleInput(
    text="Your article text here...",
    title="Example Paper",
    source_format="text",  # or "latex", "markdown", "pdf_extracted"
)

# Run full pipeline
result = await orchestrator.reason(article, llm)

# Access results
print(f"Confidence: {result.overall_confidence:.1%}")
print(f"Verified: {result.verification_report.verified} claims")
print(f"Output: {result.output_article[:500]}")
```

### Convenience Functions

```python
# Option 1: Simple text reasoning
result = await reason_about_article(
    text="Article text...",
    llm=llm,
    title="Paper Title",
    discover_new=True,
    formalize_lean=True,
)

# Option 2: PDF file reasoning
from pathlib import Path
result = await reason_about_pdf(
    pdf_path=Path("paper.pdf"),
    llm=llm,
    output_dir=Path("output/"),
)
```

### Lightweight Variants

```python
# Quick verification + discovery (no formalization)
result = await orchestrator.quick_reason(article_text, llm)

# Just verify claims
result = await orchestrator.verify_only(article_text, llm)

# Just discover new results
result = await orchestrator.discover_only(article_text, llm)
```

## Configuration

### Full Configuration Example

```python
from autoforge.engine.article_reasoning import ArticleReasoningConfig

config = ArticleReasoningConfig(
    # Phase toggles
    verify_claims=True,
    discover_new=True,
    formalize_lean=True,
    cross_verify=False,
    extend_reasoning=True,
    generate_output=True,

    # Discovery parameters
    max_discovery_rounds=10,
    min_discovery_depth="MODERATE",  # SHALLOW, MODERATE, or DEEP

    # Output format and audience
    output_format="latex",  # latex, markdown, json
    output_audience="research",  # research, survey, pedagogical

    # Advanced options
    lean_compile=False,
    run_python_verification=False,
    save_intermediate=False,
    verbose_logging=True,
)

orchestrator = ArticleReasoningOrchestrator(config)
result = await orchestrator.reason(article, llm)
```

## Output

### ArticleReasoningResult Structure

```python
result.input_article           # Original ArticleInput
result.theory_graph            # Extracted TheoryGraph
result.verification_report     # ArticleVerificationReport (claims verified)
result.discovery_results       # list[DiscoveryResult] from autonomous discovery
result.formalization_report    # FormalizationReport (Lean 4 formalizations)
result.reasoning_rounds        # list[ReasoningRound] from extension engine
result.output_article          # Generated article (LaTeX/Markdown/JSON)
result.output_lean_files       # dict[filename: str, code: str]
result.overall_confidence      # 0-1 aggregate confidence
result.timing                  # dict[phase: float] execution times
result.pipeline_log            # list[str] phase summaries
result.errors                  # list[str] errors encountered
```

### Result Methods

```python
# Serialize to JSON (with all metadata)
json_str = result.to_json()

# Generate markdown summary
summary_md = result.to_summary()

# Save all artifacts to disk
import asyncio
await orchestrator._save_outputs(result, Path("output/"))
# Generates:
#   - reasoning_report.json
#   - summary.md
#   - output_article.{tex,md,json}
#   - lean/*.lean (Lean 4 files)
```

## Pipeline Phases

### Phase 1: Preprocess

Normalizes text from various formats:

- **LaTeX**: Removes preamble, comments, normalizes commands
- **Markdown**: Basic whitespace normalization
- **PDF extracted**: Fixes hyphenation, collapses whitespace
- **Plain text**: Minimal processing
- Extracts main body (skips abstract, bibliography for discovery)

```python
preprocessor = ArticlePreprocessor()
normalized = await preprocessor.preprocess(article)
```

### Phase 2: Parse

Extracts a `TheoryGraph` using `ArticleParser` from `theoretical_reasoning.py`:

- Parses definitions, theorems, lemmas, propositions
- Builds concept hierarchy
- Establishes cross-domain connections

### Phase 3: Verify (Optional)

Verifies mathematical claims using `ArticleVerifier` from `article_verifier.py`:

- Extracts claims (theorems, lemmas, propositions)
- Auto-formalizes to Lean 4
- Compiles and checks for errors
- Iteratively repairs failed formalizations
- Optionally cross-verifies with multiple provers

Produces `ArticleVerificationReport` with per-claim status.

### Phase 4: Reconcile (Optional)

Maps verified claims to theory graph concepts:

- Uses semantic similarity to match claims with concepts
- Updates concepts with verification status and confidence
- Enriches theory graph with verification metadata

### Phase 5: Formalize (Optional)

Generates Lean 4 code using `PaperFormalizer` from `paper_formalizer.py`:

- Extracts formalizable statements (definitions, theorems, propositions)
- Generates Lean 4 code for each
- Optionally compiles and produces Python verification scripts
- Produces `FormalizationReport` with per-statement status

### Phase 6: Discover (Optional)

Finds new results using `DiscoveryOrchestrator` from `autonomous_discovery.py`:

- Generates conjectures from frontier nodes
- Evaluates novelty and depth
- Iterates up to `max_discovery_rounds`
- Stops early if no new deep results found

### Phase 7: Extend (Optional)

Grows reasoning using `ReasoningExtensionEngine` from `reasoning_extension.py`:

- Starts from minimal kernel (axioms + key theorems)
- Applies growth operators (lift, fold, specialize, dualize, etc.)
- Generates numbered conclusions with proof sketches
- Evaluates publication worthiness
- Produces `ReasoningRound` with accepted conclusions

### Phase 8: Generate (Optional)

Creates output article using `ArticleGenerator` from `theoretical_reasoning.py`:

- Combines original theory graph with discovery/extension results
- Formats for specified audience (research, survey, pedagogical)
- Outputs in specified format (LaTeX, Markdown, JSON)

## Error Handling & Logging

### Graceful Degradation

Each phase is wrapped in try/except:

```python
try:
    # Phase implementation
except Exception as e:
    logger.warning(f"[Phase] Failed: {e}")
    # Continue pipeline with available results
```

If a phase fails, the pipeline:
- Logs the warning
- Continues to next phase
- Records error in `result.errors`
- Still produces output from successful phases

### Logging

All phases log activity:

```python
logger.info(f"[Preprocess] Normalized {len(text)} chars")
logger.info(f"[Parse] Created TheoryGraph with {len(theory_graph._nodes)} concepts")
logger.info(f"[Verify] Verified {report.verified} / {report.total_claims} claims")
logger.info(f"[Formalize] Generated {len(lean_files)} Lean files")
```

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom Phase Configuration

Disable expensive phases to speed up analysis:

```python
config = ArticleReasoningConfig(
    verify_claims=False,        # Skip verification
    formalize_lean=False,       # Skip Lean formalization
    extend_reasoning=False,     # Skip reasoning extension
    discover_new=False,         # Skip discovery
)

orchestrator = ArticleReasoningOrchestrator(config)
result = await orchestrator.reason(article, llm)
```

### Discovery with Custom Depth

```python
config = ArticleReasoningConfig(
    discover_new=True,
    max_discovery_rounds=20,
    min_discovery_depth="DEEP",
)
```

### Cross-Verification

Enable multi-prover verification for high-confidence results:

```python
config = ArticleReasoningConfig(
    verify_claims=True,
    cross_verify=True,  # Verify with multiple backends (Lean, Coq, Z3, etc.)
)
```

### Output to Specific Format

```python
config = ArticleReasoningConfig(
    output_format="markdown",        # or "latex", "json"
    output_audience="pedagogical",   # or "research", "survey"
)
```

### Save All Artifacts

```python
from pathlib import Path

result = await orchestrator.reason(
    article,
    llm,
    output_dir=Path("analysis_output/")
)

# Automatically saves:
# - reasoning_report.json
# - summary.md
# - output_article.md (or .tex, .json)
# - lean/*.lean
```

## Integration with AutoForge

The orchestrator integrates seamlessly with AutoForge's other systems:

### With TheoryGraph

```python
# Use a pre-built TheoryGraph
from autoforge.engine.theoretical_reasoning import TheoryGraph

theory_graph = TheoryGraph(title="My Theory")
# ... populate with concepts ...

# Skip parsing, start from verification
result.theory_graph = theory_graph
result = await orchestrator._verify_phase(text, article, llm)
```

### With LLM Router

```python
from autoforge.engine.llm_router import LLMRouter

llm = LLMRouter()  # Auto-routes simple→Sonnet, complex→Opus
result = await orchestrator.reason(article, llm)
```

### With Formal Verification

The orchestrator automatically uses available provers:

- Lean 4 (if `lean` binary on PATH)
- Cloud Lean (if CloudProver configured)
- Z3/SMT (if multi-prover enabled)
- Coq, Isabelle (if available)

## Performance & Timing

Typical timings for a 5000-word article:

| Phase | Time |
|-------|------|
| Preprocess | 0.1s |
| Parse | 2-5s |
| Verify claims | 5-15s (depends on claim count) |
| Reconcile | 0.5-1s |
| Formalize | 10-30s (Lean compilation) |
| Discover | 5-10s/round |
| Extend | 5-10s/round |
| Generate | 2-5s |
| **Total** | **30-90s** (all phases) |

To speed up:
- Disable expensive phases (formalize_lean, extend_reasoning)
- Reduce max_discovery_rounds
- Use quick_reason() instead of full pipeline

## Examples

### Example 1: Verify a Math Paper

```python
from pathlib import Path
from autoforge.engine.article_reasoning import reason_about_pdf

result = await reason_about_pdf(
    pdf_path=Path("arxiv_paper.pdf"),
    llm=llm,
    verify_claims=True,
    discover_new=False,
    formalize_lean=True,
)

print(f"Verified: {result.verification_report.verified}/{result.verification_report.total_claims}")
print(f"Confidence: {result.overall_confidence:.1%}")
print(f"Lean files: {list(result.output_lean_files.keys())}")
```

### Example 2: Find Novel Results

```python
result = await reason_about_article(
    text=article_text,
    llm=llm,
    title="My Theory",
    discover_new=True,
    extend_reasoning=True,
    verify_claims=False,  # Skip expensive verification
)

for i, result in enumerate(result.discovery_results, 1):
    print(f"Round {i}: {result.conjectures_generated} new conjectures")

for i, rnd in enumerate(result.reasoning_rounds, 1):
    print(f"Round {i}: {rnd.accepted} conclusions accepted")
```

### Example 3: Generate Survey Article

```python
config = ArticleReasoningConfig(
    output_format="markdown",
    output_audience="survey",
    verify_claims=False,
    formalize_lean=False,
)

orchestrator = ArticleReasoningOrchestrator(config)
result = await orchestrator.reason(article, llm, output_dir=Path("survey/"))

# result.output_article is a markdown survey
# summary.md has overview
# reasoning_report.json has full metadata
```

## Troubleshooting

### Pipeline Too Slow

- Disable formalize_lean (most expensive)
- Reduce max_discovery_rounds
- Use quick_reason() for quick analysis

### Lean Compilation Failures

- Ensure Lean 4 + Mathlib installed: `elan toolchain install leanprover/lean4:latest`
- Or disable lean_compile: `config.lean_compile = False`

### PDF Extraction Issues

- Install pdfplumber: `pip install pdfplumber`
- Or manually extract text and use reason_about_article()

### Memory Issues with Large Articles

- Process in chunks (split article into sections)
- Disable discovery/extension for initial verify pass

### No Results from Discovery

- Increase max_discovery_rounds (default 10)
- Lower min_discovery_depth (try "SHALLOW")
- Check that verify_claims passed (provides context)

## Extensibility

### Custom Preprocessing

```python
from autoforge.engine.article_reasoning import ArticlePreprocessor

class CustomPreprocessor(ArticlePreprocessor):
    def _normalize_latex(self, text: str) -> str:
        # Your custom LaTeX normalization
        return text
```

### Custom Reconciliation

```python
from autoforge.engine.article_reasoning import TheoryGraphReconciler

class CustomReconciler(TheoryGraphReconciler):
    async def _match_claim_to_concept(self, claim, concepts, llm):
        # Your custom matching logic
        return best_concept_id
```

## Files & Locations

- **Module**: `/autoforge/engine/article_reasoning.py` (990 lines)
- **Config**: `ArticleReasoningConfig` dataclass
- **Entry points**: `ArticleReasoningOrchestrator` class, `reason_about_article()`, `reason_about_pdf()`
- **Dependencies**:
  - `article_verifier.py` (verify claims)
  - `theoretical_reasoning.py` (parse, generate)
  - `paper_formalizer.py` (formalize to Lean)
  - `autonomous_discovery.py` (discover new results)
  - `reasoning_extension.py` (extend reasoning)

## References

- **ArticleVerifier**: `autoforge/engine/article_verifier.py`
- **TheoryGraph**: `autoforge/engine/theoretical_reasoning.py`
- **PaperFormalizer**: `autoforge/engine/paper_formalizer.py`
- **DiscoveryOrchestrator**: `autoforge/engine/autonomous_discovery.py`
- **ReasoningExtensionEngine**: `autoforge/engine/reasoning_extension.py`
- **LLMRouter**: `autoforge/engine/llm_router.py`
