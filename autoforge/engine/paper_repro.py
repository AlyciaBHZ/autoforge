"""ICLR paper inference and reproduction utilities.

This module enables three related capabilities:
1. Infer likely papers from only a high-level goal description.
2. Build a reproduction brief/prompt for code generation.
3. Run lightweight benchmarks to measure inference quality and gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any
from urllib import parse, request


OPENREVIEW_BASE = "https://api2.openreview.net/notes"
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
    "our",
    "via",
    "using",
}
_COMMON_DATASETS = (
    "imagenet",
    "cifar-10",
    "cifar10",
    "cifar-100",
    "mnist",
    "fashion-mnist",
    "wmt",
    "squad",
    "glue",
    "superglue",
    "ms coco",
    "coco",
    "cityscapes",
    "kitti",
    "librispeech",
    "wikitext",
    "ptb",
    "mmlu",
    "gsm8k",
    "arc",
    "hellaswag",
)
_COMMON_METRICS = (
    "accuracy",
    "f1",
    "auc",
    "auroc",
    "bleu",
    "rouge",
    "perplexity",
    "wer",
    "cer",
    "iou",
    "mse",
    "mae",
    "top-1",
    "top-5",
    "latency",
    "throughput",
)
_GOAL_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "long-context": ("long", "context", "kv", "cache", "million-length"),
    "reasoning": ("chain-of-thought", "cot", "planning", "inference-time"),
    "sparse": ("pruning", "top-k", "token-selection", "sparsity"),
    "fast": ("latency", "throughput", "efficient", "acceleration"),
    "robust": ("noise", "corruption", "adversarial", "distribution-shift"),
    "graph": ("gnn", "node", "edge", "graph-structured"),
    "tool": ("agent", "planner", "external", "api"),
    "editing": ("edit", "instruction-based", "image-editing"),
}
_METHOD_TERMS = (
    "transformer",
    "diffusion",
    "gnn",
    "graph neural network",
    "sparse attention",
    "mixture of experts",
    "rlhf",
    "chain-of-thought",
    "speculative decoding",
    "retrieval-augmented",
)
_HARDWARE_TERMS = (
    "a100",
    "v100",
    "h100",
    "gpu",
    "tpu",
    "cuda",
    "memory",
    "batch size",
)


def _field_value(content: dict[str, Any], key: str) -> str:
    raw = content.get(key, "")
    if isinstance(raw, dict):
        value = raw.get("value", "")
        return value if isinstance(value, str) else str(value)
    if isinstance(raw, list):
        return ", ".join(str(x) for x in raw)
    return raw if isinstance(raw, str) else str(raw)


def _field_list(content: dict[str, Any], key: str) -> list[str]:
    raw = content.get(key, [])
    if isinstance(raw, dict):
        value = raw.get("value", [])
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        return []
    if isinstance(raw, list):
        return [str(v).strip() for v in raw if str(v).strip()]
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-\+\.]{1,}", _normalize(text))
    return [w for w in words if w not in _STOPWORDS]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _expanded_goal_tokens(goal: str) -> set[str]:
    base = set(_tokens(goal))
    norm = _normalize(goal)
    expanded = set(base)
    for trigger, adds in _GOAL_EXPANSIONS.items():
        if trigger in norm or trigger.replace("-", " ") in norm:
            expanded.update(_tokens(" ".join(adds)))
    return expanded


@dataclass
class PaperRecord:
    """Normalized paper metadata from OpenReview."""

    note_id: str
    title: str
    abstract: str
    keywords: list[str]
    year: int
    openreview_url: str
    pdf_url: str


@dataclass
class InferenceResult:
    """One ranked paper candidate for a goal."""

    paper: PaperRecord
    score: float
    matched_terms: list[str]


@dataclass
class BenchmarkCase:
    """One benchmark query outcome."""

    case_id: str
    goal: str
    gold_title: str
    top_titles: list[str]
    hit_at_1: bool
    hit_at_k: bool
    top_score: float


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report."""

    year: int
    sample_size: int
    top_k: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    cases: list[BenchmarkCase]
    gaps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "sample_size": self.sample_size,
            "top_k": self.top_k,
            "hit_at_1": self.hit_at_1,
            "hit_at_k": self.hit_at_k,
            "mrr": self.mrr,
            "cases": [
                {
                    "case_id": c.case_id,
                    "goal": c.goal,
                    "gold_title": c.gold_title,
                    "top_titles": c.top_titles,
                    "hit_at_1": c.hit_at_1,
                    "hit_at_k": c.hit_at_k,
                    "top_score": c.top_score,
                }
                for c in self.cases
            ],
            "gaps": self.gaps,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class PaperSignals:
    """Structured extraction of reproducibility signals from paper text."""

    datasets: list[str]
    metrics: list[str]
    methods: list[str]
    hardware_hints: list[str]
    claimed_metrics: list[dict[str, str]]
    text_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasets": self.datasets,
            "metrics": self.metrics,
            "methods": self.methods,
            "hardware_hints": self.hardware_hints,
            "claimed_metrics": self.claimed_metrics,
            "text_source": self.text_source,
        }


def fetch_iclr_papers(
    *,
    year: int,
    limit: int = 600,
    timeout_seconds: int = 25,
    use_cache: bool = True,
    cache_max_age_hours: int = 24,
) -> list[PaperRecord]:
    """Fetch ICLR papers from OpenReview."""
    cache_file = Path(".autoforge") / "cache" / f"iclr_{year}.json"
    if use_cache and cache_file.exists():
        age_seconds = max(0.0, datetime.now().timestamp() - cache_file.stat().st_mtime)
        if age_seconds <= float(cache_max_age_hours * 3600):
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                papers = [PaperRecord(**row) for row in cached if isinstance(row, dict)]
                if len(papers) >= min(limit, 100):
                    return papers[:limit]
            except Exception:
                pass

    papers: list[PaperRecord] = []
    offset = 0
    page_size = min(max(limit, 1), 1000)
    venue = f"ICLR.cc/{year}/Conference"

    while len(papers) < limit:
        params = {
            "content.venueid": venue,
            "limit": str(min(page_size, limit - len(papers))),
            "offset": str(offset),
        }
        url = f"{OPENREVIEW_BASE}?{parse.urlencode(params)}"
        req = request.Request(url, headers={"User-Agent": "AutoForge/2.8"})
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        notes = payload.get("notes", [])
        if not notes:
            break

        for note in notes:
            content = note.get("content", {})
            title = _field_value(content, "title")
            abstract = _field_value(content, "abstract")
            if not title or not abstract:
                continue
            note_id = str(note.get("id", "")).strip()
            if not note_id:
                continue
            papers.append(
                PaperRecord(
                    note_id=note_id,
                    title=title.strip(),
                    abstract=abstract.strip(),
                    keywords=_field_list(content, "keywords"),
                    year=year,
                    openreview_url=f"https://openreview.net/forum?id={note_id}",
                    pdf_url=f"https://openreview.net/pdf?id={note_id}",
                )
            )

        offset += len(notes)
        if len(notes) < page_size:
            break

    if use_cache:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps([p.__dict__ for p in papers], ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    return papers[:limit]


def fetch_pdf_text(
    pdf_url: str,
    *,
    timeout_seconds: int = 25,
    max_pages: int = 12,
    max_chars: int = 120_000,
) -> str:
    """Fetch and parse paper PDF text if a PDF parser is available."""
    if not pdf_url:
        return ""
    req = request.Request(pdf_url, headers={"User-Agent": "AutoForge/2.8"})
    with request.urlopen(req, timeout=timeout_seconds) as resp:
        blob = resp.read()

    reader = None
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        reader = PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found]

            reader = PdfReader
        except Exception:
            return ""

    try:
        from io import BytesIO

        pdf = reader(BytesIO(blob))
        chunks: list[str] = []
        for page in list(pdf.pages)[:max_pages]:
            text = page.extract_text() or ""
            if text:
                chunks.append(text)
        joined = "\n".join(chunks)
        joined = re.sub(r"\s+", " ", joined).strip()
        return joined[:max_chars]
    except Exception:
        return ""


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    norm = _normalize(text)
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(0, len(norm) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _build_idf_map(papers: list[PaperRecord]) -> dict[str, float]:
    df: dict[str, int] = {}
    for p in papers:
        seen = set(_tokens(p.title + " " + " ".join(p.keywords) + " " + p.abstract))
        for t in seen:
            df[t] = df.get(t, 0) + 1
    n_docs = max(1, len(papers))
    return {t: math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0 for t, freq in df.items()}


def infer_papers_from_goal(
    goal: str,
    papers: list[PaperRecord],
    *,
    top_k: int = 5,
) -> list[InferenceResult]:
    """Rank candidate papers from a goal string."""
    goal_set = _expanded_goal_tokens(goal)
    if not goal_set:
        return []
    idf = _build_idf_map(papers)
    goal_norm = _normalize(goal)
    goal_ngrams = _char_ngrams(goal_norm)

    results: list[InferenceResult] = []
    for p in papers:
        title_tokens = _tokens(p.title)
        abs_tokens = _tokens(p.abstract)
        kw_tokens = _tokens(" ".join(p.keywords))

        title_set = set(title_tokens)
        abs_set = set(abs_tokens)
        kw_set = set(kw_tokens)

        # Weighted lexical overlap (idf-weighted).
        title_overlap = sum(idf.get(t, 1.0) for t in (goal_set & title_set))
        abs_overlap = sum(idf.get(t, 1.0) for t in (goal_set & abs_set))
        kw_overlap = sum(idf.get(t, 1.0) for t in (goal_set & kw_set))

        # Lightweight semantic similarity using char n-grams.
        title_sem = _jaccard(goal_ngrams, _char_ngrams(p.title))
        abs_sem = _jaccard(goal_ngrams, _char_ngrams(p.abstract[:1200]))

        phrase_bonus = 0.0
        if goal_norm and goal_norm in _normalize(p.abstract):
            phrase_bonus = 2.0
        elif goal_norm and goal_norm in _normalize(p.title):
            phrase_bonus = 2.5

        score = (
            3.0 * title_overlap
            + 1.5 * kw_overlap
            + 1.0 * abs_overlap
            + 8.0 * title_sem
            + 4.0 * abs_sem
            + phrase_bonus
        )
        if score <= 0:
            continue

        matched = sorted(list(goal_set & (title_set | kw_set | abs_set)))
        results.append(InferenceResult(paper=p, score=float(score), matched_terms=matched))

    results.sort(
        key=lambda r: (
            r.score,
            len(r.matched_terms),
            -len(r.paper.title),
        ),
        reverse=True,
    )
    return results[:max(1, top_k)]


def extract_goal_from_abstract(abstract: str) -> str:
    """Build a short goal sentence from abstract text."""
    text = re.sub(r"\s+", " ", abstract.strip())
    if not text:
        return ""
    # Keep one to two sentence-like fragments for enough disambiguation.
    parts = [p.strip() for p in re.split(r"(?<=[\.\!\?])\s+", text) if p.strip()]
    if not parts:
        return text[:220].strip()
    goal = parts[0]
    if len(goal) < 90 and len(parts) > 1:
        goal = f"{goal} {parts[1]}".strip()
    if len(goal) > 260:
        goal = goal[:260].rsplit(" ", 1)[0].strip()
    return goal


def _extract_hits(text: str, candidates: tuple[str, ...]) -> list[str]:
    norm = _normalize(text)
    hits: list[str] = []
    for c in candidates:
        pattern = r"(?<![a-z0-9])" + re.escape(c).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        if re.search(pattern, norm):
            hits.append(c)
    # Stable unique list
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out


def _extract_claimed_metrics(text: str) -> list[dict[str, str]]:
    norm = text
    pattern = re.compile(
        r"\b(accuracy|f1|auc|auroc|bleu|rouge|perplexity|wer|cer|iou|mse|mae|latency|throughput)\b"
        r"[^0-9]{0,20}"
        r"([0-9]+(?:\.[0-9]+)?)"
        r"\s*(%|x|ms|s)?",
        flags=re.IGNORECASE,
    )
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for m in pattern.finditer(norm):
        metric = m.group(1).lower()
        value = m.group(2)
        unit = (m.group(3) or "").strip()
        key = (metric, value, unit)
        if key in seen:
            continue
        seen.add(key)
        out.append({"metric": metric, "value": value, "unit": unit})
        if len(out) >= 20:
            break
    return out


def extract_paper_signals(
    paper: PaperRecord,
    *,
    include_pdf: bool = True,
    timeout_seconds: int = 25,
) -> PaperSignals:
    """Extract reproducibility signals from abstract + optional PDF text."""
    text = f"{paper.title}\n{paper.abstract}\n{' '.join(paper.keywords)}"
    text_source = "abstract+keywords"

    if include_pdf:
        pdf_text = fetch_pdf_text(paper.pdf_url, timeout_seconds=timeout_seconds)
        if pdf_text:
            text = f"{text}\n{pdf_text}"
            text_source = "abstract+keywords+pdf"

    datasets = _extract_hits(text, _COMMON_DATASETS)
    metrics = _extract_hits(text, _COMMON_METRICS)
    methods = _extract_hits(text, _METHOD_TERMS)
    hardware = _extract_hits(text, _HARDWARE_TERMS)
    claimed = _extract_claimed_metrics(text)

    return PaperSignals(
        datasets=datasets,
        metrics=metrics,
        methods=methods,
        hardware_hints=hardware,
        claimed_metrics=claimed,
        text_source=text_source,
    )


def _ml_stack_from_text(text: str) -> list[dict[str, str]]:
    """Infer optional ML dependencies from method hints in text."""
    deps: list[dict[str, str]] = []
    ml_needed = any(t in text for t in ("transformer", "diffusion", "llm", "gnn", "rlhf"))
    if ml_needed:
        deps.extend(
            [
                {"name": "torch", "version": "2.3.*"},
                {"name": "transformers", "version": "4.4x"},
                {"name": "datasets", "version": "2.18.*"},
            ]
        )
    if "diffusion" in text:
        deps.append({"name": "diffusers", "version": "0.29.*"})
    if "graph" in text or "gnn" in text:
        deps.append({"name": "torch-geometric", "version": "2.5.*"})
    if "rlhf" in text:
        deps.append({"name": "trl", "version": "0.9.*"})

    # Keep stable order while deduplicating by package name.
    seen: set[str] = set()
    uniq: list[dict[str, str]] = []
    for dep in deps:
        name = dep["name"]
        if name in seen:
            continue
        seen.add(name)
        uniq.append(dep)
    return uniq


def build_environment_spec(
    paper: PaperRecord,
    signals: PaperSignals,
    *,
    theory_first: bool = True,
) -> dict[str, Any]:
    """Create a reproducibility-oriented environment spec."""
    base_deps = [
        {"name": "python", "version": "3.10"},
        {"name": "numpy", "version": "1.26.*"},
        {"name": "sympy", "version": "1.12.*"},
        {"name": "networkx", "version": "3.3.*"},
    ]
    if not theory_first:
        base_deps.append({"name": "pandas", "version": "2.2.*"})

    text = _normalize(f"{paper.title} {paper.abstract} {' '.join(signals.methods)}")
    ml_stack = _ml_stack_from_text(text)
    dependencies = list(base_deps)
    optional_dependencies: list[dict[str, str]] = []
    if theory_first:
        optional_dependencies = ml_stack
        platform = {
            "os": "linux/macOS/windows",
            "runtime": "cpu-first",
            "gpu": "optional (enable only for compute-heavy reproductions)",
        }
        install_policy = "minimal-default"
        determinism = [
            "Set all random seeds (python/numpy and task-specific frameworks if used).",
            "Log exact git commit, package lock, and hardware profile.",
        ]
    else:
        dependencies.extend(ml_stack)
        platform = {
            "os": "linux (recommended)",
            "cuda": "12.x (recommended for GPU workloads)",
            "gpu": "A100/V100/H100 class if available",
        }
        install_policy = "full-reproduction"
        determinism = [
            "Set all random seeds (python/numpy/torch).",
            "Log exact git commit, package lock, and hardware.",
        ]

    return {
        "paper_title": paper.title,
        "note_id": paper.note_id,
        "python": "3.10",
        "profile": "theory-first" if theory_first else "general",
        "install_policy": install_policy,
        "platform": platform,
        "dependencies": dependencies,
        "optional_dependencies": optional_dependencies,
        "seed": 42,
        "determinism": determinism,
    }


def build_verification_plan(signals: PaperSignals) -> dict[str, Any]:
    """Create verification checklist and result schema for reproduction."""
    targets = signals.claimed_metrics[:10]
    if not targets and signals.metrics:
        targets = [{"metric": m, "value": "", "unit": ""} for m in signals.metrics[:5]]
    return {
        "checklist": [
            "Reproduce at least one primary metric from the paper.",
            "Report mean/std across >=3 seeds.",
            "Record wall-clock training/eval time.",
            "Document all deviations from paper setup.",
        ],
        "expected_metrics": targets,
        "result_schema": {
            "metric": "str",
            "paper_claim": "str",
            "reproduced": "str",
            "delta": "str",
            "status": "matched|partial|failed|unknown",
        },
    }


def simulate_pipeline_feedback(
    *,
    goal: str,
    paper: PaperRecord,
    signals: PaperSignals,
    inference_score: float,
) -> dict[str, Any]:
    """Simulate pipeline feedback when no API key is available."""
    p0_status = "resolved_with_simulation"
    p1_status = "ok" if inference_score >= 12 else "risky"
    p2_status = "ok" if (signals.datasets and signals.metrics) else "needs_manual_fill"
    p3_status = "ok" if signals.claimed_metrics else "insufficient_claimed_numbers"
    p4_status = "ok" if signals.hardware_hints else "hardware_unknown"

    recommendations: list[str] = []
    if p1_status != "ok":
        recommendations.append("Expand goal text with method keywords and target domain terms.")
    if p2_status != "ok":
        recommendations.append("Manually add exact datasets/metrics from PDF tables before training.")
    if p3_status != "ok":
        recommendations.append("Extract at least one numeric claim to enable pass/fail verification.")
    if p4_status != "ok":
        recommendations.append("Add explicit GPU/CUDA/package pins for deterministic reruns.")

    return {
        "mode": "simulated_no_api_key",
        "goal": goal,
        "paper": {
            "title": paper.title,
            "note_id": paper.note_id,
            "openreview_url": paper.openreview_url,
        },
        "phases": {
            "SPEC": "completed",
            "BUILD": "simulated",
            "VERIFY": "simulated",
            "REFACTOR": "simulated",
            "DELIVER": "simulated",
        },
        "p0_p4_status": {
            "P0_api_key_runtime": p0_status,
            "P1_goal_to_paper_retrieval": p1_status,
            "P2_paper_signal_extraction": p2_status,
            "P3_closed_loop_verification": p3_status,
            "P4_environment_reproducibility": p4_status,
        },
        "recommendations": recommendations,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_reproduction_brief(
    goal: str,
    paper: PaperRecord,
    *,
    signals: PaperSignals | None = None,
) -> str:
    """Create a structured brief for downstream coding agents."""
    sig = signals or extract_paper_signals(paper, include_pdf=False)
    datasets = sig.datasets
    metrics = sig.metrics
    kw = paper.keywords[:8]

    lines = [
        "# Paper Reproduction Brief",
        "",
        f"- Goal Query: {goal}",
        f"- Inferred Paper: {paper.title}",
        f"- Year: {paper.year}",
        f"- OpenReview: {paper.openreview_url}",
        f"- PDF: {paper.pdf_url}",
        "",
        "## Objective",
        paper.abstract,
        "",
        "## Reproduction Targets",
        "- Re-implement the core method described in the paper.",
        "- Reproduce at least one main metric table/figure trend with transparent assumptions.",
        "- Document all deviations from the paper setup.",
        "",
        "## Candidate Datasets",
    ]
    if datasets:
        lines.extend([f"- {d}" for d in datasets])
    else:
        lines.append("- Not confidently identified from abstract; manual confirmation required.")
    lines.append("")
    lines.append("## Candidate Metrics")
    if metrics:
        lines.extend([f"- {m}" for m in metrics])
    else:
        lines.append("- Not confidently identified from abstract; manual confirmation required.")
    lines.append("")
    lines.append("## Keywords")
    if kw:
        lines.extend([f"- {k}" for k in kw])
    else:
        lines.append("- No keywords available")
    lines.append("")
    lines.append("## Engineering Requirements")
    lines.append("- Pin exact package versions and seed values.")
    lines.append("- Keep experiment configs in a machine-readable format.")
    lines.append("- Add scripts for data prep, train, evaluate, and report generation.")
    lines.append("- Include a short replication report with achieved vs. claimed metrics.")
    lines.append(f"- Signal source used: {sig.text_source}.")
    return "\n".join(lines)


def build_generation_prompt(
    goal: str,
    paper: PaperRecord,
    *,
    signals: PaperSignals | None = None,
) -> str:
    """Build a ready-to-run AutoForge generation prompt."""
    brief = build_reproduction_brief(goal, paper, signals=signals)
    return (
        "Reproduce this ICLR paper as an executable research codebase.\n\n"
        f"{brief}\n\n"
        "Deliverables:\n"
        "1. Training/inference pipeline code\n"
        "2. Config files for reproducible runs\n"
        "3. Evaluation scripts and summary report\n"
        "4. README with exact commands and expected outputs\n"
    )


def run_goal_inference_benchmark(
    *,
    year: int,
    sample_size: int = 5,
    corpus_size: int = 600,
    top_k: int = 5,
    seed: int = 42,
    use_cache: bool = True,
    cache_max_age_hours: int = 24,
) -> BenchmarkReport:
    """Benchmark paper inference by masking papers into short goal prompts."""
    papers = fetch_iclr_papers(
        year=year,
        limit=corpus_size,
        use_cache=use_cache,
        cache_max_age_hours=cache_max_age_hours,
    )
    if len(papers) < sample_size:
        raise ValueError(f"Not enough papers fetched: got {len(papers)}, need {sample_size}")

    rnd = random.Random(seed)
    sample = rnd.sample([p for p in papers if p.abstract], k=sample_size)
    cases: list[BenchmarkCase] = []

    reciprocal_ranks: list[float] = []
    for p in sample:
        goal = extract_goal_from_abstract(p.abstract)
        ranked = infer_papers_from_goal(goal, papers, top_k=top_k)
        titles = [r.paper.title for r in ranked]
        hit1 = bool(titles) and titles[0] == p.title
        hitk = p.title in titles

        rr = 0.0
        for idx, t in enumerate(titles, start=1):
            if t == p.title:
                rr = 1.0 / float(idx)
                break
        reciprocal_ranks.append(rr)

        cases.append(
            BenchmarkCase(
                case_id=_hash_text(p.note_id + goal),
                goal=goal,
                gold_title=p.title,
                top_titles=titles,
                hit_at_1=hit1,
                hit_at_k=hitk,
                top_score=ranked[0].score if ranked else 0.0,
            )
        )

    hit1 = sum(1 for c in cases if c.hit_at_1) / float(len(cases))
    hitk = sum(1 for c in cases if c.hit_at_k) / float(len(cases))
    mrr = sum(reciprocal_ranks) / float(len(reciprocal_ranks))

    gaps = [
        "P1: Retrieval can still miss highly abstract/paraphrased goals despite semantic mixing.",
        "P2: Method signatures/equation-level matching is not implemented yet.",
        "P2: PDF extraction depends on optional parser availability (pypdf/PyPDF2).",
        "P3: Verification currently relies on planned schemas, not automatic figure/table alignment.",
        "P4: Environment spec is generated, but container build/test execution is not enforced automatically.",
    ]

    return BenchmarkReport(
        year=year,
        sample_size=sample_size,
        top_k=top_k,
        hit_at_1=hit1,
        hit_at_k=hitk,
        mrr=mrr,
        cases=cases,
        gaps=gaps,
    )


# ============================================================================


# ============================================================================
# Reproduction Pipeline: Full orchestrator for paper reproduction
# ============================================================================

import asyncio
import logging
import subprocess
import time
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ReproductionStatus(str, Enum):
    """Status of a reproduction attempt."""
    PENDING = "pending"
    PAPER_FOUND = "paper_found"
    SIGNALS_EXTRACTED = "signals_extracted"
    CODE_GENERATED = "code_generated"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPARING = "comparing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReproductionResult:
    """Result of a paper reproduction attempt."""
    paper: PaperRecord
    signals: PaperSignals
    status: ReproductionStatus
    generated_code: str = ""
    execution_output: str = ""
    reproduced_metrics: list[dict] = field(default_factory=list)
    match_rate: float = 0.0
    figures: list[Path] = field(default_factory=list)
    report_path: Path | None = None
    total_time: float = 0.0
    error: str = ""


@dataclass
class ReproductionConfig:
    """Configuration for reproduction pipeline."""
    metric_tolerance: float = 0.05
    max_code_attempts: int = 3
    execution_timeout: int = 600
    workspace_dir: Path = field(default_factory=lambda: Path("workspace/reproductions"))
    include_pdf: bool = True
    python_executable: str = sys.executable
    auto_install_deps: bool = True


class CodeGenerator:
    """Generates Python reproduction code from paper signals."""

    async def generate_reproduction_code(
        self,
        paper: PaperRecord,
        signals: PaperSignals,
        brief: str,
        llm: Any,
    ) -> str:
        """Generate a self-contained Python reproduction script.

        Args:
            paper: The paper to reproduce.
            signals: Extracted paper signals.
            brief: Brief description of the paper and methodology.
            llm: LLM callable that takes a prompt and returns a response.

        Returns:
            Generated Python code as a string.
        """
        prompt = self._build_generation_prompt(paper, signals, brief)
        logger.info(f"Generating reproduction code for '{paper.title}'")
        response = await llm(prompt)
        return response

    async def fix_code(
        self,
        code: str,
        error: str,
        llm: Any,
    ) -> str:
        """Fix code based on execution errors.

        Args:
            code: The original code that failed.
            error: The error message from execution.
            llm: LLM callable.

        Returns:
            Fixed Python code.
        """
        prompt = f"""The following reproduction code failed with an error.
Please fix it and return only the corrected Python code.

Original code:
```python
{code}
```

Error:
{error}

Return the complete fixed code, nothing else."""
        logger.info("Fixing code based on execution error")
        response = await llm(prompt)
        return response

    def _build_generation_prompt(
        self,
        paper: PaperRecord,
        signals: PaperSignals,
        brief: str,
    ) -> str:
        """Build the LLM prompt for code generation."""
        return f"""You are an expert at reproducing machine learning papers.

Paper: {paper.title}
Authors: {', '.join(paper.authors[:3])}
Year: {paper.year}
Venue: {paper.venue}

Abstract:
{paper.abstract}

Methodology Summary:
{brief}

Claimed Metrics:
{json.dumps(signals.metrics, indent=2)}

Your task: Generate a complete, self-contained Python script that reproduces the paper's main results.

Requirements:
1. The script must be runnable end-to-end
2. Include all necessary imports and data preparation
3. Output metrics in the format: METRIC: <name> = <value>
4. Save any generated figures as PNG files
5. Use common datasets or synthetic data if the original is unavailable
6. Include comments explaining key steps

Return ONLY the Python code, nothing else."""


class MetricComparator:
    """Compares claimed vs reproduced metrics."""

    @staticmethod
    def compare(
        claimed: list[dict],
        reproduced: dict[str, float],
        tolerance: float = 0.05,
    ) -> list[dict]:
        """Compare claimed metrics against reproduced metrics.

        Args:
            claimed: List of claimed metrics with 'name' and 'value' keys.
            reproduced: Dict mapping metric names to reproduced values.
            tolerance: Relative tolerance for matching (default 0.05 = 5%).

        Returns:
            List of comparison dicts with keys:
            - metric: str
            - paper_claim: float
            - reproduced: float | None
            - delta: float | None (absolute difference)
            - status: str ("matched", "partial", or "failed")
        """
        comparisons = []
        for claim in claimed:
            metric_name = claim.get("name", "unknown")
            paper_value = claim.get("value")

            reproduced_value = reproduced.get(metric_name)

            if reproduced_value is None:
                comparisons.append({
                    "metric": metric_name,
                    "paper_claim": paper_value,
                    "reproduced": None,
                    "delta": None,
                    "status": "failed",
                })
                continue

            # Compute relative error
            if paper_value == 0:
                if reproduced_value == 0:
                    status = "matched"
                    delta = 0.0
                else:
                    status = "failed"
                    delta = abs(reproduced_value - paper_value)
            else:
                delta = abs(reproduced_value - paper_value)
                relative_error = delta / abs(paper_value)

                if relative_error <= tolerance:
                    status = "matched"
                elif relative_error <= 2 * tolerance:
                    status = "partial"
                else:
                    status = "failed"

            comparisons.append({
                "metric": metric_name,
                "paper_claim": paper_value,
                "reproduced": reproduced_value,
                "delta": delta,
                "status": status,
            })

        return comparisons

    @staticmethod
    def compute_match_rate(comparisons: list[dict]) -> float:
        """Compute fraction of metrics that matched.

        Args:
            comparisons: List of comparison dicts.

        Returns:
            Float between 0 and 1.
        """
        if not comparisons:
            return 0.0
        matched = sum(1 for c in comparisons if c["status"] == "matched")
        return matched / len(comparisons)


class ReproductionReportGenerator:
    """Generates reproduction reports in multiple formats."""

    @staticmethod
    def generate_markdown(result: ReproductionResult) -> str:
        """Generate a detailed markdown reproduction report.

        Args:
            result: Reproduction result.

        Returns:
            Markdown report as string.
        """
        paper = result.paper
        signals = result.signals

        # Determine verdict
        if result.status == ReproductionStatus.COMPLETED:
            if result.match_rate >= 0.8:
                verdict = "Successfully Reproduced"
            elif result.match_rate >= 0.5:
                verdict = "Partially Reproduced"
            else:
                verdict = "Failed to Reproduce"
        else:
            verdict = f"Failed ({result.status.value})"

        # Build metrics table
        metrics_table = "| Metric | Paper Claim | Reproduced | Delta | Status |\n"
        metrics_table += "|--------|-------------|-----------|-------|--------|\n"
        for comp in result.reproduced_metrics:
            metric = comp.get("metric", "N/A")
            claim = comp.get("paper_claim", "N/A")
            repro = comp.get("reproduced", "N/A")
            delta = comp.get("delta", "N/A")
            status = comp.get("status", "N/A")

            if isinstance(delta, float):
                delta = f"{delta:.4f}"
            if isinstance(repro, float):
                repro = f"{repro:.4f}"
            if isinstance(claim, float):
                claim = f"{claim:.4f}"

            metrics_table += f"| {metric} | {claim} | {repro} | {delta} | {status} |\n"

        # Build figures section
        figures_section = ""
        if result.figures:
            figures_section = "\n## Generated Figures\n\n"
            for fig_path in result.figures:
                figures_section += f"- {fig_path.name}\n"

        # Build error section
        error_section = ""
        if result.error:
            error_section = f"\n### Error\n{result.error}\n"

        # Build report
        report = f"""# Paper Reproduction Report

## Paper Metadata
- **Title:** {paper.title}
- **Authors:** {', '.join(paper.authors)}
- **Year:** {paper.year}
- **Venue:** {paper.venue}
- **OpenReview ID:** {paper.note_id}

## Verdict
**{verdict}** (Match Rate: {result.match_rate:.1%})

## Methodology Summary
{signals.methodology}

## Claimed Metrics
{json.dumps(signals.metrics, indent=2)}

## Reproduced Metrics

{metrics_table}

## Deviations from Paper Setup
{signals.deviations or 'None identified.'}

{figures_section}

## Execution Summary
- **Status:** {result.status.value}
- **Time Taken:** {result.total_time:.2f}s
- **Match Rate:** {result.match_rate:.1%}

{error_section}

---
*Report generated for AutoForge reproduction pipeline*
"""
        return report

    @staticmethod
    async def generate_latex(
        result: ReproductionResult,
        llm: Any,
    ) -> str:
        """Generate a LaTeX reproduction report.

        Args:
            result: Reproduction result.
            llm: LLM callable for formatting.

        Returns:
            LaTeX code as string.
        """
        paper = result.paper
        markdown = ReproductionReportGenerator.generate_markdown(result)

        prompt = f"""Convert the following markdown reproduction report into LaTeX format.
Use the article document class with appropriate sections, tables, and formatting.

Markdown:
{markdown}

Return only the LaTeX code."""
        logger.info("Generating LaTeX report via LLM")
        latex_code = await llm(prompt)
        return latex_code


class ReproductionPipeline:
    """Main orchestrator for paper reproduction."""

    def __init__(self, config: ReproductionConfig) -> None:
        """Initialize reproduction pipeline.

        Args:
            config: Reproduction configuration.
        """
        self.config = config
        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.code_gen = CodeGenerator()
        self.comparator = MetricComparator()
        self.report_gen = ReproductionReportGenerator()
        logger.info(f"Initialized ReproductionPipeline with workspace: {config.workspace_dir}")

    async def reproduce(
        self,
        goal: str,
        llm: Any,
        *,
        year: int = 2025,
        corpus_size: int = 200,
    ) -> ReproductionResult:
        """Full reproduction pipeline from goal description.

        Args:
            goal: High-level description of the paper/goal.
            llm: LLM callable for code generation and refinement.
            year: Filter papers from this year and later.
            corpus_size: Number of papers to search within.

        Returns:
            ReproductionResult with all details.
        """
        start_time = time.time()
        result = ReproductionResult(
            paper=PaperRecord(
                note_id="",
                title="",
                authors=[],
                abstract="",
                year=0,
                venue="",
                pdf_url="",
            ),
            signals=PaperSignals(
                methodology="",
                metrics=[],
                deviations="",
            ),
            status=ReproductionStatus.PENDING,
        )

        try:
            # a. Fetch papers and find best match
            logger.info(f"Fetching ICLR papers from {year} onward (corpus_size={corpus_size})")
            papers = await asyncio.to_thread(
                fetch_iclr_papers,
                year_min=year,
                limit=corpus_size,
            )
            if not papers:
                result.error = "No papers found in corpus"
                result.status = ReproductionStatus.FAILED
                result.total_time = time.time() - start_time
                return result

            ranked = infer_papers_from_goal(goal, papers, top_k=1)
            if not ranked:
                result.error = "Could not infer paper from goal"
                result.status = ReproductionStatus.FAILED
                result.total_time = time.time() - start_time
                return result

            paper = ranked[0].paper
            result.paper = paper
            result.status = ReproductionStatus.PAPER_FOUND
            logger.info(f"Selected paper: {paper.title}")

            # b. Extract signals
            logger.info("Extracting paper signals")
            signals = extract_paper_signals(paper)
            result.signals = signals
            result.status = ReproductionStatus.SIGNALS_EXTRACTED

            # c. Build reproduction brief
            brief = build_reproduction_brief(paper, signals)

            # d. Generate code
            logger.info("Generating reproduction code")
            code = await self.code_gen.generate_reproduction_code(paper, signals, brief, llm)
            result.generated_code = code
            result.status = ReproductionStatus.CODE_GENERATED

            # e. Execute code (with retry logic)
            workspace = self.config.workspace_dir / _hash_text(paper.note_id)
            workspace.mkdir(parents=True, exist_ok=True)
            result.status = ReproductionStatus.RUNNING

            attempt = 0
            stdout, stderr, exit_code = "", "", 1
            while attempt < self.config.max_code_attempts and exit_code != 0:
                attempt += 1
                logger.info(f"Execution attempt {attempt}/{self.config.max_code_attempts}")
                stdout, stderr, exit_code = await self._execute_code(code, workspace)

                if exit_code != 0 and attempt < self.config.max_code_attempts:
                    logger.warning(f"Execution failed: {stderr}")
                    code = await self.code_gen.fix_code(code, stderr, llm)

            if exit_code != 0:
                result.error = f"Code execution failed after {self.config.max_code_attempts} attempts"
                result.status = ReproductionStatus.FAILED
                result.total_time = time.time() - start_time
                return result

            result.execution_output = stdout

            # f. Parse metrics
            logger.info("Evaluating reproduced metrics")
            result.status = ReproductionStatus.EVALUATING
            reproduced_metrics = await self._parse_metrics(stdout)

            # g. Compare metrics
            logger.info("Comparing metrics against paper claims")
            result.status = ReproductionStatus.COMPARING
            comparisons = self.comparator.compare(
                signals.metrics,
                reproduced_metrics,
                tolerance=self.config.metric_tolerance,
            )
            result.reproduced_metrics = comparisons
            result.match_rate = self.comparator.compute_match_rate(comparisons)

            # h. Collect figures
            result.figures = self._collect_figures(workspace)

            # i. Generate report
            logger.info("Generating reproduction report")
            report_markdown = self.report_gen.generate_markdown(result)
            report_path = workspace / "REPRODUCTION_REPORT.md"
            report_path.write_text(report_markdown, encoding="utf-8")
            result.report_path = report_path

            result.status = ReproductionStatus.COMPLETED
            result.total_time = time.time() - start_time
            logger.info(f"Reproduction completed in {result.total_time:.2f}s")

        except Exception as e:
            logger.exception(f"Reproduction pipeline failed: {e}")
            result.error = str(e)
            result.status = ReproductionStatus.FAILED
            result.total_time = time.time() - start_time

        return result

    async def reproduce_paper(
        self,
        paper: PaperRecord,
        llm: Any,
    ) -> ReproductionResult:
        """Reproduce from a known PaperRecord directly.

        Args:
            paper: The paper to reproduce.
            llm: LLM callable.

        Returns:
            ReproductionResult.
        """
        start_time = time.time()
        result = ReproductionResult(
            paper=paper,
            signals=extract_paper_signals(paper),
            status=ReproductionStatus.SIGNALS_EXTRACTED,
        )

        try:
            signals = result.signals
            brief = build_reproduction_brief(paper, signals)

            logger.info(f"Generating code for paper: {paper.title}")
            code = await self.code_gen.generate_reproduction_code(paper, signals, brief, llm)
            result.generated_code = code
            result.status = ReproductionStatus.CODE_GENERATED

            workspace = self.config.workspace_dir / _hash_text(paper.note_id)
            workspace.mkdir(parents=True, exist_ok=True)
            result.status = ReproductionStatus.RUNNING

            attempt = 0
            stdout, stderr, exit_code = "", "", 1
            while attempt < self.config.max_code_attempts and exit_code != 0:
                attempt += 1
                stdout, stderr, exit_code = await self._execute_code(code, workspace)

                if exit_code != 0 and attempt < self.config.max_code_attempts:
                    code = await self.code_gen.fix_code(code, stderr, llm)

            if exit_code != 0:
                result.error = f"Code execution failed after {self.config.max_code_attempts} attempts"
                result.status = ReproductionStatus.FAILED
                result.total_time = time.time() - start_time
                return result

            result.execution_output = stdout
            result.status = ReproductionStatus.EVALUATING

            reproduced_metrics = await self._parse_metrics(stdout)
            result.status = ReproductionStatus.COMPARING

            comparisons = self.comparator.compare(
                signals.metrics,
                reproduced_metrics,
                tolerance=self.config.metric_tolerance,
            )
            result.reproduced_metrics = comparisons
            result.match_rate = self.comparator.compute_match_rate(comparisons)

            result.figures = self._collect_figures(workspace)

            report_markdown = self.report_gen.generate_markdown(result)
            report_path = workspace / "REPRODUCTION_REPORT.md"
            report_path.write_text(report_markdown, encoding="utf-8")
            result.report_path = report_path

            result.status = ReproductionStatus.COMPLETED
            result.total_time = time.time() - start_time

        except Exception as e:
            logger.exception(f"Reproduction failed: {e}")
            result.error = str(e)
            result.status = ReproductionStatus.FAILED
            result.total_time = time.time() - start_time

        return result

    async def _execute_code(
        self,
        code: str,
        workspace: Path,
    ) -> tuple[str, str, int]:
        """Execute code in subprocess.

        Args:
            code: Python code to execute.
            workspace: Directory to run code in.

        Returns:
            Tuple of (stdout, stderr, exit_code).
        """
        code_file = workspace / "reproduce.py"
        code_file.write_text(code, encoding="utf-8")

        logger.debug(f"Running code in {workspace}")
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    [self.config.python_executable, str(code_file)],
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                    timeout=self.config.execution_timeout,
                ),
                timeout=self.config.execution_timeout + 5,
            )
            return result.stdout, result.stderr, result.returncode
        except asyncio.TimeoutError:
            return "", f"Execution timeout after {self.config.execution_timeout}s", 1
        except Exception as e:
            return "", str(e), 1

    async def _parse_metrics(self, stdout: str) -> dict[str, float]:
        """Parse metrics from stdout.

        Expects lines like: METRIC: name = value

        Args:
            stdout: Output from reproduction code.

        Returns:
            Dict mapping metric names to values.
        """
        metrics = {}
        for line in stdout.split("\n"):
            line = line.strip()
            if line.startswith("METRIC:"):
                # Format: METRIC: accuracy = 0.95
                parts = line[7:].strip().split("=")
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        value = float(parts[1].strip())
                        metrics[name] = value
                    except ValueError:
                        pass
        return metrics

    @staticmethod
    def _collect_figures(workspace: Path) -> list[Path]:
        """Collect generated figures from workspace.

        Args:
            workspace: Directory to search for figures.

        Returns:
            List of figure paths (*.png, *.pdf, *.jpg).
        """
        figures = []
        for ext in ["png", "pdf", "jpg", "jpeg"]:
            figures.extend(workspace.glob(f"*.{ext}"))
        return sorted(figures)
