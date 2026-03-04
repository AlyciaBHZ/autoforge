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
        {"name": "python", "version": "3.11"},
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
        "python": "3.11",
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
