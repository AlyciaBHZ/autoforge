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


def fetch_iclr_papers(
    *,
    year: int,
    limit: int = 1200,
    timeout_seconds: int = 25,
) -> list[PaperRecord]:
    """Fetch ICLR papers from OpenReview."""
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

    return papers[:limit]


def infer_papers_from_goal(
    goal: str,
    papers: list[PaperRecord],
    *,
    top_k: int = 5,
) -> list[InferenceResult]:
    """Rank candidate papers from a goal string."""
    goal_tokens = _tokens(goal)
    goal_set = set(goal_tokens)
    if not goal_set:
        return []

    results: list[InferenceResult] = []
    for p in papers:
        title_tokens = _tokens(p.title)
        abs_tokens = _tokens(p.abstract)
        kw_tokens = _tokens(" ".join(p.keywords))

        title_set = set(title_tokens)
        abs_set = set(abs_tokens)
        kw_set = set(kw_tokens)

        # Weighted overlap score with a small phrase bonus.
        title_overlap = len(goal_set & title_set)
        abs_overlap = len(goal_set & abs_set)
        kw_overlap = len(goal_set & kw_set)
        phrase_bonus = 0.0
        goal_norm = _normalize(goal)
        if goal_norm and goal_norm in _normalize(p.abstract):
            phrase_bonus = 2.0

        score = (
            3.0 * title_overlap
            + 1.5 * kw_overlap
            + 1.0 * abs_overlap
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
    # Keep the first sentence-like fragment as a "goal only" prompt.
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    first = parts[0].strip()
    if len(first) > 220:
        first = first[:220].rsplit(" ", 1)[0].strip()
    return first


def _extract_hits(text: str, candidates: tuple[str, ...]) -> list[str]:
    norm = _normalize(text)
    hits = [c for c in candidates if c in norm]
    # Stable unique list
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out


def build_reproduction_brief(goal: str, paper: PaperRecord) -> str:
    """Create a structured brief for downstream coding agents."""
    datasets = _extract_hits(f"{paper.title} {paper.abstract}", _COMMON_DATASETS)
    metrics = _extract_hits(f"{paper.title} {paper.abstract}", _COMMON_METRICS)
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
    return "\n".join(lines)


def build_generation_prompt(goal: str, paper: PaperRecord) -> str:
    """Build a ready-to-run AutoForge generation prompt."""
    brief = build_reproduction_brief(goal, paper)
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
    corpus_size: int = 1200,
    top_k: int = 5,
    seed: int = 42,
) -> BenchmarkReport:
    """Benchmark paper inference by masking papers into short goal prompts."""
    papers = fetch_iclr_papers(year=year, limit=corpus_size)
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
        "Current inference is lexical overlap based; it misses semantic paraphrases.",
        "No method-level signature matching (equations, architecture motifs, loss forms).",
        "No dataset/metric parser from full PDF tables, only abstract-level hints.",
        "No automatic environment recreation (exact CUDA/library versions) yet.",
        "No closed-loop verifier to compare reproduced curves against paper figures.",
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
