"""Harness reporting + aggregation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def classify_error(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return ""
    if "budget exceeded" in t or "budget exhausted" in t:
        return "budget"
    if "timed out" in t or "timeout" in t:
        return "timeout"
    if "module not found" in t or "cannot find module" in t:
        return "missing_dependency"
    if "syntaxerror" in t:
        return "syntax"
    if "json" in t and "decode" in t:
        return "json_parse"
    return "runtime_error"


@dataclass
class CaseMetrics:
    case_id: str
    mode: str
    ok: bool
    duration_seconds: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    project_dir: str = ""
    error: str = ""
    error_type: str = ""
    visible_tests_ok: bool | None = None
    hidden_tests_ok: bool | None = None
    golden_patch_similarity: float | None = None
    diff_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mode": self.mode,
            "ok": self.ok,
            "duration_seconds": self.duration_seconds,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "project_dir": self.project_dir,
            "error": self.error,
            "error_type": self.error_type,
            "visible_tests_ok": self.visible_tests_ok,
            "hidden_tests_ok": self.hidden_tests_ok,
            "golden_patch_similarity": self.golden_patch_similarity,
            "diff_counts": self.diff_counts,
        }


@dataclass
class HarnessReport:
    run_id: str
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    cases: list[CaseMetrics] = field(default_factory=list)

    def finalize(self) -> None:
        self.completed_at = time.time()

    def summary(self) -> dict[str, Any]:
        total = len(self.cases)
        ok = sum(1 for c in self.cases if c.ok)
        visible_ok = sum(1 for c in self.cases if c.visible_tests_ok)
        hidden_ok = sum(1 for c in self.cases if c.hidden_tests_ok)
        avg_cost = (sum(c.cost_usd for c in self.cases) / total) if total else 0.0
        avg_time = (sum(c.duration_seconds for c in self.cases) / total) if total else 0.0
        error_types: dict[str, int] = {}
        for c in self.cases:
            if c.error_type:
                error_types[c.error_type] = error_types.get(c.error_type, 0) + 1
        return {
            "run_id": self.run_id,
            "total_cases": total,
            "ok_cases": ok,
            "pass_rate": (ok / total) if total else 0.0,
            "visible_test_pass": visible_ok,
            "hidden_test_pass": hidden_ok,
            "avg_cost_usd": avg_cost,
            "avg_duration_seconds": avg_time,
            "error_types": dict(
                sorted(error_types.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "started_at": self.started_at,
            "completed_at": self.completed_at or None,
            "duration_seconds": (self.completed_at - self.started_at) if self.completed_at else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "cases": [c.to_dict() for c in self.cases],
        }

