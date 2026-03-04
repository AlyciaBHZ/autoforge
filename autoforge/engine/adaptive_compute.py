"""Adaptive Test-Time Compute — difficulty-aware resource allocation.

Inspired by:
  - "Scaling LLM Test-Time Compute Optimally" (ICLR 2025)
  - Inference Scaling Laws (ICLR 2025, Snell et al.)

Key insight: allocating test-time compute *adaptively per prompt* improves
efficiency by 4× over uniform best-of-N.  Easy tasks need minimal compute;
hard tasks benefit from deeper search, more retries, and stronger models.

This module:
  1. Estimates task difficulty from the spec/description
  2. Routes tasks to appropriate compute profiles (model, retries, search depth)
  3. Tracks actual difficulty vs predicted → self-calibrates over time

Compute profiles:
  - TRIVIAL:  Haiku/fast model, 1 attempt, no search
  - STANDARD: Sonnet, 2 attempts, shallow search
  - COMPLEX:  Sonnet+, 3 attempts, MCTS search
  - EXTREME:  Opus, 5 attempts, deep MCTS + debate
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DifficultyLevel(str, Enum):
    TRIVIAL = "trivial"
    STANDARD = "standard"
    COMPLEX = "complex"
    EXTREME = "extreme"


class ReasoningType(str, Enum):
    """Task-specific reasoning strategies inspired by SciAgent (2025).

    Different reasoning tasks benefit from specialized worker systems.
    """
    SYMBOLIC = "symbolic"      # Formal logic, proofs, type systems
    NUMERICAL = "numerical"    # Math computation, optimization, algorithms
    CONCEPTUAL = "conceptual"  # Architecture design, API design, high-level planning
    GENERATIVE = "generative"  # Code generation, content creation
    VERIFICATION = "verification"  # Testing, reviewing, debugging


@dataclass
class ComputeProfile:
    """Resource allocation for a difficulty level."""

    difficulty: DifficultyLevel
    max_retries: int
    mcts_iterations: int
    search_candidates: int
    use_debate: bool
    preferred_complexity: str  # Maps to TaskComplexity for LLM routing
    builder_max_turns: int
    enable_prm: bool  # Process Reward Model
    sub_agent_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "difficulty": self.difficulty.value,
            "max_retries": self.max_retries,
            "mcts_iterations": self.mcts_iterations,
            "search_candidates": self.search_candidates,
            "use_debate": self.use_debate,
            "preferred_complexity": self.preferred_complexity,
            "builder_max_turns": self.builder_max_turns,
            "enable_prm": self.enable_prm,
            "sub_agent_hints": self.sub_agent_hints,
        }


# Default profiles
PROFILES: dict[DifficultyLevel, ComputeProfile] = {
    DifficultyLevel.TRIVIAL: ComputeProfile(
        difficulty=DifficultyLevel.TRIVIAL,
        max_retries=1,
        mcts_iterations=0,
        search_candidates=1,
        use_debate=False,
        preferred_complexity="quick",
        builder_max_turns=15,
        enable_prm=False,
    ),
    DifficultyLevel.STANDARD: ComputeProfile(
        difficulty=DifficultyLevel.STANDARD,
        max_retries=2,
        mcts_iterations=3,
        search_candidates=2,
        use_debate=False,
        preferred_complexity="standard",
        builder_max_turns=25,
        enable_prm=True,
    ),
    DifficultyLevel.COMPLEX: ComputeProfile(
        difficulty=DifficultyLevel.COMPLEX,
        max_retries=3,
        mcts_iterations=7,
        search_candidates=3,
        use_debate=True,
        preferred_complexity="complex",
        builder_max_turns=35,
        enable_prm=True,
    ),
    DifficultyLevel.EXTREME: ComputeProfile(
        difficulty=DifficultyLevel.EXTREME,
        max_retries=5,
        mcts_iterations=12,
        search_candidates=5,
        use_debate=True,
        preferred_complexity="complex",
        builder_max_turns=50,
        enable_prm=True,
    ),
}

# Sub-agent strategy hints by (difficulty, reasoning_type)
# Maps to specialized worker system configurations (SciAgent-style dispatch)
SUB_AGENT_HINTS: dict[tuple[DifficultyLevel, ReasoningType], dict[str, Any]] = {
    # VERIFICATION strategies
    (DifficultyLevel.TRIVIAL, ReasoningType.VERIFICATION): {
        "run_security_scan": False,
        "ldb_enabled": False,
        "debate_rounds": 0,
    },
    (DifficultyLevel.STANDARD, ReasoningType.VERIFICATION): {
        "run_security_scan": False,
        "ldb_enabled": False,
        "debate_rounds": 1,
    },
    (DifficultyLevel.COMPLEX, ReasoningType.VERIFICATION): {
        "run_security_scan": True,
        "ldb_enabled": True,
        "debate_rounds": 2,
        "enable_reflexion": True,
    },
    (DifficultyLevel.EXTREME, ReasoningType.VERIFICATION): {
        "run_security_scan": True,
        "ldb_enabled": True,
        "debate_rounds": 3,
        "enable_reflexion": True,
        "formal_verify": True,
    },
    # SYMBOLIC strategies (formal logic, proofs, type systems)
    (DifficultyLevel.TRIVIAL, ReasoningType.SYMBOLIC): {
        "use_lean": False,
        "debate_rounds": 0,
    },
    (DifficultyLevel.STANDARD, ReasoningType.SYMBOLIC): {
        "use_lean": False,
        "debate_rounds": 1,
    },
    (DifficultyLevel.COMPLEX, ReasoningType.SYMBOLIC): {
        "use_lean": True,
        "debate_rounds": 2,
        "enable_theorem_proving": True,
    },
    (DifficultyLevel.EXTREME, ReasoningType.SYMBOLIC): {
        "use_lean": True,
        "debate_rounds": 3,
        "enable_theorem_proving": True,
        "mcts_strategy": "proof_search",
    },
    # NUMERICAL strategies (math, optimization, algorithms)
    (DifficultyLevel.TRIVIAL, ReasoningType.NUMERICAL): {
        "enable_symbolic_compute": False,
        "mcts_iterations": 0,
    },
    (DifficultyLevel.STANDARD, ReasoningType.NUMERICAL): {
        "enable_symbolic_compute": False,
        "mcts_iterations": 3,
    },
    (DifficultyLevel.COMPLEX, ReasoningType.NUMERICAL): {
        "enable_symbolic_compute": True,
        "mcts_iterations": 7,
        "algorithm_synthesis": True,
    },
    (DifficultyLevel.EXTREME, ReasoningType.NUMERICAL): {
        "enable_symbolic_compute": True,
        "mcts_iterations": 12,
        "algorithm_synthesis": True,
        "performance_profiling": True,
    },
    # CONCEPTUAL strategies (architecture, design, planning)
    (DifficultyLevel.TRIVIAL, ReasoningType.CONCEPTUAL): {
        "debate_rounds": 0,
        "hierarchical_decomp": False,
    },
    (DifficultyLevel.STANDARD, ReasoningType.CONCEPTUAL): {
        "debate_rounds": 1,
        "hierarchical_decomp": True,
    },
    (DifficultyLevel.COMPLEX, ReasoningType.CONCEPTUAL): {
        "debate_rounds": 2,
        "hierarchical_decomp": True,
        "enable_capability_dag": True,
    },
    (DifficultyLevel.EXTREME, ReasoningType.CONCEPTUAL): {
        "debate_rounds": 3,
        "hierarchical_decomp": True,
        "enable_capability_dag": True,
        "theoretical_reasoning": True,
    },
    # GENERATIVE strategies (code generation, content creation)
    (DifficultyLevel.TRIVIAL, ReasoningType.GENERATIVE): {
        "max_retries": 1,
        "mcts_iterations": 0,
    },
    (DifficultyLevel.STANDARD, ReasoningType.GENERATIVE): {
        "max_retries": 2,
        "mcts_iterations": 3,
    },
    (DifficultyLevel.COMPLEX, ReasoningType.GENERATIVE): {
        "max_retries": 3,
        "mcts_iterations": 7,
        "enable_evomac": True,
    },
    (DifficultyLevel.EXTREME, ReasoningType.GENERATIVE): {
        "max_retries": 5,
        "mcts_iterations": 12,
        "enable_evomac": True,
        "enable_sica": True,
    },
}

# ──────────────────────────────────────────────
# Difficulty estimation heuristics
# ──────────────────────────────────────────────

# Indicators of complexity in task descriptions
COMPLEXITY_SIGNALS: dict[str, float] = {
    # High complexity signals
    "authentication": 0.3,
    "oauth": 0.4,
    "database": 0.2,
    "migration": 0.3,
    "real-time": 0.3,
    "websocket": 0.3,
    "concurrency": 0.4,
    "parallel": 0.2,
    "distributed": 0.5,
    "microservice": 0.4,
    "encryption": 0.3,
    "machine learning": 0.4,
    "neural": 0.4,
    "payment": 0.4,
    "streaming": 0.3,
    "graph": 0.2,
    "recursive": 0.2,
    "optimization": 0.3,
    "compiler": 0.5,
    "parser": 0.3,
    # Low complexity signals (negative = easier)
    "hello world": -0.3,
    "todo": -0.1,
    "calculator": -0.1,
    "static": -0.2,
    "landing page": -0.2,
    "readme": -0.3,
}

# Reasoning type indicators (keywords → reasoning strategy)
REASONING_TYPE_SIGNALS: dict[ReasoningType, list[str]] = {
    ReasoningType.VERIFICATION: [
        "test", "verify", "review", "debug", "validation", "check",
        "assert", "audit", "security", "bug", "error", "fail"
    ],
    ReasoningType.SYMBOLIC: [
        "proof", "theorem", "type", "formal", "logic", "constraint",
        "invariant", "precondition", "postcondition", "hoare"
    ],
    ReasoningType.NUMERICAL: [
        "algorithm", "optimize", "compute", "solve", "calculate", "math",
        "numerical", "matrix", "tensor", "performance", "speed"
    ],
    ReasoningType.CONCEPTUAL: [
        "design", "architect", "plan", "structure", "organize", "pattern",
        "api", "schema", "interface", "framework", "refactor"
    ],
    # GENERATIVE is default for code generation, content creation
}


@dataclass
class DifficultyEstimate:
    """Result of difficulty estimation."""

    level: DifficultyLevel
    score: float  # 0.0 = trivial, 1.0 = extreme
    confidence: float  # How confident we are in this estimate
    signals: list[str]  # What drove the estimate
    reasoning_type: ReasoningType = ReasoningType.GENERATIVE
    profile: ComputeProfile = field(default_factory=lambda: PROFILES[DifficultyLevel.STANDARD])

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "signals": self.signals,
            "reasoning_type": self.reasoning_type.value,
            "profile": self.profile.to_dict(),
        }


@dataclass
class CalibrationRecord:
    """Records predicted vs actual difficulty for self-calibration."""

    task_id: str
    predicted: DifficultyLevel
    predicted_score: float
    actual_retries: int = 0
    actual_success: bool = False
    actual_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


class AdaptiveComputeRouter:
    """Routes tasks to compute profiles based on estimated difficulty.

    Self-calibrates over time by tracking predicted vs actual outcomes.
    """

    # Thresholds for mapping score → difficulty level
    TRIVIAL_THRESHOLD = 0.2
    STANDARD_THRESHOLD = 0.45
    COMPLEX_THRESHOLD = 0.7

    def __init__(self) -> None:
        self._calibration: list[CalibrationRecord] = []
        self._bias: float = 0.0  # Learned bias from calibration
        self._profiles = dict(PROFILES)

    # ── Reasoning type classification ──────────

    def _classify_reasoning_type(self, task_description: str) -> ReasoningType:
        """Classify reasoning type using keyword signals.

        Matches task keywords against reasoning-type-specific indicators:
          - VERIFICATION: test, verify, review, debug, validation
          - SYMBOLIC: proof, theorem, type check, formal logic
          - NUMERICAL: algorithm, optimize, compute, numerical
          - CONCEPTUAL: design, architect, plan, api, schema
          - GENERATIVE: default (code generation, content creation)

        Args:
            task_description: Task or requirement description

        Returns:
            ReasoningType classification
        """
        desc_lower = task_description.lower()
        scores: dict[ReasoningType, int] = {t: 0 for t in ReasoningType}

        # Score each reasoning type by keyword matches
        for reasoning_type, keywords in REASONING_TYPE_SIGNALS.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    scores[reasoning_type] += 1

        # Find highest scoring type (or GENERATIVE if tied/no matches)
        max_score = max(scores.values()) if scores else 0
        if max_score == 0:
            return ReasoningType.GENERATIVE

        # Return first reasoning type with max score (except GENERATIVE preference)
        for reasoning_type in [
            ReasoningType.VERIFICATION,
            ReasoningType.SYMBOLIC,
            ReasoningType.NUMERICAL,
            ReasoningType.CONCEPTUAL,
        ]:
            if scores[reasoning_type] == max_score:
                return reasoning_type

        return ReasoningType.GENERATIVE

    # ── Core API ─────────────────────────────────

    def estimate_difficulty(
        self,
        task_description: str,
        spec: dict[str, Any] | None = None,
        file_count: int = 0,
    ) -> DifficultyEstimate:
        """Estimate task difficulty using heuristics.

        Uses:
          - Keyword signals in task description
          - Module/file count from spec
          - Architecture complexity indicators
          - Self-calibration bias from past runs
          - Reasoning type classification for sub-agent routing

        Returns:
            DifficultyEstimate with difficulty level and reasoning type
        """
        score = 0.5  # Baseline: standard
        signals: list[str] = []
        desc_lower = task_description.lower()

        # 1. Keyword signals
        for keyword, weight in COMPLEXITY_SIGNALS.items():
            if keyword in desc_lower:
                score += weight
                signals.append(f"keyword:{keyword}({weight:+.1f})")

        # 2. Spec-based signals
        if spec:
            modules = spec.get("modules", [])
            n_modules = len(modules)
            if n_modules > 8:
                score += 0.2
                signals.append(f"modules:{n_modules}(+0.2)")
            elif n_modules <= 2:
                score -= 0.1
                signals.append(f"modules:{n_modules}(-0.1)")

            # Tech stack complexity
            tech = spec.get("tech_stack", {})
            if isinstance(tech, dict) and len(tech) > 5:
                score += 0.15
                signals.append(f"tech_stack_size:{len(tech)}(+0.15)")

        # 3. File count (check larger threshold first)
        if file_count > 50:
            score += 0.3
            signals.append(f"files:{file_count}(+0.3)")
        elif file_count > 20:
            score += 0.15
            signals.append(f"files:{file_count}(+0.15)")

        # 4. Description length (longer descriptions often mean more complex tasks)
        desc_len = len(task_description)
        if desc_len > 500:
            score += 0.1
            signals.append(f"desc_length:{desc_len}(+0.1)")

        # 5. Apply calibration bias
        score += self._bias
        if abs(self._bias) > 0.01:
            signals.append(f"calibration_bias({self._bias:+.2f})")

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        # Map to difficulty level
        if score < self.TRIVIAL_THRESHOLD:
            level = DifficultyLevel.TRIVIAL
        elif score < self.STANDARD_THRESHOLD:
            level = DifficultyLevel.STANDARD
        elif score < self.COMPLEX_THRESHOLD:
            level = DifficultyLevel.COMPLEX
        else:
            level = DifficultyLevel.EXTREME

        profile = self._profiles[level]

        # Classify reasoning type for sub-agent routing
        reasoning_type = self._classify_reasoning_type(task_description)
        signals.append(f"reasoning_type:{reasoning_type.value}")

        # Confidence: higher when we have more calibration data
        n_cal = len(self._calibration)
        confidence = min(0.9, 0.5 + 0.04 * n_cal)

        return DifficultyEstimate(
            level=level,
            score=score,
            confidence=confidence,
            signals=signals,
            reasoning_type=reasoning_type,
            profile=profile,
        )

    async def estimate_with_llm(
        self,
        task_description: str,
        llm: Any,
        spec: dict[str, Any] | None = None,
    ) -> DifficultyEstimate:
        """Use LLM to estimate difficulty (more accurate but costs tokens).

        Falls back to heuristic if LLM call fails.
        Maintains reasoning type classification from heuristic estimate.
        """
        heuristic = self.estimate_difficulty(task_description, spec)

        try:
            prompt = f"""\
Rate the implementation difficulty of this coding task on a scale from 0.0 to 1.0:

Task: {task_description[:500]}

Consider: number of components, external APIs, algorithmic complexity,
error handling needs, testing requirements.

Reply with ONLY a JSON object: {{"score": 0.X, "reason": "brief reason"}}"""

            response = await llm.query(
                system="You are a software estimation expert. Be concise.",
                messages=[{"role": "user", "content": prompt}],
                complexity="quick",
            )

            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            # Parse LLM response
            text = text.strip()
            if "{" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]
                data = json.loads(json_str)
                llm_score = float(data.get("score", 0.5))

                # Blend: 40% heuristic + 60% LLM
                blended = 0.4 * heuristic.score + 0.6 * llm_score + self._bias
                blended = max(0.0, min(1.0, blended))

                heuristic.score = blended
                heuristic.signals.append(f"llm_estimate({llm_score:.2f})")
                heuristic.confidence = min(0.95, heuristic.confidence + 0.15)

                # Re-map level
                if blended < self.TRIVIAL_THRESHOLD:
                    heuristic.level = DifficultyLevel.TRIVIAL
                elif blended < self.STANDARD_THRESHOLD:
                    heuristic.level = DifficultyLevel.STANDARD
                elif blended < self.COMPLEX_THRESHOLD:
                    heuristic.level = DifficultyLevel.COMPLEX
                else:
                    heuristic.level = DifficultyLevel.EXTREME

                heuristic.profile = self._profiles[heuristic.level]

        except Exception as e:
            logger.debug(f"LLM difficulty estimation failed, using heuristic: {e}")

        return heuristic

    # ── Sub-agent configuration ──────────────────

    def get_sub_agent_config(self, estimate: DifficultyEstimate) -> dict[str, Any]:
        """Get merged compute profile + sub-agent hints for specialized routing.

        Implements SciAgent-style hierarchical dispatch: routes tasks to
        specialized worker systems based on both difficulty and reasoning type.

        Args:
            estimate: DifficultyEstimate with level and reasoning_type

        Returns:
            Merged dict with base ComputeProfile fields + reasoning-specific hints
        """
        # Start with base profile
        config = estimate.profile.to_dict().copy()

        # Look up reasoning-type-specific hints
        hint_key = (estimate.level, estimate.reasoning_type)
        if hint_key in SUB_AGENT_HINTS:
            hints = SUB_AGENT_HINTS[hint_key]
            config.setdefault("sub_agent_hints", {}).update(hints)
            logger.debug(
                f"[AdaptiveCompute] Sub-agent config: "
                f"({estimate.level.value}, {estimate.reasoning_type.value}) "
                f"→ {len(hints)} hints"
            )
        else:
            logger.debug(
                f"[AdaptiveCompute] No sub-agent hints for "
                f"({estimate.level.value}, {estimate.reasoning_type.value})"
            )

        return config

    # ── Calibration ──────────────────────────────

    def record_outcome(
        self,
        task_id: str,
        predicted: DifficultyEstimate,
        actual_retries: int,
        success: bool,
        elapsed: float,
    ) -> None:
        """Record actual outcome for calibration."""
        record = CalibrationRecord(
            task_id=task_id,
            predicted=predicted.level,
            predicted_score=predicted.score,
            actual_retries=actual_retries,
            actual_success=success,
            actual_time=elapsed,
        )
        self._calibration.append(record)

        # Re-calibrate after every 5 records
        if len(self._calibration) % 5 == 0:
            self._recalibrate()

    def _recalibrate(self) -> None:
        """Adjust bias based on prediction errors.

        If we consistently under-predict difficulty (tasks need more retries
        than expected), increase the bias. Vice versa.
        """
        if len(self._calibration) < 3:
            return

        recent = self._calibration[-20:]  # Use last 20 records
        errors: list[float] = []

        for rec in recent:
            # Infer actual difficulty from outcome
            if not rec.actual_success:
                actual_score = 0.9  # Failed = was hard
            elif rec.actual_retries == 0:
                actual_score = 0.2  # No retries = was easy
            elif rec.actual_retries <= 1:
                actual_score = 0.4
            elif rec.actual_retries <= 3:
                actual_score = 0.65
            else:
                actual_score = 0.85

            errors.append(actual_score - rec.predicted_score)

        # Mean error = how much we under/over-predict
        mean_error = sum(errors) / len(errors)

        # Apply exponential moving average to bias
        self._bias = 0.7 * self._bias + 0.3 * mean_error
        self._bias = max(-0.3, min(0.3, self._bias))  # Clamp

        logger.info(
            f"[AdaptiveCompute] Recalibrated: bias={self._bias:.3f} "
            f"(from {len(recent)} records, mean_error={mean_error:.3f})"
        )

    # ── Persistence ──────────────────────────────

    def save_state(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "bias": self._bias,
            "calibration": [
                {
                    "task_id": r.task_id,
                    "predicted": r.predicted.value,
                    "predicted_score": r.predicted_score,
                    "actual_retries": r.actual_retries,
                    "actual_success": r.actual_success,
                    "actual_time": r.actual_time,
                    "timestamp": r.timestamp,
                }
                for r in self._calibration[-100:]  # Keep last 100
            ],
        }
        (output_dir / "adaptive_compute_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8",
        )

    def load_state(self, state_dir: Path) -> None:
        path = state_dir / "adaptive_compute_state.json"
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            self._bias = state.get("bias", 0.0)
            self._calibration = [
                CalibrationRecord(
                    task_id=r["task_id"],
                    predicted=DifficultyLevel(r["predicted"]),
                    predicted_score=r["predicted_score"],
                    actual_retries=r.get("actual_retries", 0),
                    actual_success=r.get("actual_success", False),
                    actual_time=r.get("actual_time", 0),
                    timestamp=r.get("timestamp", 0),
                )
                for r in state.get("calibration", [])
            ]
            logger.info(
                f"[AdaptiveCompute] Loaded state: bias={self._bias:.3f}, "
                f"{len(self._calibration)} calibration records"
            )
        except Exception as e:
            logger.warning(f"[AdaptiveCompute] Failed to load state: {e}")

    def get_stats(self) -> dict[str, Any]:
        if not self._calibration:
            return {"total_tasks": 0, "bias": self._bias}

        by_level: dict[str, list[CalibrationRecord]] = {}
        for r in self._calibration:
            by_level.setdefault(r.predicted.value, []).append(r)

        return {
            "total_tasks": len(self._calibration),
            "bias": round(self._bias, 4),
            "by_level": {
                level: {
                    "count": len(records),
                    "success_rate": sum(1 for r in records if r.actual_success) / len(records),
                    "avg_retries": sum(r.actual_retries for r in records) / len(records),
                }
                for level, records in by_level.items()
            },
        }
