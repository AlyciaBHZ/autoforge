"""Prompt Optimizer — DSPy/OPRO-inspired automatic prompt self-improvement.

Instead of hand-crafted, static constitution files, this module enables
AutoForge agents to automatically optimize their system prompts over time.

The optimization loop:
  1. **Baseline**: Start with the current constitution prompt
  2. **Evaluate**: Run a project, measure fitness (quality, test pass, cost)
  3. **Propose**: Ask the LLM to suggest prompt improvements based on results
  4. **Select**: Keep improvements that increase fitness, discard regressions
  5. **Persist**: Save the optimized prompt variants to disk

This combines ideas from:
  - DSPy (Stanford NLP): Programmatic prompt optimization with MIPROv2
  - OPRO (DeepMind, ICLR 2024): LLM-as-optimizer for prompt engineering
  - SIMBA: Stochastic identification of hard examples + self-reflection
  - AMPO: Multi-branch prompt optimization with conditional logic

Key design decisions:
  - We optimize the *supplementary* prompt (dynamic constitution), not the
    base constitution. This preserves the human-crafted core while improving
    project-specific instructions.
  - Optimization is "lazy" — it only triggers after N runs with enough data.
  - Changes are conservative: we keep a history and can rollback.

Persistence: ~/.autoforge/prompt_optimization/
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────


@dataclass
class PromptVariant:
    """A single prompt variant being tested or in production."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    role: str = ""                    # Agent role: "builder", "architect", etc.
    content: str = ""                 # The prompt text
    source: str = "baseline"          # "baseline", "optimized", "mutated"
    parent_id: str | None = None      # Which variant this evolved from
    generation: int = 0
    # Performance tracking
    times_used: int = 0
    total_fitness: float = 0.0
    fitness_sq_sum: float = 0.0       # Sum of squared fitness (for variance)
    best_fitness: float = 0.0
    worst_fitness: float = float("inf")
    # Metadata
    created_at: float = field(default_factory=time.time)
    optimization_notes: str = ""      # LLM's reasoning for changes

    @property
    def avg_fitness(self) -> float:
        if self.times_used == 0:
            return 0.0
        return self.total_fitness / self.times_used

    @property
    def fitness_variance(self) -> float:
        """Compute sample variance of fitness observations."""
        if self.times_used < 2:
            return 0.25  # High-uncertainty prior for untested/single-test
        mean = self.avg_fitness
        # Var = E[X²] - E[X]²  (Welford-safe with Bessel correction)
        return max(0.0, (self.fitness_sq_sum / self.times_used - mean * mean)
                   * self.times_used / (self.times_used - 1))

    def record_fitness(self, fitness: float) -> None:
        """Record a fitness observation."""
        self.times_used += 1
        self.total_fitness += fitness
        self.fitness_sq_sum += fitness * fitness
        self.best_fitness = max(self.best_fitness, fitness)
        if self.worst_fitness == float("inf"):
            self.worst_fitness = fitness
        else:
            self.worst_fitness = min(self.worst_fitness, fitness)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "source": self.source,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "times_used": self.times_used,
            "total_fitness": self.total_fitness,
            "fitness_sq_sum": self.fitness_sq_sum,
            "best_fitness": self.best_fitness,
            "worst_fitness": self.worst_fitness if self.worst_fitness != float("inf") else 0.0,
            "created_at": self.created_at,
            "optimization_notes": self.optimization_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptVariant:
        return cls(
            id=data.get("id", uuid.uuid4().hex[:8]),
            role=data.get("role", ""),
            content=data.get("content", ""),
            source=data.get("source", "baseline"),
            parent_id=data.get("parent_id"),
            generation=data.get("generation", 0),
            times_used=data.get("times_used", 0),
            total_fitness=data.get("total_fitness", 0.0),
            fitness_sq_sum=data.get("fitness_sq_sum", 0.0),
            best_fitness=data.get("best_fitness", 0.0),
            worst_fitness=data.get("worst_fitness", float("inf")),
            created_at=data.get("created_at", time.time()),
            optimization_notes=data.get("optimization_notes", ""),
        )


@dataclass
class OptimizationRound:
    """Record of one optimization attempt."""
    round_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    role: str = ""
    baseline_id: str = ""
    candidate_id: str = ""
    baseline_fitness: float = 0.0
    candidate_fitness: float = 0.0
    accepted: bool = False
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "role": self.role,
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "baseline_fitness": self.baseline_fitness,
            "candidate_fitness": self.candidate_fitness,
            "accepted": self.accepted,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationRound:
        """Create from a dict, tolerant of extra or missing keys."""
        return cls(
            round_id=data.get("round_id", uuid.uuid4().hex[:8]),
            role=data.get("role", ""),
            baseline_id=data.get("baseline_id", ""),
            candidate_id=data.get("candidate_id", ""),
            baseline_fitness=data.get("baseline_fitness", 0.0),
            candidate_fitness=data.get("candidate_fitness", 0.0),
            accepted=data.get("accepted", False),
            reason=data.get("reason", ""),
            timestamp=data.get("timestamp", time.time()),
        )


# ──────────────────────────────────────────────
# Prompt Optimizer Engine
# ──────────────────────────────────────────────


class PromptOptimizer:
    """Automatic prompt optimization engine for agent constitutions.

    Strategy:
        1. Maintains a "population" of prompt variants per role
        2. After each project run, records fitness for the active variant
        3. When enough data is collected, proposes optimized variants
        4. Uses Thompson Sampling to balance exploration vs exploitation
        5. Persists everything to disk for cross-session learning

    Usage in orchestrator:
        optimizer = PromptOptimizer()
        # Before project: get the best prompt for each role
        prompt = optimizer.get_active_prompt("builder")
        # After project: record how well it worked
        optimizer.record_result("builder", variant_id, fitness)
        # Periodically: run optimization
        await optimizer.optimize_role("builder", llm)
    """

    _OPT_DIR = "prompt_optimization"
    _MIN_RUNS_FOR_OPTIMIZATION = 3  # Need at least N runs before optimizing
    _MAX_VARIANTS_PER_ROLE = 5      # Keep top N variants per role
    _SIGNIFICANCE_THRESHOLD = 0.10  # Welch's t-test p-value to accept improvement

    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            base_dir = Path.home() / ".autoforge"
        self.opt_dir = base_dir / self._OPT_DIR
        self.opt_dir.mkdir(parents=True, exist_ok=True)

        # Per-role variant populations
        self._variants: dict[str, list[PromptVariant]] = {}
        # Active variant per role (currently in use)
        self._active: dict[str, str] = {}
        # Optimization history
        self._history: list[OptimizationRound] = []

        self._load_state()

    # ──────── Variant Selection (Thompson Sampling) ────────

    def get_active_prompt(self, role: str) -> tuple[str, str]:
        """Get the best prompt variant for a role.

        Uses proper Thompson Sampling with Beta distribution:
        - Model each variant's fitness as a Beta(alpha, beta) distribution
        - alpha = successes (sum of fitness) + 1 (prior)
        - beta = failures (sum of 1-fitness) + 1 (prior)
        - Draw a sample from each variant's Beta distribution
        - Select the variant with the highest sample

        This naturally balances exploration vs exploitation:
        - Untested variants have wide distributions (high exploration)
        - Well-tested high-fitness variants have narrow, high distributions
        - Well-tested low-fitness variants have narrow, low distributions

        Returns: (variant_id, prompt_content)
        """
        import random

        variants = self._variants.get(role, [])
        if not variants:
            return ("", "")

        best_score = -1.0
        best_variant = variants[0]

        for v in variants:
            if v.times_used == 0:
                # Uninformative Beta(1, 1) = Uniform(0, 1) prior
                score = random.betavariate(1.0, 1.0)
            else:
                # Beta distribution parameterized from observed fitness
                # fitness is in [0, 1], so alpha ~ sum of successes, beta ~ sum of failures
                alpha = v.total_fitness + 1.0          # sum of fitness scores + prior
                beta_param = (v.times_used - v.total_fitness) + 1.0  # sum of (1-fitness) + prior
                # Clamp to valid Beta parameters (must be > 0)
                alpha = max(alpha, 0.01)
                beta_param = max(beta_param, 0.01)
                score = random.betavariate(alpha, beta_param)

            if score > best_score:
                best_score = score
                best_variant = v

        self._active[role] = best_variant.id
        self._save_state()
        return (best_variant.id, best_variant.content)

    def get_active_variant_id(self, role: str) -> str:
        """Get the ID of the currently active variant for a role."""
        return self._active.get(role, "")

    # ──────── Registering Baselines ────────

    def register_baseline(self, role: str, content: str) -> PromptVariant:
        """Register a baseline prompt variant (from static constitution).

        Called once when the system starts and the static constitutions are loaded.
        Only registers if no variants exist for this role.
        """
        if role in self._variants and self._variants[role]:
            # Already have variants — don't overwrite
            return self._variants[role][0]

        variant = PromptVariant(
            role=role,
            content=content,
            source="baseline",
            generation=0,
        )
        self._variants.setdefault(role, []).append(variant)
        self._active[role] = variant.id
        self._save_state()

        logger.info(f"[PromptOpt] Registered baseline for {role}: {variant.id}")
        return variant

    # ──────── Recording Results ────────

    def record_result(self, role: str, variant_id: str, fitness: float) -> None:
        """Record fitness for a variant after a project run.

        Also checks pending optimization rounds for statistical significance —
        if a candidate variant now has enough data to confirm improvement,
        the round is marked as accepted/rejected.

        Args:
            role: Agent role ("builder", "architect", etc.)
            variant_id: Which variant was used
            fitness: Composite fitness score (0-1)
        """
        variants = self._variants.get(role, [])
        for v in variants:
            if v.id == variant_id:
                v.record_fitness(fitness)
                logger.info(
                    f"[PromptOpt] {role}/{variant_id}: fitness={fitness:.3f} "
                    f"(avg={v.avg_fitness:.3f}, n={v.times_used}, "
                    f"var={v.fitness_variance:.4f})"
                )

                # Check if any pending optimization round can be resolved
                self._resolve_pending_rounds(role)
                self._save_state()
                return

        logger.warning(f"[PromptOpt] Variant {variant_id} not found for {role}")

    def _resolve_pending_rounds(self, role: str) -> None:
        """Check pending optimization rounds for statistical significance."""
        for rnd in self._history:
            if rnd.role != role or rnd.accepted:
                continue
            if rnd.candidate_fitness > 0:
                continue  # Already resolved

            candidate = self.get_variant(role, rnd.candidate_id)
            baseline = self.get_variant(role, rnd.baseline_id)
            if not candidate or not baseline:
                continue
            if candidate.times_used < 2 or baseline.times_used < 2:
                continue  # Need more data

            p_value = self.welch_t_test(candidate, baseline)
            if p_value < self._SIGNIFICANCE_THRESHOLD:
                rnd.accepted = True
                rnd.candidate_fitness = candidate.avg_fitness
                rnd.reason += f" [ACCEPTED: p={p_value:.4f}, Δ={candidate.avg_fitness - baseline.avg_fitness:+.3f}]"
                logger.info(
                    f"[PromptOpt] Round {rnd.round_id}: candidate {rnd.candidate_id} "
                    f"significantly better (p={p_value:.4f})"
                )
            elif candidate.times_used >= 5 and candidate.avg_fitness < baseline.avg_fitness:
                # Enough data to reject
                rnd.candidate_fitness = candidate.avg_fitness
                rnd.reason += f" [REJECTED: p={p_value:.4f}, candidate underperforms]"
                logger.info(
                    f"[PromptOpt] Round {rnd.round_id}: candidate {rnd.candidate_id} "
                    f"rejected (avg={candidate.avg_fitness:.3f} < baseline {baseline.avg_fitness:.3f})"
                )

    # ──────── Optimization (OPRO-style) ────────

    def should_optimize(self, role: str) -> bool:
        """Check if we have enough data to attempt optimization."""
        variants = self._variants.get(role, [])
        if not variants:
            return False
        total_runs = sum(v.times_used for v in variants)
        return total_runs >= self._MIN_RUNS_FOR_OPTIMIZATION

    async def optimize_role(
        self,
        role: str,
        llm: Any,
        project_context: str = "",
    ) -> PromptVariant | None:
        """Run one optimization round for a role's prompt.

        OPRO-inspired: present the LLM with the current prompt + fitness data,
        ask it to propose improvements, then evaluate the candidate.

        Args:
            role: Agent role to optimize
            llm: LLMRouter instance
            project_context: Optional recent project context for grounding

        Returns:
            New PromptVariant if optimization succeeded, None otherwise.
        """
        from autoforge.engine.llm_router import TaskComplexity

        variants = self._variants.get(role, [])
        if not variants:
            return None

        # Find the current best variant
        best = max(variants, key=lambda v: v.avg_fitness if v.times_used > 0 else 0)
        if best.times_used == 0:
            return None  # No data yet

        # Build optimization prompt (OPRO meta-prompt style)
        history_lines = self._format_variant_history(role)

        prompt = (
            f"You are an expert at optimizing AI system prompts.\n\n"
            f"## Current Best Prompt for '{role}' Agent\n"
            f"```\n{best.content[:2000]}\n```\n\n"
            f"## Performance History\n"
            f"{history_lines}\n\n"
        )

        if project_context:
            prompt += f"## Recent Project Context\n{project_context[:1000]}\n\n"

        # Include recent failures/successes for SIMBA-style analysis
        low_performers = [v for v in variants if v.times_used > 0 and v.avg_fitness < 0.5]
        if low_performers:
            prompt += "## Low-Performing Variants (learn from failures)\n"
            for lp in low_performers[:2]:
                prompt += (
                    f"- Variant {lp.id} (avg={lp.avg_fitness:.2f}): "
                    f"{lp.optimization_notes or lp.content[:100]}\n"
                )
            prompt += "\n"

        prompt += (
            f"## Task\n"
            f"Propose an improved version of the system prompt for the '{role}' agent.\n"
            f"Focus on:\n"
            f"1. Clarity: Make instructions more specific and actionable\n"
            f"2. Error prevention: Add rules to avoid common failure patterns\n"
            f"3. Efficiency: Remove unnecessary instructions that waste tokens\n"
            f"4. Quality: Emphasize aspects that correlate with high fitness\n\n"
            f"Output JSON:\n"
            f'{{"improved_prompt": "the optimized prompt text", '
            f'"changes_made": "brief description of changes", '
            f'"expected_improvement": "why this should score higher"}}'
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system=(
                    "You are a prompt engineering expert. Analyze performance data "
                    "and suggest targeted improvements. Be specific and measurable."
                ),
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            from autoforge.engine.utils import extract_json_from_text
            try:
                data = extract_json_from_text(text)
            except ValueError:
                logger.warning("[PromptOpt] Could not parse optimization response")
                return None
            improved = data.get("improved_prompt", "")
            if not improved or improved == best.content:
                logger.info("[PromptOpt] No improvement proposed")
                return None

            # Create new variant
            candidate = PromptVariant(
                role=role,
                content=improved,
                source="optimized",
                parent_id=best.id,
                generation=best.generation + 1,
                optimization_notes=data.get("changes_made", ""),
            )

            # Add to population (replace worst if at capacity)
            self._add_variant(role, candidate)

            # Record the optimization round
            self._history.append(OptimizationRound(
                role=role,
                baseline_id=best.id,
                candidate_id=candidate.id,
                baseline_fitness=best.avg_fitness,
                reason=data.get("expected_improvement", ""),
            ))

            self._save_state()

            logger.info(
                f"[PromptOpt] New variant for {role}: {candidate.id} "
                f"(gen {candidate.generation}, changes: {candidate.optimization_notes[:80]})"
            )
            return candidate

        except Exception as e:
            logger.warning(f"[PromptOpt] Optimization failed for {role}: {e}")
            return None

    # ──────── Prompt Mutation (for diversity) ────────

    async def mutate_prompt(
        self,
        role: str,
        llm: Any,
        mutation_type: str = "focus_shift",
    ) -> PromptVariant | None:
        """Create a mutated variant for exploration.

        Unlike optimize_role (which tries to improve), mutation deliberately
        explores different directions for diversity (AMPO-style branching).

        Mutation types:
            - "focus_shift": Emphasize different quality aspects
            - "simplify": Remove instructions, keep essentials
            - "elaborate": Add more detail to key areas
            - "restructure": Same content, different organization
        """
        from autoforge.engine.llm_router import TaskComplexity

        variants = self._variants.get(role, [])
        if not variants:
            return None

        # Pick the current best as base
        import random
        base = random.choice(variants)  # Random for diversity

        mutation_prompts = {
            "focus_shift": (
                "Rewrite this prompt with a DIFFERENT emphasis. If it currently "
                "focuses on correctness, shift focus to efficiency. If it focuses "
                "on code style, shift to functionality. Create a meaningfully "
                "different version that explores a new quality dimension."
            ),
            "simplify": (
                "Simplify this prompt to its essential core. Remove redundant "
                "instructions and anything that doesn't directly improve output "
                "quality. Aim for 50-70% of the original length."
            ),
            "elaborate": (
                "Expand the most important instructions with concrete examples "
                "and edge cases. Add 2-3 specific 'do this, not that' pairs."
            ),
            "restructure": (
                "Reorganize this prompt. Put the most impactful instructions first. "
                "Group related instructions together. Add clear section headers."
            ),
        }

        prompt = (
            f"## Original Prompt for '{role}' Agent\n"
            f"```\n{base.content[:2000]}\n```\n\n"
            f"## Mutation Task\n"
            f"{mutation_prompts.get(mutation_type, mutation_prompts['focus_shift'])}\n\n"
            f"Output JSON:\n"
            f'{{"mutated_prompt": "the new prompt text", '
            f'"mutation_description": "what was changed and why"}}'
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You are a prompt engineering expert creating diverse variants.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            from autoforge.engine.utils import extract_json_from_text
            try:
                data = extract_json_from_text(text)
            except ValueError:
                return None
            mutated = data.get("mutated_prompt", "")
            if not mutated:
                return None

            variant = PromptVariant(
                role=role,
                content=mutated,
                source="mutated",
                parent_id=base.id,
                generation=base.generation + 1,
                optimization_notes=f"mutation({mutation_type}): {data.get('mutation_description', '')}",
            )

            self._add_variant(role, variant)
            self._save_state()

            logger.info(
                f"[PromptOpt] Mutated variant for {role}: {variant.id} "
                f"({mutation_type})"
            )
            return variant

        except Exception as e:
            logger.warning(f"[PromptOpt] Mutation failed for {role}: {e}")
            return None

    # ──────── Statistical Testing ────────

    @staticmethod
    def welch_t_test(v_a: PromptVariant, v_b: PromptVariant) -> float:
        """Welch's t-test p-value for v_a being better than v_b (one-tailed).

        Returns approximate p-value using the t-distribution.
        Lower p = stronger evidence that v_a > v_b.
        """
        n_a, n_b = v_a.times_used, v_b.times_used
        if n_a < 2 or n_b < 2:
            return 1.0  # Not enough data

        mean_a, mean_b = v_a.avg_fitness, v_b.avg_fitness
        var_a, var_b = v_a.fitness_variance, v_b.fitness_variance

        se_a = var_a / n_a
        se_b = var_b / n_b
        se_total = se_a + se_b
        if se_total < 1e-12:
            return 0.0 if mean_a > mean_b else 1.0

        t_stat = (mean_a - mean_b) / math.sqrt(se_total)

        # Welch–Satterthwaite degrees of freedom
        df_num = se_total ** 2
        df_den = (se_a ** 2 / (n_a - 1) + se_b ** 2 / (n_b - 1)) if (n_a > 1 and n_b > 1) else 1.0
        df = max(1.0, df_num / df_den) if df_den > 0 else 1.0

        # Approximate one-tailed p-value using the incomplete beta function
        # For t > 0 (candidate better), p = P(T > t) under H0
        return PromptOptimizer._t_cdf_complement(t_stat, df)

    @staticmethod
    def _t_cdf_complement(t: float, df: float) -> float:
        """Approximate 1 - CDF of t-distribution (one-tailed upper p-value).

        Uses the regularized incomplete beta function approximation.
        For practical prompt optimization, this is more than accurate enough.
        """
        if t == 0:
            return 0.5
        if t < 0:
            return 1.0 - PromptOptimizer._t_cdf_complement(-t, df)

        # Convert t-statistic to beta function form
        x = df / (df + t * t)
        # Regularized incomplete beta: I_x(df/2, 1/2)
        # Use continued fraction approximation
        a, b = df / 2.0, 0.5
        p = 0.5 * PromptOptimizer._regularized_incomplete_beta(x, a, b)
        return max(0.0, min(1.0, p))

    @staticmethod
    def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
        """Regularized incomplete beta function via Lentz continued fraction."""
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0

        # Use the series expansion for small x, continued fraction otherwise
        # This is a simplified implementation sufficient for our use case
        # Prefactor: x^a * (1-x)^b / (a * Beta(a,b))
        try:
            ln_prefactor = (a * math.log(x) + b * math.log(1 - x)
                           + math.lgamma(a + b)
                           - math.lgamma(a) - math.lgamma(b)
                           - math.log(a))
            prefactor = math.exp(ln_prefactor)
        except (ValueError, OverflowError):
            return 0.5  # Fallback

        # Lentz's continued fraction for I_x(a, b)
        cf = 1.0
        c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        cf = d

        for m in range(1, 100):
            # Even step
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            d = 1.0 + num * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + num / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            cf *= d * c

            # Odd step
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + num / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            cf *= delta

            if abs(delta - 1.0) < 1e-8:
                break

        return min(1.0, prefactor * cf * a)

    # ──────── Variant Management ────────

    def _add_variant(self, role: str, variant: PromptVariant) -> None:
        """Add a variant, evicting the worst performer if at capacity.

        Unlike the old version, baselines CAN be evicted if they have enough
        data showing they underperform (>= 5 runs AND statistically worse).
        """
        variants = self._variants.setdefault(role, [])
        variants.append(variant)

        if len(variants) > self._MAX_VARIANTS_PER_ROLE:
            # All tested variants are candidates for eviction
            candidates = [v for v in variants if v.times_used > 0]
            if candidates:
                worst = min(candidates, key=lambda v: v.avg_fitness)
                # Protect baseline unless we have strong evidence it's worse
                if worst.source == "baseline" and worst.times_used < 5:
                    # Not enough data to evict baseline — evict next worst non-baseline
                    non_base = [c for c in candidates if c.source != "baseline"]
                    if non_base:
                        worst = min(non_base, key=lambda v: v.avg_fitness)
                    else:
                        return  # Can't evict anything
                variants.remove(worst)
                logger.info(f"[PromptOpt] Evicted variant {worst.id} (avg={worst.avg_fitness:.3f})")

    def get_variant(self, role: str, variant_id: str) -> PromptVariant | None:
        """Get a specific variant."""
        for v in self._variants.get(role, []):
            if v.id == variant_id:
                return v
        return None

    def get_all_variants(self, role: str) -> list[PromptVariant]:
        """Get all variants for a role."""
        return list(self._variants.get(role, []))

    def get_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        stats: dict[str, Any] = {
            "roles": {},
            "total_optimization_rounds": len(self._history),
            "accepted_optimizations": sum(1 for h in self._history if h.accepted),
        }
        for role, variants in self._variants.items():
            tested = [v for v in variants if v.times_used > 0]
            stats["roles"][role] = {
                "total_variants": len(variants),
                "tested_variants": len(tested),
                "best_fitness": max((v.best_fitness for v in tested), default=0),
                "avg_fitness": (
                    sum(v.avg_fitness for v in tested) / len(tested)
                    if tested else 0
                ),
                "total_runs": sum(v.times_used for v in variants),
                "active_variant": self._active.get(role, ""),
            }
        return stats

    def _format_variant_history(self, role: str) -> str:
        """Format variant performance history for the optimization prompt."""
        variants = self._variants.get(role, [])
        if not variants:
            return "No history available."

        lines = []
        for v in sorted(variants, key=lambda x: x.avg_fitness, reverse=True):
            if v.times_used == 0:
                lines.append(f"- {v.id} (gen {v.generation}): untested")
            else:
                lines.append(
                    f"- {v.id} (gen {v.generation}): "
                    f"avg={v.avg_fitness:.3f} best={v.best_fitness:.3f} "
                    f"n={v.times_used} source={v.source}"
                )
                if v.optimization_notes:
                    lines.append(f"  Notes: {v.optimization_notes[:100]}")
        return "\n".join(lines)

    # ──────── Persistence ────────

    def _save_state(self) -> None:
        """Save all optimization state to disk."""
        state = {
            "variants": {
                role: [v.to_dict() for v in variants]
                for role, variants in self._variants.items()
            },
            "active": self._active,
            "history": [h.to_dict() for h in self._history[-100:]],  # Keep last 100
        }
        path = self.opt_dir / "state.json"
        try:
            path.write_text(
                json.dumps(state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[PromptOpt] Could not save state: {e}")

    def _load_state(self) -> None:
        """Load optimization state from disk."""
        path = self.opt_dir / "state.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for role, variants_data in data.get("variants", {}).items():
                self._variants[role] = [
                    PromptVariant.from_dict(v) for v in variants_data
                ]
            self._active = data.get("active", {})
            self._history = [
                OptimizationRound.from_dict(h) for h in data.get("history", [])
            ]
            logger.info(
                f"[PromptOpt] Loaded state: "
                f"{sum(len(v) for v in self._variants.values())} variants "
                f"across {len(self._variants)} roles"
            )
        except Exception as e:
            logger.warning(f"[PromptOpt] Could not load state: {e}")
