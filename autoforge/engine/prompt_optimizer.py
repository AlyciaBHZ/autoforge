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

    def record_fitness(self, fitness: float) -> None:
        """Record a fitness observation."""
        self.times_used += 1
        self.total_fitness += fitness
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
    _EXPLORATION_BONUS = 0.1        # Bonus for under-explored variants

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

        Uses Thompson Sampling: balance exploitation (use best-known)
        with exploration (try under-sampled variants).

        Returns: (variant_id, prompt_content)
        """
        variants = self._variants.get(role, [])
        if not variants:
            return ("", "")

        # Thompson Sampling with exploration bonus
        best_score = -1.0
        best_variant = variants[0]

        import random
        for v in variants:
            if v.times_used == 0:
                # Untested variant gets high exploration priority
                score = 1.0 + random.random() * 0.5
            else:
                # Score = average fitness + exploration bonus for less-tested variants
                avg = v.avg_fitness
                exploration = self._EXPLORATION_BONUS / math.sqrt(v.times_used)
                noise = random.gauss(0, 0.05)  # Small random perturbation
                score = avg + exploration + noise

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
                    f"(avg={v.avg_fitness:.3f}, n={v.times_used})"
                )
                self._save_state()
                return

        logger.warning(f"[PromptOpt] Variant {variant_id} not found for {role}")

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

            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                logger.warning("[PromptOpt] Could not parse optimization response")
                return None

            data = json.loads(match.group())
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

            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None

            data = json.loads(match.group())
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

    # ──────── Variant Management ────────

    def _add_variant(self, role: str, variant: PromptVariant) -> None:
        """Add a variant, evicting the worst performer if at capacity."""
        variants = self._variants.setdefault(role, [])
        variants.append(variant)

        if len(variants) > self._MAX_VARIANTS_PER_ROLE:
            # Evict the worst-performing variant (but never the baseline)
            candidates_for_eviction = [
                v for v in variants
                if v.source != "baseline" and v.times_used > 0
            ]
            if candidates_for_eviction:
                worst = min(candidates_for_eviction, key=lambda v: v.avg_fitness)
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
