"""Evolution engine — continuous self-improvement of workflows and strategies.

Unlike the dynamic constitution (which learns rules from failures) and the
search tree (which explores alternatives within a single project), the evolution
engine operates ACROSS projects. It treats each project run as an organism:

  - **Genome**: The workflow configuration (which strategies were used, which
    tools were enabled, which constitution patches were active)
  - **Fitness**: A multi-dimensional score (quality, speed, cost, test pass rate)
  - **Memory**: Successful strategies are persisted and reused
  - **Mutation**: New projects start from the best-known strategies but with
    controlled variations to discover improvements
  - **Selection**: Over multiple runs, the best-performing workflows survive

This is the mechanism that lets AutoForge's agents "autonomously learn and
acquire better workflow methods" — the core loop is:

  1. Before project: Load best-known strategies from memory
  2. Mutate: Apply small variations to explore improvements
  3. Execute: Run the project with these strategies
  4. Evaluate: Measure quality, cost, speed
  5. Record: Save results to evolution history
  6. Reflect: Analyze what worked and why (LLM-assisted post-mortem)
  7. Evolve: Update the population of strategies

Persistence: All data stored in ~/.autoforge/evolution/ (global, not per-project).

Reference:
  - Evolutionary strategies (ES) for hyperparameter optimization
  - Quality-Diversity algorithms (MAP-Elites) for maintaining diverse solutions
  - LLM-assisted program synthesis and self-improvement
"""

from __future__ import annotations

import json
import logging
import math
import random
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
class WorkflowGenome:
    """A complete workflow configuration — the 'DNA' of a project run.

    Captures every strategic decision made during the pipeline.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    # Architecture strategy
    arch_strategy: str = ""         # Description of architectural approach
    arch_candidates_tried: int = 1  # How many candidates were explored
    # Constitution patches active
    active_patches: list[str] = field(default_factory=list)
    active_rules: list[str] = field(default_factory=list)
    # Build configuration
    parallel_builders: int = 2
    tdd_loops: int = 1
    checkpoints_enabled: bool = False
    search_tree_enabled: bool = False
    # Tech stack fingerprint (for matching similar projects)
    tech_fingerprint: str = ""      # e.g., "python-fastapi-react-postgres"
    project_type: str = ""          # e.g., "web-app", "cli-tool", "api-service"
    # Lineage
    parent_id: str | None = None    # Which genome this was derived from
    generation: int = 0
    mutations: list[str] = field(default_factory=list)  # What was changed
    created_at: float = field(default_factory=time.time)
    # Model routing strategy (ShinkaEvolve)
    model_preference: str = "balanced"  # "fast", "balanced", or "strong"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "arch_strategy": self.arch_strategy,
            "arch_candidates_tried": self.arch_candidates_tried,
            "active_patches": self.active_patches,
            "active_rules": self.active_rules,
            "parallel_builders": self.parallel_builders,
            "tdd_loops": self.tdd_loops,
            "checkpoints_enabled": self.checkpoints_enabled,
            "search_tree_enabled": self.search_tree_enabled,
            "tech_fingerprint": self.tech_fingerprint,
            "project_type": self.project_type,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "mutations": self.mutations,
            "created_at": self.created_at,
            "model_preference": self.model_preference,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowGenome:
        return cls(
            id=data.get("id", uuid.uuid4().hex[:10]),
            arch_strategy=data.get("arch_strategy", ""),
            arch_candidates_tried=data.get("arch_candidates_tried", 1),
            active_patches=data.get("active_patches", []),
            active_rules=data.get("active_rules", []),
            parallel_builders=data.get("parallel_builders", 2),
            tdd_loops=data.get("tdd_loops", 1),
            checkpoints_enabled=data.get("checkpoints_enabled", False),
            search_tree_enabled=data.get("search_tree_enabled", False),
            tech_fingerprint=data.get("tech_fingerprint", ""),
            project_type=data.get("project_type", ""),
            parent_id=data.get("parent_id"),
            generation=data.get("generation", 0),
            mutations=data.get("mutations", []),
            created_at=data.get("created_at", time.time()),
            model_preference=data.get("model_preference", "balanced"),
        )


@dataclass
class FitnessScore:
    """Multi-dimensional fitness measurement for a project run."""
    quality_score: float = 0.0       # Code review score (0-10)
    test_pass_rate: float = 0.0      # Fraction of tests that passed (0-1)
    cost_usd: float = 0.0            # Total API cost
    duration_seconds: float = 0.0    # Wall-clock time
    tasks_completed: int = 0         # Number of build tasks done
    tasks_total: int = 0             # Total build tasks
    build_success_rate: float = 0.0  # tasks_completed / tasks_total
    refactor_needed: bool = True     # Whether refactoring phase was needed

    @property
    def composite_score(self) -> float:
        """Single composite fitness value for ranking.

        Weighted formula prioritizing quality > completion > cost efficiency.
        """
        quality_norm = self.quality_score / 10.0  # 0-1
        completion = self.build_success_rate
        cost_efficiency = max(0, 1.0 - (self.cost_usd / 5.0))  # Penalize over $5
        speed_bonus = 0.1 if self.duration_seconds < 300 else 0.0

        return (
            quality_norm * 0.40
            + completion * 0.30
            + self.test_pass_rate * 0.15
            + cost_efficiency * 0.10
            + speed_bonus * 0.05
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_score": self.quality_score,
            "test_pass_rate": self.test_pass_rate,
            "cost_usd": self.cost_usd,
            "duration_seconds": self.duration_seconds,
            "tasks_completed": self.tasks_completed,
            "tasks_total": self.tasks_total,
            "build_success_rate": self.build_success_rate,
            "refactor_needed": self.refactor_needed,
            "composite_score": self.composite_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitnessScore:
        return cls(
            quality_score=data.get("quality_score", 0),
            test_pass_rate=data.get("test_pass_rate", 0),
            cost_usd=data.get("cost_usd", 0),
            duration_seconds=data.get("duration_seconds", 0),
            tasks_completed=data.get("tasks_completed", 0),
            tasks_total=data.get("tasks_total", 0),
            build_success_rate=data.get("build_success_rate", 0),
            refactor_needed=data.get("refactor_needed", True),
        )


@dataclass
class EvolutionRecord:
    """A single project run record in evolution history."""
    genome: WorkflowGenome
    fitness: FitnessScore
    project_name: str = ""
    reflection: str = ""        # LLM-generated post-mortem
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome": self.genome.to_dict(),
            "fitness": self.fitness.to_dict(),
            "project_name": self.project_name,
            "reflection": self.reflection,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolutionRecord:
        return cls(
            genome=WorkflowGenome.from_dict(data.get("genome", {})),
            fitness=FitnessScore.from_dict(data.get("fitness", {})),
            project_name=data.get("project_name", ""),
            reflection=data.get("reflection", ""),
            timestamp=data.get("timestamp", time.time()),
        )


# ──────────────────────────────────────────────
# Strategy Memory — "what worked before"
# ──────────────────────────────────────────────


@dataclass
class StrategyMemory:
    """Persistent memory of successful strategies, indexed by project type.

    This is the "population" in evolutionary terms. It maintains a diverse
    set of high-performing strategies for different project categories.
    """
    # Best strategies per project type (MAP-Elites style niche preservation)
    niches: dict[str, list[EvolutionRecord]] = field(default_factory=dict)
    # Global top strategies regardless of type
    hall_of_fame: list[EvolutionRecord] = field(default_factory=list)
    max_per_niche: int = 5
    max_hall_of_fame: int = 10

    def record(self, entry: EvolutionRecord) -> None:
        """Record a completed project run."""
        niche_key = entry.genome.tech_fingerprint or entry.genome.project_type or "general"

        # Add to niche
        if niche_key not in self.niches:
            self.niches[niche_key] = []
        self.niches[niche_key].append(entry)

        # Keep only top performers per niche
        self.niches[niche_key].sort(
            key=lambda r: r.fitness.composite_score, reverse=True
        )
        self.niches[niche_key] = self.niches[niche_key][:self.max_per_niche]

        # Update hall of fame
        self.hall_of_fame.append(entry)
        self.hall_of_fame.sort(
            key=lambda r: r.fitness.composite_score, reverse=True
        )
        self.hall_of_fame = self.hall_of_fame[:self.max_hall_of_fame]

    def get_best_for_type(self, project_type: str, tech_fingerprint: str = "") -> EvolutionRecord | None:
        """Find the best-performing strategy for a similar project."""
        # Exact niche match first
        for key in [tech_fingerprint, project_type, "general"]:
            if key and key in self.niches and self.niches[key]:
                return self.niches[key][0]

        # Fuzzy match: find niches with overlapping tech
        if tech_fingerprint:
            target_techs = set(tech_fingerprint.split("-"))
            best_match = None
            best_overlap = 0
            for niche_key, records in self.niches.items():
                niche_techs = set(niche_key.split("-"))
                overlap = len(target_techs & niche_techs)
                if overlap > best_overlap and records:
                    best_overlap = overlap
                    best_match = records[0]
            if best_match:
                return best_match

        # Fall back to hall of fame
        return self.hall_of_fame[0] if self.hall_of_fame else None

    def get_diverse_strategies(self, n: int = 3) -> list[EvolutionRecord]:
        """Get N diverse strategies from different niches (for crossover)."""
        strategies = []
        for niche_key in sorted(self.niches.keys()):
            if self.niches[niche_key]:
                strategies.append(self.niches[niche_key][0])
            if len(strategies) >= n:
                break
        return strategies

    def to_dict(self) -> dict[str, Any]:
        return {
            "niches": {
                k: [r.to_dict() for r in records]
                for k, records in self.niches.items()
            },
            "hall_of_fame": [r.to_dict() for r in self.hall_of_fame],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyMemory:
        mem = cls()
        for k, records in data.get("niches", {}).items():
            mem.niches[k] = [EvolutionRecord.from_dict(r) for r in records]
        mem.hall_of_fame = [
            EvolutionRecord.from_dict(r) for r in data.get("hall_of_fame", [])
        ]
        return mem


# ──────────────────────────────────────────────
# Evolution Engine
# ──────────────────────────────────────────────


class EvolutionEngine:
    """Core evolution engine — manages the genome lifecycle across projects.

    The engine follows a biological evolution metaphor:

    1. **Recall**: Load the best-known genome for a similar project type
    2. **Mutate**: Apply controlled mutations to explore improvements
    3. **Execute**: The orchestrator runs the project with this genome
    4. **Evaluate**: Measure the fitness of the result
    5. **Reflect**: LLM analyzes what worked and what didn't
    6. **Record**: Save the genome + fitness to evolution memory
    7. **Prune**: Remove low-performing strategies from memory
    """

    # Global evolution data directory (persists across all projects)
    _EVOLUTION_DIR_NAME = ".autoforge"
    _MEMORY_FILE = "evolution_memory.json"
    _HISTORY_FILE = "evolution_history.jsonl"

    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            base_dir = Path.home() / self._EVOLUTION_DIR_NAME
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.memory = self._load_memory()
        self._current_genome: WorkflowGenome | None = None
        self._current_start_time: float = 0

    # ──────── Phase 1: Recall + Mutate ────────

    def _get_niche_candidates(self, project_type: str, tech_fingerprint: str) -> list[EvolutionRecord]:
        """Get all candidate genomes from the niche (ShinkaEvolve fitness-proportional sampling).

        Returns candidates that could be used for parent selection, ordered by fitness.
        """
        candidates = []

        # Exact niche match first
        for key in [tech_fingerprint, project_type, "general"]:
            if key and key in self.memory.niches and self.memory.niches[key]:
                candidates.extend(self.memory.niches[key])
                if candidates:
                    break

        # Fuzzy match if no exact match
        if not candidates and tech_fingerprint:
            target_techs = set(tech_fingerprint.split("-"))
            for niche_key, records in self.memory.niches.items():
                niche_techs = set(niche_key.split("-"))
                if len(target_techs & niche_techs) > 0 and records:
                    candidates.extend(records)

        # Fall back to hall of fame
        if not candidates:
            candidates = list(self.memory.hall_of_fame)

        return candidates

    def _fitness_proportional_sample(self, candidates: list[EvolutionRecord]) -> EvolutionRecord | None:
        """Fitness-proportional (softmax) parent selection from candidates (ShinkaEvolve trick 1).

        Instead of just picking the single best, use softmax-weighted probability distribution
        to promote diversity while still favoring better-performing genomes.
        """
        import random

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Extract fitness scores and normalize
        scores = [max(0.1, r.fitness.composite_score) for r in candidates]
        max_score = max(scores)
        min_score = min(scores)

        # Normalize to [0, 1] range
        if max_score > min_score:
            normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        else:
            normalized = [1.0] * len(scores)

        # Apply softmax (temperature-scaled exponential)
        temperature = 0.5  # Lower = more concentrated on best; higher = more diverse
        exp_scores = [math.exp(x / temperature) for x in normalized]
        total = sum(exp_scores)
        probabilities = [e / total for e in exp_scores]

        # Sample according to probabilities
        selected = random.choices(candidates, weights=probabilities, k=1)[0]
        return selected

    def prepare_genome(
        self,
        project_type: str = "",
        tech_fingerprint: str = "",
        config: Any = None,
    ) -> WorkflowGenome:
        """Prepare a genome for a new project run.

        If evolution memory has a good match, start from that and mutate.
        Otherwise, create a fresh genome from the current config.

        Uses fitness-proportional sampling (ShinkaEvolve trick 1) to select ancestors.
        """
        # Try to recall successful strategies (ShinkaEvolve: fitness-proportional sampling)
        candidates = self._get_niche_candidates(project_type, tech_fingerprint)
        ancestor = self._fitness_proportional_sample(candidates)

        if ancestor and ancestor.fitness.composite_score > 0.5:
            # Found a good ancestor — inherit and mutate
            genome = self._inherit(ancestor.genome, project_type, tech_fingerprint)
            genome = self._mutate(genome)
            logger.info(
                f"[Evolution] Inherited from genome {ancestor.genome.id} "
                f"(gen {ancestor.genome.generation}, fitness={ancestor.fitness.composite_score:.2f})"
            )
        else:
            # No good ancestor — create from scratch
            genome = WorkflowGenome(
                tech_fingerprint=tech_fingerprint,
                project_type=project_type,
                generation=0,
            )
            # Apply config defaults if available
            if config:
                genome.parallel_builders = getattr(config, "max_agents", 2)
                genome.tdd_loops = getattr(config, "build_test_loops", 1)
                genome.checkpoints_enabled = getattr(config, "checkpoints_enabled", False)
                genome.search_tree_enabled = getattr(config, "search_tree_enabled", False)
            logger.info("[Evolution] Created fresh genome (no suitable ancestor)")

        self._current_genome = genome
        self._current_start_time = time.time()
        return genome

    def _inherit(
        self,
        parent: WorkflowGenome,
        project_type: str,
        tech_fingerprint: str,
    ) -> WorkflowGenome:
        """Create a child genome from a parent (inheritance without mutation)."""
        return WorkflowGenome(
            arch_strategy=parent.arch_strategy,
            arch_candidates_tried=parent.arch_candidates_tried,
            active_patches=list(parent.active_patches),
            active_rules=list(parent.active_rules),
            parallel_builders=parent.parallel_builders,
            tdd_loops=parent.tdd_loops,
            checkpoints_enabled=parent.checkpoints_enabled,
            search_tree_enabled=parent.search_tree_enabled,
            tech_fingerprint=tech_fingerprint or parent.tech_fingerprint,
            project_type=project_type or parent.project_type,
            parent_id=parent.id,
            generation=parent.generation + 1,
            model_preference=parent.model_preference,
        )

    def _genome_fingerprint(self, genome: WorkflowGenome) -> frozenset[str]:
        """Create a fingerprint for a genome (ShinkaEvolve trick 2: novelty rejection).

        Returns a frozenset of genome characteristics for Jaccard distance comparison.
        """
        fingerprint = {
            f"search_tree:{genome.search_tree_enabled}",
            f"checkpoints:{genome.checkpoints_enabled}",
            f"candidates:{genome.arch_candidates_tried}",
            f"tdd_loops:{genome.tdd_loops}",
            f"parallel:{genome.parallel_builders}",
            f"model:{genome.model_preference}",
        }
        if genome.arch_strategy:
            fingerprint.add(f"arch:{genome.arch_strategy}")
        return frozenset(fingerprint)

    def _novelty_check(
        self,
        candidate: WorkflowGenome,
        recent_genomes: list[WorkflowGenome],
        jaccard_threshold: float = 0.7,
    ) -> bool:
        """Check if a candidate genome is sufficiently novel (ShinkaEvolve trick 2).

        Uses Jaccard distance between fingerprints to ensure diversity.
        Returns True if the genome is novel (sufficiently different from recent ones).
        """
        if not recent_genomes:
            return True

        candidate_fp = self._genome_fingerprint(candidate)

        for recent in recent_genomes:
            recent_fp = self._genome_fingerprint(recent)

            # Compute Jaccard similarity: intersection / union
            intersection = len(candidate_fp & recent_fp)
            union = len(candidate_fp | recent_fp)

            if union > 0:
                jaccard_similarity = intersection / union
                if jaccard_similarity > jaccard_threshold:
                    # Too similar to a recent genome — not novel
                    return False

        return True

    def _mutate(self, genome: WorkflowGenome) -> WorkflowGenome:
        """Apply small random mutations to a genome.

        Mutations are conservative — only change 1-2 parameters at a time.
        The idea is to explore the neighborhood of known-good solutions.

        Includes novelty rejection sampling (ShinkaEvolve trick 2) to avoid
        re-exploring similar genomes.
        """
        import random

        # Collect recent genomes from memory for novelty check
        recent_genomes = []
        for records in self.memory.niches.values():
            recent_genomes.extend([r.genome for r in records[:3]])  # Top 3 from each niche
        recent_genomes.extend([r.genome for r in self.memory.hall_of_fame[:5]])

        # Novelty rejection: up to 3 attempts to generate a novel genome
        max_attempts = 3
        for attempt in range(max_attempts):
            mutations = []
            candidate = WorkflowGenome(
                arch_strategy=genome.arch_strategy,
                arch_candidates_tried=genome.arch_candidates_tried,
                active_patches=list(genome.active_patches),
                active_rules=list(genome.active_rules),
                parallel_builders=genome.parallel_builders,
                tdd_loops=genome.tdd_loops,
                checkpoints_enabled=genome.checkpoints_enabled,
                search_tree_enabled=genome.search_tree_enabled,
                tech_fingerprint=genome.tech_fingerprint,
                project_type=genome.project_type,
                parent_id=genome.parent_id,
                generation=genome.generation,
                model_preference=genome.model_preference,
            )

            # Mutation 1: Toggle search tree (10% chance)
            if random.random() < 0.10:
                candidate.search_tree_enabled = not candidate.search_tree_enabled
                mutations.append(f"search_tree={'on' if candidate.search_tree_enabled else 'off'}")

            # Mutation 2: Adjust candidate count (15% chance)
            if random.random() < 0.15 and candidate.search_tree_enabled:
                delta = random.choice([-1, 1])
                candidate.arch_candidates_tried = max(2, min(5, candidate.arch_candidates_tried + delta))
                mutations.append(f"candidates={candidate.arch_candidates_tried}")

            # Mutation 3: Toggle checkpoints (10% chance)
            if random.random() < 0.10:
                candidate.checkpoints_enabled = not candidate.checkpoints_enabled
                mutations.append(f"checkpoints={'on' if candidate.checkpoints_enabled else 'off'}")

            # Mutation 4: Adjust TDD loops (15% chance)
            if random.random() < 0.15:
                delta = random.choice([-1, 1])
                candidate.tdd_loops = max(0, min(3, candidate.tdd_loops + delta))
                mutations.append(f"tdd_loops={candidate.tdd_loops}")

            # Mutation 5: Adjust parallelism (10% chance)
            if random.random() < 0.10:
                delta = random.choice([-1, 1])
                candidate.parallel_builders = max(1, min(4, candidate.parallel_builders + delta))
                mutations.append(f"parallel={candidate.parallel_builders}")

            # Mutation 6: Adaptive model ensemble selection (15% chance) (ShinkaEvolve trick 3)
            if random.random() < 0.15:
                candidate.model_preference = random.choice(["fast", "balanced", "strong"])
                mutations.append(f"model_preference={candidate.model_preference}")

            # Check novelty (ShinkaEvolve trick 2: novelty rejection sampling)
            if self._novelty_check(candidate, recent_genomes):
                candidate.mutations = mutations
                if mutations:
                    logger.info(f"[Evolution] Mutations (attempt {attempt + 1}): {', '.join(mutations)}")
                return candidate
            else:
                if attempt < max_attempts - 1:
                    logger.debug(f"[Evolution] Mutation not novel enough (attempt {attempt + 1}/{max_attempts}), re-rolling...")

        # If all attempts failed novelty check, return the last candidate anyway
        candidate.mutations = mutations
        if mutations:
            logger.info(f"[Evolution] Mutations (final, novelty check exhausted): {', '.join(mutations)}")
        return candidate

    # ──────── Phase 2: Record + Evaluate ────────

    def record_result(
        self,
        project_name: str,
        fitness: FitnessScore,
        genome: WorkflowGenome | None = None,
    ) -> EvolutionRecord:
        """Record a completed project run."""
        genome = genome or self._current_genome or WorkflowGenome()

        record = EvolutionRecord(
            genome=genome,
            fitness=fitness,
            project_name=project_name,
        )

        self.memory.record(record)
        self._save_memory()
        self._append_history(record)

        logger.info(
            f"[Evolution] Recorded {project_name}: "
            f"fitness={fitness.composite_score:.2f} gen={genome.generation}"
        )
        return record

    # ──────── Phase 3: Reflect ────────

    async def reflect(
        self,
        record: EvolutionRecord,
        llm: Any,
    ) -> str:
        """LLM-assisted post-mortem — analyze what worked and what to improve.

        This is a key differentiator from blind evolutionary search: the LLM
        can reason about WHY a strategy worked or failed, not just its score.
        """
        from autoforge.engine.llm_router import TaskComplexity

        # Gather context
        history_context = self._get_relevant_history(record.genome.tech_fingerprint, limit=5)

        prompt = (
            f"Analyze this AutoForge project run and suggest workflow improvements.\n\n"
            f"## Project: {record.project_name}\n"
            f"## Genome (Configuration)\n"
            f"- Architecture strategy: {record.genome.arch_strategy or 'default'}\n"
            f"- Search tree: {'enabled' if record.genome.search_tree_enabled else 'disabled'} "
            f"(tried {record.genome.arch_candidates_tried} candidates)\n"
            f"- TDD loops: {record.genome.tdd_loops}\n"
            f"- Checkpoints: {'enabled' if record.genome.checkpoints_enabled else 'disabled'}\n"
            f"- Parallel builders: {record.genome.parallel_builders}\n"
            f"- Generation: {record.genome.generation}\n"
            f"- Mutations applied: {record.genome.mutations or 'none'}\n\n"
            f"## Fitness Results\n"
            f"- Quality: {record.fitness.quality_score}/10\n"
            f"- Test pass rate: {record.fitness.test_pass_rate:.0%}\n"
            f"- Build success: {record.fitness.tasks_completed}/{record.fitness.tasks_total}\n"
            f"- Cost: ${record.fitness.cost_usd:.4f}\n"
            f"- Duration: {record.fitness.duration_seconds:.0f}s\n"
            f"- Refactoring needed: {'yes' if record.fitness.refactor_needed else 'no'}\n"
            f"- Composite fitness: {record.fitness.composite_score:.2f}\n\n"
        )

        if history_context:
            prompt += (
                f"## Previous Runs (same project type)\n"
                f"{history_context}\n\n"
            )

        prompt += (
            f"## Instructions\n"
            f"1. What worked well in this run?\n"
            f"2. What could be improved?\n"
            f"3. Suggest specific parameter changes for the next run\n"
            f"4. Are there any workflow patterns worth remembering?\n\n"
            f"Be concise — 3-5 bullet points total."
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system=(
                    "You are a DevOps analyst reviewing an AI code generation run. "
                    "Focus on actionable workflow improvements, not code quality."
                ),
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            record.reflection = text.strip()
            self._save_memory()
            logger.info(f"[Evolution] Reflection completed for {record.project_name}")
            return text.strip()

        except Exception as e:
            logger.warning(f"[Evolution] Reflection failed: {e}")
            return ""

    # ──────── Phase 4: Crossover (advanced) ────────

    def crossover(
        self,
        genome_a: WorkflowGenome,
        genome_b: WorkflowGenome,
    ) -> WorkflowGenome:
        """Combine two successful genomes into a new one.

        Inspired by genetic algorithms: take the best traits from each parent.
        This is used when the engine has enough history to identify which
        specific parameters contributed to success.
        """
        child = WorkflowGenome(
            # Take architecture from the higher-scoring parent
            arch_strategy=genome_a.arch_strategy or genome_b.arch_strategy,
            arch_candidates_tried=max(
                genome_a.arch_candidates_tried,
                genome_b.arch_candidates_tried,
            ),
            # Merge patches/rules from both (union)
            active_patches=list(set(genome_a.active_patches + genome_b.active_patches)),
            active_rules=list(set(genome_a.active_rules + genome_b.active_rules)),
            # Randomly choose numerical params from either parent
            parallel_builders=random.choice([
                genome_a.parallel_builders, genome_b.parallel_builders
            ]),
            tdd_loops=random.choice([genome_a.tdd_loops, genome_b.tdd_loops]),
            checkpoints_enabled=random.choice([
                genome_a.checkpoints_enabled, genome_b.checkpoints_enabled
            ]),
            search_tree_enabled=random.choice([
                genome_a.search_tree_enabled, genome_b.search_tree_enabled
            ]),
            # Inherit tech info
            tech_fingerprint=genome_a.tech_fingerprint or genome_b.tech_fingerprint,
            project_type=genome_a.project_type or genome_b.project_type,
            parent_id=genome_a.id,
            generation=max(genome_a.generation, genome_b.generation) + 1,
            mutations=[f"crossover({genome_a.id}x{genome_b.id})"],
            model_preference=random.choice([genome_a.model_preference, genome_b.model_preference]),
        )
        logger.info(
            f"[Evolution] Crossover: {genome_a.id} x {genome_b.id} -> {child.id}"
        )
        return child

    # ──────── Genome → Config Application ────────

    def apply_genome_to_config(self, genome: WorkflowGenome, config: Any) -> None:
        """Apply a genome's parameters to a ForgeConfig instance.

        This is how the evolution engine actually influences the pipeline:
        it adjusts config parameters before the orchestrator runs.
        """
        if hasattr(config, "max_agents"):
            config.max_agents = genome.parallel_builders
        if hasattr(config, "build_test_loops"):
            config.build_test_loops = genome.tdd_loops
        if hasattr(config, "checkpoints_enabled"):
            config.checkpoints_enabled = genome.checkpoints_enabled
        if hasattr(config, "search_tree_enabled"):
            config.search_tree_enabled = genome.search_tree_enabled
        if hasattr(config, "search_tree_max_candidates"):
            config.search_tree_max_candidates = genome.arch_candidates_tried

        # Note: genome.model_preference ("strong"/"fast"/"balanced") is an
        # advisory signal logged for analysis.  We do NOT override
        # config.model_strong / config.model_fast here because those are
        # model-name strings, not booleans.  The LLM router already picks
        # the right tier per-agent based on task complexity.

        logger.info(
            f"[Evolution] Applied genome {genome.id} to config: "
            f"parallel={genome.parallel_builders} tdd={genome.tdd_loops} "
            f"search_tree={genome.search_tree_enabled} checkpoints={genome.checkpoints_enabled} "
            f"model_preference={genome.model_preference}"
        )

    # ──────── Tech Fingerprint Extraction ────────

    @staticmethod
    def extract_tech_fingerprint(spec: dict[str, Any]) -> str:
        """Extract a canonical tech fingerprint from a project spec.

        Creates a normalized string like "python-fastapi-react-postgres"
        that can be used to match similar projects.
        """
        parts = []

        tech_stack = spec.get("tech_stack", {})
        if isinstance(tech_stack, dict):
            for key in sorted(tech_stack.keys()):
                val = tech_stack[key]
                if isinstance(val, str):
                    parts.append(val.lower().split()[0])  # First word
                elif isinstance(val, list):
                    parts.extend(v.lower().split()[0] for v in val if isinstance(v, str))
        elif isinstance(tech_stack, str):
            parts.extend(tech_stack.lower().replace(",", " ").split())

        # Deduplicate and sort
        parts = sorted(set(parts))
        return "-".join(parts) if parts else "general"

    @staticmethod
    def infer_project_type(spec: dict[str, Any]) -> str:
        """Infer the project type from the spec."""
        desc = (spec.get("description", "") + " " + spec.get("project_name", "")).lower()

        type_keywords = {
            "web-app": ["web", "frontend", "dashboard", "portal", "website"],
            "api-service": ["api", "backend", "rest", "graphql", "server"],
            "cli-tool": ["cli", "command", "terminal", "shell"],
            "mobile-app": ["mobile", "ios", "android", "react native", "flutter"],
            "data-pipeline": ["data", "etl", "pipeline", "analytics"],
            "library": ["library", "package", "sdk", "module", "framework"],
        }

        for project_type, keywords in type_keywords.items():
            if any(kw in desc for kw in keywords):
                return project_type

        return "general"

    # ──────── Stats & Reporting ────────

    def get_evolution_stats(self) -> dict[str, Any]:
        """Get statistics about the evolution history."""
        all_records = []
        for records in self.memory.niches.values():
            all_records.extend(records)

        if not all_records:
            return {"total_runs": 0, "niches": 0, "message": "No evolution history yet"}

        scores = [r.fitness.composite_score for r in all_records]
        generations = [r.genome.generation for r in all_records]

        return {
            "total_runs": len(all_records),
            "niches": len(self.memory.niches),
            "hall_of_fame_size": len(self.memory.hall_of_fame),
            "best_fitness": max(scores),
            "avg_fitness": sum(scores) / len(scores),
            "max_generation": max(generations) if generations else 0,
            "niche_keys": list(self.memory.niches.keys()),
        }

    # ──────── Persistence ────────

    def _load_memory(self) -> StrategyMemory:
        """Load evolution memory from disk."""
        path = self.base_dir / self._MEMORY_FILE
        if not path.exists():
            return StrategyMemory()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return StrategyMemory.from_dict(data)
        except Exception as e:
            logger.warning(f"[Evolution] Could not load memory: {e}")
            return StrategyMemory()

    def _save_memory(self) -> None:
        """Save evolution memory to disk."""
        path = self.base_dir / self._MEMORY_FILE
        try:
            path.write_text(
                json.dumps(self.memory.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[Evolution] Could not save memory: {e}")

    def _append_history(self, record: EvolutionRecord) -> None:
        """Append a record to the JSONL history file (append-only log)."""
        path = self.base_dir / self._HISTORY_FILE
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[Evolution] Could not append history: {e}")

    def _get_relevant_history(self, tech_fingerprint: str, limit: int = 5) -> str:
        """Get formatted history of similar project runs for reflection context."""
        records = self.memory.niches.get(tech_fingerprint, [])
        if not records:
            records = self.memory.hall_of_fame[:limit]
        else:
            records = records[:limit]

        if not records:
            return ""

        lines = []
        for r in records:
            lines.append(
                f"- {r.project_name}: fitness={r.fitness.composite_score:.2f} "
                f"quality={r.fitness.quality_score}/10 "
                f"gen={r.genome.generation}"
            )
            if r.reflection:
                lines.append(f"  Reflection: {r.reflection[:150]}...")
        return "\n".join(lines)
