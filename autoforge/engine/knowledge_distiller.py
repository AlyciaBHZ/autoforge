"""KnowledgeDistiller — Cross-module generalizable knowledge extraction.

Aggregates patterns from all self-evolution subsystems (Reflexion, SICA,
DynamicConstitution, Evolution, EvoMAC) and distills them into reusable
principles that transfer across projects and contexts.

Key insight: each module collects domain-specific memories, but none of them
abstract from specific cases to general principles. This module bridges that
gap by:
  1. Clustering similar failure reflections into failure archetypes
  2. Extracting parameter correlations from evolution strategy memory
  3. Aggregating learned rules into ranked best-practice lists
  4. Cross-referencing gradient effectiveness with rule confidence

The output is a structured KnowledgeBase that can be:
  - Injected into new project prompts as "institutional knowledge"
  - Used to warm-start evolution for similar projects
  - Exported as a human-readable report

References:
  - Experience replay (DQN, Mnih 2015) — but with natural language
  - Knowledge distillation (Hinton 2015) — teacher → student via soft labels
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────


@dataclass
class DistilledPrinciple:
    """A generalizable principle extracted from multiple observations.

    This is the atomic unit of distilled knowledge — a single insight
    supported by evidence from one or more evolution subsystems.
    """

    id: str
    principle: str  # The actual knowledge statement
    category: str  # "failure_pattern" | "strategy_insight" | "best_practice" | "anti_pattern"
    confidence: float  # 0-1, based on evidence strength
    evidence_count: int  # Number of supporting observations
    source_modules: list[str] = field(default_factory=list)  # Which modules contributed
    applicable_contexts: list[str] = field(default_factory=list)  # Project types / tech stacks
    created_at: float = field(default_factory=time.time)
    last_validated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "principle": self.principle,
            "category": self.category,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "source_modules": self.source_modules,
            "applicable_contexts": self.applicable_contexts,
            "created_at": self.created_at,
            "last_validated": self.last_validated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DistilledPrinciple:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FailureArchetype:
    """A cluster of similar failures abstracted into a reusable pattern."""

    id: str
    archetype: str  # Generalized failure description
    frequency: int  # How often this pattern occurs
    affected_roles: list[str] = field(default_factory=list)
    common_tags: list[str] = field(default_factory=list)
    resolution_strategies: list[str] = field(default_factory=list)
    resolution_success_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "archetype": self.archetype,
            "frequency": self.frequency,
            "affected_roles": self.affected_roles,
            "common_tags": self.common_tags,
            "resolution_strategies": self.resolution_strategies,
            "resolution_success_rate": self.resolution_success_rate,
        }


@dataclass
class StrategyInsight:
    """A parameter correlation discovered from evolution strategy memory."""

    id: str
    insight: str  # Natural language description
    parameter: str  # Which genome parameter
    correlation: str  # "positive" | "negative" | "conditional"
    effect_size: float  # How much it affects fitness (Cohen's d or similar)
    conditions: list[str] = field(default_factory=list)  # When this applies
    sample_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "insight": self.insight,
            "parameter": self.parameter,
            "correlation": self.correlation,
            "effect_size": self.effect_size,
            "conditions": self.conditions,
            "sample_size": self.sample_size,
        }


# ──────────────────────────────────────────────
# Knowledge Distiller
# ──────────────────────────────────────────────


class KnowledgeDistiller:
    """Distills generalizable knowledge from all self-evolution subsystems.

    This is the "institutional memory" of AutoForge — it aggregates specific
    experiences from Reflexion, SICA, DynamicConstitution, Evolution, and
    EvoMAC into reusable principles.

    Usage:
        distiller = KnowledgeDistiller(project_dir)
        distiller.ingest_reflections(reflexion_engine.get_recent_memories(50))
        distiller.ingest_learned_rules(dynamic_constitution._learned_rules)
        distiller.ingest_strategy_memory(evolution_engine.strategy_memory)
        distiller.ingest_edge_effectiveness(evomac_engine)

        distiller.distill()  # Run the distillation process

        # Get knowledge for a new project
        context = distiller.build_knowledge_context("python-fastapi-web")
    """

    MIN_EVIDENCE_FOR_PRINCIPLE = 3  # Minimum observations to form a principle
    CONFIDENCE_DECAY_DAYS = 60  # Days before principles start losing confidence

    def __init__(self, project_dir: Path | None = None) -> None:
        self._principles: list[DistilledPrinciple] = []
        self._failure_archetypes: list[FailureArchetype] = []
        self._strategy_insights: list[StrategyInsight] = []
        self._persistence_path = (
            project_dir / ".autoforge" / "distilled_knowledge.json"
            if project_dir
            else None
        )

        # Raw ingested data (cleared after distillation)
        self._raw_reflections: list[dict[str, Any]] = []
        self._raw_rules: list[dict[str, Any]] = []
        self._raw_strategies: list[dict[str, Any]] = []
        self._raw_edge_data: list[dict[str, Any]] = []

        self._load_state()

    # ──── Ingestion ──────────────────────────────

    def ingest_reflections(self, reflections: list[Any]) -> None:
        """Ingest reflections from Reflexion engine."""
        for r in reflections:
            self._raw_reflections.append({
                "task": getattr(r, "task_description", ""),
                "failure": getattr(r, "failure_summary", ""),
                "reflection": getattr(r, "reflection", ""),
                "outcome": getattr(r, "outcome", "pending"),
                "tags": getattr(r, "tags", []),
                "project": getattr(r, "project", ""),
            })

    def ingest_learned_rules(self, rules: list[Any]) -> None:
        """Ingest learned rules from DynamicConstitution."""
        for r in rules:
            self._raw_rules.append({
                "pattern": getattr(r, "pattern", ""),
                "rule": getattr(r, "rule", ""),
                "confidence": getattr(r, "confidence", 0.0),
                "times_applied": getattr(r, "times_applied", 0),
                "times_helped": getattr(r, "times_helped", 0),
                "source_role": getattr(r, "source_role", ""),
            })

    def ingest_strategy_memory(self, strategy_memory: Any) -> None:
        """Ingest strategy records from Evolution engine's StrategyMemory.

        Records may appear in both hall_of_fame and niches, so we deduplicate
        by (project_type, tech_fingerprint, composite_score) to avoid biasing
        the statistical analysis.
        """
        seen_fingerprints: set[tuple[str, str, float]] = set()

        def _ingest_record(record: Any, niche_key: str = "") -> None:
            genome = getattr(record, "genome", None)
            fitness = getattr(record, "fitness", None)
            if not genome or not fitness:
                return
            proj = getattr(genome, "project_type", "")
            tech = niche_key or getattr(genome, "tech_fingerprint", "")
            score = getattr(fitness, "composite_score", 0.0)
            dedup_key = (proj, tech, round(score, 6))
            if dedup_key in seen_fingerprints:
                return
            seen_fingerprints.add(dedup_key)
            self._raw_strategies.append({
                "project_type": proj,
                "tech_fingerprint": tech,
                "search_tree": getattr(genome, "search_tree_enabled", False),
                "checkpoints": getattr(genome, "checkpoints_enabled", False),
                "parallel_builders": getattr(genome, "parallel_builders", 1),
                "tdd_loops": getattr(genome, "tdd_loops", 1),
                "arch_candidates": getattr(genome, "arch_candidates_tried", 1),
                "model_preference": getattr(genome, "model_preference", "balanced"),
                "composite_score": score,
                "test_pass_rate": getattr(fitness, "test_pass_rate", 0.0),
            })

        if hasattr(strategy_memory, "hall_of_fame"):
            for record in strategy_memory.hall_of_fame:
                _ingest_record(record)

        if hasattr(strategy_memory, "niches"):
            for niche_key, records in strategy_memory.niches.items():
                for record in records:
                    _ingest_record(record, niche_key=niche_key)

    def ingest_edge_effectiveness(self, evomac_engine: Any) -> None:
        """Ingest edge effectiveness data from EvoMAC engine."""
        if hasattr(evomac_engine, "_topology"):
            for edge in evomac_engine._topology:
                self._raw_edge_data.append({
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "active": edge.active,
                    "effectiveness": edge.effectiveness,
                    "gradients_sent": edge.gradients_sent,
                    "gradients_helped": edge.gradients_helped,
                })

    # ──── Distillation ──────────────────────────

    def distill(self) -> dict[str, int]:
        """Run the full distillation process.

        Processes all ingested raw data and extracts generalizable principles.
        Returns counts of extracted knowledge items.
        """
        counts = {
            "failure_archetypes": 0,
            "strategy_insights": 0,
            "best_practices": 0,
            "anti_patterns": 0,
        }

        # Clear previous distillation results to avoid accumulation on
        # repeated calls.  Persisted state is rebuilt from current raw data.
        self._failure_archetypes.clear()
        self._strategy_insights.clear()
        self._principles.clear()

        # 1. Cluster failure reflections into archetypes
        archetypes = self._cluster_failures()
        self._failure_archetypes.extend(archetypes)
        counts["failure_archetypes"] = len(archetypes)

        # 2. Extract parameter correlations from strategies
        insights = self._extract_strategy_insights()
        self._strategy_insights.extend(insights)
        counts["strategy_insights"] = len(insights)

        # 3. Promote high-confidence learned rules to principles
        best_practices = self._promote_rules_to_principles()
        self._principles.extend(best_practices)
        counts["best_practices"] = len(best_practices)

        # 4. Extract anti-patterns from consistently failing edges
        anti_patterns = self._extract_anti_patterns()
        self._principles.extend(anti_patterns)
        counts["anti_patterns"] = len(anti_patterns)

        # 5. Convert archetypes and insights to principles
        for arch in archetypes:
            if arch.frequency >= self.MIN_EVIDENCE_FOR_PRINCIPLE:
                principle = DistilledPrinciple(
                    id=f"fp-{arch.id}",
                    principle=arch.archetype,
                    category="failure_pattern",
                    confidence=min(1.0, arch.frequency / 10.0),
                    evidence_count=arch.frequency,
                    source_modules=["reflexion"],
                    applicable_contexts=arch.common_tags,
                )
                self._principles.append(principle)

        for ins in insights:
            if ins.sample_size >= self.MIN_EVIDENCE_FOR_PRINCIPLE:
                principle = DistilledPrinciple(
                    id=f"si-{ins.id}",
                    principle=ins.insight,
                    category="strategy_insight",
                    confidence=min(1.0, abs(ins.effect_size)),
                    evidence_count=ins.sample_size,
                    source_modules=["evolution"],
                    applicable_contexts=ins.conditions,
                )
                self._principles.append(principle)

        # Deduplicate principles
        self._deduplicate_principles()

        # Clear raw data
        self._raw_reflections.clear()
        self._raw_rules.clear()
        self._raw_strategies.clear()
        self._raw_edge_data.clear()

        self._save_state()
        logger.info(f"[KnowledgeDistiller] Distilled: {counts}")
        return counts

    def _cluster_failures(self) -> list[FailureArchetype]:
        """Cluster similar failure reflections into archetypes.

        Uses tag-based clustering: reflections with overlapping tags are
        grouped together, then the most common pattern is extracted.
        """
        if not self._raw_reflections:
            return []

        # Group by primary tag
        tag_groups: dict[str, list[dict[str, Any]]] = {}
        for ref in self._raw_reflections:
            tags = ref.get("tags", [])
            primary_tag = tags[0] if tags else "unknown"
            tag_groups.setdefault(primary_tag, []).append(ref)

        archetypes = []
        for tag, group in tag_groups.items():
            if len(group) < 2:
                continue

            # Count affected roles
            role_counts = Counter()
            all_tags: list[str] = []
            resolutions: list[str] = []
            resolved_count = 0

            for ref in group:
                # Extract role from task description heuristically
                task = ref.get("task", "").lower()
                for role in ["builder", "architect", "tester", "reviewer", "gardener"]:
                    if role in task:
                        role_counts[role] += 1
                all_tags.extend(ref.get("tags", []))
                if ref.get("outcome") == "resolved":
                    resolved_count += 1
                    resolutions.append(ref.get("reflection", "")[:200])

            common_tags = [t for t, _ in Counter(all_tags).most_common(5)]

            archetype = FailureArchetype(
                id=f"arch-{tag}-{len(group)}",
                archetype=f"Recurring {tag} failures ({len(group)} occurrences)",
                frequency=len(group),
                affected_roles=list(role_counts.keys()),
                common_tags=common_tags,
                resolution_strategies=resolutions[:3],
                resolution_success_rate=resolved_count / len(group) if group else 0,
            )
            archetypes.append(archetype)

        return archetypes

    def _extract_strategy_insights(self) -> list[StrategyInsight]:
        """Extract parameter correlations from strategy memory.

        For each boolean/numeric parameter, compute correlation with
        composite_score using point-biserial or Pearson correlation.
        """
        if len(self._raw_strategies) < 3:
            return []

        insights = []
        params_to_check = [
            ("search_tree", "Search tree (MCTS)"),
            ("checkpoints", "Mid-task checkpoints"),
        ]

        for param_key, param_name in params_to_check:
            # Split into enabled vs disabled groups
            enabled = [s for s in self._raw_strategies if s.get(param_key)]
            disabled = [s for s in self._raw_strategies if not s.get(param_key)]

            if len(enabled) < 2 or len(disabled) < 2:
                continue

            avg_enabled = sum(s["composite_score"] for s in enabled) / len(enabled)
            avg_disabled = sum(s["composite_score"] for s in disabled) / len(disabled)

            delta = avg_enabled - avg_disabled
            # Cohen's d approximation
            pooled_var = (
                self._variance([s["composite_score"] for s in enabled])
                + self._variance([s["composite_score"] for s in disabled])
            ) / 2
            effect_size = delta / max(math.sqrt(pooled_var), 0.01)

            if abs(effect_size) > 0.2:  # At least small effect size
                correlation = "positive" if effect_size > 0 else "negative"
                insight = StrategyInsight(
                    id=f"si-{param_key}",
                    insight=f"{param_name} has a {correlation} effect on project quality "
                            f"(avg score {avg_enabled:.2f} enabled vs {avg_disabled:.2f} disabled, "
                            f"Cohen's d = {effect_size:.2f})",
                    parameter=param_key,
                    correlation=correlation,
                    effect_size=effect_size,
                    sample_size=len(enabled) + len(disabled),
                )
                insights.append(insight)

        # Check numeric parameters (parallel_builders, tdd_loops)
        for param_key, param_name in [
            ("parallel_builders", "Parallel builders"),
            ("tdd_loops", "TDD loop count"),
            ("arch_candidates", "Architecture candidates"),
        ]:
            values = [(s.get(param_key, 1), s["composite_score"])
                      for s in self._raw_strategies if param_key in s]
            if len(values) < 3:
                continue

            # Simple Pearson correlation
            r = self._pearson_correlation(
                [v[0] for v in values], [v[1] for v in values]
            )
            if abs(r) > 0.3:  # At least moderate correlation
                correlation = "positive" if r > 0 else "negative"
                insight = StrategyInsight(
                    id=f"si-{param_key}",
                    insight=f"Higher {param_name} {'improves' if r > 0 else 'degrades'} "
                            f"project quality (r = {r:.2f})",
                    parameter=param_key,
                    correlation=correlation,
                    effect_size=r,
                    sample_size=len(values),
                )
                insights.append(insight)

        return insights

    def _promote_rules_to_principles(self) -> list[DistilledPrinciple]:
        """Promote high-confidence learned rules to generalizable principles."""
        principles = []
        for rule in self._raw_rules:
            confidence = rule.get("confidence", 0)
            applied = rule.get("times_applied", 0)
            helped = rule.get("times_helped", 0)

            # Only promote rules with strong evidence
            if applied >= self.MIN_EVIDENCE_FOR_PRINCIPLE and confidence >= 0.6:
                principle = DistilledPrinciple(
                    id=f"bp-{hash(rule.get('pattern', '')) % 100000}",
                    principle=rule.get("rule", ""),
                    category="best_practice",
                    confidence=confidence,
                    evidence_count=applied,
                    source_modules=["dynamic_constitution"],
                    applicable_contexts=[rule.get("source_role", "all")],
                )
                principles.append(principle)

        return principles

    def _extract_anti_patterns(self) -> list[DistilledPrinciple]:
        """Extract anti-patterns from consistently failing feedback edges."""
        anti_patterns = []
        for edge in self._raw_edge_data:
            sent = edge.get("gradients_sent", 0)
            effectiveness = edge.get("effectiveness", 0.5)

            # Low effectiveness with sufficient data → anti-pattern
            if sent >= 5 and effectiveness < 0.2:
                principle = DistilledPrinciple(
                    id=f"ap-{edge['source']}-{edge['target']}",
                    principle=(
                        f"Feedback from {edge['source']} to {edge['target']} is "
                        f"consistently ineffective ({effectiveness:.0%} success rate "
                        f"over {sent} attempts). Consider restructuring this "
                        f"feedback channel or changing the feedback format."
                    ),
                    category="anti_pattern",
                    confidence=min(1.0, sent / 10.0),
                    evidence_count=sent,
                    source_modules=["evomac"],
                )
                anti_patterns.append(principle)

        return anti_patterns

    def _deduplicate_principles(self) -> None:
        """Remove near-duplicate principles using word overlap."""
        if len(self._principles) <= 1:
            return

        unique: list[DistilledPrinciple] = []
        seen_words: list[set[str]] = []

        # Sort by confidence (highest first) so we keep the most confident version
        self._principles.sort(key=lambda p: p.confidence, reverse=True)

        for p in self._principles:
            p_words = set(p.principle.lower().split())
            is_dup = False
            for sw in seen_words:
                if not p_words or not sw:
                    continue
                overlap = len(p_words & sw) / max(len(p_words | sw), 1)
                if overlap > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(p)
                seen_words.append(p_words)

        self._principles = unique

    # ──── Knowledge Retrieval ────────────────────

    def build_knowledge_context(
        self,
        project_type: str = "",
        tech_fingerprint: str = "",
        role: str = "",
        max_principles: int = 10,
    ) -> str:
        """Build a knowledge context string for injection into agent prompts.

        This is the primary output — a concise summary of institutional
        knowledge relevant to the current project/role.
        """
        relevant = self._get_relevant_principles(
            project_type, tech_fingerprint, role, max_principles
        )

        if not relevant:
            return ""

        parts = ["\n\n## Institutional Knowledge (Distilled from Past Projects)\n"]

        # Group by category
        by_category: dict[str, list[DistilledPrinciple]] = {}
        for p in relevant:
            by_category.setdefault(p.category, []).append(p)

        category_labels = {
            "best_practice": "Best Practices",
            "failure_pattern": "Common Pitfalls",
            "strategy_insight": "Strategy Insights",
            "anti_pattern": "Known Anti-Patterns",
        }

        for cat, label in category_labels.items():
            if cat not in by_category:
                continue
            parts.append(f"\n### {label}\n")
            for p in by_category[cat]:
                conf_str = f"[{p.confidence:.0%} confidence, {p.evidence_count} observations]"
                parts.append(f"- {p.principle} {conf_str}\n")

        return "".join(parts)

    def _get_relevant_principles(
        self,
        project_type: str,
        tech_fingerprint: str,
        role: str,
        top_k: int,
    ) -> list[DistilledPrinciple]:
        """Retrieve principles relevant to the current context."""
        now = time.time()

        scored: list[tuple[float, DistilledPrinciple]] = []
        for p in self._principles:
            score = p.confidence

            # Context matching bonus
            if project_type and project_type in p.applicable_contexts:
                score += 0.2
            if tech_fingerprint:
                tech_words = set(tech_fingerprint.lower().split("-"))
                context_words = set(
                    w.lower() for ctx in p.applicable_contexts for w in ctx.split("-")
                )
                if tech_words & context_words:
                    score += 0.15
            if role and role in p.applicable_contexts:
                score += 0.1

            # Evidence strength bonus
            score += min(0.2, p.evidence_count / 20.0)

            # Temporal decay
            age_days = (now - p.last_validated) / 86400.0
            if age_days > self.CONFIDENCE_DECAY_DAYS:
                decay = math.exp(-(age_days - self.CONFIDENCE_DECAY_DAYS) / 90.0)
                score *= decay

            scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def get_failure_archetypes(self) -> list[FailureArchetype]:
        """Return all distilled failure archetypes."""
        return self._failure_archetypes

    def get_strategy_insights(self) -> list[StrategyInsight]:
        """Return all distilled strategy insights."""
        return self._strategy_insights

    def get_all_principles(self) -> list[DistilledPrinciple]:
        """Return all distilled principles."""
        return self._principles

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        return {
            "total_principles": len(self._principles),
            "failure_archetypes": len(self._failure_archetypes),
            "strategy_insights": len(self._strategy_insights),
            "by_category": Counter(p.category for p in self._principles),
            "avg_confidence": (
                sum(p.confidence for p in self._principles) / max(len(self._principles), 1)
            ),
            "source_modules": list(set(
                m for p in self._principles for m in p.source_modules
            )),
        }

    # ──── Persistence ────────────────────────────

    def _save_state(self) -> None:
        """Persist distilled knowledge to disk."""
        if not self._persistence_path:
            return
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "principles": [p.to_dict() for p in self._principles],
                "failure_archetypes": [a.to_dict() for a in self._failure_archetypes],
                "strategy_insights": [s.to_dict() for s in self._strategy_insights],
                "distilled_at": time.time(),
            }
            self._persistence_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            logger.info(
                f"[KnowledgeDistiller] Saved {len(self._principles)} principles"
            )
        except Exception as e:
            logger.warning(f"[KnowledgeDistiller] Save failed: {e}")

    def _load_state(self) -> None:
        """Load persisted distilled knowledge."""
        if not self._persistence_path or not self._persistence_path.exists():
            return
        try:
            data = json.loads(
                self._persistence_path.read_text(encoding="utf-8")
            )
            self._principles = [
                DistilledPrinciple.from_dict(p)
                for p in data.get("principles", [])
            ]
            self._failure_archetypes = [
                FailureArchetype(**{
                    k: v for k, v in a.items()
                    if k in FailureArchetype.__dataclass_fields__
                })
                for a in data.get("failure_archetypes", [])
            ]
            self._strategy_insights = [
                StrategyInsight(**{
                    k: v for k, v in s.items()
                    if k in StrategyInsight.__dataclass_fields__
                })
                for s in data.get("strategy_insights", [])
            ]
            logger.info(
                f"[KnowledgeDistiller] Loaded {len(self._principles)} principles"
            )
        except Exception as e:
            logger.warning(f"[KnowledgeDistiller] Load failed: {e}")

    # ──── Utilities ──────────────────────────────

    @staticmethod
    def _variance(values: list[float]) -> float:
        """Compute population variance."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    @staticmethod
    def _pearson_correlation(x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 3 or len(y) != n:
            return 0.0
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denom_x = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
        denom_y = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))
        denominator = denom_x * denom_y
        if denominator < 1e-10:
            return 0.0
        return numerator / denominator
