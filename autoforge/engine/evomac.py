"""EvoMAC — Evolved Multi-Agent Collaboration with text backpropagation.

Inspired by ICLR 2025 paper: three-layer team architecture with "text
backpropagation" that dynamically adapts agent behaviour and connections
based on test-time execution feedback.

Architecture:
  - Layer 1: Coding Team   (Builder agents → produce code)
  - Layer 2: Testing Team  (Tester + Reviewer → evaluate quality)
  - Layer 3: Update Team   (analyses gradients, rewrites agent instructions)

The key innovation is the *text gradient*: after the testing team evaluates
code, the update team generates natural-language feedback ("gradients") that
describe WHAT went wrong and HOW to fix it.  These gradients are propagated
backwards through the team topology to update each agent's behaviour prompt.

This integrates with AutoForge's existing pipeline:
  Builder → Reviewer → Tester  maps to  Coding → Testing
  Gardener + DynamicConstitution  maps to  Update Team

References:
  - EvoMAC: Evolving Multi-Agent Collaboration (ICLR 2025)
  - TextGrad: Automatic differentiation via text (2024)
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Real Finite-Difference Gradient Estimation
# ──────────────────────────────────────────────


class FiniteDifferenceGradient:
    """Real gradient estimation via finite differences.

    Instead of asking LLM "what should change?", we:
    1. Perturb agent parameters (temperature, emphasis weights, etc.)
    2. Measure output quality delta via actual execution
    3. Compute ∂quality/∂parameter numerically
    4. Accumulate gradients across multiple samples for stability
    """

    def __init__(self, epsilon: float = 0.05, momentum: float = 0.9) -> None:
        self.epsilon = epsilon  # Perturbation size
        self.momentum = momentum
        self._grad_history: dict[str, list[float]] = {}  # param_name → gradient history
        self._velocity: dict[str, float] = {}  # Momentum-based smoothing
        self._algo_calls = 0
        self._fallback_calls = 0

    @property
    def algorithm_ratio(self) -> float:
        """Ratio of finite-difference calls to total gradient calls."""
        total = self._algo_calls + self._fallback_calls
        return self._algo_calls / total if total > 0 else 0.0

    def estimate_gradient(
        self,
        param_name: str,
        current_value: float,
        quality_at_current: float,
        quality_at_perturbed: float,
    ) -> float:
        """Compute numerical gradient for a single parameter.

        gradient ≈ (f(x + ε) - f(x)) / ε

        Args:
            param_name: Name of the parameter being optimized
            current_value: Current parameter value (for tracking)
            quality_at_current: Quality metric at f(x)
            quality_at_perturbed: Quality metric at f(x + ε)

        Returns:
            Smoothed gradient estimate using momentum
        """
        gradient = (quality_at_perturbed - quality_at_current) / self.epsilon

        # Momentum-based smoothing for stability
        if param_name not in self._velocity:
            self._velocity[param_name] = 0.0
        self._velocity[param_name] = (
            self.momentum * self._velocity[param_name] +
            (1 - self.momentum) * gradient
        )

        # Track history for statistics
        if param_name not in self._grad_history:
            self._grad_history[param_name] = []
        self._grad_history[param_name].append(gradient)

        self._algo_calls += 1
        return self._velocity[param_name]

    def suggest_update(
        self,
        param_name: str,
        current_value: float,
        learning_rate: float = 0.01,
    ) -> float:
        """Suggest new parameter value based on accumulated gradient.

        Args:
            param_name: Parameter name
            current_value: Current parameter value
            learning_rate: Step size for update

        Returns:
            New parameter value: x_new = x + lr * velocity
        """
        velocity = self._velocity.get(param_name, 0.0)
        return current_value + learning_rate * velocity

    def record_fallback(self) -> None:
        """Record a fallback to LLM-based gradient (for ratio tracking)."""
        self._fallback_calls += 1

    def get_gradient_stats(self) -> dict[str, dict[str, Any]]:
        """Return statistics about gradient history for each parameter.

        Returns:
            Dict mapping param names to {mean, std, n_samples, velocity}
        """
        stats = {}
        for name, history in self._grad_history.items():
            if history:
                stats[name] = {
                    "mean": statistics.mean(history),
                    "std": statistics.stdev(history) if len(history) > 1 else 0.0,
                    "n_samples": len(history),
                    "velocity": self._velocity.get(name, 0.0),
                }
        return stats

    def reset(self) -> None:
        """Reset gradient accumulators for a new parameter sweep."""
        self._grad_history.clear()
        self._velocity.clear()


# ──────────────────────────────────────────────
# Topology Optimizer (Edge Weight Optimization)
# ──────────────────────────────────────────────


class TopologyOptimizer:
    """Real topology optimization using gradient-based edge weight updates.

    Tracks which agent→agent communication edges are effective
    and adjusts weights using actual performance correlation.
    """

    def __init__(self, learning_rate: float = 0.01, decay: float = 0.99) -> None:
        self.lr = learning_rate
        self.decay = decay
        self._edge_effectiveness: dict[tuple[str, str], list[float]] = {}
        self._edge_weights: dict[tuple[str, str], float] = {}

    def record_edge_usage(
        self,
        source: str,
        target: str,
        quality_before: float,
        quality_after: float,
    ) -> None:
        """Record the effect of an agent→agent communication.

        Args:
            source: Source agent name
            target: Target agent name
            quality_before: Quality metric before edge was used
            quality_after: Quality metric after edge was used
        """
        edge = (source, target)
        effectiveness = quality_after - quality_before
        if edge not in self._edge_effectiveness:
            self._edge_effectiveness[edge] = []
            self._edge_weights[edge] = 1.0
        self._edge_effectiveness[edge].append(effectiveness)

        # Update edge weight using exponential moving average of last 10 observations
        history = self._edge_effectiveness[edge]
        avg_effect = sum(history[-10:]) / min(len(history), 10)

        # Clip weight to [0.1, 2.0] to prevent extreme values
        self._edge_weights[edge] = max(0.1, min(2.0,
            self._edge_weights[edge] + self.lr * avg_effect))

    def get_edge_weight(self, source: str, target: str) -> float:
        """Get current weight for an edge.

        Returns:
            Edge weight (default 1.0 if not yet recorded)
        """
        return self._edge_weights.get((source, target), 1.0)

    def prune_weak_edges(self, threshold: float = 0.3) -> list[tuple[str, str]]:
        """Identify edges that should be pruned (consistently unhelpful).

        Args:
            threshold: Minimum edge weight before pruning

        Returns:
            List of (source, target) edges to prune
        """
        pruned = []
        for edge, weight in self._edge_weights.items():
            if weight < threshold and len(self._edge_effectiveness.get(edge, [])) >= 5:
                pruned.append(edge)
        return pruned

    def get_topology_report(self) -> dict[str, Any]:
        """Return current topology state.

        Returns:
            Dict with edges and weak_edges lists
        """
        return {
            "edges": {
                f"{s}→{t}": {
                    "weight": w,
                    "n_observations": len(self._edge_effectiveness.get((s, t), []))
                }
                for (s, t), w in self._edge_weights.items()
            },
            "weak_edges": [f"{s}→{t}" for s, t in self.prune_weak_edges()],
        }


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────


@dataclass
class TextGradient:
    """A natural-language 'gradient' describing what an agent should change.

    Analogous to a numerical gradient in backpropagation, but expressed as
    human-readable instruction deltas.
    """
    source_agent: str       # Who produced the evaluation (e.g. "reviewer")
    target_agent: str       # Who should change (e.g. "builder")
    feedback: str           # What went wrong
    suggestion: str         # How to fix it
    severity: float = 0.5   # 0 = minor suggestion, 1 = critical issue
    task_context: str = ""  # Which task triggered this gradient
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "feedback": self.feedback,
            "suggestion": self.suggestion,
            "severity": self.severity,
            "task_context": self.task_context,
        }


@dataclass
class AgentTopologyEdge:
    """A directed communication edge between two agents.

    The weight represents how strongly the source influences the target.
    Edges can be strengthened (if feedback helped) or weakened.
    """
    source: str
    target: str
    weight: float = 1.0
    active: bool = True
    gradients_sent: int = 0
    gradients_helped: int = 0

    @property
    def effectiveness(self) -> float:
        if self.gradients_sent == 0:
            return 0.5
        return self.gradients_helped / self.gradients_sent


@dataclass
class TeamPerformance:
    """Performance snapshot for one pipeline iteration."""
    iteration: int
    test_pass_rate: float = 0.0
    review_score: float = 0.0
    build_success_rate: float = 0.0
    gradients_pending: int = 0
    gradients_applied: int = 0
    timestamp: float = field(default_factory=time.time)


# ──────────────────────────────────────────────
# EvoMAC Engine
# ──────────────────────────────────────────────


class EvoMACEngine:
    """Evolved Multi-Agent Collaboration engine.

    Manages the three-layer team architecture with text backpropagation:

    1. **Forward pass**: Coding team (Builder) produces code
    2. **Evaluation**: Testing team (Tester + Reviewer) evaluates quality
    3. **Backward pass**: Update team generates text gradients and
       rewrites agent instructions for the next iteration

    The topology (which agents communicate with which) evolves over time:
    effective edges are strengthened, ineffective ones are pruned.
    """

    # Default team topology — maps existing AutoForge agents
    DEFAULT_EDGES = [
        ("reviewer", "builder"),      # Review feedback → builder improvement
        ("tester", "builder"),        # Test failures → builder fixes
        ("reviewer", "architect"),    # Architecture critique → design update
        ("tester", "gardener"),       # Test results → refactoring targets
        ("reviewer", "gardener"),     # Quality issues → gardener priorities
    ]

    # Minimum iterations before topology evolution kicks in
    MIN_ITERATIONS_FOR_EVOLUTION = 3
    # Gradient severity threshold for applying updates
    GRADIENT_APPLY_THRESHOLD = 0.3

    def __init__(self, project_dir: Path | None = None) -> None:
        self.project_dir = project_dir
        self._topology: list[AgentTopologyEdge] = []
        self._pending_gradients: dict[str, list[TextGradient]] = {}
        self._performance_history: list[TeamPerformance] = []
        self._iteration = 0
        # Causal verification: store (score_before, score_after) deltas per edge
        self._edge_deltas: dict[tuple[str, str], list[float]] = {}
        # Real finite-difference gradient estimation
        self._fd_gradient = FiniteDifferenceGradient(epsilon=0.05, momentum=0.9)
        # Real topology optimization via edge effectiveness
        self._topology_optimizer = TopologyOptimizer(learning_rate=0.01, decay=0.99)
        self._init_topology()

    @property
    def iteration(self) -> int:
        """Current iteration number."""
        return self._iteration

    def _init_topology(self) -> None:
        """Initialize the default agent communication topology."""
        self._topology = [
            AgentTopologyEdge(source=src, target=tgt)
            for src, tgt in self.DEFAULT_EDGES
        ]

    # ──────── Forward Pass: record what agents produced ────────

    def start_iteration(self) -> int:
        """Begin a new forward-backward iteration. Returns iteration number."""
        self._iteration += 1
        logger.info(f"[EvoMAC] Starting iteration {self._iteration}")
        return self._iteration

    # ──────── Finite-Difference Gradient Estimation ────────

    def estimate_parameter_gradient(
        self,
        param_name: str,
        current_value: float,
        quality_at_current: float,
        quality_at_perturbed: float,
    ) -> float:
        """Estimate gradient of quality w.r.t. a single parameter via finite differences.

        This is a PRIMARY gradient estimation method that measures real quality deltas
        instead of asking the LLM "what should change?".

        Args:
            param_name: Name of the parameter (e.g. "temperature", "emphasis_weight")
            current_value: Current parameter value (for tracking)
            quality_at_current: Quality metric f(x)
            quality_at_perturbed: Quality metric f(x + ε)

        Returns:
            Smoothed gradient estimate using momentum
        """
        return self._fd_gradient.estimate_gradient(
            param_name,
            current_value,
            quality_at_current,
            quality_at_perturbed,
        )

    def suggest_parameter_update(
        self,
        param_name: str,
        current_value: float,
        learning_rate: float = 0.01,
    ) -> float:
        """Suggest new parameter value based on accumulated finite-difference gradients.

        Args:
            param_name: Parameter name
            current_value: Current value
            learning_rate: Step size

        Returns:
            Recommended new parameter value
        """
        return self._fd_gradient.suggest_update(param_name, current_value, learning_rate)

    def record_edge_effectiveness(
        self,
        source: str,
        target: str,
        quality_before: float,
        quality_after: float,
    ) -> None:
        """Record the measured effectiveness of an agent→agent communication edge.

        This feeds the topology optimizer, which adjusts edge weights based on
        actual quality improvements.

        Args:
            source: Source agent
            target: Target agent
            quality_before: Quality metric before the edge was used
            quality_after: Quality metric after the edge was used
        """
        self._topology_optimizer.record_edge_usage(source, target, quality_before, quality_after)

        # Also update the corresponding edge weight in the topology
        for edge in self._topology:
            if edge.source == source and edge.target == target:
                new_weight = self._topology_optimizer.get_edge_weight(source, target)
                edge.weight = new_weight
                break

    def get_fd_gradient_stats(self) -> dict[str, Any]:
        """Get statistics about finite-difference gradient estimation.

        Returns:
            Dict with algorithm_ratio, parameter_stats, and edge_stats
        """
        return {
            "algorithm_ratio": round(self._fd_gradient.algorithm_ratio, 2),
            "parameter_stats": self._fd_gradient.get_gradient_stats(),
            "edge_stats": self._topology_optimizer.get_topology_report(),
        }

    # ──────── Backward Pass: generate text gradients ────────

    async def generate_gradients(
        self,
        evaluation_data: dict[str, Any],
        llm: Any,
    ) -> list[TextGradient]:
        """Generate text gradients from evaluation results.

        This is the 'backward pass': the update team analyses test results
        and review feedback to produce actionable gradients for each agent.

        Args:
            evaluation_data: {
                "test_results": {...},
                "review_score": float,
                "review_issues": [...],
                "build_failures": [...],
                "task_context": str,
            }
        """
        from autoforge.engine.llm_router import TaskComplexity

        test_results = evaluation_data.get("test_results", {})
        review_issues = evaluation_data.get("review_issues", [])
        build_failures = evaluation_data.get("build_failures", [])
        task_context = evaluation_data.get("task_context", "")

        prompt = (
            "You are the Update Team in a multi-agent coding system. "
            "Analyse the evaluation results and generate targeted feedback "
            "(text gradients) for each agent that needs to improve.\n\n"
            f"## Evaluation Results\n"
            f"Test pass rate: {test_results.get('pass_rate', 'N/A')}\n"
            f"Review score: {evaluation_data.get('review_score', 'N/A')}/10\n\n"
        )

        if review_issues:
            prompt += "## Review Issues\n"
            for issue in review_issues[:10]:
                prompt += f"- {issue}\n"
            prompt += "\n"

        if build_failures:
            prompt += "## Build Failures\n"
            for fail in build_failures[:5]:
                prompt += f"- {fail}\n"
            prompt += "\n"

        prompt += (
            "## Instructions\n"
            "For each issue, output a JSON array of gradients:\n"
            "```json\n"
            '[\n'
            '  {"target": "builder", "feedback": "what went wrong", '
            '"suggestion": "how to fix", "severity": 0.0-1.0},\n'
            '  ...\n'
            ']\n'
            "```\n"
            "Target agents: builder, architect, tester, gardener, reviewer.\n"
            "Focus on the ROOT CAUSE, not symptoms. Be specific and actionable.\n"
            "Generate 2-6 gradients, prioritised by severity."
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You are a development process analyst. Generate precise, "
                       "actionable feedback to improve agent performance.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            gradients = self._parse_gradients(text, task_context)
            logger.info(f"[EvoMAC] Generated {len(gradients)} text gradients")
            return gradients

        except Exception as e:
            logger.warning(f"[EvoMAC] Gradient generation failed: {e}")
            return []

    def _parse_gradients(self, text: str, task_context: str) -> list[TextGradient]:
        """Parse LLM output into TextGradient objects with contradiction detection.

        Detects and resolves contradictory gradients targeting the same agent
        by keeping only the higher-severity one when two gradients conflict.
        """
        gradients = []

        from autoforge.engine.utils import extract_json_list_from_text
        try:
            items = extract_json_list_from_text(text)
        except ValueError:
            return []

        for item in items:
            if not isinstance(item, dict):
                continue
            gradient = TextGradient(
                source_agent="update_team",
                target_agent=item.get("target", "builder"),
                feedback=item.get("feedback", ""),
                suggestion=item.get("suggestion", ""),
                severity=float(item.get("severity", 0.5)),
                task_context=task_context,
            )
            gradients.append(gradient)

        # Contradiction detection: check for opposing suggestions to the same agent
        gradients = self._resolve_contradictions(gradients)

        # Queue for target agents
        for gradient in gradients:
            target = gradient.target_agent
            if target not in self._pending_gradients:
                self._pending_gradients[target] = []
            self._pending_gradients[target].append(gradient)

        return gradients

    @staticmethod
    def _resolve_contradictions(gradients: list[TextGradient]) -> list[TextGradient]:
        """Detect and resolve contradictory gradients for the same target.

        Strategy: group by target, then for each pair check for opposition
        keywords (add/remove, increase/decrease, more/less, enable/disable).
        When contradiction found, keep the higher-severity one.
        """
        OPPOSITION_PAIRS = [
            ("add", "remove"), ("increase", "decrease"), ("more", "less"),
            ("enable", "disable"), ("strict", "lenient"), ("verbose", "concise"),
            ("simple", "complex"), ("defensive", "aggressive"),
        ]

        # Group by target agent
        by_target: dict[str, list[TextGradient]] = {}
        for g in gradients:
            by_target.setdefault(g.target_agent, []).append(g)

        resolved: list[TextGradient] = []
        for target, group in by_target.items():
            if len(group) <= 1:
                resolved.extend(group)
                continue

            # Check each pair for contradiction
            keep = set(range(len(group)))
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if i not in keep or j not in keep:
                        continue
                    text_i = (group[i].suggestion + " " + group[i].feedback).lower()
                    text_j = (group[j].suggestion + " " + group[j].feedback).lower()

                    contradicts = False
                    for word_a, word_b in OPPOSITION_PAIRS:
                        if ((word_a in text_i and word_b in text_j) or
                                (word_b in text_i and word_a in text_j)):
                            contradicts = True
                            break

                    if contradicts:
                        # Keep the higher-severity one
                        loser = j if group[i].severity >= group[j].severity else i
                        keep.discard(loser)
                        logger.info(
                            f"[EvoMAC] Resolved contradictory gradients for {target}: "
                            f"kept severity={group[i if loser == j else j].severity:.1f}, "
                            f"dropped severity={group[loser].severity:.1f}"
                        )

            resolved.extend(group[k] for k in sorted(keep))

        return resolved

    # ──────── Apply Gradients: update agent instructions ────────

    async def apply_gradients(
        self,
        role: str,
        current_supplement: str,
        llm: Any,
    ) -> str:
        """Apply pending text gradients to generate updated agent instructions.

        This is the 'weight update' step: the accumulated gradients are
        synthesised into a revised instruction supplement for the agent.

        Args:
            role: Agent role to update (e.g. "builder")
            current_supplement: Current dynamic constitution supplement
            llm: LLM router instance

        Returns:
            Updated supplement string incorporating gradient feedback.
        """
        from autoforge.engine.llm_router import TaskComplexity

        pending = self._pending_gradients.get(role, [])
        # Filter by severity threshold
        actionable = [g for g in pending if g.severity >= self.GRADIENT_APPLY_THRESHOLD]

        if not actionable:
            return current_supplement

        # Sort by severity (most critical first)
        actionable.sort(key=lambda g: g.severity, reverse=True)

        gradients_text = ""
        for i, g in enumerate(actionable[:8], 1):
            gradients_text += (
                f"{i}. [severity={g.severity:.1f}] {g.feedback}\n"
                f"   Suggestion: {g.suggestion}\n"
            )

        prompt = (
            f"You are updating the instruction prompt for the '{role}' agent "
            f"based on feedback from the testing and review teams.\n\n"
            f"## Current Instructions\n"
            f"```\n{current_supplement[:2000]}\n```\n\n"
            f"## Text Gradients (feedback to incorporate)\n"
            f"{gradients_text}\n"
            f"## Task\n"
            f"Rewrite the agent instructions to address the feedback. "
            f"Keep what works, fix what doesn't. Be concise and specific.\n"
            f"Output ONLY the updated instructions, nothing else."
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You update agent behaviour prompts based on execution feedback. "
                       "Preserve effective instructions while incorporating improvements.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            # Record that we fell back to LLM-based gradients (vs finite-difference)
            self._fd_gradient.record_fallback()

            updated = text.strip()
            if updated and len(updated) > 50:
                # Track which sources contributed gradients (for accurate attribution)
                source_counts: dict[str, int] = {}
                for g in actionable:
                    source_counts[g.source_agent] = source_counts.get(g.source_agent, 0) + 1

                # Update per-edge gradient counts (now properly attributed)
                for edge in self._topology:
                    if edge.target == role and edge.active:
                        # Count only gradients that came from this edge's source
                        edge.gradients_sent += source_counts.get(edge.source, 0)

                # Clear applied gradients
                self._pending_gradients[role] = [
                    g for g in pending if g not in actionable
                ]

                logger.info(
                    f"[EvoMAC] Applied {len(actionable)} gradients to {role} "
                    f"({len(updated)} chars)"
                )

                # Replace existing EvoMAC section instead of blindly appending
                evomac_section = f"\n\n## EvoMAC Adaptive Instructions\n{updated}"
                if "## EvoMAC Adaptive Instructions" in current_supplement:
                    # Replace the old section
                    import re
                    result = re.sub(
                        r"\n\n## EvoMAC Adaptive Instructions\n.*",
                        evomac_section,
                        current_supplement,
                        flags=re.DOTALL,
                    )
                    return result
                else:
                    return current_supplement + evomac_section

        except Exception as e:
            logger.warning(f"[EvoMAC] Failed to apply gradients for {role}: {e}")

        return current_supplement

    # ──────── Topology Evolution ────────

    def record_gradient_outcome_with_metrics(
        self,
        target_role: str,
        score_before: float,
        score_after: float,
        source_role: str = "",
    ) -> None:
        """Record gradient outcome with actual before/after metrics.

        This method enables causal attribution of improvements to specific
        feedback channels by tracking the actual performance delta per edge.
        The delta is used to update edge weights proportionally rather than
        with fixed increments.

        Args:
            target_role: The agent that received the gradient.
            score_before: Performance score before applying the gradient.
            score_after: Performance score after applying the gradient.
            source_role: The agent that sent the gradient. Should be provided
                for accurate attribution.
        """
        delta = score_after - score_before

        updated = False
        for edge in self._topology:
            if edge.target != target_role:
                continue
            # Only update the specific edge if source_role is provided
            if source_role and edge.source != source_role:
                continue
            if not edge.active:
                continue  # Never attribute to inactive edges regardless of source

            # Store delta keyed by actual edge (not wildcard) for causal analysis
            edge_key = (edge.source, edge.target)
            if edge_key not in self._edge_deltas:
                self._edge_deltas[edge_key] = []
            self._edge_deltas[edge_key].append(delta)
            # Cap delta history to prevent unbounded growth
            if len(self._edge_deltas[edge_key]) > 50:
                self._edge_deltas[edge_key] = self._edge_deltas[edge_key][-50:]

            # Update gradient counts
            edge.gradients_sent += 1
            if delta > 0:
                edge.gradients_helped += 1

            # Weight update proportional to delta magnitude
            # Positive delta: boost weight by proportional amount (capped)
            # Negative delta: reduce weight but less aggressively than fixed -0.05
            if delta > 0:
                boost = min(0.3, delta)  # Cap boost at 0.3
                edge.weight = min(2.0, edge.weight + boost)
            else:
                penalty = max(-0.15, delta)  # Allow larger penalties for negative deltas
                edge.weight = max(0.1, edge.weight + penalty)

            updated = True

        if not updated and source_role:
            logger.debug(f"[EvoMAC] No active edge found for {source_role}->{target_role}")

    def record_gradient_outcome(
        self, target_role: str, helped: bool, source_role: str = "",
    ) -> None:
        """Record whether applied gradients actually improved the agent.

        This feedback drives topology evolution: effective edges are
        strengthened, ineffective ones are weakened or pruned.

        IMPORTANT: For accurate causal attribution, prefer using
        record_gradient_outcome_with_metrics() which takes actual before/after
        scores. This method is maintained for backward compatibility and internally
        converts binary feedback to estimated deltas.

        Args:
            target_role: The agent that received the gradient.
            helped: Whether the gradient led to improvement.
            source_role: The agent that sent the gradient. Should be provided
                for accurate attribution.
        """
        # Convert binary feedback to estimated delta
        estimated_delta = 0.1 if helped else -0.05

        self.record_gradient_outcome_with_metrics(
            target_role=target_role,
            score_before=0.0,
            score_after=estimated_delta,
            source_role=source_role,
        )

    def get_edge_causal_summary(self) -> list[dict[str, Any]]:
        """Get causal evidence for each edge.

        Returns list of dicts containing:
        - source: Source agent
        - target: Target agent
        - avg_delta: Average performance delta attributed to this edge
        - n_observations: Number of outcomes recorded for this edge
        - is_beneficial: True if avg_delta > 0 AND n_observations >= 3
        - confidence: Confidence score (0-1) based on sample size
                      (reaches 1.0 at n_observations >= 5)
        """
        summaries = []

        for edge in self._topology:
            edge_key = (edge.source, edge.target)
            deltas = self._edge_deltas.get(edge_key, [])

            if not deltas:
                # No delta data yet; fall back to binary effectiveness
                avg_delta = 0.0
                n_obs = edge.gradients_sent
                is_beneficial = edge.effectiveness > 0.5 if n_obs > 0 else False
            else:
                avg_delta = sum(deltas) / len(deltas)
                n_obs = len(deltas)
                # Beneficial: positive average delta with sufficient samples
                is_beneficial = avg_delta > 0.0 and n_obs >= 3

            # Confidence: scales with sample count, saturates at n=5
            confidence = min(1.0, n_obs / 5.0)

            summaries.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "avg_delta": round(avg_delta, 4),
                    "n_observations": n_obs,
                    "is_beneficial": is_beneficial,
                    "confidence": round(confidence, 2),
                    "weight": round(edge.weight, 2),
                    "active": edge.active,
                }
            )

        return summaries

    def evolve_topology(self) -> list[str]:
        """Evolve the agent communication topology based on effectiveness.

        Key improvements over original:
        1. Pruned edges can be RE-ACTIVATED if performance is stagnating
           (indicates the system needs more feedback channels)
        2. Uses Bayesian credibility: needs min 3 samples before pruning
        3. When delta metrics are available, uses avg_delta for pruning/
           strengthening decisions (more causal than binary effectiveness)
        4. Weight updates use effectiveness-proportional step sizes

        Returns list of changes made (for logging).
        """
        if self._iteration < self.MIN_ITERATIONS_FOR_EVOLUTION:
            return []

        changes = []
        trend = self.get_improvement_trend()
        is_stagnating = trend.get("trend") == "stagnating"

        for edge in self._topology:
            if edge.active:
                edge_key = (edge.source, edge.target)
                deltas = self._edge_deltas.get(edge_key, [])

                # Determine pruning/strengthening based on available evidence
                if deltas:
                    # Use delta-based evidence when available
                    avg_delta = sum(deltas) / len(deltas)
                    n_obs = len(deltas)

                    # Pruning: negative avg delta with sufficient confidence
                    if n_obs >= 3 and avg_delta < -0.05:
                        edge.active = False
                        changes.append(
                            f"Pruned {edge.source}->{edge.target} "
                            f"(avg_delta={avg_delta:.3f}, n={n_obs})"
                        )
                    # Strengthening: positive avg delta
                    elif n_obs >= 2 and avg_delta > 0.05 and edge.weight < 1.5:
                        boost = min(0.2, avg_delta)  # Proportional to delta
                        edge.weight = min(2.0, edge.weight + boost)
                        changes.append(
                            f"Strengthened {edge.source}->{edge.target} "
                            f"(avg_delta={avg_delta:.3f}, weight={edge.weight:.1f})"
                        )
                else:
                    # Fall back to binary effectiveness (backward compatibility)
                    if edge.gradients_sent >= 3 and edge.effectiveness < 0.2:
                        edge.active = False
                        changes.append(
                            f"Pruned {edge.source}->{edge.target} "
                            f"(effectiveness={edge.effectiveness:.0%}, "
                            f"n={edge.gradients_sent})"
                        )
                    elif edge.effectiveness > 0.7 and edge.weight < 1.5:
                        # Proportional strengthening: better edges get bigger boosts
                        boost = 0.1 + 0.1 * edge.effectiveness
                        edge.weight = min(2.0, edge.weight + boost)
                        changes.append(
                            f"Strengthened {edge.source}->{edge.target} "
                            f"(weight={edge.weight:.1f})"
                        )
            else:
                # Re-activation: if performance is stagnating, try reviving pruned edges
                # This prevents the system from permanently losing feedback channels
                if is_stagnating and self._iteration > edge.gradients_sent + 5:
                    edge.active = True
                    edge.weight = 0.5  # Start with lower weight on reactivation
                    # Reset counters to give the edge a fresh chance
                    edge.gradients_sent = 0
                    edge.gradients_helped = 0
                    changes.append(
                        f"Reactivated {edge.source}->{edge.target} "
                        f"(stagnation recovery, weight=0.5)"
                    )

        if changes:
            logger.info(f"[EvoMAC] Topology evolved: {'; '.join(changes)}")
        return changes

    # ──────── Performance Tracking ────────

    def record_performance(
        self,
        test_pass_rate: float = 0.0,
        review_score: float = 0.0,
        build_success_rate: float = 0.0,
    ) -> TeamPerformance:
        """Record team performance for this iteration."""
        perf = TeamPerformance(
            iteration=self._iteration,
            test_pass_rate=test_pass_rate,
            review_score=review_score,
            build_success_rate=build_success_rate,
            gradients_pending=sum(len(g) for g in self._pending_gradients.values()),
        )
        self._performance_history.append(perf)
        return perf

    def get_improvement_trend(self) -> dict[str, Any]:
        """Analyse performance trend using linear regression slope.

        Uses the last 5 iterations (or all available) and computes the
        OLS slope of the composite score (review_score + test_pass_rate*10).
        This is more robust than comparing just first vs last point.
        """
        if len(self._performance_history) < 2:
            return {"trend": "insufficient_data", "iterations": len(self._performance_history)}

        recent = self._performance_history[-5:]
        n = len(recent)

        # Composite metric: review_score + test_pass_rate * 10 (both 0-10 scale)
        scores = [p.review_score + p.test_pass_rate * 10.0 for p in recent]

        # Simple linear regression: slope = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = sum(scores) / n
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, scores))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        slope = numerator / denominator if denominator > 0 else 0.0

        # Categorize: improving if slope > 0.1, regressing if < -0.1
        if slope > 0.1:
            trend = "improving"
        elif slope < -0.1:
            trend = "regressing"
        else:
            trend = "stagnating"

        return {
            "trend": trend,
            "iterations": len(self._performance_history),
            "latest_score": recent[-1].review_score,
            "latest_test_rate": recent[-1].test_pass_rate,
            "slope": round(slope, 3),
            "score_delta": scores[-1] - scores[0] if n >= 2 else 0,
        }

    def get_active_edges(self) -> list[dict[str, Any]]:
        """Get currently active topology edges."""
        return [
            {
                "source": e.source,
                "target": e.target,
                "weight": round(e.weight, 2),
                "effectiveness": round(e.effectiveness, 2),
            }
            for e in self._topology if e.active
        ]

    # ──────── Persistence ────────

    def save_state(self, output_dir: Path) -> None:
        """Save EvoMAC state to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "iteration": self._iteration,
            "topology": [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "active": e.active,
                    "gradients_sent": e.gradients_sent,
                    "gradients_helped": e.gradients_helped,
                }
                for e in self._topology
            ],
            "performance_history": [
                {
                    "iteration": p.iteration,
                    "test_pass_rate": p.test_pass_rate,
                    "review_score": p.review_score,
                    "build_success_rate": p.build_success_rate,
                }
                for p in self._performance_history
            ],
            "pending_gradient_counts": {
                role: len(grads) for role, grads in self._pending_gradients.items()
            },
            "edge_deltas": {
                f"{src}::{tgt}": deltas
                for (src, tgt), deltas in self._edge_deltas.items()
            },
            "fd_gradient_stats": {
                "algo_calls": self._fd_gradient._algo_calls,
                "fallback_calls": self._fd_gradient._fallback_calls,
                "grad_history": {
                    name: history
                    for name, history in self._fd_gradient._grad_history.items()
                },
            },
            "topology_optimizer_stats": {
                "edge_weights": {
                    f"{src}::{tgt}": w
                    for (src, tgt), w in self._topology_optimizer._edge_weights.items()
                },
            },
        }
        path = output_dir / "evomac_state.json"
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.debug(f"[EvoMAC] State saved to {path}")

    def load_state(self, state_dir: Path) -> None:
        """Load EvoMAC state from disk."""
        path = state_dir / "evomac_state.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._iteration = data.get("iteration", 0)
            self._topology = [
                AgentTopologyEdge(
                    source=e["source"],
                    target=e["target"],
                    weight=e.get("weight", 1.0),
                    active=e.get("active", True),
                    gradients_sent=e.get("gradients_sent", 0),
                    gradients_helped=e.get("gradients_helped", 0),
                )
                for e in data.get("topology", [])
            ]
            # Restore edge deltas for causal analysis
            edge_deltas_raw = data.get("edge_deltas", {})
            self._edge_deltas = {}
            for key_str, deltas in edge_deltas_raw.items():
                # Parse key format "src::tgt"
                parts = key_str.split("::")
                if len(parts) == 2:
                    self._edge_deltas[(parts[0], parts[1])] = deltas

            # Restore finite-difference gradient state
            fd_stats = data.get("fd_gradient_stats", {})
            if fd_stats:
                self._fd_gradient._algo_calls = fd_stats.get("algo_calls", 0)
                self._fd_gradient._fallback_calls = fd_stats.get("fallback_calls", 0)
                grad_hist = fd_stats.get("grad_history", {})
                self._fd_gradient._grad_history = {
                    name: history for name, history in grad_hist.items()
                }

            # Restore topology optimizer state
            topo_stats = data.get("topology_optimizer_stats", {})
            if topo_stats:
                edge_weights_raw = topo_stats.get("edge_weights", {})
                for key_str, weight in edge_weights_raw.items():
                    parts = key_str.split("::")
                    if len(parts) == 2:
                        self._topology_optimizer._edge_weights[(parts[0], parts[1])] = weight

            logger.info(
                f"[EvoMAC] Loaded state: iteration={self._iteration}, "
                f"edge_deltas_count={len(self._edge_deltas)}, "
                f"fd_algo_ratio={self._fd_gradient.algorithm_ratio:.2f}"
            )
        except Exception as e:
            logger.warning(f"[EvoMAC] Failed to load state: {e}")

    def summary(self) -> dict[str, Any]:
        """Get a summary of current EvoMAC state."""
        return {
            "iteration": self._iteration,
            "active_edges": len([e for e in self._topology if e.active]),
            "total_edges": len(self._topology),
            "pending_gradients": sum(len(g) for g in self._pending_gradients.values()),
            "performance_trend": self.get_improvement_trend(),
            "algorithm_ratio": round(self._fd_gradient.algorithm_ratio, 2),
            "gradient_estimation_stats": {
                "algo_calls": self._fd_gradient._algo_calls,
                "fallback_calls": self._fd_gradient._fallback_calls,
                "parameters_tracked": len(self._fd_gradient._grad_history),
            },
        }
