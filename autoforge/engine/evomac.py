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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    gradients_generated: int = 0
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
    # Maximum accumulated gradients per agent before forced update
    MAX_PENDING_GRADIENTS = 10

    def __init__(self, project_dir: Path | None = None) -> None:
        self.project_dir = project_dir
        self._topology: list[AgentTopologyEdge] = []
        self._pending_gradients: dict[str, list[TextGradient]] = {}
        self._performance_history: list[TeamPerformance] = []
        self._iteration = 0
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
        """Parse LLM output into TextGradient objects."""
        gradients = []

        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            raw_text = match.group(1).strip()
        else:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                raw_text = text[start:end + 1]
            else:
                return []

        try:
            items = json.loads(raw_text)
        except json.JSONDecodeError:
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

            # Queue for the target agent
            target = gradient.target_agent
            if target not in self._pending_gradients:
                self._pending_gradients[target] = []
            self._pending_gradients[target].append(gradient)

        return gradients

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

            updated = text.strip()
            if updated and len(updated) > 50:
                # Mark gradients as applied
                for edge in self._topology:
                    if edge.target == role and edge.active:
                        edge.gradients_sent += len(actionable)

                # Clear applied gradients
                self._pending_gradients[role] = [
                    g for g in pending if g not in actionable
                ]

                logger.info(
                    f"[EvoMAC] Applied {len(actionable)} gradients to {role} "
                    f"({len(updated)} chars)"
                )
                return f"\n\n## EvoMAC Adaptive Instructions\n{updated}"

        except Exception as e:
            logger.warning(f"[EvoMAC] Failed to apply gradients for {role}: {e}")

        return current_supplement

    # ──────── Topology Evolution ────────

    def record_gradient_outcome(self, target_role: str, helped: bool) -> None:
        """Record whether applied gradients actually improved the agent.

        This feedback drives topology evolution: effective edges are
        strengthened, ineffective ones are weakened or pruned.
        """
        for edge in self._topology:
            if edge.target == target_role and edge.active:
                if helped:
                    edge.gradients_helped += 1
                    edge.weight = min(2.0, edge.weight + 0.1)
                else:
                    edge.weight = max(0.1, edge.weight - 0.05)

    def evolve_topology(self) -> list[str]:
        """Evolve the agent communication topology based on effectiveness.

        Returns list of changes made (for logging).
        """
        if self._iteration < self.MIN_ITERATIONS_FOR_EVOLUTION:
            return []

        changes = []
        for edge in self._topology:
            if edge.gradients_sent >= 3 and edge.effectiveness < 0.2:
                edge.active = False
                changes.append(
                    f"Pruned edge {edge.source}->{edge.target} "
                    f"(effectiveness={edge.effectiveness:.0%})"
                )
            elif edge.effectiveness > 0.7 and edge.weight < 1.5:
                edge.weight = min(2.0, edge.weight + 0.2)
                changes.append(
                    f"Strengthened {edge.source}->{edge.target} "
                    f"(weight={edge.weight:.1f})"
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
            gradients_generated=sum(len(g) for g in self._pending_gradients.values()),
        )
        self._performance_history.append(perf)
        return perf

    def get_improvement_trend(self) -> dict[str, Any]:
        """Analyse performance trend across iterations."""
        if len(self._performance_history) < 2:
            return {"trend": "insufficient_data", "iterations": len(self._performance_history)}

        recent = self._performance_history[-3:]
        scores = [p.review_score for p in recent]
        test_rates = [p.test_pass_rate for p in recent]

        improving = (
            (len(scores) >= 2 and scores[-1] > scores[0])
            or (len(test_rates) >= 2 and test_rates[-1] > test_rates[0])
        )

        return {
            "trend": "improving" if improving else "stagnating",
            "iterations": len(self._performance_history),
            "latest_score": scores[-1] if scores else 0,
            "latest_test_rate": test_rates[-1] if test_rates else 0,
            "score_delta": scores[-1] - scores[0] if len(scores) >= 2 else 0,
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
            logger.info(f"[EvoMAC] Loaded state: iteration={self._iteration}")
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
        }
