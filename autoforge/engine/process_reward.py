"""Process Reward Model — step-level evaluation for code generation.

Traditional evaluation only checks the final output (did tests pass?).
CodePRM evaluates EACH STEP of the generation process, providing:
  - Intermediate feedback during code generation
  - Step-level scores that identify where things go wrong
  - Execution-grounded evaluation (actually run partial code)
  - Reward signals for the search tree to guide exploration

This module implements ideas from:
  - CodePRM (ACL 2025): Execution feedback-enhanced process reward
  - FunPRM (2025): Function-as-step process reward model
  - Process Reward Models (Lightman et al.): Step-by-step verification

Architecture:
  1. StepTracker: Records each generation step with context
  2. ProcessRewardModel: Evaluates steps using LLM + execution feedback
  3. RewardAggregator: Combines step scores into trajectory-level signals
  4. Integration with CheckpointManager for seamless pipeline use

Key insight: Instead of training a separate reward model (expensive),
we use the LLM itself as a process verifier, grounded by actual code
execution results. This gives us PRM-quality feedback without training.
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


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────


class StepType(Enum):
    """Types of generation steps that can be evaluated."""
    PLANNING = "planning"          # Thinking about approach
    FILE_CREATE = "file_create"    # Creating a new file
    FILE_MODIFY = "file_modify"    # Modifying existing file
    DEPENDENCY = "dependency"      # Adding dependency/import
    TEST_WRITE = "test_write"      # Writing test code
    CONFIG = "config"              # Configuration changes
    REFACTOR = "refactor"          # Refactoring existing code
    DEBUG = "debug"                # Debugging/fixing errors


@dataclass
class GenerationStep:
    """A single step in the code generation process.

    Inspired by FunPRM: each step is a function-level unit of work,
    making evaluation more natural than arbitrary token boundaries.
    """
    id: str = ""
    step_number: int = 0
    step_type: StepType = StepType.PLANNING
    description: str = ""          # What the agent is trying to do
    # Content produced in this step
    files_touched: list[str] = field(default_factory=list)
    code_snippet: str = ""         # Key code written (truncated)
    tool_calls: list[str] = field(default_factory=list)
    # Execution feedback (CodePRM core idea)
    execution_attempted: bool = False
    execution_success: bool = False
    execution_output: str = ""     # stdout/stderr from running the code
    syntax_valid: bool = True
    # Reward scores
    process_reward: float = 0.0    # Step-level reward (0.0 to 1.0)
    confidence: float = 0.0        # How confident the reward is
    reward_reason: str = ""        # Explanation of the reward
    # Timing
    started_at: float = field(default_factory=time.time)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "description": self.description,
            "files_touched": self.files_touched,
            "code_snippet": self.code_snippet[:500],
            "execution_attempted": self.execution_attempted,
            "execution_success": self.execution_success,
            "execution_output": self.execution_output[:300],
            "syntax_valid": self.syntax_valid,
            "process_reward": self.process_reward,
            "confidence": self.confidence,
            "reward_reason": self.reward_reason,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class StepTrajectory:
    """Complete trajectory of steps for a task.

    The trajectory captures the full story of how code was generated,
    enabling post-hoc analysis and reward signal computation.
    """
    task_id: str = ""
    task_description: str = ""
    steps: list[GenerationStep] = field(default_factory=list)
    # Aggregate metrics
    total_reward: float = 0.0
    avg_reward: float = 0.0
    min_reward: float = 1.0
    reward_trend: str = ""         # "improving", "degrading", "stable"
    # Trajectory-level evaluation
    final_outcome: str = ""        # "success", "partial", "failure"
    final_score: float = 0.0
    started_at: float = field(default_factory=time.time)

    def add_step(self, step: GenerationStep) -> None:
        """Add a step and update aggregate metrics."""
        self.steps.append(step)
        rewards = [s.process_reward for s in self.steps if s.process_reward > 0]
        if rewards:
            self.total_reward = sum(rewards)
            self.avg_reward = self.total_reward / len(rewards)
            self.min_reward = min(rewards)

            # Compute trend from last 3 steps
            if len(rewards) >= 3:
                recent = rewards[-3:]
                if recent[-1] > recent[0] + 0.05:
                    self.reward_trend = "improving"
                elif recent[-1] < recent[0] - 0.05:
                    self.reward_trend = "degrading"
                else:
                    self.reward_trend = "stable"

    def get_bottleneck_steps(self, threshold: float = 0.4) -> list[GenerationStep]:
        """Find steps with low process rewards (potential issues)."""
        return [s for s in self.steps if s.process_reward < threshold and s.process_reward > 0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "num_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "min_reward": self.min_reward,
            "reward_trend": self.reward_trend,
            "final_outcome": self.final_outcome,
            "final_score": self.final_score,
        }


# ──────────────────────────────────────────────
# Process Reward Model
# ──────────────────────────────────────────────


class ProcessRewardModel:
    """LLM-based process reward model for code generation.

    Instead of training a separate reward model, we use the LLM itself
    as a verifier, grounded by execution feedback. This approach:
      - Requires no training data
      - Adapts to any programming language
      - Can explain its reasoning
      - Improves as the base LLM improves

    Evaluation dimensions (CodePRM-inspired):
      1. Correctness: Does this step move toward the goal?
      2. Efficiency: Is this the most direct path?
      3. Coherence: Does this fit with previous steps?
      4. Execution: Does the code actually run?
    """

    def __init__(
        self,
        config: Any,
        llm: Any,
        sandbox: Any | None = None,
        working_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.sandbox = sandbox
        self.working_dir = working_dir
        self._trajectories: dict[str, StepTrajectory] = {}

    def start_trajectory(self, task_id: str, task_description: str) -> StepTrajectory:
        """Begin tracking a new task trajectory."""
        trajectory = StepTrajectory(
            task_id=task_id,
            task_description=task_description,
        )
        self._trajectories[task_id] = trajectory
        logger.debug(f"[PRM] Started trajectory for {task_id}")
        return trajectory

    def record_step(
        self,
        task_id: str,
        step_type: StepType,
        description: str,
        files_touched: list[str] | None = None,
        code_snippet: str = "",
        tool_calls: list[str] | None = None,
    ) -> GenerationStep:
        """Record a generation step (before evaluation).

        Called by the agent loop whenever a meaningful action is taken.
        """
        trajectory = self._trajectories.get(task_id)
        if trajectory is None:
            trajectory = self.start_trajectory(task_id, "")

        step = GenerationStep(
            id=f"{task_id}-step-{len(trajectory.steps):03d}",
            step_number=len(trajectory.steps),
            step_type=step_type,
            description=description,
            files_touched=files_touched or [],
            code_snippet=code_snippet,
            tool_calls=tool_calls or [],
        )
        trajectory.add_step(step)
        return step

    async def evaluate_step(
        self,
        task_id: str,
        step: GenerationStep,
        context: str = "",
    ) -> float:
        """Evaluate a single step and assign a process reward.

        This is the core of CodePRM: combine LLM judgment with execution
        feedback for grounded step-level evaluation.

        Returns: process reward score (0.0 to 1.0)
        """
        trajectory = self._trajectories.get(task_id)
        if trajectory is None:
            return 0.5

        # Phase 1: Execution feedback (if possible)
        if self.sandbox and step.files_touched and step.step_type in (
            StepType.FILE_CREATE, StepType.FILE_MODIFY, StepType.TEST_WRITE,
        ):
            await self._check_execution(step)

        # Phase 2: LLM evaluation
        reward = await self._llm_evaluate_step(step, trajectory, context)

        # Phase 3: Combine execution + LLM signals
        if step.execution_attempted:
            exec_bonus = 0.15 if step.execution_success else -0.15
            syntax_bonus = 0.05 if step.syntax_valid else -0.10
            reward = max(0.0, min(1.0, reward + exec_bonus + syntax_bonus))

        step.process_reward = reward
        logger.info(
            f"[PRM] Step {step.id}: reward={reward:.2f} "
            f"(exec={'ok' if step.execution_success else 'fail' if step.execution_attempted else 'skip'})"
        )
        return reward

    async def evaluate_trajectory(
        self,
        task_id: str,
        final_outcome: str = "",
    ) -> dict[str, Any]:
        """Evaluate the complete trajectory and generate a summary.

        Called after a task completes. Provides:
          - Overall trajectory score
          - Bottleneck identification
          - Improvement suggestions for future runs
        """
        from autoforge.engine.llm_router import TaskComplexity

        trajectory = self._trajectories.get(task_id)
        if trajectory is None:
            return {"error": "No trajectory found"}

        trajectory.final_outcome = final_outcome

        # Compute trajectory-level score
        if trajectory.steps:
            rewarded = [s for s in trajectory.steps if s.process_reward > 0]
            if rewarded:
                # Weighted: later steps matter more (they build on earlier ones)
                weights = [1.0 + 0.1 * i for i in range(len(rewarded))]
                weighted_sum = sum(s.process_reward * w for s, w in zip(rewarded, weights))
                trajectory.final_score = weighted_sum / sum(weights)
            else:
                trajectory.final_score = 0.5

        # Identify bottlenecks
        bottlenecks = trajectory.get_bottleneck_steps()

        result = {
            "task_id": task_id,
            "final_score": trajectory.final_score,
            "avg_reward": trajectory.avg_reward,
            "min_reward": trajectory.min_reward,
            "reward_trend": trajectory.reward_trend,
            "total_steps": len(trajectory.steps),
            "bottleneck_count": len(bottlenecks),
            "final_outcome": final_outcome,
        }

        if bottlenecks:
            result["bottleneck_steps"] = [
                {
                    "step": s.step_number,
                    "type": s.step_type.value,
                    "description": s.description[:80],
                    "reward": s.process_reward,
                    "reason": s.reward_reason[:100],
                }
                for s in bottlenecks[:5]
            ]

        logger.info(
            f"[PRM] Trajectory {task_id}: score={trajectory.final_score:.2f} "
            f"trend={trajectory.reward_trend} bottlenecks={len(bottlenecks)}"
        )
        return result

    def get_trajectory(self, task_id: str) -> StepTrajectory | None:
        """Get the trajectory for a task."""
        return self._trajectories.get(task_id)

    def get_reward_signal_for_search(self, task_id: str) -> float:
        """Get an aggregate reward signal for the search tree.

        Used by RethinkMCTS to evaluate and guide the search direction.
        Returns a normalized score suitable for MCTS value estimation.
        """
        trajectory = self._trajectories.get(task_id)
        if trajectory is None or not trajectory.steps:
            return 0.5  # Neutral prior

        rewarded = [s for s in trajectory.steps if s.process_reward > 0]
        if not rewarded:
            return 0.5

        # Recent steps weighted higher (reflects current direction)
        recent = rewarded[-5:]
        if len(recent) >= 2:
            recent_avg = sum(s.process_reward for s in recent) / len(recent)
            overall_avg = trajectory.avg_reward
            # Blend: 60% recent, 40% overall
            return 0.6 * recent_avg + 0.4 * overall_avg
        return trajectory.avg_reward

    def should_course_correct(self, task_id: str) -> tuple[bool, str]:
        """Check if the trajectory suggests course correction is needed.

        Returns (should_correct, reason).
        Used as an early warning system before the checkpoint manager triggers.
        """
        trajectory = self._trajectories.get(task_id)
        if trajectory is None or len(trajectory.steps) < 3:
            return (False, "")

        # Check for sustained low rewards
        recent = trajectory.steps[-3:]
        recent_rewards = [s.process_reward for s in recent if s.process_reward > 0]

        if recent_rewards and max(recent_rewards) < 0.3:
            return (True, "Last 3 steps all scored below 0.3 — approach may be fundamentally wrong")

        if trajectory.reward_trend == "degrading" and trajectory.avg_reward < 0.5:
            return (True, f"Reward trend is degrading (avg={trajectory.avg_reward:.2f})")

        # Check for repeated failures
        recent_exec = [s for s in recent if s.execution_attempted]
        if len(recent_exec) >= 2 and all(not s.execution_success for s in recent_exec):
            return (True, "Multiple consecutive execution failures")

        return (False, "")

    # ──────── Internal: Execution Check ────────

    async def _check_execution(self, step: GenerationStep) -> None:
        """Run execution checks on code produced by a step.

        CodePRM's key insight: actual execution provides far stronger
        signal than static analysis alone.
        """
        if not self.sandbox or not self.working_dir:
            return

        step.execution_attempted = True

        # Check syntax for Python files
        for filepath in step.files_touched:
            full_path = self.working_dir / filepath
            if not full_path.exists():
                continue

            if filepath.endswith(".py"):
                try:
                    result = await self.sandbox.exec(
                        f"python -m py_compile {full_path}",
                        timeout=10,
                    )
                    step.syntax_valid = result.exit_code == 0
                    if not step.syntax_valid:
                        step.execution_output += f"Syntax error in {filepath}: {result.stderr[:200]}\n"
                except Exception:
                    pass

            elif filepath.endswith((".js", ".jsx")):
                try:
                    result = await self.sandbox.exec(
                        f"node --check {full_path}",
                        timeout=10,
                    )
                    step.syntax_valid = result.exit_code == 0
                    if not step.syntax_valid:
                        step.execution_output += f"Syntax error in {filepath}: {result.stderr[:200]}\n"
                except Exception:
                    pass

        # For test files, try running them
        if step.step_type == StepType.TEST_WRITE:
            test_files = [f for f in step.files_touched if "test" in f.lower()]
            if test_files:
                try:
                    test_file = test_files[0]
                    if test_file.endswith(".py"):
                        result = await self.sandbox.exec(
                            f"python -m pytest {self.working_dir / test_file} --tb=short -q 2>&1 || true",
                            timeout=30,
                        )
                    else:
                        result = await self.sandbox.exec(
                            f"cd {self.working_dir} && npm test -- {test_file} 2>&1 || true",
                            timeout=30,
                        )
                    step.execution_success = result.exit_code == 0
                    step.execution_output += result.stdout[:300]
                except Exception as e:
                    step.execution_output += f"Test execution error: {e}"

        # If all syntax checks passed but no execution was actually performed
        # (no test output), mark as unknown rather than success to avoid
        # conflating "syntax valid" with "execution succeeded".
        if step.syntax_valid and not step.execution_output:
            step.execution_success = None

    # ──────── Internal: LLM Evaluation ────────

    async def _llm_evaluate_step(
        self,
        step: GenerationStep,
        trajectory: StepTrajectory,
        context: str = "",
    ) -> float:
        """Use the LLM as a process verifier for a step.

        Evaluates: correctness, efficiency, coherence, quality.
        """
        from autoforge.engine.llm_router import TaskComplexity

        # Build evaluation context from previous steps
        prev_summary = self._summarize_previous_steps(trajectory, step.step_number)

        prompt = (
            f"Evaluate this code generation step.\n\n"
            f"## Task Goal\n{trajectory.task_description[:500]}\n\n"
            f"## Previous Steps Summary\n{prev_summary}\n\n"
            f"## Current Step (#{step.step_number})\n"
            f"Type: {step.step_type.value}\n"
            f"Description: {step.description}\n"
            f"Files: {', '.join(step.files_touched) if step.files_touched else 'none'}\n"
        )

        if step.code_snippet:
            prompt += f"Code:\n```\n{step.code_snippet[:800]}\n```\n"

        if step.execution_attempted:
            prompt += (
                f"\n## Execution Results\n"
                f"Syntax valid: {step.syntax_valid}\n"
                f"Execution success: {step.execution_success}\n"
            )
            if step.execution_output:
                prompt += f"Output:\n```\n{step.execution_output[:300]}\n```\n"

        if context:
            prompt += f"\n## Project Context\n{context[:500]}\n"

        prompt += (
            f"\n## Evaluate (0.0 to 1.0)\n"
            f"1. Correctness: Does this step move toward the task goal?\n"
            f"2. Efficiency: Is this the most direct approach?\n"
            f"3. Coherence: Does this fit with previous steps?\n"
            f"4. Quality: Is the code/action well-structured?\n\n"
            f"Output JSON: {{\"score\": 0.0-1.0, \"confidence\": 0.0-1.0, "
            f"\"reason\": \"brief explanation\"}}"
        )

        try:
            response = await self.llm.call(
                complexity=TaskComplexity.STANDARD,
                system=(
                    "You are a code review expert evaluating a single generation step. "
                    "Be precise and constructive. Respond with JSON only."
                ),
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            import re
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                step.confidence = float(data.get("confidence", 0.5))
                step.reward_reason = str(data.get("reason", ""))
                return float(data.get("score", 0.5))

        except Exception as e:
            logger.debug(f"[PRM] LLM evaluation failed: {e}")

        return 0.5  # Neutral default

    def _summarize_previous_steps(
        self,
        trajectory: StepTrajectory,
        current_step: int,
    ) -> str:
        """Create a compact summary of previous steps for context."""
        if current_step == 0:
            return "This is the first step."

        lines = []
        for step in trajectory.steps[:current_step]:
            emoji = "+" if step.process_reward >= 0.6 else ("-" if step.process_reward < 0.4 else "~")
            lines.append(
                f"  {emoji} Step {step.step_number} ({step.step_type.value}): "
                f"{step.description[:60]} "
                f"[reward={step.process_reward:.2f}]"
            )

        return "\n".join(lines[-8:])  # Last 8 steps for context window

    # ──────── Persistence ────────

    def save_trajectory(self, task_id: str, output_dir: Path) -> None:
        """Save a trajectory to disk for post-hoc analysis."""
        trajectory = self._trajectories.get(task_id)
        if trajectory is None:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"trajectory_{task_id}.json"
        try:
            path.write_text(
                json.dumps(trajectory.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug(f"[PRM] Saved trajectory to {path}")
        except Exception as e:
            logger.warning(f"[PRM] Could not save trajectory: {e}")
