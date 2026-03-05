"""Closed-loop experiment execution pipeline for AI research agent framework.

Implements the AI Scientist v2 paradigm: hypothesis → code → run → analyze → iterate.

This module enables autonomous scientific discovery by:
1. Generating initial hypotheses from a research question
2. Writing executable experiment code for each hypothesis
3. Running experiments in a sandboxed environment
4. Analyzing results and comparing with predictions
5. Iteratively refining hypotheses or code based on outcomes
6. Tracking full provenance of the experimental process
7. Running ablation studies to determine factor importance

Key classes:
- ExperimentLoop: Main orchestrator for the closed-loop process
- HypothesisGenerator: LLM-powered hypothesis creation and refinement
- ExperimentCoder: Generates self-contained Python experiment scripts
- ExperimentRunner: Sandboxed execution with metrics collection
- ResultAnalyzer: LLM-based analysis of experimental results
- AblationStudy: Systematic factor importance analysis
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from autoforge.engine.llm_router import TaskComplexity
from autoforge.engine.sandbox import SubprocessSandbox, SandboxResult

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment in the execution pipeline."""

    PENDING = "pending"           # Waiting to start
    HYPOTHESIS = "hypothesis"     # Generating initial hypotheses
    CODING = "coding"             # Generating or fixing code
    RUNNING = "running"           # Code is executing
    ANALYZING = "analyzing"       # Analyzing results
    ITERATING = "iterating"       # Refining hypothesis or code
    COMPLETED = "completed"       # Successfully finished
    FAILED = "failed"             # Terminal failure


@dataclass
class Hypothesis:
    """A testable scientific hypothesis for an experiment."""

    id: str
    """Unique identifier for this hypothesis."""

    statement: str
    """Clear statement of the hypothesis (e.g., 'Larger batch size improves convergence')."""

    rationale: str
    """Scientific reasoning behind the hypothesis."""

    predicted_outcome: str
    """Expected outcome if hypothesis is true (e.g., 'Training loss < 0.5 at epoch 10')."""

    confidence: float
    """Confidence level in hypothesis, 0.0-1.0."""

    parent_id: str = ""
    """ID of parent hypothesis if this is a refinement."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (e.g., {'domain': 'deep_learning', 'task': 'classification'})."""

    created_at: float = field(default_factory=time.time)
    """Timestamp when hypothesis was created."""

    def is_refined(self) -> bool:
        """Check if this is a refined hypothesis (has a parent)."""
        return bool(self.parent_id)


@dataclass
class ExperimentResult:
    """Result of executing a single experiment."""

    hypothesis_id: str
    """ID of the hypothesis being tested."""

    code_path: Path
    """Path to the executed code file."""

    stdout: str
    """Standard output from the code execution."""

    stderr: str
    """Standard error output."""

    exit_code: int
    """Process exit code (0 = success)."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Extracted metrics from execution (e.g., {'accuracy': 0.95, 'loss': 0.1})."""

    artifacts: list[Path] = field(default_factory=list)
    """Generated files (figures, data, etc.) from the experiment."""

    execution_time: float = 0.0
    """Total execution time in seconds."""

    success: bool = False
    """Whether execution completed successfully (exit_code == 0)."""

    timestamp: float = field(default_factory=time.time)
    """When the result was recorded."""

    def has_metrics(self) -> bool:
        """Check if any metrics were captured."""
        return bool(self.metrics)

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Safely get a metric value."""
        return self.metrics.get(name, default)


@dataclass
class ExperimentIteration:
    """Single iteration in the experiment loop."""

    round_number: int
    """Iteration number (1-indexed)."""

    hypothesis: Hypothesis
    """The hypothesis being tested in this iteration."""

    code: str
    """The Python code that was generated and executed."""

    result: ExperimentResult | None = None
    """Result of executing the code (None if not yet run)."""

    analysis: str = ""
    """LLM analysis of the results."""

    next_action: str = "continue"
    """Next step: 'refine_hypothesis', 'refine_code', 'accept', 'reject', 'continue'."""

    improvement_delta: float = 0.0
    """Improvement from previous iteration (positive = better)."""

    code_attempt: int = 1
    """Which code generation attempt was this (1 = first)."""

    timestamp: float = field(default_factory=time.time)
    """When this iteration was executed."""

    def is_successful(self) -> bool:
        """Check if this iteration succeeded and produced valid results."""
        return self.result is not None and self.result.success and self.result.has_metrics()

    def should_accept(self) -> bool:
        """Check if the hypothesis should be accepted based on this iteration."""
        return self.next_action == "accept"


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""

    workspace_dir: Path
    """Base directory for all experiments."""

    max_iterations: int = 10
    """Maximum number of iterations to run."""

    max_code_attempts: int = 3
    """Maximum attempts to fix failing code."""

    execution_timeout: int = 300
    """Timeout for code execution in seconds."""

    improvement_threshold: float = 0.01
    """Minimum improvement to continue iterating (0.01 = 1%)."""

    sandbox_enabled: bool = True
    """Whether to run code in sandboxed environment."""

    python_executable: str = "python3"
    """Path to Python executable."""

    metrics_prefix: str = "METRIC:"
    """Prefix for metric lines in stdout."""

    artifact_extensions: list[str] = field(
        default_factory=lambda: [".png", ".jpg", ".pdf", ".json", ".csv", ".txt"]
    )
    """File extensions to collect as artifacts."""

    log_level: str = "INFO"
    """Logging level."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.workspace_dir.exists():
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.execution_timeout < 10:
            raise ValueError("execution_timeout must be >= 10")
        if not (0.0 <= self.improvement_threshold <= 1.0):
            raise ValueError("improvement_threshold must be 0.0-1.0")


class HypothesisGenerator:
    """Generates and refines scientific hypotheses using LLM guidance."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.HypothesisGenerator")

    async def generate_initial(
        self,
        research_question: str,
        context: str,
        llm: Any,
    ) -> list[Hypothesis]:
        """Generate initial hypotheses from a research question.

        Args:
            research_question: The scientific question to explore
            context: Domain context (e.g., 'deep learning', 'optimization')
            llm: LLM instance with async call(prompt, complexity=TaskComplexity.HIGH)

        Returns:
            List of generated hypotheses
        """
        self.logger.info(f"Generating hypotheses for: {research_question}")

        prompt = f"""You are an expert AI research scientist. Generate 3-5 diverse, testable hypotheses 
for the following research question.

Research Question:
{research_question}

Domain Context:
{context}

For each hypothesis, provide:
1. A clear statement (1-2 sentences)
2. Scientific rationale (3-5 sentences explaining why this might be true)
3. Predicted outcome (specific, measurable expectation)
4. Confidence level (0.0-1.0)

Format your response as JSON with this structure:
{{
  "hypotheses": [
    {{
      "statement": "...",
      "rationale": "...",
      "predicted_outcome": "...",
      "confidence": 0.8
    }},
    ...
  ]
}}

Ensure hypotheses are:
- Scientifically sound and testable
- Diverse in approach and assumptions
- Specific enough to be verifiable with experiments
- Non-obvious (interesting research contributions)
"""

        try:
            response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            content = response.content if hasattr(response, 'content') else str(response)

            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                self.logger.error("No JSON found in hypothesis generation response")
                return []

            data = json.loads(json_match.group())
            hypotheses = []

            for i, h_data in enumerate(data.get('hypotheses', [])):
                hypothesis = Hypothesis(
                    id=f"hyp-{uuid.uuid4().hex[:8]}",
                    statement=h_data['statement'],
                    rationale=h_data['rationale'],
                    predicted_outcome=h_data['predicted_outcome'],
                    confidence=float(h_data['confidence']),
                    metadata={
                        'generation': 'initial',
                        'index': i,
                        'research_question': research_question,
                    },
                )
                hypotheses.append(hypothesis)
                self.logger.debug(f"Generated hypothesis {i+1}: {hypothesis.statement}")

            self.logger.info(f"Generated {len(hypotheses)} hypotheses")
            return hypotheses

        except Exception as e:
            self.logger.error(f"Failed to generate hypotheses: {e}", exc_info=True)
            return []

    async def refine(
        self,
        hypothesis: Hypothesis,
        result: ExperimentResult,
        analysis: str,
        llm: Any,
    ) -> Hypothesis:
        """Refine a hypothesis based on experimental results.

        Args:
            hypothesis: The original hypothesis
            result: Results from testing the hypothesis
            analysis: LLM analysis of the results
            llm: LLM instance

        Returns:
            Refined hypothesis with parent_id pointing to original
        """
        self.logger.info(f"Refining hypothesis {hypothesis.id}")

        prompt = f"""You are an expert AI research scientist analyzing experimental results.

Original Hypothesis:
{hypothesis.statement}

Rationale:
{hypothesis.rationale}

Predicted Outcome:
{hypothesis.predicted_outcome}

Experimental Results:
{json.dumps(result.metrics, indent=2)}

Analysis:
{analysis}

Based on the results, generate a refined hypothesis that:
1. Incorporates insights from the failed/partial experiment
2. Adjusts assumptions that weren't supported
3. Proposes a more nuanced or modified claim
4. Remains testable in a similar experimental setup

Provide the refined hypothesis in JSON format:
{{
  "statement": "...",
  "rationale": "...",
  "predicted_outcome": "...",
  "confidence": 0.7
}}
"""

        try:
            response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            content = response.content if hasattr(response, 'content') else str(response)

            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                self.logger.error("No JSON in refined hypothesis response")
                return hypothesis

            data = json.loads(json_match.group())

            refined = Hypothesis(
                id=f"hyp-{uuid.uuid4().hex[:8]}",
                statement=data['statement'],
                rationale=data['rationale'],
                predicted_outcome=data['predicted_outcome'],
                confidence=float(data['confidence']),
                parent_id=hypothesis.id,
                metadata={
                    'generation': 'refined',
                    'refinement_round': hypothesis.metadata.get('refinement_round', 0) + 1,
                    'based_on_metrics': result.metrics,
                },
            )

            self.logger.debug(f"Refined hypothesis: {refined.statement}")
            return refined

        except Exception as e:
            self.logger.error(f"Failed to refine hypothesis: {e}", exc_info=True)
            return hypothesis


class ExperimentCoder:
    """Generates and fixes Python code for experiments."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ExperimentCoder")

    async def generate_code(
        self,
        hypothesis: Hypothesis,
        llm: Any,
        previous_code: str = "",
        error_feedback: str = "",
    ) -> str:
        """Generate self-contained Python experiment code.

        Args:
            hypothesis: The hypothesis to test
            llm: LLM instance
            previous_code: Previous code attempt if fixing
            error_feedback: Error message if this is a fix attempt

        Returns:
            Complete Python script as string
        """
        self.logger.info(f"Generating code for hypothesis {hypothesis.id}")

        if error_feedback:
            prompt = f"""You are an expert Python developer. Fix the following code that failed with an error.

Original Hypothesis:
{hypothesis.statement}

Previous Code:
```python
{previous_code}
```

Error:
{error_feedback}

Generate corrected Python code that:
1. Fixes the error
2. Maintains the experimental design from the hypothesis
3. Outputs metrics as JSON lines with format: {{"metric": "<name>", "value": <float>}}
4. Generates visualizations with proper labels and saved as PNG
5. Is completely self-contained (no external data dependencies)
6. Has clear comments explaining the experimental flow
7. Includes proper error handling

Wrap the complete code in triple backticks:
```python
...
```
"""
        else:
            prompt = f"""You are an expert experimental designer. Generate Python code to test this hypothesis:

Hypothesis: {hypothesis.statement}

Rationale: {hypothesis.rationale}

Predicted Outcome: {hypothesis.predicted_outcome}

Generate complete, self-contained Python code that:
1. Sets up the experimental environment (random seeds, data generation, etc.)
2. Implements the core experiment to test the hypothesis
3. Collects metrics that would validate/refute the hypothesis
4. Outputs metrics as JSON lines with format: {{"metric": "<name>", "value": <float>}}
5. Generates at least one visualization (use matplotlib) saved as PNG
6. Includes proper documentation and comments
7. Uses only standard libraries (numpy, scipy, matplotlib, scikit-learn allowed)
8. Includes error handling and logging

The experiment should:
- Be reproducible (set random seeds)
- Run in under {self.config.execution_timeout} seconds
- Output progress/status to stdout
- Save any figures as PNG files in the working directory
- Return metrics that directly address the hypothesis

Wrap the complete code in triple backticks:
```python
...
```
"""

        try:
            response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            content = response.content if hasattr(response, 'content') else str(response)

            # Extract code from markdown code fence
            code_match = re.search(r'```python\n([\s\S]*?)\n```', content)
            if code_match:
                code = code_match.group(1)
            else:
                # Try without language specifier
                code_match = re.search(r'```\n([\s\S]*?)\n```', content)
                if code_match:
                    code = code_match.group(1)
                else:
                    self.logger.error("Could not extract code from response")
                    return ""

            self.logger.debug(f"Generated {len(code.splitlines())} lines of code")
            return code

        except Exception as e:
            self.logger.error(f"Failed to generate code: {e}", exc_info=True)
            return ""

    async def fix_code(
        self,
        code: str,
        error: str,
        llm: Any,
    ) -> str:
        """Fix code that failed during execution.

        Args:
            code: The failing code
            error: Error message from execution
            llm: LLM instance

        Returns:
            Fixed Python code
        """
        self.logger.info("Attempting to fix failing code")
        # Create a temporary hypothesis to reuse generate_code logic
        temp_hypothesis = Hypothesis(
            id="temp",
            statement="Fix the failing experiment",
            rationale="Previous execution failed",
            predicted_outcome="Code should run without errors",
            confidence=0.5,
        )
        return await self.generate_code(
            temp_hypothesis,
            llm,
            previous_code=code,
            error_feedback=error,
        )


class ExperimentRunner:
    """Executes Python experiment code in a sandboxed environment."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ExperimentRunner")

    async def run(self, code: str, config: ExperimentConfig | None = None) -> ExperimentResult:
        """Execute experiment code and collect results.

        Args:
            code: Python code to execute
            config: ExperimentConfig (uses self.config if None)

        Returns:
            ExperimentResult with metrics and artifacts
        """
        if config is None:
            config = self.config

        exec_config = config or self.config
        run_id = uuid.uuid4().hex[:8]
        work_dir = exec_config.workspace_dir / f"run-{run_id}"
        code_path = work_dir / "experiment.py"

        try:
            # Create working directory
            work_dir.mkdir(parents=True, exist_ok=True)
            code_path.write_text(code)
            self.logger.debug(f"Wrote code to {code_path}")

            # Execute code in sandbox
            start_time = time.time()
            sandbox = SubprocessSandbox(work_dir)
            await sandbox.start()

            command = f"{exec_config.python_executable} experiment.py"
            result = await sandbox.exec(command, timeout=exec_config.execution_timeout)
            await sandbox.stop()

            execution_time = time.time() - start_time

            # Parse metrics from stdout
            metrics = self._parse_metrics(result.stdout)

            # Collect artifacts
            artifacts = self._collect_artifacts(work_dir, exec_config)

            experiment_result = ExperimentResult(
                hypothesis_id="",  # Will be set by caller
                code_path=code_path,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                metrics=metrics,
                artifacts=artifacts,
                execution_time=execution_time,
                success=result.exit_code == 0,
            )

            self.logger.info(
                f"Execution complete: exit_code={result.exit_code}, "
                f"metrics={len(metrics)}, artifacts={len(artifacts)}, "
                f"time={execution_time:.2f}s"
            )

            return experiment_result

        except Exception as e:
            self.logger.error(f"Failed to run experiment: {e}", exc_info=True)
            return ExperimentResult(
                hypothesis_id="",
                code_path=code_path,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                success=False,
            )

    def _parse_metrics(self, stdout: str) -> dict[str, float]:
        """Parse metrics from stdout.

        Expected format: lines like 'METRIC: {"metric": "accuracy", "value": 0.95}'
        """
        metrics = {}
        for line in stdout.splitlines():
            if self.config.metrics_prefix in line:
                try:
                    json_str = line.split(self.config.metrics_prefix, 1)[1].strip()
                    data = json.loads(json_str)
                    if 'metric' in data and 'value' in data:
                        metrics[data['metric']] = float(data['value'])
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    self.logger.debug(f"Failed to parse metric line: {line}, error: {e}")
        return metrics

    def _collect_artifacts(self, work_dir: Path, config: ExperimentConfig) -> list[Path]:
        """Collect generated artifact files from the working directory."""
        artifacts = []
        try:
            for ext in config.artifact_extensions:
                for artifact in work_dir.glob(f"*{ext}"):
                    if artifact.is_file():
                        artifacts.append(artifact)
                        self.logger.debug(f"Collected artifact: {artifact}")
        except Exception as e:
            self.logger.warning(f"Failed to collect artifacts: {e}")
        return artifacts


class ResultAnalyzer:
    """Analyzes experimental results and determines next steps."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ResultAnalyzer")

    async def analyze(
        self,
        hypothesis: Hypothesis,
        result: ExperimentResult,
        previous_best_metric: float | None = None,
        llm: Any | None = None,
    ) -> tuple[str, str, float]:
        """Analyze experimental results and generate next action.

        Args:
            hypothesis: The tested hypothesis
            result: Experimental results
            previous_best_metric: Best metric value from previous iteration
            llm: LLM instance for analysis

        Returns:
            Tuple of (analysis_text, next_action, improvement_delta)
            next_action: 'refine_hypothesis', 'refine_code', 'accept', 'reject'
        """
        self.logger.info(f"Analyzing results for hypothesis {hypothesis.id}")

        # Basic analysis without LLM if not available
        if not llm:
            return await self._analyze_basic(
                hypothesis, result, previous_best_metric
            )

        # Use LLM for deeper analysis
        prompt = f"""You are an expert AI research scientist analyzing experimental results.

Hypothesis:
{hypothesis.statement}

Predicted Outcome:
{hypothesis.predicted_outcome}

Experimental Results:
Metrics: {json.dumps(result.metrics, indent=2)}
Exit Code: {result.exit_code}
Execution Time: {result.execution_time:.2f}s

stdout: {result.stdout[:500]}
stderr: {result.stderr[:500] if result.stderr else "None"}

Previous Best Metric: {previous_best_metric if previous_best_metric else "None (first iteration)"}

Provide analysis in JSON format:
{{
  "prediction_met": true/false,
  "analysis": "Detailed explanation of what the results show",
  "evidence": "Key metrics that support or refute the hypothesis",
  "next_action": "refine_hypothesis|refine_code|accept|reject",
  "reasoning": "Why this action should be taken",
  "improvement_delta": 0.15
}}

Guidelines:
- If execution failed (non-zero exit), suggest refine_code or reject
- If metrics don't match predictions, suggest refine_hypothesis
- If metrics support the hypothesis, suggest accept
- improvement_delta should be positive if metrics improved, negative otherwise
- Range from -1.0 to 1.0 where 0 = no change
"""

        try:
            response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            content = response.content if hasattr(response, 'content') else str(response)

            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                self.logger.warning("No JSON in analysis response, falling back to basic")
                return await self._analyze_basic(hypothesis, result, previous_best_metric)

            data = json.loads(json_match.group())
            return (
                data.get('analysis', ''),
                data.get('next_action', 'refine_hypothesis'),
                float(data.get('improvement_delta', 0.0)),
            )

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}", exc_info=True)
            return await self._analyze_basic(hypothesis, result, previous_best_metric)

    async def _analyze_basic(
        self,
        hypothesis: Hypothesis,
        result: ExperimentResult,
        previous_best_metric: float | None = None,
    ) -> tuple[str, str, float]:
        """Basic analysis without LLM."""
        if not result.success:
            return (
                f"Code execution failed with exit code {result.exit_code}: {result.stderr}",
                "refine_code",
                -1.0,
            )

        if not result.has_metrics():
            return (
                "No metrics collected from execution",
                "refine_code",
                0.0,
            )

        # Simple heuristic: if we have metrics, assume partial success
        first_metric_val = list(result.metrics.values())[0] if result.metrics else 0.0

        improvement = 0.0
        if previous_best_metric is not None:
            improvement = (first_metric_val - previous_best_metric) / abs(previous_best_metric + 1e-6)

        analysis = f"Executed successfully. Collected {len(result.metrics)} metrics. "
        if improvement > self.config.improvement_threshold:
            analysis += f"Metrics improved by {improvement:.1%}."
            next_action = "refine_hypothesis"
        else:
            analysis += "Limited improvement from previous iteration."
            next_action = "refine_hypothesis"

        return analysis, next_action, improvement


class ExperimentLoop:
    """Main orchestrator for the closed-loop experiment execution pipeline."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ExperimentLoop")

        self.hypothesis_gen = HypothesisGenerator(config)
        self.coder = ExperimentCoder(config)
        self.runner = ExperimentRunner(config)
        self.analyzer = ResultAnalyzer(config)

    async def run(
        self,
        research_question: str,
        llm: Any,
        context: str = "",
    ) -> list[ExperimentIteration]:
        """Execute the full experiment loop.

        Args:
            research_question: The research question to investigate
            llm: LLM instance for hypothesis generation, code generation, and analysis
            context: Domain context information

        Returns:
            List of all experiment iterations with full provenance
        """
        self.logger.info(f"Starting experiment loop for: {research_question}")

        iterations: list[ExperimentIteration] = []
        best_iteration: ExperimentIteration | None = None
        best_metric: float = 0.0

        # Phase 1: Generate initial hypotheses
        self.logger.info("Phase 1: Generating initial hypotheses")
        hypotheses = await self.hypothesis_gen.generate_initial(
            research_question, context, llm
        )

        if not hypotheses:
            self.logger.error("Failed to generate any hypotheses")
            return iterations

        self.logger.info(f"Generated {len(hypotheses)} hypotheses")

        # Select best hypothesis (highest confidence initially)
        current_hypothesis = max(hypotheses, key=lambda h: h.confidence)
        self.logger.info(
            f"Selected hypothesis {current_hypothesis.id}: {current_hypothesis.statement}"
        )

        # Phase 2-5: Iterate until stopping condition
        code_attempts = 0
        for iteration_num in range(1, self.config.max_iterations + 1):
            self.logger.info(f"=== Iteration {iteration_num} ===")

            # Phase 2: Generate/fix code
            self.logger.info("Phase 2: Generating experiment code")
            if code_attempts > 0:
                # Previous code failed, try to fix it
                if iterations and iterations[-1].code:
                    prev_code = iterations[-1].code
                    prev_error = iterations[-1].result.stderr if iterations[-1].result else ""
                    code = await self.coder.fix_code(prev_code, prev_error, llm)
                else:
                    code = await self.coder.generate_code(current_hypothesis, llm)
            else:
                code = await self.coder.generate_code(current_hypothesis, llm)

            if not code:
                self.logger.error("Failed to generate code")
                code_attempts += 1
                if code_attempts >= self.config.max_code_attempts:
                    break
                continue

            # Phase 3: Run experiment
            self.logger.info("Phase 3: Running experiment")
            result = await self.runner.run(code, self.config)
            result.hypothesis_id = current_hypothesis.id

            # Phase 4: Analyze results
            self.logger.info("Phase 4: Analyzing results")
            analysis, next_action, improvement_delta = await self.analyzer.analyze(
                current_hypothesis, result, best_metric, llm
            )

            # Create iteration record
            iteration = ExperimentIteration(
                round_number=iteration_num,
                hypothesis=current_hypothesis,
                code=code,
                result=result,
                analysis=analysis,
                next_action=next_action,
                improvement_delta=improvement_delta,
                code_attempt=code_attempts + 1,
            )

            iterations.append(iteration)
            self.logger.info(f"Iteration {iteration_num} complete. Next action: {next_action}")

            # Track best result
            if result.success and result.has_metrics():
                first_metric = list(result.metrics.values())[0]
                if first_metric > best_metric:
                    best_metric = first_metric
                    best_iteration = iteration
                    code_attempts = 0

            # Phase 5: Decide whether to continue/iterate
            if not self._should_continue(iterations, current_hypothesis):
                self.logger.info("Stopping condition met")
                break

            # Phase 5: Iterate (refine hypothesis or code)
            self.logger.info("Phase 5: Preparing next iteration")
            if next_action == "refine_hypothesis":
                self.logger.info("Refining hypothesis")
                current_hypothesis = await self.hypothesis_gen.refine(
                    current_hypothesis, result, analysis, llm
                )
                code_attempts = 0
            elif next_action == "refine_code":
                self.logger.info("Refining code (attempt fix)")
                code_attempts += 1
                if code_attempts >= self.config.max_code_attempts:
                    self.logger.warning("Max code attempts reached")
                    break
                iteration_num -= 1  # Don't count failed attempts as iterations
            elif next_action in ("accept", "reject"):
                self.logger.info(f"Hypothesis {next_action}ed")
                break

        self.logger.info(
            f"Experiment loop complete. Total iterations: {len(iterations)}, "
            f"best metric: {best_metric:.4f}"
        )

        return iterations

    def _should_continue(
        self,
        iterations: list[ExperimentIteration],
        hypothesis: Hypothesis,
    ) -> bool:
        """Determine if the loop should continue to the next iteration."""
        if len(iterations) >= self.config.max_iterations:
            self.logger.info("Max iterations reached")
            return False

        if not iterations:
            return True

        last_iteration = iterations[-1]

        # Stop if hypothesis accepted
        if last_iteration.next_action == "accept":
            return False

        # Stop if hypothesis rejected
        if last_iteration.next_action == "reject":
            return False

        # Continue by default
        return True

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of the last run."""
        return {
            "config": {
                "max_iterations": self.config.max_iterations,
                "execution_timeout": self.config.execution_timeout,
            },
            "generated_at": time.time(),
        }


class AblationStudy:
    """Runs systematic ablation studies to determine factor importance."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AblationStudy")
        self.runner = ExperimentRunner(config)
        self.analyzer = ResultAnalyzer(config)

    async def run(
        self,
        base_experiment: ExperimentIteration,
        factors: list[str],
        llm: Any,
        config: ExperimentConfig | None = None,
    ) -> dict[str, Any]:
        """Run ablation study by removing one factor at a time.

        Args:
            base_experiment: The baseline experiment iteration
            factors: List of factor names to ablate
            llm: LLM instance for code modification
            config: ExperimentConfig (uses self.config if None)

        Returns:
            Dictionary with ablation results and factor importance ranking
        """
        if config is None:
            config = self.config

        self.logger.info(f"Running ablation study on {len(factors)} factors")

        base_metrics = base_experiment.result.metrics if base_experiment.result else {}
        ablation_results: dict[str, dict[str, Any]] = {}

        for factor in factors:
            self.logger.info(f"Ablating factor: {factor}")

            # Generate modified code without this factor
            prompt = f"""You are an expert experimental designer. Modify this experiment code to remove 
the '{factor}' factor/component.

Original Code:
```python
{base_experiment.code}
```

Generate modified code that:
1. Removes or disables the '{factor}' component
2. Maintains all other aspects of the experiment
3. Still outputs metrics in the same format
4. Preserves the experimental setup otherwise

Wrap the code in triple backticks:
```python
...
```
"""

            try:
                response = await llm.call(prompt, complexity=TaskComplexity.STANDARD)
                content = response.content if hasattr(response, 'content') else str(response)

                # Extract code
                code_match = re.search(r'```python\n([\s\S]*?)\n```', content)
                if not code_match:
                    code_match = re.search(r'```\n([\s\S]*?)\n```', content)
                if not code_match:
                    self.logger.warning(f"Could not extract code for factor {factor}")
                    continue

                modified_code = code_match.group(1)

                # Run ablated experiment
                result = await self.runner.run(modified_code, config)

                # Compare metrics
                importance = self._compute_importance(
                    base_metrics, result.metrics
                )

                ablation_results[factor] = {
                    "metrics": result.metrics,
                    "importance": importance,
                    "success": result.success,
                }

                self.logger.debug(
                    f"Factor '{factor}' importance: {importance:.4f}"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to ablate factor '{factor}': {e}", exc_info=True
                )

        # Rank factors by importance
        ranked_factors = sorted(
            ablation_results.items(),
            key=lambda x: x[1]['importance'],
            reverse=True,
        )

        return {
            "base_metrics": base_metrics,
            "ablation_results": ablation_results,
            "ranked_factors": [name for name, _ in ranked_factors],
            "importance_scores": {name: data['importance'] for name, data in ranked_factors},
        }

    def _compute_importance(
        self,
        base_metrics: dict[str, float],
        ablated_metrics: dict[str, float],
    ) -> float:
        """Compute importance of a factor based on performance drop.

        Returns a score from 0.0 (not important) to 1.0 (very important).
        """
        if not base_metrics or not ablated_metrics:
            return 0.0

        # Simple approach: average relative change across all metrics
        changes = []
        for metric_name, base_val in base_metrics.items():
            ablated_val = ablated_metrics.get(metric_name)
            if ablated_val is not None and base_val != 0:
                relative_change = abs(ablated_val - base_val) / abs(base_val)
                changes.append(relative_change)

        if not changes:
            return 0.0

        # Normalize to 0-1 range (cap at 1.0)
        importance = min(sum(changes) / len(changes), 1.0)
        return importance
