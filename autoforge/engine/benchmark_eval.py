"""
Benchmark Evaluation Harness

Provides standardized evaluation on mathematical reasoning benchmarks:
- miniF2F (244 problems: 122 valid + 122 test)
- PutnamBench (440 Putnam competition problems)
- LeanWorkbook (10K+ Lean 4 problems)
- ProofNet (50K+ theorems)
- Custom JSON-based benchmarks

Usage:
    suite = BenchmarkSuite()
    reports = await suite.run_all(prover, [BenchmarkType.MINIF2F])
    print(reports[BenchmarkType.MINIF2F].to_markdown())
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from math import comb
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Standard mathematical reasoning benchmarks."""
    MINIF2F = "minif2f"
    PUTNAM_BENCH = "putnam_bench"
    LEAN_WORKBOOK = "lean_workbook"
    PROOF_NET = "proof_net"
    CUSTOM = "custom"


@dataclass
class BenchmarkProblem:
    """A single benchmark problem with formal and informal statements."""
    id: str
    name: str
    formal_statement: str  # Lean 4 statement
    informal_statement: str = ""
    informal_proof: str = ""
    difficulty: str = ""  # e.g., "imo", "amc", "aime", "putnam"
    source: str = ""  # e.g., "imo_2019_p1"
    split: str = "test"  # train/valid/test
    tags: list[str] = field(default_factory=list)
    ground_truth_proof: str = ""

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class ProofResult:
    """Result of a proof attempt."""
    problem_id: str
    success: bool
    proof: str = ""
    has_sorry: bool = False
    compilation_verified: bool = False
    attempts: int = 0
    time_seconds: float = 0.0
    error: str = ""
    tactic_trace: list[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Aggregated results from a benchmark run."""
    benchmark: BenchmarkType
    total_problems: int
    attempted: int
    solved: int
    pass_at_1: float
    pass_at_k: dict[int, float] = field(default_factory=dict)  # k -> pass rate
    solve_rate_by_difficulty: dict[str, float] = field(default_factory=dict)
    solve_rate_by_tag: dict[str, float] = field(default_factory=dict)
    avg_time_per_problem: float = 0.0
    avg_attempts_per_solve: float = 0.0
    results: list[ProofResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "benchmark": self.benchmark.value,
            "total_problems": self.total_problems,
            "attempted": self.attempted,
            "solved": self.solved,
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "solve_rate_by_difficulty": self.solve_rate_by_difficulty,
            "solve_rate_by_tag": self.solve_rate_by_tag,
            "avg_time_per_problem": self.avg_time_per_problem,
            "avg_attempts_per_solve": self.avg_attempts_per_solve,
            "timestamp": self.timestamp,
            "results_count": len(self.results),
        }

    def to_markdown(self) -> str:
        """Generate markdown-formatted report."""
        lines = [
            f"# Benchmark Report: {self.benchmark.value.upper()}",
            "",
            f"**Timestamp:** {datetime.fromtimestamp(self.timestamp).isoformat()}",
            "",
            "## Summary Statistics",
            "",
            f"- **Total Problems:** {self.total_problems}",
            f"- **Attempted:** {self.attempted}",
            f"- **Solved:** {self.solved}",
            f"- **Pass@1:** {self.pass_at_1:.1%}",
            "",
        ]

        if self.pass_at_k:
            lines.extend([
                "## Pass@k Results",
                "",
            ])
            for k in sorted(self.pass_at_k.keys()):
                lines.append(f"- **Pass@{k}:** {self.pass_at_k[k]:.1%}")
            lines.append("")

        if self.solve_rate_by_difficulty:
            lines.extend([
                "## Solve Rate by Difficulty",
                "",
            ])
            for difficulty in sorted(self.solve_rate_by_difficulty.keys()):
                rate = self.solve_rate_by_difficulty[difficulty]
                lines.append(f"- **{difficulty}:** {rate:.1%}")
            lines.append("")

        if self.solve_rate_by_tag:
            lines.extend([
                "## Solve Rate by Tag",
                "",
            ])
            for tag in sorted(self.solve_rate_by_tag.keys()):
                rate = self.solve_rate_by_tag[tag]
                lines.append(f"- **{tag}:** {rate:.1%}")
            lines.append("")

        lines.extend([
            "## Timing Statistics",
            "",
            f"- **Avg Time per Problem:** {self.avg_time_per_problem:.2f}s",
            f"- **Avg Attempts per Solve:** {self.avg_attempts_per_solve:.2f}",
            "",
        ])

        if len(self.results) <= 50:
            lines.extend([
                "## Problem Results",
                "",
                "| Problem ID | Success | Time (s) | Attempts |",
                "|---|---|---|---|",
            ])
            for result in self.results:
                status = "✓" if result.success else "✗"
                lines.append(
                    f"| {result.problem_id} | {status} | {result.time_seconds:.2f} | {result.attempts} |"
                )

        return "\n".join(lines)


@dataclass
class BenchmarkRunConfig:
    """Configuration for running benchmarks."""
    benchmark_type: BenchmarkType
    n_samples: int = 1  # For pass@k, need n >= k
    max_concurrent: int = 4
    timeout_per_problem: int = 300
    lean_project_dir: Path | None = None
    save_proofs: bool = True
    output_dir: Path | None = None


class PassAtKEstimator:
    """Utility for computing pass@k metrics."""

    @staticmethod
    def compute_pass_at_k(n: int, c: int, k: int) -> float:
        """
        Compute pass@k using unbiased estimator.

        Args:
            n: Total number of attempts
            c: Number of successes
            k: k for pass@k metric

        Returns:
            Pass@k probability (0 to 1)
        """
        if n <= 0 or k <= 0 or c <= 0:
            return 0.0

        # k cannot exceed available samples for a problem.
        k_eff = min(k, n)
        c_eff = min(c, n)

        # If failures are fewer than draws, at least one success is guaranteed.
        if (n - c_eff) < k_eff:
            return 1.0

        # Use unbiased estimator: 1 - comb(n-c, k) / comb(n, k)
        try:
            return max(0.0, 1.0 - comb(n - c_eff, k_eff) / comb(n, k_eff))
        except (ValueError, OverflowError):
            return 1.0 if c_eff > 0 else 0.0

    @staticmethod
    def compute_from_results(
        results: list[list[bool]], k_values: list[int]
    ) -> dict[int, float]:
        """
        Compute pass@k from n independent trial results.

        Args:
            results: List of [n_samples] bool lists for each problem
            k_values: Values of k to compute pass@k for

        Returns:
            Dictionary mapping k -> average pass@k
        """
        pass_at_k_scores = {k: [] for k in k_values}

        for problem_results in results:
            n = len(problem_results)
            c = sum(problem_results)

            for k in k_values:
                score = PassAtKEstimator.compute_pass_at_k(n, c, k)
                pass_at_k_scores[k].append(score)

        return {
            k: (sum(scores) / len(scores) if scores else 0.0)
            for k, scores in pass_at_k_scores.items()
        }


class BenchmarkLoader:
    """Loads benchmark problems from various sources."""

    # Built-in example problems for testing without full downloads
    BUILTIN_EXAMPLES = [
        BenchmarkProblem(
            id="imo_2019_p1",
            name="IMO 2019 Problem 1",
            formal_statement="""theorem imo_2019_p1 (f : ℕ → ℕ) (h1 : ∀ n : ℕ, f (n + 1) > f n)
    (h2 : ∀ n : ℕ, f (f n) = f n + 1) : f 1 = 2 := by
  sorry""",
            informal_statement="Find all functions f from ℕ to ℕ such that f(n+1) > f(n) and f(f(n)) = f(n) + 1",
            difficulty="imo",
            source="imo_2019_p1",
            split="test",
            tags=["functions", "imo"],
        ),
        BenchmarkProblem(
            id="imo_2020_p2",
            name="IMO 2020 Problem 2",
            formal_statement="""theorem imo_2020_p2 : ∀ (a b : ℝ), a > 0 → b > 0 →
    (a + b)^2 / (a * b) ≥ 4 := by
  sorry""",
            informal_statement="Prove AM-GM inequality",
            difficulty="imo",
            source="imo_2020_p2",
            split="test",
            tags=["inequalities", "imo"],
        ),
        BenchmarkProblem(
            id="amc_12_2021_p1",
            name="AMC 12 2021 Problem 1",
            formal_statement="""theorem amc_12_2021_p1 : (2^3)^2 = 64 := by
  norm_num""",
            informal_statement="Compute (2^3)^2",
            difficulty="amc",
            source="amc_12_2021_p1",
            split="test",
            tags=["arithmetic", "amc"],
        ),
        BenchmarkProblem(
            id="aime_2022_p1",
            name="AIME 2022 Problem 1",
            formal_statement="""theorem aime_2022_p1 : (20^2 + 22^2 + 24^2 + 26^2 + 28^2) / 5 = 600 := by
  norm_num""",
            informal_statement="Compute the average of squares",
            difficulty="aime",
            source="aime_2022_p1",
            split="test",
            tags=["arithmetic", "aime"],
        ),
        BenchmarkProblem(
            id="putnam_2020_a1",
            name="Putnam 2020 Problem A1",
            formal_statement="""theorem putnam_2020_a1 : ∀ (x y z : ℚ),
    (x + 1) * (y + 1) * (z + 1) = 10 →
    x * y + x * z + y * z ≤ 7 := by
  sorry""",
            informal_statement="Find the maximum of xy + xz + yz given (x+1)(y+1)(z+1) = 10",
            difficulty="putnam",
            source="putnam_2020_a1",
            split="test",
            tags=["optimization", "putnam"],
        ),
        BenchmarkProblem(
            id="algebra_basic_1",
            name="Basic Algebra: Quadratic Formula",
            formal_statement="""theorem algebra_basic_1 : ∀ (a b c x : ℝ),
    a ≠ 0 → a * x^2 + b * x + c = 0 →
    x = (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a) ∨
    x = (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a) := by
  sorry""",
            informal_statement="Quadratic formula",
            difficulty="algebra",
            source="algebra_basic_1",
            split="train",
            tags=["algebra", "equations"],
        ),
        BenchmarkProblem(
            id="geometry_basic_1",
            name="Basic Geometry: Pythagorean Theorem",
            formal_statement="""theorem geometry_basic_1 : ∀ (a b c : ℝ),
    a > 0 → b > 0 → c > 0 →
    a^2 + b^2 = c^2 ↔
    (∃ (x y z : ℝ), (x : ℕ)^2 + (y : ℕ)^2 = (z : ℕ)^2) := by
  sorry""",
            informal_statement="Pythagorean theorem",
            difficulty="geometry",
            source="geometry_basic_1",
            split="train",
            tags=["geometry", "triangles"],
        ),
        BenchmarkProblem(
            id="number_theory_primes",
            name="Number Theory: Primes",
            formal_statement="""theorem number_theory_primes : ∃ (p : ℕ), Nat.Prime p ∧ p > 2 ∧ (p + 2) ∈ Nat.Prime := by
  sorry""",
            informal_statement="Twin primes conjecture (existence)",
            difficulty="number_theory",
            source="number_theory_primes",
            split="test",
            tags=["number_theory", "primes"],
        ),
        BenchmarkProblem(
            id="combinatorics_basic_1",
            name="Combinatorics: Binomial Coefficient",
            formal_statement="""theorem combinatorics_basic_1 : ∀ (n k : ℕ),
    k ≤ n → (Nat.choose n k : ℚ) = (n.factorial : ℚ) / ((k.factorial : ℚ) * ((n - k).factorial : ℚ)) := by
  sorry""",
            informal_statement="Binomial coefficient definition",
            difficulty="combinatorics",
            source="combinatorics_basic_1",
            split="train",
            tags=["combinatorics", "binomial"],
        ),
        BenchmarkProblem(
            id="calc_limits",
            name="Calculus: Limits",
            formal_statement="""theorem calc_limits : Filter.Tendsto (fun x : ℝ => (x^2 - 1) / (x - 1))
    (𝓝[≠] 1) (𝓝 2) := by
  sorry""",
            informal_statement="Compute lim_{x→1} (x²-1)/(x-1) = 2",
            difficulty="calculus",
            source="calc_limits",
            split="test",
            tags=["calculus", "limits"],
        ),
        BenchmarkProblem(
            id="logic_basic_1",
            name="Logic: De Morgan's Laws",
            formal_statement="""theorem logic_basic_1 : ∀ (p q : Prop),
    (¬(p ∨ q)) ↔ (¬p ∧ ¬q) := by
  tauto""",
            informal_statement="De Morgan's law: ¬(p ∨ q) ↔ ¬p ∧ ¬q",
            difficulty="logic",
            source="logic_basic_1",
            split="train",
            tags=["logic", "boolean"],
        ),
        BenchmarkProblem(
            id="set_theory_1",
            name="Set Theory: Union and Intersection",
            formal_statement="""theorem set_theory_1 : ∀ (A B C : Set ℕ),
    A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := by
  ext x
  tauto""",
            informal_statement="Set distributivity: A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)",
            difficulty="set_theory",
            source="set_theory_1",
            split="train",
            tags=["sets", "logic"],
        ),
        BenchmarkProblem(
            id="graph_theory_1",
            name="Graph Theory: Handshaking Lemma",
            formal_statement="""theorem graph_theory_1 : ∀ (n : ℕ) (degrees : Fin n → ℕ),
    (Finset.sum Finset.univ fun i => degrees i) % 2 = 0 := by
  sorry""",
            informal_statement="Sum of vertex degrees is even",
            difficulty="graph_theory",
            source="graph_theory_1",
            split="test",
            tags=["graph_theory", "combinatorics"],
        ),
        BenchmarkProblem(
            id="linear_algebra_1",
            name="Linear Algebra: Matrix Determinant",
            formal_statement="""theorem linear_algebra_1 : ∀ (A B : Matrix (Fin 2) (Fin 2) ℝ),
    Matrix.det (A * B) = Matrix.det A * Matrix.det B := by
  sorry""",
            informal_statement="Multiplicativity of determinant",
            difficulty="linear_algebra",
            source="linear_algebra_1",
            split="train",
            tags=["linear_algebra", "matrices"],
        ),
        BenchmarkProblem(
            id="topology_1",
            name="Topology: Open Sets",
            formal_statement="""theorem topology_1 : ∀ (X : Type*) [TopologicalSpace X] (s : Set X),
    IsOpen (sᶜ) ↔ IsClosed s := by
  exact isOpen_compl_iff""",
            informal_statement="Complement of open set is closed",
            difficulty="topology",
            source="topology_1",
            split="train",
            tags=["topology", "open_sets"],
        ),
        BenchmarkProblem(
            id="probability_1",
            name="Probability: Bayes' Rule",
            formal_statement="""theorem probability_1 : ∀ (A B : Set Ω) (p : Ω → ℝ),
    p B > 0 →
    p (A ∩ B) / p B = p A * (p B / p A) / p B := by
  sorry""",
            informal_statement="Bayes' theorem statement",
            difficulty="probability",
            source="probability_1",
            split="test",
            tags=["probability", "conditional"],
        ),
    ]

    @staticmethod
    async def load_minif2f(data_dir: Path | None = None) -> list[BenchmarkProblem]:
        """
        Load miniF2F benchmark (244 problems: 122 valid + 122 test).

        If data_dir not provided, returns built-in examples and logs warning.
        """
        if data_dir is None or not data_dir.exists():
            logger.warning(
                "miniF2F data not found. Using built-in examples. "
                "For full benchmark, clone: https://github.com/facebookresearch/miniF2F"
            )
            return BenchmarkLoader.BUILTIN_EXAMPLES[:5]

        problems = []
        try:
            for split in ["valid", "test"]:
                split_dir = data_dir / split
                if not split_dir.exists():
                    continue

                for lean_file in split_dir.glob("*.lean"):
                    parsed = await BenchmarkLoader._parse_lean_file(lean_file)
                    for p in parsed:
                        p.split = split
                        p.difficulty = "imo"
                        problems.append(p)

            logger.info(f"Loaded {len(problems)} miniF2F problems from {data_dir}")
        except Exception as e:
            logger.error(f"Error loading miniF2F: {e}. Using built-in examples.")
            return BenchmarkLoader.BUILTIN_EXAMPLES[:5]

        return problems if problems else BenchmarkLoader.BUILTIN_EXAMPLES[:5]

    @staticmethod
    async def load_putnam_bench(data_dir: Path | None = None) -> list[BenchmarkProblem]:
        """Load PutnamBench (440 Putnam competition problems)."""
        if data_dir is None or not data_dir.exists():
            logger.warning(
                "PutnamBench data not found. Using built-in examples. "
                "For full benchmark: https://github.com/leanprover-community/putnam-bench"
            )
            return [p for p in BenchmarkLoader.BUILTIN_EXAMPLES if "putnam" in p.difficulty.lower()]

        problems = []
        try:
            for lean_file in data_dir.glob("**/*.lean"):
                parsed = await BenchmarkLoader._parse_lean_file(lean_file)
                for p in parsed:
                    p.difficulty = "putnam"
                    problems.append(p)

            logger.info(f"Loaded {len(problems)} Putnam problems from {data_dir}")
        except Exception as e:
            logger.error(f"Error loading PutnamBench: {e}. Using built-in examples.")
            return [p for p in BenchmarkLoader.BUILTIN_EXAMPLES if "putnam" in p.difficulty.lower()]

        return problems

    @staticmethod
    async def load_lean_workbook(
        data_dir: Path | None = None, limit: int = 1000
    ) -> list[BenchmarkProblem]:
        """Load LeanWorkbook (10K+ Lean 4 problems, optionally limited)."""
        if data_dir is None or not data_dir.exists():
            logger.warning(
                f"LeanWorkbook data not found. Using built-in examples. "
                f"For full benchmark: https://github.com/leanprover-community/lean-workbook"
            )
            return BenchmarkLoader.BUILTIN_EXAMPLES

        problems = []
        try:
            for lean_file in sorted(data_dir.glob("**/*.lean"))[:limit]:
                parsed = await BenchmarkLoader._parse_lean_file(lean_file)
                problems.extend(parsed)

            logger.info(f"Loaded {len(problems)} Lean Workbook problems from {data_dir} (limit: {limit})")
        except Exception as e:
            logger.error(f"Error loading LeanWorkbook: {e}. Using built-in examples.")
            return BenchmarkLoader.BUILTIN_EXAMPLES

        return problems[:limit]

    @staticmethod
    async def load_proof_net(data_dir: Path | None = None) -> list[BenchmarkProblem]:
        """Load ProofNet (50K+ theorems)."""
        if data_dir is None or not data_dir.exists():
            logger.warning(
                "ProofNet data not found. Using built-in examples. "
                "For full benchmark: https://github.com/leanprover-community/proof-net"
            )
            return BenchmarkLoader.BUILTIN_EXAMPLES

        problems = []
        try:
            for lean_file in data_dir.glob("**/*.lean"):
                parsed = await BenchmarkLoader._parse_lean_file(lean_file)
                problems.extend(parsed)

            logger.info(f"Loaded {len(problems)} ProofNet problems from {data_dir}")
        except Exception as e:
            logger.error(f"Error loading ProofNet: {e}. Using built-in examples.")
            return BenchmarkLoader.BUILTIN_EXAMPLES

        return problems

    @staticmethod
    async def load_custom(problems_json: Path) -> list[BenchmarkProblem]:
        """Load problems from custom JSON file."""
        if not problems_json.exists():
            raise FileNotFoundError(f"Custom problems file not found: {problems_json}")

        try:
            with open(problems_json, "r") as f:
                data = json.load(f)

            problems = []
            for item in data if isinstance(data, list) else data.get("problems", []):
                p = BenchmarkProblem(
                    id=item["id"],
                    name=item.get("name", item["id"]),
                    formal_statement=item["formal_statement"],
                    informal_statement=item.get("informal_statement", ""),
                    informal_proof=item.get("informal_proof", ""),
                    difficulty=item.get("difficulty", ""),
                    source=item.get("source", ""),
                    split=item.get("split", "test"),
                    tags=item.get("tags", []),
                    ground_truth_proof=item.get("ground_truth_proof", ""),
                )
                problems.append(p)

            logger.info(f"Loaded {len(problems)} custom problems from {problems_json}")
            return problems

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid custom problems JSON: {e}") from e

    @staticmethod
    async def _parse_lean_file(path: Path) -> list[BenchmarkProblem]:
        """
        Parse .lean file to extract theorem statements.

        Extracts theorem declarations and their formal statements.
        """
        problems = []
        try:
            with open(path, "r") as f:
                content = f.read()

            # Simple regex-based extraction of theorem statements
            import re

            # Match: theorem name (args) : statement := proof
            pattern = r"theorem\s+(\w+)\s*([^:]*?):\s*([^:=]+)(?::=\s*(.+?))?(?:\n(?=theorem|\w+)|\Z)"
            matches = re.finditer(pattern, content, re.DOTALL)

            for match in matches:
                name = match.group(1)
                formal_stmt = match.group(3).strip()

                if formal_stmt:
                    p = BenchmarkProblem(
                        id=f"{path.stem}_{name}",
                        name=name,
                        formal_statement=f"theorem {name}{match.group(2).strip()}: {formal_stmt}",
                        source=path.stem,
                    )
                    problems.append(p)

        except Exception as e:
            logger.debug(f"Error parsing {path}: {e}")

        return problems

    @staticmethod
    async def _download_benchmark(repo_url: str, target_dir: Path) -> Path:
        """
        Download/clone benchmark data from GitHub.

        Args:
            repo_url: GitHub repository URL
            target_dir: Directory to clone into

        Returns:
            Path to cloned repository
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "clone", repo_url, str(target_dir)],
                capture_output=True,
                timeout=300,
            )

            if result.returncode == 0:
                logger.info(f"Downloaded benchmark from {repo_url} to {target_dir}")
                return target_dir
            else:
                logger.error(f"Failed to clone {repo_url}: {result.stderr.decode()}")
                return target_dir

        except subprocess.TimeoutExpired:
            logger.error(f"Download timeout for {repo_url}")
            return target_dir
        except Exception as e:
            logger.error(f"Error downloading benchmark: {e}")
            return target_dir


class BenchmarkRunner:
    """Runs proofs against benchmark problems."""

    def __init__(self, prover: Any, config: BenchmarkRunConfig) -> None:
        """
        Initialize runner.

        Args:
            prover: Object with async prove(statement, informal_hint, max_attempts) method
            config: Benchmark configuration
        """
        self.prover = prover
        self.config = config

    async def run_benchmark(
        self,
        problems: list[BenchmarkProblem],
        *,
        n_samples: int = 1,
        max_concurrent: int = 4,
    ) -> BenchmarkReport:
        """
        Run benchmark on all problems.

        Args:
            problems: List of benchmark problems
            n_samples: Number of proof attempts per problem (for pass@k)
            max_concurrent: Max concurrent problems

        Returns:
            Aggregated report
        """
        logger.info(f"Starting benchmark: {len(problems)} problems, {n_samples} samples each")

        results: dict[str, list[ProofResult]] = {}
        semaphore = asyncio.Semaphore(max_concurrent)

        async def solve_with_semaphore(problem: BenchmarkProblem) -> None:
            async with semaphore:
                proof_results = await self._solve_problem(problem, n_samples)
                results[problem.id] = proof_results

        # Run all problems concurrently
        tasks = [solve_with_semaphore(p) for p in problems]
        await asyncio.gather(*tasks)

        # Compute report
        report = self._compute_report(results, self.config.benchmark_type)
        logger.info(f"Benchmark complete. Solved: {report.solved}/{report.total_problems} "
                   f"(Pass@1: {report.pass_at_1:.1%})")

        return report

    async def _solve_problem(
        self, problem: BenchmarkProblem, n_samples: int
    ) -> list[ProofResult]:
        """Attempt to solve a problem n_samples times."""
        results = []

        for attempt_num in range(n_samples):
            start_time = time.time()
            result = ProofResult(
                problem_id=problem.id,
                success=False,
                attempts=attempt_num + 1,
            )

            try:
                # Call prover (assume it has async prove method)
                if hasattr(self.prover, "prove"):
                    proof = await asyncio.wait_for(
                        self.prover.prove(
                            problem.formal_statement,
                            problem.informal_proof,
                            max_attempts=self.config.timeout_per_problem,
                        ),
                        timeout=self.config.timeout_per_problem,
                    )

                    if proof:
                        result.proof = proof
                        result.success = await self._verify_proof(
                            problem.formal_statement, proof
                        )
                        result.compilation_verified = result.success
                else:
                    logger.warning(f"Prover has no prove method: {type(self.prover)}")

            except asyncio.TimeoutError:
                result.error = "Timeout"
            except Exception as e:
                result.error = str(e)
                logger.debug(f"Error solving {problem.id}: {e}")

            result.time_seconds = time.time() - start_time
            results.append(result)

        return results

    async def _verify_proof(self, statement: str, proof: str) -> bool:
        """
        Verify proof by attempting Lean 4 compilation.

        Returns True if proof compiles without errors.
        """
        if not proof or not statement:
            return False

        # Check for sorry (incomplete proof)
        if "sorry" in proof.lower():
            return False

        try:
            # Create temp Lean file and attempt compilation
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lean", delete=False
            ) as f:
                f.write(statement + "\n" + proof)
                temp_file = f.name

            # Attempt to compile with Lean 4
            result = await asyncio.to_thread(
                subprocess.run,
                ["lean", temp_file],
                capture_output=True,
                timeout=10,
            )

            Path(temp_file).unlink(missing_ok=True)
            return result.returncode == 0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Lean not available or timeout
            return False
        except Exception as e:
            logger.debug(f"Verification error: {e}")
            return False

    def _compute_report(
        self, results: dict[str, list[ProofResult]], benchmark_type: BenchmarkType
    ) -> BenchmarkReport:
        """Aggregate results into report."""
        total = len(results)
        attempted = sum(1 for res_list in results.values() if res_list)
        solved = sum(1 for res_list in results.values() if any(r.success for r in res_list))

        # Compute pass@1 and pass@k
        all_results = []
        success_by_sample = []

        for problem_id, res_list in results.items():
            if res_list:
                all_results.extend(res_list)
                success_by_sample.append([r.success for r in res_list])

        # pass@1 must use first sample only (not "any of n samples").
        first_sample_successes = sum(
            1
            for res_list in results.values()
            if res_list and res_list[0].success
        )
        pass_at_1 = first_sample_successes / total if total > 0 else 0.0

        # Compute pass@k
        pass_at_k = {}
        if success_by_sample:
            k_values = [1, 8, 32, 128]
            pass_at_k = PassAtKEstimator.compute_from_results(
                success_by_sample, k_values
            )

        # Compute timing statistics
        total_time = sum(r.time_seconds for r in all_results)
        avg_time = total_time / len(all_results) if all_results else 0.0

        solved_results = [r for r in all_results if r.success]
        avg_attempts = (
            sum(r.attempts for r in solved_results) / len(solved_results)
            if solved_results
            else 0.0
        )

        return BenchmarkReport(
            benchmark=benchmark_type,
            total_problems=total,
            attempted=attempted,
            solved=solved,
            pass_at_1=pass_at_1,
            pass_at_k=pass_at_k,
            avg_time_per_problem=avg_time,
            avg_attempts_per_solve=avg_attempts,
            results=all_results,
        )


class BenchmarkSuite:
    """Convenience orchestrator for running multiple benchmarks."""

    async def run_all(
        self,
        prover: Any,
        benchmarks: list[BenchmarkType] | None = None,
        config: BenchmarkRunConfig | None = None,
    ) -> dict[BenchmarkType, BenchmarkReport]:
        """
        Run all specified benchmarks.

        Args:
            prover: Proof generator
            benchmarks: Benchmarks to run (default: [MINIF2F])
            config: Shared configuration

        Returns:
            Dictionary of benchmark type -> report
        """
        if benchmarks is None:
            benchmarks = [BenchmarkType.MINIF2F]

        reports = {}

        for benchmark_type in benchmarks:
            logger.info(f"Running {benchmark_type.value} benchmark...")

            # Load problems
            if benchmark_type == BenchmarkType.MINIF2F:
                problems = await BenchmarkLoader.load_minif2f()
            elif benchmark_type == BenchmarkType.PUTNAM_BENCH:
                problems = await BenchmarkLoader.load_putnam_bench()
            elif benchmark_type == BenchmarkType.LEAN_WORKBOOK:
                problems = await BenchmarkLoader.load_lean_workbook()
            elif benchmark_type == BenchmarkType.PROOF_NET:
                problems = await BenchmarkLoader.load_proof_net()
            else:
                logger.warning(f"Unknown benchmark type: {benchmark_type}")
                continue

            # Create config if not provided
            if config is None:
                cfg = BenchmarkRunConfig(benchmark_type=benchmark_type)
            else:
                cfg = BenchmarkRunConfig(
                    benchmark_type=benchmark_type,
                    n_samples=config.n_samples,
                    max_concurrent=config.max_concurrent,
                    timeout_per_problem=config.timeout_per_problem,
                    lean_project_dir=config.lean_project_dir,
                    save_proofs=config.save_proofs,
                    output_dir=config.output_dir,
                )

            # Run benchmark
            runner = BenchmarkRunner(prover, cfg)
            report = await runner.run_benchmark(problems)
            reports[benchmark_type] = report

        return reports

    async def compare_provers(
        self,
        provers: dict[str, Any],
        benchmark: BenchmarkType,
        config: BenchmarkRunConfig | None = None,
    ) -> dict[str, BenchmarkReport]:
        """
        Compare multiple provers on a single benchmark.

        Args:
            provers: Dictionary of prover_name -> prover_object
            benchmark: Benchmark to use
            config: Shared configuration

        Returns:
            Dictionary of prover_name -> report
        """
        logger.info(f"Comparing {len(provers)} provers on {benchmark.value}...")

        # Load problems once
        if benchmark == BenchmarkType.MINIF2F:
            problems = await BenchmarkLoader.load_minif2f()
        elif benchmark == BenchmarkType.PUTNAM_BENCH:
            problems = await BenchmarkLoader.load_putnam_bench()
        elif benchmark == BenchmarkType.LEAN_WORKBOOK:
            problems = await BenchmarkLoader.load_lean_workbook()
        elif benchmark == BenchmarkType.PROOF_NET:
            problems = await BenchmarkLoader.load_proof_net()
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark}")

        reports = {}

        for prover_name, prover in provers.items():
            logger.info(f"Running {prover_name}...")

            cfg = BenchmarkRunConfig(benchmark_type=benchmark)
            if config:
                cfg = BenchmarkRunConfig(
                    benchmark_type=benchmark,
                    n_samples=config.n_samples,
                    max_concurrent=config.max_concurrent,
                    timeout_per_problem=config.timeout_per_problem,
                    lean_project_dir=config.lean_project_dir,
                    save_proofs=config.save_proofs,
                    output_dir=config.output_dir,
                )

            runner = BenchmarkRunner(prover, cfg)
            report = await runner.run_benchmark(problems)
            reports[prover_name] = report

        return reports

    @staticmethod
    def _format_comparison_table(reports: dict[str, BenchmarkReport]) -> str:
        """Format prover comparison as markdown table."""
        lines = [
            "# Prover Comparison",
            "",
            "| Prover | Solved | Pass@1 | Pass@8 | Avg Time (s) |",
            "|---|---|---|---|---|",
        ]

        for prover_name, report in sorted(reports.items()):
            pass_at_8 = report.pass_at_k.get(8, 0.0)
            lines.append(
                f"| {prover_name} | {report.solved}/{report.total_problems} | "
                f"{report.pass_at_1:.1%} | {pass_at_8:.1%} | {report.avg_time_per_problem:.2f} |"
            )

        return "\n".join(lines)


# Example usage in main or tests
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        """Example: run built-in benchmark suite."""
        # Create a dummy prover for testing
        class DummyProver:
            async def prove(self, statement, hint, max_attempts):
                # Dummy implementation that always fails
                return None

        # Run mini benchmark
        suite = BenchmarkSuite()
        reports = await suite.run_all(DummyProver(), [BenchmarkType.MINIF2F])

        # Print report
        report = reports[BenchmarkType.MINIF2F]
        print(report.to_markdown())
        print("\n" + report.to_dict().__repr__())

    asyncio.run(main())
