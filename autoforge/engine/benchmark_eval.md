# Benchmark Evaluation Harness

Standardized evaluation framework for mathematical reasoning provers on benchmark datasets.

## Quick Start

```python
import asyncio
from autoforge.engine.benchmark_eval import (
    BenchmarkType,
    BenchmarkSuite,
    BenchmarkRunConfig,
)

async def main():
    suite = BenchmarkSuite()
    
    # Run on miniF2F (244 problems)
    reports = await suite.run_all(
        prover=my_prover,
        benchmarks=[BenchmarkType.MINIF2F],
    )
    
    # Print results
    report = reports[BenchmarkType.MINIF2F]
    print(report.to_markdown())

asyncio.run(main())
```

## Benchmarks

### miniF2F
- **244 problems**: 122 valid + 122 test split
- **Source**: [facebookresearch/miniF2F](https://github.com/facebookresearch/miniF2F)
- **Covers**: IMO, AIME, AMC problems
- **Format**: Lean 4 theorem statements

### PutnamBench
- **440 problems**: Putnam exam questions
- **Source**: [leanprover-community/putnam-bench](https://github.com/leanprover-community/putnam-bench)
- **Difficulty**: Academic competition level
- **Format**: Lean 4 proofs

### LeanWorkbook
- **10K+ problems**: Large collection of Lean theorems
- **Source**: [leanprover-community/lean-workbook](https://github.com/leanprover-community/lean-workbook)
- **Variety**: Multiple mathematical domains
- **Customizable**: Load with `limit` parameter

### ProofNet
- **50K+ theorems**: Large-scale theorem corpus
- **Source**: [leanprover-community/proof-net](https://github.com/leanprover-community/proof-net)
- **Scale**: For large-scale evaluation
- **Format**: Lean 4 statements

### Custom
Load your own benchmark from JSON:

```python
custom_problems = await BenchmarkLoader.load_custom(
    Path("my_problems.json")
)
```

JSON format:
```json
[
  {
    "id": "problem_1",
    "name": "My Problem",
    "formal_statement": "theorem problem_1 : ... := by sorry",
    "informal_statement": "Prove that...",
    "difficulty": "intermediate",
    "source": "my_dataset",
    "tags": ["algebra", "equations"],
    "split": "test"
  }
]
```

## API Reference

### Enums

#### `BenchmarkType`
```python
BenchmarkType.MINIF2F          # 244 mathematical problems
BenchmarkType.PUTNAM_BENCH     # 440 Putnam problems
BenchmarkType.LEAN_WORKBOOK    # 10K+ Lean theorems
BenchmarkType.PROOF_NET        # 50K+ theorems
BenchmarkType.CUSTOM           # Custom JSON problems
```

### Dataclasses

#### `BenchmarkProblem`
```python
@dataclass
class BenchmarkProblem:
    id: str                        # Unique problem identifier
    name: str                      # Human-readable name
    formal_statement: str          # Lean 4 theorem statement
    informal_statement: str = ""   # Natural language description
    informal_proof: str = ""       # Hint or proof sketch
    difficulty: str = ""           # "imo", "amc", "aime", "putnam"
    source: str = ""               # Dataset origin
    split: str = "test"            # "train", "valid", or "test"
    tags: list[str] = []           # Problem categories
    ground_truth_proof: str = ""   # Reference proof (if available)
```

#### `ProofResult`
```python
@dataclass
class ProofResult:
    problem_id: str                # Which problem this solves
    success: bool                  # Whether proof succeeded
    proof: str = ""                # Generated proof text
    has_sorry: bool = False        # Contains incomplete sorry
    compilation_verified: bool = False  # Lean compiled successfully
    attempts: int = 0              # Number of attempts
    time_seconds: float = 0.0      # Elapsed time
    error: str = ""                # Error message if failed
    tactic_trace: list[str] = []   # Tactic sequence
```

#### `BenchmarkReport`
```python
@dataclass
class BenchmarkReport:
    benchmark: BenchmarkType
    total_problems: int
    attempted: int
    solved: int
    pass_at_1: float               # Pass@1 metric
    pass_at_k: dict[int, float]    # Pass@k for k=1,8,32,128
    solve_rate_by_difficulty: dict[str, float]
    solve_rate_by_tag: dict[str, float]
    avg_time_per_problem: float
    avg_attempts_per_solve: float
    results: list[ProofResult]
    timestamp: float
    
    # Methods:
    def to_dict(self) -> dict      # Convert to dictionary
    def to_markdown(self) -> str   # Markdown-formatted report
```

#### `BenchmarkRunConfig`
```python
@dataclass
class BenchmarkRunConfig:
    benchmark_type: BenchmarkType
    n_samples: int = 1             # Attempts per problem
    max_concurrent: int = 4        # Parallel problems
    timeout_per_problem: int = 300 # Seconds per problem
    lean_project_dir: Path = None  # Lean project root
    save_proofs: bool = True       # Save generated proofs
    output_dir: Path = None        # Output directory
```

### Classes

#### `BenchmarkLoader`
Load problems from various sources.

```python
# From standard benchmarks
problems = await BenchmarkLoader.load_minif2f(data_dir=None)
problems = await BenchmarkLoader.load_putnam_bench(data_dir=None)
problems = await BenchmarkLoader.load_lean_workbook(data_dir=None, limit=1000)
problems = await BenchmarkLoader.load_proof_net(data_dir=None)

# From custom source
problems = await BenchmarkLoader.load_custom(Path("problems.json"))

# Download/clone benchmark
await BenchmarkLoader._download_benchmark(
    "https://github.com/org/repo",
    Path("./data")
)

# Parse .lean files
problems = await BenchmarkLoader._parse_lean_file(Path("theorems.lean"))
```

**Note**: If benchmark data not available, uses built-in examples (~16 problems) with warning.

#### `PassAtKEstimator`
Compute pass@k metrics.

```python
# Compute for single problem: n attempts, c correct, k threshold
score = PassAtKEstimator.compute_pass_at_k(n=10, c=5, k=8)  # 0.978

# Compute from trial results: list of [n_samples] bools per problem
results = [
    [True, True],      # Problem 1: 2/2 correct
    [True, False],     # Problem 2: 1/2 correct
    [False, False],    # Problem 3: 0/2 correct
]
pass_at_k = PassAtKEstimator.compute_from_results(results, [1, 2, 4])
# {1: 0.333, 2: 0.667, 4: 1.0}
```

#### `BenchmarkRunner`
Run proofs against benchmark problems.

```python
config = BenchmarkRunConfig(
    benchmark_type=BenchmarkType.MINIF2F,
    n_samples=8,        # 8 attempts per problem
    max_concurrent=4,   # 4 parallel
    timeout_per_problem=300,
)

runner = BenchmarkRunner(prover=my_prover, config=config)

report = await runner.run_benchmark(
    problems=problems,
    n_samples=8,
    max_concurrent=4,
)

print(f"Solved: {report.solved}/{report.total_problems}")
print(f"Pass@1: {report.pass_at_1:.1%}")
print(report.to_markdown())
```

#### `BenchmarkSuite`
Orchestrate multiple benchmarks.

```python
suite = BenchmarkSuite()

# Run multiple benchmarks
reports = await suite.run_all(
    prover=my_prover,
    benchmarks=[
        BenchmarkType.MINIF2F,
        BenchmarkType.PUTNAM_BENCH,
    ],
)

for benchmark, report in reports.items():
    print(f"\n{benchmark.value}:")
    print(report.to_markdown())

# Compare multiple provers
comparison = await suite.compare_provers(
    provers={
        "ProverA": prover_a,
        "ProverB": prover_b,
    },
    benchmark=BenchmarkType.MINIF2F,
)

# Print comparison table
table = suite._format_comparison_table(comparison)
print(table)
```

## Complete Example

```python
import asyncio
import logging
from pathlib import Path
from autoforge.engine.benchmark_eval import (
    BenchmarkType,
    BenchmarkSuite,
    BenchmarkRunConfig,
    BenchmarkLoader,
)

logging.basicConfig(level=logging.INFO)

class MyProver:
    """Example prover implementation."""
    
    async def prove(self, statement: str, hint: str = "", max_attempts: int = 300):
        """Generate proof for statement."""
        # Your proof generation logic here
        # Return proof string or None if unsuccessful
        pass

async def main():
    # Initialize
    prover = MyProver()
    suite = BenchmarkSuite()
    
    # Run on miniF2F with 8 samples for pass@k
    print("Evaluating prover on miniF2F...")
    
    config = BenchmarkRunConfig(
        benchmark_type=BenchmarkType.MINIF2F,
        n_samples=8,
        max_concurrent=4,
        timeout_per_problem=300,
        output_dir=Path("./results"),
        save_proofs=True,
    )
    
    # Load problems
    problems = await BenchmarkLoader.load_minif2f(
        data_dir=Path("./benchmarks/minif2f")
    )
    
    # Run benchmark
    from autoforge.engine.benchmark_eval import BenchmarkRunner
    runner = BenchmarkRunner(prover, config)
    report = await runner.run_benchmark(problems, n_samples=8)
    
    # Print results
    print(report.to_markdown())
    
    # Save report
    with open("report.json", "w") as f:
        import json
        json.dump(report.to_dict(), f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
```

## Built-in Example Problems

The module includes ~16 built-in example problems for quick testing:
- **IMO problems**: imo_2019_p1, imo_2020_p2
- **AIME/AMC**: amc_12_2021_p1, aime_2022_p1
- **Putnam**: putnam_2020_a1
- **Basic Math**: algebra, geometry, combinatorics, etc.

These are automatically used when benchmark data is not available, with a warning logged.

## Metrics Explained

### Pass@k
The probability that at least one of k sampled proofs is correct. Useful for measuring quality diversity.

Formula (unbiased estimator):
```
Pass@k = 1 - C(n-c, k) / C(n, k)
```
Where:
- n = total samples
- c = correct samples
- C(n,k) = binomial coefficient

## Performance Tips

1. **Concurrent Evaluation**: Increase `max_concurrent` for parallel problem solving
2. **Timeout Tuning**: Adjust `timeout_per_problem` based on prover complexity
3. **Sampling**: Use `n_samples=1` for quick eval, `n_samples=32+` for pass@k research
4. **Data Caching**: Pre-download benchmarks to avoid repeated downloads
5. **Logging**: Set `logging.INFO` to track progress

## Integration with AutoForge

This module is designed to evaluate provers within the AutoForge framework:

```python
from autoforge.engine.lean_prover import LeanProver
from autoforge.engine.benchmark_eval import BenchmarkSuite, BenchmarkType

# Use with AutoForge's Lean prover
lean_prover = LeanProver(config)
suite = BenchmarkSuite()
reports = await suite.run_all(lean_prover, [BenchmarkType.MINIF2F])
```

## License

Part of AutoForge. See LICENSE in root directory.
