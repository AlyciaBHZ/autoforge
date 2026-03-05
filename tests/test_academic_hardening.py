from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from autoforge.engine.benchmark_eval import (
    BenchmarkRunConfig,
    BenchmarkRunner,
    BenchmarkType,
    PassAtKEstimator,
    ProofResult,
)
from autoforge.engine.dense_retrieval import (
    PremiseEntry,
    PremiseIndex,
    RetrievalConfig,
    TFIDFProvider,
)
from autoforge.engine.experiment_loop import ExperimentConfig
from autoforge.engine.literature_search import CitationGraph
from autoforge.engine.repro_contract import (
    REQUIRED_ARTIFACT_FILES,
    build_repro_report,
    validate_contract_artifacts,
)


class TestBenchmarkHardening(unittest.TestCase):
    def test_pass_at_k_regression_no_longer_overestimates(self) -> None:
        n, c, k = 100, 6, 5
        observed = PassAtKEstimator.compute_pass_at_k(n, c, k)
        expected = 1.0 - math.comb(n - c, k) / math.comb(n, k)
        self.assertAlmostEqual(observed, expected, places=12)
        self.assertLess(observed, 1.0)

    def test_pass_at_1_uses_first_sample_only(self) -> None:
        runner = BenchmarkRunner(
            prover=None,
            config=BenchmarkRunConfig(benchmark_type=BenchmarkType.MINIF2F),
        )
        results = {
            "p1": [ProofResult(problem_id="p1", success=False), ProofResult(problem_id="p1", success=True)],
            "p2": [ProofResult(problem_id="p2", success=False), ProofResult(problem_id="p2", success=False)],
        }
        report = runner._compute_report(results, BenchmarkType.MINIF2F)
        self.assertEqual(report.pass_at_1, 0.0)


class TestDenseRetrieverHardening(unittest.IsolatedAsyncioTestCase):
    async def test_tfidf_query_space_consistency(self) -> None:
        provider = TFIDFProvider(embedding_dim=8)
        index = PremiseIndex(RetrievalConfig(embedding_dim=8, top_k=3, similarity_threshold=-1.0))

        premises = [
            PremiseEntry(
                id=str(i),
                name=f"N{i}",
                statement=f"unique theorem token{i} prime matrix graph sequence",
                module_path="Mathlib/Test",
            )
            for i in range(40)
        ]
        await index.build_index(premises, provider)

        query = "N0: unique theorem token0 prime matrix graph sequence"
        hits = await index.search(query, top_k=1, provider=provider)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0].id, "0")
        self.assertGreater(hits[0][1], 0.0)


class TestContractHardening(unittest.TestCase):
    def test_simulated_mode_cannot_pass_contract(self) -> None:
        report = build_repro_report(
            run_id="run-1",
            paper_id="paper-1",
            goal="goal",
            mode="simulated_no_api_key",
            profile="theory-first",
            output_dir=Path("."),
            strict_contract=True,
            p0_p4_status={
                "P0_api_key_runtime": "resolved_with_simulation",
                "P1_goal_to_paper_retrieval": "ok",
                "P2_paper_signal_extraction": "ok",
                "P3_closed_loop_verification": "ok",
                "P4_environment_reproducibility": "ok",
            },
            artifacts_complete=True,
            failure_reasons=[],
        )
        self.assertEqual(report["pass_fail"], "fail")
        reasons = report.get("failure_reasons", [])
        self.assertTrue(any("Simulated run" in reason for reason in reasons))

    def test_validator_rejects_pass_with_non_generated_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name in REQUIRED_ARTIFACT_FILES:
                (base / name).write_text("{}", encoding="utf-8")

            manifest = {
                "required_artifacts": list(REQUIRED_ARTIFACT_FILES),
                "artifacts_written": list(REQUIRED_ARTIFACT_FILES),
            }
            (base / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )

            report = {
                "schema_version": "paper-repro-contract-v1",
                "run_id": "r1",
                "paper_id": "p1",
                "goal": "g",
                "mode": "artifact_only",
                "profile": "theory-first",
                "artifacts_complete": True,
                "pass_fail": "pass",
                "p0_p4_status": {
                    "P0_api_key_runtime": "not_requested",
                    "P1_goal_to_paper_retrieval": "ok",
                    "P2_paper_signal_extraction": "ok",
                    "P3_closed_loop_verification": "ok",
                    "P4_environment_reproducibility": "ok",
                },
                "failure_reasons": [],
                "strict_contract": True,
                "output_dir": str(base),
                "manifest_path": str(base / "run_manifest.json"),
                "report_path": str(base / "repro_report.json"),
                "generated_at": "2026-03-05T00:00:00Z",
            }
            (base / "repro_report.json").write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )

            result = validate_contract_artifacts(base)
            self.assertFalse(result.ok)
            self.assertTrue(
                any("requires mode=generated_with_api_key" in err for err in result.errors)
            )


class TestExecutionDefaults(unittest.TestCase):
    def test_experiment_config_uses_current_python(self) -> None:
        cfg = ExperimentConfig(workspace_dir=Path(tempfile.mkdtemp()))
        import sys

        self.assertEqual(cfg.python_executable, sys.executable)


class _ProbeCitationGraph(CitationGraph):
    def __init__(self) -> None:
        super().__init__()
        self.last_url = ""

    async def _http_get(self, url: str) -> str:  # type: ignore[override]
        self.last_url = url
        return '{"data":[]}'


class TestLiteratureHardening(unittest.IsolatedAsyncioTestCase):
    async def test_frontier_query_is_url_encoded(self) -> None:
        graph = _ProbeCitationGraph()
        await graph.find_research_frontier("graph neural networks + sparse")
        self.assertIn("query=graph+neural+networks+%2B+sparse", graph.last_url)
        self.assertIn("fields=paperId", graph.last_url)


if __name__ == "__main__":
    unittest.main()
