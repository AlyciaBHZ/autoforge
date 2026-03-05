"""Unit and integration tests for paper reproduction contract v1."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoforge.engine.config import ForgeConfig
from autoforge.engine.paper_repro import InferenceResult, PaperRecord
from autoforge.engine.repro_contract import (
    ContractValidationResult,
    REQUIRED_ARTIFACT_FILES,
    build_repro_report,
    build_run_manifest,
    validate_contract_artifacts,
    validate_report_schema,
    write_contract_outputs,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_core_artifacts(base: Path) -> None:
    _write_json(base / "candidate.json", {"paper": {"note_id": "abc123"}, "goal": "test"})
    _write_json(base / "paper_signals.json", {"datasets": ["mmlu"], "metrics": ["accuracy"]})
    _write_json(base / "verification_plan.json", {"checklist": ["reproduce metric"]})
    _write_json(base / "environment_spec.json", {"profile": "theory-first"})


def _p0_p4_ok() -> dict[str, str]:
    return {
        "P0_api_key_runtime": "resolved_with_simulation",
        "P1_goal_to_paper_retrieval": "ok",
        "P2_paper_signal_extraction": "ok",
        "P3_closed_loop_verification": "ok",
        "P4_environment_reproducibility": "ok",
    }


class TestReproContractPassCases(unittest.TestCase):
    def test_validate_contract_ok_simulated_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _write_core_artifacts(out_dir)
            manifest = build_run_manifest(
                run_id="run-1",
                goal="test goal",
                mode="simulated_no_api_key",
                profile="theory-first",
                strict_contract=True,
                output_dir=out_dir,
                paper={"note_id": "abc123", "title": "Test Paper", "year": 2025},
                artifacts_written=list(REQUIRED_ARTIFACT_FILES),
            )
            report = build_repro_report(
                run_id="run-1",
                paper_id="abc123",
                goal="test goal",
                mode="simulated_no_api_key",
                profile="theory-first",
                output_dir=out_dir,
                strict_contract=True,
                p0_p4_status=_p0_p4_ok(),
                artifacts_complete=True,
                failure_reasons=[],
            )
            write_contract_outputs(output_dir=out_dir, manifest=manifest, report=report)
            result = validate_contract_artifacts(out_dir)
            self.assertTrue(result.ok, result.errors)

    def test_validate_contract_ok_artifact_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _write_core_artifacts(out_dir)
            manifest = build_run_manifest(
                run_id="run-2",
                goal="artifact only",
                mode="artifact_only",
                profile="theory-first",
                strict_contract=False,
                output_dir=out_dir,
                paper={"note_id": "abc123", "title": "Test Paper", "year": 2025},
                artifacts_written=list(REQUIRED_ARTIFACT_FILES),
            )
            report = build_repro_report(
                run_id="run-2",
                paper_id="abc123",
                goal="artifact only",
                mode="artifact_only",
                profile="theory-first",
                output_dir=out_dir,
                strict_contract=False,
                p0_p4_status=_p0_p4_ok(),
                artifacts_complete=True,
                failure_reasons=[],
            )
            write_contract_outputs(output_dir=out_dir, manifest=manifest, report=report)
            self.assertEqual(validate_report_schema(report), [])
            self.assertTrue(validate_contract_artifacts(out_dir).ok)

    def test_validate_contract_ok_generation_failed_mode_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            report = build_repro_report(
                run_id="run-3",
                paper_id="abc123",
                goal="failed run",
                mode="generation_failed",
                profile="theory-first",
                output_dir=out_dir,
                strict_contract=True,
                p0_p4_status=_p0_p4_ok(),
                artifacts_complete=False,
                failure_reasons=["generation failed: timeout"],
            )
            self.assertEqual(validate_report_schema(report), [])


class TestReproContractFailCases(unittest.TestCase):
    def test_validate_contract_fails_on_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _write_core_artifacts(out_dir)
            (out_dir / "candidate.json").unlink()
            manifest = build_run_manifest(
                run_id="run-4",
                goal="missing artifact",
                mode="artifact_only",
                profile="theory-first",
                strict_contract=True,
                output_dir=out_dir,
                paper={"note_id": "abc123", "title": "Test Paper", "year": 2025},
                artifacts_written=list(REQUIRED_ARTIFACT_FILES),
            )
            report = build_repro_report(
                run_id="run-4",
                paper_id="abc123",
                goal="missing artifact",
                mode="artifact_only",
                profile="theory-first",
                output_dir=out_dir,
                strict_contract=True,
                p0_p4_status=_p0_p4_ok(),
                artifacts_complete=False,
                failure_reasons=["candidate missing"],
            )
            write_contract_outputs(output_dir=out_dir, manifest=manifest, report=report)
            result = validate_contract_artifacts(out_dir)
            self.assertFalse(result.ok)
            self.assertIn("candidate.json", result.missing_artifacts)

    def test_validate_contract_fails_on_report_schema_missing_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _write_core_artifacts(out_dir)
            manifest = build_run_manifest(
                run_id="run-5",
                goal="schema missing key",
                mode="artifact_only",
                profile="theory-first",
                strict_contract=True,
                output_dir=out_dir,
                paper={"note_id": "abc123", "title": "Test Paper", "year": 2025},
                artifacts_written=list(REQUIRED_ARTIFACT_FILES),
            )
            report = build_repro_report(
                run_id="run-5",
                paper_id="abc123",
                goal="schema missing key",
                mode="artifact_only",
                profile="theory-first",
                output_dir=out_dir,
                strict_contract=True,
                p0_p4_status=_p0_p4_ok(),
                artifacts_complete=True,
                failure_reasons=[],
            )
            report.pop("pass_fail", None)
            write_contract_outputs(output_dir=out_dir, manifest=manifest, report=report)
            result = validate_contract_artifacts(out_dir)
            self.assertFalse(result.ok)
            self.assertTrue(any("pass_fail" in err for err in result.errors))

    def test_validate_contract_fails_on_invalid_mode_enum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _write_core_artifacts(out_dir)
            manifest = build_run_manifest(
                run_id="run-6",
                goal="invalid mode",
                mode="artifact_only",
                profile="theory-first",
                strict_contract=True,
                output_dir=out_dir,
                paper={"note_id": "abc123", "title": "Test Paper", "year": 2025},
                artifacts_written=list(REQUIRED_ARTIFACT_FILES),
            )
            report = build_repro_report(
                run_id="run-6",
                paper_id="abc123",
                goal="invalid mode",
                mode="artifact_only",
                profile="theory-first",
                output_dir=out_dir,
                strict_contract=True,
                p0_p4_status=_p0_p4_ok(),
                artifacts_complete=True,
                failure_reasons=[],
            )
            report["mode"] = "invalid_mode"
            write_contract_outputs(output_dir=out_dir, manifest=manifest, report=report)
            result = validate_contract_artifacts(out_dir)
            self.assertFalse(result.ok)
            self.assertTrue(any("enum" in err and "mode" in err for err in result.errors))


class TestReproContractIntegration(unittest.TestCase):
    def test_reproduce_strict_contract_no_key_generates_all_artifacts(self) -> None:
        from autoforge.cli.app import _run_paper_reproduce

        paper = PaperRecord(
            note_id="abc123",
            title="Sparse Long Context Reasoning",
            abstract="We improve long-context sparse reasoning accuracy on MMLU by 3.0%.",
            keywords=["sparse", "reasoning", "mmlu"],
            year=2025,
            openreview_url="https://openreview.net/forum?id=abc123",
            pdf_url="https://openreview.net/pdf?id=abc123",
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            config = ForgeConfig(api_keys={})
            args = argparse.Namespace(
                goal="improve long-context sparse reasoning",
                year=2025,
                top_k=3,
                pick=1,
                corpus_size=100,
                cache_hours=1,
                refresh_corpus=False,
                with_pdf=False,
                output_dir=str(out_dir),
                run_generate=True,
                strict_contract=True,
            )

            with patch("autoforge.engine.paper_repro.fetch_iclr_papers", return_value=[paper]):
                with patch(
                    "autoforge.engine.paper_repro.infer_papers_from_goal",
                    return_value=[
                        InferenceResult(
                            paper=paper,
                            score=18.0,
                            matched_terms=["sparse", "reasoning"],
                        )
                    ],
                ):
                    exit_code = asyncio.run(_run_paper_reproduce(config, args))

            self.assertEqual(exit_code, 0)
            for name in REQUIRED_ARTIFACT_FILES:
                self.assertTrue((out_dir / name).exists(), f"missing {name}")
            self.assertTrue((out_dir / "simulated_pipeline_feedback.json").exists())
            self.assertTrue(validate_contract_artifacts(out_dir).ok)

    def test_reproduce_strict_contract_stops_with_exit_2_on_validation_error(self) -> None:
        from autoforge.cli.app import _run_paper_reproduce

        paper = PaperRecord(
            note_id="abc123",
            title="Sparse Long Context Reasoning",
            abstract="We improve long-context sparse reasoning accuracy on MMLU by 3.0%.",
            keywords=["sparse", "reasoning", "mmlu"],
            year=2025,
            openreview_url="https://openreview.net/forum?id=abc123",
            pdf_url="https://openreview.net/pdf?id=abc123",
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            config = ForgeConfig(api_keys={})
            args = argparse.Namespace(
                goal="improve long-context sparse reasoning",
                year=2025,
                top_k=3,
                pick=1,
                corpus_size=100,
                cache_hours=1,
                refresh_corpus=False,
                with_pdf=False,
                output_dir=str(out_dir),
                run_generate=False,
                strict_contract=True,
            )

            with patch("autoforge.engine.paper_repro.fetch_iclr_papers", return_value=[paper]):
                with patch(
                    "autoforge.engine.paper_repro.infer_papers_from_goal",
                    return_value=[
                        InferenceResult(
                            paper=paper,
                            score=18.0,
                            matched_terms=["sparse", "reasoning"],
                        )
                    ],
                ):
                    with patch(
                        "autoforge.engine.repro_contract.validate_contract_artifacts",
                        return_value=ContractValidationResult(
                            ok=False,
                            errors=["Missing required artifact: candidate.json"],
                            missing_artifacts=["candidate.json"],
                        ),
                    ):
                        exit_code = asyncio.run(_run_paper_reproduce(config, args))

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
