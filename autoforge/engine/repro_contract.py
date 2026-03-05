"""Strict build contract helpers for paper reproduction runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA_VERSION = "paper-repro-contract-v1"
REQUIRED_ARTIFACT_FILES: tuple[str, ...] = (
    "candidate.json",
    "paper_signals.json",
    "verification_plan.json",
    "environment_spec.json",
    "run_manifest.json",
    "repro_report.json",
    "repro_report.md",
)
P0_P4_KEYS: tuple[str, ...] = (
    "P0_api_key_runtime",
    "P1_goal_to_paper_retrieval",
    "P2_paper_signal_extraction",
    "P3_closed_loop_verification",
    "P4_environment_reproducibility",
)


@dataclass
class ContractValidationResult:
    """Validation result for one reproduction output directory."""

    ok: bool
    errors: list[str]
    missing_artifacts: list[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def contract_schema_path() -> Path:
    """Return the absolute path to the contract schema file."""
    return Path(__file__).resolve().parents[1] / "contracts" / "paper_repro_contract.schema.json"


def load_contract_schema() -> dict[str, Any]:
    """Load paper repro contract schema."""
    schema_path = contract_schema_path()
    try:
        return json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to load contract schema: {schema_path} ({exc})") from exc


def derive_contract_status(
    *,
    inference_score: float,
    datasets: list[str],
    metrics: list[str],
    claimed_metrics: list[dict[str, str]],
    hardware_hints: list[str],
    run_generate: bool,
    has_api_key: bool,
) -> dict[str, str]:
    """Derive baseline P0-P4 status map for contract reporting."""
    if run_generate and has_api_key:
        p0 = "ok"
    elif run_generate:
        p0 = "resolved_with_simulation"
    else:
        p0 = "not_requested"

    return {
        "P0_api_key_runtime": p0,
        "P1_goal_to_paper_retrieval": "ok" if inference_score >= 12.0 else "risky",
        "P2_paper_signal_extraction": "ok" if (datasets and metrics) else "needs_manual_fill",
        "P3_closed_loop_verification": "ok" if claimed_metrics else "insufficient_claimed_numbers",
        "P4_environment_reproducibility": "ok" if hardware_hints else "hardware_unknown",
    }


def build_run_manifest(
    *,
    run_id: str,
    goal: str,
    mode: str,
    profile: str,
    strict_contract: bool,
    output_dir: Path,
    paper: dict[str, Any],
    artifacts_written: list[str],
) -> dict[str, Any]:
    """Build run_manifest.json payload."""
    artifacts_unique = sorted(set(artifacts_written))
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "run_id": run_id,
        "goal": goal,
        "mode": mode,
        "profile": profile,
        "strict_contract": strict_contract,
        "output_dir": str(output_dir),
        "paper": paper,
        "required_artifacts": list(REQUIRED_ARTIFACT_FILES),
        "artifacts_written": artifacts_unique,
        "stop_conditions": [
            "Missing required artifact file",
            "Schema validation failed for repro_report.json",
        ],
        "generated_at": _utc_now_iso(),
    }


def build_repro_report(
    *,
    run_id: str,
    paper_id: str,
    goal: str,
    mode: str,
    profile: str,
    output_dir: Path,
    strict_contract: bool,
    p0_p4_status: dict[str, str],
    artifacts_complete: bool,
    failure_reasons: list[str],
) -> dict[str, Any]:
    """Build repro_report.json payload."""
    dedup_failures: list[str] = []
    seen: set[str] = set()
    for reason in failure_reasons:
        text = reason.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        dedup_failures.append(text)

    # Contract semantics: only full generated runs can pass reproducibility.
    if mode == "simulated_no_api_key":
        simulation_reason = (
            "Simulated run without API key cannot satisfy full closed-loop reproduction."
        )
        if simulation_reason not in seen:
            dedup_failures.append(simulation_reason)
            seen.add(simulation_reason)
    elif mode == "generation_failed":
        generation_reason = "Generation failed; reproduction run incomplete."
        if generation_reason not in seen:
            dedup_failures.append(generation_reason)
            seen.add(generation_reason)

    mode_can_pass = mode == "generated_with_api_key"
    pass_fail = "pass" if (artifacts_complete and not dedup_failures and mode_can_pass) else "fail"
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "run_id": run_id,
        "paper_id": paper_id,
        "goal": goal,
        "mode": mode,
        "profile": profile,
        "artifacts_complete": artifacts_complete,
        "pass_fail": pass_fail,
        "p0_p4_status": p0_p4_status,
        "failure_reasons": dedup_failures,
        "strict_contract": strict_contract,
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "run_manifest.json"),
        "report_path": str(output_dir / "repro_report.json"),
        "generated_at": _utc_now_iso(),
    }


def render_repro_report_markdown(
    *,
    report: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    """Render a concise markdown report for human review."""
    lines: list[str] = [
        "# Reproduction Contract Report",
        "",
        "## Summary",
        f"- Run ID: {report.get('run_id', '')}",
        f"- Mode: {report.get('mode', '')}",
        f"- Profile: {report.get('profile', '')}",
        f"- Strict Contract: {report.get('strict_contract', False)}",
        f"- Artifacts Complete: {report.get('artifacts_complete', False)}",
        f"- Result: {report.get('pass_fail', 'fail')}",
        "",
        "## Input",
        f"- Goal: {report.get('goal', '')}",
        f"- Paper ID: {report.get('paper_id', '')}",
        "",
        "## Artifact Checklist",
    ]
    written = set(manifest.get("artifacts_written", []))
    for name in REQUIRED_ARTIFACT_FILES:
        mark = "x" if name in written else " "
        lines.append(f"- [{mark}] {name}")

    lines.extend(
        [
            "",
            "## P0-P4 Status",
        ]
    )
    status = report.get("p0_p4_status", {})
    for key in P0_P4_KEYS:
        lines.append(f"- {key}: {status.get(key, 'unknown')}")

    failures = report.get("failure_reasons", [])
    lines.extend(
        [
            "",
            "## Failure Reasons",
        ]
    )
    if failures:
        for reason in failures:
            lines.append(f"- {reason}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Stop Conditions",
        ]
    )
    for stop_condition in manifest.get("stop_conditions", []):
        lines.append(f"- {stop_condition}")

    return "\n".join(lines) + "\n"


def write_contract_outputs(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> None:
    """Write run_manifest.json, repro_report.json, and repro_report.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "repro_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "repro_report.md").write_text(
        render_repro_report_markdown(report=report, manifest=manifest),
        encoding="utf-8",
    )


def _type_matches(expected: str, value: Any) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _validate_json_schema(
    *,
    schema: dict[str, Any],
    value: Any,
    path: str,
    errors: list[str],
) -> None:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if not any(_type_matches(t, value) for t in schema_type):
            errors.append(f"{path}: expected one of {schema_type}, got {type(value).__name__}")
            return
    elif isinstance(schema_type, str):
        if not _type_matches(schema_type, value):
            errors.append(f"{path}: expected {schema_type}, got {type(value).__name__}")
            return

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path}: value {value!r} not in enum {enum_values}")

    if isinstance(value, str):
        min_len = schema.get("minLength")
        if isinstance(min_len, int) and len(value) < min_len:
            errors.append(f"{path}: string shorter than minLength={min_len}")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{path}: array shorter than minItems={min_items}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                _validate_json_schema(
                    schema=item_schema,
                    value=item,
                    path=f"{path}[{idx}]",
                    errors=errors,
                )

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")

        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in properties:
                prop_schema = properties[key]
                if isinstance(prop_schema, dict):
                    _validate_json_schema(
                        schema=prop_schema,
                        value=child,
                        path=child_path,
                        errors=errors,
                    )
            else:
                if additional is False:
                    errors.append(f"{path}: unexpected key {key!r}")
                elif isinstance(additional, dict):
                    _validate_json_schema(
                        schema=additional,
                        value=child,
                        path=child_path,
                        errors=errors,
                    )


def validate_report_schema(report_payload: dict[str, Any]) -> list[str]:
    """Validate repro_report.json against contract schema."""
    schema = load_contract_schema()
    errors: list[str] = []
    _validate_json_schema(schema=schema, value=report_payload, path="report", errors=errors)
    return errors


def validate_contract_artifacts(output_dir: Path) -> ContractValidationResult:
    """Validate required artifacts and report schema for one run directory."""
    errors: list[str] = []

    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (output_dir / name).exists()]
    for name in missing:
        errors.append(f"Missing required artifact: {name}")

    manifest_payload: dict[str, Any] | None = None
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"run_manifest.json is invalid JSON: {exc}")
    else:
        errors.append("run_manifest.json missing")

    report_path = output_dir / "repro_report.json"
    report_payload: dict[str, Any] | None = None
    if report_path.exists():
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"repro_report.json is invalid JSON: {exc}")
    else:
        errors.append("repro_report.json missing")

    if report_payload is not None:
        errors.extend(validate_report_schema(report_payload))

    if manifest_payload is not None:
        required = manifest_payload.get("required_artifacts", [])
        if sorted(required) != sorted(REQUIRED_ARTIFACT_FILES):
            errors.append("run_manifest.json required_artifacts does not match contract list")

    if report_payload is not None:
        reported_complete = bool(report_payload.get("artifacts_complete", False))
        actual_complete = len(missing) == 0
        if reported_complete != actual_complete:
            errors.append(
                "repro_report.json artifacts_complete mismatch "
                f"(reported={reported_complete}, actual={actual_complete})"
            )

        pass_fail = report_payload.get("pass_fail")
        mode = report_payload.get("mode")
        if pass_fail == "pass":
            if mode != "generated_with_api_key":
                errors.append(
                    "repro_report.json pass_fail=pass requires mode=generated_with_api_key"
                )
            reasons = report_payload.get("failure_reasons", [])
            if isinstance(reasons, list) and reasons:
                errors.append(
                    "repro_report.json pass_fail=pass requires empty failure_reasons"
                )

    return ContractValidationResult(
        ok=(len(errors) == 0),
        errors=errors,
        missing_artifacts=missing,
    )
