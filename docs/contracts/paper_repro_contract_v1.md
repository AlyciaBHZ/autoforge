# Paper Reproduction Contract v1

This contract defines the minimum overnight-sized acceptance bar for `forgeai paper reproduce`.

## Scope

- Validate and publish a strict artifact contract for each reproduction run.
- Produce machine-readable and human-readable reports.
- Enforce stop conditions under strict mode.

Out of scope:

- New model training strategies.
- New heavyweight dependencies.
- Infrastructure provisioning.

## Required Deliverables

Every run must produce these files inside the run output directory:

1. `candidate.json`
2. `paper_signals.json`
3. `verification_plan.json`
4. `environment_spec.json`
5. `run_manifest.json`
6. `repro_report.json`
7. `repro_report.md`

## CLI Contract

`forgeai paper reproduce ... --strict-contract`

- When enabled, missing required artifacts or schema violations are fatal.
- Fatal contract errors exit with code `2`.

## Report Contract

`repro_report.json` must conform to:

- `autoforge/contracts/paper_repro_contract.schema.json`

Core required fields:

- `run_id`, `paper_id`, `goal`
- `mode`, `profile`
- `artifacts_complete`, `pass_fail`
- `p0_p4_status`
- `failure_reasons`
- `strict_contract`
- `output_dir`, `manifest_path`, `report_path`
- `generated_at`

## Stop Conditions

Strict mode must stop immediately when either condition is true:

1. Any required file is missing.
2. `repro_report.json` fails schema validation.

## Test Expectations

Minimum validation before merge:

1. Contract validator has pass and fail unit tests.
2. No-API-key simulated path writes all required artifacts.
3. Existing smoke suite remains green.
