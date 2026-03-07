# Harness Refactor TODO

Status date: 2026-03-07

## Completed

- [x] Establish kernel profiles with explicit phase/success/artifact contracts
- [x] Persist repo-local kernel manifests, environment locks, plans, evidence packs
- [x] Add kernel inspection, event streaming, replay/judge bundle export
- [x] Add kernel-aware CLI commands and daemon watch tailing
- [x] Introduce a repo-local `RunStore` as the kernel control-plane index
- [x] Add `lineage_id / parent_run_id / project_id` identity to all kernel runs
- [x] Add `contract_verdict.json` and make close-out validate contracts
- [x] Feed harness judge outputs back into kernel artifacts and verdicts
- [x] Make daemon/project registry consume kernel run lineage instead of parallel semantics
- [x] Add schema versioning and backward-compatible reads for core kernel artifacts
- [x] Introduce `RunController + ProfileRunner + PhaseExecutor` as the top-level execution control layer
- [x] Attach `handler / resume_markers` metadata to phase graph nodes and use graph-driven resume phase mapping
- [x] Add repo-local kernel checkpoints and make resume prefer kernel checkpoint substrate
- [x] Turn phase graphs into executable public run paths via `PhaseExecutor`
- [x] Move public `generate / review / import / resume` sequencing onto profile-aware phase execution
- [x] Add structured verification judge artifacts and counterexample exports
- [x] Expand `doctor / healthcheck` with kernel/runtime readiness checks
- [x] Add regression coverage for kernel checkpoints and phase-executor resume sequencing

## Next

- [x] Continue shrinking `Orchestrator` by moving step helper bodies into dedicated runner modules
- [x] Add real daemon crash/resume/lease-recovery end-to-end tests (not only targeted regression)
- [x] Add research-profile execution on top of the same phase-executor substrate
- [ ] Remove legacy executor helper duplication from `Orchestrator` once compatibility shims are no longer needed
- [ ] Expand daemon recovery coverage to multi-run lineage takeover and concurrent worker contention

## UI Harness Next

- [x] Add a `ui_harness` overlay on top of `development` instead of a new top-level profile
- [x] Add `design_context_refs` ingest for design links, screenshot bundles, and brand assets
- [x] Emit repo-local UI artifacts: `design_brief.json`, `style_guide.json`, `design_tokens.json`, `component_inventory.json`, `ui_judge_report.json`, `ui_handoff.md`
- [ ] Add Storybook/visual-baseline generation for frontend-capable outputs
- [ ] Add stronger UI verification contracts backed by Playwright/axe instead of heuristic-only outcomes
- [ ] Add optional token sync pipeline: Figma Variables / Tokens Studio -> Style Dictionary -> code outputs

## Development Harness Next

- [x] Add an `agent_harness` control plane with per-agent run manifests, tool-call traces, and schema-backed agent verdicts
- [x] Add an `llm_harness` layer with provider fallback receipts, prompt/response bundles, and token-cost ledgers
- [x] Turn `TaskDAG` into a schema-versioned `task_harness` artifact set with task contracts, attempt logs, and task verdicts
- [x] Add an `execution_harness` across sandbox/git/locks with command receipts, worktree manifests, lease artifacts, and merge verdicts
- [x] Add a `delivery_harness` for deploy output with `deploy_manifest.json`, environment requirements, and publish verdicts
- [x] Add a `daemon_harness` compare/recovery view for multi-run lineage takeover, worker contention, and queue-to-kernel traceability

## Development Harness Next Up

- [ ] Register development-harness artifacts into kernel manifests for cross-surface inspection
- [ ] Add CLI inspection/rendering for agent/llm/task/execution/delivery/daemon harness outputs
- [ ] Add compare views across runs for agent verdict drift, budget drift, and task-plan drift
- [ ] Add stronger UI verification backends: Storybook, Playwright baselines, and real `axe` artifacts

Reference design note:

- `docs/UI_HARNESS_INTEGRATION.md`
- `docs/DEVELOPMENT_HARNESS_REVIEW.md`
