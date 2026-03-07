# AutoForge Kernel Architecture

## Goal

The kernel turns AutoForge's runtime primitives into a first-class execution
substrate that is:

- durable
- inspectable
- profile-driven
- multi-surface friendly
- compatible with queue/daemon and local CLI execution

This follows a harness-oriented engineering style:

- repository-local manifests instead of hidden state
- append-only event logs for replay/debugging
- explicit profile contracts instead of mode-specific branching
- environment locks and workspace leases for reproducibility

## Kernel Surface

The kernel is modeled as five cooperating surfaces:

1. `queue / daemon`
2. `run registry / event log / lease / resume`
3. `workspace / sandbox / environment lock`
4. `trace / artifact / harness`
5. `async bridge / checkpoint / inbox`

In the current codebase these surfaces live across:

- `autoforge.engine.daemon`
- `autoforge.engine.project_registry`
- `autoforge.engine.runtime.*`
- `autoforge.engine.harness.*`
- `autoforge.engine.kernel.*`

## Profiles

Profiles define exactly three things:

1. phase graph
2. success contract
3. artifact contract

Implemented profiles:

- `development`
- `verification`
- `research`

Profile definitions live in:

- `autoforge/engine/kernel/contracts.py`
- `autoforge/engine/kernel/profiles.py`

## Landed Components

### 1. Kernel contracts

`KernelProfile` now makes phase graph, success contract, and artifact contract
explicit and serializable.

### 2. Kernel session

`KernelSession` creates a durable per-run control plane under:

`<project>/.autoforge/kernel/runs/<run_id>/`

It writes:

- `manifest.json`
- `execution_plan.json`
- `profile.json`
- `contracts.json`
- `environment.lock.json`
- `artifact_manifest.json`
- `events.jsonl`
- `inbox.json`

### 3. Workspace lease

`WorkspaceLock` prevents concurrent writers against the same workspace and
supports stale-lock recovery through TTL + heartbeat.

### 4. Runtime unification

Orchestrator runtime bootstrapping is centralized through:

- `create_runtime_from_config(...)`
- `Orchestrator._open_runtime(...)`
- `Orchestrator._close_runtime(...)`

This removes duplicated runtime wiring and makes profile/session setup uniform
across:

- generate
- review
- import
- resume

### 5. Async inbox and checkpoint visibility

Daemon message polling now mirrors inbox messages into kernel state.
Checkpoint decisions are also recorded in the kernel event log.

### 6. Contract-backed deliverables

`development` runs now generate `DEPLOY_GUIDE.md` during `deliver`.
`verification` runs now emit a structured obligation ledger.

### 7. Execution plans

Each kernel run now persists a repository-local `execution_plan.json` that
captures:

- operation and surface
- objective and summary
- profile phase graph
- success and artifact contracts
- phase state progression
- checkpoints and execution constraints

The plan is updated incrementally as phases advance, so CLI, daemon, and future
surfaces can all inspect the same control plane.

### 8. Inspector and evidence packs

The kernel now exposes reusable inspection/export primitives:

- `inspect_kernel_run(...)`
- `load_kernel_event_stream(...)`
- `render_kernel_event_stream(...)`
- `render_kernel_inspection(...)`
- `export_evidence_pack(...)`
- `export_replay_bundle(...)`

The kernel now also persists:

- `contract_verdict.json`
- repo-local `run_store.sqlite3`
- optional `harness_judge.json` overlays for harness-evaluated runs

Research runs also auto-materialize:

- `.autoforge/research/brief.md`
- `.autoforge/research/metrics.json`
- `.autoforge/research/evidence_pack/<run_id>/`

CLI entry points:

- `autoforgeai kernel inspect <workspace-or-run>`
- `autoforgeai kernel events <workspace-or-run>`
- `autoforgeai kernel evidence-pack <workspace-or-run>`
- `autoforgeai kernel replay-bundle <workspace-or-run>`

### 9. Replay and judge bundles

Kernel replay bundles now package:

- kernel control-plane files
- optional workspace artifacts
- local harness judge outputs when the workspace sits inside a harness case run
- saved trajectories under `.autoforge/trajectory_*.json`
- `summary.json`
- `judge_rubric.json`
- OpenAI-friendly `items.jsonl` + `item_schema.json`

This makes it possible to hand a completed run to external replay/judge
tooling without reconstructing the original workspace state by hand.

### 10. Run identity and lineage

Kernel runs now carry explicit identity fields:

- `run_id`
- `lineage_id`
- `parent_run_id`
- `project_id`

These fields are written into manifests, environment locks, contract verdicts,
and the repo-local `run_store.sqlite3` index. This is the basis for joining:

- daemon projects
- CLI runs
- resume runs
- harness case runs
- replay/judge exports

For daemon-backed runs, the queue still creates a planned placeholder
`run_<project_id>` record at intake time, but execution claims that placeholder
into the actual kernel `run_id` before orchestration starts. This keeps queue
durability while ensuring the registry, kernel artifacts, and harness exports
all describe the same run lineage.

### 11. Schema-Versioned Kernel Artifacts

Core repo-local kernel JSON artifacts now carry `schema_version` and
`artifact_type` fields:

- `manifest.json`
- `execution_plan.json`
- `environment.lock.json`
- `artifact_manifest.json`
- `inbox.json`
- `contract_verdict.json`

`inspect_kernel_run()` and related tooling use backward-compatible loaders, so
older runs without explicit schema metadata are normalized at read time instead
of breaking replay/inspection.

### 12. Execution Control Layer

Top-level orchestration entrypoints now dispatch through a dedicated control
layer:

- `autoforge.engine.run_controller.RunController`
- `autoforge.engine.profile_runner`
- `autoforge.engine.phase_executor`

The heavy phase implementations still live in `Orchestrator`, but the public
entry surface for `generate / review / import / resume` is now split from the
underlying phase methods. This is the first step toward making kernel-driven
execution reusable across CLI, daemon, and harness surfaces.

Phase graph nodes now also carry `handler` and `resume_markers` metadata. The
resume path mapper consults profile graph metadata first, instead of relying
only on hard-coded string heuristics inside `Orchestrator`.

### 13. Kernel Checkpoint Substrate

Runs now persist a repo-local `checkpoint.json` under the kernel run dir:

- `.autoforge/kernel/runs/<run_id>/checkpoint.json`

The checkpoint carries:

- run identity (`run_id / lineage_id / parent_run_id / project_id`)
- profile + operation
- latest state marker + state version
- the resumable state payload

`resume` now prefers this kernel checkpoint substrate first, then falls back to
the durable journal snapshot, and finally to `.forge_state.json` for backward
compatibility.

This makes kernel state the primary resume source instead of a sidecar file.

### 14. Graph-Driven Public Execution

The public `generate / review / import / resume` paths now execute through
profile-aware `PhaseExecutor` plans instead of direct monolithic pipeline
methods.

That means:

- phase ordering is derived from the profile graph
- resume starts from the successor phase implied by the saved marker
- kernel phase transitions are emitted centrally by the executor
- post-phase checkpoints are driven from the control layer instead of being
  scattered through public entrypoints

`Orchestrator` still owns most heavy implementation details, but sequencing is
no longer the responsibility of the monolithic public pipeline methods.

### 15. Runner-Owned Phase Implementations

The public execution path now calls phase implementations owned by
`autoforge.engine.profile_runner` instead of the older `_executor_*` helper
methods on `Orchestrator`.

That means:

- `development` phase bodies now live under `DevelopmentProfileRunner`
- `verification` phase bodies now live under `VerificationProfileRunner`
- `research` now runs on the same `PhaseExecutor` substrate through
  `ResearchProfileRunner`
- `resume` routes back into the same profile-aware phase plans, including
  research checkpoints

`Orchestrator` still retains compatibility helpers, but the active control path
for CLI/daemon/harness-facing runs is now runner-owned.

### 16. Daemon Recovery Proof

Daemon recovery is now covered by an end-to-end regression that exercises:

- stale BUILDING project requeue
- workspace lease reacquisition
- kernel checkpoint detection
- daemon resume path selection
- registry/run-record completion after resumed execution

This closes the previous gap where crash recovery behavior existed in code but
was only covered by targeted unit-style regression.

### 17. UI Harness Overlay

Frontend-capable development runs can now enable a `ui_harness` overlay through
config or CLI flags.

The overlay currently contributes:

- `design_context_refs` input capture
- repo-local UI artifacts such as `design_brief.json`, `style_guide.json`,
  `design_tokens.json`, `component_inventory.json`, `ui_judge_report.json`,
  and `ui_handoff.md`
- UI-oriented outcomes for design consistency, responsive layout, and
  accessibility contract heuristics

This keeps UI intent and UI verification inside the same repo-local control
plane as the rest of the kernel run instead of leaving visual quality as an
untracked side effect.

## Multi-Surface Model

Kernel sessions are surface-aware through `client_surface`:

- `cli`
- `daemon`
- future surfaces such as webhook, desktop, mobile, or remote workers

The surface is recorded in thread metadata and environment locks, so replay and
audit tools can reason about where the run came from.

## Recommended Next Integrations

These are the most valuable next upgrades on top of the landed kernel:

### Event-stream consumers

Expose `events.jsonl` through a thin UI/server adapter so local CLI, daemon,
webhook, and future mobile surfaces can all render the same run state.

### Replay and judge bundles

Add richer coupling between kernel replay bundles and harness benchmark runs so
judge outcomes, hidden-test results, and kernel traces can be compared under a
single run lineage.

### Unified watch visibility

`autoforgeai watch --tail` now tails both:

- `.forge_task_transition_log.jsonl`
- active kernel `events.jsonl`

This keeps daemon watch mode aligned with the same kernel event stream that the
direct `kernel events` command consumes.

### Microagent or skill overlays

Add profile-scoped reusable behaviors that can be injected without changing the
core orchestrator:

- research source collection
- verification obligation extraction
- deployment hardening
- UI harness overlays for Storybook generation, screenshot baselines, stronger design judges, and token sync

### Remote worker surfaces

Add first-class remote worker and mobile/webhook surfaces on top of the same
kernel contracts so queue, inspection, and resume semantics stay uniform across
clients.

## External Signals Worth Tracking

The following ecosystems are worth borrowing from:

- OpenAI app-server and execution-plan style repository-local artifacts
- OpenHands event-stream runtime and microagent overlays
- SWE-agent trajectory and benchmark replay discipline
- GPT Researcher evidence-first reporting and source traceability
- PaperBench style rubric-backed verification of research outcomes
- Figma/v0/Builder style design-context ingestion and UI review workflows

## Non-Goals

The kernel does not replace the orchestrator. It standardizes runtime
contracts underneath it.

The kernel also does not force a single pipeline. Profiles remain free to map
their contracts onto different internal execution strategies.
