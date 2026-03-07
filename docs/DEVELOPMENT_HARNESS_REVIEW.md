# Development Harness Review

Status date: 2026-03-07

## Update

The module-level harness work described in this review has now been landed in
code for:

- `agent_harness`
- `llm_harness`
- `task_harness`
- `execution_harness`
- `delivery_harness`
- `daemon_harness`

The rest of this document remains useful as the rationale for those changes and
as a guide for the next refinement layer.

## Conclusion

AutoForge is no longer only a monolithic pipeline.

The `kernel`, `phase_executor`, and `ui_harness` layers now provide a real
control plane for run identity, contracts, checkpoints, research evidence, and
UI-facing overlays.

What is still missing is harness coverage for the supporting development
modules. Today many of those modules are reliable components, but not yet
first-class harness surfaces.

That means they can execute work, but they do not yet all provide:

- a schema-versioned artifact surface
- explicit success or failure contracts
- replay/export friendly traces
- judgeable outputs that can be compared across runs

## Current State

### Already largely harnessed

- `kernel/*`
  - run identity, lineage, artifact contracts, verdicts, checkpoints, replay
- `phase_executor.py` and `profile_runner.py`
  - profile-aware sequencing, phase-local outputs, research substrate
- `ui_harness.py`
  - design context refs, style artifacts, UI judge outputs, handoff files
- `daemon.py` plus `project_registry.py`
  - queue-to-run lineage alignment, recovery proof, registry event flow

### Recently harnessed in this refactor

- `agent_base.py`
- `llm_router.py`
- `task_dag.py`
- `sandbox.py`
- `git_manager.py`
- `lock_manager.py`
- `deploy_guide.py`
- parts of `profile_runner.py` that still emit ad hoc phase files

## Module Review

### 1. Agent layer: `agent_base.py`

Current strength:

- structured `AgentResult`
- tool-use loop with schema repair
- anti-spin detection and mid-task checkpoints

Current gap:

- artifacts are an untyped in-memory dict
- tool calls are not emitted as a repo-local typed trace artifact
- checkpoint feedback is local to the loop, not exported as a kernel-joinable
  agent trace
- there is no agent-level success contract beyond "did the loop finish"

What harnessing should add:

- `agent_run.json`
- `tool_call_log.jsonl`
- `agent_checkpoint_log.json`
- `agent_verdict.json`
- explicit output contract per agent role

Why this matters:

Without this, you cannot compare builder or reviewer behavior across runs the
same way you can compare kernel or harness runs.

### 2. LLM layer: `llm_router.py`

Current strength:

- provider routing
- budget reservation
- rate limiting
- telemetry hooks
- model fallback handling

Current gap:

- telemetry is best-effort and sink-driven, not a guaranteed repo-local harness
  artifact
- prompt/model fallback decisions are not preserved as a typed replay bundle
- there is no call-level verdict artifact for budget exhaustion, fallback use,
  schema repair success, or provider degradation

What harnessing should add:

- `llm_call_manifest.jsonl`
- `llm_budget_ledger.json`
- `provider_fallback_receipts.json`
- `llm_replay_bundle/`
- `llm_verdict.json`

Why this matters:

The LLM layer is where cost, determinism, and provider drift happen. If it is
not harnessed, benchmark and production regressions are hard to explain.

### 3. Planning layer: `task_dag.py`

Current strength:

- clean task model
- cycle validation
- save/load for resume

Current gap:

- persisted DAG is just a raw list, not a versioned artifact type
- task contracts are shallow: `acceptance_criteria` and `exports` are strings
- there is no attempt ledger, no claim history, and no task-level verdict set

What harnessing should add:

- `task_graph.json`
- `task_contracts.json`
- `task_attempts.jsonl`
- `task_claims.jsonl`
- `task_verdicts.json`

Why this matters:

Task decomposition is one of the core places where AutoForge should be judged.
Right now task planning is serializable, but not benchmark-grade.

### 4. Execution layer: `sandbox.py`

Current strength:

- multiple backends
- capability allowlists
- telemetry hooks
- execution platform awareness

Current gap:

- command traces are not guaranteed repo-local artifacts
- execution environment, capability, and allowlist decisions are not exported
  as typed receipts
- there is no execution verdict artifact explaining whether a failure was a
  code issue, environment issue, timeout, or policy block

What harnessing should add:

- `command_receipts.jsonl`
- `execution_environment.json`
- `sandbox_policy.json`
- `execution_failures.json`
- `runtime_dependency_manifest.json`

Why this matters:

A harness-grade system needs to separate "the code is bad" from "the runtime
surface is constrained or misconfigured".

### 5. Worktree and merge layer: `git_manager.py`

Current strength:

- isolated worktrees
- commit and merge helpers

Current gap:

- worktree lifecycle is invisible to the kernel artifact plane
- no manifest ties task IDs, branches, commits, and merge outcomes together
- merge conflicts are runtime exceptions, not typed verdict artifacts

What harnessing should add:

- `worktree_manifest.json`
- `commit_receipts.jsonl`
- `merge_verdict.json`
- `branch_lineage.json`

Why this matters:

Parallel builder quality is not only about generated code. It is also about
merge reliability and traceable ownership of changes.

### 6. Lease and lock layer: `lock_manager.py`

Current strength:

- cross-platform atomic claims
- stale-lock cleanup

Current gap:

- lock state only lives in the filesystem and ephemeral cache
- there is no exported artifact for lease history, stale-lock takeover, or
  ownership transfer

What harnessing should add:

- `task_lock_events.jsonl`
- `lease_manifest.json`
- `takeover_verdict.json`

Why this matters:

If concurrency bugs appear, you need a replayable lock story, not only a best
effort local lock file.

### 7. Delivery layer: `deploy_guide.py`

Current strength:

- lightweight framework detection
- generated deploy guidance

Current gap:

- deploy output is markdown only
- there is no structured provider contract, environment requirement manifest,
  or publishability verdict
- delivery cannot currently be compared across runs except by diffing prose

What harnessing should add:

- `deploy_manifest.json`
- `environment_requirements.json`
- `publish_targets.json`
- `delivery_verdict.json`

Why this matters:

Delivery is the last phase users actually consume. A harnessed system should be
able to say whether a project is deploy-ready, not just print a guide.

### 8. Phase outputs still needing cleanup: `profile_runner.py`

Current strength:

- more public phase logic now lives in runners
- research outputs are structured and kernel-registered
- UI artifacts are now runner-owned

Current gap:

- several files are still written ad hoc, for example `spec.json`, research
  notes, and markdown outputs
- not every phase file has a schema version, artifact type, or explicit
  contract registration path

What harnessing should add:

- make every phase output either:
  - a schema-versioned JSON artifact, or
  - a typed text artifact with manifest metadata
- remove remaining write-path drift between runtime artifacts and kernel
  artifacts

## Priority Order

This is the order I would implement next.

1. `agent_harness`
2. `llm_harness`
3. `task_harness`
4. `execution_harness`
5. `delivery_harness`
6. `daemon_harness` compare and recovery view

## Short Form

The project is no longer blocked only on kernel work.

The next maturity jump is to harness the development support layers so AutoForge
can explain, replay, score, and compare:

- what each agent did
- what each model call cost and returned
- how tasks were planned and retried
- how commands were executed
- how worktrees merged
- whether delivery is actually publish-ready

That is the path from "kernel harnessed" to "whole development stack
harnessed".
