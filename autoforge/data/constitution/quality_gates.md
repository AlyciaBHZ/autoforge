# Quality Gates

Each phase transition must pass its quality gate before proceeding.

## Gate: SPEC → BUILD

| Check | Criteria |
|-------|----------|
| Spec validity | spec.json is valid JSON with all required fields |
| Module count | At least 1 module defined |
| File coverage | Each module lists at least 1 file |
| Tech stack | Framework, language, and runtime specified |
| Naming | project_name is kebab-case, valid as directory name |

## Gate: BUILD → VERIFY

| Check | Criteria |
|-------|----------|
| Task completion | All BUILD tasks are "done" |
| File existence | All files listed in tasks actually exist |
| No blocked tasks | No tasks stuck in "blocked" state |
| Review pass | All code reviewed and approved (score >= 6) |

## Gate: VERIFY → REFACTOR

| Check | Criteria |
|-------|----------|
| Build success | Project compiles/builds without errors |
| Start success | Application starts without crashing |
| Test pass | No critical test failures |

## Gate: REFACTOR → DELIVER

| Check | Criteria |
|-------|----------|
| Quality score | Overall quality >= threshold (default 0.7) |
| Regression free | All tests still pass after refactoring |

## Enforcement Status

Each gate is enforced programmatically in `orchestrator.py`:

| Gate | Enforcement |
|------|-------------|
| SPEC → BUILD | `_phase_spec()` validates spec.json fields |
| BUILD → VERIFY | **`_enforce_build_gate()`** — independent file existence audit, all tasks DONE check |
| VERIFY → REFACTOR | `_phase_verify()` checks test results |
| REFACTOR → DELIVER | `_phase_refactor()` checks quality score threshold |

Additional pipeline hardening:
- **Smoke Check**: Before review, files are checked for existence + syntax validity
- **Anti-Spin Detection**: Builder agents that don't write files for 10+ turns are nudged; 20+ turns = forced failure
- **File Overlap Detection**: Tasks claiming the same files are serialized to prevent merge conflicts
- **TDD Loop**: When `--tdd N` is set, builder runs tests N times before review, fixing failures each iteration
- **Human Checkpoints**: When `--confirm phase1,phase2` is set, pipeline pauses after each specified phase for user review

## Agent Capabilities

| Agent | Core Tools | Search Tools | Web Tools |
|-------|-----------|-------------|-----------|
| Director | (text only) | — | `search_web`, `fetch_url` |
| Architect | `read_template` | — | `search_web`, `fetch_url` |
| Builder | `write_file`, `read_file`, `list_files`, `run_command` | `grep_search` | `fetch_url` |
| Scanner | `read_file`, `list_files`, `run_command` | `grep_search` | — |
| Tester | `run_command`, `read_file` | — | — |
| Gardener | `write_file`, `read_file`, `list_files` | `grep_search` | `fetch_url` |

## Failure Handling

1. **First failure**: Retry the phase (up to 3 times)
2. **Persistent failure**: Degrade — simplify the approach
3. **Degradation fails**: Mark as blocked, report to user
4. **Budget exceeded**: Stop immediately, deliver what's available
