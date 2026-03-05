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
| Build contract | `build_contract` present with `deliverables`, `test_requirements`, `reports`, `stop_conditions` |
| Scope limit | Module count ≤ `stop_conditions.max_modules` (default 8, hard cap 10) |
| Task cap | `stop_conditions.max_tasks` ≤ 20 |
| Justification | `scope_justification` is a non-empty string |

## Gate: BUILD → VERIFY

| Check | Criteria |
|-------|----------|
| Task completion | All BUILD tasks are "done" |
| Task count | Task count ≤ `build_contract.stop_conditions.max_tasks` |
| File existence | All files listed in tasks actually exist |
| File count | Total source files ≤ `build_contract.stop_conditions.max_source_files` |
| No blocked tasks | No tasks stuck in "blocked" state |
| Review pass | All code reviewed and approved (score >= 6) |
| Deliverables | All items in `build_contract.deliverables` are present |

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
| Director | (no file tools) | — | `search_web`, `fetch_url`, `search_github`, `inspect_repo` |
| Architect | `read_template` | — | `search_web`, `fetch_url` |
| Builder | `write_file`, `read_file`, `list_files`, `run_command` | `grep_search` | `fetch_url` |
| Reviewer | `read_file`, `list_files`, `run_check` | `grep_search` | — |
| Scanner | `read_file`, `list_files`, `run_command` | `grep_search` | — |
| Tester | `run_command`, `read_file` | — | — |
| Gardener | `write_file`, `read_file`, `list_files` | `grep_search` | `fetch_url` |

## Failure Handling

1. **First failure**: Retry the phase (up to `max_build_resets` times, default 3)
2. **Persistent failure**: Mark individual task as BLOCKED after failing every reset cycle (fail-fast)
3. **All tasks blocked**: Stop build, deliver what's available
4. **Budget exceeded**: Stop immediately, deliver what's available

## Error Recovery & Feedback Flow

### BUILD Phase Recovery
- **Builder fails to write files**: Anti-spin detection nudges after 10 turns, forces failure after 20 turns
- **Reviewer rejects (score < 7)**: Builder receives specific issues list, revises code, re-submits (up to 3 cycles)
- **Smoke check fails (syntax error)**: Builder gets error output and file path, attempts automatic fix before review
- **TDD loop failure**: Builder retries all configured iterations; failures don't abort the loop early
- **Task fails all resets**: Marked BLOCKED permanently (fail-fast), other tasks continue

### VERIFY Phase Recovery
- **Tests fail**: Director creates fix tasks → Builder applies fixes → Tester re-runs (up to `max_retries` attempts)
- **Security scan finds critical issues**: Builder receives findings and applies minimal targeted fixes
- **Integration check fails**: Cross-module import mismatches are logged with specific file:line references

### REFACTOR Phase Recovery
- **Quality score < threshold after refactoring**: Gardener iterates; post-refactor syntax verification auto-fixes obvious breaks
- **Regression detected**: Changes are reported; tests re-run to confirm

### Data Flow Between Phases
```
SPEC: Director → spec.json
  ↓
BUILD: Architect reads spec.json → dev_plan.json (tasks + architecture)
       Builder reads dev_plan.json + dependency file contents → source files
       Reviewer reads source files → review JSON (score + issues)
       If rejected: issues → Builder → revised files → Reviewer (loop)
  ↓
VERIFY: Tester reads source files → test_results.json
        If failures: Director reads test_results.json → fix tasks → Builder → re-test
        Security scanner reads source files → security_report.json
        If critical: Builder reads security_report.json → fixes
  ↓
REFACTOR: Reviewer reads all source files → quality score
          If score < threshold: Gardener reads source files + issues → improved files
  ↓
DELIVER: Package source + test_results.json + README → final output
```
