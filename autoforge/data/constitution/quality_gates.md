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

## Failure Handling

1. **First failure**: Retry the phase (up to 3 times)
2. **Persistent failure**: Degrade — simplify the approach
3. **Degradation fails**: Mark as blocked, report to user
4. **Budget exceeded**: Stop immediately, deliver what's available
