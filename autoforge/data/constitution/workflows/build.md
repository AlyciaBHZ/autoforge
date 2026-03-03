# Phase 2: BUILD — Parallel Development

## Input
- `spec.json` from Phase 1
- Architecture design from Architect

## Process
1. **Architect** receives spec, designs architecture and task DAG
2. Architect produces `dev_plan.json` with tasks, dependencies, and file ownership
3. **Builder** agents claim tasks from the DAG (via LockManager)
4. Each Builder works in its own git worktree
5. After completing a task, **Reviewer** checks the code
6. If approved: merge to main, mark task done
7. If rejected: Builder revises based on feedback, re-submit

## Parallel Execution
- Tasks with no unmet dependencies can execute simultaneously
- Each Builder claims exactly 1 task at a time (hard rule)
- The orchestrator polls for ready tasks and assigns idle Builders

## Output
- Source code files in the project directory
- `dev_plan.json` updated with task statuses

## Quality Gate
- [ ] All BUILD-phase tasks are marked "done"
- [ ] All expected source files exist
- [ ] No tasks in "failed" or "blocked" state
- [ ] All code has passed Reviewer review
