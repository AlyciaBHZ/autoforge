# Phase 3: VERIFY — Testing and Validation

## Input
Generated source code from Phase 2.

## Process
1. **Tester** agent examines the project structure
2. Tester installs dependencies in the sandbox
3. Tester runs build commands
4. Tester runs test suites (if any)
5. Tester verifies the application starts without errors
6. On failure: Director creates fix tasks, Builder fixes, re-test (up to 3 retries)

## Output
- `test_results.json` with pass/fail status per step

## Quality Gate
- [ ] Dependencies install without errors
- [ ] Project builds without errors
- [ ] Application starts without crashing
- [ ] No critical test failures
