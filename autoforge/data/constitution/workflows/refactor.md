# Phase 4: REFACTOR — Quality Improvement

## Input
- Verified source code from Phase 3
- Review feedback (if any remaining issues)

## Process
1. **Reviewer** performs a final code quality assessment
2. If quality score < threshold: **Gardener** refactors
3. Gardener makes targeted improvements
4. **Tester** runs regression tests after refactoring
5. If regression detected: revert changes, keep original code

## Output
- Improved source code (or unchanged if already passing quality threshold)

## Quality Gate
- [ ] Quality score >= threshold (default 0.7)
- [ ] All tests still pass after refactoring
- [ ] No new issues introduced
