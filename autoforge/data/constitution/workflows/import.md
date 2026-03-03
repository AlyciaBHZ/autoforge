# Import Workflow

Import and improve existing projects at any completion level.

## Pipeline

```
SCAN → REVIEW → [ENHANCE] → VERIFY → [REFACTOR] → DELIVER
```

## Phases

### Phase 1: SCAN
- Scanner Agent analyzes the existing codebase
- Produces spec.json describing current state
- Identifies gaps and incomplete features
- Assesses completeness percentage

### Phase 2: REVIEW
- Reviewer Agent performs full code review
- Identifies issues, security vulnerabilities, code quality problems
- Produces quality score and improvement recommendations

### Phase 3: ENHANCE (Optional, if enhancement description provided)
- Director Agent merges existing spec with enhancement requests
- Architect Agent designs task DAG for new features only
- Builder Agents implement new features while respecting existing code
- Reviewer checks new code against existing patterns

### Phase 4: VERIFY
- Tester Agent runs existing test suite
- Verifies project builds and starts
- If tests fail: fix cycle (Director + Builder)

### Phase 5: REFACTOR (Developer mode only)
- If quality score < threshold
- Gardener Agent applies targeted improvements
- Focuses on issues found in Phase 2

### Phase 6: DELIVER
- Generate/update README
- Save analysis reports to .autoforge/
- Print summary with before/after quality scores

## Key Principles

1. **Respect existing code**: Don't rewrite what works
2. **Preserve patterns**: New code follows existing conventions
3. **Minimal changes**: Fix issues surgically, not wholesale
4. **Git history**: Preserve existing git history where possible
5. **Backward compatible**: Don't break existing functionality
