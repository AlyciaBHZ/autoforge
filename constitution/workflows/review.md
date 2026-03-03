# Review Workflow

Standalone code review for existing projects.

## Pipeline

```
SCAN → REVIEW → [REFACTOR] → REPORT
```

## Phases

### Phase 1: SCAN
- Scanner Agent analyzes the project structure
- Produces spec.json describing current state
- Identifies tech stack, modules, gaps

### Phase 2: REVIEW
- Reviewer Agent examines the full codebase
- Checks: correctness, security, error handling, code quality, conventions
- Produces per-module scores and overall quality score
- Lists all issues with severity, location, and suggestions

### Phase 3: REFACTOR (Developer mode only)
- If quality score < threshold AND mode is "developer"
- Gardener Agent applies targeted fixes
- Re-runs review to verify improvements

### Phase 4: REPORT
- Generate formatted review report
- Include: overall score, issues by severity, recommendations
- Save to project directory as `.autoforge/review_report.json`

## Quality Gate

No quality gate for review — the review itself IS the quality assessment.
Reports are always generated regardless of score.

## Modes

- **Research mode**: SCAN → REVIEW → REPORT (no modifications)
- **Developer mode**: SCAN → REVIEW → REFACTOR → REPORT (applies fixes)
