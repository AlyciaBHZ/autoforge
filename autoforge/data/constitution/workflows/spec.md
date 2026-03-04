# Phase 1: SPEC — Requirement Analysis

## Input
Natural language project description from the user.

## Process
1. **Director** receives the user description
2. Director analyzes requirements, infers tech stack, identifies modules
3. Director produces `spec.json` with structured specification
4. Director defines a **build contract** — deliverables, test expectations, reports, and stop conditions
5. Director identifies what is out of scope (MVP boundary)

## Output
- `workspace/<project>/spec.json` — Structured project specification with build contract

## Quality Gate
- [ ] spec.json is valid JSON
- [ ] spec.json contains at least 1 module
- [ ] Each module has a name, description, and file list
- [ ] Tech stack is fully specified
- [ ] project_name is a valid directory name (kebab-case)
- [ ] `build_contract` is present and contains: `deliverables`, `test_requirements`, `reports`, `stop_conditions`
- [ ] `stop_conditions.max_tasks` ≤ 20 (overnight-sized scope)
- [ ] `stop_conditions.max_modules` ≤ 10
- [ ] Module count does not exceed `stop_conditions.max_modules`
