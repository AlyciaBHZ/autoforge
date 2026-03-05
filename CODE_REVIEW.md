# AutoForge Code Review Report

## Architecture-Level Issues

### 1. Orchestrator is a 3174-line God Class (58 methods)

`engine/orchestrator.py` simultaneously handles: pipeline flow control, engine
initialization/lifecycle (28 `self._xxx: Any = None` attributes), task
scheduling, evolution/genetic algorithms, context budget calculation, file
overlap detection, formal verification/security scanning/theorem proving,
state persistence, knowledge graph ingestion, and three operating modes.

`_init_engines()` (lines 117-274) is a 157-line if-import-assign waterfall
with zero abstraction.

### 2. Feature Flag Hell — 32 boolean toggles in AdvancedConfig

`engine/config.py:166-231` has 32 boolean flags. No documentation on
interactions between flags. No way to know if disabling one flag breaks
another feature.

### 3. `Any` type abuse — 20+ core attributes untyped

`orchestrator.py:79-99` — 20 consecutive `Any`-typed attributes. Could use
`TYPE_CHECKING` conditional imports to maintain type safety with lazy loading.

---

## Critical/Major Bugs

### BUG-1: Mutable class variable shared across all LLMRouter instances

**`engine/llm_router.py:151`**
```python
class LLMRouter:
    _custom_providers: dict[str, LLMProvider] = {}
```
Not a dataclass. All instances share the same dict. Provider registration in
one pipeline contaminates all others. **Severity: Critical.**

### BUG-2: 54 `except Exception:` silently swallow engine init failures

**`engine/orchestrator.py:161-273`** — 20+ engines silently set to `None` on
any exception, including programming bugs (TypeError, NameError). No logging.

### BUG-3: Attributes conditionally created without `__init__` defaults

**`engine/orchestrator.py:187-273`** — Attributes like `self._world_model`,
`self._recursive_decomp_prover` only exist when their flag is True. Accessing
them when False raises `AttributeError` instead of returning `None`.

### BUG-4: Sandbox security filter is a bypassable blacklist

**`engine/sandbox.py:78-99`** — Regex blacklist can be bypassed via:
- `rm -rf /*` (not `/`)
- `bash -c "rm -rf /"`
- Base64 encoding: `eval $(echo "..." | base64 -d)`
- Python: `python -c "import shutil; shutil.rmtree('/')"`

### BUG-5: `exec_args` joins args into string for regex matching

**`engine/sandbox.py:193`** — Defeats the purpose of argument-vector execution
by reconstructing a shell-like string.

### BUG-6: `exec` and `exec_args` are 60 lines of copy-pasted code

**`engine/sandbox.py:118-186 vs 188-246`** — Nearly identical timeout handling,
process group killing, and exception handling duplicated.

---

## Design Smells

### SMELL-1: Config `__getattr__`/`__setattr__` proxy creates invisible API

**`engine/config.py:353-381`** — Attribute access silently delegates to
sub-configs. Name collisions resolve by search order. IDE completion and
static analysis broken.

### SMELL-2: `from_env()` factory method is 300+ lines

### SMELL-3: Tests use global variables instead of test framework

**`tests/smoke_test.py:25-26`** — `PASSED = 0`, `FAILED = 0` globals.

### SMELL-4: 45 `# noqa`/`# type: ignore` in smoke_test.py

### SMELL-5: Tool schemas duplicated across agents

Each agent defines `write_file`, `read_file`, `list_files` schemas inline
with near-identical JSON dicts.

### SMELL-6: Builder reads env vars directly, bypassing config

**`engine/agents/builder.py:151`** — `os.environ.get("GITHUB_TOKEN")` instead
of using `config.tools.github_token`.

### SMELL-7: Keyword-based heuristics for context budget

**`engine/orchestrator.py:762-767`** — Fragile keyword matching to determine
task type.

---

## Code Duplication

### DUP-1: `_init_engines()` pattern repeated 27 times

Same if-try-import-assign-except template, never abstracted.

### DUP-2: `_run_xxx()` methods are 7 copies of the same template

### DUP-3: `WorkflowGenome.to_dict()` manually serializes dataclass fields

Could use `dataclasses.asdict()`.

---

## Inconsistencies

### INCON-1: Exception handling split personality in `_init_engines()`

Lines 120-156: failures crash the program. Lines 161-273: failures silently
swallowed. No documented rationale.

### INCON-2: Agent constructor signatures vary without interface contract

### INCON-3: Log levels inconsistent across error paths

- Engine init failure: silent
- Security scan failure: `logger.warning`
- DAG ingestion failure: `logger.debug`
- Build task failure: `logger.error`

---

## Performance Issues

### PERF-1: `_list_project_files()` called before every task (full rglob)

### PERF-2: `json.dumps` on entire message history per LLM call for cost estimation

---

## Test Quality

### TEST-1: smoke_test.py is 92,526 lines with custom test framework

### TEST-2: No integration tests for end-to-end pipeline

### TEST-3: Three different test styles in tests/ directory

---

## Summary Scores

| Dimension       | Score | Notes                                       |
|----------------|-------|---------------------------------------------|
| Architecture   | 3/10  | God Class, feature flag hell                |
| Type Safety    | 2/10  | 20+ `Any`, `__getattr__` proxy             |
| Error Handling | 2/10  | 54x `except Exception:`, inconsistent logs  |
| Security       | 4/10  | Path traversal OK, sandbox blacklist broken |
| Code Duplication| 2/10 | init waterfall, tool schemas, run templates |
| Test Quality   | 3/10  | Quantity exists, quality poor               |
| Maintainability| 2/10  | Adding engine requires 4+ file changes      |
| Consistency    | 3/10  | Constructors, exceptions, logs all differ   |

## Priority Fixes

1. Fix `LLMRouter._custom_providers` shared mutable class variable
2. Add `logger.warning` to silent `except Exception:` blocks in `_init_engines`
3. Pre-initialize all engine attributes to `None` in `__init__`
4. Replace sandbox blacklist with whitelist or proper sandboxing
