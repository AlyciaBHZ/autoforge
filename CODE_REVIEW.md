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

### BUG-7: JSON extraction regex duplicated 7 times across agents

The same JSON-from-LLM-output extraction pattern is copy-pasted across
`director.py:114-135`, `director.py:175-192`, `architect.py:158-167`,
`reviewer.py:159-184`, `tester.py:169-193`, `gardener.py:196-212`,
`scanner.py:223-240`. If the regex needs updating, 7 places must change.

### BUG-8: `UnicodeDecodeError` not caught in 4 of 5 agents' `read_file`

`builder.py:194`, `reviewer.py:84`, `tester.py:110`, `gardener.py:143` all
call `read_text(encoding="utf-8")` without catching `UnicodeDecodeError`.
Reading any binary file crashes the agent. Only `scanner.py:121-128` handles
this correctly. **Severity: Major.**

### BUG-9: Path traversal errors not caught in builder/gardener write handlers

`builder.py:183` and `gardener.py:132` — `_validate_path()` raises
`ValueError` on traversal, but no try/except wraps it. The raw exception
propagates instead of returning a structured JSON error. `reviewer.py:78-81`
and `scanner.py:133-135` handle it correctly — inconsistent security behavior.

### BUG-10: `DirectorFixAgent` loads wrong constitution file

`director.py:145` — `DirectorFixAgent` sets `ROLE = "director"` (same as
`DirectorAgent` on line 25). Both load `constitution/agents/director.md`.
`DirectorFixAgent` has a completely different purpose (fix task generation)
but gets the generic director prompt.

### BUG-11: Scanner command blocklist trivially bypassable

`scanner.py:158-159` — Substring matching blocklist:
```python
blocked = ["rm ", "mv ", "cp ", "write", "install", "npm ", "pip ", "delete", "> ", ">> "]
```
Bypassed by: `/bin/rm`, `$(rm -rf /)`, backtick expansion, `python -c "..."`,
pipes, semicolons. The string `"write"` blocks `grep write` false positives.

### BUG-12: `parse_architecture` raises raw exceptions unlike all sibling methods

`architect.py:156-167` — Raises raw `json.JSONDecodeError` instead of returning
a structured error object. All other agents' parse methods (`reviewer.py:173-177`,
`tester.py`, etc.) catch `JSONDecodeError` and return a default failure object.

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

### DUP-4: `_handle_list_files` copied verbatim across 4 agents

`builder.py:197-213`, `reviewer.py:86-105`, `gardener.py:145-161`,
`scanner.py:130-149` — all do the same rglob + filter .git + symlink check.
Only difference: scanner also filters `node_modules` and caps at 500.

### DUP-5: `_handle_read_file` duplicated across 5 agents

`builder.py:189-195`, `reviewer.py:76-84`, `tester.py:102-110`,
`gardener.py:138-143`, `scanner.py:113-128`. Scanner adds truncation and
`UnicodeDecodeError` handling that all others are missing.

### DUP-6: Web/GitHub tool registration duplicated across 4 agents

`director.py:32-92`, `architect.py:63-125`, `builder.py:134-167`,
`gardener.py:101-116` — same conditional import, same env var read, same
ToolDefinition construction with slightly different descriptions. Should be
a `_register_web_tools()` helper on AgentBase.

### DUP-7: `MAX_FILE_SIZE = 2 * 1024 * 1024` defined identically in 2 files

`builder.py:174` and `gardener.py:119` — same constant, defined twice.

---

## Inconsistencies

### INCON-1: Exception handling split personality in `_init_engines()`

Lines 120-156: failures crash the program. Lines 161-273: failures silently
swallowed. No documented rationale.

### INCON-2: Agent constructor signatures vary without interface contract

| Agent | Extra params |
|-------|-------------|
| DirectorAgent | none |
| ArchitectAgent | `templates_dir` |
| BuilderAgent | `working_dir`, `sandbox`, `agent_id` |
| ReviewerAgent | `working_dir` |
| TesterAgent | `working_dir`, `sandbox` |
| GardenerAgent | `working_dir` |
| ScannerAgent | `working_dir` |

Makes `AGENT_REGISTRY` unusable for generic instantiation.

### INCON-4: Missing type hints on agent `__init__` parameters

All agents take `config` and `llm` without type hints (`config: ForgeConfig`,
`llm: LLMRouter`), violating the project's own "type hints on all function
signatures" convention.

### INCON-5: No abstract `parse_output` interface

Most agents have `parse_spec`, `parse_architecture`, `parse_review`, etc. but
this is not an abstract method on AgentBase. Director has two parse methods,
Builder has zero. No generic orchestration possible.

### INCON-6: `_handle_write_file` response format differs between agents

`builder.py:187` returns `"bytes": len(content)`. `gardener.py:136` returns
only `"status": "ok", "path": rel_path` with no size info.

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
5. Add `UnicodeDecodeError` handling to all agents' `_handle_read_file`
6. Extract shared tool handlers (`read_file`, `write_file`, `list_files`) into AgentBase or mixin
7. Extract JSON-from-LLM regex into a single shared utility
8. Give `DirectorFixAgent` its own ROLE and constitution file
9. Add type hints to all agent `__init__` parameters
