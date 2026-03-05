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

### BUG-13: TOCTOU race condition in LockManager.release()

**`engine/lock_manager.py:71-84`** — Between reading the lock owner and
unlinking the file, another process can release and re-claim the lock:
```python
owner = self._read_owner(lock_path)  # read
if owner == agent_id:
    lock_path.unlink()                # delete — but lock may have been re-claimed
```
Classic time-of-check-time-of-use race. **Severity: Critical.**

### BUG-14: LockManager task_id path traversal — no sanitization

**`engine/lock_manager.py:44`** — `task_id` used directly in filename:
```python
lock_path = self.lock_dir / f"{task_id}.lock"
```
A task ID like `../../etc/cron.d/evil` creates a file outside `lock_dir`.
No `validate_path` call unlike `agent_base.py`. **Severity: Major.**

### BUG-15: `BudgetExceededError` swallowed by agent_base catch-all

**`engine/agent_base.py:378`** — The outermost `except Exception as e` in the
agent loop catches `BudgetExceededError`, reporting it as a generic failure
instead of propagating it to the orchestrator for proper handling.

### BUG-16: Checkpoint reset corrupts `files_written_list`

**`engine/agent_base.py:341-356`** — On checkpoint reset, `files_written` counter
resets to 0 and `last_write_turn` resets to 0, but `files_written_list` (the
list of file paths) is never cleared. Stale paths from the pre-reset
conversation feed incorrect data to subsequent checkpoint evaluations.

### BUG-17: UCB1 crashes with `math.log(0)` when parent unvisited

**`engine/search_tree.py:650-651`** — If root node has `visit_count=0`,
`math.log(parent_visits)` raises `ValueError: math domain error`. The child
guard (`visit_count == 0 → return inf`) does not protect against zero parent.

### BUG-18: JSON extraction regex `\{.*?\}` breaks on nested objects

**`engine/search_tree.py:455,850`, `engine/checkpoints.py:180`** — The non-greedy
`\{.*?\}` matches first `{` to first `}`. For `{"key": {"nested": true}}` it
returns `{"key": {"nested": true}` — invalid JSON. Used in 3+ places.

### BUG-19: `reset_failed` raises raw KeyError unlike sibling methods

**`engine/task_dag.py:155-160`** — `self._tasks[task_id]` raises `KeyError`
if task doesn't exist. All other methods (lines 129, 135, 143) check and raise
`ValueError` with a clear message.

### BUG-20: LockManager is fully synchronous in an async codebase

**`engine/lock_manager.py`** — All filesystem operations are synchronous. In an
async codebase (project convention: "All async"), these block the event loop.
No `asyncio.to_thread()` wrapper.

### BUG-21: Async HTTP clients never closed — resource leak in daemon mode

**`engine/llm_router.py:227-264`** — `AsyncAnthropic`, `AsyncOpenAI`,
`genai.Client` are created and cached but never closed. No `close()` or
`__aexit__` on `LLMRouter`. In long-running daemon mode, connection pools
accumulate indefinitely.

### BUG-22: `detect_provider` false positive on "o"-prefixed models

**`engine/llm_router.py:128`** — `model_lower.startswith(("o1", "o3", "o4"))`
matches any model starting with these prefixes (e.g., `o100`, hypothetical
`opus-4`). The exact-match set `_OPENAI_MODELS` already covers the intended
models, making the prefix check overly greedy.

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

### SMELL-8: Tool lookup is O(n) linear scan per call

**`engine/agent_base.py:161-162`** — `_execute_tool` iterates through
`self._tools` list for every tool call. Should use a dict lookup.

### SMELL-9: Checkpoint evaluator returns "all clear" on failure

**`engine/checkpoints.py:198-203`** — When the LLM checkpoint check fails
(exception), returns `on_track=True`. A failed safety check should be
conservative (low score or propagate error), not optimistic.

### SMELL-10: Checkpoints accumulate in memory with no eviction

**`engine/checkpoints.py:104`** — `copy.deepcopy(messages)` called every
`checkpoint_interval` turns. Messages contain full conversation history. For
a 25-turn agent, each checkpoint can be megabytes. Never evicted.

### SMELL-11: LockManager cache is computed but never actually used

**`engine/lock_manager.py:127-130`** — `cached_count` is computed then
immediately discarded by a full filesystem scan. The cache gives a false
sense of optimization.

### SMELL-12: `TaskDAG` has three confusingly similar completion methods

**`engine/task_dag.py:105-120`** — `is_finished()`, `is_all_done()`,
`is_complete()` with subtly different semantics. `is_complete` is a
"backward-compatible alias" that does something different from both others.

### SMELL-13: f-string logging wastes CPU for disabled log levels

Throughout the codebase, `logger.debug(f"...")` interpolates the string
unconditionally even when DEBUG is disabled. Should use `%`-style:
`logger.debug("msg: %s", value)`. Affects `agent_base.py:117`,
`llm_router.py:165`, `search_tree.py:325`, and many more.

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

### DUP-8: Broken JSON regex `\{.*?\}` duplicated in 3+ files

`search_tree.py:455,850`, `checkpoints.py:180`, plus 7x in agents (BUG-7).
All use the same broken non-greedy pattern that fails on nested JSON.

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

### INCON-7: `TaskDAG` error style inconsistent — KeyError vs ValueError

`reset_failed()` raises raw `KeyError`, all other methods raise `ValueError`
with descriptive message.

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

| Dimension       | Score | Notes                                          |
|----------------|-------|-------------------------------------------------|
| Architecture   | 3/10  | God Class, feature flag hell                    |
| Type Safety    | 2/10  | 20+ `Any`, `__getattr__` proxy                  |
| Error Handling | 2/10  | 54x `except Exception:`, inconsistent logs      |
| Security       | 3/10  | Path traversal OK, sandbox/lockmanager broken   |
| Code Duplication| 2/10 | init waterfall, tool schemas, run templates     |
| Test Quality   | 3/10  | Quantity exists, quality poor                   |
| Maintainability| 2/10  | Adding engine requires 4+ file changes          |
| Consistency    | 3/10  | Constructors, exceptions, logs all differ       |
| Concurrency    | 2/10  | TOCTOU in locks, sync ops in async, no cleanup  |

**Total bugs found: 22** (3 critical, 10 major, 9 minor)
**Design smells: 13** | **Duplications: 8** | **Inconsistencies: 7**

## Priority Fixes (by severity)

### P0 — Critical (fix immediately)
1. Fix `LLMRouter._custom_providers` shared mutable class variable (BUG-1)
2. Fix TOCTOU race in `LockManager.release()` (BUG-13)
3. Sanitize `task_id` in `LockManager` to prevent path traversal (BUG-14)

### P1 — Major (fix before next release)
4. Add `logger.warning` to silent `except Exception:` blocks in `_init_engines` (BUG-2)
5. Pre-initialize all engine attributes to `None` in `__init__` (BUG-3)
6. Replace sandbox blacklist with whitelist or proper sandboxing (BUG-4)
7. Fix JSON extraction regex to handle nested objects (BUG-18)
8. Don't swallow `BudgetExceededError` in `agent_base.py` catch-all (BUG-15)
9. Add `UnicodeDecodeError` handling to all agents' `_handle_read_file` (BUG-8)
10. Add `LLMRouter.close()` for async client cleanup (BUG-21)
11. Guard `math.log(0)` in UCB1 calculation (BUG-17)

### P2 — Refactoring (next sprint)
12. Extract shared tool handlers into AgentBase or mixin (DUP-4,5,6)
13. Extract JSON-from-LLM regex into a single shared utility (DUP-8)
14. Abstract `_init_engines()` pattern into a generic loader (DUP-1)
15. Give `DirectorFixAgent` its own ROLE and constitution file (BUG-10)
16. Wrap `LockManager` in async (BUG-20)
17. Add type hints to all agent `__init__` parameters (INCON-4)
