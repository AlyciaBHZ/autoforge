# AutoForge — AI Multi-Agent Project Generator

## What This Project Is

AutoForge is a Python framework that uses 6 AI agents to automatically generate complete, runnable software projects from a natural language description. Run `python forge.py "your idea"` and get a working project.

## Commands

```bash
# Setup (first time)
./setup.sh          # macOS/Linux
setup.bat           # Windows

# Run
source .venv/bin/activate
python forge.py "project description"       # Generate a project
python forge.py "desc" --budget 5.00        # With budget limit
python forge.py --status                    # Show all projects
python forge.py --resume                    # Resume interrupted run

# Test
python tests/smoke_test.py                  # 31-check smoke test suite

# Daemon mode (24/7 background service)
python forge.py daemon start                # Start daemon
python forge.py daemon status               # Check status
python forge.py queue "project description" # Add to build queue
python forge.py projects                    # List all projects
python forge.py deploy <project_id>         # Show deploy guide

# Version control (git sync)
python scripts/git_sync.py status           # Show branch sync status vs main
python scripts/git_sync.py changelog        # Show changes since last sync
python scripts/git_sync.py merge-main       # Merge main into current branch
python scripts/git_sync.py sync             # Full sync: fetch + merge + push
python scripts/git_sync.py cherry-pick <sha> # Cherry-pick specific commits
python scripts/git_sync.py pick-range 0,2-4 # Cherry-pick by index from main
```

## Architecture

5-phase pipeline: **SPEC → BUILD → VERIFY → REFACTOR → DELIVER**

6 agents: Director (Opus, requirements), Architect (Opus, design), Builder (Sonnet, code), Reviewer (Sonnet, review), Tester (Sonnet, test), Gardener (Sonnet, refactor).

## Key Files

```
forge.py                    Entry point — CLI argument parsing, orchestrator launch
engine/orchestrator.py      Pipeline controller — 5 phases, state persistence, resume
engine/config.py            ForgeConfig dataclass — models, budget, paths
engine/llm_router.py        LLM routing — Opus for complex, Sonnet for routine tasks
engine/agent_base.py        AgentBase — agentic tool-use loop (send → tool_use → execute → repeat)
engine/task_dag.py           TaskDAG — dependency graph, scheduling, persistence
engine/lock_manager.py      Cross-platform atomic task locking (symlink on POSIX, O_CREAT|O_EXCL on Windows)
engine/git_manager.py       Git worktree isolation for parallel builders
engine/sandbox.py           Subprocess + Docker sandbox for safe code execution
engine/project_registry.py  SQLite-backed multi-project management (daemon mode)
engine/daemon.py            24/7 daemon controller — queue → build → notify → repeat
engine/deploy_guide.py      Vercel deployment guide generator
engine/channels/            Input channels (Telegram bot, webhook API)
engine/agents/              6 agent implementations (director, architect, builder, reviewer, tester, gardener)
constitution/               Agent behavior rules, workflow definitions, quality gates
constitution/agents/*.md    Per-agent system prompts (loaded by agent_base.py)
constitution/workflows/*.md Phase definitions (spec, build, verify, refactor, deliver)
services/                   systemd + launchd service configs for daemon mode
tests/smoke_test.py         31-check validation suite (no API key needed)
scripts/git_sync.py         Automated git merge/cherry-pick workflow tool
```

## Coding Conventions

- Python 3.11+ with `from __future__ import annotations`
- All async — use `async/await` and `asyncio` throughout
- Type hints on all function signatures
- `pathlib.Path` for all file paths (never string concatenation)
- `rich` library for terminal output
- Cross-platform: use `sys.platform` checks where needed, `tempfile` for temp dirs
- Error handling: catch specific exceptions, log with `logging` module
- Constitution files (`.md`) control agent behavior — edit these to customize

## Important Rules

- Never hardcode API keys or secrets
- All file operations in agents must validate paths stay within workspace (path traversal prevention)
- Budget tracking: every LLM call records token usage via `config.record_usage()`
- Lock manager ensures one task per agent at a time
- Generated projects go in `workspace/` (gitignored)
- Test changes with `python tests/smoke_test.py` before committing
