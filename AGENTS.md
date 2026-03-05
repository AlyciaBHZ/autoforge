# AutoForge

AI multi-agent project generation framework. Describe what you want in natural language, and 6 AI agents build it for you.

## Quick Reference

```bash
# Setup
./setup.sh                  # macOS/Linux
setup.bat                   # Windows

# Run
source .venv/bin/activate   # macOS/Linux (.venv\Scripts\activate.bat on Windows)
python forge.py "Build a Todo app with user login"
python forge.py "desc" --budget 5.00 --agents 3 --verbose
python forge.py --status
python forge.py --resume

# Test
python tests/smoke_test.py

# Daemon mode
python forge.py daemon start                # 24/7 background service
python forge.py queue "project description" # Add to build queue
python forge.py projects                    # List all projects
python forge.py deploy <project_id>         # Show deploy guide
```

## Architecture

**Pipeline:** SPEC → BUILD → VERIFY → REFACTOR → DELIVER

**Agents:**
| Agent | Model | Role |
|-------|-------|------|
| Director | Opus | Requirement analysis, module decomposition |
| Architect | Opus | Architecture design, task DAG generation |
| Builder | Sonnet | Code implementation (parallel, git worktree isolated) |
| Reviewer | Sonnet | Code review, quality scoring |
| Tester | Sonnet | Build verification, test execution in sandbox |
| Gardener | Sonnet | Refactoring based on review feedback |

## Project Structure

```
forge.py                    CLI entry point
engine/
  orchestrator.py           5-phase pipeline controller
  config.py                 ForgeConfig — models, budget, paths
  llm_router.py             Model routing (Opus/Sonnet) + budget enforcement
  agent_base.py             Agentic tool-use loop base class
  task_dag.py               Task dependency graph + scheduler
  lock_manager.py           Cross-platform atomic task locking
  git_manager.py            Git worktree management for parallel builders
  sandbox.py                Subprocess + Docker command execution
  project_registry.py       SQLite multi-project management (daemon mode)
  daemon.py                 24/7 daemon controller
  deploy_guide.py           Vercel deployment guide generator
  channels/                 Input channels (Telegram bot, webhook API)
  agents/                   6 agent implementations
constitution/
  CONSTITUTION.md           Core principles and hard rules
  agents/*.md               Per-agent system prompts
  workflows/*.md            Phase workflow definitions
  quality_gates.md          Phase transition criteria
services/                   systemd + launchd service configs
tests/smoke_test.py         31-check smoke test (no API key needed)
```

## Coding Standards

- Python 3.10+, `from __future__ import annotations`
- Fully async (`async/await`, `asyncio`)
- Type hints on all signatures
- `pathlib.Path` for file paths
- `rich` for terminal UI
- Cross-platform (Windows + macOS + Linux)
- Path traversal prevention in all agent file operations
- Budget tracking on every LLM call

## Key Design Decisions

- **LLM SDK**: Multi-provider support (Anthropic, OpenAI, Google) with native async clients for full tool-use loop control
- **Parallelism**: `asyncio` with `asyncio.wait()` for concurrent builder tasks
- **Task locks**: `os.symlink()` on POSIX, `os.open(O_CREAT|O_EXCL)` on Windows
- **Sandbox**: SubprocessSandbox (default) or DockerSandbox (optional, `--network none`)
- **State**: JSON persistence for resume capability
- **Constitution**: Markdown files in `constitution/` define agent behavior — editable

## Supported Project Types

web-app, api-server, cli-tool, static-site, mobile-scaffold, desktop-scaffold, library.
Languages: TypeScript, Python, Go, Java, Dart. Tech stacks auto-selected by Director agent.
