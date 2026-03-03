# AutoForge — AI Multi-Agent Project Generator

AutoForge uses 6 AI agents to build complete projects from natural language. Run `python forge.py "your idea"` to generate a project.

## Commands

- Setup: `./setup.sh` (macOS/Linux) or `setup.bat` (Windows)
- Run: `python forge.py "project description"`
- Test: `python tests/smoke_test.py` (22 checks, no API key needed)
- Status: `python forge.py --status`
- Resume: `python forge.py --resume`

## Architecture

5-phase pipeline (SPEC → BUILD → VERIFY → REFACTOR → DELIVER) with 6 agents:
Director (Opus, requirements), Architect (Opus, design), Builder (Sonnet, code),
Reviewer (Sonnet, review), Tester (Sonnet, test), Gardener (Sonnet, refactor).

## Key Files

- `forge.py` — CLI entry point
- `engine/orchestrator.py` — Pipeline controller
- `engine/config.py` — Configuration + budget tracking
- `engine/agent_base.py` — Agentic tool-use loop
- `engine/agents/` — Agent implementations
- `constitution/` — Agent behavior rules (editable markdown)

## Coding Standards

- Python 3.11+, `from __future__ import annotations`
- Async throughout (`async/await`, `asyncio`)
- Type hints on all signatures
- `pathlib.Path` for file paths
- `rich` for terminal UI
- Cross-platform (Windows + macOS + Linux)
- Path traversal prevention in agent file ops
- Always run `python tests/smoke_test.py` after changes
