# AutoForge — AI Multi-Agent Project Generator

## What This Project Is

AutoForge is a Python framework that uses 6 AI agents to automatically generate complete, runnable software projects from a natural language description. Run `forgeai` to start an interactive session.

## Commands

```bash
# Install
pip install -e .                             # From source
pip install -e ".[all]"                      # From source (all extras)

# Run
forgeai                                      # Interactive session (recommended)
forgeai generate "project description"       # Generate a project
forgeai generate "desc" --budget 5.00        # With budget limit
forgeai status                               # Show all projects
forgeai resume                               # Resume interrupted run
forgeai setup                                # Reconfigure settings

# Test
python tests/smoke_test.py                  # 127-check smoke test suite
python -m pytest tests/test_engines.py      # 112 behavioral/unit tests

# Daemon mode (24/7 background service)
forgeai daemon start                         # Start daemon
forgeai daemon status                        # Check status
forgeai queue "project description"          # Add to build queue
forgeai projects                             # List all projects
forgeai deploy <project_id>                  # Show deploy guide

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
engine/config.py            ForgeConfig dataclass — models, budget, paths, search tree & checkpoint settings
engine/llm_router.py        LLM routing — Opus for complex, Sonnet for routine tasks
engine/agent_base.py        AgentBase — agentic tool-use loop + checkpoints + dynamic constitution
engine/search_tree.py       Search tree + RethinkMCTS — branching, evaluation, backtracking, thought refinement
engine/checkpoints.py       Mid-task checkpoints — Process Reward Model style direction checking
engine/dynamic_constitution.py  Dynamic constitution + meta-learning knowledge base
engine/evolution.py             Evolution engine — cross-project workflow self-improvement
engine/prompt_optimizer.py      DSPy/OPRO-style automatic prompt self-optimization
engine/process_reward.py        CodePRM — step-level process reward model for code generation
engine/evomac.py                EvoMAC — text backpropagation, natural-language gradient feedback between agents
engine/sica.py                  SICA — self-improving coding agent, constitution self-editing + rollback
engine/rag_retrieval.py         Library-level RAG — BM25+TF-IDF hybrid cross-project code retrieval
engine/formal_verify.py         Formal verification — multi-level linting, type checking, LLM formal analysis
engine/reasoning_extension.py   Autonomous reasoning extension — minimal kernel self-growth, numbered conclusions, publication gate
engine/article_verifier.py      Article verification pipeline — claim extraction, Lean 4 formalization, cross-prover verification
engine/agent_debate.py          Conditional debate — reward-guided multi-agent architecture debate
engine/security_scan.py         RedCode security scanning — pattern matching + LLM deep vulnerability analysis
engine/reflexion.py             Reflexion — verbal RL with episodic memory for failure-informed retries
engine/adaptive_compute.py      Adaptive test-time compute — difficulty-aware resource allocation + self-calibration
engine/ldb_debugger.py          LDB — block-level fault localization via runtime simulation
engine/speculative_pipeline.py  Speculative pipeline — overlapping phase pre-execution for speed
engine/hierarchical_decomp.py   Hierarchical decomposition — Parsel-style function-level task planning
engine/lean_prover.py           Lean 4 theorem proving — Hilbert+COPRA+MCTS+STP+Pantograph+PaperReview+ProofRepair
engine/multi_prover.py          Multi-prover formal verification — Coq, Isabelle, TLA+, Z3/SMT, Dafny cross-verification
engine/autonomous_discovery.py  Autonomous theorem discovery — DomainContext-templated prompts, minimal kernel → conjecture generation → novelty filter → depth evaluation
engine/paper_formalizer.py      Paper-specific Lean 4 formalization — real Lean compile, theorem extraction, Lean codegen, Python reproducibility, markdown report
engine/cloud_prover.py          Cloud Lean 4 compilation — Docker, SSH, GitHub Codespaces backends with proof caching
engine/capability_dag.py        CapabilityDAG — self-growing universal knowledge graph, community-mergeable
engine/theoretical_reasoning.py Cross-domain scientific reasoning — TheoryGraph, multi-modal verification, theory evolution & article generation
engine/experiment_loop.py       Closed-loop experiment pipeline — AI Scientist v2-style hypothesis→code→run→analyze→iterate with ablation studies
engine/paper_writer.py          Automated paper writing — LaTeX generation, BibTeX management, figure generation, NeurIPS/ICML/ICLR templates
engine/dense_retrieval.py       Dense embedding premise selection — ReProver/LeanDojo-style retrieval replacing Jaccard, FAISS support
engine/benchmark_eval.py        Standard benchmark evaluation — miniF2F, PutnamBench, LeanWorkbook, ProofNet harnesses with Pass@k
engine/rl_proof_search.py       RL proof search — AlphaProof/DeepSeek-Prover-V2 style PUCT-MCTS with policy/value networks, expert iteration
engine/literature_search.py     Enhanced literature search — citation graph traversal, SPECTER2 embedding search, full-text analysis, gap detection
engine/paper_repro.py           Enhanced paper reproduction — full pipeline: goal→paper→signals→code→execute→compare→report
engine/provers/lean_lake.py     Real Lean 4 Lake integration — proper Mathlib project scaffolding, Lake build, project pool, 32 import mappings
engine/task_dag.py           TaskDAG — dependency graph, scheduling, persistence
engine/lock_manager.py      Cross-platform atomic task locking (symlink on POSIX, O_CREAT|O_EXCL on Windows)
engine/git_manager.py       Git worktree isolation for parallel builders
engine/sandbox.py           Subprocess + Docker sandbox for safe code execution
engine/project_registry.py  SQLite-backed multi-project management (daemon mode)
engine/daemon.py            24/7 daemon controller — queue → build → notify → repeat
engine/deploy_guide.py      Vercel deployment guide generator
engine/channels/            Input channels (Telegram bot, webhook API)
engine/tools/web.py         Web search (DuckDuckGo/Google) + URL fetching
engine/tools/search.py      Project code search (grep)
engine/tools/github_search.py  GitHub repository & code search API integration
engine/agents/              6 agent implementations (director, architect, builder, reviewer, tester, gardener)
constitution/               Agent behavior rules, workflow definitions, quality gates
constitution/agents/*.md    Per-agent system prompts (loaded by agent_base.py)
constitution/workflows/*.md Phase definitions (spec, build, verify, refactor, deliver)
services/                   systemd + launchd service configs for daemon mode
tests/smoke_test.py         127-check validation suite (no API key needed)
tests/test_engines.py       112 behavioral tests (SearchTree, MCTS, Evolution, etc.)
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
