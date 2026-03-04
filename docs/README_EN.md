```
     _         _        _____
    / \  _   _| |_ ___ |  ___|__  _ __ __ _  ___
   / _ \| | | | __/ _ \| |_ / _ \| '__/ _` |/ _ \
  / ___ \ |_| | || (_) |  _| (_) | | | (_| |  __/
 /_/   \_\__,_|\__\___/|_|  \___/|_|  \__, |\___|
                                       |___/
```

**One command. Six agents. A complete, runnable software project.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)
[![Tests](https://img.shields.io/badge/tests-105%20checks-brightgreen.svg)](../tests/smoke_test.py)

[中文](../README.md) | [Developer Docs](../CLAUDE.md)

---

## Quick Start

```bash
./setup.sh                                    # Install dependencies
python forge.py "Todo app with user auth"     # Generate a project
```

On first run, the setup wizard guides you through API key configuration (Anthropic / OpenAI / Google — any one will do). You can skip and configure later. Output lands in `workspace/`, ready to run.

---

## What It Does

AutoForge orchestrates 6 specialized AI agents through a 5-phase pipeline to turn a natural-language idea into a working codebase. Requirements analysis, architecture design, parallel code generation, testing, code review, refactoring, and deployment packaging — fully automated.

**Highlights:**
- Full-stack web apps, APIs, CLI tools, and mobile apps from a single prompt
- Multi-provider LLM support with cross-provider mixing (strong model for planning, fast model for coding)
- Budget controls with real-time cost tracking (`--budget 5.00`)
- Sandbox execution (Docker or subprocess) for safe testing
- Interrupt-safe — resume any run without losing progress
- 24/7 daemon mode with build queue, Telegram bot, and REST API

---

## Architecture

```
 "Build a todo app with auth"
          |
          v
 ┌─────────────────────────────────────────────┐
 │  SPEC      Director analyses requirements    │
 ├─────────────────────────────────────────────┤
 │  BUILD     Architect designs, Builders code  │
 │            Reviewer checks each task         │
 ├─────────────────────────────────────────────┤
 │  VERIFY    Tester runs build + tests         │
 │            Failures auto-generate fix tasks   │
 ├─────────────────────────────────────────────┤
 │  REFACTOR  Gardener optimises code quality   │
 ├─────────────────────────────────────────────┤
 │  DELIVER   README, structure, cost report    │
 └─────────────────────────────────────────────┘
          |
          v
    workspace/my-todo-app/
```

| Agent | Role | Model Tier |
|-------|------|------------|
| **Director** | Requirements & scoping | Strong |
| **Architect** | System design & task DAG | Strong |
| **Builder** | Code generation (parallelised) | Fast |
| **Reviewer** | Code review & scoring | Fast |
| **Tester** | Build, test, auto-fix loop | Fast |
| **Gardener** | Refactoring & security fixes | Fast |

---

## Supported LLM Providers

AutoForge supports multiple LLM providers. Mix and match freely:

| Provider | Environment Variable | Recommended Models |
|----------|---------------------|--------------------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude Opus 4 (strong), Claude Sonnet 4 (fast) |
| **OpenAI** | `OPENAI_API_KEY` | GPT-4o / o3 (strong), GPT-4o-mini (fast) |
| **Google** | `GOOGLE_API_KEY` | Gemini 2.5 Pro (strong), Gemini 2.5 Flash (fast) |

Keys can also be stored in `~/.autoforge/config.toml`. Cross-provider example:

```bash
export FORGE_MODEL_STRONG=gpt-4o          # Strong model from OpenAI
export FORGE_MODEL_FAST=gemini-2.5-flash  # Fast model from Google
```

---

## Usage

```bash
# Generate projects
python forge.py "REST API for a bookstore with JWT auth"
python forge.py "Landing page for a SaaS product" --budget 3.00

# Manage runs
python forge.py --status          # Show all projects
python forge.py --resume          # Resume interrupted run

# Daemon mode (24/7 background service)
python forge.py daemon start
python forge.py queue "Blog with markdown support"
python forge.py projects
python forge.py deploy <project_id>
```

<details>
<summary>Cost estimates</summary>

| Complexity | Example | Estimated Cost |
|------------|---------|:--------------:|
| Simple | Todo app, landing page | $2--3 |
| Medium | Blog system, booking platform | $4--6 |
| Complex | E-commerce MVP, multi-role platform | $7--10 |

Default budget cap: $10. Override with `--budget`.

</details>

---

## Academic Foundations

AutoForge integrates techniques from recent AI and software engineering research:

| Engine | Source | Paper / Reference |
|--------|--------|-------------------|
| **RethinkMCTS** | `search_tree.py` | RethinkMCTS (2024) — thought refinement via execution feedback in MCTS |
| **EvoMAC** | `evomac.py` | EvoMAC (ICLR 2025) — text backpropagation with natural-language gradients |
| **SICA** | `sica.py` | SICA (ICLR 2025 Workshop) + STO (NeurIPS 2025) — self-improving coding agents |
| **Reflexion** | `reflexion.py` | Reflexion (NeurIPS 2023) — verbal reinforcement learning |
| **CodePRM** | `process_reward.py` | CodePRM (ACL 2025) — step-level process reward model for code |
| **LDB** | `ldb_debugger.py` | LDB (ACL 2024) — block-level fault localisation |
| **Adaptive Compute** | `adaptive_compute.py` | Scaling LLM Test-Time Compute (ICLR 2025) — difficulty-aware resource allocation |
| **Speculative Pipeline** | `speculative_pipeline.py` | Speculative Actions — overlapping phase pre-execution |
| **Parsel** | `hierarchical_decomp.py` | Parsel (NeurIPS 2023) + CodePlan (ACM 2024) — function-level decomposition |
| **Lean Prover** | `lean_prover.py` | Hilbert (NeurIPS 2025) + COPRA (COLM 2024) + DeepSeek-Prover (ICLR 2025) |
| **CapabilityDAG** | `capability_dag.py` | Voyager (NeurIPS 2023) + FunSearch (Nature 2024) — self-growing knowledge graph |
| **TheoryGraph** | `theoretical_reasoning.py` | Cross-domain scientific reasoning with multi-modal verification |

<details>
<summary>Additional techniques</summary>

- **DSPy/OPRO prompt optimisation** (`prompt_optimizer.py`) — automatic prompt self-improvement
- **Cross-project RAG** (`rag_retrieval.py`) — BM25+TF-IDF hybrid code retrieval
- **Formal verification** (`formal_verify.py`) — multi-level linting and type checking
- **Conditional debate** (`agent_debate.py`) — reward-guided multi-agent architecture debate (ICLR 2025)
- **RedCode security scanning** (`security_scan.py`) — pattern matching + LLM vulnerability analysis (NeurIPS 2024)
- **Evolution engine** (`evolution.py`) — MAP-Elites cross-project workflow self-improvement

</details>

---

## Requirements

- **Python 3.11+** — [python.org](https://python.org)
- **At least one LLM API key** — [Anthropic](https://console.anthropic.com/) / [OpenAI](https://platform.openai.com/api-keys) / [Google](https://aistudio.google.com/apikey)
- **Git** (recommended) — for worktree isolation
- **Docker** (optional) — for sandbox execution

---

## License

MIT
