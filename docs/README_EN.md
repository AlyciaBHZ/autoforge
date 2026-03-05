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
[![Tests](https://img.shields.io/badge/tests-217%20checks-brightgreen.svg)](../tests/)

[中文](../README.md) | [Developer Docs](../CLAUDE.md)

---

## Quick Start

```bash
git clone https://github.com/AlyciaBHZ/autoforge.git
cd autoforge
pip install -e .                              # Install from source
forgeai                                       # Start interactive session
```

On first launch, the setup wizard guides you through API key configuration (Anthropic / OpenAI / Google — any one), GitHub environment, and operating mode. Then you're dropped into a session where you describe a project and it gets built.

<details>
<summary>Optional dependencies</summary>

```bash
pip install -e ".[openai]"       # OpenAI support
pip install -e ".[google]"       # Google Gemini support
pip install -e ".[search]"       # Web search capabilities
pip install -e ".[channels]"     # Telegram / Webhook channels
pip install -e ".[all]"          # Install everything
```

</details>

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
# Interactive session (recommended)
forgeai                           # Select mode → describe project → build

# Direct generation
forgeai generate "REST API for a bookstore with JWT auth"
forgeai generate "Landing page for a SaaS product" --budget 3.00

# Manage runs
forgeai status                    # Show all projects
forgeai resume                    # Resume interrupted run

# Daemon mode (24/7 background service)
forgeai daemon start
forgeai queue "Blog with markdown support"
forgeai projects
forgeai deploy <project_id>
```

<details>
<summary>Cost estimates</summary>

| Complexity | Example | Estimated Cost |
|------------|---------|:--------------:|
| Simple | Todo app, landing page | $2–3 |
| Medium | Blog system, booking platform | $4–6 |
| Complex | E-commerce MVP, multi-role platform | $7–10 |

Default budget cap: $10. Override with `--budget`.

</details>

---

## Intelligent Engines

AutoForge includes multiple intelligent engines that work together across the code generation pipeline:

- **MCTS Search Tree** — Architecture exploration with execution-feedback-driven thought refinement
- **Natural-Language Gradient Feedback** — Agents improve each other's output via text "backpropagation"
- **Process Reward Model** — Step-level code quality evaluation, not just final results
- **Adaptive Compute Allocation** — Dynamically adjusts reasoning depth based on task difficulty
- **Verbal Reinforcement Learning** — Extracts lessons from failures to avoid known mistakes on retry
- **Block-Level Fault Localisation** — Pinpoints code defects to the block level
- **Function-Level Task Decomposition** — Breaks complex requirements into independently verifiable subtasks
- **Speculative Pipeline** — Overlapping phase execution for faster builds

<details>
<summary>Advanced reasoning capabilities</summary>

- **Theorem Proving** — Lean 4 formal proofs with MCTS tactic search and auto-repair
- **Multi-Prover Verification** — Cross-verification via Coq, Isabelle, TLA+, Z3/SMT, Dafny
- **Cross-Domain Scientific Reasoning** — Theory graph construction, multi-modal verification, theory evolution
- **Self-Growing Knowledge Graph** — Cross-project capability accumulation, community-mergeable
- **Prompt Self-Optimisation** — Automatic A/B testing and evolutionary prompt improvement
- **Cross-Project RAG Retrieval** — BM25+TF-IDF hybrid code retrieval from past projects
- **Multi-Agent Debate** — Reward-guided architecture debate from multiple perspectives
- **Security Vulnerability Scanning** — Pattern matching + LLM deep analysis
- **Workflow Self-Evolution** — Cross-project strategy evolution based on historical fitness

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
