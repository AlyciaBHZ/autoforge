```
     _         _        _____
    / \  _   _| |_ ___ |  ___|__  _ __ __ _  ___
   / _ \| | | | __/ _ \| |_ / _ \| '__/ _` |/ _ \
  / ___ \ |_| | || (_) |  _| (_) | | | (_| |  __/
 /_/   \_\__,_|\__\___/|_|  \___/|_|  \__, |\___|
                                       |___/
```

**AI Multi-Agent Framework — Autonomous Research Reasoning · Formal Proving · Full-Stack Project Generation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)
[![Tests](https://img.shields.io/badge/tests-217%20checks-brightgreen.svg)](../tests/)
[![Engines](https://img.shields.io/badge/engines-47%20modules-orange.svg)](../autoforge/engine/)

[中文](../README.md) | [Developer Docs](../CLAUDE.md)

---

## Table of Contents

- [Installation & Configuration](#installation--configuration)
  - [Installation](#installation)
  - [First-Run Setup Wizard](#first-run-setup-wizard)
  - [Supported LLM Providers](#supported-llm-providers)
  - [Requirements](#requirements)
- [Three Operating Modes](#three-operating-modes)
- [Academic & Research Capabilities](#academic--research-capabilities)
  - [End-to-End Article Reasoning](#end-to-end-article-reasoning)
  - [Formal Verification & Theorem Proving](#formal-verification--theorem-proving)
  - [Autonomous Research Discovery](#autonomous-research-discovery)
  - [Full Paper Pipeline](#full-paper-pipeline)
  - [Paper Reproduction Pipeline](#paper-reproduction-pipeline)
  - [Key Techniques & Inspirations](#key-techniques--inspirations)
- [Engineering Capabilities](#engineering-capabilities)
  - [5-Phase Pipeline](#5-phase-pipeline)
  - [6-Agent Collaboration](#6-agent-collaboration)
  - [Intelligent Engines](#intelligent-engines)
- [CLI Reference](#cli-reference)
- [Daemon Mode](#daemon-mode)

---

## Installation & Configuration

### Installation

```bash
pip install forgeai          # Install from PyPI
forgeai                      # Launch (first run enters setup wizard automatically)
```

<details>
<summary>Optional dependencies</summary>

```bash
pip install "forgeai[openai]"    # OpenAI support
pip install "forgeai[google]"    # Google Gemini support
pip install "forgeai[search]"    # Web search capabilities
pip install "forgeai[channels]"  # Telegram / Webhook channels
pip install "forgeai[all]"       # Install everything
```

</details>

<details>
<summary>Install from source</summary>

```bash
git clone https://github.com/AlyciaBHZ/autoforge.git
cd autoforge
pip install -e ".[all]"
```

</details>

### First-Run Setup Wizard

Running `forgeai` for the first time launches an interactive setup wizard. Every step is optional (Ctrl+C to skip). Reconfigure anytime with `forgeai setup`:

```
Step 1 │ Configure LLM providers (optional)
       │   Select Anthropic / OpenAI / Google (multi-select)
       │   Choose auth method per provider (API Key, OAuth, Bedrock, Vertex AI, etc.)
       │   Pick strong model (for Director/Architect) and fast model (for Builder/Tester)
       │
Step 2 │ Budget limit (default $10)
       │
Step 3 │ Max parallel Builders (default 3, up to 8)
       │
Step 4 │ Docker sandbox (optional, for isolated builds)
       │
Step 5 │ GitHub environment
       │   Auto-detects git and gh CLI
       │   Optional auto-push to GitHub
```

Configuration is saved to `~/.autoforge/config.toml` and can be overridden via environment variables.

### Supported LLM Providers

| Provider | Environment Variable | Strong Models | Fast Models |
|----------|---------------------|---------------|-------------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude Opus 4.6 | Claude Sonnet 4.5 |
| **OpenAI** | `OPENAI_API_KEY` | Codex 5.3, o3, GPT-4o | o4-mini, GPT-4o-mini |
| **Google** | `GOOGLE_API_KEY` | Gemini 2.5 Pro | Gemini 2.5 Flash, Gemini 2.0 Flash |

**Supported auth methods:**

| Auth Method | Providers | Description |
|-------------|-----------|-------------|
| API Key | All | Simplest option, recommended for getting started |
| Codex OAuth | OpenAI | Browser login, uses ChatGPT subscription quota |
| Device Code | OpenAI | For headless/SSH environments |
| OAuth2 Client Credentials | Anthropic, OpenAI | Enterprise-grade |
| Bearer Token + Custom URL | Anthropic, OpenAI | Azure, LiteLLM proxies |
| Amazon Bedrock | Anthropic | AWS Profile / Access Key / Instance Role |
| Google Vertex AI | Anthropic | GCP Project + ADC |
| ADC / Service Account | Google | Google Cloud native auth |

Cross-provider model mixing is supported:

```bash
export FORGE_MODEL_STRONG=o3              # Strong model from OpenAI
export FORGE_MODEL_FAST=gemini-2.5-flash  # Fast model from Google
```

### Requirements

- **Python 3.11+** — [python.org](https://python.org)
- **At least one LLM API key** — [Anthropic](https://console.anthropic.com/) / [OpenAI](https://platform.openai.com/api-keys) / [Google](https://aistudio.google.com/apikey)
- **Git** (recommended) — for worktree isolation in parallel builds
- **Docker** (optional) — for sandbox execution
- **Lean 4** (optional) — for formal theorem proving

---

## Three Operating Modes

After configuration, `forgeai` enters an interactive session where you first select a mode:

```
? Select mode:
❯ Development — generate complete runnable projects
  Academic — scientific reasoning, theorem proving, theory evolution
  Verification — review & verify existing codebases
```

| Mode | Purpose | Supported Actions |
|------|---------|-------------------|
| **Development** | Full-stack project generation | Generate new projects, import & enhance existing ones |
| **Academic** | Research reasoning & theorem proving | Generate research projects, analyse existing codebases |
| **Verification** | Code review & verification | Review project quality, security, architecture |

Each mode allows further budget and parallelism configuration before you describe your task in natural language.

---

## Academic & Research Capabilities

AutoForge includes a complete academic research pipeline that serves as an AI-powered autonomous research assistant — from ingesting a paper to producing a new one, fully automated.

### End-to-End Article Reasoning

> Feed any paper → automatic parsing → TheoryGraph construction → claim verification → interleaved formal/informal reasoning → Lean 4 formalisation → autonomous discovery → Elo hypothesis ranking → reasoning extension → peer review → output new paper

Core orchestrator: [`article_reasoning.py`](../autoforge/engine/article_reasoning.py) — unified 8-phase pipeline

### Formal Verification & Theorem Proving

| Module | Capability | Method |
|--------|-----------|--------|
| [Lean 4 MCTS Proof Search](../autoforge/engine/provers/proof_search.py) | Monte Carlo tree search over tactic space | HILBERT recursive decomposition + COPRA + STP |
| [Lean Lake Integration](../autoforge/engine/provers/lean_lake.py) | Real Lean 4 compilation with Mathlib | Lake build system, 32 concept→import mappings |
| [Pantograph REPL](../autoforge/engine/provers/pantograph_repl.py) | Incremental tactic application, no full recompilation | TACAS 2025, machine-to-machine Lean 4 interaction, BFS/DFS search |
| [GRPO Verifiable Reward Training](../autoforge/engine/rl_proof_search.py) | Group Relative Policy Optimization + scaffolded progressive RL | DeepSeek-Prover-V2 (88.9% miniF2F) + Scaf-GRPO |
| [Kimina Interleaved Reasoning](../autoforge/engine/recursive_decomp_prover.py) | Informal-formal interleaved single-pass proof generation | Kimina-Prover (80.7% miniF2F) |
| [DPO Tactic Optimization](../autoforge/engine/proof_embedding.py) | Direct Preference Optimization, no reward model needed | BFS-Prover-V2 state-tactic DPO |
| [Multi-Prover Cross-Verification](../autoforge/engine/provers/multi_prover.py) | Coq, Isabelle, TLA+, Z3/SMT, Dafny | 6-backend parallel verification |
| [Dense Embedding Retrieval](../autoforge/engine/dense_retrieval.py) | Premise selection replacing Jaccard | ReProver/LeanDojo style + FAISS |
| [Proof Embedding Transfer](../autoforge/engine/proof_embedding.py) | Cross-domain tactic transfer learning | Vector memory bank + FAISS + experience tracking |
| [Standard Benchmarks](../autoforge/engine/benchmark_eval.py) | miniF2F / PutnamBench / LeanWorkbook / ProofNet | Unbiased Pass@k estimation |

### Autonomous Research Discovery

| Module | Capability | Method |
|--------|-----------|--------|
| [Autonomous Theorem Discovery](../autoforge/engine/autonomous_discovery.py) | Extract kernel → generate conjectures → novelty filter → depth evaluation | DomainContext templates + Thompson sampling |
| [Elo Hypothesis Tournament](../autoforge/engine/autonomous_discovery.py) | Pairwise hypothesis competition → Elo ranking → select best | Google AI Co-Scientist (2025) style |
| [Self-Play Conjecturing](../autoforge/engine/self_play_conjecture.py) | Dual-agent Conjecturer/Prover game | STP (ICML 2025) + Bayesian difficulty calibration |
| [Reasoning Extension](../autoforge/engine/reasoning_extension.py) | Minimal axiom kernel → iterative deep conclusions | Thompson sampling + publication-gate quality control |
| [Cross-Domain Scientific Reasoning](../autoforge/engine/theoretical_reasoning.py) | TheoryGraph + hypergraph n-ary relations + 12 reasoning strategies | SciAgents HyperEdge + 10 verification modes |
| [Structured World Model](../autoforge/engine/world_model.py) | Temporal query layer for TheoryGraph + cross-session persistence | Kosmos (2025) |
| [Curriculum Learning](../autoforge/engine/curriculum_learning.py) | Complexity-ordered proving + positive transfer tracking | LeanAgent (ICLR 2025) |

### Full Paper Pipeline

| Module | Capability | Method |
|--------|-----------|--------|
| [Closed-Loop Experiments](../autoforge/engine/experiment_loop.py) | Hypothesis → code → run → analyse → ablation → iterate | AI Scientist v2 |
| [Automated Paper Writing](../autoforge/engine/paper_writer.py) | LaTeX generation + BibTeX + figures + templates | NeurIPS/ICML/ICLR/ArXiv templates |
| [Literature Search & Analysis](../autoforge/engine/literature_search.py) | Citation graph traversal + SPECTER2 semantic search + full-text analysis + gap detection | Semantic Scholar API + arXiv |
| [VLM Figure Analysis](../autoforge/engine/vlm_figure.py) | Figure extraction → visual analysis → data extraction → reproduction → verification | Vision-Language Models |
| [Symbolic Computation](../autoforge/engine/symbolic_compute.py) | SymPy/SageMath integration, LaTeX↔SymPy bidirectional conversion | Algebraic identity verification + limits/series |
| [Peer Review Simulation](../autoforge/engine/peer_review.py) | Multi-reviewer + author rebuttal + meta-review + iterative revision | 6 specialised reviewer roles |

### Paper Reproduction Pipeline

AutoForge can infer relevant papers from a high-level research goal and build full reproduction plans:

```bash
forgeai paper infer "improve sample efficiency in offline RL"   # Infer relevant ICLR papers
forgeai paper benchmark                                         # Evaluate inference quality
forgeai paper reproduce "goal" --with-pdf --run-generate        # End-to-end reproduction
```

Pipeline: research goal → OpenReview paper retrieval → TF-IDF ranking → signal extraction → reproduction plan → optional auto-execution

### Key Techniques & Inspirations

| Technique | Source | Key Innovation |
|-----------|--------|----------------|
| **GRPO Verifiable Rewards** | DeepSeek-Prover-V2 (2025, 88.9% miniF2F) | Group-relative advantage replaces PPO critic |
| **Interleaved Reasoning** | Kimina-Prover (2025, 80.7% miniF2F) | Informal + formal interleaved single-pass |
| **Pantograph REPL** | TACAS 2025 | Incremental tactic application, 10x+ speedup |
| **DPO Tactic Preference** | BFS-Prover-V2 (72.95% miniF2F) | Direct preference optimization, no reward model |
| **Elo Hypothesis Ranking** | Google AI Co-Scientist (2025) | Pairwise competition for dynamic ranking |
| **Hypergraph Knowledge** | SciAgents + Hypergraph KG (2025) | N-ary relations replace binary |
| **Darwin Self-Rewriting** | Darwin Gödel Machine (2025) | Evolutionary self-rewriting agent constitutions |
| **Scaffolded Progressive RL** | Scaf-GRPO (2025, 44.3%↑ AIME) | Tiered hints + progressive scaffold removal |
| **PUCT-MCTS** | AlphaProof (DeepMind, 2024) | AlphaZero adapted to tactic space |
| **Recursive Decomposition** | HILBERT (NeurIPS 2025) | informal reasoner + prover + verifier + retriever |
| **Self-Play Conjecturing** | STP (ICML 2025, 28.5% LeanWorkbook) | Bayesian calibration at 50% sweet spot |
| **Dense Premise Retrieval** | ReProver / LeanDojo (NeurIPS 2023) | FAISS replacing Jaccard overlap |
| **Curriculum Learning** | LeanAgent (ICLR 2025) | Complexity-ordered lifelong learning |
| **Closed-Loop Experiments** | AI Scientist v2 (2025) | Hypothesis → code → execute → analyse → ablation → iterate |
| **Process Reward Model** | CodePRM (2024) | Step-level quality evaluation |
| **Verbal RL** | Reflexion (NeurIPS 2023) | Linguistic memory + failure pattern avoidance |

---

## Engineering Capabilities

AutoForge is also a full-stack code generation engine — 6 AI agents collaborate to turn a natural-language description into a complete runnable codebase.

### 5-Phase Pipeline

```
  "Build a todo app with auth"
           │
           ▼
  ┌────────────────────────────────────────────────┐
  │  SPEC       Director analyses requirements      │
  ├────────────────────────────────────────────────┤
  │  BUILD      Architect designs, Builders code    │
  │             Reviewer checks each task           │
  ├────────────────────────────────────────────────┤
  │  VERIFY     Tester runs build + tests           │
  │             Failures auto-generate fix tasks     │
  ├────────────────────────────────────────────────┤
  │  REFACTOR   Gardener optimises code quality     │
  ├────────────────────────────────────────────────┤
  │  DELIVER    README, structure, cost report      │
  └────────────────────────────────────────────────┘
           │
           ▼
     workspace/my-todo-app/
```

Two additional specialised pipelines:
- **Review pipeline:** SCAN → REVIEW → REFACTOR → REPORT
- **Import pipeline:** SCAN → REVIEW → ENHANCE → VERIFY → REFACTOR → DELIVER

### 6-Agent Collaboration

| Agent | Role | Model Tier |
|-------|------|------------|
| **Director** | Requirements & scoping | Strong |
| **Architect** | System design & task DAG | Strong |
| **Builder** | Code generation (parallelised, up to 8) | Fast |
| **Reviewer** | Code review & scoring | Fast |
| **Tester** | Build, test, auto-fix loop | Fast |
| **Gardener** | Refactoring & security fixes | Fast |

### Intelligent Engines

47 built-in engines work together across the build pipeline:

- **MCTS Search Tree** — Architecture exploration with execution-feedback-driven refinement
- **Natural-Language Gradient Feedback (EvoMAC)** — Agents improve each other via text backpropagation
- **Process Reward Model (CodePRM)** — Step-level code quality evaluation
- **Adaptive Compute Allocation** — Dynamically adjusts reasoning depth by task difficulty
- **Verbal RL (Reflexion)** — Extracts lessons from failures, avoids known mistakes on retry
- **Block-Level Fault Localisation (LDB)** — Pinpoints defects to the block level
- **Function-Level Decomposition** — Breaks complex requirements into independently verifiable subtasks
- **Speculative Pipeline** — Overlapping phase execution for faster builds
- **Darwin Self-Rewriting (SICA)** — Evolutionary self-rewriting agent constitutions
- **Security Scanning (RedCode)** — Pattern matching + LLM deep vulnerability analysis
- **Cross-Project RAG** — BM25+TF-IDF hybrid code retrieval across projects
- **Self-Growing Knowledge Graph (CapabilityDAG)** — Cross-project capability accumulation, community-mergeable

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

## CLI Reference

```bash
# Interactive (recommended)
forgeai                                        # Guided session

# Project generation
forgeai generate "REST API for a bookstore with JWT auth"
forgeai generate "Landing page for SaaS" --budget 3.00

# Code review
forgeai review ./my-project

# Import & enhance
forgeai import ./my-project --enhance "add dark mode"

# Run management
forgeai status                                 # Show all projects
forgeai resume                                 # Resume interrupted run
forgeai setup                                  # Reconfigure

# Paper reproduction
forgeai paper infer "research goal"            # Infer relevant papers
forgeai paper benchmark                        # Evaluate inference quality
forgeai paper reproduce "goal" --run-generate  # End-to-end reproduction
```

Global flags: `--budget`, `--agents`, `--model`, `--mode`, `--mobile`, `--tdd`, `--verbose`

---

## Daemon Mode

AutoForge can run as a 24/7 background service, accepting build requests via CLI, Telegram, or Webhook:

```bash
forgeai daemon start                           # Start daemon
forgeai daemon status                          # Check status
forgeai daemon stop                            # Stop

forgeai queue "Blog with markdown support"     # Queue a build
forgeai projects                               # List all projects
forgeai deploy <project_id>                    # Show deployment guide
```

Supports systemd (Linux) and launchd (macOS) service installation — see `services/` directory.

---

## License

MIT
