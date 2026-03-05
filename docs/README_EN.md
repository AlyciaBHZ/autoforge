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

- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [LLM Provider Configuration](#llm-provider-configuration)
  - [Requirements](#requirements)
- [Academic & Research Capabilities](#academic--research-capabilities)
  - [End-to-End Article Reasoning](#end-to-end-article-reasoning)
  - [Formal Verification & Theorem Proving](#formal-verification--theorem-proving)
  - [Autonomous Research Discovery](#autonomous-research-discovery)
  - [Full Paper Pipeline](#full-paper-pipeline)
  - [Key Techniques & Inspirations](#key-techniques--inspirations)
- [Engineering Capabilities](#engineering-capabilities)
  - [Architecture](#architecture)
  - [Usage](#usage)
  - [Intelligent Engines](#intelligent-engines)

---

## Quick Start

### Installation

```bash
pip install forgeai                           # Install from PyPI
forgeai                                       # Start interactive session
```

On first launch, the setup wizard guides you through API key configuration (Anthropic / OpenAI / Google — any one), GitHub environment, and operating mode. Then you're dropped into a session where you describe a project and it gets built.

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
<summary>Install from source (developers)</summary>

```bash
git clone https://github.com/AlyciaBHZ/autoforge.git
cd autoforge
pip install -e ".[all]"
```

</details>

### LLM Provider Configuration

| Provider | Environment Variable | Strong Models | Fast Models |
|----------|---------------------|---------------|-------------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude Opus 4.6 | Claude Sonnet 4.5 |
| **OpenAI** | `OPENAI_API_KEY` | Codex 5.3, o3, GPT-4o | o4-mini, GPT-4o-mini |
| **Google** | `GOOGLE_API_KEY` | Gemini 2.5 Pro | Gemini 2.5 Flash, Gemini 2.0 Flash |

**Auth methods:** API Key, Codex OAuth (browser login, uses ChatGPT subscription), Device Code (headless/SSH), OAuth2 Client Credentials, Azure/LiteLLM Bearer Token, Google ADC/Service Account, AWS Bedrock, and Google Vertex AI are all supported.

Keys can also be stored in `~/.autoforge/config.toml`. Cross-provider example:

```bash
export FORGE_MODEL_STRONG=o3              # Strong model from OpenAI
export FORGE_MODEL_FAST=gemini-2.5-flash  # Fast model from Google
```

### Requirements

- **Python 3.11+** — [python.org](https://python.org)
- **At least one LLM API key** — [Anthropic](https://console.anthropic.com/) / [OpenAI](https://platform.openai.com/api-keys) / [Google](https://aistudio.google.com/apikey)
- **Git** (recommended) — for worktree isolation
- **Docker** (optional) — for sandbox execution
- **Lean 4** (optional) — for formal theorem proving

---

## Academic & Research Capabilities

AutoForge includes a complete academic research pipeline that serves as an AI-powered autonomous research assistant. It covers the full workflow from paper reading, formal verification, and autonomous theorem discovery to paper writing — from ingesting a paper to producing a new one, fully automated.

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
| [Paper Reproduction](../autoforge/engine/paper_repro.py) | Goal inference → signal extraction → code generation → execution → metric comparison → report | OpenReview API integration |
| [VLM Figure Analysis](../autoforge/engine/vlm_figure.py) | Figure extraction → visual analysis → data extraction → reproduction → verification | Vision-Language Models |
| [Symbolic Computation](../autoforge/engine/symbolic_compute.py) | SymPy/SageMath integration, LaTeX↔SymPy bidirectional conversion | Algebraic identity verification + limits/series |
| [Peer Review Simulation](../autoforge/engine/peer_review.py) | Multi-reviewer + author rebuttal + meta-review + iterative revision | 6 specialised reviewer roles |

### Key Techniques & Inspirations

| Technique | Source | Key Innovation |
|-----------|--------|----------------|
| **GRPO Verifiable Rewards** | DeepSeek-Prover-V2 (2025, 88.9% miniF2F) | Group-relative advantage replaces PPO critic, verifiable reward training |
| **Interleaved Reasoning Pattern** | Kimina-Prover (2025, 80.7% miniF2F) | Informal + formal interleaved single-pass generation |
| **Pantograph REPL** | TACAS 2025 | Incremental tactic application, 10x+ compilation speedup |
| **DPO Tactic Preference** | BFS-Prover-V2 (72.95% miniF2F) | Direct preference optimization without reward model |
| **Elo Hypothesis Ranking** | Google AI Co-Scientist (2025) | Pairwise competition for dynamic hypothesis ranking |
| **Hypergraph Knowledge Representation** | SciAgents + Hypergraph KG (2025) | N-ary relations replace binary relations |
| **Darwin Self-Rewriting** | Darwin Gödel Machine (2025) | Evolutionary self-rewriting agent constitutions |
| **Scaffolded Progressive RL** | Scaf-GRPO (2025, 44.3%↑ AIME) | Tiered hints + progressive scaffold removal |
| **PUCT-MCTS Proof Search** | AlphaProof (DeepMind, 2024) | AlphaZero formula adapted to tactic space search |
| **Recursive Decomposition** | HILBERT (NeurIPS 2025) | 4-component: informal reasoner + prover + verifier + retriever |
| **Self-Play Conjecturing** | STP (ICML 2025, 28.5% LeanWorkbook) | Dual-agent + Bayesian calibration targeting 50% success sweet spot |
| **Dense Premise Retrieval** | ReProver / LeanDojo (NeurIPS 2023) | Dense embeddings + FAISS replacing Jaccard overlap |
| **Curriculum Learning** | LeanAgent (ICLR 2025) | Complexity-ordered proving with positive transfer |
| **Closed-Loop Experiments** | AI Scientist v2 (2025) | Hypothesis → code → execute → analyse → ablation → iterate loop |
| **Process Reward Model** | CodePRM (2024) | Step-level code quality, not just outcome-level |
| **Verbal RL** | Reflexion (NeurIPS 2023) | Linguistic memory + failure pattern avoidance |
| **Self-Growing Knowledge Graph** | CapabilityDAG (internal) | Cross-project capability accumulation, community-mergeable |

---

## Engineering Capabilities

AutoForge is also a full-stack code generation engine — 6 specialised AI agents collaborate through a 5-phase pipeline to turn a natural-language description into a complete codebase.

### Architecture

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

### Usage

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

### Intelligent Engines

AutoForge includes multiple intelligent engines that work together across the code generation pipeline:

- **MCTS Search Tree** — Architecture exploration with execution-feedback-driven thought refinement
- **Natural-Language Gradient Feedback** — Agents improve each other's output via text "backpropagation"
- **Process Reward Model** — Step-level code quality evaluation, not just final results
- **Adaptive Compute Allocation** — Dynamically adjusts reasoning depth based on task difficulty
- **Verbal Reinforcement Learning** — Extracts lessons from failures to avoid known mistakes on retry
- **Block-Level Fault Localisation** — Pinpoints code defects to the block level
- **Function-Level Task Decomposition** — Breaks complex requirements into independently verifiable subtasks
- **Speculative Pipeline** — Overlapping phase execution for faster builds
- **Darwin Self-Rewriting** — Evolutionary self-rewriting agent constitutions and workflows

---

## License

MIT
