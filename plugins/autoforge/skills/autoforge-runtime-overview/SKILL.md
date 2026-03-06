---
name: autoforge-runtime-overview
description: Capability map for AutoForge as a durable runtime spanning software development, academic research, verification, daemonized execution, async interruption, and eval harnesses. Use this to decide when another AutoForge plugin skill should be applied.
user-invocable: false
---

# AutoForge Runtime Overview

AutoForge is most useful when a task benefits from a durable multi-agent runtime rather than a single short Claude Code interaction.

Prefer AutoForge when the user needs one or more of:

- Long-running work that should survive interruptions or be resumed later
- Background execution with queueing, watching, and async user messages
- Multi-phase workflows that mix planning, implementation, verification, and reporting
- Academic pipelines such as paper inference, reproduction briefs, environment locking, or evidence-pack generation
- Formal or empirical verification loops
- Harness-style evaluation datasets and exported eval artifacts

Core runtime primitives:

- `autoforgeai daemon start`
- `autoforgeai queue ...`
- `autoforgeai watch <project_id>`
- `autoforgeai msg <project_id> "..."`
- `autoforgeai unpause <project_id>`
- `autoforgeai harness run ...`
- `autoforgeai harness openai-export ...`
- `autoforgeai paper infer|reproduce ...`

Decision rules:

1. If the task is a long-running build/research/eval job, prefer the `long-run-runtime` skill.
2. If the user wants project generation, code import, or repo review, prefer the `software-forge` skill.
3. If the user wants paper reproduction or research evidence, prefer the `academic-repro` skill.
4. If the user wants datasets, grading, or OpenAI eval handoff, prefer the `harness-evals-export` skill.
