---
name: harness-evals-export
description: Run AutoForge harness datasets and export OpenAI-friendly eval bundles from datasets or completed harness runs.
---

# Harness and Eval Export

Use this skill when the user wants benchmarking, dataset runs, or eval handoff artifacts.

AutoForge supports:

- local harness dataset execution
- deterministic environment setup and traces
- export of datasets or completed runs as OpenAI-friendly eval bundles

Preferred commands:

- Run a dataset:
  - `autoforgeai harness run <dataset.jsonl>`
- Prewarm referenced images:
  - `autoforgeai harness prewarm <dataset.jsonl>`
- Export a dataset or run to an eval bundle:
  - `autoforgeai harness openai-export <dataset-or-run-path>`

Expected exported artifacts:

- `items.jsonl`
- `item_schema.json`
- `bundle_manifest.json`

When a harness run has already completed, expect AutoForge to emit:

- `<run_dir>/openai_eval_bundle/`

Always surface:

- source dataset or run directory
- exported bundle path
- whether traces, judge results, and raw case metadata are preserved
