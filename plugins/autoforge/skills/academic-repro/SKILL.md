---
name: academic-repro
description: Run AutoForge as an academic research workflow engine for paper inference, reproduction planning, evidence-pack generation, and long-running research jobs.
---

# Academic Reproduction with AutoForge

Use this skill when the user wants to:

- infer likely papers from a research goal
- reproduce a paper or build a reproduction brief
- extract claims, datasets, metrics, and environment assumptions
- queue a long-running research workflow and monitor it over time

Preferred workflow:

1. If needed, start the durable runtime:
   - `autoforgeai daemon start`
2. For paper discovery or reproduction artifacts:
   - `autoforgeai paper infer "<goal>"`
   - `autoforgeai paper reproduce "<goal>" --top-k 5 --pick 1`
3. For long-running work, prefer queue/watch over one-shot execution:
   - `autoforgeai queue "<research objective>"`
   - `autoforgeai watch <project_id>`
   - `autoforgeai msg <project_id> "<new constraint or direction>"`
4. Preserve and summarize the resulting artifacts:
   - `candidate.json`
   - `reproduction_brief.md`
   - `generation_prompt.txt`
   - `paper_signals.json`
   - `verification_plan.json`
   - `environment_spec.json`
   - `run_manifest.json`
   - `repro_report.json`

Operational guidance:

- Use the daemon path when the research workflow may take more than one interactive Claude turn.
- Prefer evidence and artifact paths over vague summaries.
- Call out whether the result is reproduced, partially reproduced, or not reproduced.
- If the user wants interactive correction mid-run, use `msg` or `unpause` rather than restarting from scratch.
