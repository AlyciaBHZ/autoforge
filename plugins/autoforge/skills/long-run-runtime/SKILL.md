---
name: long-run-runtime
description: Operate AutoForge as a durable long-running agent runtime with queueing, watch mode, async messages, checkpoint replies, and resumable execution.
---

# Long-Run Runtime

Use this skill when the user needs a workflow that should continue beyond the current Claude session.

AutoForge supports:

- queue-based background execution
- periodic progress updates
- async user interference through messages
- checkpoint/bridge responses
- paused runs with later resume

Canonical runtime flow:

1. Start daemon if not already running:
   - `autoforgeai daemon start`
2. Queue work:
   - `autoforgeai queue "<objective>" --wait --tail`
3. Inspect and follow:
   - `autoforgeai projects`
   - `autoforgeai watch <project_id> --tail`
4. Interfere asynchronously:
   - `autoforgeai msg <project_id> "<updated guidance>"`
5. Resume if paused:
   - `autoforgeai unpause <project_id>`

Use this mode whenever the user cares about:

- not losing state
- seeing incremental output
- injecting late feedback
- having one runtime pattern for software, research, and verification tasks
