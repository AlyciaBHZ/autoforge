---
name: software-forge
description: Use AutoForge for multi-agent software generation, repo import, review, queue-based builds, and durable delivery workflows.
---

# Software Forge

Use this skill when the user wants:

- a new software project generated from natural language
- an existing project imported and improved
- an existing repo reviewed by a multi-agent workflow
- a durable queue/watch/resume loop instead of a single transient coding turn

Preferred commands:

- Generate: `autoforgeai generate "<project description>"`
- Review: `autoforgeai review <path>`
- Import: `autoforgeai import <path> --enhance "<changes>"`
- Queue for background build: `autoforgeai queue "<project description>" --wait --tail`
- Watch an existing run: `autoforgeai watch <project_id> --tail`
- Send async direction: `autoforgeai msg <project_id> "<instruction>"`

Use the daemon runtime when:

- the user wants background execution
- the user may intervene asynchronously
- the work should be resumed after pauses or checkpoints
- the output should remain observable through progress updates and logs

When handing off results, always include:

- project id or workspace path
- completion or paused status
- deploy guide path if present
- next command to continue watching or resuming
