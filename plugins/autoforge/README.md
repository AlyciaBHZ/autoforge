# AutoForge Claude Code Plugin

AutoForge packages long-running multi-agent workflows for Claude Code as a shareable plugin.

## What this plugin exposes

- Research and paper reproduction workflows
- Software project generation and review workflows
- Durable daemon-based queue/watch/msg flows
- Harness execution and OpenAI eval bundle export

## Install from this repository marketplace

Add the repository as a marketplace:

```text
/plugin marketplace add AlyciaBHZ/autoforge
```

Install the plugin:

```text
/plugin install autoforge@autoforge
```

After installation, Claude Code can invoke the packaged AutoForge skills such as:

- `/autoforge:academic-repro`
- `/autoforge:software-forge`
- `/autoforge:long-run-runtime`
- `/autoforge:harness-evals-export`

## Official marketplace

This repository contains a complete plugin package and marketplace manifest. To submit the plugin to Anthropic's official marketplace, use the in-app submission forms documented by Claude Code:

- `https://claude.ai/settings/plugins/submit`
- `https://platform.claude.com/plugins/submit`

## Notes

- `plugin` = installable package (`plugins/autoforge`)
- `marketplace` = catalog/distribution manifest (`.claude-plugin/marketplace.json`)
- This repository also includes `.claude/settings.json` so Claude Code users who trust the repo can be prompted to enable the AutoForge marketplace and plugin automatically.
