# Claude Plugin Distribution

## Plugin vs marketplace

- `plugin`: the installable Claude Code package under `plugins/autoforge`
- `marketplace`: the distribution catalog under `.claude-plugin/marketplace.json`

The plugin is the thing users install. The marketplace is the thing Claude Code reads in order to discover the plugin.

## Current setup

- Marketplace root: `.claude-plugin/marketplace.json`
- Plugin root: `plugins/autoforge`
- Auto-enable hint for repo users: `.claude/settings.json`

This repository can already be used as a GitHub marketplace:

```text
/plugin marketplace add AlyciaBHZ/autoforge
/plugin install autoforge@autoforge
```

## Official Anthropic marketplace

Claude Code documents official marketplace submission through in-app forms:

- `https://claude.ai/settings/plugins/submit`
- `https://platform.claude.com/plugins/submit`

Reference:

- Plugins: `https://docs.claude.com/en/docs/claude-code/plugins`
- Plugin marketplaces: `https://docs.claude.com/en/docs/claude-code/plugin-marketplaces`
- Plugins reference: `https://docs.claude.com/en/docs/claude-code/plugins-reference`

## Promotion strategy

For this repository, the best immediate path is:

1. Ship the plugin package in-repo
2. Ship the marketplace manifest in-repo
3. Auto-suggest the marketplace/plugin to Claude Code users via `.claude/settings.json`
4. Submit the same plugin to the official marketplace once screenshots, description, and maintainer details are ready
