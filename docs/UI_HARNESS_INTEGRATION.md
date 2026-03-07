# UI Harness Integration

Status date: 2026-03-07

## Current implementation status

Phase 1 is now partially landed in the codebase.

Implemented:

- `ui_harness_enabled`
- `design_context_refs`
- repo-local `design_brief.json`
- repo-local `style_guide.json`
- repo-local `design_tokens.json`
- repo-local `component_inventory.json`
- repo-local `ui_judge_report.json`
- repo-local `ui_handoff.md`
- heuristic UI outcomes for responsive layout, design consistency, and
  accessibility contract checks

Still missing:

- Storybook generation
- visual baseline capture
- real `axe` reports
- token sync from external design systems
- Figma MCP ingestion

## Why this matters

AutoForge is already strong at:

- turning requirements into runnable code
- decomposing work into phases
- validating execution with kernel artifacts and harness-style checks

What it still lacks for app and web work is a durable answer to:

- where visual taste comes from
- how design intent is kept aligned with code
- how UI quality is judged beyond "it runs"

For frontend work, "looks coherent, feels deliberate, and survives iteration"
needs to become an engineering surface, not an afterthought.

My recommendation is:

- do not create a fourth top-level profile just for design
- add a `UI harness` overlay on top of `development`
- let it feed design context into generation and add visual/a11y/polish judges

## Market landscape

### 1. Design context into code

These tools reduce "AI guessed the UI" failure modes.

| Tool | What it helps with | Best fit |
| --- | --- | --- |
| Figma Dev Mode + MCP | Pull design context directly into coding agents | Product teams already using Figma |
| Builder DSI MCP | Expose component docs, tokens, and design-system context to AI | Teams with mature component systems |
| Builder Figma Plugin / Visual Copilot | Turn Figma designs into responsive content/code faster | Figma-heavy teams that want import/export workflows |

### 2. Style and concept generation

These tools help with the "taste gap" before code is finalized.

| Tool | What it helps with | Best fit |
| --- | --- | --- |
| v0 Design Mode + design systems | Prompt-to-UI plus visual fine-tuning inside a code-oriented workflow | Code-first React/Next teams |
| Relume Style Guide / Site Builder | Sitemap, wireframe, and style-guide generation | Marketing sites, landing pages, Webflow/React handoff |
| Uizard | Fast multi-screen mockups, screenshot-to-editable UI, theme generation | PMs, non-designers, early product exploration |

### 3. Design system source of truth

These tools stop brand drift and random spacing/color decisions.

| Tool | What it helps with | Best fit |
| --- | --- | --- |
| Tokens Studio | Token authoring and synchronization across design/code | Teams managing multi-brand or multi-theme systems |
| Style Dictionary | Export tokens to CSS, JS, iOS, Android, etc. | Build pipelines that need cross-platform delivery |
| Figma Variables | Native token-like system for themes and modes | Teams centered on Figma workflows |

### 4. UI verification and sign-off

These tools make UI quality measurable instead of subjective-only.

| Tool | What it helps with | Best fit |
| --- | --- | --- |
| Storybook | Isolated component development and documentation | Componentized frontend work |
| Chromatic | Visual, interaction, and component accessibility regression checks | Teams needing reviewable UI baselines in CI |
| Playwright visual comparisons | Local screenshot baselines and page-level visual regression | Self-hosted or lower-cost regression checks |
| Polypane | Responsive, accessibility, contrast, and emulation debugging | Designers/devs doing polish and responsive QA |
| axe / Deque | Accessibility linting and automated issue detection | Web and mobile compliance workflows |

## Recommended stack for AutoForge

### Tier 1: highest leverage

This is the stack I would integrate first.

1. `Figma Dev Mode MCP` for design-context ingest when a design file exists.
2. `Tokens Studio + Style Dictionary` for design tokens into code artifacts.
3. `Storybook + Chromatic` for component-level UI regression and sign-off.
4. `Playwright` for page-level visual baselines in local/offline workflows.
5. `axe` checks as part of the UI verification contract.

### Tier 2: situational but valuable

1. `v0` for fast React-first UI ideation and local design tweaks.
2. `Builder Visual Copilot / DSI MCP` for Figma-heavy enterprise teams with an existing component system.
3. `Relume` for marketing-site structure and style-guide exploration.
4. `Uizard` for early low-fidelity ideation when product/design maturity is still low.
5. `Polypane` for manual polish, responsive debugging, and accessibility spot checks.

## How to integrate this into AutoForge

## Principle

UI should be treated like research:

- gather context
- generate a structured plan
- build artifacts
- run judges
- keep evidence

But UI should remain an overlay on `development`, not a separate top-level
profile.

## Proposed model: `ui_harness` overlay

Apply `ui_harness` when:

- the target is a web app, site, or mobile-style frontend
- the user provides a design link, screenshot set, style reference, or asks for strong UI quality

### Overlay phases

1. `intent_capture`
   - extract target audience, tone, device emphasis, platform, references
2. `design_context_ingest`
   - Figma MCP, screenshots, brand assets, token bundles, existing app styles
3. `style_system_build`
   - palette, typography, spacing scale, radii, shadows, motion rules, layout density
4. `component_plan`
   - map design system to actual component primitives and page sections
5. `ui_build`
   - generate components/pages using the style system and component constraints
6. `ui_review`
   - visual diff, responsive diff, accessibility checks, consistency checks
7. `ui_handoff`
   - stories, tokens, baselines, screenshots, sign-off notes

## Proposed kernel artifacts

These should be repo-local artifacts, just like existing kernel outputs.

- `design_brief.json`
- `design_context_manifest.json`
- `style_guide.json`
- `design_tokens.json`
- `component_inventory.json`
- `story_manifest.json`
- `visual_baseline_manifest.json`
- `a11y_report.json`
- `ui_judge_report.json`
- `ui_handoff.md`

## Proposed success contract additions

For frontend-capable development runs, success should not stop at:

- repo runnable
- tests pass

It should also gate on UI-specific outcomes when `ui_harness` is active:

- `responsive_layout_pass`
- `visual_regression_pass`
- `a11y_contract_pass`
- `design_consistency_pass`

## Proposed UI judges

These should be explicit judges, not vague prompts.

### Objective judges

- accessibility: axe/Chromatic accessibility tests
- responsive layout: viewport matrix screenshot checks
- visual regression: Storybook or Playwright baselines
- token consistency: verify colors/spacing/fonts map to allowed token set
- interaction sanity: key flows render and behave correctly

### Heuristic judges

These are still LLM- or rubric-assisted, but structured.

- hierarchy clarity
- spacing rhythm
- typography consistency
- density balance
- component reuse discipline
- novelty vs template-likeness

Heuristic outputs should be scored and justified, not free-form.

## What AutoForge should implement first

### Phase 1

- add `ui_harness_enabled`
- add `design_context_refs` input support
- emit `design_brief.json` and `style_guide.json`
- add screenshot-set artifact support

### Phase 2

- generate Storybook stories for frontend outputs
- add Playwright screenshot baselines
- add `ui_judge_report.json`
- add `axe` results into verification artifacts

### Phase 3

- add Figma MCP adapter
- add token sync pipeline: `Figma/Tokens Studio -> Style Dictionary -> CSS variables`
- add Chromatic integration as an optional hosted review backend
- add compare view across runs for UI regressions

## What I would not do

- do not rely on one AI generator to "have taste"
- do not mix raw visual prompts straight into build without a style-system artifact
- do not make UI approval fully subjective
- do not create frontend code without component stories or screenshot baselines

## Short recommendation

If you want the fastest practical path:

1. make `UI harness` a `development` overlay
2. use Figma MCP when design files exist
3. use tokens as the design source of truth
4. require Storybook + screenshot baselines for generated frontend code
5. use Chromatic or Playwright + axe as the merge gate

That combination is the cleanest bridge between AI generation, design fidelity,
and engineering-grade verification.

## Sources

- Figma Dev Mode: https://www.figma.com/dev-mode/
- Figma MCP server guide: https://help.figma.com/hc/en-us/articles/32132100833559-Guide-to-the-Dev-Mode-MCP-Server
- Figma Make: https://www.figma.com/solutions/ai-to-do-app-builder/
- v0 Design mode: https://chat.v0.dev/docs/design-mode
- v0 Design systems: https://v0.dev/docs/design-systems
- Vercel v0 announcement: https://vercel.com/blog/announcing-v0-generative-ui
- Builder Figma to Code / Visual Copilot: https://www.builder.io/figma-to-code
- Builder import from Figma: https://site.builder.io/c/docs/import-from-figma
- Builder DSI MCP: https://www.builder.io/c/docs/builder-dsi-mcp
- Relume Style Guide: https://www.relume.io/style-guide
- Relume Site Builder: https://www.relume.io/
- Uizard AI UI design: https://uizard.io/ai-design/
- Tokens Studio: https://tokens.studio/
- Tokens Studio docs: https://documentation.tokens.studio/
- Style Dictionary: https://styledictionary.com/
- Storybook: https://storybook.js.org/
- Chromatic UI tests: https://www.chromatic.com/docs/test
- Chromatic accessibility: https://www.chromatic.com/docs/accessibility
- Playwright visual comparisons: https://playwright.dev/docs/test-snapshots
- Polypane: https://polypane.app/
- Polypane accessibility panel: https://polypane.app/docs/accessibility-panel/
- Deque docs / axe: https://docs.deque.com/
