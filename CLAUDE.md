# CLAUDE.md

Claude Code and other agentic tools should follow the repository guidance in
`AGENTS.md`.

High-signal reminders:

- This is a Python package with a Typer CLI in `robosmith/cmd/` and agentic
  LangGraph workflows in `robosmith/agent/`.
- Keep optional robotics dependencies lazy and preserve helpful missing-package
  errors.
- Use existing Pydantic models, graph state types, registries, and trainer/env
  interfaces instead of inventing parallel contracts.
- Active docs live in `robosmith-docs/` using Astro Starlight. The old MkDocs
  setup is obsolete.
- Verify Python changes with focused `pytest` runs, and docs changes with
  `cd robosmith-docs && npm run check && npm run build`.
- Do not commit `.env.local`, generated run artifacts, docs builds, caches, or
  local tool settings.
