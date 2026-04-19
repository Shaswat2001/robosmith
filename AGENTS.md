# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

RoboSmith is a Python robotics toolchain for turning natural-language robot
tasks into trained policies, plus tools for inspecting and integrating existing
robotics artifacts.

The main code lives in `robosmith/`. The documentation site lives in
`robosmith-docs/` and uses Astro Starlight.

Important areas:

- `robosmith/cmd/`: Typer CLI entry points.
- `robosmith/agent/`: LangGraph workflows, graph state, and LLM-backed agents.
- `robosmith/stages/`: deterministic pipeline stage implementations.
- `robosmith/envs/`: environment registry, adapters, and reward wrapper.
- `robosmith/trainers/`: training backend interfaces and implementations.
- `robosmith/inspect/`: artifact inspection and compatibility models.
- `robosmith/generators/`: adapter/wrapper generation.
- `tests/`: pytest suite.
- `configs/`: example configuration and environment registry.

## Development Commands

Use the smallest relevant verification command for the change.

```bash
pytest
pytest tests/test_agent.py
pytest tests/test_config.py
ruff check robosmith tests
```

Docs:

```bash
cd robosmith-docs
npm run check
npm run build
```

CLI smoke checks:

```bash
robosmith --help
robosmith envs
robosmith config
```

## Coding Guidelines

- Prefer existing patterns over new abstractions.
- Keep optional robotics dependencies lazy. Import heavy simulator, training, or
  Hub packages inside functions or adapters when possible.
- Preserve clear failure messages for missing optional dependencies.
- Keep Typer CLI output human-readable with Rich, and keep `--json` output
  machine-readable.
- Use Pydantic models and typed dataclasses already present in the codebase
  instead of ad hoc dictionaries when crossing module boundaries.
- Do not hide important workflow behavior inside prompts. LangGraph nodes,
  routing conditions, and state fields should make the flow inspectable.
- Add or update focused tests when changing graph routing, CLI behavior,
  config parsing, registries, or generated-code contracts.

## Agentic Workflow Notes

`robosmith run` is a LangGraph workflow built from
`robosmith.agent.graphs.run.pipeline`. It uses `PipelineState` from
`robosmith.agent.state` and persists checkpoints under the run directory.

`robosmith auto integrate` is built in
`robosmith.agent.graphs.auto_integrate` and uses `IntegrateState`.

LLM-facing logic should usually go through:

- `BaseAgent` for LiteLLM calls, retries, JSON parsing, and usage tracking.
- `RewardAgent` for reward-code generation and evolution.
- `DecisionAgent` for accept/refine/switch decisions after evaluation.

When adding graph nodes:

- Return partial state dictionaries; do not mutate state in place.
- Include concise `steps_log` entries.
- Set `status` and `status_message` on recoverable failures.
- Keep generated artifacts in the run/output directory and store paths in state.

## Docs Cleanup

The active docs source is `robosmith-docs/`. Do not add new MkDocs pages or rely
on the old top-level `docs/` tree. If documentation is needed, add it under
`robosmith-docs/src/content/docs/` and wire it through
`robosmith-docs/astro.config.mjs`.

The old `mkdocs.yml` and top-level `docs/assets/` files are obsolete.

## Safety

- Do not commit secrets from `.env.local` or local run artifacts.
- Treat `robosmith_runs/`, `site/`, `robosmith-docs/dist/`, caches, and local
  environment files as generated or local-only.
- Avoid destructive git commands unless the user explicitly asks for them.
