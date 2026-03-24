# Contributing

## Development Setup

```bash
git clone https://github.com/Shaswat2001/robosmith.git
cd robosmith
pip install -e ".[dev]"
```

## Running Tests

```bash
# Full test suite (140 tests)
pytest tests/

# Specific test file
pytest tests/test_evaluation.py

# With coverage
pytest tests/ --cov=robosmith --cov-report=html
```

## Code Style

We use `ruff` for linting and formatting:

```bash
ruff check robosmith/
ruff format robosmith/
```

## Project Structure

```
robosmith/
├── agents/          # LLM agents (reward, decision, base)
├── envs/            # Environment registry, adapters, wrappers
│   └── adapters/    # Framework-specific adapters
├── stages/          # Pipeline stages (intake → delivery)
├── trainers/        # Training backend abstractions
├── config.py        # Pydantic models (TaskSpec, ForgeConfig, RunState)
├── controller.py    # Pipeline orchestrator
└── cli.py           # Typer CLI
```

## Adding a New Training Backend

See [Custom Trainers](extending/trainers.md).

## Adding a New Environment Adapter

See [Custom Environments](extending/environments.md).

## Adding a New Pipeline Stage

1. Create `robosmith/stages/my_stage.py` with a `run_my_stage()` function
2. Add the stage name to `STAGES` in `controller.py`
3. Add a `_stage_my_stage()` method to `ForgeController`
4. Add tests in `tests/test_my_stage.py`
