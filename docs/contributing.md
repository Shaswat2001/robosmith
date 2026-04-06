# Contributing

Thank you for your interest in contributing to RoboSmith. This guide covers development setup, testing, code style, and how to add new components.

## Development setup

```bash
git clone https://github.com/Shaswat2001/robosmith.git
cd robosmith
pip install -e ".[dev,sim,train]"
```

The `[dev]` extra installs pytest, ruff, mypy, and pre-commit hooks.

## Project structure

```
robosmith/
├── agents/              # LLM agents (reward, decision, base)
│   ├── base.py          # BaseAgent — LiteLLM wrapper with retry + JSON parsing
│   ├── reward_agent.py  # RewardAgent — generates and evolves reward functions
│   └── decision_agent.py# DecisionAgent — pipeline iteration decisions
├── envs/                # Environment registry, adapters, wrappers
│   ├── adapters/        # Framework-specific adapters (gymnasium, isaac_lab, etc.)
│   ├── registry.py      # YAML-driven environment catalog
│   ├── adapter_registry.py # Singleton adapter discovery and routing
│   ├── wrapper.py       # make_env() — the single entry point for env creation
│   └── reward_wrapper.py# ForgeRewardWrapper — injects custom rewards
├── stages/              # Pipeline stages (each is a standalone module)
│   ├── intake/          # Stage 1: natural language → TaskSpec
│   ├── scout/           # Stage 2: literature search via Semantic Scholar
│   ├── env_synthesis/   # Stage 3: task → environment matching
│   ├── reward_design/   # Stage 4: evolutionary reward function search
│   ├── training/        # Stage 5: RL training with backend abstraction
│   ├── evaluation/      # Stage 6: behavioral success detection
│   └── delivery/        # Stage 7: artifact packaging
├── trainers/            # Training backend abstractions
│   ├── base.py          # Trainer ABC, TrainingConfig, TrainingResult, Policy
│   ├── registry.py      # TrainerRegistry singleton
│   ├── selector.py      # PolicySelector — task-aware algorithm selection
│   ├── sb3_trainer.py   # Stable Baselines3 backend
│   ├── cleanrl_trainer.py# CleanRL (pure PyTorch PPO)
│   ├── rl_games_trainer.py# NVIDIA rl_games backend
│   ├── il_trainer.py    # Imitation learning (BC, DAgger)
│   └── offline_rl_trainer.py # Offline RL (TD3+BC, CQL, IQL)
├── config.py            # Pydantic models (TaskSpec, ForgeConfig, RunState, enums)
├── controller.py        # ForgeController — pipeline orchestrator
└── cli.py               # Typer CLI
```

## Running tests

```bash
# Full test suite
pytest tests/

# Specific test file
pytest tests/test_evaluation.py

# With coverage
pytest tests/ --cov=robosmith --cov-report=html

# Only fast tests (skip integration tests that need LLM or MuJoCo)
pytest tests/ -m "not integration"

# Verbose output
pytest tests/ -v
```

The test suite uses mocks for LLM calls and environment creation, so most tests run without API keys or simulation dependencies.

## Code style

We use `ruff` for linting and formatting:

```bash
# Check for issues
ruff check robosmith/

# Auto-fix issues
ruff check robosmith/ --fix

# Format code
ruff format robosmith/
```

Key style rules:

- Line length: 100 characters
- Python 3.11+ target
- Import sorting with isort (via ruff)
- Type annotations on all public functions
- Docstrings on all public classes and functions

## Type checking

```bash
mypy robosmith/
```

The project uses strict mypy settings. All public APIs should have complete type annotations.

## Adding a new training backend

1. Create `robosmith/trainers/my_trainer.py`:

```python
from robosmith.trainers.base import (
    Trainer, TrainingConfig, TrainingResult, LearningParadigm, Policy
)

class MyTrainer(Trainer):
    name = "my_trainer"
    paradigm = LearningParadigm.REINFORCEMENT_LEARNING
    algorithms = ["my_algo"]
    requires = ["my_package"]
    description = "My custom trainer"

    def train(self, config: TrainingConfig) -> TrainingResult:
        # Your training logic here
        ...

    def load_policy(self, path: Path) -> Policy:
        # Load a saved checkpoint
        ...
```

2. Register it in `robosmith/trainers/registry.py`:

```python
self._known_backends = {
    ...
    "my_trainer": ("robosmith.trainers.my_trainer", "MyTrainer"),
}
```

3. Add tests in `tests/test_my_trainer.py`

See [Custom Trainers](extending/trainers.md) for the full guide.

## Adding a new environment adapter

1. Create `robosmith/envs/adapters/my_adapter.py`:

```python
from robosmith.envs.adapters import EnvAdapter, EnvConfig

class MyEnvAdapter(EnvAdapter):
    name = "my_framework"
    frameworks = ["my_framework"]
    requires = ["my_package"]
    description = "My custom simulation framework"

    def make(self, env_id: str, config: EnvConfig | None = None) -> Any:
        ...

    def list_envs(self) -> list[str]:
        ...
```

2. Register it in `robosmith/envs/adapter_registry.py`

3. Add environment entries to `configs/env_registry.yaml`

4. Add tests in `tests/test_my_adapter.py`

See [Custom Environments](extending/environments.md) for the full guide.

## Adding a new pipeline stage

1. Create `robosmith/stages/my_stage.py` (or a package `robosmith/stages/my_stage/`) with a `run_my_stage()` function
2. Add the stage name to `STAGES` in `controller.py`
3. Add a `_stage_my_stage()` method to `ForgeController`
4. Add tests in `tests/test_my_stage.py`

Stages should follow these conventions:

- Have a single public entry point (`run_my_stage()`)
- Accept a `TaskSpec` and return a dataclass with results
- Write their results to `RunState` metadata
- Handle their own errors gracefully (log and continue, don't crash the pipeline)
- Be independently testable with mocked dependencies

## Commit guidelines

- Write clear commit messages that explain the "why", not just the "what"
- Keep commits focused — one logical change per commit
- Run `ruff check` and `pytest` before committing

## Submitting changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes with tests
4. Run `ruff check robosmith/` and `pytest tests/`
5. Commit and push
6. Open a pull request with a clear description of what and why

## Questions?

Open an issue on [GitHub](https://github.com/Shaswat2001/robosmith/issues) for bug reports, feature requests, or questions.
