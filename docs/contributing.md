# Contributing

Thank you for your interest in contributing to RoboSmith. This guide covers development setup, project structure, testing, and how to extend the system with new components.

## Development setup

```bash
git clone https://github.com/Shaswat2001/robosmith.git
cd robosmith
pip install -e ".[dev,sim,train]"
```

The `[dev]` extra installs pytest, ruff, mypy, and pre-commit hooks.

---

## Project structure

```
robosmith/
├── cmd/                          # CLI entry points
│   ├── robosmith_cli.py          # Main Typer app: run, envs, deps, trainers, version, config
│   └── cli/
│       ├── inspect.py            # robosmith inspect subcommands
│       ├── diag.py               # robosmith diag subcommands
│       ├── gen.py                # robosmith gen subcommands
│       └── auto.py               # robosmith auto subcommands
│
├── agent/                        # Agentic layer (LangGraph)
│   ├── llm.py                    # LiteLLM wrapper with retry and JSON parsing
│   └── graphs/
│       ├── run.py                # Main training pipeline StateGraph
│       └── auto_integrate.py     # Auto-integrate workflow StateGraph
│
├── stages/                       # Pipeline stage implementations
│   ├── intake/                   # Stage 1: NL → TaskSpec
│   ├── env_synthesis/            # Stage 3: task → EnvEntry
│   ├── evaluation/               # Stage 6: behavioral success detection
│   └── delivery/                 # Stage 7: artifact packaging
│
├── envs/                         # Environment registry and adapters
│   ├── registry.py               # EnvRegistry — YAML catalog with substring search
│   ├── adapter_registry.py       # Adapter discovery and routing
│   ├── wrapper.py                # make_env() — single entry point for env creation
│   ├── reward_wrapper.py         # ForgeRewardWrapper — injects custom reward functions
│   └── adapters/                 # Framework adapters
│       ├── gymnasium_adapter.py
│       ├── isaac_lab_adapter.py
│       ├── libero_adapter.py
│       ├── maniskill_adapter.py
│       └── custom_mjcf_adapter.py
│
├── trainers/                     # Training backend abstractions
│   ├── base.py                   # Trainer ABC, TrainingConfig, TrainingResult, Policy
│   ├── registry.py               # TrainerRegistry singleton
│   ├── selector.py               # Task-aware algorithm selection
│   ├── sb3_trainer.py            # Stable Baselines3 backend
│   ├── cleanrl_trainer.py        # CleanRL (pure PyTorch PPO)
│   ├── rl_games_trainer.py       # NVIDIA rl_games (GPU-parallel)
│   ├── il_trainer.py             # Imitation learning (BC, DAgger)
│   └── offline_rl_trainer.py     # Offline RL (TD3+BC, CQL, IQL)
│
├── inspect/                      # Artifact inspection subsystem
│   ├── dispatch.py               # inspect_dataset(), inspect_env(), inspect_policy(), inspect_robot()
│   ├── compat.py                 # check_compatibility() — finds mismatches
│   ├── formatter.py              # Rich table formatting for inspect output
│   ├── models.py                 # DatasetInspectResult, EnvInspectResult, CompatReport, etc.
│   ├── registry.py               # dataset_registry, env_registry
│   └── inspectors/
│       ├── lerobot.py            # LeRobot dataset inspector (v2 + v3, Hub)
│       ├── lerobot_policy.py     # LeRobot policy inspector
│       └── gymnasium_env.py      # Gymnasium environment inspector
│
├── diagnostics/                  # Rollout analysis
│   ├── trajectory_analyzer.py    # analyze_trajectory(), compare_trajectories()
│   ├── trajectory_reader.py      # HDF5 and LeRobot dataset readers
│   └── diag_models.py            # TrajectoryDiagResult, CompareResult, ActionStats, etc.
│
├── generators/                   # Code generation
│   └── gen_wrapper.py            # generate_wrapper() — template + LLM-based adapter generation
│
├── config.py                     # Pydantic models: TaskSpec, ForgeConfig, LLMConfig, etc.
├── utils.py                      # banner() — ASCII art + info panel
└── __init__.py                   # __version__, public API
```

---

## Running tests

```bash
# Full test suite
pytest tests/

# Specific file
pytest tests/test_inspect.py

# With coverage report
pytest tests/ --cov=robosmith --cov-report=html

# Skip tests that require LLM keys or MuJoCo
pytest tests/ -m "not integration"

# Verbose output
pytest tests/ -v
```

Most tests mock LLM calls and environment creation, so they run without API keys or simulation dependencies.

---

## Code style

```bash
# Check for issues
ruff check robosmith/

# Auto-fix
ruff check robosmith/ --fix

# Format
ruff format robosmith/
```

Key rules: 100-character line length, Python 3.11+ target, type annotations on all public functions, docstrings on all public classes and functions.

```bash
mypy robosmith/
```

---

## Adding a new training backend

Training backends are lazy-loaded at runtime — the core pipeline doesn't import SB3, PyTorch, or any backend directly.

**1.** Create `robosmith/trainers/my_trainer.py`:

```python
from robosmith.trainers.base import Trainer, TrainingConfig, TrainingResult, LearningParadigm, Policy

class MyTrainer(Trainer):
    name = "my_trainer"
    paradigm = LearningParadigm.REINFORCEMENT_LEARNING
    algorithms = ["my_algo"]
    requires = ["my_package"]
    description = "My custom trainer"

    def train(self, config: TrainingConfig) -> TrainingResult:
        ...

    def load_policy(self, path: Path) -> Policy:
        ...
```

**2.** Register it in `robosmith/trainers/registry.py`:

```python
self._known_backends = {
    ...
    "my_trainer": ("robosmith.trainers.my_trainer", "MyTrainer"),
}
```

**3.** Add tests in `tests/test_my_trainer.py`.

See [Custom Trainers](extending/trainers.md) for the full guide.

---

## Adding a new environment adapter

Environment adapters are also lazy-loaded. Each adapter handles a specific simulation framework.

**1.** Create `robosmith/envs/adapters/my_adapter.py`:

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

**2.** Register it in `robosmith/envs/adapter_registry.py`.

**3.** Add environment entries to `configs/env_registry.yaml`.

**4.** Add tests in `tests/test_my_adapter.py`.

See [Custom Environments](extending/environments.md) for the full guide.

---

## Adding a new inspector

Inspectors live in `robosmith/inspect/inspectors/`. Each inspector handles a specific artifact type (a dataset format, an environment type, a policy format).

**1.** Create `robosmith/inspect/inspectors/my_inspector.py`:

```python
from robosmith.inspect.registry import BaseDatasetInspector, dataset_registry
from robosmith.inspect.models import DatasetInspectResult

class MyFormatInspector(BaseDatasetInspector):
    name = "my_format"

    def can_handle(self, identifier: str, **kwargs) -> bool:
        # Return True if this inspector can handle the given identifier
        ...

    def inspect(self, identifier: str, **kwargs) -> DatasetInspectResult:
        # Return a populated DatasetInspectResult
        ...

dataset_registry.register("my_format", MyFormatInspector)
```

**2.** Import the inspector in `robosmith/inspect/dispatch.py` so it's registered on import.

**3.** Add formatting support in `robosmith/inspect/formatter.py` if the result type is new.

**4.** Add tests in `tests/test_inspect.py`.

---

## Adding a new pipeline stage

Pipeline stages are nodes in the LangGraph defined in `robosmith/agent/graphs/run.py`.

**1.** Create `robosmith/stages/my_stage/` with a `run_my_stage()` function.

**2.** Add the stage as a node in `robosmith/agent/graphs/run.py`:

```python
graph.add_node("my_stage", my_stage_node)
graph.add_edge("previous_stage", "my_stage")
graph.add_edge("my_stage", "next_stage")
```

**3.** Add tests in `tests/test_my_stage.py`.

Stages should: have a single public entry point, read from and write to the shared `PipelineState`, handle their own errors gracefully without crashing the pipeline, and be independently testable with mocked dependencies.

---

## Commit guidelines

- Write commit messages that explain the *why*, not just the *what*
- Keep commits focused — one logical change per commit
- Run `ruff check robosmith/` and `pytest tests/` before committing

## Submitting changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes with tests
4. Run `ruff check robosmith/` and `pytest tests/`
5. Commit and push
6. Open a pull request with a clear description of what and why

## Questions?

Open an issue on [GitHub](https://github.com/Shaswat2001/robosmith/issues) for bug reports, feature requests, or questions.
