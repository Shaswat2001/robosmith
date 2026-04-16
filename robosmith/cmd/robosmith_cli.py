"""
Forge CLI — the main entry point for RoboSmith.
"""

from __future__ import annotations

import yaml
import typer
import logging
from pathlib import Path
from loguru import logger
from typing import Optional
from rich.console import Console

from robosmith import __version__
from robosmith.utils import banner
from robosmith.envs.registry import EnvRegistry
from robosmith.config import RewardSearchConfig, LLMConfig
from robosmith.config import Algorithm, ForgeConfig, RobotType, TaskSpec
from .cli import diag_app, gen_app, auto_app, inspect_app

app = typer.Typer(
    name="robosmith",
    help="RoboSmith — Natural language → trained robot policy.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(diag_app)
app.add_typer(inspect_app)
app.add_typer(gen_app)
app.add_typer(auto_app)

console = Console()

@app.command()
def envs(
    robot: Optional[str] = typer.Option(None, "--robot", "-r", help="Filter by robot type"),
    framework: Optional[str] = typer.Option(None, "--framework", "-f", help="Filter by framework"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags to search"),
    env_type: Optional[str] = typer.Option(None, "--env-type", "-e", help="Filter by env type"),
) -> None:
    """List and search available simulation environments."""

    banner()
    registry = EnvRegistry()

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    results = registry.search(
        robot_type=robot,
        framework=framework,
        env_type=env_type,
        tags=tag_list,
    )

    if not results:
        results = registry.list_all()

    table = Table(title=f"Environments ({len(results)} found)", border_style="dim")
    table.add_column("ID", style="bold")
    table.add_column("Framework")
    table.add_column("Robot")
    table.add_column("Type")
    table.add_column("Description", max_width=40)

    for e in results:
        fw_style = {"gymnasium": "green", "isaac_lab": "cyan", "mjlab": "magenta"}.get(
            e.framework, "white"
        )
        table.add_row(
            e.id,
            f"[{fw_style}]{e.framework}[/{fw_style}]",
            f"{e.robot_model or e.robot_type}",
            e.env_type,
            e.description[:40],
        )

    console.print()
    console.print(table)

@app.command()
def run(
    task: str = typer.Option(..., "--task", "-t", help="Natural language task description"),
    robot: Optional[str] = typer.Option(None, "--robot", "-r", help="Robot type: arm, quadruped, biped, dexterous_hand, mobile_base"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific robot model: franka, unitree_go2, shadow_hand"),
    algorithm: Optional[str] = typer.Option(None, "--algo", "-a", help="RL algorithm: ppo, sac, td3, auto"),
    time_budget: int = typer.Option(60, "--time-budget", help="Max training time in minutes"),
    num_envs: int = typer.Option(1024, "--num-envs", help="Number of parallel sim environments"),
    push_to_hub: Optional[str] = typer.Option(None, "--push-to-hub", help="HuggingFace repo ID to push to"),
    candidates: int = typer.Option(4, "--candidates", "-c", help="Number of reward function candidates per iteration"),
    skip: Optional[list[str]] = typer.Option(None, "--skip", "-s", help="Stages to skip: scout, intake, delivery"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Training backend: sb3, cleanrl (default: auto)"),
    llm: Optional[str] = typer.Option(None, "--llm", "-L", help="LLM provider or model: anthropic, openai, gemini, groq, or 'openai/gpt-4o'"),
    scout: Optional[str] = typer.Option(None, "--scout", help="Literature search backend: semantic_scholar, arxiv, both"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Path to robosmith.yaml config file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse and plan only, don't train"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logs"),
) -> None:
    """
    Run the full RoboSmith pipeline.

    Provide a natural language task description, and RoboSmith handles everything:
    environment setup, reward design, training, evaluation, and packaging.
    """
    banner()

    # Suppress all noisy loggers — we handle terminal output via on_step
    logger.remove()
    # Always write logs to file; --verbose drops the level to DEBUG
    log_path = Path("robosmith_runs") / "latest.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(str(log_path), level=log_level, format="{time:HH:mm:ss} | {level:<7} | {message}", mode="w")
    console.print(f"  [dim]Logs → {log_path}[/dim]\n")

    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    # Load .env.local / .env, then resolve the LLM model to use
    from robosmith.env_loader import load_env_local, resolve_llm
    loaded_vars = load_env_local()
    if loaded_vars:
        console.print(f"  [dim]Loaded {len(loaded_vars)} vars from .env.local[/dim]")

    try:
        robot_type = RobotType(robot) if robot else RobotType.ARM
    except ValueError:
        valid = ", ".join(r.value for r in RobotType)
        console.print(f"  [red]Invalid robot type '{robot}'. Valid: {valid}[/red]")
        raise typer.Exit(1)

    try:
        algo = Algorithm(algorithm) if algorithm else Algorithm.AUTO
    except ValueError:
        valid = ", ".join(a.value for a in Algorithm)
        console.print(f"  [red]Invalid algorithm '{algorithm}'. Valid: {valid}[/red]")
        raise typer.Exit(1)

    # Build TaskSpec
    task_spec = TaskSpec(
        task_description=task,
        raw_input=task,
        robot_type=robot_type,
        robot_model=model,
        algorithm=algo,
        time_budget_minutes=time_budget,
        num_envs=num_envs,
        push_to_hub=push_to_hub,
    )

    # Load config file if provided, auto-detect robosmith.yaml in cwd
    file_config: dict = {}
    _config_path = config_file
    if _config_path is None:
        for name in ("robosmith.yaml", "robosmith.yml"):
            if Path(name).exists():
                _config_path = Path(name)
                break

    if _config_path and _config_path.exists():
        with open(_config_path) as f:
            file_config = yaml.safe_load(f) or {}
        console.print(f"  [dim]Config loaded from {_config_path}[/dim]\n")

    # Resolve LLM: --llm flag > robosmith.yaml > auto-detect from API keys
    llm_cfg = file_config.get("llm", {})
    config_model = llm_cfg.get("model") if llm_cfg.get("model") else None
    resolved_model, resolved_fast_model = resolve_llm(llm_arg=llm, config_model=config_model)
    llm_config = LLMConfig(
        provider=resolved_model.split("/")[0] if "/" in resolved_model else "anthropic",
        model=resolved_model,
        fast_model=resolved_fast_model,
        temperature=llm_cfg.get("temperature", 0.7),
        max_retries=llm_cfg.get("max_retries", 3),
    )

    # Reward search config — CLI --candidates overrides file
    rs_cfg = file_config.get("reward_search", {})
    reward_config = RewardSearchConfig(
        candidates_per_iteration=candidates if candidates != 4 else rs_cfg.get("candidates_per_iteration", 4),
        num_iterations=rs_cfg.get("num_iterations", 3),
        eval_timesteps=rs_cfg.get("eval_timesteps", 50_000),
        eval_time_minutes=rs_cfg.get("eval_time_minutes", 2.0),
    )

    # Skip stages — merge file + CLI
    skip_stages = list(file_config.get("skip_stages", []))
    SKIPPABLE = {"scout", "intake", "delivery"}
    if skip:
        for s in skip:
            if s not in skip_stages:
                skip_stages.append(s)
    for s in skip_stages[:]:
        if s not in SKIPPABLE:
            console.print(f"  [yellow]Warning: '{s}' cannot be skipped (core stage). Ignoring.[/yellow]")
            skip_stages.remove(s)

    # Scout source: --scout flag > robosmith.yaml > default
    VALID_SCOUT = {"semantic_scholar", "arxiv", "both"}
    scout_source = scout or file_config.get("scout_source", "semantic_scholar")
    if scout_source not in VALID_SCOUT:
        console.print(f"  [red]Invalid --scout '{scout_source}'. Valid: {', '.join(VALID_SCOUT)}[/red]")
        raise typer.Exit(1)

    # Paths from file
    runs_dir = Path(file_config.get("runs_dir", "./robosmith_runs"))
    env_registry_path = file_config.get("env_registry_path")

    config = ForgeConfig(
        llm=llm_config,
        reward_search=reward_config,
        verbose=verbose,
        dry_run=dry_run,
        skip_stages=skip_stages,
        scout_source=scout_source,
        runs_dir=runs_dir,
        env_registry_path=Path(env_registry_path) if env_registry_path else None,
        max_iterations=file_config.get("max_iterations", 3),
    )

    # Store training backend preference (not a Pydantic field, just an attribute)
    if backend or file_config.get("training_backend"):
        config._training_backend = backend or file_config.get("training_backend")

    # Show minimal header — LLM hasn't parsed the task yet so don't show derived fields
    console.print(f"  Task:   [bold]{task}[/bold]")
    parts = [f"llm={resolved_model}"]
    if algorithm:
        parts.append(f"algo={algorithm}")
    if robot:
        parts.append(f"robot={robot}")
    parts.append(f"budget={time_budget}m")
    parts.append(f"candidates={candidates}")
    parts.append(f"scout={scout_source}")
    if backend:
        parts.append(f"backend={backend}")
    console.print(f"  Config: [dim]{' · '.join(parts)}[/dim]")
    console.print()

    if dry_run:
        console.print("[yellow]Dry run — stopping before execution.[/yellow]")
        return

    # Node → Rich color for stage progress lines
    _NODE_COLOR = {
        "intake": "white",
        "scout": "blue",
        "env_synthesis": "cyan",
        "inspect_env": "cyan",
        "reward_design": "magenta",
        "training": "yellow",
        "evaluation": "green",
        "delivery": "bold green",
    }

    def _on_step(node_name: str, line: str) -> None:
        color = _NODE_COLOR.get(node_name, "white")
        console.print(f"  [{color}]{line}[/{color}]")

    from robosmith.agent.graphs.run import run_pipeline
    result = run_pipeline(task_spec, config, on_step=_on_step)

    # ── Summary ──
    console.print()
    eval_report = result.get("eval_report")
    if eval_report and hasattr(eval_report, "success_rate"):
        decision = getattr(eval_report, "decision", "")
        decision_str = decision.value if hasattr(decision, "value") else str(decision)
        console.print(
            f"  Success: [bold]{eval_report.success_rate:.0%}[/bold]"
            f"  |  Reward: [bold]{eval_report.mean_reward:.2f}[/bold]"
            f"  |  Decision: [bold]{decision_str}[/bold]"
        )

    console.print(f"  Run: [dim]{result.get('run_id', '')}[/dim]")
    artifacts_dir = result.get("artifacts_dir", "")
    if artifacts_dir:
        console.print(f"  Artifacts: [dim]{artifacts_dir}[/dim]")
    console.print()

@app.command()
def version() -> None:
    """Show RoboSmith version."""
    console.print(f"RoboSmith v{__version__}")

@app.command()
def config() -> None:
    """Show default configuration."""
    banner()
    cfg = ForgeConfig()
    console.print_json(cfg.model_dump_json(indent=2))

@app.command()
def deps() -> None:
    """Show installed dependencies and how to install missing ones."""
    banner()
    console.print()

    # ── Environment adapters ──
    console.print("[bold]Environment Adapters[/bold]")
    console.print()

    env_deps = [
        ("gymnasium", "gymnasium", "pip install gymnasium", "Core simulation (MuJoCo, classic control)"),
        ("mujoco", "mujoco", "pip install mujoco", "MuJoCo physics engine"),
        ("gymnasium_robotics", "gymnasium-robotics", "pip install gymnasium-robotics", "Fetch, Shadow Hand envs"),
        ("mani_skill", "mani-skill", "pip install mani-skill", "ManiSkill manipulation envs"),
        ("libero", "libero", "pip install libero", "LIBERO benchmark (130 tasks)"),
        ("isaaclab", "isaaclab", "See: https://isaac-sim.github.io/IsaacLab/", "Isaac Lab GPU-parallel envs"),
    ]

    for module, package, install_cmd, description in env_deps:
        try:
            __import__(module)
            console.print(f"  [green]✓[/green] {package:25} {description}")
        except ImportError:
            console.print(f"  [red]✗[/red] {package:25} {description}")
            console.print(f"    [dim]{install_cmd}[/dim]")

    # ── Training backends ──
    console.print()
    console.print("[bold]Training Backends[/bold]")
    console.print()

    train_deps = [
        ("stable_baselines3", "stable-baselines3", "pip install stable-baselines3", "PPO, SAC, TD3, A2C, DQN"),
        ("torch", "pytorch", "pip install torch", "PyTorch (CleanRL, IL, Offline RL)"),
        ("rl_games", "rl-games", "pip install rl-games", "GPU-accelerated PPO"),
    ]

    for module, package, install_cmd, description in train_deps:
        try:
            __import__(module)
            console.print(f"  [green]✓[/green] {package:25} {description}")
        except ImportError:
            console.print(f"  [red]✗[/red] {package:25} {description}")
            console.print(f"    [dim]{install_cmd}[/dim]")

    # ── Extras ──
    console.print()
    console.print("[bold]Extras[/bold]")
    console.print()

    extras = [
        ("imageio", "imageio", "pip install imageio[ffmpeg]", "Video recording"),
        ("moviepy", "moviepy", "pip install moviepy", "Video encoding (RecordVideo)"),
        ("huggingface_hub", "huggingface-hub", "pip install huggingface-hub", "Model sharing"),
        ("httpx", "httpx", "pip install httpx", "Semantic Scholar API (scout)"),
    ]

    for module, package, install_cmd, description in extras:
        try:
            __import__(module)
            console.print(f"  [green]✓[/green] {package:25} {description}")
        except ImportError:
            console.print(f"  [red]✗[/red] {package:25} {description}")
            console.print(f"    [dim]{install_cmd}[/dim]")

    console.print()
    console.print("[dim]Quick install groups:[/dim]")
    console.print('  [dim]pip install robosmith[sim]       # MuJoCo + Gymnasium[/dim]')
    console.print('  [dim]pip install robosmith[train]     # SB3 + PyTorch[/dim]')
    console.print('  [dim]pip install robosmith[robotics]  # Gymnasium-Robotics[/dim]')
    console.print('  [dim]pip install robosmith[video]     # Video recording[/dim]')
    console.print('  [dim]pip install robosmith[all]       # Everything[/dim]')
    console.print()


@app.command()
def trainers() -> None:
    """Show available training backends and algorithms."""
    banner()
    console.print()

    from robosmith.trainers.registry import TrainerRegistry
    registry = TrainerRegistry()

    console.print("[bold]Training Backends[/bold]")
    console.print()

    for info in registry.list_all():
        status = "[green]✓[/green]" if info["available"] else "[red]✗[/red]"
        algos = ", ".join(info["algorithms"])
        console.print(f"  {status} [bold]{info['name']:15}[/bold] | {info['paradigm']:12} | {algos}")
        if not info["available"]:
            console.print(f"    [dim]Requires: {', '.join(info['requires'])}[/dim]")

    console.print()


if __name__ == "__main__":
    app()