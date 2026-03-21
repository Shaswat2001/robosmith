"""
Forge CLI — the main entry point for RoboSmith.

Usage::

    # Full pipeline
    forge run --task "A Franka arm that picks up a red cube"

    # Dry run (parse + plan only)
    forge run --task "Quadruped navigating rubble" --dry-run

    # Show version
    forge version

    # Show config
    forge config
"""

from __future__ import annotations

import sys
import typer
import logging
import pyfiglet
import time as _time
from pathlib import Path
from loguru import logger
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from robosmith import __version__
from robosmith.config import StageStatus
from robosmith.envs.registry import EnvRegistry
from robosmith.config import RewardSearchConfig
from robosmith.controller import ForgeController, STAGES
from robosmith.config import Algorithm, ForgeConfig, RobotType, TaskSpec

app = typer.Typer(
    name="robosmith",
    help="RoboSmith — Natural language → trained robot policy.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()

# Banner
def _banner() -> None:
    ascii_art = pyfiglet.figlet_format("ROBOSMITH", font="ansi_shadow")
    text = Text()
    text.append(ascii_art, style="bold cyan")
    text.append(f"  RoboSmith v{__version__}\n", style="dim")
    text.append("  Natural language → trained robot policy", style="italic bright_black")
    console.print(Panel(text, border_style="cyan", padding=(0, 2)))

@app.command()
def envs(
    robot: Optional[str] = typer.Option(None, "--robot", "-r", help="Filter by robot type"),
    framework: Optional[str] = typer.Option(None, "--framework", "-f", help="Filter by framework"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags to search"),
    env_type: Optional[str] = typer.Option(None, "--env-type", "-e", help="Filter by env type"),
) -> None:
    """List and search available simulation environments."""

    _banner()
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
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse and plan only, don't train"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logs"),
) -> None:
    """
    Run the full RoboSmith pipeline.

    Provide a natural language task description, and RoboSmith handles everything:
    environment setup, reward design, training, evaluation, and packaging.
    """
    _banner()

    # Suppress all noisy loggers — we handle output ourselves
    logger.remove()
    if verbose:
        log_path = Path("robosmith_runs") / "latest.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(log_path), level="DEBUG", format="{time:HH:mm:ss} | {level:<7} | {message}", mode="w")
        console.print(f"  [dim]Verbose logs → {log_path}[/dim]\n")

    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    # Build TaskSpec
    task_spec = TaskSpec(
        task_description=task,
        raw_input=task,
        robot_type=RobotType(robot) if robot else RobotType.ARM,
        robot_model=model,
        algorithm=Algorithm(algorithm) if algorithm else Algorithm.AUTO,
        time_budget_minutes=time_budget,
        num_envs=num_envs,
        push_to_hub=push_to_hub,
    )

    # Build config
    # Validate and process skip stages
    skip_stages = []
    SKIPPABLE = {"scout", "intake", "delivery"}
    if skip:
        for s in skip:
            if s in SKIPPABLE:
                skip_stages.append(s)
            else:
                console.print(f"  [yellow]Warning: '{s}' cannot be skipped (core stage). Ignoring.[/yellow]")

    config = ForgeConfig(
        verbose=verbose,
        dry_run=dry_run,
        skip_stages=skip_stages,
        reward_search=RewardSearchConfig(candidates_per_iteration=candidates),
    )

    # Show plan
    _show_task_spec(task_spec)

    if dry_run:
        console.print("\n[yellow]Dry run — stopping before execution.[/yellow]")
        return

    # Run pipeline with live stage-by-stage progress
    controller = ForgeController(task_spec, config)
    console.print()

    STAGE_LABELS = {
        "intake": "Parsing task description",
        "scout": "Searching literature",
        "env_synthesis": "Finding simulation environment",
        "reward_design": "Designing reward functions",
        "training": "Training RL policy",
        "evaluation": "Evaluating policy",
        "delivery": "Packaging artifacts",
    }

    STAGE_COLORS = {
        "intake": "cyan",
        "scout": "blue",
        "env_synthesis": "magenta",
        "reward_design": "yellow",
        "training": "green",
        "evaluation": "blue",
        "delivery": "cyan",
    }

    # Run the pipeline stage by stage with live output
    _critical_failure = False

    while not controller.state.is_complete() and not _critical_failure:
        controller.state.iteration += 1
        if controller.state.iteration > 1:
            console.print(f"\n  [dim]Iteration {controller.state.iteration}/{controller.state.max_iterations}[/dim]")

        for stage_name in STAGES:
            if controller._should_skip_stage(stage_name):
                continue

            label = STAGE_LABELS.get(stage_name, stage_name)
            color = STAGE_COLORS.get(stage_name, "white")

            # Show "running" status
            console.print(f"  [cyan]⟳[/cyan] [{color}]{label}[/{color}]...", end="")

            start = _time.time()
            controller._run_stage(stage_name)
            elapsed = _time.time() - start

            record = controller.state.stages.get(stage_name)
            if record is None:
                console.print(" [dim]?[/dim]")
                continue

            # Overwrite the line with result
            if record.status == StageStatus.COMPLETED:
                # Show extra info for key stages
                extra = ""
                if stage_name == "intake" and controller.task_spec:
                    extra = f" → [dim]{controller.task_spec.summary()}[/dim]"
                elif stage_name == "scout" and record.metadata.get("num_papers"):
                    n = record.metadata["num_papers"]
                    extra = f" → [bold]{n}[/bold] relevant papers found"
                elif stage_name == "env_synthesis" and record.metadata.get("env_gym_id"):
                    extra = f" → [bold]{record.metadata['env_gym_id']}[/bold]"
                elif stage_name == "reward_design" and record.metadata.get("best_score") is not None:
                    extra = f" → best score: [bold]{record.metadata['best_score']:.2f}[/bold]"
                elif stage_name == "training" and record.metadata.get("algorithm"):
                    extra = f" → {record.metadata['algorithm']}, reward={record.metadata.get('final_mean_reward', 0):.2f}"
                elif stage_name == "evaluation" and record.metadata.get("success_rate") is not None:
                    sr = record.metadata["success_rate"]
                    decision = record.metadata.get("decision", "")
                    extra = f" → success={sr:.0%}, decision=[bold]{decision}[/bold]"

                console.print(f"\r  [green]✓[/green] [{color}]{label}[/{color}] [dim]({elapsed:.1f}s)[/dim]{extra}")

            elif record.status == StageStatus.FAILED:
                console.print(f"\r  [red]✗[/red] [{color}]{label}[/{color}] [dim]({elapsed:.1f}s)[/dim]")
                if record.error:
                    err = record.error.split("\n")[0][:80]
                    console.print(f"    [dim red]{err}[/dim red]")

                if stage_name in ("env_synthesis", "reward_design", "training"):
                    _critical_failure = True
                    break

            elif record.status == StageStatus.SKIPPED:
                console.print(f"\r  [dim]–[/dim] [dim]{label} (skipped)[/dim]")

            # Check if evaluation decided to iterate
            if controller._needs_iteration():
                break

    # Save state
    controller._save_state()

    # Show remaining stages that weren't reached
    for stage_name in STAGES:
        if stage_name not in controller.state.stages:
            label = STAGE_LABELS.get(stage_name, stage_name)
            console.print(f"  [dim]○ {label}[/dim]")

    # Summary
    console.print()
    result = controller.state

    eval_stage = result.stages.get("evaluation")
    if eval_stage and eval_stage.status == StageStatus.COMPLETED:
        sr = eval_stage.metadata.get("success_rate")
        mr = eval_stage.metadata.get("mean_reward")
        decision = eval_stage.metadata.get("decision", "")
        if sr is not None:
            console.print(f"  Success: [bold]{sr:.0%}[/bold]  |  Reward: [bold]{mr:.2f}[/bold]  |  Decision: [bold]{decision}[/bold]")

    console.print(f"  Run: [dim]{result.run_id}[/dim]")
    if result.artifacts_dir:
        console.print(f"  Artifacts: [dim]{result.artifacts_dir}[/dim]")
    console.print()

@app.command()
def version() -> None:
    """Show RoboSmith version."""
    console.print(f"RoboSmith v{__version__}")

@app.command()
def config() -> None:
    """Show default configuration."""
    _banner()
    cfg = ForgeConfig()
    console.print_json(cfg.model_dump_json(indent=2))

# Display helpers
def _show_task_spec(spec: TaskSpec) -> None:
    """Pretty-print the parsed task specification."""
    table = Table(title="Task specification", border_style="dim", show_header=False, pad_edge=False)
    table.add_column("Field", style="bold", width=20)
    table.add_column("Value")

    table.add_row("Task", spec.task_description)
    table.add_row("Robot type", spec.robot_type.value)
    table.add_row("Robot model", spec.robot_model or "(auto)")
    table.add_row("Environment", spec.environment_type.value)
    table.add_row("Algorithm", spec.algorithm.value)
    table.add_row("Time budget", f"{spec.time_budget_minutes} min")
    table.add_row("Parallel envs", str(spec.num_envs))
    table.add_row(
        "Success criteria",
        ", ".join(str(c) for c in spec.success_criteria),
    )
    if spec.push_to_hub:
        table.add_row("Push to hub", spec.push_to_hub)

    console.print()
    console.print(table)


if __name__ == "__main__":
    app()