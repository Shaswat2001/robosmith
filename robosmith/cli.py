"""
Forge CLI — the main entry point for RoboSmith.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import pyfiglet
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from robosmith import __version__
from robosmith.config import RewardSearchConfig
from robosmith.controller import ForgeController
from robosmith.envs.registry import EnvRegistry
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

# Commands
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
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse and plan only, don't train"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Verbose output"),
) -> None:
    """
    Run the full Forge pipeline.

    Provide a natural language task description, and Forge handles everything:
    environment setup, reward design, training, evaluation, and packaging.
    """
    _banner()

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
    config = ForgeConfig(
        verbose=verbose, 
        dry_run=dry_run, 
        reward_search=RewardSearchConfig(candidates_per_iteration=candidates),
    )

    # Show plan
    _show_task_spec(task_spec)

    if dry_run:
        console.print("\n[yellow]Dry run — stopping before execution.[/yellow]")
        return

    # Run pipeline
    controller = ForgeController(task_spec, config)

    console.print()
    with console.status("[bold red]Running Forge pipeline...", spinner="dots"):
        result = controller.run()

    _show_result(result)

@app.command()
def version() -> None:
    """Show Forge version."""
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

def _show_result(state) -> None:  # noqa: ANN001
    """Pretty-print the pipeline result."""
    from robosmith.config import StageStatus

    console.print()

    table = Table(title="Pipeline result", border_style="dim")
    table.add_column("Stage", style="bold")
    table.add_column("Status")
    table.add_column("Time")

    status_styles = {
        StageStatus.COMPLETED: "[green]completed[/green]",
        StageStatus.FAILED: "[red]failed[/red]",
        StageStatus.SKIPPED: "[yellow]skipped[/yellow]",
        StageStatus.PENDING: "[dim]pending[/dim]",
        StageStatus.RUNNING: "[blue]running[/blue]",
    }

    for name, record in state.stages.items():
        table.add_row(
            name,
            status_styles.get(record.status, str(record.status)),
            f"{record.duration_seconds:.1f}s",
        )

    console.print(table)
    console.print(f"\nRun ID: [bold]{state.run_id}[/bold]")
    console.print(f"Artifacts: [dim]{state.artifacts_dir}[/dim]")

if __name__ == "__main__":
    app()
