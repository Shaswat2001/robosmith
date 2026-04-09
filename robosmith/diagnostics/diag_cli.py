"""
CLI commands for `robosmith diag`.

Trajectory diagnostics, training diagnostics, and comparison tools.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

diag_app = typer.Typer(
    name="diag",
    help="Diagnostics for trajectories, training runs, and policies.",
    no_args_is_help=True,
)

JsonFlag = Annotated[
    bool,
    typer.Option("--json", "-j", help="Output as machine-readable JSON"),
]

@diag_app.command("trajectory")
def diag_trajectory_cmd(
    path: Annotated[str, typer.Argument(help="Path to HDF5 file, LeRobot dataset dir, or Hub repo_id")],
    json_output: JsonFlag = False,
) -> None:
    """Analyze trajectory rollouts: success rate, action stats, failure modes."""
    from robosmith.diagnostics.trajectory_analyzer import analyze_trajectory

    try:
        result = analyze_trajectory(path)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        console.print(result.model_dump_json(indent=2, exclude_none=True))
    else:
        _format_trajectory_result(result)


@diag_app.command("compare")
def diag_compare_cmd(
    path_a: Annotated[str, typer.Argument(help="First rollout path")],
    path_b: Annotated[str, typer.Argument(help="Second rollout path")],
    json_output: JsonFlag = False,
) -> None:
    """Compare two trajectory sets side by side."""
    from robosmith.diagnostics.trajectory_analyzer import compare_trajectories

    try:
        result = compare_trajectories(path_a, path_b)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        console.print(result.model_dump_json(indent=2, exclude_none=True))
    else:
        _format_compare_result(result)


# ── Formatters ────────────────────────────────────────────────


def _format_trajectory_result(result: Any) -> None:
    """Pretty-print trajectory diagnostic results."""
    # Overview table
    table = Table(title=f"Trajectory Diagnostics: {result.source}", show_header=False, border_style="dim")
    table.add_column("Key", style="bold cyan", width=22)
    table.add_column("Value")

    table.add_row("Format", result.format)
    table.add_row("Episodes", str(result.num_episodes))
    table.add_row("Total Frames", str(result.total_frames))

    if result.success_rate is not None:
        sr_color = "green" if result.success_rate >= 0.8 else "yellow" if result.success_rate >= 0.5 else "red"
        table.add_row("Success Rate", Text(f"{result.success_rate:.1%} ({result.successes}/{result.successes + result.failures})", style=sr_color))

    table.add_row("Episode Length", f"{result.episode_length_mean:.1f} ± {result.episode_length_std:.1f} [{result.episode_length_min}, {result.episode_length_max}]")

    if result.reward_mean is not None:
        table.add_row("Reward", f"{result.reward_mean:.3f} ± {result.reward_std:.3f} [{result.reward_min:.3f}, {result.reward_max:.3f}]")

    if result.action_dim is not None:
        table.add_row("Action Dim", str(result.action_dim))

    console.print(table)

    # Action stats
    if result.action_stats:
        console.print()
        act_table = Table(title="Action Statistics", border_style="dim")
        act_table.add_column("Dim", width=5)
        act_table.add_column("Mean", width=10)
        act_table.add_column("Std", width=10)
        act_table.add_column("Min", width=10)
        act_table.add_column("Max", width=10)
        act_table.add_column("Clip %", width=8)

        for s in result.action_stats:
            clip_style = "red" if s.clipping_rate > 0.1 else ""
            act_table.add_row(
                str(s.dim),
                f"{s.mean:.4f}",
                f"{s.std:.4f}",
                f"{s.min:.4f}",
                f"{s.max:.4f}",
                Text(f"{s.clipping_rate:.1%}", style=clip_style),
            )
        console.print(act_table)

    # Failure clusters
    if result.failure_clusters:
        console.print()
        console.print(Panel("[bold]Failure Analysis[/bold]", border_style="red"))
        for cluster in result.failure_clusters:
            console.print(f"  [red]Cluster {cluster.cluster_id}[/red]: {cluster.description} ({cluster.count} episodes)")
            if cluster.example_episodes:
                console.print(f"    Examples: episodes {cluster.example_episodes}")


def _format_compare_result(result: Any) -> None:
    """Pretty-print trajectory comparison results."""
    table = Table(title="Trajectory Comparison", border_style="dim")
    table.add_column("Metric", style="bold cyan")
    table.add_column(result.source_a, width=20)
    table.add_column(result.source_b, width=20)
    table.add_column("Delta", width=15)

    if result.success_rate_a is not None:
        delta_str = ""
        if result.success_rate_delta is not None:
            color = "green" if result.success_rate_delta >= 0 else "red"
            delta_str = f"[{color}]{result.success_rate_delta:+.1%}[/{color}]"
        table.add_row(
            "Success Rate",
            f"{result.success_rate_a:.1%}",
            f"{result.success_rate_b:.1%}",
            delta_str,
        )

    table.add_row(
        "Avg Episode Length",
        f"{result.episode_length_mean_a:.1f}",
        f"{result.episode_length_mean_b:.1f}",
        f"{result.episode_length_mean_b - result.episode_length_mean_a:+.1f}",
    )

    console.print(table)

    if result.biggest_degradation:
        console.print(f"\n[yellow]Biggest change:[/yellow] {result.biggest_degradation}")

    if result.action_divergence:
        console.print(f"\n[dim]Action divergence per dim: {[f'{d:.4f}' for d in result.action_divergence]}[/dim]")
