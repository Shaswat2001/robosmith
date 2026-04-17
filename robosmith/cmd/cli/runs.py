"""
`robosmith runs` — inspect and manage past pipeline runs.

Every `robosmith run` produces a directory under `robosmith_runs/run_*/`
containing at minimum `run_state.json`.  Delivery also writes
`eval_report.json` and `task_spec.json`.  These commands read those files
so you can list, inspect, compare, and clean up runs without opening files
manually.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax

runs_app = typer.Typer(
    name="runs",
    help="List, inspect, compare, and clean past pipeline runs.",
    no_args_is_help=True,
)
console = Console()

# Helpers

def _default_runs_dir() -> Path:
    return Path("./robosmith_runs")

def _load_run(run_dir: Path) -> dict:
    """Load all available metadata for a run directory."""
    data: dict = {"run_dir": run_dir, "run_id": run_dir.name}

    for fname in ("run_state.json", "eval_report.json", "task_spec.json", "checkpoint.json"):
        fpath = run_dir / fname
        if fpath.exists():
            try:
                data[fname.replace(".json", "").replace(".", "_")] = json.loads(fpath.read_text())
            except Exception:
                pass

    return data

def _find_run(run_id: str, runs_dir: Path) -> Path | None:
    """Locate a run directory by exact or prefix match."""
    exact = runs_dir / run_id
    if exact.exists():
        return exact
    matches = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(run_id)],
        key=lambda d: d.name,
    )
    return matches[0] if matches else None

def _parse_run_date(run_id: str) -> datetime | None:
    """Extract the datetime embedded in a run_id like run_20250416_143022_abc."""
    parts = run_id.split("_")
    if len(parts) >= 3:
        try:
            return datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    return None

def _format_date(run_id: str) -> str:
    dt = _parse_run_date(run_id)
    if dt:
        return dt.strftime("%Y-%m-%d %H:%M")
    return "unknown"

def _status_color(status: str) -> str:
    return {"success": "green", "failed": "red", "running": "yellow"}.get(status, "white")

def _decision_color(decision: str) -> str:
    return {"accept": "green", "refine_reward": "yellow",
            "switch_algo": "cyan", "adjust_env": "blue"}.get(decision, "white")


def _list_all_runs(runs_dir: Path) -> list[dict]:
    if not runs_dir.exists():
        return []
    dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
        reverse=True,
    )
    return [_load_run(d) for d in dirs]

# Commands
@runs_app.command("list")
def runs_list(
    runs_dir: Path = typer.Option(_default_runs_dir(), "--runs-dir", "-d", help="Base runs directory"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status: success, failed, running"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum rows to show"),
) -> None:
    """List all pipeline runs with key metrics."""
    runs = _list_all_runs(runs_dir)

    if not runs:
        console.print(f"  [yellow]No runs found in {runs_dir}[/yellow]")
        raise typer.Exit(0)

    if status:
        runs = [r for r in runs if r.get("run_state", {}).get("status", "") == status]
        if not runs:
            console.print(f"  [yellow]No runs with status '{status}'[/yellow]")
            raise typer.Exit(0)

    runs = runs[:limit]

    table = Table(
        title=f"Pipeline Runs ({len(runs)} shown)",
        border_style="dim",
        show_lines=False,
    )
    table.add_column("Run ID", style="dim", max_width=36)
    table.add_column("Date")
    table.add_column("Task", max_width=35)
    table.add_column("Status")
    table.add_column("Decision")
    table.add_column("Success%")
    table.add_column("Reward")
    table.add_column("Iters", justify="right")

    for r in runs:
        run_id = r["run_id"]
        state = r.get("run_state", {})
        eval_rep = r.get("eval_report", {})
        task = r.get("task_spec", {})

        status_str = state.get("status", "unknown")
        sc = _status_color(status_str)

        decision = eval_rep.get("decision", "")
        dc = _decision_color(decision)

        success_rate = eval_rep.get("success_rate")
        mean_reward = eval_rep.get("mean_reward")
        iteration = state.get("iteration", "—")

        task_desc = task.get("task_description", state.get("steps_log", [""])[0])[:35]

        table.add_row(
            run_id[-30:],
            _format_date(run_id),
            task_desc,
            f"[{sc}]{status_str}[/{sc}]",
            f"[{dc}]{decision}[/{dc}]" if decision else "—",
            f"{success_rate:.0%}" if success_rate is not None else "—",
            f"{mean_reward:.1f}" if mean_reward is not None else "—",
            str(iteration),
        )

    console.print()
    console.print(table)
    console.print()
    console.print(f"  [dim]Run dir: {runs_dir.resolve()}[/dim]")
    console.print(f"  [dim]robosmith runs inspect <run_id>  — full details[/dim]")
    console.print(f"  [dim]robosmith runs compare <id1> <id2>  — side-by-side[/dim]")
    console.print()

@runs_app.command("inspect")
def runs_inspect(
    run_id: str = typer.Argument(help="Run ID or unique prefix"),
    runs_dir: Path = typer.Option(_default_runs_dir(), "--runs-dir", "-d"),
    show_log: bool = typer.Option(False, "--log", "-l", help="Print the full steps log"),
    show_reward: bool = typer.Option(False, "--reward", "-r", help="Print the reward function code"),
) -> None:
    """Show full details for a single run."""
    run_dir = _find_run(run_id, runs_dir)
    if run_dir is None:
        console.print(f"  [red]Run '{run_id}' not found in {runs_dir}[/red]")
        raise typer.Exit(1)

    r = _load_run(run_dir)
    state = r.get("run_state", {})
    eval_rep = r.get("eval_report", {})
    task = r.get("task_spec", {})
    checkpoint = r.get("checkpoint", {})

    # ── Header ──
    status_str = state.get("status", "unknown")
    sc = _status_color(status_str)
    console.print()
    console.print(Panel(
        f"[bold]{run_dir.name}[/bold]  [{sc}]{status_str}[/{sc}]  {_format_date(run_dir.name)}",
        border_style="cyan",
        expand=False,
    ))

    # ── Task ──
    console.print()
    console.print("[bold]Task[/bold]")
    console.print(f"  Description:  {task.get('task_description', '—')}")
    console.print(f"  Robot:        {task.get('robot_type', '—')}")
    console.print(f"  Algorithm:    {task.get('algorithm', '—')}")
    console.print(f"  Budget:       {task.get('time_budget_minutes', '—')} min")

    # ── Environment ──
    env_match = state.get("env_match", {})
    if env_match:
        console.print()
        console.print("[bold]Environment[/bold]")
        console.print(f"  Gym ID:       {env_match.get('env_gym_id', '—')}")
        console.print(f"  Framework:    {env_match.get('framework', '—')}")
        console.print(f"  Match score:  {env_match.get('score', '—')}")
        console.print(f"  Reason:       {env_match.get('reason', '—')}")

    # ── Evaluation ──
    if eval_rep:
        decision = eval_rep.get("decision", "")
        dc = _decision_color(decision)
        console.print()
        console.print("[bold]Evaluation[/bold]")
        console.print(f"  Success rate: [bold]{eval_rep.get('success_rate', 0):.0%}[/bold]")
        console.print(f"  Mean reward:  {eval_rep.get('mean_reward', '—'):.2f}  ±  {eval_rep.get('std_reward', 0):.2f}")
        console.print(f"  Best/worst:   {eval_rep.get('best_reward', '—'):.2f} / {eval_rep.get('worst_reward', '—'):.2f}")
        console.print(f"  Episodes:     {eval_rep.get('num_episodes', '—')}")
        console.print(f"  Decision:     [{dc}]{decision}[/{dc}]")
        console.print(f"  Reason:       {eval_rep.get('decision_reason', '—')}")

        criteria = eval_rep.get("criteria_results", {})
        if criteria:
            console.print()
            console.print("[bold]Criteria[/bold]")
            for criterion, result in criteria.items():
                passed = result.get("passed")
                icon = "[green]✓[/green]" if passed else ("[red]✗[/red]" if passed is False else "[dim]?[/dim]")
                value = result.get("value")
                val_str = f"{value:.3f}" if isinstance(value, float) else str(value)
                console.print(f"  {icon} {criterion}  (got {val_str})")

    # ── Iteration / resume status ──
    completed = state.get("completed_nodes") or checkpoint.get("completed_nodes", [])
    if completed:
        console.print()
        console.print("[bold]Pipeline progress[/bold]")
        all_nodes = ["intake", "scout", "env_synthesis", "inspect_env",
                     "reward_design", "training", "evaluation", "delivery"]
        for node in all_nodes:
            icon = "[green]✓[/green]" if node in completed else "[dim]○[/dim]"
            console.print(f"  {icon}  {node}")

        if status_str != "success" and len(completed) < len(all_nodes):
            console.print()
            console.print(f"  [yellow]Run did not finish — resume with:[/yellow]")
            console.print(f"  [dim]robosmith resume {run_dir.name}[/dim]")

    # ── Artifacts ──
    console.print()
    console.print("[bold]Artifacts[/bold]")
    for f in sorted(run_dir.iterdir()):
        size = f.stat().st_size
        size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} B"
        console.print(f"  {f.name:<35} {size_str}")

    # ── Optional: steps log ──
    if show_log:
        steps = state.get("steps_log") or checkpoint.get("steps_log", [])
        if steps:
            console.print()
            console.print("[bold]Steps log[/bold]")
            for line in steps:
                console.print(f"  {line}")

    # ── Optional: reward function code ──
    if show_reward:
        reward_path = run_dir / "reward_function.py"
        if reward_path.exists():
            console.print()
            console.print("[bold]Reward function[/bold]")
            console.print(Syntax(reward_path.read_text(), "python", theme="monokai", line_numbers=True))
        else:
            console.print("\n  [dim]reward_function.py not found (delivery may have been skipped)[/dim]")

    console.print()

@runs_app.command("compare")
def runs_compare(
    run_id_a: str = typer.Argument(help="First run ID or prefix"),
    run_id_b: str = typer.Argument(help="Second run ID or prefix"),
    runs_dir: Path = typer.Option(_default_runs_dir(), "--runs-dir", "-d"),
) -> None:
    """Compare two runs side by side."""
    dir_a = _find_run(run_id_a, runs_dir)
    dir_b = _find_run(run_id_b, runs_dir)

    missing = []
    if dir_a is None:
        missing.append(run_id_a)
    if dir_b is None:
        missing.append(run_id_b)
    if missing:
        console.print(f"  [red]Runs not found: {', '.join(missing)}[/red]")
        raise typer.Exit(1)

    ra = _load_run(dir_a)
    rb = _load_run(dir_b)

    def _get(r: dict, *keys: str, default: str = "—") -> str:
        """Traverse nested dicts or return default."""
        for key_path in keys:
            parts = key_path.split(".")
            val = r
            for p in parts:
                if isinstance(val, dict):
                    val = val.get(p)
                else:
                    val = None
                if val is None:
                    break
            if val is not None:
                if isinstance(val, float):
                    return f"{val:.3f}"
                return str(val)
        return default

    table = Table(border_style="cyan", show_header=True)
    table.add_column("Metric", style="bold", min_width=28)
    table.add_column(dir_a.name[-30:], min_width=22)
    table.add_column(dir_b.name[-30:], min_width=22)

    rows = [
        ("Date",              _format_date(dir_a.name),                           _format_date(dir_b.name)),
        ("Status",            _get(ra, "run_state.status"),                        _get(rb, "run_state.status")),
        ("Task",              _get(ra, "task_spec.task_description")[:30],         _get(rb, "task_spec.task_description")[:30]),
        ("Robot",             _get(ra, "task_spec.robot_type"),                    _get(rb, "task_spec.robot_type")),
        ("Algorithm",         _get(ra, "task_spec.algorithm"),                     _get(rb, "task_spec.algorithm")),
        ("Environment",       _get(ra, "run_state.env_match.env_gym_id"),          _get(rb, "run_state.env_match.env_gym_id")),
        ("Decision",          _get(ra, "eval_report.decision"),                    _get(rb, "eval_report.decision")),
        ("Success rate",      _get(ra, "eval_report.success_rate"),                _get(rb, "eval_report.success_rate")),
        ("Mean reward",       _get(ra, "eval_report.mean_reward"),                 _get(rb, "eval_report.mean_reward")),
        ("Std reward",        _get(ra, "eval_report.std_reward"),                  _get(rb, "eval_report.std_reward")),
        ("Best reward",       _get(ra, "eval_report.best_reward"),                 _get(rb, "eval_report.best_reward")),
        ("Worst reward",      _get(ra, "eval_report.worst_reward"),                _get(rb, "eval_report.worst_reward")),
        ("Episodes",          _get(ra, "eval_report.num_episodes"),                _get(rb, "eval_report.num_episodes")),
        ("Iterations",        _get(ra, "run_state.iteration"),                     _get(rb, "run_state.iteration")),
    ]

    for label, val_a, val_b in rows:
        # Highlight differing numeric values
        try:
            fa, fb = float(val_a), float(val_b)
            if fa > fb:
                val_a = f"[green]{val_a}[/green]"
                val_b = f"[red]{val_b}[/red]"
            elif fb > fa:
                val_a = f"[red]{val_a}[/red]"
                val_b = f"[green]{val_b}[/green]"
        except (ValueError, TypeError):
            pass
        table.add_row(label, val_a, val_b)

    console.print()
    console.print(table)
    console.print()

@runs_app.command("clean")
def runs_clean(
    runs_dir: Path = typer.Option(_default_runs_dir(), "--runs-dir", "-d"),
    older_than: int = typer.Option(7, "--older-than", help="Delete runs older than N days"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without deleting"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Delete runs older than N days to free disk space."""
    runs = _list_all_runs(runs_dir)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=older_than)

    to_delete = []
    for r in runs:
        run_id = r["run_id"]
        dt = _parse_run_date(run_id)
        if dt and dt < cutoff:
            to_delete.append(r["run_dir"])

    if not to_delete:
        console.print(f"  [green]Nothing to delete — no runs older than {older_than} days.[/green]")
        raise typer.Exit(0)

    console.print(f"\n  Found [bold]{len(to_delete)}[/bold] runs older than {older_than} days:\n")
    total_size = 0
    for d in to_delete:
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        total_size += size
        size_mb = size / (1024 * 1024)
        console.print(f"  [dim]{d.name}[/dim]  ({size_mb:.1f} MB)")

    console.print(f"\n  Total: [bold]{total_size / (1024*1024):.1f} MB[/bold] would be freed.\n")

    if dry_run:
        console.print("  [yellow]Dry run — nothing deleted.[/yellow]")
        raise typer.Exit(0)

    if not yes:
        confirmed = typer.confirm(f"  Delete {len(to_delete)} run(s)?")
        if not confirmed:
            console.print("  Cancelled.")
            raise typer.Exit(0)

    deleted = 0
    for d in to_delete:
        try:
            shutil.rmtree(d)
            deleted += 1
        except Exception as exc:
            console.print(f"  [red]Failed to delete {d.name}: {exc}[/red]")

    console.print(f"\n  [green]Deleted {deleted} run(s).[/green]\n")
