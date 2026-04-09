"""
Rich formatters for inspection results.

Converts Pydantic models into pretty Rich tables/panels for terminal output.
The --json flag bypasses this entirely and dumps model.model_dump_json().
"""

from __future__ import annotations

from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.console import Console

from robosmith.inspect.models import (
    CompatReport,
    DatasetInspectResult,
    EnvInspectResult,
    PolicyInspectResult,
    RobotInspectResult,
    Severity,
)

console = Console()

SEVERITY_COLORS = {
    Severity.CRITICAL: "red",
    Severity.WARNING: "yellow",
    Severity.INFO: "blue",
}

def format_dataset(result: DatasetInspectResult) -> None:
    """Pretty-print a dataset inspection result."""
    table = Table(title=f"Dataset: {result.repo_id}", show_header=False, border_style="dim")
    table.add_column("Key", style="bold cyan", width=20)
    table.add_column("Value")

    table.add_row("Format", result.dataset_format.value)
    table.add_row("Episodes", str(result.episodes))
    table.add_row("Total Frames", str(result.total_frames))
    table.add_row("FPS", str(result.fps) if result.fps else "unknown")
    table.add_row("Action Dim", str(result.action_dim) if result.action_dim else "unknown")
    table.add_row("State Dim", str(result.state_dim) if result.state_dim else "unknown")

    if result.cameras:
        cam_str = ", ".join(
            f"{name} ({spec.width}x{spec.height})"
            for name, spec in result.cameras.items()
        )
        table.add_row("Cameras", cam_str)

    if result.task_descriptions:
        tasks = result.task_descriptions[:5]  # Show first 5
        for i, task in enumerate(tasks):
            label = "Tasks" if i == 0 else ""
            table.add_row(label, task)
        if len(result.task_descriptions) > 5:
            table.add_row("", f"... and {len(result.task_descriptions) - 5} more")

    if result.storage:
        size_str = f"{result.storage.size_gb} GB" if result.storage.size_gb else "unknown"
        table.add_row("Storage", f"{result.storage.format} ({size_str})")

    if result.action_keys:
        table.add_row("Action Keys", ", ".join(result.action_keys[:10]))

    console.print(table)

    # Quality issues
    if result.quality_issues:
        console.print()
        issue_table = Table(title="Quality Issues", border_style="dim")
        issue_table.add_column("Severity", width=10)
        issue_table.add_column("Type", width=20)
        issue_table.add_column("Detail")

        for issue in result.quality_issues:
            color = SEVERITY_COLORS.get(issue.severity, "white")
            issue_table.add_row(
                Text(issue.severity.value.upper(), style=color),
                issue.issue_type,
                issue.detail,
            )
        console.print(issue_table)

def format_env(result: EnvInspectResult) -> None:
    """Pretty-print an environment inspection result."""
    table = Table(title=f"Environment: {result.env_id}", show_header=False, border_style="dim")
    table.add_column("Key", style="bold cyan", width=20)
    table.add_column("Value")

    table.add_row("Framework", result.framework)
    table.add_row("Max Steps", str(result.max_episode_steps) if result.max_episode_steps else "unlimited")
    table.add_row("Success Function", "yes" if result.has_success_fn else "no")
    table.add_row("FPS", str(result.fps) if result.fps else "unknown")
    table.add_row("Render Modes", ", ".join(result.render_modes) if result.render_modes else "none")

    if result.action_space:
        a = result.action_space
        table.add_row("Action Space", f"shape={a.shape}, dtype={a.dtype}, range=[{a.low}, {a.high}]")

    if result.action_semantics:
        table.add_row("Action Semantics", ", ".join(result.action_semantics))

    console.print(table)

    # Obs space
    if result.obs_space:
        obs_table = Table(title="Observation Space", border_style="dim")
        obs_table.add_column("Key", style="bold")
        obs_table.add_column("Shape")
        obs_table.add_column("Dtype")
        obs_table.add_column("Range")

        for key, spec in result.obs_space.items():
            range_str = f"[{spec.low}, {spec.high}]" if spec.low is not None else ""
            obs_table.add_row(key, str(spec.shape), spec.dtype, range_str)
        console.print(obs_table)

def format_policy(result: PolicyInspectResult) -> None:
    """Pretty-print a policy inspection result."""
    table = Table(title=f"Policy: {result.model_id}", show_header=False, border_style="dim")
    table.add_column("Key", style="bold cyan", width=22)
    table.add_column("Value")

    table.add_row("Architecture", result.architecture)
    if result.base_vlm:
        table.add_row("Base VLM", result.base_vlm)
    table.add_row("Action Head", result.action_head.value)
    table.add_row("Action Dim", str(result.action_dim) if result.action_dim else "unknown")
    if result.action_chunk_size:
        table.add_row("Action Chunk Size", str(result.action_chunk_size))
    table.add_row("Cameras", ", ".join(result.expected_cameras) if result.expected_cameras else "none")
    table.add_row("State Keys", ", ".join(result.expected_state_keys) if result.expected_state_keys else "none")
    table.add_row("Normalization", result.normalization or "none")
    table.add_row("Language Input", "yes" if result.accepts_language_instruction else "no")
    if result.parameters:
        table.add_row("Parameters", result.parameters)
    if result.inference_dtype:
        table.add_row("Inference Dtype", result.inference_dtype)

    console.print(table)

def format_robot(result: RobotInspectResult) -> None:
    """Pretty-print a robot inspection result."""
    table = Table(title=f"Robot: {result.name}", show_header=False, border_style="dim")
    table.add_column("Key", style="bold cyan", width=16)
    table.add_column("Value")

    table.add_row("Source", result.source_file)
    table.add_row("DOF", str(result.dof))
    table.add_row("Total Links", str(result.total_links))
    if result.end_effector:
        table.add_row("End Effector", result.end_effector)
    if result.gripper:
        g = result.gripper
        table.add_row("Gripper", f"{g.gripper_type} (dof={g.dof})")

    console.print(table)

    if result.joints:
        joint_table = Table(title="Joints", border_style="dim")
        joint_table.add_column("Name")
        joint_table.add_column("Type")
        joint_table.add_column("Limits")

        for j in result.joints:
            limits_str = f"[{j.limits[0]:.3f}, {j.limits[1]:.3f}]" if j.limits else "none"
            joint_table.add_row(j.name, j.joint_type, limits_str)
        console.print(joint_table)

def format_compat(result: CompatReport) -> None:
    """Pretty-print a compatibility report."""
    status = "[green]COMPATIBLE[/green]" if result.compatible else "[red]INCOMPATIBLE[/red]"
    title = f"Compatibility: {result.artifact_a} ↔ {result.artifact_b}"
    if result.artifact_c:
        title += f" ↔ {result.artifact_c}"

    console.print(Panel(f"{title}\nStatus: {status}", border_style="dim"))

    all_issues = (
        [(i, "error") for i in result.errors]
        + [(i, "warning") for i in result.warnings]
        + [(i, "info") for i in result.info]
    )

    if all_issues:
        issue_table = Table(border_style="dim")
        issue_table.add_column("Severity", width=10)
        issue_table.add_column("Type", width=25)
        issue_table.add_column("Detail")
        issue_table.add_column("Fix Hint", style="dim")

        for issue, _ in all_issues:
            color = SEVERITY_COLORS.get(issue.severity, "white")
            issue_table.add_row(
                Text(issue.severity.value.upper(), style=color),
                issue.issue_type,
                issue.detail,
                issue.fix_hint or "",
            )
        console.print(issue_table)
    else:
        console.print("[green]No issues found.[/green]")