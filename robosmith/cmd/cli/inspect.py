"""
CLI commands for `robosmith inspect`.

This module defines the Typer subgroup that plugs into robosmith's main CLI.
Each command outputs human-readable Rich tables by default, or JSON with --json.

Usage:
    robosmith inspect dataset lerobot/aloha_mobile_cabinet
    robosmith inspect dataset lerobot/aloha_mobile_cabinet --json
    robosmith inspect dataset lerobot/aloha_mobile_cabinet --schema
    robosmith inspect dataset lerobot/aloha_mobile_cabinet --quality
    robosmith inspect compat <policy_id> <dataset_id>
"""

from __future__ import annotations

from typing import Annotated, Optional

import typer
from rich.console import Console

from robosmith.utils import banner
from robosmith.inspect.compat import check_compatibility
from robosmith.inspect.formatter import format_dataset, format_env, format_policy, format_robot, format_compat
from robosmith.inspect.models import DatasetInspectResult, EnvInspectResult, PolicyInspectResult, RobotInspectResult
from robosmith.inspect.dispatch import inspect_dataset, inspect_env, _find_inspector, inspect_policy, inspect_robot
from robosmith.inspect.registry import dataset_registry, env_registry, BaseDatasetInspector, BaseEnvInspector

console = Console()

inspect_app = typer.Typer(
    name="inspect",
    help="Inspect datasets, environments, policies, and robots.",
    no_args_is_help=True,
)

JsonFlag = Annotated[
    bool,
    typer.Option("--json", "-j", help="Output as machine-readable JSON"),
]

@inspect_app.command("dataset")
def inspect_dataset_cmd(
    identifier: Annotated[str, typer.Argument(help="Dataset repo_id (e.g. lerobot/aloha_mobile_cabinet)")],
    json_output: JsonFlag = False,
    schema: Annotated[bool, typer.Option("--schema", help="Show detailed column-level schema")] = False,
    quality: Annotated[bool, typer.Option("--quality", help="Run data quality checks")] = False,
    sample: Annotated[Optional[int], typer.Option("--sample", help="Dump N sample frames")] = None,
) -> None:
    """Inspect a robotics dataset: cameras, action/state dims, episodes, tasks."""

    banner()
    try:
        result = inspect_dataset(identifier)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
 
    if not isinstance(result, DatasetInspectResult):
        console.print(f"[red]Unexpected result type: {type(result)}[/red]")
        raise typer.Exit(1)
 
    # Extended inspection flags
    if schema or quality or sample:
        inspector = _find_inspector(dataset_registry, identifier)
        if inspector and isinstance(inspector, BaseDatasetInspector):
            if schema:
                result.column_stats = inspector.inspect_schema(identifier)
            if quality:
                result.quality_issues = inspector.inspect_quality(identifier)
            if sample:
                result.sample_frames = inspector.inspect_sample(identifier, n=sample)
 
    if json_output:
        console.print(result.model_dump_json(indent=2, exclude_none=True))
    else:
        format_dataset(result)

@inspect_app.command("env")
def inspect_env_cmd(
    identifier: Annotated[str, typer.Argument(help="Environment ID (e.g. LIBERO_Kitchen, Ant-v5)")],
    json_output: JsonFlag = False,
    obs_docs: Annotated[bool, typer.Option("--obs-docs", help="Show obs dimension descriptions")] = False,
    sample_step: Annotated[bool, typer.Option("--sample", help="Run one step and dump obs/reward/info")] = False,
) -> None:
    """Inspect a simulation environment: obs/action spaces, success fn, render modes."""

    banner()
    try:
        result = inspect_env(identifier)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
 
    if not isinstance(result, EnvInspectResult):
        console.print(f"[red]Unexpected result type: {type(result)}[/red]")
        raise typer.Exit(1)
 
    # Extended inspection flags
    if obs_docs or sample_step:
        inspector = _find_inspector(env_registry, identifier)
        if inspector and isinstance(inspector, BaseEnvInspector):
            if obs_docs:
                result.obs_docs = inspector.inspect_obs_docs(identifier)
            if sample_step:
                result.sample_obs = inspector.inspect_sample_step(identifier)
 
    if json_output:
        console.print(result.model_dump_json(indent=2, exclude_none=True))
    else:
        format_env(result)

@inspect_app.command("policy")
def inspect_policy_cmd(
    identifier: Annotated[str, typer.Argument(help="Policy checkpoint or Hub model ID (e.g. lerobot/smolvla_base)")],
    json_output: JsonFlag = False,
    config: Annotated[bool, typer.Option("--config", help="Show full training config")] = False,
    requirements: Annotated[bool, typer.Option("--requirements", help="Show package requirements")] = False,
) -> None:
    """Inspect a policy: architecture, action head, expected inputs/outputs."""

    banner()
    try:
        result = inspect_policy(identifier)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
 
    if not isinstance(result, PolicyInspectResult):
        console.print(f"[red]Unexpected result type: {type(result)}[/red]")
        raise typer.Exit(1)
 
    if json_output:
        console.print(result.model_dump_json(indent=2, exclude_none=True))
    else:
        format_policy(result)

@inspect_app.command("robot")
def inspect_robot_cmd(
    identifier: Annotated[str, typer.Argument(help="Path to URDF or MJCF file")],
    json_output: JsonFlag = False,
) -> None:
    """Inspect a robot description: joints, DOF, end effector, gripper."""

    banner()
    try:
        result = inspect_robot(identifier)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not isinstance(result, RobotInspectResult):
        console.print(f"[red]Unexpected result type: {type(result)}[/red]")
        raise typer.Exit(1)

    if json_output:
        console.print(result.model_dump_json(indent=2, exclude_none=True))
    else:
        format_robot(result)

@inspect_app.command("compat")
def inspect_compat_cmd(
    artifact_a: Annotated[str, typer.Argument(help="First artifact (policy, dataset, or env ID)")],
    artifact_b: Annotated[str, typer.Argument(help="Second artifact (policy, dataset, or env ID)")],
    artifact_c: Annotated[Optional[str], typer.Argument(help="Optional third artifact for 3-way check")] = None,
    json_output: JsonFlag = False,
    fix: Annotated[bool, typer.Option("--fix", help="Auto-generate wrapper to resolve mismatches")] = False,
) -> None:
    """Check compatibility between policy, dataset, and/or environment."""

    banner()
    try:
        result = check_compatibility(artifact_a, artifact_b, artifact_c)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        console.print(result.model_dump_json(indent=2, exclude_none=True))
    else:
        format_compat(result)

    if fix and not result.compatible:
        from rich.syntax import Syntax
        from robosmith.generators.gen_wrapper import generate_wrapper
        console.print("\n[yellow]Generating adapter wrapper...[/yellow]")
        try:
            code = generate_wrapper(artifact_a, artifact_b, use_llm=False)
            console.print("[bold]Generated adapter[/bold] [dim](use robosmith gen wrapper ... -o <file> to save)[/dim]")
            console.print(Syntax(code, "python", theme="monokai", line_numbers=True))
        except Exception as e:
            console.print(f"[red]Wrapper generation failed:[/red] {e}")
