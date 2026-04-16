"""
CLI commands for `robosmith auto`.

Compound agentic commands that orchestrate multiple inspection,
diagnostic, and generation steps via LangGraph.
"""

from __future__ import annotations

import json
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from robosmith.utils import banner
from robosmith.agent.graphs.auto_integrate import run_auto_integrate

console = Console()

auto_app = typer.Typer(
    name="auto",
    help="Agentic workflows: auto-integrate, auto-debug, auto-eval.",
    no_args_is_help=True,
)

JsonFlag = Annotated[
    bool,
    typer.Option("--json", "-j", help="Output as machine-readable JSON"),
]

@auto_app.command("integrate")
def auto_integrate_cmd(
    policy_id: Annotated[str, typer.Argument(help="Policy model ID (e.g. lerobot/smolvla_base)")],
    target_id: Annotated[str, typer.Argument(help="Dataset repo_id or env ID to integrate with")],
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="Write wrapper to file")] = None,
    json_output: JsonFlag = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show step-by-step execution")] = False,
) -> None:
    """Auto-integrate a policy with a dataset or environment.

    Inspects both artifacts, checks compatibility, generates an adapter
    wrapper if needed, and produces a complete integration package.

    This is the compound agentic command that chains:
    inspect_policy → inspect_target → check_compat → gen_wrapper → validate
    """

    banner()
    console.print(Panel(
        f"[bold]auto-integrate[/bold]\n"
        f"Policy: {policy_id}\n"
        f"Target: {target_id}",
        border_style="cyan",
    ))

    try:
        final_state = run_auto_integrate(
            policy_id=policy_id,
            target_id=target_id,
            verbose=verbose,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        # Output the full state as JSON
        output_data = {
            "status": final_state["status"],
            "status_message": final_state["status_message"],
            "is_compatible": final_state["is_compatible"],
            "errors": final_state["errors"],
            "warnings": final_state["warnings"],
            "output_files": final_state["output_files"],
            "steps_log": final_state["steps_log"],
        }
        if final_state.get("wrapper_code"):
            output_data["wrapper_code_length"] = len(final_state["wrapper_code"])
        console.print(json.dumps(output_data, indent=2))
        return

    # ── Pretty output ──
    console.print()

    # Steps log
    for line in final_state["steps_log"]:
        if line.startswith("✓"):
            console.print(f"  [green]{line}[/green]")
        elif line.startswith("✗"):
            console.print(f"  [red]{line}[/red]")
        else:
            console.print(f"  [dim]{line}[/dim]")

    console.print()

    # Status
    status = final_state["status"]
    msg = final_state["status_message"]
    if status == "success":
        console.print(Panel(f"[green]{msg}[/green]", title="Result", border_style="green"))
    else:
        console.print(Panel(f"[red]{msg}[/red]", title="Result", border_style="red"))

    # Show wrapper code
    if final_state.get("wrapper_code"):
        console.print()
        if output:
            from pathlib import Path
            Path(output).write_text(final_state["wrapper_code"])
            console.print(f"[green]✓ Wrapper written to {output}[/green]")
        else:
            console.print("[bold]Generated adapter[/bold] [dim](use -o <file> to save)[/dim]")
            console.print(Syntax(
                final_state["wrapper_code"],
                "python",
                theme="monokai",
                line_numbers=True,
            ))

    # Show warnings
    if final_state.get("warnings"):
        console.print()
        console.print("[yellow]Warnings to review:[/yellow]")
        for w in final_state["warnings"]:
            console.print(f"  [yellow]•[/yellow] {w.get('detail', w)}")
