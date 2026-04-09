"""
CLI commands for `robosmith gen`.

Code and config generators that use inspection results + LLM
to produce runnable boilerplate.
"""

from __future__ import annotations

import typer
from rich.syntax import Syntax
from rich.console import Console
from typing import Annotated, Optional

from robosmith.utils import banner
from robosmith.generators.gen_wrapper import generate_wrapper

console = Console()

gen_app = typer.Typer(
    name="gen",
    help="Generate code: wrappers, eval scripts, configs.",
    no_args_is_help=True,
)

@gen_app.command("wrapper")
def gen_wrapper_cmd(
    policy_id: Annotated[str, typer.Argument(help="Policy model ID (e.g. lerobot/smolvla_base)")],
    target_id: Annotated[str, typer.Argument(help="Dataset repo_id or env ID to adapt to")],
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="Write to file instead of stdout")] = None,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Use template-based generation (no API key needed)")] = False,
) -> None:
    """Generate an adapter wrapper to resolve policy/dataset/env mismatches.

    Runs inspect compat internally, then generates Python code that
    handles camera key remapping, action dimension adaptation,
    normalization, and image resizing.
    """
    
    banner()
    try:
        code = generate_wrapper(
            policy_id=policy_id,
            target_id=target_id,
            output_path=output,
            use_llm=not no_llm,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Generation failed:[/red] {e}")
        raise typer.Exit(1)

    if output:
        console.print(f"[green]Wrapper written to {output}[/green]")
    else:
        console.print(Syntax(code, "python", theme="monokai", line_numbers=True))