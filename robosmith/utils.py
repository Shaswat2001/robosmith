import os
import sys
import platform

import pyfiglet
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.console import Console

from robosmith import __version__

console = Console()

_CMDS = [
    ("run",      "natural language → policy"),
    ("envs",     "list & search environments"),
    ("deps",     "check installed dependencies"),
    ("trainers", "training backends & algos"),
    ("diag",     "trajectory diagnostics"),
    ("inspect",  "inspect policies and envs"),
    ("gen",      "generate reward functions"),
    ("auto",     "autonomous experiment loop"),
]

_AGENTS = [
    ("intake", "scout"),
    ("env_synthesis",),
    ("reward_design",),
    ("training", "evaluation"),
    ("delivery",),
]


def banner() -> None:
    ascii_art = pyfiglet.figlet_format("robosmith", font="ansi_shadow")

    # ── Large ASCII header ───────────────────────────────────────────────────
    console.print(f"[bold cyan]{ascii_art}[/bold cyan]", justify="center")

    # ── Two-column info panel ────────────────────────────────────────────────
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    plat    = platform.system().lower()
    machine = platform.machine()
    ncpu    = os.cpu_count() or "?"

    # Left — system / agent info
    left = Text()
    left.append("version   ", style="dim")
    left.append(f"{__version__}\n", style="bright_white")
    left.append("python    ", style="dim")
    left.append(f"{py_ver}\n", style="bright_white")
    left.append("platform  ", style="dim")
    left.append(f"{plat} · {machine}\n", style="bright_white")
    left.append("cpu       ", style="dim")
    left.append(f"{ncpu} cores\n", style="bright_white")
    left.append("\n")
    left.append("Agents\n", style="bold bright_white")
    for row in _AGENTS:
        left.append("  " + " · ".join(row) + "\n", style="dim")

    # Right — workflow commands
    right = Text()
    right.append("Workflows\n", style="bold bright_white")
    for cmd, desc in _CMDS:
        right.append(f"  /{cmd:<9}", style="cyan")
        right.append(f"{desc}\n", style="dim")

    # Columns sized so the panel is ~76 chars — fits neatly inside most
    # terminals and gets genuinely centred on wider ones.
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=24, no_wrap=True)
    grid.add_column(width=41, no_wrap=True)
    grid.add_row(left, right)

    panel = Panel(
        grid,
        border_style="cyan",
        padding=(1, 2),
        subtitle=f"[dim]v{__version__}[/dim]",
        expand=False,
    )

    console.print(panel, justify="center")
    console.print()
