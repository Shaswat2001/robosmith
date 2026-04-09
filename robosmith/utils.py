import pyfiglet
from rich.text import Text
from rich.panel import Panel
from rich.console import Console

from robosmith import __version__

console = Console()

def banner() -> None:
    ascii_art = pyfiglet.figlet_format("ROBOSMITH", font="ansi_shadow")
    text = Text()
    text.append(ascii_art, style="bold cyan")
    text.append(f"  RoboSmith v{__version__}\n", style="dim")
    text.append("  Natural language → trained robot policy", style="italic bright_black")
    console.print(Panel(text, border_style="cyan", padding=(0, 2)))