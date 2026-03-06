"""Shared display components for AutoForge CLI."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

BANNER = r"""
    _         _        _____
   / \  _   _| |_ ___ |  ___|__  _ __ __ _  ___
  / _ \| | | | __/ _ \| |_ / _ \| '__/ _` |/ _ \
 / ___ \ |_| | || (_) |  _| (_) | | | (_| |  __/
/_/   \_\__,_|\__\___/|_|  \___/|_|  \__, |\___|
                                      |___/
"""


def show_banner() -> None:
    """Display the AutoForge welcome banner."""
    console.print(Text(BANNER, style="bold cyan"))
    console.print("  AI-powered multi-agent development platform", style="dim")
    console.print()


def show_phase_progress(phase: str, status: str) -> None:
    """Display phase progress indicator."""
    icons = {
        "running": "[bold yellow]...[/bold yellow]",
        "done": "[bold green]OK[/bold green]",
        "failed": "[bold red]FAIL[/bold red]",
        "skipped": "[dim]SKIP[/dim]",
    }
    icon = icons.get(status, "")
    console.print(f"  {icon} {phase}")


def show_cost_tracker(cost: float, budget: float) -> None:
    """Display cost tracking info."""
    pct = (cost / budget * 100) if budget > 0 else 0
    color = "green" if pct < 70 else "yellow" if pct < 90 else "red"
    console.print(f"  Cost: [{color}]${cost:.4f}[/{color}] / ${budget:.2f} ({pct:.1f}%)")


def show_startup_info(
    mode: str,
    action: str,
    description: str,
    budget: float,
    agents: int,
    model_strong: str,
    model_fast: str,
    mobile_target: str = "none",
    run_id: str = "",
    trace_enabled: bool = False,
    trace_llm: bool = False,
    trace_cmd: bool = False,
    trace_fs: bool = False,
) -> None:
    """Display startup configuration panel."""
    from autoforge.engine.git_manager import is_git_available

    git_status = "[green]detected[/green]" if is_git_available() else "[yellow]not found[/yellow] (install: git-scm.com)"

    lines = [
        f"[bold]Mode:[/bold]    {mode}",
        f"[bold]Action:[/bold]  {action}",
        f"[bold]Budget:[/bold]  ${budget:.2f}",
        f"[bold]Agents:[/bold]  {agents}",
        f"[bold]Models:[/bold]  {model_strong} / {model_fast}",
        f"[bold]Git:[/bold]     {git_status}",
    ]
    if run_id:
        lines.append(f"[bold]Run ID:[/bold]  {run_id}")
    if trace_enabled:
        flags = []
        if trace_llm:
            flags.append("llm")
        if trace_cmd:
            flags.append("cmd")
        if trace_fs:
            flags.append("fs")
        suffix = f" ({', '.join(flags)})" if flags else ""
        lines.append(f"[bold]Trace:[/bold]   enabled{suffix}")
    if mobile_target != "none":
        lines.append(f"[bold]Mobile:[/bold]  {mobile_target}")
    if description:
        desc = description[:80] + ("..." if len(description) > 80 else "")
        lines.insert(2, f"[bold]Target:[/bold]  {desc}")

    console.print(Panel("\n".join(lines), title="AutoForge", border_style="cyan"))


def show_review_report(results: dict[str, Any]) -> None:
    """Display a formatted review report."""
    score = results.get("score", 0)
    issues = results.get("issues", [])
    summary = results.get("summary", "")

    # Score color
    if score >= 8:
        score_style = "bold green"
    elif score >= 6:
        score_style = "bold yellow"
    else:
        score_style = "bold red"

    console.print()
    console.print(Panel(
        f"[{score_style}]Score: {score}/10[/{score_style}]\n\n{summary}",
        title="Code Review Report",
        border_style="cyan",
    ))

    if issues:
        table = Table(title="Issues Found")
        table.add_column("Severity", style="bold", width=10)
        table.add_column("File", style="cyan")
        table.add_column("Description")
        table.add_column("Suggestion", style="dim")

        for issue in issues:
            severity = issue.get("severity", "info")
            sev_style = {
                "critical": "bold red",
                "warning": "yellow",
                "info": "dim",
            }.get(severity, "")
            table.add_row(
                f"[{sev_style}]{severity}[/{sev_style}]",
                issue.get("file", ""),
                issue.get("description", ""),
                issue.get("suggestion", ""),
            )

        console.print(table)
    console.print()
