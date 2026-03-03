#!/usr/bin/env python3
"""AutoForge — AI-powered multi-agent project generation.

Usage:
    python forge.py "Build a Todo app with user login and task management"
    python forge.py "做一个心理咨询预约平台" --budget 5.00
    python forge.py --resume
    python forge.py --status

Daemon mode:
    python forge.py daemon start       # Start 24/7 daemon
    python forge.py daemon stop        # Stop daemon
    python forge.py daemon status      # Check daemon status
    python forge.py daemon install     # Install as system service

Queue management:
    python forge.py queue "project description"   # Add to queue
    python forge.py projects                      # List all projects
    python forge.py deploy <project_id>           # Show deploy guide
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from engine.config import ForgeConfig
from engine.orchestrator import Orchestrator

console = Console()


def setup_logging(level: str = "INFO", verbose: bool = False) -> None:
    """Configure logging output."""
    log_level = "DEBUG" if verbose else level.upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


_SUBCOMMANDS = {"daemon", "queue", "projects", "deploy"}


def parse_args() -> argparse.Namespace:
    # Detect if the first positional arg is a subcommand
    argv = sys.argv[1:]
    is_subcommand = bool(argv) and argv[0] in _SUBCOMMANDS

    if is_subcommand:
        return _parse_subcommand_args()
    return _parse_direct_args()


def _parse_subcommand_args() -> argparse.Namespace:
    """Parse subcommand-style arguments: daemon, queue, projects, deploy."""
    parser = argparse.ArgumentParser(
        description="AutoForge: AI-powered project generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # daemon
    daemon_parser = subparsers.add_parser("daemon", help="Manage the AutoForge daemon")
    daemon_parser.add_argument(
        "action", choices=["start", "stop", "status", "install"], help="Daemon action",
    )

    # queue
    queue_parser = subparsers.add_parser("queue", help="Add a project to the build queue")
    queue_parser.add_argument("queue_description", help="Project description")
    queue_parser.add_argument("--budget", type=float, default=None, help="Budget limit in USD")

    # projects
    subparsers.add_parser("projects", help="List all projects in the registry")

    # deploy
    deploy_parser = subparsers.add_parser("deploy", help="Show deployment guide for a project")
    deploy_parser.add_argument("project_id", help="Project ID")

    return parser.parse_args()


def _parse_direct_args() -> argparse.Namespace:
    """Parse direct-mode arguments: description, --budget, --resume, etc."""
    parser = argparse.ArgumentParser(
        description="AutoForge: AI-powered project generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python forge.py "Build a Todo app with user login"\n'
            '  python forge.py "做一个博客网站" --budget 5.00\n'
            "  python forge.py --status\n"
            "  python forge.py --resume\n"
            "\n"
            "Daemon mode:\n"
            "  python forge.py daemon start\n"
            '  python forge.py queue "Build a blog"\n'
            "  python forge.py projects\n"
        ),
    )
    parser.add_argument("description", nargs="?", help="Natural language project description")
    parser.add_argument("--budget", type=float, default=None, help="Budget limit in USD (default: from .env or 10.0)")
    parser.add_argument("--agents", type=int, default=None, help="Number of parallel builder agents (default: 3)")
    parser.add_argument("--model", default=None, help="Default model for routine tasks")
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume a previous run")
    parser.add_argument("--status", action="store_true", help="Show current project status")
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")

    args = parser.parse_args()
    args.command = None  # Mark as direct mode
    return args


async def async_main() -> int:
    args = parse_args()

    # Build config from environment with CLI overrides
    overrides = {}
    if hasattr(args, "budget") and args.budget is not None:
        overrides["budget_limit_usd"] = args.budget
    if hasattr(args, "agents") and args.agents is not None:
        overrides["max_agents"] = args.agents
    if hasattr(args, "model") and args.model is not None:
        overrides["model_fast"] = args.model
    if hasattr(args, "verbose") and args.verbose:
        overrides["verbose"] = True
    if hasattr(args, "log_level") and args.log_level:
        overrides["log_level"] = args.log_level

    config = ForgeConfig.from_env(**overrides)
    setup_logging(config.log_level, config.verbose)

    # ── Subcommands ──

    if args.command == "daemon":
        return await _handle_daemon(args, config)

    if args.command == "queue":
        return await _handle_queue(args, config)

    if args.command == "projects":
        return await _handle_projects(config)

    if args.command == "deploy":
        return await _handle_deploy(args, config)

    # ── Legacy direct mode ──

    orchestrator = Orchestrator(config)

    # Handle --status
    if hasattr(args, "status") and args.status:
        orchestrator.show_status()
        return 0

    # Handle --resume
    if hasattr(args, "resume") and args.resume is not None:
        if not config.anthropic_api_key:
            console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set. Edit .env or run setup.sh")
            return 1

        resume_path = None if args.resume == "auto" else Path(args.resume)
        try:
            await orchestrator.resume(resume_path)
            return 0
        except Exception as e:
            console.print(f"[red]Resume failed:[/red] {e}")
            return 1

    # Handle new project
    if not args.description or not args.description.strip():
        console.print("[red]Error:[/red] Please provide a project description or use --resume/--status")
        console.print("Usage: python forge.py \"your project description\"")
        console.print()
        console.print("Daemon mode:")
        console.print("  python forge.py daemon start       # Start 24/7 daemon")
        console.print('  python forge.py queue "project"     # Add to build queue')
        console.print("  python forge.py projects            # List all projects")
        return 1

    # Validate CLI overrides
    if hasattr(args, "budget") and args.budget is not None and args.budget <= 0:
        console.print("[red]Error:[/red] Budget must be a positive number")
        return 1
    if hasattr(args, "agents") and args.agents is not None and args.agents < 1:
        console.print("[red]Error:[/red] Agent count must be at least 1")
        return 1

    if not config.anthropic_api_key:
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set. Edit .env or run setup.sh")
        return 1

    # Print startup info
    console.print()
    console.print("[bold]AutoForge[/bold] — AI-powered project generation")
    console.print(f"  Requirement: {args.description[:80]}{'...' if len(args.description) > 80 else ''}")
    console.print(f"  Budget:      ${config.budget_limit_usd:.2f}")
    console.print(f"  Agents:      {config.max_agents}")
    console.print(f"  Models:      {config.model_strong} / {config.model_fast}")
    console.print()

    try:
        project_dir = await orchestrator.run(args.description)

        # Generate deploy guide for completed projects
        from engine.deploy_guide import generate_deploy_guide
        guide = generate_deploy_guide(project_dir, project_dir.name)
        guide_path = project_dir / "DEPLOY_GUIDE.md"
        guide_path.write_text(guide, encoding="utf-8")
        console.print(f"  Deploy guide: {guide_path}")

        return 0
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.getLogger(__name__).debug("Full traceback:", exc_info=True)
        return 1


# ── Subcommand handlers ──


async def _handle_daemon(args: argparse.Namespace, config: ForgeConfig) -> int:
    """Handle daemon subcommand."""
    action = args.action

    if action == "start":
        if not config.anthropic_api_key:
            console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set. Edit .env or run setup.sh")
            return 1

        from engine.daemon import ForgeDaemon
        daemon = ForgeDaemon(config)
        try:
            await daemon.start()
        except KeyboardInterrupt:
            await daemon.stop()
        return 0

    elif action == "stop":
        console.print("To stop the daemon:")
        if sys.platform == "darwin":
            console.print("  launchctl unload ~/Library/LaunchAgents/com.autoforge.daemon.plist")
        elif sys.platform == "linux":
            console.print("  sudo systemctl stop autoforge")
        else:
            console.print("  Kill the daemon process (Ctrl+C or Task Manager)")
        return 0

    elif action == "status":
        from engine.project_registry import ProjectRegistry, ProjectStatus
        async with ProjectRegistry(config.db_path) as reg:
            building = await reg.list_by_status(ProjectStatus.BUILDING)
            queued = await reg.list_by_status(ProjectStatus.QUEUED)
            total = await reg.total_cost()

        console.print("[bold]AutoForge Daemon Status[/bold]")
        console.print(f"  Building: {len(building)} project(s)")
        console.print(f"  Queued:   {len(queued)} project(s)")
        console.print(f"  Total cost: ${total:.4f}")

        if building:
            for p in building:
                console.print(f"  Current: [{p.id}] {p.description[:60]} ({p.phase})")
        return 0

    elif action == "install":
        return _install_service()

    return 1


def _install_service() -> int:
    """Install the daemon as a system service."""
    project_root = Path(__file__).parent

    if sys.platform == "darwin":
        src = project_root / "services" / "com.autoforge.daemon.plist"
        dest = Path.home() / "Library" / "LaunchAgents" / "com.autoforge.daemon.plist"
        console.print("Installing launchd service...")
        console.print(f"  Copy: {src} → {dest}")
        console.print()
        console.print("Run these commands:")
        console.print(f"  cp {src} {dest}")
        console.print(f"  launchctl load {dest}")
        console.print()
        console.print("To uninstall:")
        console.print(f"  launchctl unload {dest}")
        console.print(f"  rm {dest}")

    elif sys.platform == "linux":
        src = project_root / "services" / "autoforge.service"
        console.print("Installing systemd service...")
        console.print()
        console.print("Run these commands:")
        console.print(f"  sudo cp {src} /etc/systemd/system/autoforge@.service")
        console.print("  sudo systemctl daemon-reload")
        console.print("  sudo systemctl enable autoforge@$USER")
        console.print("  sudo systemctl start autoforge@$USER")
        console.print()
        console.print("To check status:")
        console.print("  sudo systemctl status autoforge@$USER")
        console.print("  journalctl -u autoforge@$USER -f")

    else:
        console.print("Windows: Use Task Scheduler or NSSM to run as a service.")
        console.print(f"  Command: {sys.executable} {project_root / 'forge.py'} daemon start")

    return 0


async def _handle_queue(args: argparse.Namespace, config: ForgeConfig) -> int:
    """Handle queue subcommand — add a project to the build queue."""
    from engine.project_registry import ProjectRegistry

    budget = args.budget if args.budget is not None else config.budget_limit_usd

    async with ProjectRegistry(config.db_path) as reg:
        project = await reg.enqueue(
            description=args.queue_description,
            requested_by="cli",
            budget_usd=budget,
        )
        queue_size = await reg.queue_size()

    console.print("[green]Project queued![/green]")
    console.print(f"  ID:       {project.id}")
    console.print(f"  Budget:   ${project.budget_usd:.2f}")
    console.print(f"  Position: {queue_size}")
    console.print()
    console.print("Start the daemon to process: python forge.py daemon start")
    return 0


async def _handle_projects(config: ForgeConfig) -> int:
    """Handle projects subcommand — list all projects."""
    from engine.project_registry import ProjectRegistry

    async with ProjectRegistry(config.db_path) as reg:
        projects = await reg.list_all()

    if not projects:
        console.print("No projects yet. Use 'python forge.py queue \"description\"' to add one.")
        return 0

    table = Table(title="AutoForge Projects")
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Phase")
    table.add_column("Description")
    table.add_column("Cost", style="yellow")
    table.add_column("Created")

    for p in projects:
        status_style = {
            "queued": "dim",
            "building": "bold blue",
            "completed": "green",
            "failed": "red",
            "cancelled": "dim red",
        }.get(p.status.value, "")

        desc = p.description[:40] + ("..." if len(p.description) > 40 else "")
        created = p.created_at[:16] if p.created_at else ""

        table.add_row(
            p.id,
            f"[{status_style}]{p.status.value}[/{status_style}]",
            p.phase or "—",
            desc,
            f"${p.cost_usd:.2f}" if p.cost_usd > 0 else "—",
            created,
        )

    console.print(table)
    return 0


async def _handle_deploy(args: argparse.Namespace, config: ForgeConfig) -> int:
    """Handle deploy subcommand — show deployment guide."""
    from engine.project_registry import ProjectRegistry, ProjectStatus

    async with ProjectRegistry(config.db_path) as reg:
        try:
            project = await reg.get(args.project_id)
        except KeyError:
            console.print(f"[red]Project not found:[/red] {args.project_id}")
            return 1

    if project.status != ProjectStatus.COMPLETED:
        console.print(f"[yellow]Project not completed yet[/yellow] (status: {project.status.value})")
        return 1

    guide_path = Path(project.workspace_path) / "DEPLOY_GUIDE.md"
    if not guide_path.exists():
        # Generate on the fly
        from engine.deploy_guide import generate_deploy_guide
        guide = generate_deploy_guide(Path(project.workspace_path), project.name)
        guide_path.write_text(guide, encoding="utf-8")

    from rich.markdown import Markdown
    console.print(Markdown(guide_path.read_text(encoding="utf-8")))
    return 0


def main() -> None:
    sys.exit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
