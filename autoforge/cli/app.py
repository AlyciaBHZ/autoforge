"""AutoForge CLI — interactive multi-mode entry point.

Subcommands:
    (no args)           Interactive mode with InquirerPy menus
    setup               First-run configuration wizard
    generate <desc>     Generate a new project
    review <path>       Review an existing project
    import <path>       Import & improve an existing project
    status              Show project status
    resume [path]       Resume an interrupted run
    daemon <action>     Run daemon lifecycle commands
    queue <desc>        Queue project for daemon execution
    projects            List queued/completed projects from registry
    deploy <id>         Print deploy guide for a completed project
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def setup_logging(level: str = "INFO", verbose: bool = False) -> None:
    """Configure logging output."""
    log_level = "DEBUG" if verbose else level.upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="AutoForge: AI-powered multi-agent development platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  autoforge                                   # Interactive mode\n"
            '  autoforge generate "Build a Todo app"       # Generate new project\n'
            "  autoforge review ./my-project               # Review existing project\n"
            "  autoforge import ./my-project               # Import & improve\n"
            "  autoforge setup                             # Configure settings\n"
            "  autoforge status                            # Show project status\n"
            "  autoforge daemon start                      # Start queue daemon\n"
            '  autoforge queue "Build an API"              # Queue project\n'
            "  autoforge projects                          # List queued/history\n"
            "  autoforge deploy <project_id>               # Print deploy guide\n"
        ),
    )

    # Global flags
    parser.add_argument("--budget", type=float, default=None, help="Budget limit in USD")
    parser.add_argument("--agents", type=int, default=None, help="Number of parallel builder agents")
    parser.add_argument("--model", default=None, help="Model for routine tasks")
    parser.add_argument(
        "--mode",
        choices=["developer", "research"],
        default=None,
        help="Operating mode: developer (read-write) or research (read-only)",
    )
    parser.add_argument(
        "--mobile",
        choices=["none", "ios", "android", "both"],
        default=None,
        help="Mobile app target platform",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument(
        "--confirm",
        default=None,
        help="Phases to pause for confirmation: spec,build,verify,refactor,all",
    )
    parser.add_argument(
        "--tdd",
        type=int,
        default=None,
        help="TDD iterations per build task (0=disabled, 1-3 recommended)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # setup
    subparsers.add_parser("setup", help="Run the configuration wizard")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate a new project")
    gen_parser.add_argument("description", help="Natural language project description")

    # review
    rev_parser = subparsers.add_parser("review", help="Review an existing project")
    rev_parser.add_argument("path", help="Path to the project directory")

    # import
    imp_parser = subparsers.add_parser("import", help="Import & improve an existing project")
    imp_parser.add_argument("path", help="Path to the project directory")
    imp_parser.add_argument("--enhance", default="", help="Description of enhancements to add")

    # status
    subparsers.add_parser("status", help="Show current project status")

    # resume
    res_parser = subparsers.add_parser("resume", help="Resume a previous run")
    res_parser.add_argument("path", nargs="?", default=None, help="Workspace path to resume")

    # daemon
    daemon_parser = subparsers.add_parser("daemon", help="Daemon lifecycle commands")
    daemon_subparsers = daemon_parser.add_subparsers(dest="daemon_action", required=True)
    daemon_subparsers.add_parser("start", help="Start daemon in foreground")
    daemon_subparsers.add_parser("status", help="Show daemon/process status")
    daemon_subparsers.add_parser("stop", help="Stop daemon using PID file")
    daemon_subparsers.add_parser("install", help="Print service installation hints")

    # queue
    queue_parser = subparsers.add_parser("queue", help="Queue a project for daemon build")
    queue_parser.add_argument("description", help="Natural language project description")
    queue_parser.add_argument("--budget", type=float, default=None, help="Per-project budget in USD")
    queue_parser.add_argument(
        "--requester",
        default=None,
        help="Requester id override (default: current user)",
    )
    queue_parser.add_argument(
        "--idempotency-key",
        default=None,
        help="Optional idempotency key to de-duplicate retries",
    )

    # projects
    projects_parser = subparsers.add_parser("projects", help="List projects from registry")
    projects_parser.add_argument("--limit", type=int, default=50, help="Max rows to show")
    projects_parser.add_argument(
        "--requester",
        default=None,
        help="Filter by requester id (default: current user)",
    )
    projects_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all projects (admin/local use)",
    )

    # deploy
    deploy_parser = subparsers.add_parser("deploy", help="Show deployment guide for project id")
    deploy_parser.add_argument("project_id", help="Project id")
    deploy_parser.add_argument(
        "--requester",
        default=None,
        help="Requester id for authorization checks (default: current user)",
    )
    deploy_parser.add_argument(
        "--all",
        action="store_true",
        help="Bypass requester filter and read any project id",
    )

    # Backward compatibility flags
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        dest="legacy_resume",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--status", action="store_true", dest="legacy_status", help=argparse.SUPPRESS)

    return parser


# Known subcommands for legacy detection
_KNOWN_COMMANDS = {
    "setup",
    "generate",
    "review",
    "import",
    "status",
    "resume",
    "daemon",
    "queue",
    "projects",
    "deploy",
}


def _build_config_overrides(args: argparse.Namespace) -> dict:
    """Build config overrides from CLI args."""
    overrides = {}
    if args.budget is not None and getattr(args, "command", None) != "queue":
        overrides["budget_limit_usd"] = args.budget
    if args.agents is not None:
        overrides["max_agents"] = args.agents
    if args.model is not None:
        overrides["model_fast"] = args.model
    if args.mode is not None:
        overrides["mode"] = args.mode
    if args.mobile is not None:
        overrides["mobile_target"] = args.mobile
    if args.verbose:
        overrides["verbose"] = True
    if args.log_level:
        overrides["log_level"] = args.log_level
    if getattr(args, "confirm", None) is not None:
        overrides["confirm_phases"] = [p.strip() for p in args.confirm.split(",")]
    if getattr(args, "tdd", None) is not None:
        overrides["build_test_loops"] = max(0, min(args.tdd, 5))
    return overrides


async def _run_generate(config, description: str) -> int:
    """Run the generate pipeline."""
    from autoforge.cli.display import show_startup_info
    from autoforge.engine.orchestrator import Orchestrator

    if not config.has_api_key:
        console.print("[red]Error:[/red] No API key configured. Run: autoforge setup")
        return 1

    show_startup_info(
        mode=config.mode,
        action="Generate new project",
        description=description,
        budget=config.budget_limit_usd,
        agents=config.max_agents,
        model_strong=config.model_strong,
        model_fast=config.model_fast,
        mobile_target=config.mobile_target,
    )

    orchestrator = Orchestrator(config)
    try:
        await orchestrator.run(description)
        return 0
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.getLogger(__name__).debug("Full traceback:", exc_info=True)
        return 1


async def _run_review(config, project_path: str) -> int:
    """Run the review pipeline."""
    from autoforge.cli.display import show_startup_info
    from autoforge.engine.orchestrator import Orchestrator

    if not config.has_api_key:
        console.print("[red]Error:[/red] No API key configured. Run: autoforge setup")
        return 1

    path = Path(project_path).resolve()
    if not path.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {project_path}")
        return 1

    show_startup_info(
        mode=config.mode,
        action="Review existing project",
        description=str(path),
        budget=config.budget_limit_usd,
        agents=config.max_agents,
        model_strong=config.model_strong,
        model_fast=config.model_fast,
    )

    orchestrator = Orchestrator(config)
    try:
        await orchestrator.review_project(str(path))
        return 0
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.getLogger(__name__).debug("Full traceback:", exc_info=True)
        return 1


async def _run_import(config, project_path: str, enhance: str = "") -> int:
    """Run the import pipeline."""
    from autoforge.cli.display import show_startup_info
    from autoforge.engine.orchestrator import Orchestrator

    if not config.has_api_key:
        console.print("[red]Error:[/red] No API key configured. Run: autoforge setup")
        return 1

    path = Path(project_path).resolve()
    if not path.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {project_path}")
        return 1

    show_startup_info(
        mode=config.mode,
        action="Import & improve project",
        description=f"{path}" + (f" + {enhance[:60]}" if enhance else ""),
        budget=config.budget_limit_usd,
        agents=config.max_agents,
        model_strong=config.model_strong,
        model_fast=config.model_fast,
        mobile_target=config.mobile_target,
    )

    orchestrator = Orchestrator(config)
    try:
        await orchestrator.import_project(str(path), enhance)
        return 0
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.getLogger(__name__).debug("Full traceback:", exc_info=True)
        return 1


def _default_cli_requester() -> str:
    """Return default requester id for local CLI usage."""
    try:
        username = getpass.getuser()
    except Exception:
        username = os.getenv("USERNAME") or os.getenv("USER") or "local"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in username)
    safe = safe.strip("._-") or "local"
    return f"cli:{safe}"


def _normalize_cli_requester(raw: str | None) -> str:
    """Normalize requester override into ``channel:identifier`` format."""
    if not raw:
        return _default_cli_requester()
    value = raw.strip()
    if not value:
        return _default_cli_requester()
    if ":" in value:
        return value
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in value)
    safe = safe.strip("._-") or "local"
    return f"cli:{safe}"


def _read_daemon_pid_file(pid_path: Path | None) -> int | None:
    if pid_path is None or not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but current user cannot signal it.
        return True
    except OSError:
        return False
    return True


async def _run_daemon_start(config) -> int:
    """Start daemon in foreground."""
    from autoforge.engine.daemon import ForgeDaemon

    if not config.has_api_key:
        console.print("[red]Error:[/red] No API key configured. Run: autoforge setup")
        return 1

    running_pid = _read_daemon_pid_file(config.daemon_pid_file)
    if running_pid and _is_pid_running(running_pid):
        console.print(f"[red]Daemon already running with PID {running_pid}[/red]")
        return 1

    daemon = ForgeDaemon(config)
    await daemon.start()
    return 0


async def _run_daemon_status(config) -> int:
    """Show daemon status and queue counters."""
    from autoforge.engine.project_registry import ProjectRegistry, ProjectStatus

    pid = _read_daemon_pid_file(config.daemon_pid_file)
    running = bool(pid and _is_pid_running(pid))

    async with ProjectRegistry(config.db_path) as registry:
        queued = await registry.queue_size()
        building = len(await registry.list_by_status(ProjectStatus.BUILDING))
        completed = len(await registry.list_by_status(ProjectStatus.COMPLETED))
        failed = len(await registry.list_by_status(ProjectStatus.FAILED))

    state = "running" if running else "stopped"
    console.print(f"Daemon: [bold]{state}[/bold]")
    if pid:
        console.print(f"PID file: {config.daemon_pid_file} (pid={pid})")
    else:
        console.print(f"PID file: {config.daemon_pid_file} (not found)")
    console.print(f"Queue: {queued} queued, {building} building")
    console.print(f"History: {completed} completed, {failed} failed")
    return 0


async def _run_daemon_stop(config) -> int:
    """Stop daemon process using PID file."""
    pid = _read_daemon_pid_file(config.daemon_pid_file)
    if pid is None:
        console.print("[yellow]Daemon PID file not found; nothing to stop.[/yellow]")
        return 0

    if not _is_pid_running(pid):
        console.print("[yellow]Stale daemon PID file found; cleaning up.[/yellow]")
        try:
            config.daemon_pid_file.unlink(missing_ok=True)
        except OSError:
            pass
        return 0

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        console.print(f"[red]Failed to stop daemon PID {pid}: {exc}[/red]")
        return 1

    for _ in range(25):
        await asyncio.sleep(0.2)
        if not _is_pid_running(pid):
            break

    if _is_pid_running(pid):
        console.print(f"[red]Daemon PID {pid} did not stop after SIGTERM.[/red]")
        return 1

    try:
        config.daemon_pid_file.unlink(missing_ok=True)
    except OSError:
        pass
    console.print(f"[green]Daemon stopped (pid {pid}).[/green]")
    return 0


async def _run_daemon_install(config) -> int:
    """Print service installation instructions for current OS."""
    project_root = Path(__file__).resolve().parents[2]
    systemd_service = project_root / "services" / "autoforge.service"
    launchd_plist = project_root / "services" / "com.autoforge.daemon.plist"

    if sys.platform.startswith("linux"):
        console.print("Systemd install:")
        console.print(f"  sudo cp {systemd_service} /etc/systemd/system/autoforge@.service")
        console.print("  sudo systemctl daemon-reload")
        console.print("  sudo systemctl enable autoforge@$USER")
        console.print("  sudo systemctl start autoforge@$USER")
    elif sys.platform == "darwin":
        console.print("launchd install:")
        console.print(f"  cp {launchd_plist} ~/Library/LaunchAgents/")
        console.print("  launchctl load ~/Library/LaunchAgents/com.autoforge.daemon.plist")
    else:
        console.print("Windows service install is not automated yet.")
        console.print("Run daemon manually: autoforge daemon start")
    return 0


async def _run_queue(config, args: argparse.Namespace) -> int:
    """Queue a project request in the daemon registry."""
    from autoforge.engine.project_registry import ProjectRegistry
    from autoforge.engine.request_intake import IntakePolicyError, RequestIntakeService

    requested_by = _normalize_cli_requester(getattr(args, "requester", None))
    channel, requester_hint = requested_by.split(":", 1)
    idempotency_key = getattr(args, "idempotency_key", None)
    budget = getattr(args, "budget", None)

    async with ProjectRegistry(config.db_path) as registry:
        intake = RequestIntakeService(config, registry)
        try:
            result = await intake.enqueue(
                channel=channel,
                requester_hint=requester_hint,
                fallback_hint="local",
                description=args.description,
                budget=budget,
                idempotency_key=idempotency_key,
            )
        except IntakePolicyError as exc:
            console.print(f"[red]Queue rejected:[/red] {exc}")
            return 1

    if result.deduplicated:
        console.print("[yellow]Reused existing queued/completed request (idempotency match).[/yellow]")
    console.print(
        f"Queued project [bold]{result.project.id}[/bold] "
        f"(budget=${result.project.budget_usd:.2f}, queue_position={result.queue_size})"
    )
    return 0


async def _run_projects(config, args: argparse.Namespace) -> int:
    """List projects from registry."""
    from autoforge.engine.project_registry import ProjectRegistry

    limit = max(1, min(int(getattr(args, "limit", 50)), 500))
    requested_by = _normalize_cli_requester(getattr(args, "requester", None))
    show_all = bool(getattr(args, "all", False))

    async with ProjectRegistry(config.db_path) as registry:
        if show_all:
            projects = await registry.list_all(limit=limit)
        else:
            projects = await registry.list_for_requester(requested_by, limit=limit)

    if not projects:
        console.print("No projects found.")
        return 0

    for p in projects:
        phase = f" ({p.phase})" if p.phase else ""
        console.print(
            f"[{p.id}] {p.status.value}{phase} "
            f"cost=${p.cost_usd:.4f} requester={p.requested_by}",
            markup=False,
        )
        console.print(f"  {p.description[:120]}", markup=False)
    return 0


async def _run_deploy(config, args: argparse.Namespace) -> int:
    """Print deploy guide for a completed project."""
    from autoforge.engine.project_registry import ProjectRegistry, ProjectStatus

    project_id = args.project_id
    requested_by = _normalize_cli_requester(getattr(args, "requester", None))
    allow_all = bool(getattr(args, "all", False))

    async with ProjectRegistry(config.db_path) as registry:
        try:
            if allow_all:
                project = await registry.get(project_id)
            else:
                project = await registry.get_for_requester(project_id, requested_by)
        except KeyError:
            console.print("[red]Project not found.[/red]")
            return 1

    if project.status != ProjectStatus.COMPLETED:
        console.print(f"[red]Project not completed:[/red] {project.status.value}")
        return 1

    workspace = Path(project.workspace_path).resolve()
    guide_path = (workspace / "DEPLOY_GUIDE.md").resolve()
    if not str(guide_path).startswith(str(workspace)):
        console.print("[red]Invalid deploy guide path.[/red]")
        return 1
    if not guide_path.exists():
        console.print("[red]Deploy guide not found.[/red]")
        return 1

    try:
        console.print(guide_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        console.print(f"[red]Failed to read deploy guide:[/red] {exc}")
        return 1
    return 0


def _resolve_command(argv: list[str]) -> tuple[argparse.Namespace, str | None]:
    """Parse CLI args, handling legacy bare-description mode.

    Returns (parsed_args, legacy_description_or_None).
    """
    parser = build_parser()

    # Pre-scan for legacy bare description: python forge.py "Build a todo app"
    # If the first non-flag arg isn't a known subcommand, treat as generate.
    legacy_description = None
    clean_argv = list(argv)
    for i, arg in enumerate(argv):
        if arg.startswith("-"):
            continue
        if arg not in _KNOWN_COMMANDS:
            legacy_description = arg
            clean_argv = argv[:i] + argv[i + 1:]
        break

    args = parser.parse_args(clean_argv)
    return args, legacy_description


def _run_interactive_sync(args: argparse.Namespace) -> int:
    """Run interactive mode — all InquirerPy prompts happen here (sync).

    Returns exit code. May call asyncio.run() for the actual pipeline.
    """
    from autoforge.cli.display import show_banner
    from autoforge.cli.setup_wizard import needs_setup

    show_banner()

    # Check if setup is needed — runs InquirerPy (sync, no event loop)
    if needs_setup():
        console.print("[yellow]First time? Let's set up AutoForge.[/yellow]\n")
        from autoforge.cli.setup_wizard import run_setup_wizard
        run_setup_wizard()
        console.print()

    try:
        from autoforge.cli.interactive import run_interactive
    except ImportError:
        console.print("[red]Error:[/red] InquirerPy not installed. Run: pip install InquirerPy")
        console.print("Or use subcommands directly: autoforge generate \"your description\"")
        return 1

    # Run InquirerPy menus (sync, no event loop)
    choices = run_interactive()
    action = choices.get("action")

    if action == "setup":
        from autoforge.cli.setup_wizard import run_setup_wizard
        run_setup_wizard()
        return 0

    # Build config from global config + interactive choices
    from autoforge.engine.config import ForgeConfig

    overrides = _build_config_overrides(args)
    # Apply interactive choices as overrides
    if "budget" in choices:
        overrides["budget_limit_usd"] = choices["budget"]
    if "max_agents" in choices:
        overrides["max_agents"] = choices["max_agents"]
    if "mode" in choices:
        overrides["mode"] = choices["mode"]
    if "mobile_target" in choices:
        overrides["mobile_target"] = choices["mobile_target"]

    config = ForgeConfig.from_env(**overrides)
    setup_logging(config.log_level, config.verbose)

    # Now enter the async event loop only for the pipeline execution
    if action == "generate":
        return asyncio.run(_run_generate(config, choices["description"]))
    elif action == "review":
        return asyncio.run(_run_review(config, choices["project_path"]))
    elif action == "import":
        return asyncio.run(_run_import(
            config,
            choices["project_path"],
            choices.get("enhance_description", ""),
        ))

    return 0


async def _async_dispatch(args: argparse.Namespace, overrides: dict) -> int:
    """Dispatch a resolved subcommand to its async handler."""
    from autoforge.engine.config import ForgeConfig

    config = ForgeConfig.from_env(**overrides)
    setup_logging(config.log_level, config.verbose)

    if args.command == "generate":
        return await _run_generate(config, args.description)

    elif args.command == "review":
        return await _run_review(config, args.path)

    elif args.command == "import":
        enhance = getattr(args, "enhance", "")
        return await _run_import(config, args.path, enhance)

    elif args.command == "status":
        from autoforge.engine.orchestrator import Orchestrator
        orchestrator = Orchestrator(config)
        orchestrator.show_status()
        return 0

    elif args.command == "resume":
        from autoforge.engine.orchestrator import Orchestrator

        if not config.has_api_key:
            console.print("[red]Error:[/red] No API key configured. Run: autoforge setup")
            return 1

        orchestrator = Orchestrator(config)
        resume_path = getattr(args, "path", None)
        resume_path = Path(resume_path) if resume_path else None
        try:
            await orchestrator.resume(resume_path)
            return 0
        except Exception as e:
            console.print(f"[red]Resume failed:[/red] {e}")
            return 1

    elif args.command == "daemon":
        action = getattr(args, "daemon_action", "")
        if action == "start":
            return await _run_daemon_start(config)
        if action == "status":
            return await _run_daemon_status(config)
        if action == "stop":
            return await _run_daemon_stop(config)
        if action == "install":
            return await _run_daemon_install(config)
        console.print(f"[red]Unknown daemon action:[/red] {action}")
        return 1

    elif args.command == "queue":
        return await _run_queue(config, args)

    elif args.command == "projects":
        return await _run_projects(config, args)

    elif args.command == "deploy":
        return await _run_deploy(config, args)

    return 1


def main() -> None:
    """Synchronous entry point.

    All InquirerPy interactions (setup wizard, interactive menus) run here
    in the sync context. Only orchestrator pipelines use asyncio.run().
    """
    args, legacy_description = _resolve_command(sys.argv[1:])

    # Handle: no command at all -> interactive mode or legacy mode
    if args.command is None:
        # Legacy: python forge.py --status
        if args.legacy_status:
            args.command = "status"
        # Legacy: python forge.py --resume
        elif args.legacy_resume is not None:
            args.command = "resume"
            args.path = None if args.legacy_resume == "auto" else args.legacy_resume
        # Legacy: python forge.py "description"
        elif legacy_description:
            args.command = "generate"
            args.description = legacy_description
        # No args at all -> interactive mode (sync, InquirerPy runs here)
        else:
            sys.exit(_run_interactive_sync(args))
            return

    # Subcommand: "setup" runs InquirerPy directly (sync, no event loop)
    if args.command == "setup":
        from autoforge.cli.setup_wizard import run_setup_wizard
        run_setup_wizard()
        sys.exit(0)

    # All other subcommands dispatch through asyncio
    overrides = _build_config_overrides(args)
    sys.exit(asyncio.run(_async_dispatch(args, overrides)))


if __name__ == "__main__":
    main()
