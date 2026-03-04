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
    paper <action>      Infer/reproduce ICLR papers from high-level goals
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
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
            "  forgeai                                     # Interactive session\n"
            '  forgeai generate "Build a Todo app"         # Generate new project\n'
            "  forgeai review ./my-project                 # Review existing project\n"
            "  forgeai import ./my-project                 # Import & improve\n"
            "  forgeai setup                               # Configure settings\n"
            "  forgeai status                              # Show project status\n"
            "  forgeai daemon start                        # Start queue daemon\n"
            '  forgeai queue "Build an API"                # Queue project\n'
            "  forgeai projects                            # List queued/history\n"
            "  forgeai deploy <project_id>                 # Print deploy guide\n"
            '  forgeai paper infer "goal text"             # Infer likely ICLR papers\n'
            "  forgeai paper benchmark                     # Evaluate paper inference quality\n"
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

    # paper
    from datetime import datetime

    default_year = datetime.now().year - 1
    paper_parser = subparsers.add_parser(
        "paper",
        help="Infer/reproduce ICLR papers from high-level goals",
    )
    paper_subparsers = paper_parser.add_subparsers(dest="paper_action", required=True)

    paper_infer = paper_subparsers.add_parser("infer", help="Infer likely ICLR papers from a goal")
    paper_infer.add_argument("goal", help="High-level research goal (no paper title)")
    paper_infer.add_argument("--year", type=int, default=default_year, help="ICLR year (default: previous year)")
    paper_infer.add_argument("--top-k", type=int, default=5, help="Show top-k paper candidates")
    paper_infer.add_argument(
        "--corpus-size",
        type=int,
        default=600,
        help="Number of papers to fetch from OpenReview (smaller is faster)",
    )
    paper_infer.add_argument(
        "--cache-hours",
        type=int,
        default=48,
        help="Reuse local corpus cache up to N hours (default: 48)",
    )
    paper_infer.add_argument(
        "--refresh-corpus",
        action="store_true",
        help="Bypass local cache and fetch fresh metadata from OpenReview",
    )

    paper_bench = paper_subparsers.add_parser(
        "benchmark",
        help="Benchmark goal->paper inference on real ICLR papers",
    )
    paper_bench.add_argument("--year", type=int, default=default_year, help="ICLR year")
    paper_bench.add_argument("--sample-size", type=int, default=5, help="How many benchmark cases to evaluate")
    paper_bench.add_argument("--top-k", type=int, default=5, help="Hit@k threshold")
    paper_bench.add_argument(
        "--corpus-size",
        type=int,
        default=600,
        help="Number of papers to fetch (smaller is faster)",
    )
    paper_bench.add_argument(
        "--cache-hours",
        type=int,
        default=48,
        help="Reuse local corpus cache up to N hours (default: 48)",
    )
    paper_bench.add_argument(
        "--refresh-corpus",
        action="store_true",
        help="Bypass local cache and fetch fresh metadata from OpenReview",
    )
    paper_bench.add_argument("--seed", type=int, default=42, help="Random seed for case sampling")
    paper_bench.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON benchmark report",
    )

    paper_repro = paper_subparsers.add_parser(
        "reproduce",
        help="Infer a paper and build reproduction brief/prompt",
    )
    paper_repro.add_argument("goal", help="High-level research goal")
    paper_repro.add_argument("--year", type=int, default=default_year, help="ICLR year")
    paper_repro.add_argument("--top-k", type=int, default=5, help="Candidate pool size")
    paper_repro.add_argument("--pick", type=int, default=1, help="1-based candidate index to reproduce")
    paper_repro.add_argument(
        "--corpus-size",
        type=int,
        default=600,
        help="Number of papers to fetch (smaller is faster)",
    )
    paper_repro.add_argument(
        "--cache-hours",
        type=int,
        default=48,
        help="Reuse local corpus cache up to N hours (default: 48)",
    )
    paper_repro.add_argument(
        "--refresh-corpus",
        action="store_true",
        help="Bypass local cache and fetch fresh metadata from OpenReview",
    )
    paper_repro.add_argument(
        "--with-pdf",
        action="store_true",
        help="Parse PDF text for richer signal extraction (extra network/CPU cost)",
    )
    paper_repro.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write reproduction artifacts",
    )
    paper_repro.add_argument(
        "--run-generate",
        action="store_true",
        help="Run full AutoForge generation using the built reproduction prompt",
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
    "paper",
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
        console.print("[red]Error:[/red] No API key configured. Run: forgeai setup")
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
        console.print("[red]Error:[/red] No API key configured. Run: forgeai setup")
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
        console.print("[red]Error:[/red] No API key configured. Run: forgeai setup")
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
        console.print("[red]Error:[/red] No API key configured. Run: forgeai setup")
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
        console.print("Run daemon manually: forgeai daemon start")
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


def _default_paper_report_path(config, year: int) -> Path:
    return config.project_root / ".autoforge" / "reports" / f"paper_benchmark_iclr{year}.json"


def _safe_slug(text: str, max_len: int = 64) -> str:
    cleaned = "".join(c.lower() if c.isalnum() else "-" for c in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part).strip("-")
    return (cleaned[:max_len].strip("-") or "paper")


async def _run_paper_infer(config, args: argparse.Namespace) -> int:
    """Infer likely ICLR papers from a short goal description."""
    from autoforge.engine.paper_repro import fetch_iclr_papers, infer_papers_from_goal

    goal = (args.goal or "").strip()
    if not goal:
        console.print("[red]Goal is required.[/red]")
        return 1

    year = int(getattr(args, "year", 0) or 0)
    top_k = max(1, min(int(getattr(args, "top_k", 5)), 20))
    corpus_size = max(50, min(int(getattr(args, "corpus_size", 600)), 2000))
    cache_hours = max(1, min(int(getattr(args, "cache_hours", 48)), 24 * 14))
    refresh = bool(getattr(args, "refresh_corpus", False))

    try:
        papers = await asyncio.to_thread(
            fetch_iclr_papers,
            year=year,
            limit=corpus_size,
            use_cache=not refresh,
            cache_max_age_hours=cache_hours,
        )
    except Exception as exc:
        console.print(f"[red]Failed to fetch ICLR papers:[/red] {exc}")
        return 1

    if not papers:
        console.print("[red]No papers fetched from OpenReview.[/red]")
        return 1

    ranked = infer_papers_from_goal(goal, papers, top_k=top_k)
    if not ranked:
        console.print("[yellow]No strong paper match found for this goal.[/yellow]")
        return 0

    console.print(f"[bold]Goal:[/bold] {goal}")
    console.print(f"[bold]ICLR {year} candidates:[/bold]")
    for idx, r in enumerate(ranked, start=1):
        matched = ", ".join(r.matched_terms[:8]) if r.matched_terms else "-"
        console.print(f"{idx}. score={r.score:.2f}  {r.paper.title}")
        console.print(f"   matched: {matched}")
        console.print(f"   {r.paper.openreview_url}")
    return 0


async def _run_paper_benchmark(config, args: argparse.Namespace) -> int:
    """Benchmark goal->paper inference on real ICLR papers."""
    from autoforge.engine.paper_repro import run_goal_inference_benchmark

    year = int(getattr(args, "year", 0) or 0)
    sample_size = max(3, min(int(getattr(args, "sample_size", 5)), 20))
    top_k = max(1, min(int(getattr(args, "top_k", 5)), 20))
    min_corpus = max(100, sample_size * 20)
    corpus_size = max(min_corpus, min(int(getattr(args, "corpus_size", 600)), 2000))
    seed = int(getattr(args, "seed", 42))
    cache_hours = max(1, min(int(getattr(args, "cache_hours", 48)), 24 * 14))
    refresh = bool(getattr(args, "refresh_corpus", False))

    try:
        report = await asyncio.to_thread(
            run_goal_inference_benchmark,
            year=year,
            sample_size=sample_size,
            corpus_size=corpus_size,
            top_k=top_k,
            seed=seed,
            use_cache=not refresh,
            cache_max_age_hours=cache_hours,
        )
    except Exception as exc:
        console.print(f"[red]Benchmark failed:[/red] {exc}")
        return 1

    out_arg = getattr(args, "output", None)
    out_path = Path(out_arg) if out_arg else _default_paper_report_path(config, year)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        console.print(f"[red]Failed to write benchmark report:[/red] {exc}")
        return 1

    console.print(f"[bold]ICLR {year} Goal-Inference Benchmark[/bold]")
    console.print(
        f"sample={report.sample_size} top_k={report.top_k} "
        f"hit@1={report.hit_at_1:.2%} hit@{report.top_k}={report.hit_at_k:.2%} "
        f"mrr={report.mrr:.3f}"
    )
    console.print(f"report: {out_path}")

    console.print("\nTop shortcomings:")
    for i, gap in enumerate(report.gaps, start=1):
        console.print(f"{i}. {gap}")
    return 0


async def _run_paper_reproduce(config, args: argparse.Namespace) -> int:
    """Infer one paper and generate reproduction artifacts/prompt."""
    from autoforge.engine.paper_repro import (
        build_environment_spec,
        build_generation_prompt,
        build_reproduction_brief,
        build_verification_plan,
        extract_paper_signals,
        fetch_iclr_papers,
        infer_papers_from_goal,
        simulate_pipeline_feedback,
    )

    goal = (args.goal or "").strip()
    if not goal:
        console.print("[red]Goal is required.[/red]")
        return 1

    year = int(getattr(args, "year", 0) or 0)
    top_k = max(1, min(int(getattr(args, "top_k", 5)), 20))
    pick = max(1, int(getattr(args, "pick", 1)))
    corpus_size = max(100, min(int(getattr(args, "corpus_size", 600)), 2000))
    cache_hours = max(1, min(int(getattr(args, "cache_hours", 48)), 24 * 14))
    refresh = bool(getattr(args, "refresh_corpus", False))

    try:
        papers = await asyncio.to_thread(
            fetch_iclr_papers,
            year=year,
            limit=corpus_size,
            use_cache=not refresh,
            cache_max_age_hours=cache_hours,
        )
    except Exception as exc:
        console.print(f"[red]Failed to fetch ICLR papers:[/red] {exc}")
        return 1

    ranked = infer_papers_from_goal(goal, papers, top_k=top_k)
    if not ranked:
        console.print("[red]No candidate paper inferred. Try a richer goal description.[/red]")
        return 1
    if pick > len(ranked):
        console.print(f"[red]--pick {pick} exceeds candidate count {len(ranked)}.[/red]")
        return 1

    chosen = ranked[pick - 1]
    paper = chosen.paper

    include_pdf = bool(getattr(args, "with_pdf", False))
    try:
        signals = await asyncio.to_thread(extract_paper_signals, paper, include_pdf=include_pdf)
    except Exception:
        signals = await asyncio.to_thread(extract_paper_signals, paper, include_pdf=False)

    brief = build_reproduction_brief(goal, paper, signals=signals)
    prompt = build_generation_prompt(goal, paper, signals=signals)
    verification_plan = build_verification_plan(signals)
    env_spec = build_environment_spec(paper, signals, theory_first=True)

    out_arg = getattr(args, "output_dir", None)
    if out_arg:
        out_dir = Path(out_arg)
    else:
        slug = _safe_slug(paper.title, max_len=48)
        out_dir = config.project_root / ".autoforge" / "paper_runs" / f"iclr{year}_{slug}_{paper.note_id}"

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidate.json").write_text(
            json.dumps(
                {
                    "goal": goal,
                    "score": chosen.score,
                    "matched_terms": chosen.matched_terms,
                    "paper": {
                        "title": paper.title,
                        "year": paper.year,
                        "openreview_url": paper.openreview_url,
                        "pdf_url": paper.pdf_url,
                        "note_id": paper.note_id,
                        "keywords": paper.keywords,
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (out_dir / "reproduction_brief.md").write_text(brief, encoding="utf-8")
        (out_dir / "generation_prompt.txt").write_text(prompt, encoding="utf-8")
        (out_dir / "paper_signals.json").write_text(
            json.dumps(signals.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / "verification_plan.json").write_text(
            json.dumps(verification_plan, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / "environment_spec.json").write_text(
            json.dumps(env_spec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        console.print(f"[red]Failed to write reproduction artifacts:[/red] {exc}")
        return 1

    console.print(f"[bold]Selected paper:[/bold] {paper.title}")
    console.print(f"score={chosen.score:.2f}  matched={', '.join(chosen.matched_terms[:10]) or '-'}")
    console.print(f"openreview: {paper.openreview_url}")
    console.print(f"signal_source: {signals.text_source}")
    console.print(f"artifacts: {out_dir}")

    if not getattr(args, "run_generate", False):
        return 0

    if not config.has_api_key:
        feedback = simulate_pipeline_feedback(
            goal=goal,
            paper=paper,
            signals=signals,
            inference_score=chosen.score,
        )
        try:
            (out_dir / "simulated_pipeline_feedback.json").write_text(
                json.dumps(feedback, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            console.print(f"[red]Failed to write simulated feedback:[/red] {exc}")
            return 1
        console.print(
            "[yellow]No API key detected.[/yellow] "
            "Generated simulated pipeline feedback at "
            f"{out_dir / 'simulated_pipeline_feedback.json'}"
        )
        return 0

    from autoforge.engine.orchestrator import Orchestrator

    console.print("[cyan]Running full AutoForge generation from inferred paper prompt...[/cyan]")
    orchestrator = Orchestrator(config)
    try:
        project_dir = await orchestrator.run(prompt)
    except Exception as exc:
        console.print(f"[red]Generation failed:[/red] {exc}")
        return 1

    console.print(f"[green]Generation completed:[/green] {project_dir}")
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
    """Run interactive session — setup on first run, then session loop.

    Flow: banner → first-run setup → mode select → build → (repeat)
    """
    from autoforge.cli.display import show_banner
    from autoforge.cli.setup_wizard import needs_setup

    show_banner()

    # First-run setup: API key, GitHub, preferences
    if needs_setup():
        console.print("[yellow]First time? Let's set up ForgeAI.[/yellow]\n")
        from autoforge.cli.setup_wizard import run_setup_wizard
        run_setup_wizard()
        console.print()

    try:
        from autoforge.cli.interactive import run_interactive
    except ImportError:
        console.print("[red]Error:[/red] InquirerPy not installed. Run: pip install InquirerPy")
        console.print("Or use subcommands directly: forgeai generate \"your description\"")
        return 1

    # Session loop: mode → action → build → next session
    while True:
        try:
            choices = run_interactive()
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            return 0

        action = choices.get("action")
        if action == "setup":
            from autoforge.cli.setup_wizard import run_setup_wizard
            run_setup_wizard()
            continue

        # Build config from global config + interactive choices
        from autoforge.engine.config import ForgeConfig

        overrides = _build_config_overrides(args)
        if "budget" in choices:
            overrides["budget_limit_usd"] = choices["budget"]
        if "max_agents" in choices:
            overrides["max_agents"] = choices["max_agents"]

        # Map interactive modes to engine modes
        mode = choices.get("mode", "developer")
        if mode == "academic":
            overrides["mode"] = "research"
            # Enable academic engines
            overrides["lean_prover_enabled"] = True
            overrides["theoretical_reasoning_enabled"] = True
            overrides["formal_verify_enabled"] = True
        elif mode == "verification":
            overrides["mode"] = "research"
            overrides["formal_verify_enabled"] = True
            overrides["security_scan_enabled"] = True
        else:
            overrides["mode"] = mode

        if "mobile_target" in choices:
            overrides["mobile_target"] = choices["mobile_target"]

        config = ForgeConfig.from_env(**overrides)
        setup_logging(config.log_level, config.verbose)

        # Execute the pipeline
        exit_code = 0
        if action == "generate":
            exit_code = asyncio.run(_run_generate(config, choices["description"]))
        elif action == "review":
            exit_code = asyncio.run(_run_review(config, choices["project_path"]))
        elif action == "import":
            exit_code = asyncio.run(_run_import(
                config,
                choices["project_path"],
                choices.get("enhance_description", ""),
            ))

        # Session complete — prompt for next
        console.print()
        if exit_code == 0:
            console.print("[bold green]Session complete![/bold green]")
        else:
            console.print("[bold yellow]Session ended with errors.[/bold yellow]")

        console.print("[dim]Starting next session... (Ctrl+C to exit)[/dim]\n")


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
            console.print("[red]Error:[/red] No API key configured. Run: forgeai setup")
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

    elif args.command == "paper":
        action = getattr(args, "paper_action", "")
        if action == "infer":
            return await _run_paper_infer(config, args)
        if action == "benchmark":
            return await _run_paper_benchmark(config, args)
        if action == "reproduce":
            return await _run_paper_reproduce(config, args)
        console.print(f"[red]Unknown paper action:[/red] {action}")
        return 1

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
