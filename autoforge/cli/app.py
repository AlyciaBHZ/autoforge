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
    harness <action>    Run evaluation harness over datasets (JSONL)
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import inspect
import json
import logging
import os
import signal
import sys
import time
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
            "  autoforgeai                                 # Interactive session\n"
            '  autoforgeai generate "Build a Todo app"     # Generate new project\n'
            "  autoforgeai review ./my-project             # Review existing project\n"
            "  autoforgeai import ./my-project             # Import & improve\n"
            "  autoforgeai setup                           # Configure settings\n"
            "  autoforgeai status                          # Show project status\n"
            "  autoforgeai daemon start                    # Start queue daemon\n"
            '  autoforgeai queue "Build an API"            # Queue project\n'
            "  autoforgeai projects                        # List queued/history\n"
            "  autoforgeai deploy <project_id>             # Print deploy guide\n"
            '  autoforgeai paper infer "goal text"         # Infer likely ICLR papers\n'
            "  autoforgeai paper benchmark                 # Evaluate paper inference quality\n"
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
        "--trace",
        action="store_true",
        help="Enable trace export (FORGE_TRACE). Writes JSONL under .autoforge/traces/",
    )
    parser.add_argument(
        "--trace-llm",
        dest="trace_llm",
        action="store_true",
        help="Include LLM content in trace (FORGE_TRACE_LLM). May capture sensitive data.",
    )
    parser.add_argument(
        "--trace-cmd",
        dest="trace_cmd",
        action="store_true",
        help="Include command output in trace (FORGE_TRACE_CMD).",
    )
    parser.add_argument(
        "--trace-fs",
        dest="trace_fs",
        action="store_true",
        help="Include filesystem snapshots in trace (FORGE_TRACE_FS).",
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
    parser.add_argument(
        "--backend",
        choices=["auto", "docker", "subprocess", "slurm"],
        default=None,
        help="Execution backend for running commands: auto, docker, subprocess, slurm",
    )
    parser.add_argument("--slurm-partition", default=None, help="Slurm partition (backend=slurm)")
    parser.add_argument("--slurm-account", default=None, help="Slurm account (backend=slurm)")
    parser.add_argument("--slurm-qos", default=None, help="Slurm QoS (backend=slurm)")
    parser.add_argument("--slurm-cpus", type=int, default=None, help="Slurm cpus-per-task (backend=slurm)")
    parser.add_argument("--slurm-mem", default=None, help="Slurm memory, e.g. 4G (backend=slurm)")
    parser.add_argument("--slurm-gres", default=None, help="Slurm gres, e.g. gpu:1 (backend=slurm)")
    parser.add_argument("--slurm-queue-timeout", type=int, default=None, help="Extra seconds to wait for Slurm queueing")
    parser.add_argument(
        "--slurm-poll-interval",
        type=float,
        default=None,
        help="Polling interval seconds for Slurm job status",
    )
    parser.add_argument(
        "--slurm-local-in-alloc",
        dest="slurm_local_in_alloc",
        action="store_true",
        default=None,
        help="If inside a Slurm allocation, run commands locally instead of submitting sbatch",
    )
    parser.add_argument(
        "--no-slurm-local-in-alloc",
        dest="slurm_local_in_alloc",
        action="store_false",
        default=None,
        help="Always submit sbatch even inside a Slurm allocation",
    )
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=None,
        help="Deterministic mode: forces temperature=0 and stabilizes backoff jitter",
    )
    parser.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        default=None,
        help="Disable deterministic mode",
    )
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed (FORGE_SEED)")
    parser.add_argument(
        "--source-date-epoch",
        type=int,
        default=None,
        help="Set SOURCE_DATE_EPOCH inside sandboxes (FORGE_SOURCE_DATE_EPOCH)",
    )
    parser.add_argument("--pip-index-url", default=None, help="PIP_INDEX_URL injected into sandboxes (FORGE_PIP_INDEX_URL)")
    parser.add_argument("--pip-cache-dir", default=None, help="PIP_CACHE_DIR injected into sandboxes (FORGE_PIP_CACHE_DIR)")
    parser.add_argument("--npm-registry", default=None, help="NPM_CONFIG_REGISTRY injected into sandboxes (FORGE_NPM_REGISTRY)")
    parser.add_argument("--npm-cache-dir", default=None, help="NPM_CONFIG_CACHE injected into sandboxes (FORGE_NPM_CACHE_DIR)")
    parser.add_argument(
        "--llm-rate-limit",
        dest="llm_rate_limit",
        action="store_true",
        default=None,
        help="Enable global LLM rate limiting (FORGE_LLM_RATE_LIMIT)",
    )
    parser.add_argument(
        "--no-llm-rate-limit",
        dest="llm_rate_limit",
        action="store_false",
        default=None,
        help="Disable global LLM rate limiting",
    )
    parser.add_argument("--llm-rpm", type=int, default=None, help="Global LLM requests/minute (FORGE_LLM_RPM)")
    parser.add_argument("--llm-tpm", type=int, default=None, help="Global LLM tokens/minute (FORGE_LLM_TPM)")
    parser.add_argument("--llm-rate-db", default=None, help="SQLite path for global rate limiting (FORGE_LLM_RATE_LIMIT_DB)")
    parser.add_argument(
        "--llm-rate-namespace",
        default=None,
        help="Rate limiter namespace key (FORGE_LLM_RATE_LIMIT_NAMESPACE)",
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
    queue_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait and stream status updates until the project completes/fails",
    )
    queue_parser.add_argument(
        "--poll",
        type=float,
        default=3.0,
        help="Polling interval in seconds when using --wait (default: 3.0)",
    )
    queue_parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Timeout in seconds for --wait (0=never, default: 0)",
    )
    queue_parser.add_argument(
        "--tail",
        action="store_true",
        help="When using --wait, also tail task transition log once workspace is available",
    )

    # watch
    watch_parser = subparsers.add_parser(
        "watch",
        help="Watch a queued/building project until completion",
    )
    watch_parser.add_argument("project_id", help="Project id")
    watch_parser.add_argument(
        "--poll",
        type=float,
        default=3.0,
        help="Polling interval in seconds (default: 3.0)",
    )
    watch_parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Timeout in seconds (0=never, default: 0)",
    )
    watch_parser.add_argument(
        "--requester",
        default=None,
        help="Requester id for authorization checks (default: current user)",
    )
    watch_parser.add_argument(
        "--all",
        action="store_true",
        help="Bypass requester filter and read any project id",
    )
    watch_parser.add_argument(
        "--tail",
        action="store_true",
        help="Also tail task transition log once workspace is available",
    )

    # msg
    msg_parser = subparsers.add_parser(
        "msg",
        help="Send an async message/note to a queued/building project",
    )
    msg_parser.add_argument("project_id", help="Project id")
    msg_parser.add_argument("text", nargs="+", help="Message text")
    msg_parser.add_argument(
        "--requester",
        default=None,
        help="Requester id for authorization checks (default: current user)",
    )
    msg_parser.add_argument(
        "--all",
        action="store_true",
        help="Bypass requester filter and read any project id",
    )

    # unpause
    unpause_parser = subparsers.add_parser(
        "unpause",
        help="Re-queue a paused project so the daemon can resume it",
    )
    unpause_parser.add_argument("project_id", help="Project id")
    unpause_parser.add_argument(
        "--requester",
        default=None,
        help="Requester id for authorization checks (default: current user)",
    )
    unpause_parser.add_argument(
        "--all",
        action="store_true",
        help="Bypass requester filter and unpause any project id",
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

    # harness
    harness_parser = subparsers.add_parser("harness", help="Evaluation harness (batch datasets)")
    harness_subparsers = harness_parser.add_subparsers(dest="harness_action", required=True)

    harness_run = harness_subparsers.add_parser("run", help="Run a JSONL dataset")
    harness_run.add_argument("dataset", help="Path to dataset.jsonl")
    harness_run.add_argument("--out", default=None, help="Output directory (default: .autoforge/harness/runs/<run_id>)")
    harness_run.add_argument("--concurrency", type=int, default=1, help="Number of cases to run in parallel")
    harness_run.add_argument("--allow-subprocess", action="store_true", help="Allow falling back to SubprocessSandbox if Docker is unavailable (not recommended)")
    harness_run.add_argument("--no-trace", action="store_true", help="Disable trace export (not recommended)")

    harness_prewarm = harness_subparsers.add_parser("prewarm", help="Build/prewarm docker images referenced by dataset")
    harness_prewarm.add_argument("dataset", help="Path to dataset.jsonl")
    harness_prewarm.add_argument("--out", default=None, help="Output directory (default: .autoforge/harness/)")

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
    paper_repro.add_argument(
        "--strict-contract",
        action="store_true",
        help="Enforce contract artifacts/report schema; fail with exit code 2 on violations",
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
    "watch",
    "msg",
    "unpause",
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
    if getattr(args, "trace", False):
        overrides["trace_enabled"] = True
    if getattr(args, "trace_llm", False):
        overrides["trace_enabled"] = True
        overrides["trace_capture_llm_content"] = True
    if getattr(args, "trace_cmd", False):
        overrides["trace_enabled"] = True
        overrides["trace_capture_command_output"] = True
    if getattr(args, "trace_fs", False):
        overrides["trace_enabled"] = True
        overrides["trace_capture_fs_snapshots"] = True
    if getattr(args, "confirm", None) is not None:
        overrides["confirm_phases"] = [p.strip() for p in args.confirm.split(",")]
    if getattr(args, "tdd", None) is not None:
        overrides["build_test_loops"] = max(0, min(args.tdd, 5))
    if getattr(args, "backend", None) is not None:
        overrides["execution_backend"] = args.backend
    if getattr(args, "slurm_partition", None) is not None:
        overrides["slurm_partition"] = args.slurm_partition
    if getattr(args, "slurm_account", None) is not None:
        overrides["slurm_account"] = args.slurm_account
    if getattr(args, "slurm_qos", None) is not None:
        overrides["slurm_qos"] = args.slurm_qos
    if getattr(args, "slurm_cpus", None) is not None:
        overrides["slurm_cpus_per_task"] = args.slurm_cpus
    if getattr(args, "slurm_mem", None) is not None:
        overrides["slurm_mem"] = args.slurm_mem
    if getattr(args, "slurm_gres", None) is not None:
        overrides["slurm_gres"] = args.slurm_gres
    if getattr(args, "slurm_queue_timeout", None) is not None:
        overrides["slurm_queue_timeout_seconds"] = args.slurm_queue_timeout
    if getattr(args, "slurm_poll_interval", None) is not None:
        overrides["slurm_poll_interval_seconds"] = args.slurm_poll_interval
    if getattr(args, "slurm_local_in_alloc", None) is not None:
        overrides["slurm_use_local_in_allocation"] = bool(args.slurm_local_in_alloc)
    if getattr(args, "deterministic", None) is not None:
        overrides["deterministic"] = bool(args.deterministic)
    if getattr(args, "seed", None) is not None:
        overrides["deterministic_seed"] = int(args.seed)
    if getattr(args, "source_date_epoch", None) is not None:
        overrides["deterministic_source_date_epoch"] = int(args.source_date_epoch)
    if getattr(args, "pip_index_url", None) is not None:
        overrides["pip_index_url"] = args.pip_index_url
    if getattr(args, "pip_cache_dir", None) is not None:
        overrides["pip_cache_dir"] = args.pip_cache_dir
    if getattr(args, "npm_registry", None) is not None:
        overrides["npm_registry"] = args.npm_registry
    if getattr(args, "npm_cache_dir", None) is not None:
        overrides["npm_cache_dir"] = args.npm_cache_dir
    if getattr(args, "llm_rate_limit", None) is not None:
        overrides["llm_rate_limit_enabled"] = bool(args.llm_rate_limit)
    if getattr(args, "llm_rpm", None) is not None:
        overrides["llm_rpm_limit"] = int(args.llm_rpm)
    if getattr(args, "llm_tpm", None) is not None:
        overrides["llm_tpm_limit"] = int(args.llm_tpm)
    if getattr(args, "llm_rate_db", None) is not None:
        overrides["llm_rate_limit_db_path"] = args.llm_rate_db
    if getattr(args, "llm_rate_namespace", None) is not None:
        overrides["llm_rate_limit_namespace"] = args.llm_rate_namespace
    return overrides


async def _close_orchestrator_llm(orchestrator: Any) -> None:
    """Best-effort close of LLM client pools for interactive asyncio.run loops."""
    llm = getattr(orchestrator, "llm", None)
    if llm is None:
        return
    close_fn = getattr(llm, "close", None)
    if close_fn is None:
        return
    try:
        result = close_fn()
        if inspect.isawaitable(result):
            await result
    except Exception:
        logging.getLogger(__name__).debug("Failed to close orchestrator LLM clients", exc_info=True)


async def _run_generate(config, description: str) -> int:
    """Run the generate pipeline."""
    from autoforge.cli.display import show_startup_info
    from autoforge.engine.orchestrator import Orchestrator

    if not config.has_api_key:
        console.print("[red]Error:[/red] No API key configured. Run: autoforgeai setup")
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
        run_id=str(getattr(config, "run_id", "")),
        trace_enabled=bool(getattr(config, "trace_enabled", False)),
        trace_llm=bool(getattr(config, "trace_capture_llm_content", False)),
        trace_cmd=bool(getattr(config, "trace_capture_command_output", False)),
        trace_fs=bool(getattr(config, "trace_capture_fs_snapshots", False)),
    )

    orchestrator = Orchestrator(config)
    try:
        await orchestrator.run(description)
        return 0
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.getLogger(__name__).debug("Full traceback:", exc_info=True)
        return 1
    finally:
        await _close_orchestrator_llm(orchestrator)


async def _run_review(config, project_path: str) -> int:
    """Run the review pipeline."""
    from autoforge.cli.display import show_startup_info
    from autoforge.engine.orchestrator import Orchestrator

    if not config.has_api_key:
        console.print("[red]Error:[/red] No API key configured. Run: autoforgeai setup")
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
        run_id=str(getattr(config, "run_id", "")),
        trace_enabled=bool(getattr(config, "trace_enabled", False)),
        trace_llm=bool(getattr(config, "trace_capture_llm_content", False)),
        trace_cmd=bool(getattr(config, "trace_capture_command_output", False)),
        trace_fs=bool(getattr(config, "trace_capture_fs_snapshots", False)),
    )

    orchestrator = Orchestrator(config)
    try:
        await orchestrator.review_project(str(path))
        return 0
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.getLogger(__name__).debug("Full traceback:", exc_info=True)
        return 1
    finally:
        await _close_orchestrator_llm(orchestrator)


async def _run_import(config, project_path: str, enhance: str = "") -> int:
    """Run the import pipeline."""
    from autoforge.cli.display import show_startup_info
    from autoforge.engine.orchestrator import Orchestrator

    if not config.has_api_key:
        console.print("[red]Error:[/red] No API key configured. Run: autoforgeai setup")
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
        run_id=str(getattr(config, "run_id", "")),
        trace_enabled=bool(getattr(config, "trace_enabled", False)),
        trace_llm=bool(getattr(config, "trace_capture_llm_content", False)),
        trace_cmd=bool(getattr(config, "trace_capture_command_output", False)),
        trace_fs=bool(getattr(config, "trace_capture_fs_snapshots", False)),
    )

    orchestrator = Orchestrator(config)
    try:
        await orchestrator.import_project(str(path), enhance)
        return 0
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.getLogger(__name__).debug("Full traceback:", exc_info=True)
        return 1
    finally:
        await _close_orchestrator_llm(orchestrator)


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
        console.print("[red]Error:[/red] No API key configured. Run: autoforgeai setup")
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
        console.print("Run daemon manually: autoforgeai daemon start")
    return 0


def _clamp_float(value: float, *, minimum: float, maximum: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(v, maximum))


async def _tail_task_transitions(
    *,
    project_dir: Path,
    stop_event: asyncio.Event,
    poll_seconds: float = 0.5,
    replay_last: int = 50,
) -> None:
    """Tail ``.forge_task_transition_log.jsonl`` and print task transitions.

    Designed for codex-like visibility when using daemon mode.
    """
    from rich.text import Text

    from autoforge.engine.task_dag import TaskDAG

    poll = _clamp_float(poll_seconds, minimum=0.1, maximum=5.0)
    replay_last = max(0, min(int(replay_last), 500))

    log_path = project_dir / ".forge_task_transition_log.jsonl"
    plan_path = project_dir / "dev_plan.json"
    plan_mtime: float | None = None
    task_map: dict[str, tuple[str, list[str]]] = {}

    def _maybe_reload_plan() -> None:
        nonlocal plan_mtime, task_map
        try:
            if not plan_path.exists():
                return
            mtime = float(plan_path.stat().st_mtime)
            if plan_mtime is not None and mtime == plan_mtime:
                return
            dag = TaskDAG.load(plan_path)
            task_map = {
                t.id: (str(t.description or ""), list(t.files or []))
                for t in dag.get_all_tasks()
            }
            plan_mtime = mtime
        except Exception:
            return

    while not stop_event.is_set() and not log_path.exists():
        await asyncio.sleep(poll)
    if stop_event.is_set():
        return

    console.print(Text(f"[tail] {log_path}", style="dim"))

    try:
        _maybe_reload_plan()
    except Exception:
        pass

    def _print_entry(entry: dict[str, Any]) -> None:
        try:
            ts = float(entry.get("ts", 0.0) or 0.0)
        except Exception:
            ts = 0.0
        stamp = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else time.strftime("%H:%M:%S")
        task_id = str(entry.get("task_id", "") or "")
        to_status = str(entry.get("to_status", "") or "")
        reason = str(entry.get("reason", "") or "")
        agent_id = str(entry.get("agent_id", "") or "")

        status_style = {
            "in_progress": "yellow",
            "done": "green",
            "failed": "red",
            "blocked": "red",
            "todo": "dim",
        }.get(to_status, "cyan")

        desc, files = task_map.get(task_id, ("", []))
        desc = (desc or "").strip()
        if len(desc) > 90:
            desc = desc[:90] + "..."

        files_part = ""
        if files:
            preview = ", ".join(str(f) for f in files[:3] if str(f))
            suffix = "…" if len(files) > 3 else ""
            files_part = f"[{preview}{suffix}]"

        reason_part = ""
        if reason:
            reason_short = reason.replace("\n", " ").strip()
            if len(reason_short) > 120:
                reason_short = reason_short[:120] + "..."
            reason_part = f" ({reason_short})"

        line = Text()
        line.append(stamp, style="dim")
        line.append(" ")
        line.append(to_status or "transition", style=status_style)
        if task_id:
            line.append(" ")
            line.append(task_id, style="bold")
        if agent_id:
            line.append(f" @{agent_id}", style="dim")
        if desc:
            line.append(" - ")
            line.append(desc)
        if files_part:
            line.append(" ")
            line.append(files_part, style="dim")
        if reason_part:
            line.append(reason_part, style="dim")
        console.print(line)

    try:
        if replay_last > 0:
            try:
                existing = log_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                existing = []
            for raw in existing[-replay_last:]:
                s = (raw or "").strip()
                if not s:
                    continue
                try:
                    payload = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    _maybe_reload_plan()
                    _print_entry(payload)
    except Exception:
        pass

    with log_path.open("r", encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        while not stop_event.is_set():
            line = f.readline()
            if not line:
                await asyncio.sleep(poll)
                continue
            s = line.strip()
            if not s:
                continue
            try:
                payload = json.loads(s)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            _maybe_reload_plan()
            _print_entry(payload)


async def _watch_project(
    *,
    config,
    project_id: str,
    requested_by: str,
    poll_seconds: float = 3.0,
    timeout_seconds: float = 0.0,
    allow_all: bool = False,
    tail: bool = False,
) -> int:
    """Poll the daemon registry and print project status changes."""
    from autoforge.engine.project_registry import ProjectRegistry, ProjectStatus
    from rich.text import Text

    poll_seconds = _clamp_float(poll_seconds, minimum=0.5, maximum=60.0)
    timeout_seconds = _clamp_float(timeout_seconds, minimum=0.0, maximum=7 * 24 * 3600.0)
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None

    console.print(f"[dim]Watching project[/dim] [bold]{project_id}[/bold] [dim](Ctrl+C to stop)[/dim]")
    console.print("[dim]Tip:[/dim] if it stays queued, run: [bold]autoforgeai daemon start[/bold]")

    last: tuple[str, str, float, str, str] | None = None
    stop_event = asyncio.Event()
    tail_task: asyncio.Task[None] | None = None
    async with ProjectRegistry(config.db_path) as registry:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                console.print("[yellow]Watch timed out.[/yellow]")
                stop_event.set()
                if tail_task is not None:
                    tail_task.cancel()
                    await asyncio.gather(tail_task, return_exceptions=True)
                return 2

            try:
                project = await (registry.get(project_id) if allow_all else registry.get_for_requester(project_id, requested_by))
            except KeyError:
                console.print("[red]Project not found.[/red]")
                stop_event.set()
                if tail_task is not None:
                    tail_task.cancel()
                    await asyncio.gather(tail_task, return_exceptions=True)
                return 1

            status = project.status.value
            phase = project.phase or ""
            cost = float(project.cost_usd or 0.0)
            workspace = project.workspace_path or ""
            error = project.error or ""

            cur = (status, phase, cost, workspace, error)
            if cur != last:
                stamp = time.strftime("%H:%M:%S")
                phase_part = f" ({phase})" if phase else ""
                status_style = {
                    ProjectStatus.QUEUED: "yellow",
                    ProjectStatus.BUILDING: "cyan",
                    ProjectStatus.COMPLETED: "green",
                    ProjectStatus.FAILED: "red",
                    ProjectStatus.CANCELLED: "red",
                    ProjectStatus.PAUSED: "yellow",
                }.get(project.status, "")

                line = Text()
                line.append(stamp, style="dim")
                line.append(" ")
                line.append(status, style=status_style)
                if phase_part:
                    line.append(phase_part, style="dim")
                line.append(f" cost=${cost:.4f}", style="magenta")
                console.print(line)

                if error and project.status in {ProjectStatus.FAILED, ProjectStatus.CANCELLED, ProjectStatus.PAUSED}:
                    err_style = "red" if project.status != ProjectStatus.PAUSED else "yellow"
                    console.print(Text("  ") + Text(error[:200], style=err_style))
                if workspace:
                    console.print(Text("  workspace: ", style="dim") + Text(workspace))
                last = cur

            if tail and workspace and tail_task is None:
                try:
                    project_dir = Path(workspace)
                except Exception:
                    project_dir = None
                if project_dir is not None and project_dir.exists():
                    tail_task = asyncio.create_task(
                        _tail_task_transitions(project_dir=project_dir, stop_event=stop_event),
                    )

            if project.status in {ProjectStatus.COMPLETED, ProjectStatus.FAILED, ProjectStatus.CANCELLED, ProjectStatus.PAUSED}:
                stop_event.set()
                if tail_task is not None:
                    tail_task.cancel()
                    await asyncio.gather(tail_task, return_exceptions=True)
                if project.status == ProjectStatus.COMPLETED:
                    return 0
                if project.status == ProjectStatus.PAUSED:
                    return 2
                return 1

            await asyncio.sleep(poll_seconds)


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
    if bool(getattr(args, "wait", False)):
        try:
            return await _watch_project(
                config=config,
                project_id=result.project.id,
                requested_by=requested_by,
                poll_seconds=float(getattr(args, "poll", 3.0)),
                timeout_seconds=float(getattr(args, "timeout", 0.0)),
                allow_all=False,
                tail=bool(getattr(args, "tail", False)),
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped watching.[/dim]")
            return 0
    return 0


async def _run_watch(config, args: argparse.Namespace) -> int:
    requested_by = _normalize_cli_requester(getattr(args, "requester", None))
    allow_all = bool(getattr(args, "all", False))
    try:
        return await _watch_project(
            config=config,
            project_id=str(getattr(args, "project_id", "")).strip(),
            requested_by=requested_by,
            poll_seconds=float(getattr(args, "poll", 3.0)),
            timeout_seconds=float(getattr(args, "timeout", 0.0)),
            allow_all=allow_all,
            tail=bool(getattr(args, "tail", False)),
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching.[/dim]")
        return 0


async def _run_msg(config, args: argparse.Namespace) -> int:
    """Send an async message to a queued/building project."""
    from autoforge.engine.project_registry import ProjectRegistry, ProjectStatus

    project_id = str(getattr(args, "project_id", "")).strip()
    raw_text = getattr(args, "text", [])
    text = " ".join(str(t) for t in (raw_text or [])).strip()
    if not project_id:
        console.print("[red]Project id is required.[/red]")
        return 1
    if not text:
        console.print("[red]Message text is required.[/red]")
        console.print("Usage: autoforgeai msg <project_id> <text>")
        return 1

    requested_by = _normalize_cli_requester(getattr(args, "requester", None))
    allow_all = bool(getattr(args, "all", False))
    async with ProjectRegistry(config.db_path) as registry:
        try:
            project = await (registry.get(project_id) if allow_all else registry.get_for_requester(project_id, requested_by))
        except KeyError:
            console.print("[red]Project not found.[/red]")
            return 1

        if project.status not in {ProjectStatus.QUEUED, ProjectStatus.BUILDING, ProjectStatus.PAUSED}:
            console.print(
                f"[red]Project is {project.status.value}; cannot accept messages for this state.[/red]",
                markup=False,
            )
            return 1

        try:
            await registry.add_message(
                project_id,
                text=text,
                kind="user_note",
                source=requested_by,
            )
        except Exception as exc:
            console.print(f"[red]Failed to enqueue message:[/red] {exc}")
            return 1

    console.print("[green]Message queued.[/green] It will be applied at the next safe point.")
    return 0


async def _run_unpause(config, args: argparse.Namespace) -> int:
    """Re-queue a paused project for daemon resume."""
    from autoforge.engine.project_registry import ProjectRegistry

    project_id = str(getattr(args, "project_id", "")).strip()
    if not project_id:
        console.print("[red]Project id is required.[/red]")
        return 1

    requested_by = _normalize_cli_requester(getattr(args, "requester", None))
    allow_all = bool(getattr(args, "all", False))
    async with ProjectRegistry(config.db_path) as registry:
        try:
            ok = await (registry.unpause(project_id) if allow_all else registry.unpause_for_requester(project_id, requested_by))
        except Exception as exc:
            console.print(f"[red]Failed to unpause:[/red] {exc}")
            return 1

    if ok:
        console.print(f"[green]Project {project_id} re-queued.[/green]")
        return 0
    console.print("[yellow]Nothing to unpause.[/yellow] (not paused or not owned by you)")
    return 1


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
    from autoforge.engine.repro_contract import (
        REQUIRED_ARTIFACT_FILES,
        build_repro_report,
        build_run_manifest,
        derive_contract_status,
        validate_contract_artifacts,
        write_contract_outputs,
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
    strict_contract = bool(getattr(args, "strict_contract", False))

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

    artifacts_written: list[str] = []

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
        artifacts_written.append("candidate.json")
        (out_dir / "reproduction_brief.md").write_text(brief, encoding="utf-8")
        artifacts_written.append("reproduction_brief.md")
        (out_dir / "generation_prompt.txt").write_text(prompt, encoding="utf-8")
        artifacts_written.append("generation_prompt.txt")
        (out_dir / "paper_signals.json").write_text(
            json.dumps(signals.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifacts_written.append("paper_signals.json")
        (out_dir / "verification_plan.json").write_text(
            json.dumps(verification_plan, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifacts_written.append("verification_plan.json")
        (out_dir / "environment_spec.json").write_text(
            json.dumps(env_spec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifacts_written.append("environment_spec.json")
    except OSError as exc:
        console.print(f"[red]Failed to write reproduction artifacts:[/red] {exc}")
        return 1

    console.print(f"[bold]Selected paper:[/bold] {paper.title}")
    console.print(f"score={chosen.score:.2f}  matched={', '.join(chosen.matched_terms[:10]) or '-'}")
    console.print(f"openreview: {paper.openreview_url}")
    console.print(f"signal_source: {signals.text_source}")
    console.print(f"artifacts: {out_dir}")

    run_generate = bool(getattr(args, "run_generate", False))
    run_mode = "artifact_only"
    exit_code = 0
    failure_reasons: list[str] = []
    p0_p4_status = derive_contract_status(
        inference_score=chosen.score,
        datasets=signals.datasets,
        metrics=signals.metrics,
        claimed_metrics=signals.claimed_metrics,
        hardware_hints=signals.hardware_hints,
        run_generate=run_generate,
        has_api_key=bool(config.has_api_key),
    )

    if run_generate:
        if not config.has_api_key:
            run_mode = "simulated_no_api_key"
            feedback = simulate_pipeline_feedback(
                goal=goal,
                paper=paper,
                signals=signals,
                inference_score=chosen.score,
            )
            p0_p4_status = feedback.get("p0_p4_status", p0_p4_status)
            try:
                (out_dir / "simulated_pipeline_feedback.json").write_text(
                    json.dumps(feedback, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                artifacts_written.append("simulated_pipeline_feedback.json")
            except OSError as exc:
                console.print(f"[red]Failed to write simulated feedback:[/red] {exc}")
                failure_reasons.append(f"simulated feedback write failed: {exc}")
                exit_code = 1
            else:
                console.print(
                    "[yellow]No API key detected.[/yellow] "
                    "Generated simulated pipeline feedback at "
                    f"{out_dir / 'simulated_pipeline_feedback.json'}"
                )
        else:
            from autoforge.engine.orchestrator import Orchestrator

            run_mode = "generated_with_api_key"
            console.print("[cyan]Running full AutoForge generation from inferred paper prompt...[/cyan]")
            orchestrator = Orchestrator(config)
            try:
                project_dir = await orchestrator.run(prompt)
                console.print(f"[green]Generation completed:[/green] {project_dir}")
            except Exception as exc:
                run_mode = "generation_failed"
                failure_reasons.append(f"generation failed: {exc}")
                console.print(f"[red]Generation failed:[/red] {exc}")
                exit_code = 1
            finally:
                await _close_orchestrator_llm(orchestrator)

    manifest = build_run_manifest(
        run_id=f"{config.run_id}-{paper.note_id}",
        goal=goal,
        mode=run_mode,
        profile=env_spec.get("profile", "theory-first"),
        strict_contract=strict_contract,
        output_dir=out_dir,
        paper={
            "note_id": paper.note_id,
            "title": paper.title,
            "year": paper.year,
            "openreview_url": paper.openreview_url,
            "pdf_url": paper.pdf_url,
        },
        artifacts_written=artifacts_written + ["run_manifest.json", "repro_report.json", "repro_report.md"],
    )
    report = build_repro_report(
        run_id=f"{config.run_id}-{paper.note_id}",
        paper_id=paper.note_id,
        goal=goal,
        mode=run_mode,
        profile=env_spec.get("profile", "theory-first"),
        output_dir=out_dir,
        strict_contract=strict_contract,
        p0_p4_status=p0_p4_status,
        artifacts_complete=True,
        failure_reasons=failure_reasons,
    )
    try:
        write_contract_outputs(output_dir=out_dir, manifest=manifest, report=report)
        artifacts_written.extend(["run_manifest.json", "repro_report.json", "repro_report.md"])
    except OSError as exc:
        console.print(f"[red]Failed to write contract outputs:[/red] {exc}")
        return 1

    validation = validate_contract_artifacts(out_dir)
    if not validation.ok:
        merged_failures = list(failure_reasons) + validation.errors
        report = build_repro_report(
            run_id=f"{config.run_id}-{paper.note_id}",
            paper_id=paper.note_id,
            goal=goal,
            mode=run_mode,
            profile=env_spec.get("profile", "theory-first"),
            output_dir=out_dir,
            strict_contract=strict_contract,
            p0_p4_status=p0_p4_status,
            artifacts_complete=False,
            failure_reasons=merged_failures,
        )
        manifest = build_run_manifest(
            run_id=f"{config.run_id}-{paper.note_id}",
            goal=goal,
            mode=run_mode,
            profile=env_spec.get("profile", "theory-first"),
            strict_contract=strict_contract,
            output_dir=out_dir,
            paper={
                "note_id": paper.note_id,
                "title": paper.title,
                "year": paper.year,
                "openreview_url": paper.openreview_url,
                "pdf_url": paper.pdf_url,
            },
            artifacts_written=artifacts_written,
        )
        try:
            write_contract_outputs(output_dir=out_dir, manifest=manifest, report=report)
        except OSError:
            pass

        console.print("[red]Contract validation failed.[/red]")
        for err in validation.errors:
            console.print(f"  - {err}")
        if strict_contract:
            return 2
        console.print("[yellow]Continuing because --strict-contract is not enabled.[/yellow]")
    elif strict_contract and report.get("pass_fail") != "pass":
        console.print(
            "[red]Strict contract failed:[/red] run artifacts are valid, "
            "but reproduction outcome is not marked PASS."
        )
        for reason in report.get("failure_reasons", []):
            console.print(f"  - {reason}")
        return 2

    expected = ", ".join(REQUIRED_ARTIFACT_FILES)
    console.print(f"[dim]contract artifacts:[/dim] {expected}")
    return exit_code


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
        console.print("Or use subcommands directly: autoforgeai generate \"your description\"")
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
            console.print("[red]Error:[/red] No API key configured. Run: autoforgeai setup")
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
        finally:
            await _close_orchestrator_llm(orchestrator)

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

    elif args.command == "watch":
        return await _run_watch(config, args)
    elif args.command == "msg":
        return await _run_msg(config, args)
    elif args.command == "unpause":
        return await _run_unpause(config, args)

    elif args.command == "projects":
        return await _run_projects(config, args)

    elif args.command == "deploy":
        return await _run_deploy(config, args)

    elif args.command == "harness":
        from autoforge.engine.harness.runner import (
            HarnessRunConfig,
            prewarm_dataset_images,
            run_dataset,
        )

        action = getattr(args, "harness_action", "")
        dataset_path = Path(getattr(args, "dataset", "")).resolve()
        if not dataset_path.is_file():
            console.print(f"[red]Error:[/red] Dataset not found: {dataset_path}")
            return 1

        if action == "prewarm":
            out = getattr(args, "out", None)
            out_dir = (
                Path(out).resolve()
                if out
                else (config.project_root / ".autoforge" / "harness").resolve()
            )
            try:
                mapping = await prewarm_dataset_images(dataset_path, out_dir=out_dir)
                console.print(f"[green]Prewarm complete[/green] ({len(mapping)} images)")
                return 0
            except Exception as e:
                console.print(f"[red]Prewarm failed:[/red] {e}")
                logging.getLogger(__name__).debug("Harness prewarm failed", exc_info=True)
                return 1

        if action == "run":
            if not config.has_api_key:
                console.print("[red]Error:[/red] No API key configured. Run: autoforgeai setup")
                return 1

            out = getattr(args, "out", None)
            out_dir = Path(out).resolve() if out else None
            hcfg = HarnessRunConfig(
                concurrency=max(1, int(getattr(args, "concurrency", 1) or 1)),
                out_dir=out_dir,
                docker_required=not bool(getattr(args, "allow_subprocess", False)),
                trace_enabled=not bool(getattr(args, "no_trace", False)),
            )
            try:
                _ = await run_dataset(config, dataset_path, cfg=hcfg)
                return 0
            except Exception as e:
                console.print(f"[red]Harness run failed:[/red] {e}")
                logging.getLogger(__name__).debug("Harness run failed", exc_info=True)
                return 1

        console.print(f"[red]Unknown harness action:[/red] {action}")
        return 1

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
