"""AutoForge CLI — interactive multi-mode entry point.

Subcommands:
    (no args)           Interactive mode with InquirerPy menus
    setup               First-run configuration wizard
    generate <desc>     Generate a new project
    review <path>       Review an existing project
    import <path>       Import & improve an existing project
    status              Show project status
    resume [path]       Resume an interrupted run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

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
            "  autoforge status                            # Show projects\n"
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
_KNOWN_COMMANDS = {"setup", "generate", "review", "import", "status", "resume"}


def _build_config_overrides(args: argparse.Namespace) -> dict:
    """Build config overrides from CLI args."""
    overrides = {}
    if args.budget is not None:
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
