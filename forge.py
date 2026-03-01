#!/usr/bin/env python3
"""AutoForge — AI-powered multi-agent project generation.

Usage:
    python forge.py "Build a Todo app with user login and task management"
    python forge.py "做一个心理咨询预约平台" --budget 5.00
    python forge.py --resume
    python forge.py --status
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoForge: AI-powered project generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python forge.py "Build a Todo app with user login"\n'
            '  python forge.py "做一个博客网站" --budget 5.00\n'
            "  python forge.py --status\n"
            "  python forge.py --resume\n"
        ),
    )
    parser.add_argument(
        "description",
        nargs="?",
        help="Natural language project description",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Budget limit in USD (default: from .env or 10.0)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=None,
        help="Number of parallel builder agents (default: 3)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Default model for routine tasks",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume a previous run (optionally specify workspace path)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current project status",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed logs",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()

    # Build config from environment with CLI overrides
    overrides = {}
    if args.budget is not None:
        overrides["budget_limit_usd"] = args.budget
    if args.agents is not None:
        overrides["max_agents"] = args.agents
    if args.model is not None:
        overrides["model_fast"] = args.model
    if args.verbose:
        overrides["verbose"] = True
    if args.log_level:
        overrides["log_level"] = args.log_level

    config = ForgeConfig.from_env(**overrides)
    setup_logging(config.log_level, config.verbose)

    orchestrator = Orchestrator(config)

    # Handle --status
    if args.status:
        orchestrator.show_status()
        return 0

    # Handle --resume
    if args.resume is not None:
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
    if not args.description:
        console.print("[red]Error:[/red] Please provide a project description or use --resume/--status")
        console.print("Usage: python forge.py \"your project description\"")
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
        await orchestrator.run(args.description)
        return 0
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.getLogger(__name__).debug("Full traceback:", exc_info=True)
        return 1


def main() -> None:
    sys.exit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
