"""ForgeDaemon — persistent background service for 24/7 project building.

Runs an async event loop that:
1. Polls the project queue (SQLite)
2. Dequeues the next project
3. Runs the Orchestrator pipeline
4. Records results + notifies the requester
5. Generates deployment guide
6. Repeats
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from autoforge.engine.config import ForgeConfig
from autoforge.engine.deploy_guide import generate_deploy_guide
from autoforge.engine.orchestrator import Orchestrator
from autoforge.engine.project_registry import ProjectRegistry, ProjectStatus

logger = logging.getLogger(__name__)
console = Console()


class ForgeDaemon:
    """Persistent daemon that processes build requests from the queue."""

    def __init__(self, config: ForgeConfig) -> None:
        self.config = config
        self.registry = ProjectRegistry(config.db_path)
        self._running = False
        self._current_project_id: str | None = None
        self._notify_callback: Any = None  # set by channels

    async def start(self) -> None:
        """Start the daemon event loop."""
        self._running = True
        await self.registry.open()

        console.print("[bold green]AutoForge Daemon started[/bold green]")
        console.print(f"  Database: {self.config.db_path}")
        console.print(f"  Workspace: {self.config.workspace_dir}")
        console.print(f"  Poll interval: {self.config.daemon_poll_interval}s")

        # Start channel tasks
        channel_tasks = []

        if self.config.telegram_token:
            try:
                from autoforge.engine.channels.telegram_bot import start_telegram_bot
                task = asyncio.create_task(
                    start_telegram_bot(self.config, self.registry, self._on_notify)
                )
                channel_tasks.append(task)
                console.print("  Telegram bot: [green]active[/green]")
            except ImportError:
                console.print("  Telegram bot: [yellow]skipped[/yellow] (pip install python-telegram-bot)")
                logger.warning("python-telegram-bot not installed, skipping Telegram")
            except Exception as e:
                console.print(f"  Telegram bot: [red]failed[/red] ({e})")
                logger.error(f"Failed to start Telegram bot: {e}", exc_info=True)

        if self.config.webhook_enabled:
            try:
                from autoforge.engine.channels.webhook import start_webhook_server
                task = asyncio.create_task(
                    start_webhook_server(self.config, self.registry)
                )
                channel_tasks.append(task)
                console.print(f"  Webhook API: [green]http://{self.config.webhook_host}:{self.config.webhook_port}[/green]")
            except ImportError:
                console.print("  Webhook API: [yellow]skipped[/yellow] (pip install fastapi uvicorn)")
                logger.warning("fastapi/uvicorn not installed, skipping webhook")
            except Exception as e:
                console.print(f"  Webhook API: [red]failed[/red] ({e})")
                logger.error(f"Failed to start webhook: {e}", exc_info=True)

        console.print()

        # Install signal handlers
        if sys.platform != "win32":
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, self._handle_signal)

        # Main build loop
        try:
            await self._build_loop()
        finally:
            self._running = False
            for t in channel_tasks:
                t.cancel()
            if channel_tasks:
                await asyncio.gather(*channel_tasks, return_exceptions=True)
            try:
                await self.registry.close()
            except Exception as e:
                logger.error(f"Failed to close registry: {e}")
            console.print("[bold]AutoForge Daemon stopped[/bold]")

    async def _build_loop(self) -> None:
        """Main loop: dequeue → build → record → notify → repeat."""
        while self._running:
            project = await self.registry.dequeue()

            if project is None:
                # Nothing in queue — wait and try again
                await asyncio.sleep(self.config.daemon_poll_interval)
                continue

            self._current_project_id = project.id
            console.print(f"[bold blue]Building:[/bold blue] {project.id} — {project.description[:80]}")

            project_config = None
            try:
                # Create a per-project config with its own budget
                project_config = ForgeConfig.from_env(
                    budget_limit_usd=project.budget_usd,
                )

                orchestrator = Orchestrator(project_config)

                # Hook into orchestrator to track phase changes
                original_save = orchestrator._save_state

                def _track_phase(phase: str, _orig=original_save, _pid=project.id) -> None:
                    _orig(phase)
                    asyncio.ensure_future(self.registry.update_phase(_pid, phase))

                orchestrator._save_state = _track_phase  # type: ignore[assignment]

                # Run the full pipeline
                project_dir = await orchestrator.run(project.description)
                cost = project_config.estimated_cost_usd
                project_name = project_dir.name

                # Update registry
                await self.registry.update_name(project.id, project_name, str(project_dir))
                await self.registry.mark_completed(project.id, cost)

                # Generate deploy guide
                guide = generate_deploy_guide(project_dir, project_name)
                guide_path = project_dir / "DEPLOY_GUIDE.md"
                guide_path.write_text(guide, encoding="utf-8")

                console.print(f"[bold green]Completed:[/bold green] {project_name} (${cost:.4f})")
                console.print(f"  Location: {project_dir}")
                console.print(f"  Deploy guide: {guide_path}")

                # Notify requester
                await self._notify(
                    project.requested_by,
                    f"Project '{project_name}' completed!\n"
                    f"Cost: ${cost:.4f}\n"
                    f"Location: {project_dir}\n"
                    f"Deploy guide saved to DEPLOY_GUIDE.md",
                )

            except Exception as e:
                error_msg = str(e)[:500]
                cost = project_config.estimated_cost_usd if project_config is not None else 0.0
                logger.error(f"Build failed for {project.id}: {error_msg}", exc_info=True)
                await self.registry.mark_failed(project.id, error_msg, cost_usd=cost)

                console.print(f"[bold red]Failed:[/bold red] {project.id} — {error_msg[:80]}")

                await self._notify(
                    project.requested_by,
                    f"Build failed: {error_msg[:200]}",
                )

            finally:
                self._current_project_id = None

    def set_notify_callback(self, callback: Any) -> None:
        """Set the notification callback (used by Telegram bot)."""
        self._notify_callback = callback

    async def _notify(self, requested_by: str, message: str) -> None:
        """Send a notification to the requester."""
        if self._notify_callback:
            try:
                await self._notify_callback(requested_by, message)
            except Exception as e:
                logger.warning(f"Notification failed: {e}")

    async def _on_notify(self, requested_by: str, message: str) -> None:
        """Notification handler (passed to channels) — delegates to _notify."""
        await self._notify(requested_by, message)

    def _handle_signal(self) -> None:
        """Handle shutdown signal."""
        console.print("\n[yellow]Shutting down...[/yellow]")
        self._running = False

    async def stop(self) -> None:
        """Stop the daemon."""
        self._running = False
