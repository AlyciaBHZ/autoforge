"""ForgeDaemon - persistent background service for project queue processing."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Awaitable, Callable

from rich.console import Console

from autoforge.engine.config import ForgeConfig
from autoforge.engine.deploy_guide import generate_deploy_guide
from autoforge.engine.orchestrator import Orchestrator
from autoforge.engine.project_registry import Project, ProjectRegistry

logger = logging.getLogger(__name__)
console = Console()

Notifier = Callable[[str, str], Awaitable[None]]


class ForgeDaemon:
    """Persistent daemon that processes build requests from the queue."""

    def __init__(self, config: ForgeConfig) -> None:
        self.config = config
        self.registry = ProjectRegistry(config.db_path)
        self._running = False
        self._active_builds: dict[str, asyncio.Task[None]] = {}
        self._notifiers: dict[str, Notifier] = {}

    def register_notifier(self, channel: str, notifier: Notifier) -> None:
        """Register channel-specific completion notifier, e.g. ``telegram``."""
        self._notifiers[channel] = notifier

    def set_notify_callback(self, callback: Notifier) -> None:
        """Backward-compatible notifier registration."""
        self.register_notifier("*", callback)

    def _write_pid_file(self) -> None:
        pid_file = self.config.daemon_pid_file
        if pid_file is None:
            return
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()), encoding="utf-8")

    def _clear_pid_file(self) -> None:
        pid_file = self.config.daemon_pid_file
        if pid_file is None:
            return
        try:
            pid = pid_file.read_text(encoding="utf-8").strip()
            if pid and int(pid) == os.getpid():
                pid_file.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass

    def _fire_and_forget(self, coro: Awaitable[None], label: str) -> None:
        """Launch background task and log exceptions instead of dropping them."""
        task = asyncio.create_task(coro)

        def _done(t: asyncio.Task[None]) -> None:
            try:
                t.result()
            except Exception as exc:
                logger.debug("%s failed: %s", label, exc, exc_info=True)

        task.add_done_callback(_done)

    async def start(self) -> None:
        """Start the daemon event loop."""
        self._running = True
        await self.registry.open()
        self._write_pid_file()

        console.print("[bold green]AutoForge Daemon started[/bold green]")
        console.print(f"  Database: {self.config.db_path}")
        console.print(f"  Workspace: {self.config.workspace_dir}")
        console.print(f"  Poll interval: {self.config.daemon_poll_interval}s")
        console.print(f"  Max concurrent projects: {self.config.daemon_max_concurrent_projects}")

        channel_tasks: list[asyncio.Task[None]] = []

        if self.config.telegram_token:
            try:
                from autoforge.engine.channels.telegram_bot import start_telegram_bot

                task = asyncio.create_task(
                    start_telegram_bot(
                        self.config,
                        self.registry,
                        register_notifier=self.register_notifier,
                    )
                )
                channel_tasks.append(task)
                console.print("  Telegram bot: [green]active[/green]")
            except ImportError:
                console.print("  Telegram bot: [yellow]skipped[/yellow] (pip install autoforgeai[channels])")
                logger.warning("python-telegram-bot not installed, skipping Telegram")
            except Exception as exc:
                console.print(f"  Telegram bot: [red]failed[/red] ({exc})")
                logger.error("Failed to start Telegram bot: %s", exc, exc_info=True)

        if self.config.webhook_enabled:
            try:
                from autoforge.engine.channels.webhook import start_webhook_server

                task = asyncio.create_task(start_webhook_server(self.config, self.registry))
                channel_tasks.append(task)
                console.print(
                    f"  Webhook API: [green]http://{self.config.webhook_host}:{self.config.webhook_port}[/green]"
                )
            except ImportError:
                console.print("  Webhook API: [yellow]skipped[/yellow] (pip install autoforgeai[channels])")
                logger.warning("fastapi/uvicorn not installed, skipping webhook")
            except Exception as exc:
                console.print(f"  Webhook API: [red]failed[/red] ({exc})")
                logger.error("Failed to start webhook: %s", exc, exc_info=True)

        console.print()
        self._install_signal_handlers()

        try:
            await self._build_loop()
        finally:
            self._running = False
            if self._active_builds:
                await asyncio.gather(*self._active_builds.values(), return_exceptions=True)
                self._active_builds.clear()

            for task in channel_tasks:
                task.cancel()
            if channel_tasks:
                await asyncio.gather(*channel_tasks, return_exceptions=True)

            try:
                await self.registry.close()
            except Exception as exc:
                logger.error("Failed to close registry: %s", exc)

            self._clear_pid_file()
            console.print("[bold]AutoForge Daemon stopped[/bold]")

    def _install_signal_handlers(self) -> None:
        if sys.platform == "win32":
            signal.signal(signal.SIGTERM, lambda s, f: self._handle_signal())
            signal.signal(signal.SIGINT, lambda s, f: self._handle_signal())
            return

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal)

    async def _build_loop(self) -> None:
        """Queue polling loop with bounded concurrent project workers."""
        while self._running:
            while (
                self._running
                and len(self._active_builds) < self.config.daemon_max_concurrent_projects
            ):
                project = await self.registry.dequeue()
                if project is None:
                    break
                task = asyncio.create_task(self._process_project(project))
                self._active_builds[project.id] = task

            if not self._active_builds:
                await asyncio.sleep(self.config.daemon_poll_interval)
                continue

            done, _ = await asyncio.wait(
                self._active_builds.values(),
                timeout=self.config.daemon_poll_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                continue

            for project_id, task in list(self._active_builds.items()):
                if task not in done:
                    continue
                del self._active_builds[project_id]
                try:
                    task.result()
                except Exception as exc:
                    logger.error("Worker task crashed for %s: %s", project_id, exc, exc_info=True)

    async def _process_project(self, project: Project) -> None:
        """Run one project through the orchestrator pipeline."""
        console.print(f"[bold blue]Building:[/bold blue] {project.id} - {project.description[:80]}")
        project_config = None

        try:
            project_config = ForgeConfig.from_env(
                budget_limit_usd=project.budget_usd,
            )
            orchestrator = Orchestrator(project_config)
            original_save = orchestrator._save_state

            def _track_phase(phase: str, _orig=original_save, _pid=project.id) -> None:
                _orig(phase)
                self._fire_and_forget(self.registry.update_phase(_pid, phase), "update_phase")

            orchestrator._save_state = _track_phase  # type: ignore[assignment]

            project_dir = await orchestrator.run(project.description)
            cost = project_config.estimated_cost_usd
            project_name = project_dir.name

            await self.registry.update_name(project.id, project_name, str(project_dir))
            await self.registry.mark_completed(project.id, cost)

            guide = generate_deploy_guide(project_dir, project_name)
            guide_path = project_dir / "DEPLOY_GUIDE.md"
            guide_path.write_text(guide, encoding="utf-8")

            console.print(f"[bold green]Completed:[/bold green] {project_name} (${cost:.4f})")
            console.print(f"  Location: {project_dir}")
            console.print(f"  Deploy guide: {guide_path}")

            await self._notify(
                project.requested_by,
                f"Project '{project_name}' completed!\n"
                f"Cost: ${cost:.4f}\n"
                f"Location: {project_dir}\n"
                f"Deploy guide saved to DEPLOY_GUIDE.md",
            )
        except Exception as exc:
            error_msg = str(exc)[:500]
            cost = project_config.estimated_cost_usd if project_config is not None else 0.0
            logger.error("Build failed for %s: %s", project.id, error_msg, exc_info=True)
            await self.registry.mark_failed(project.id, error_msg, cost_usd=cost)
            console.print(f"[bold red]Failed:[/bold red] {project.id} - {error_msg[:80]}")
            await self._notify(project.requested_by, f"Build failed: {error_msg[:200]}")

    async def _notify(self, requested_by: str, message: str) -> None:
        """Dispatch completion/failure notifications by requester channel."""
        channel = requested_by.split(":", 1)[0] if ":" in requested_by else requested_by
        notifier = self._notifiers.get(channel) or self._notifiers.get("*")
        if notifier is None:
            logger.debug("No notifier for channel %s (requested_by=%s)", channel, requested_by)
            return
        try:
            await notifier(requested_by, message)
        except Exception as exc:
            logger.warning("Notification failed for %s: %s", requested_by, exc)

    def _handle_signal(self) -> None:
        """Handle shutdown signal."""
        console.print("\n[yellow]Shutting down...[/yellow]")
        self._running = False

    async def stop(self) -> None:
        """Stop the daemon loop."""
        self._running = False
