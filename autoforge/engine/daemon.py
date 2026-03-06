"""ForgeDaemon - persistent background service for project queue processing."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

from rich.console import Console

from autoforge.engine.channels.bridge_agent import AsyncBridgeAgent, BridgeRequest, BridgeResponse, BridgeTimeoutEvent
from autoforge.engine.config import ForgeConfig
from autoforge.engine.deploy_guide import generate_deploy_guide
from autoforge.engine.hil import CheckpointRequest, CheckpointResponder
from autoforge.engine.orchestrator import Orchestrator
from autoforge.engine.project_registry import Project, ProjectMessage, ProjectRegistry, RunStage

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
        self._bridge_context: dict[str, dict[str, Any]] = {}
        self._bridge = AsyncBridgeAgent[dict[str, Any], dict[str, Any]](
            timeout_seconds=float(getattr(config, "bridge_timeout_seconds", 60.0) or 60.0),
            on_timeout=self._on_bridge_timeout,
            on_response=self._on_bridge_response,
            on_late_response=self._on_bridge_late_response,
        )

    async def _on_bridge_timeout(self, event: BridgeTimeoutEvent) -> None:
        ctx = self._bridge_context.get(event.request_id)
        if not ctx:
            return
        requested_by = str(ctx.get("requested_by", "") or "")
        kind = str(ctx.get("kind", "") or "")
        if requested_by and kind == "checkpoint":
            phase = str(ctx.get("phase", "") or "")
            project_id = str(ctx.get("project_id", "") or "")
            proj_part = f" {project_id}" if project_id else ""
            await self._notify(
                requested_by,
                f"[checkpoint]{proj_part} timed out waiting for reply (phase={phase}). "
                "Continuing with default policy.",
            )

    async def _on_bridge_response(self, response: BridgeResponse[dict[str, Any]]) -> None:
        # Normal response: clear context to avoid leaks.
        self._bridge_context.pop(response.request_id, None)

    async def _on_bridge_late_response(self, response: BridgeResponse[dict[str, Any]]) -> None:
        ctx = self._bridge_context.pop(response.request_id, None) or {}
        requested_by = str(ctx.get("requested_by", "") or "")
        kind = str(ctx.get("kind", "") or "")
        if requested_by and kind == "checkpoint":
            phase = str(ctx.get("phase", "") or "")
            project_id = str(ctx.get("project_id", "") or "")
            proj_part = f" {project_id}" if project_id else ""
            raw = str(response.payload.get("raw", "") or response.payload.get("text", "") or "")
            await self._notify(
                requested_by,
                f"[checkpoint]{proj_part} late reply received (phase={phase}): {raw[:200]}",
            )

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
        # Use O_EXCL to detect if another daemon is already running
        try:
            fd = os.open(str(pid_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
        except FileExistsError:
            # PID file exists — check if the process is still alive
            try:
                existing_pid = int(pid_file.read_text(encoding="utf-8").strip())
                os.kill(existing_pid, 0)  # Check if process exists
                raise RuntimeError(
                    f"Another daemon is already running (PID {existing_pid}). "
                    f"Stop it first or remove {pid_file}"
                )
            except (OSError, ValueError):
                # Process is dead or PID file is corrupt — safe to overwrite
                logger.warning("Stale PID file found, overwriting")
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
                logger.warning("%s failed: %s", label, exc, exc_info=True)

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
                        bridge_receive=self._bridge_receive,
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
                await self._bridge.close()
            except Exception:
                pass

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

    async def _bridge_receive(self, request_id: str, payload: dict[str, Any]) -> None:
        await self._bridge.receive(str(request_id), dict(payload))

    def _checkpoint_responder_for(self, *, requested_by: str, project_id: str) -> CheckpointResponder | None:
        """Return a checkpoint responder for the requester channel (if supported)."""
        channel = requested_by.split(":", 1)[0] if ":" in requested_by else requested_by
        if channel != "telegram":
            return None

        daemon = self

        class _BridgeCheckpointResponder:
            def __init__(self, requested_by: str, project_id: str) -> None:
                self._requested_by = requested_by
                self._project_id = project_id

            async def confirm_checkpoint(self, request: CheckpointRequest) -> bool:
                req_id = uuid4().hex
                daemon._bridge_context[req_id] = {
                    "kind": "checkpoint",
                    "requested_by": self._requested_by,
                    "phase": request.phase,
                    "run_id": request.run_id,
                    "project_id": self._project_id,
                }
                # Keep context map bounded; dicts preserve insertion order.
                if len(daemon._bridge_context) > 2048:
                    daemon._bridge_context.pop(next(iter(daemon._bridge_context)), None)

                async def _sender(br: BridgeRequest[dict[str, Any]]) -> None:
                    payload = br.payload
                    phase = str(payload.get("phase", "") or "")
                    summary = str(payload.get("summary", "") or "")
                    project_name = ""
                    try:
                        if request.project_dir is not None:
                            project_name = request.project_dir.name
                    except Exception:
                        project_name = ""
                    name_part = f" ({project_name})" if project_name else ""
                    msg = (
                        f"[checkpoint] {self._project_id}{name_part} {phase} complete\n"
                        f"{summary}\n\n"
                        f"Reply within {daemon._bridge.timeout_seconds:.0f}s:\n"
                        f"/reply {br.request_id} yes\n"
                        f"/reply {br.request_id} no\n"
                        "(default: continue)"
                    )
                    await daemon._notify(self._requested_by, msg)

                try:
                    handle = await daemon._bridge.send(
                        {
                            "kind": "checkpoint",
                            "project_id": self._project_id,
                            "phase": request.phase,
                            "summary": request.summary,
                            "run_id": request.run_id,
                            "requested_by": self._requested_by,
                        },
                        sender=_sender,
                        request_id=req_id,
                    )
                except Exception as exc:
                    logger.warning("Failed to send checkpoint prompt: %s", exc, exc_info=True)
                    daemon._bridge_context.pop(req_id, None)
                    return True
                try:
                    resp = await handle.wait()
                except asyncio.TimeoutError:
                    return True
                except Exception as exc:
                    logger.warning("Checkpoint wait failed: %s", exc, exc_info=True)
                    daemon._bridge_context.pop(req_id, None)
                    return True
                proceed = bool(resp.payload.get("proceed", True))
                return proceed

        return _BridgeCheckpointResponder(requested_by, project_id)

    async def _build_loop(self) -> None:
        """Queue polling loop with bounded concurrent project workers."""
        while self._running:
            try:
                recovered = await self.registry.requeue_stale_builds(max_age_seconds=600)
                if recovered:
                    logger.warning("Recovered %d stale BUILDING projects back to queue", recovered)
            except Exception as exc:
                logger.debug("stale build recovery skipped: %s", exc)

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

    async def _progress_loop(
        self,
        *,
        project: Project,
        project_config: ForgeConfig,
        orchestrator: Orchestrator,
        phase_ref: dict[str, str],
        interval_seconds: float = 120.0,
    ) -> None:
        """Periodically push a short progress summary via the requester notifier."""
        channel = project.requested_by.split(":", 1)[0] if ":" in project.requested_by else project.requested_by
        if self._notifiers.get(channel) is None and self._notifiers.get("*") is None:
            return

        interval_raw = float(interval_seconds or 0.0)
        if interval_raw <= 0:
            return
        interval = max(15.0, interval_raw)
        last_fingerprint: tuple[str, int, int] | None = None

        while True:
            await asyncio.sleep(interval)

            phase = str(phase_ref.get("phase", "") or "")
            done = 0
            total = 0
            active_part = ""
            if orchestrator.dag is not None:
                tasks = orchestrator.dag.get_all_tasks()
                total = len(tasks)
                done = sum(1 for t in tasks if t.status.value == "done")
                active = [t for t in tasks if t.status.value == "in_progress"]
                if active:
                    t0 = active[0]
                    desc = str(getattr(t0, "description", "") or "").strip()
                    if len(desc) > 60:
                        desc = desc[:60] + "..."
                    active_part = f" active={t0.id}:{desc}"
                    if len(active) > 1:
                        active_part += f"(+{len(active) - 1})"

            fp = (phase, done, total)
            if fp == last_fingerprint:
                continue
            last_fingerprint = fp

            cost = float(getattr(project_config, "estimated_cost_usd", 0.0) or 0.0)
            task_part = f" tasks={done}/{total}" if total else ""
            msg = f"[progress] {project.id} phase={phase} cost=${cost:.4f}{task_part}{active_part}"
            await self._notify(project.requested_by, msg)

    async def _message_loop(
        self,
        *,
        project: Project,
        project_config: ForgeConfig,
        orchestrator: Orchestrator,
        poll_seconds: float = 2.0,
    ) -> None:
        """Poll project inbox and apply messages as 'user_updates' to the running config.

        Messages are applied asynchronously and will take effect on the next
        agent run / prompt build (safe point).
        """
        interval = max(1.0, float(poll_seconds or 0.0))
        while True:
            try:
                messages = await self.registry.list_unhandled_messages(
                    project.id,
                    after_id=0,
                    limit=50,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                messages = []

            if messages:
                applied: list[str] = []
                for msg in messages:
                    if not isinstance(msg, ProjectMessage):
                        continue
                    try:
                        await self.registry.mark_message_handled(msg.id)
                    except Exception:
                        continue

                    text = str(msg.text or "").strip()
                    if not text:
                        continue
                    updates = getattr(project_config, "user_updates", None)
                    if not isinstance(updates, list):
                        updates = []
                        try:
                            setattr(project_config, "user_updates", updates)
                        except Exception:
                            pass
                    updates.append(text)
                    if len(updates) > 50:
                        del updates[:-50]

                    try:
                        if orchestrator.runtime is not None:
                            orchestrator.runtime.telemetry.record(
                                "user_message",
                                {
                                    "project_id": project.id,
                                    "source": msg.source,
                                    "kind": msg.kind,
                                    "text": text[:500],
                                },
                            )
                    except Exception:
                        pass
                    applied.append(text[:200])

                if applied:
                    body = "\n".join(f"- {t}" for t in applied)
                    await self._notify(
                        project.requested_by,
                        "[message] queued for next safe point:\n" + body,
                    )

            await asyncio.sleep(interval)

    async def _process_project(self, project: Project) -> None:
        """Run one project through the orchestrator pipeline."""
        console.print(f"[bold blue]Building:[/bold blue] {project.id} - {project.description[:80]}")
        project_config = None
        progress_task: asyncio.Task[None] | None = None
        message_task: asyncio.Task[None] | None = None

        lease_holder = f"daemon:{os.getpid()}"
        lease_ok = await self.registry.acquire_task_lease(project.id, lease_holder, ttl_seconds=120)
        if not lease_ok:
            logger.info("Skipping %s; lease already held", project.id)
            return

        heartbeat_task: asyncio.Task[None] | None = None
        try:
            async def _heartbeat() -> None:
                while self._running:
                    await asyncio.sleep(30)
                    await self.registry.heartbeat_task_lease(project.id, lease_holder, ttl_seconds=120)

            heartbeat_task = asyncio.create_task(_heartbeat())
            await self.registry.set_run_stage(f"run_{project.id}", RunStage.RUNNING)
            await self.registry.append_event(f"run_{project.id}", "RunStarted", {"project_id": project.id})

            project_config = ForgeConfig.from_env(
                budget_limit_usd=project.budget_usd,
            )
            checkpoint_responder = self._checkpoint_responder_for(
                requested_by=project.requested_by,
                project_id=project.id,
            )
            orchestrator = Orchestrator(project_config, checkpoint_responder=checkpoint_responder)
            original_save = orchestrator._save_state
            phase_ref: dict[str, str] = {"phase": ""}

            def _track_phase(phase: str, _orig=original_save, _pid=project.id) -> None:
                _orig(phase)
                phase_ref["phase"] = str(phase)
                self._fire_and_forget(self.registry.update_phase(_pid, phase), "update_phase")
                if phase == "spec_complete":
                    # As soon as SPEC completes we know the workspace path; store it for downstream channels.
                    try:
                        if orchestrator.project_dir is not None:
                            project_name = orchestrator.project_dir.name
                            self._fire_and_forget(
                                self.registry.update_name(_pid, project_name, str(orchestrator.project_dir)),
                                "update_name",
                            )
                    except Exception:
                        pass
                    try:
                        project_name = orchestrator.project_dir.name if orchestrator.project_dir is not None else ""
                        workspace = str(orchestrator.project_dir) if orchestrator.project_dir is not None else ""
                        n_modules = len((orchestrator.spec or {}).get("modules", []))
                        cost = float(getattr(project_config, "estimated_cost_usd", 0.0) or 0.0)
                        ws_part = f" workspace={workspace}" if workspace else ""
                        msg = (
                            f"[spec] {project.id} name={project_name} modules={n_modules} "
                            f"cost=${cost:.4f}{ws_part}"
                        )
                        self._fire_and_forget(self._notify(project.requested_by, msg), "notify_spec")
                    except Exception:
                        pass
                elif phase in {"build_complete", "verify_complete", "refactor_complete", "complete", "budget_exceeded"}:
                    try:
                        cost = float(getattr(project_config, "estimated_cost_usd", 0.0) or 0.0)
                        done = 0
                        total = 0
                        if orchestrator.dag is not None:
                            tasks = orchestrator.dag.get_all_tasks()
                            total = len(tasks)
                            done = sum(1 for t in tasks if t.status.value == "done")
                        task_part = f" tasks={done}/{total}" if total else ""
                        msg = f"[phase] {project.id} {phase} cost=${cost:.4f}{task_part}"
                        self._fire_and_forget(self._notify(project.requested_by, msg), "notify_phase")
                    except Exception:
                        pass

            orchestrator._save_state = _track_phase  # type: ignore[assignment]

            progress_task = asyncio.create_task(
                self._progress_loop(
                    project=project,
                    project_config=project_config,
                    orchestrator=orchestrator,
                    phase_ref=phase_ref,
                    interval_seconds=float(getattr(self.config, "progress_notify_interval_seconds", 120.0)),
                )
            )
            message_task = asyncio.create_task(
                self._message_loop(
                    project=project,
                    project_config=project_config,
                    orchestrator=orchestrator,
                )
            )

            workspace_raw = str(getattr(project, "workspace_path", "") or "").strip()
            if workspace_raw:
                workspace = Path(workspace_raw).resolve()
                state_file = workspace / ".forge_state.json"
                if workspace.is_dir() and state_file.exists():
                    self._fire_and_forget(
                        self._notify(project.requested_by, f"[resume] {project.id} resuming from {workspace}"),
                        "notify_resume",
                    )
                    project_dir = await orchestrator.resume(workspace)
                else:
                    project_dir = await orchestrator.run(project.description)
            else:
                project_dir = await orchestrator.run(project.description)
            cost = project_config.estimated_cost_usd
            project_name = project_dir.name

            final_phase = str(phase_ref.get("phase", "") or "")
            if final_phase.startswith("paused_after_"):
                await self.registry.update_name(project.id, project_name, str(project_dir))
                await self.registry.mark_paused(
                    project.id,
                    error=final_phase,
                    cost_usd=cost,
                )
                await self.registry.set_run_stage(f"run_{project.id}", RunStage.PAUSED)
                await self.registry.append_event(
                    f"run_{project.id}",
                    "RunPaused",
                    {"project_id": project.id, "phase": final_phase, "workspace": str(project_dir)},
                )
                console.print(f"[bold yellow]Paused:[/bold yellow] {project_name} (${cost:.4f})")
                await self._notify(
                    project.requested_by,
                    f"[paused] {project.id} paused at {final_phase}.\n"
                    f"Location: {project_dir}\n"
                    f"Resume with: autoforgeai resume {project_dir}",
                )
                return

            await self.registry.update_name(project.id, project_name, str(project_dir))
            await self.registry.mark_completed(project.id, cost)
            await self.registry.set_run_stage(f"run_{project.id}", RunStage.RUNTIME_VERIFIED)
            await self.registry.append_event(
                f"run_{project.id}",
                "RunCompleted",
                {"project_id": project.id, "cost_usd": cost, "workspace": str(project_dir)},
            )

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
            await self.registry.set_run_stage(f"run_{project.id}", RunStage.FAILED)
            await self.registry.append_event(
                f"run_{project.id}",
                "RunFailed",
                {"project_id": project.id, "error": error_msg},
            )
            console.print(f"[bold red]Failed:[/bold red] {project.id} - {error_msg[:80]}")
            await self._notify(project.requested_by, f"Build failed: {error_msg[:200]}")
        finally:
            if progress_task is not None:
                progress_task.cancel()
                await asyncio.gather(progress_task, return_exceptions=True)
            if message_task is not None:
                message_task.cancel()
                await asyncio.gather(message_task, return_exceptions=True)
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            await self.registry.release_task_lease(project.id, lease_holder)

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
