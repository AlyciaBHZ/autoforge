"""Telegram bot channel for daemon request intake and notifications."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine

from autoforge.engine.config import ForgeConfig
from autoforge.engine.project_registry import ProjectRegistry, ProjectStatus
from autoforge.engine.request_intake import IntakePolicyError, RequestIntakeService

logger = logging.getLogger(__name__)

NotifierRegistrar = Callable[
    [str, Callable[[str, str], Coroutine[Any, Any, None]]],
    None,
]


async def start_telegram_bot(
    config: ForgeConfig,
    registry: ProjectRegistry,
    register_notifier: NotifierRegistrar | None = None,
) -> None:
    """Start Telegram bot in long-polling mode."""
    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, ContextTypes
    except ImportError:
        raise ImportError(
            "Telegram bot requires 'python-telegram-bot'. "
            "Install it with: pip install forgeai[channels]"
        ) from None

    intake = RequestIntakeService(config, registry)
    allowed_users = set(config.telegram_allowed_users) if config.telegram_allowed_users else None
    requester_to_chat: dict[str, int] = {}

    def _remember_chat(update: Update) -> None:
        requester_id = _requester_from_update(update)
        chat = update.effective_chat
        if chat is None:
            return
        requester_to_chat[requester_id] = chat.id

    def _requester_from_update(update: Update) -> str:
        user = update.effective_user
        if user is None:
            return "telegram:unknown"
        return f"telegram:{user.id}"

    def _is_authorized(update: Update) -> bool:
        if allowed_users is None:
            return bool(config.telegram_allow_public)
        user = update.effective_user
        if user is None:
            return False
        username = user.username or ""
        return str(user.id) in allowed_users or username in allowed_users

    async def _send_reply(update: Update, text: str) -> None:
        if update.message is not None:
            await update.message.reply_text(text)

    async def _send_long(chat_id: int, text: str) -> None:
        max_len = 3900
        parts = [text[i : i + max_len] for i in range(0, len(text), max_len)] or [text]
        for part in parts:
            await app.bot.send_message(chat_id=chat_id, text=part)

    async def _notify_requester(requested_by: str, message: str) -> None:
        chat_id = requester_to_chat.get(requested_by)
        if chat_id is None and requested_by.startswith("telegram:"):
            raw = requested_by.split(":", 1)[1]
            if raw.isdigit():
                chat_id = int(raw)
        if chat_id is None:
            logger.debug("No Telegram chat mapping for requester %s", requested_by)
            return
        await _send_long(chat_id, message)

    async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _is_authorized(update):
            if allowed_users is None and not config.telegram_allow_public:
                await _send_reply(
                    update,
                    "Unauthorized. Admin must set FORGE_TELEGRAM_ALLOWED_USERS "
                    "or FORGE_TELEGRAM_ALLOW_PUBLIC=true.",
                )
            else:
                await _send_reply(update, "Unauthorized. Contact the admin.")
            return

        _remember_chat(update)
        await _send_reply(
            update,
            "Welcome to AutoForge.\n\n"
            "Commands:\n"
            "/build <description> - Queue a new project\n"
            "/status - Show your recent projects\n"
            "/queue - Show your queued/building requests\n"
            "/budget - Show your total spending\n"
            "/cancel <id> - Cancel one queued project\n"
            "/deploy <id> - Get deploy guide\n"
            "/help - Show this message",
        )

    async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await cmd_start(update, context)

    async def cmd_build(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _is_authorized(update):
            await _send_reply(update, "Unauthorized.")
            return

        _remember_chat(update)
        message_text = update.message.text if update.message else ""
        text = message_text.replace("/build", "", 1).strip()
        if not text:
            await _send_reply(update, "Usage: /build <project description>")
            return

        budget_override = None
        if text.startswith("$"):
            parts = text.split(" ", 1)
            try:
                budget_override = float(parts[0][1:])
            except ValueError:
                await _send_reply(update, "Invalid budget format. Use: /build $5 description")
                return
            text = parts[1] if len(parts) > 1 else ""

        if not text:
            await _send_reply(update, "Please provide a project description.")
            return

        user_id = str(update.effective_user.id) if update.effective_user else "unknown"
        try:
            result = await intake.enqueue(
                channel="telegram",
                requester_hint=user_id,
                fallback_hint="unknown",
                description=text,
                budget=budget_override,
            )
        except IntakePolicyError as exc:
            await _send_reply(update, f"Request rejected: {exc}")
            return

        if result.deduplicated:
            await _send_reply(
                update,
                f"Duplicate request detected, reusing existing project.\n"
                f"ID: {result.project.id}\n"
                f"Status: {result.project.status.value}",
            )
            return

        await _send_reply(
            update,
            f"Project queued.\n"
            f"ID: {result.project.id}\n"
            f"Budget: ${result.project.budget_usd:.2f}\n"
            f"Queue position: {result.queue_size}\n\n"
            f"You will get a completion notification here.",
        )

    async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _is_authorized(update):
            await _send_reply(update, "Unauthorized.")
            return

        _remember_chat(update)
        requester = _requester_from_update(update)
        projects = await registry.list_for_requester(requester, limit=10)
        if not projects:
            await _send_reply(update, "No projects yet. Use /build to create one.")
            return

        lines = ["Your recent projects:\n"]
        for p in projects:
            status_emoji = {
                ProjectStatus.QUEUED: "⏳",
                ProjectStatus.BUILDING: "🔨",
                ProjectStatus.COMPLETED: "✅",
                ProjectStatus.FAILED: "❌",
                ProjectStatus.CANCELLED: "🚫",
            }.get(p.status, "❓")
            desc = p.description[:40] + ("..." if len(p.description) > 40 else "")
            line = f"{status_emoji} [{p.id}] {desc}"
            if p.status == ProjectStatus.COMPLETED:
                line += f" (${p.cost_usd:.2f})"
            elif p.status == ProjectStatus.BUILDING:
                line += f" ({p.phase})"
            lines.append(line)
        await _send_reply(update, "\n".join(lines))

    async def cmd_queue(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _is_authorized(update):
            await _send_reply(update, "Unauthorized.")
            return

        _remember_chat(update)
        requester = _requester_from_update(update)
        queued = await registry.list_by_status_for_requester(ProjectStatus.QUEUED, requester)
        building = await registry.list_by_status_for_requester(ProjectStatus.BUILDING, requester)

        lines: list[str] = []
        if building:
            lines.append("Building now:")
            for p in building:
                lines.append(f"  🔨 [{p.id}] {p.description[:50]} ({p.phase})")
        if queued:
            lines.append(f"\nQueued ({len(queued)}):")
            for i, p in enumerate(queued, 1):
                lines.append(f"  {i}. [{p.id}] {p.description[:50]}")
        elif not building:
            lines.append("Queue is empty. Use /build to add a project.")

        await _send_reply(update, "\n".join(lines))

    async def cmd_budget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _is_authorized(update):
            await _send_reply(update, "Unauthorized.")
            return

        _remember_chat(update)
        requester = _requester_from_update(update)
        total = await registry.total_cost_for_requester(requester)
        projects = await registry.list_for_requester(requester, limit=200)
        completed = sum(1 for p in projects if p.status == ProjectStatus.COMPLETED)
        failed = sum(1 for p in projects if p.status == ProjectStatus.FAILED)

        await _send_reply(
            update,
            f"Your total spending: ${total:.4f}\n"
            f"Completed: {completed}\n"
            f"Failed: {failed}\n"
            f"Default budget per project: ${config.budget_limit_usd:.2f}",
        )

    async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _is_authorized(update):
            await _send_reply(update, "Unauthorized.")
            return

        _remember_chat(update)
        message_text = update.message.text if update.message else ""
        project_id = message_text.replace("/cancel", "", 1).strip()
        if not project_id:
            await _send_reply(update, "Usage: /cancel <project_id>")
            return

        requester = _requester_from_update(update)
        ok = await registry.cancel_for_requester(project_id, requester)
        if ok:
            await _send_reply(update, f"Project {project_id} cancelled.")
        else:
            await _send_reply(
                update,
                f"Cannot cancel {project_id}. It may not be queued or not owned by you.",
            )

    async def cmd_deploy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _is_authorized(update):
            await _send_reply(update, "Unauthorized.")
            return

        _remember_chat(update)
        message_text = update.message.text if update.message else ""
        project_id = message_text.replace("/deploy", "", 1).strip()
        if not project_id:
            await _send_reply(update, "Usage: /deploy <project_id>")
            return

        requester = _requester_from_update(update)
        try:
            project = await registry.get_for_requester(project_id, requester)
        except KeyError:
            await _send_reply(update, f"Project {project_id} not found.")
            return

        if project.status != ProjectStatus.COMPLETED:
            await _send_reply(
                update,
                f"Project {project_id} is not completed yet (status: {project.status.value}).",
            )
            return

        workspace = Path(project.workspace_path).resolve()
        guide_path = (workspace / "DEPLOY_GUIDE.md").resolve()
        if not str(guide_path).startswith(str(workspace)):
            await _send_reply(update, "Invalid workspace path.")
            return
        if not guide_path.exists():
            await _send_reply(
                update,
                "Deploy guide not found. The project may have been built before this feature.",
            )
            return

        try:
            guide = guide_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            await _send_reply(update, f"Error reading deploy guide: {exc}")
            return
        if update.effective_chat is None:
            await _send_reply(update, guide[:3900])
            return
        await _send_long(update.effective_chat.id, guide)

    app = Application.builder().token(config.telegram_token).build()

    if register_notifier is not None:
        register_notifier("telegram", _notify_requester)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("build", cmd_build))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("queue", cmd_queue))
    app.add_handler(CommandHandler("budget", cmd_budget))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("deploy", cmd_deploy))

    logger.info("Telegram bot starting...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
