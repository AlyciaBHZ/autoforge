"""Telegram Bot channel for AutoForge daemon.

Commands:
  /build <description>  — Queue a new project
  /status               — Show all projects
  /queue                — Show build queue
  /budget               — Show total spending
  /cancel <id>          — Cancel a queued project
  /deploy <id>          — Get deployment guide
  /help                 — Show available commands
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine

from engine.config import ForgeConfig
from engine.project_registry import ProjectRegistry, ProjectStatus

logger = logging.getLogger(__name__)


async def start_telegram_bot(
    config: ForgeConfig,
    registry: ProjectRegistry,
    notify_callback: Callable[..., Coroutine] | None = None,
) -> None:
    """Start the Telegram bot (long-polling mode)."""
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )

    allowed_users = set(config.telegram_allowed_users) if config.telegram_allowed_users else None

    def _check_auth(update: Update) -> bool:
        """Check if the user is authorized."""
        if allowed_users is None:
            return True  # No restriction
        user = update.effective_user
        if user is None:
            return False
        return str(user.id) in allowed_users or (user.username and user.username in allowed_users)

    async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _check_auth(update):
            await update.message.reply_text("Unauthorized. Contact the admin.")
            return
        await update.message.reply_text(
            "Welcome to AutoForge!\n\n"
            "Commands:\n"
            "/build <description> — Queue a new project\n"
            "/status — Show all projects\n"
            "/queue — Show build queue\n"
            "/budget — Show total spending\n"
            "/cancel <id> — Cancel a queued project\n"
            "/deploy <id> — Get deployment guide\n"
            "/help — Show this message"
        )

    async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await cmd_start(update, context)

    async def cmd_build(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _check_auth(update):
            await update.message.reply_text("Unauthorized.")
            return

        text = update.message.text.replace("/build", "", 1).strip()
        if not text:
            await update.message.reply_text("Usage: /build <project description>")
            return

        # Parse optional budget: /build $5 description
        budget = config.budget_limit_usd
        if text.startswith("$"):
            parts = text.split(" ", 1)
            try:
                budget = float(parts[0][1:])
                text = parts[1] if len(parts) > 1 else ""
            except ValueError:
                pass

        if not text:
            await update.message.reply_text("Please provide a project description.")
            return

        user_id = str(update.effective_user.id)
        project = await registry.enqueue(
            description=text,
            requested_by=f"telegram:{user_id}",
            budget_usd=budget,
        )
        queue_size = await registry.queue_size()

        await update.message.reply_text(
            f"Project queued!\n"
            f"ID: {project.id}\n"
            f"Budget: ${budget:.2f}\n"
            f"Queue position: {queue_size}\n\n"
            f"I'll notify you when it's done."
        )

    async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _check_auth(update):
            await update.message.reply_text("Unauthorized.")
            return

        projects = await registry.list_all(limit=10)
        if not projects:
            await update.message.reply_text("No projects yet. Use /build to create one.")
            return

        lines = ["Recent projects:\n"]
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

        await update.message.reply_text("\n".join(lines))

    async def cmd_queue(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _check_auth(update):
            await update.message.reply_text("Unauthorized.")
            return

        queued = await registry.list_by_status(ProjectStatus.QUEUED)
        building = await registry.list_by_status(ProjectStatus.BUILDING)

        lines = []
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

        await update.message.reply_text("\n".join(lines))

    async def cmd_budget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _check_auth(update):
            await update.message.reply_text("Unauthorized.")
            return

        total = await registry.total_cost()
        projects = await registry.list_all(limit=100)
        completed = sum(1 for p in projects if p.status == ProjectStatus.COMPLETED)
        failed = sum(1 for p in projects if p.status == ProjectStatus.FAILED)

        await update.message.reply_text(
            f"Total spending: ${total:.4f}\n"
            f"Completed: {completed}\n"
            f"Failed: {failed}\n"
            f"Default budget per project: ${config.budget_limit_usd:.2f}"
        )

    async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _check_auth(update):
            await update.message.reply_text("Unauthorized.")
            return

        project_id = update.message.text.replace("/cancel", "", 1).strip()
        if not project_id:
            await update.message.reply_text("Usage: /cancel <project_id>")
            return

        ok = await registry.cancel(project_id)
        if ok:
            await update.message.reply_text(f"Project {project_id} cancelled.")
        else:
            await update.message.reply_text(
                f"Cannot cancel {project_id}. Only queued projects can be cancelled."
            )

    async def cmd_deploy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _check_auth(update):
            await update.message.reply_text("Unauthorized.")
            return

        project_id = update.message.text.replace("/deploy", "", 1).strip()
        if not project_id:
            await update.message.reply_text("Usage: /deploy <project_id>")
            return

        try:
            project = await registry.get(project_id)
        except KeyError:
            await update.message.reply_text(f"Project {project_id} not found.")
            return

        if project.status != ProjectStatus.COMPLETED:
            await update.message.reply_text(
                f"Project {project_id} is not completed yet (status: {project.status.value})."
            )
            return

        from pathlib import Path
        guide_path = Path(project.workspace_path) / "DEPLOY_GUIDE.md"
        if guide_path.exists():
            guide = guide_path.read_text(encoding="utf-8")
            # Telegram has 4096 char limit per message
            if len(guide) > 4000:
                guide = guide[:3990] + "\n\n[...truncated]"
            await update.message.reply_text(guide)
        else:
            await update.message.reply_text(
                "Deploy guide not found. The project may have been built before this feature was added."
            )

    # Build the application
    app = Application.builder().token(config.telegram_token).build()
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

    # Keep running until cancelled
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


# Need asyncio for the sleep in start_telegram_bot
import asyncio
