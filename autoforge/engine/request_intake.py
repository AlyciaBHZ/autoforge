"""Centralized intake policy for queueing build requests.

This module keeps request validation, quota checks, and idempotency in one
place so CLI, Telegram, and Webhook channels share the same behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any

from autoforge.engine.config import ForgeConfig
from autoforge.engine.project_registry import Project, ProjectRegistry, ProjectStatus

_SAFE_REQUESTER_RE = re.compile(r"[^a-zA-Z0-9_.-]")


class IntakePolicyError(ValueError):
    """Raised when a request violates queue policy."""


@dataclass
class IntakeResult:
    """Result returned after a queue intake attempt."""

    project: Project
    queue_size: int
    deduplicated: bool = False


class RequestIntakeService:
    """Validates requests and enqueues them with anti-abuse safeguards."""

    def __init__(self, config: ForgeConfig, registry: ProjectRegistry) -> None:
        self.config = config
        self.registry = registry

    @staticmethod
    def _sanitize_requester_id(raw: str, fallback: str = "unknown") -> str:
        """Sanitize a requester hint into a stable, storage-safe identifier."""
        candidate = (raw or "").strip()
        if not candidate:
            return fallback
        candidate = candidate[:128]
        candidate = _SAFE_REQUESTER_RE.sub("_", candidate)
        candidate = candidate.strip("._-")
        return candidate or fallback

    def normalize_requester(
        self,
        *,
        channel: str,
        requester_hint: str | None,
        fallback_hint: str = "unknown",
    ) -> str:
        """Build canonical requester id like ``telegram:123456``."""
        safe_channel = self._sanitize_requester_id(channel, "unknown")
        hint = requester_hint if requester_hint else fallback_hint
        safe_hint = self._sanitize_requester_id(hint, "unknown")
        return f"{safe_channel}:{safe_hint}"

    def validate_description(self, description: str) -> str:
        text = (description or "").strip()
        if not text:
            raise IntakePolicyError("description is required")
        if len(text) > self.config.request_max_description_chars:
            raise IntakePolicyError(
                f"description too long (max {self.config.request_max_description_chars} chars)"
            )
        return text

    def normalize_budget(self, budget: Any) -> float:
        if budget in (None, ""):
            value = float(self.config.budget_limit_usd)
        else:
            try:
                value = float(budget)
            except (TypeError, ValueError) as exc:
                raise IntakePolicyError("budget must be a number") from exc
        if value <= 0:
            raise IntakePolicyError("budget must be positive")
        if value > self.config.request_max_budget_usd:
            raise IntakePolicyError(
                f"budget too high (max ${self.config.request_max_budget_usd:.2f})"
            )
        return value

    @staticmethod
    def _normalize_idempotency_key(idempotency_key: str | None) -> str | None:
        if not idempotency_key:
            return None
        key = idempotency_key.strip()
        if not key:
            return None
        if len(key) > 128:
            raise IntakePolicyError("idempotency key too long (max 128 chars)")
        return key

    async def _enforce_limits(self, requester: str) -> None:
        queue_size = await self.registry.queue_size()
        if queue_size >= self.config.queue_max_size:
            raise IntakePolicyError("queue is full, try again later")

        active_for_requester = await self.registry.count_by_status_for_requester(
            requester,
            statuses=[ProjectStatus.QUEUED, ProjectStatus.BUILDING],
        )
        if active_for_requester >= self.config.requester_queue_limit:
            raise IntakePolicyError(
                f"active request limit reached ({self.config.requester_queue_limit})"
            )

        now = datetime.now(timezone.utc)

        window_start = (now - timedelta(seconds=self.config.requester_rate_window_seconds)).isoformat()
        burst_count = await self.registry.count_created_since(requester, window_start)
        if burst_count >= self.config.requester_rate_limit:
            raise IntakePolicyError("rate limit exceeded, please wait before retrying")

        day_start = (now - timedelta(days=1)).isoformat()
        daily_count = await self.registry.count_created_since(requester, day_start)
        if daily_count >= self.config.requester_daily_limit:
            raise IntakePolicyError("daily request quota exceeded")

    async def enqueue(
        self,
        *,
        channel: str,
        requester_hint: str | None,
        fallback_hint: str = "unknown",
        description: str,
        budget: Any = None,
        idempotency_key: str | None = None,
    ) -> IntakeResult:
        """Validate and enqueue a request, honoring idempotency and quotas."""
        requested_by = self.normalize_requester(
            channel=channel,
            requester_hint=requester_hint,
            fallback_hint=fallback_hint,
        )
        description_text = self.validate_description(description)
        budget_value = self.normalize_budget(budget)
        idem = self._normalize_idempotency_key(idempotency_key)

        if idem:
            existing = await self.registry.get_by_idempotency(requested_by, idem)
            if existing is not None and existing.status in {
                ProjectStatus.QUEUED,
                ProjectStatus.BUILDING,
                ProjectStatus.COMPLETED,
            }:
                return IntakeResult(
                    project=existing,
                    queue_size=await self.registry.queue_size(),
                    deduplicated=True,
                )

        await self._enforce_limits(requested_by)
        project = await self.registry.enqueue_run(
            description=description_text,
            requested_by=requested_by,
            budget_usd=budget_value,
            idempotency_key=idem,
        )
        return IntakeResult(
            project=project,
            queue_size=await self.registry.queue_size(),
            deduplicated=False,
        )
