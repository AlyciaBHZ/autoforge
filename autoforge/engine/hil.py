"""Human-in-the-loop (HIL) interfaces for AutoForge.

This module defines small, dependency-free protocols that let the core
Orchestrator ask for user input without coupling to any specific channel
implementation (CLI, Telegram, Webhook, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class CheckpointRequest:
    """A phase-boundary checkpoint prompt emitted by the Orchestrator."""

    run_id: str
    phase: str
    summary: str
    project_dir: Path | None


class CheckpointResponder(Protocol):
    """Async checkpoint decision provider.

    Return True to proceed, False to pause/stop.
    """

    async def confirm_checkpoint(self, request: CheckpointRequest) -> bool: ...

