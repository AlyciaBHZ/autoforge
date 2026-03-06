"""Typed error model for the AutoForge engine.

The goal is to make failures:
  - classified (stable error codes)
  - actionable (retryable vs fatal)
  - attributable (phase/task/attempt/run context)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    UNKNOWN = "UNKNOWN"

    # Durable execution / state
    STATE_CORRUPT = "STATE_CORRUPT"
    STATE_INCOMPATIBLE = "STATE_INCOMPATIBLE"
    STATE_IO = "STATE_IO"

    # Sandbox / command execution
    COMMAND_NOT_FOUND = "COMMAND_NOT_FOUND"
    COMMAND_TIMEOUT = "COMMAND_TIMEOUT"
    COMMAND_BLOCKED = "COMMAND_BLOCKED"
    COMMAND_FAILED = "COMMAND_FAILED"

    # Git
    GIT_FAILED = "GIT_FAILED"

    # LLM
    LLM_BUDGET_EXCEEDED = "LLM_BUDGET_EXCEEDED"
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    LLM_SCHEMA_ERROR = "LLM_SCHEMA_ERROR"

    # Dependency resolution / environment preparation
    DEPENDENCY_SETUP_FAILED = "DEPENDENCY_SETUP_FAILED"


@dataclass(frozen=True)
class ErrorContext:
    run_id: str | None = None
    phase: str | None = None
    task_id: str | None = None
    attempt_id: str | None = None
    project_dir: str | None = None


class ForgeError(Exception):
    """Base exception with error code + retry semantics."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        context: ErrorContext | None = None,
        cause: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.context = context or ErrorContext()
        self.cause = cause
        self.details = details or {}

    def __str__(self) -> str:
        base = f"{self.code.value}: {self.message}"
        ctx_parts = []
        if self.context.run_id:
            ctx_parts.append(f"run_id={self.context.run_id}")
        if self.context.phase:
            ctx_parts.append(f"phase={self.context.phase}")
        if self.context.task_id:
            ctx_parts.append(f"task_id={self.context.task_id}")
        if self.context.attempt_id:
            ctx_parts.append(f"attempt_id={self.context.attempt_id}")
        if ctx_parts:
            base += " (" + ", ".join(ctx_parts) + ")"
        return base

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "context": {
                "run_id": self.context.run_id,
                "phase": self.context.phase,
                "task_id": self.context.task_id,
                "attempt_id": self.context.attempt_id,
                "project_dir": self.context.project_dir,
            },
            "details": dict(self.details),
            "cause": str(self.cause) if self.cause else "",
        }

