"""LLM Router — model selection, API calls, and budget enforcement."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from anthropic import AsyncAnthropic

from autoforge.engine.config import ForgeConfig

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity determines which model to use."""

    HIGH = "high"  # Uses model_strong (e.g. Opus)
    STANDARD = "standard"  # Uses model_fast (e.g. Sonnet)


class BudgetExceededError(Exception):
    """Raised when the API budget limit is reached."""


class LLMRouter:
    """Routes LLM calls to appropriate models and tracks usage.

    All agent interactions with the Anthropic API go through this class.
    It handles model selection, budget enforcement, and usage logging.
    """

    def __init__(self, config: ForgeConfig) -> None:
        self.config = config
        if not config.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or run setup.sh"
            )
        self.client = AsyncAnthropic(api_key=config.anthropic_api_key)
        self._call_count = 0

    def _select_model(self, complexity: TaskComplexity) -> tuple[str, int]:
        """Return (model_name, max_tokens) for the given complexity."""
        if complexity == TaskComplexity.HIGH:
            return self.config.model_strong, self.config.max_tokens_strong
        return self.config.model_fast, self.config.max_tokens_fast

    async def call(
        self,
        *,
        complexity: TaskComplexity,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Make an LLM call with model routing and budget checking.

        Args:
            complexity: Determines which model to use.
            system: System prompt for the conversation.
            messages: Message history in Anthropic API format.
            tools: Optional tool definitions in Anthropic API format.

        Returns:
            Anthropic Message response object.

        Raises:
            BudgetExceededError: If the budget limit has been reached.
        """
        if not self.config.check_budget():
            raise BudgetExceededError(
                f"Budget exhausted: ${self.config.estimated_cost_usd:.2f} "
                f"of ${self.config.budget_limit_usd:.2f} used"
            )

        model, max_tokens = self._select_model(complexity)
        self._call_count += 1
        call_id = self._call_count

        logger.info(
            f"[LLM #{call_id}] model={model} messages={len(messages)} "
            f"budget_remaining=${self.config.budget_remaining:.2f}"
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.client.messages.create(**kwargs)

        # Track usage
        usage = response.usage
        self.config.record_usage(model, usage.input_tokens, usage.output_tokens)

        logger.info(
            f"[LLM #{call_id}] stop_reason={response.stop_reason} "
            f"tokens_in={usage.input_tokens} tokens_out={usage.output_tokens} "
            f"cost_total=${self.config.estimated_cost_usd:.4f}"
        )

        return response

    @property
    def total_cost(self) -> float:
        return self.config.estimated_cost_usd

    @property
    def call_count(self) -> int:
        return self._call_count
