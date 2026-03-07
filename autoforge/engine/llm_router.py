"""LLM Router - multi-provider model selection, API calls, and budget enforcement.

Supports Anthropic (Claude), OpenAI (GPT/o-series), and Google (Gemini).
All responses are normalized to a common format so agents don't need
provider-specific code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from autoforge.engine.config import DEFAULT_PRICING, MODEL_PRICING, ForgeConfig
from autoforge.engine.development_harness import (
    append_development_jsonl,
    resolve_development_harness_root,
    serialize_jsonable,
    write_development_json,
)
from autoforge.engine.runtime.rate_limit import RateLimitSpec, SqliteRateLimiter
from autoforge.engine.runtime.telemetry import TelemetrySink

logger = logging.getLogger(__name__)


# ── LLM Provider Protocol ──────────────────────────────────────────
# Implement this protocol to add a new LLM provider (e.g., Mistral, Cohere).
# Then register it via LLMRouter.register_provider().


class LLMProvider:
    """Protocol for LLM provider implementations.

    To add a new provider:
        1. Subclass LLMProvider
        2. Implement get_client(), call(), convert_messages(), convert_tools()
        3. Register via LLMRouter.register_provider("my_provider", MyProvider())
        4. Add model patterns to detect_provider() or use explicit provider selection
    """

    def get_client(self, config: ForgeConfig, auth: Any) -> Any:
        """Create or return a cached async client."""
        raise NotImplementedError

    async def call(
        self,
        client: Any,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        response_json_schema: dict[str, Any] | None = None,
    ) -> "LLMResponse":
        """Make an API call and return a normalized LLMResponse."""
        raise NotImplementedError


# ── Normalized response types ──────────────────────────────────────


@dataclass
class ContentBlock:
    """A block of content in an LLM response (text, tool_use, or image)."""

    type: str  # "text", "tool_use", or "image"
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    # Vision fields - used when type == "image"
    media_type: str = ""  # e.g. "image/png", "image/jpeg"
    image_data: str = ""  # base64-encoded image bytes


@dataclass
class Usage:
    """Token usage for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    """Normalized LLM response - same attribute interface as Anthropic's Message."""

    content: list[ContentBlock] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" or "tool_use"
    usage: Usage = field(default_factory=Usage)


# ── Enums ──────────────────────────────────────────────────────────


class TaskComplexity(Enum):
    """Task complexity determines which model to use."""

    HIGH = "high"  # Uses model_strong (e.g. Opus, GPT-4o, Gemini Pro)
    STANDARD = "standard"  # Uses model_fast (e.g. Sonnet, GPT-4o-mini, Flash)
    COMPLEX = "complex"  # Alias of HIGH for backward compatibility
    MODERATE = "moderate"  # Alias of STANDARD for backward compatibility
    LOW = "low"  # Alias of STANDARD for lightweight tasks


class BudgetExceededError(Exception):
    """Raised when the API budget limit is reached."""


# ── Retry configuration ───────────────────────────────────────────

MAX_RETRIES = 3
"""Maximum number of retry attempts for transient API errors."""

_RETRY_BASE_DELAY = 1.0  # seconds - base delay for exponential backoff


# ── Provider detection ─────────────────────────────────────────────

# Known OpenAI model prefixes/names
_OPENAI_MODELS = {
    "gpt-5.4",
    "gpt-5.3",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2-codex",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5-codex",
    "codex-5.3",
    "o3",
    "o3-mini",
    "o4",
    "o4-mini",
    "o1",
    "o1-mini",
    "o1-preview",
}

_OPENAI_MODEL_ALIASES = {
    # Historical alias from older AutoForge releases.
    "codex-5.3": "gpt-5.3-codex",
}

_OPENAI_MODEL_FALLBACKS = [
    "gpt-5.1-codex-max",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.1-codex-mini",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
]


def detect_provider(model: str) -> str:
    """Detect LLM provider from model name.

    Returns "anthropic", "openai", or "google".
    """
    model_lower = model.lower()

    # OpenAI: check exact matches and prefixes
    if model_lower in _OPENAI_MODELS or model_lower.startswith(("gpt-", "codex-", "o1", "o3", "o4", "o5")):
        return "openai"

    # Google Gemini
    if model_lower.startswith("gemini"):
        return "google"

    # Default: Anthropic (claude-* and anything else)
    if not model_lower.startswith("claude"):
        logger.warning(
            "Unknown model %r does not match any known provider pattern. "
            "Defaulting to Anthropic - verify this is correct.",
            model,
        )
    return "anthropic"


# ── LLM Router ─────────────────────────────────────────────────────


class LLMRouter:
    """Routes LLM calls to appropriate providers and tracks usage.

    All agent interactions with LLM APIs go through this class.
    It handles provider detection, model selection, format conversion,
    budget enforcement, and usage logging.
    """

    # Custom provider registry - populated via register_provider()
    # NOTE: class-level registry is intentionally shared so providers
    # registered once are available to all router instances.
    _custom_providers: dict[str, LLMProvider] = {}
    _custom_providers_initialized: bool = False

    @classmethod
    def register_provider(cls, name: str, provider: LLMProvider) -> None:
        """Register a custom LLM provider.

        Once registered, models that detect_provider() maps to `name`
        will be routed through this provider's call() method.

        Args:
            name: Provider identifier (e.g., "mistral", "cohere").
            provider: An LLMProvider implementation.
        """
        # Copy-on-write to avoid mutating a shared default dict
        if not cls._custom_providers_initialized:
            cls._custom_providers = {}
            cls._custom_providers_initialized = True
        cls._custom_providers[name] = provider
        logger.info("Registered custom LLM provider: %s", name)

    def __init__(self, config: ForgeConfig) -> None:
        self.config = config
        self._call_count = 0
        self._clients: dict[str, Any] = {}
        self._auth_providers: dict[str, Any] = {}  # Cached AuthProvider per provider
        # Per-run cache: if a configured OpenAI model is unavailable (404),
        # remember the first working fallback to avoid repeated retries.
        self._openai_model_overrides: dict[str, str] = {}
        self._budget_lock: asyncio.Lock | None = None
        self._client_lock: asyncio.Lock | None = None
        self._reserved_budget_usd: float = 0.0
        self._telemetry: TelemetrySink | None = None
        self._rng = random.Random()
        self._harness_project_dir: Path | None = None
        self._last_openai_candidate: str = ""
        if bool(getattr(config, "deterministic", False)):
            try:
                self._rng.seed(int(getattr(config, "deterministic_seed", 0) or 0))
            except (ValueError, TypeError):
                self._rng.seed(0)
        self._rate_limiter: SqliteRateLimiter | None = None

    def set_telemetry_sink(self, sink: TelemetrySink | None) -> None:
        """Attach a best-effort telemetry sink (durable journal, trace exporter, etc)."""
        self._telemetry = sink

    def set_harness_project_dir(self, project_dir: Path | None) -> None:
        self._harness_project_dir = project_dir

    def _llm_harness_root(self) -> Path:
        root = resolve_development_harness_root(
            config=self.config,
            project_dir=self._harness_project_dir,
        )
        return root / "llm_harness"

    def _record_call_bundle(
        self,
        *,
        call_id: int,
        provider: str,
        requested_model: str,
        actual_model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        response: LLMResponse | None,
        error_text: str,
        duration_seconds: float,
        max_tokens: int,
        response_json_schema: dict[str, Any] | None,
    ) -> None:
        calls_dir = self._llm_harness_root() / "calls"
        bundle_path = calls_dir / f"call_{call_id:05d}.json"
        payload: dict[str, Any] = {
            "run_id": str(getattr(self.config, "run_id", "") or ""),
            "lineage_id": str(getattr(self.config, "lineage_id", "") or ""),
            "project_id": str(getattr(self.config, "project_id", "") or ""),
            "call_id": int(call_id),
            "provider": str(provider or ""),
            "requested_model": str(requested_model or ""),
            "actual_model": str(actual_model or requested_model or ""),
            "system": system,
            "messages": serialize_jsonable(messages),
            "tools": serialize_jsonable(tools or []),
            "max_tokens": int(max_tokens),
            "schema_enforced": response_json_schema is not None,
            "response_json_schema": serialize_jsonable(response_json_schema or {}),
            "ok": response is not None,
            "error": str(error_text or ""),
            "duration_seconds": float(duration_seconds),
        }
        if response is not None:
            payload.update(
                {
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": int(response.usage.input_tokens),
                        "output_tokens": int(response.usage.output_tokens),
                    },
                    "response_content": serialize_jsonable(response.content),
                }
            )
        write_development_json(bundle_path, payload, artifact_type="llm_call_bundle")

    def _actual_call_cost_usd(self, *, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        return (
            int(input_tokens) * float(pricing["input"]) / 1_000_000
            + int(output_tokens) * float(pricing["output"]) / 1_000_000
        )

    def _record_budget_ledger(
        self,
        *,
        call_id: int,
        provider: str,
        actual_model: str,
        response: LLMResponse | None,
        budget_before: float,
    ) -> None:
        if response is None:
            return
        actual_cost = self._actual_call_cost_usd(
            model=actual_model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        ledger_path = self._llm_harness_root() / "budget_ledger.jsonl"
        entry = {
            "run_id": str(getattr(self.config, "run_id", "") or ""),
            "call_id": int(call_id),
            "provider": str(provider or ""),
            "model": str(actual_model or ""),
            "input_tokens": int(response.usage.input_tokens),
            "output_tokens": int(response.usage.output_tokens),
            "actual_cost_usd": float(actual_cost),
            "budget_before_usd": float(budget_before),
            "budget_after_usd": float(getattr(self.config, "estimated_cost_usd", 0.0)),
            "budget_limit_usd": float(getattr(self.config, "budget_limit_usd", 0.0)),
        }
        append_development_jsonl(ledger_path, entry, event_type="budget_entry")
        write_development_json(
            self._llm_harness_root() / "budget_summary.json",
            {
                "run_id": str(getattr(self.config, "run_id", "") or ""),
                "call_count": int(self._call_count),
                "budget_limit_usd": float(getattr(self.config, "budget_limit_usd", 0.0)),
                "budget_used_usd": float(getattr(self.config, "estimated_cost_usd", 0.0)),
                "last_call_cost_usd": float(actual_cost),
                "last_call_id": int(call_id),
            },
            artifact_type="llm_budget_summary",
        )

    def _record_fallback_receipt(
        self,
        *,
        requested_model: str,
        failed_candidate: str,
        fallback_candidate: str,
        error: str,
    ) -> None:
        append_development_jsonl(
            self._llm_harness_root() / "fallback_receipts.jsonl",
            {
                "run_id": str(getattr(self.config, "run_id", "") or ""),
                "lineage_id": str(getattr(self.config, "lineage_id", "") or ""),
                "requested_model": str(requested_model or ""),
                "failed_candidate": str(failed_candidate or ""),
                "fallback_candidate": str(fallback_candidate or ""),
                "error": str(error or ""),
            },
            event_type="model_fallback",
        )

    async def close(self) -> None:
        """Close all cached async HTTP clients to release connection pools."""
        for provider, client in self._clients.items():
            try:
                if hasattr(client, "close"):
                    await client.close()
                elif hasattr(client, "aclose"):
                    await client.aclose()
            except Exception:
                logger.debug("Failed to close %s client", provider)
        self._clients.clear()

    async def __aenter__(self) -> LLMRouter:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _get_budget_lock(self) -> asyncio.Lock:
        """Lazily create lock to avoid cross-event-loop lock reuse."""
        if self._budget_lock is None:
            self._budget_lock = asyncio.Lock()
        return self._budget_lock

    def _is_openai_subscription_auth(self) -> bool:
        auth = self.config.auth_config.get("openai", {})
        method = str(auth.get("auth_method", "")).strip().lower()
        return method in ("codex_oauth", "device_code")

    def _is_budget_enforced(self, provider: str) -> bool:
        if provider == "openai" and self._is_openai_subscription_auth():
            return False
        return True

    def _is_openai_reasoning_model(self, model: str) -> bool:
        model_lower = model.lower()
        if model_lower.startswith(("o1", "o3", "o4", "o5", "codex-", "gpt-5")):
            return True
        if "codex" in model_lower:
            return True
        return False

    def _is_openai_codex_unsupported_model_error(self, exc: Exception) -> bool:
        """Detect the Codex backend error for an unsupported model.

        When using ChatGPT-subscription Codex auth, OpenAI may return HTTP 400/403
        with a message like "model is not supported when using Codex with a ChatGPT account".
        """
        status_code = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        detail = ""
        if isinstance(body, dict) and "detail" in body:
            detail = str(body.get("detail", "")).lower()
        text = str(exc).lower()
        combined = " ".join(x for x in (text, detail) if x)
        if "not supported" not in combined:
            return False
        if "codex" not in combined:
            return False
        if "chatgpt" not in combined:
            return False
        if status_code in (400, 403):
            return True
        # Some client wrappers don't expose status_code/body reliably.
        if "not supported when using codex" in combined:
            return True
        return False

    def _canonicalize_openai_model(self, model: str) -> str:
        """Normalize legacy OpenAI aliases into canonical API model names."""
        return _OPENAI_MODEL_ALIASES.get(model.lower(), model)

    def _openai_model_candidates(self, model: str) -> list[str]:
        """Build fallback model candidates for OpenAI model access failures."""
        canonical = self._canonicalize_openai_model(model)
        if not canonical:
            return []
        candidates: list[str] = [canonical]

        # Respect user-selected models in subscription auth mode: no silent model rewrite,
        # and no automatic fallback to other models.
        if self._is_openai_subscription_auth():
            return candidates

        for fallback in _OPENAI_MODEL_FALLBACKS:
            if fallback not in candidates:
                candidates.append(fallback)
        return candidates

    def _is_openai_model_access_error(self, exc: Exception) -> bool:
        """Return True when OpenAI says the model is unavailable to this account."""
        status_code = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        code = ""
        detail = ""
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                code = str(err.get("code", "")).lower()
            if "detail" in body:
                detail = str(body.get("detail", "")).lower()

        text = str(exc).lower()
        combined = " ".join(x for x in (text, detail, code) if x)
        if code == "model_not_found":
            return True
        if status_code == 404 and ("model" in text and ("not found" in text or "does not exist" in text)):
            return True
        if "do not have access" in text or "does not exist or you do not have access" in text:
            return True
        if status_code in (400, 403) and "codex" in combined and "not supported" in combined:
            # ChatGPT-subscription Codex backend rejects unsupported models with 400s.
            return True
        if "not supported when using codex" in combined and "chatgpt" in combined:
            return True
        return False

    def _estimate_call_cost_usd(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> float:
        """Estimate upper-bound call cost for reservation.

        Uses a conservative chars->tokens approximation to reduce overspend races.
        """
        total_chars = len(system)
        total_chars += sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
        estimated_input_tokens = max(1, total_chars // 4)
        estimated_output_tokens = max(1, max_tokens)
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        return (
            estimated_input_tokens * pricing["input"] / 1_000_000
            + estimated_output_tokens * pricing["output"] / 1_000_000
        )

    def _estimate_call_tokens(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> int:
        """Estimate total tokens for rate limiting (input approx + max_tokens)."""
        total_chars = len(system)
        total_chars += sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
        estimated_input_tokens = max(1, total_chars // 4)
        return int(estimated_input_tokens) + max(1, int(max_tokens or 1))

    def _get_rate_limiter(self) -> SqliteRateLimiter | None:
        enabled = bool(getattr(self.config, "llm_rate_limit_enabled", False))
        rpm = int(getattr(self.config, "llm_rpm_limit", 0) or 0)
        tpm = int(getattr(self.config, "llm_tpm_limit", 0) or 0)
        if not enabled and rpm <= 0 and tpm <= 0:
            return None

        db_path = getattr(self.config, "llm_rate_limit_db_path", None)
        if db_path is None:
            return None
        ns = str(getattr(self.config, "llm_rate_limit_namespace", "global") or "global").strip() or "global"

        if self._rate_limiter is None:
            spec = RateLimitSpec(
                enabled=True,
                namespace=ns,
                requests_per_minute=max(0, rpm),
                tokens_per_minute=max(0, tpm),
                db_path=db_path,
            )
            self._rate_limiter = SqliteRateLimiter(spec=spec)
        return self._rate_limiter

    def _get_client_lock(self) -> asyncio.Lock:
        """Lazily create client lock to avoid cross-event-loop lock reuse."""
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        return self._client_lock

    def _get_client(self, provider: str) -> Any:
        """Get or create an async client for the given provider.

        Uses the auth module to determine authentication method (API key,
        OAuth bearer, OAuth2 client_credentials, Google ADC/service account,
        AWS Bedrock, Google Vertex AI, Codex OAuth, Device Code).

        Note: Callers should use _get_or_create_client() for async-safe access.
        This sync method is kept for backward compatibility but may race under
        concurrent calls.
        """
        if provider in self._clients:
            return self._clients[provider]

        from autoforge.engine.auth import (
            AWSBedrockAuth, CodexOAuthAuth, DeviceCodeAuth, VertexAIAuth,
            create_auth_provider,
        )

        auth_config = getattr(self.config, "auth_config", {})
        auth = create_auth_provider(provider, self.config.api_keys, auth_config)
        self._auth_providers[provider] = auth
        client_kwargs = auth.get_client_kwargs()

        if provider == "anthropic":
            # Check for Bedrock or Vertex AI auth
            if isinstance(auth, AWSBedrockAuth):
                from anthropic import AsyncAnthropicBedrock
                client = AsyncAnthropicBedrock(**client_kwargs)
            elif isinstance(auth, VertexAIAuth):
                from anthropic import AsyncAnthropicVertex
                client = AsyncAnthropicVertex(**client_kwargs)
            else:
                from anthropic import AsyncAnthropic
                client = AsyncAnthropic(**client_kwargs)

        elif provider == "openai":
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI provider requires the 'openai' package. "
                    "Install it with: pip install openai"
                ) from None
            if isinstance(auth, (CodexOAuthAuth, DeviceCodeAuth)):
                # Token-based auth: client gets a dummy key, real token
                # injected via _ensure_fresh_token before each call
                client = AsyncOpenAI(api_key="placeholder", **client_kwargs)
            else:
                client = AsyncOpenAI(**client_kwargs)

        elif provider == "google":
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "Google provider requires the 'google-genai' package. "
                    "Install it with: pip install autoforgeai[google]"
                ) from None
            auth_method = auth_config.get("google", {}).get("auth_method", "api_key")
            if auth_method in ("adc", "service_account"):
                # ADC: genai.Client() auto-discovers credentials
                client = genai.Client()
            else:
                key = self.config.api_keys.get("google", "")
                client = genai.Client(api_key=key)

        else:
            raise ValueError(f"Unknown provider: {provider}")

        self._clients[provider] = client
        return client

    async def _ensure_fresh_token(self, provider: str) -> None:
        """Refresh OAuth/dynamic token if needed and update the client."""
        from autoforge.engine.auth import (
            CodexOAuthAuth, DeviceCodeAuth, OAuth2ClientCredentialsAuth,
        )

        auth = self._auth_providers.get(provider)
        if auth is None:
            return

        # Only refresh for token-based auth providers
        if not isinstance(auth, (OAuth2ClientCredentialsAuth, CodexOAuthAuth, DeviceCodeAuth)):
            return

        token = await auth.get_token()
        # Update the client's API key with the fresh token
        if provider in self._clients:
            client = self._clients[provider]
            if hasattr(client, "api_key"):
                client.api_key = token.access_token

    async def _retry_with_backoff(
        self,
        fn: Callable[[], Coroutine[Any, Any, Any]],
        *,
        max_retries: int = MAX_RETRIES,
    ) -> Any:
        """Retry an async callable with exponential backoff for transient errors.

        Retries on rate-limit (HTTP 429), server errors (5xx), and network
        errors (ConnectionError, TimeoutError, OSError).  Non-transient errors
        (400, 401, 403, etc.) are re-raised immediately.

        Backoff schedule: 1s, 2s, 4s (with +-25 % jitter).
        """
        last_exc: BaseException | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return await fn()
            except Exception as exc:
                last_exc = exc

                # --- Classify the error ---
                status: int | None = None
                if hasattr(exc, "status_code"):          # Anthropic / httpx
                    status = exc.status_code
                elif hasattr(exc, "status"):              # OpenAI / aiohttp
                    status = exc.status
                elif hasattr(exc, "code"):                # Google / gRPC
                    code = getattr(exc, "code", None)
                    if isinstance(code, int):
                        status = code

                is_transient = False

                # Rate-limit or server error
                if status is not None and (status == 429 or 500 <= status < 600):
                    is_transient = True

                # Network-level failures
                if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
                    is_transient = True

                if not is_transient or attempt == max_retries:
                    raise

                # Exponential backoff with jitter (+-25 %)
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                jitter = delay * 0.25 * (2 * self._rng.random() - 1)
                delay = max(0, delay + jitter)

                logger.warning(
                    "Transient error on attempt %d/%d (status=%s): %s - "
                    "retrying in %.1fs",
                    attempt, max_retries, status, exc, delay,
                )
                await asyncio.sleep(delay)

        # Should never reach here, but satisfy type checker
        assert last_exc is not None
        raise last_exc

    def _select_model(self, complexity: TaskComplexity) -> tuple[str, int]:
        """Return (model_name, max_tokens) for the given complexity."""
        if complexity in (TaskComplexity.HIGH, TaskComplexity.COMPLEX):
            return self.config.model_strong, self.config.max_tokens_strong
        return self.config.model_fast, self.config.max_tokens_fast

    async def generate(
        self,
        prompt: str,
        *,
        complexity: TaskComplexity = TaskComplexity.HIGH,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Convenience alias: ``await llm.generate(prompt, ...)``

        Maps to :meth:`call` with the prompt wrapped as a single user
        message.  Extra kwargs (``max_tokens``, ``temperature``, etc.) are
        accepted for call-site compatibility but do not affect routing -
        model selection is governed solely by *complexity*.
        """
        return await self.call(
            prompt,
            complexity=complexity,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def call(
        self,
        prompt_or_nothing: str | None = None,
        *,
        complexity: TaskComplexity,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Make an LLM call with automatic provider routing.

        Supports two calling conventions:
          - Full:   ``await llm.call(complexity=..., system=..., messages=[...])``
          - Short:  ``await llm.call(prompt, complexity=...)``
            (auto-wrapped into ``messages=[{"role": "user", "content": prompt}]``)

        Args:
            prompt_or_nothing: Optional positional prompt string (short form).
            complexity: Determines which model to use.
            system: System prompt.
            messages: Message history (normalized format).
            tools: Tool definitions (internal format).

        Returns:
            LLMResponse with normalized content blocks.
        """
        if kwargs:
            if "max_completion_tokens" in kwargs and max_tokens is None:
                alias_tokens = kwargs.pop("max_completion_tokens")
                if isinstance(alias_tokens, int):
                    max_tokens = alias_tokens
                else:
                    raise TypeError(
                        "max_completion_tokens must be int when used as "
                        "llm.call() compatibility alias"
                    )

            if kwargs:
                raise TypeError(
                    "llm.call() got unexpected keyword arguments: "
                    + ", ".join(sorted(kwargs))
                )

        # ── short-form compat: llm.call(prompt, complexity=...) ──
        if prompt_or_nothing is not None:
            if messages is not None:
                raise ValueError(
                    "Cannot pass both a positional prompt and messages=",
                )
            messages = [{"role": "user", "content": prompt_or_nothing}]
        elif messages is None:
            raise ValueError("Either a positional prompt or messages= is required")
        model, default_max_tokens = self._select_model(complexity)
        effective_max_tokens = default_max_tokens if max_tokens is None else max_tokens
        if effective_max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if temperature is None and bool(getattr(self.config, "deterministic", False)):
            temperature = 0.0
        provider = detect_provider(model)
        if provider == "openai":
            model = self._canonicalize_openai_model(model)
            self._last_openai_candidate = ""
        budget_enforced = self._is_budget_enforced(provider)
        reservation = 0.0
        actual_model = model
        budget_before = float(getattr(self.config, "estimated_cost_usd", 0.0))
        if response_json_schema is not None and provider == "anthropic":
            schema_hint = json.dumps(response_json_schema, ensure_ascii=False)
            if system:
                system = (
                    system
                    + "\n\nReturn exactly one JSON object that matches the schema below "
                    "and contains no extra prose.\n"
                    f"Schema:\n```json\n{schema_hint}\n```"
                )
            else:
                system = (
                    "Return exactly one JSON object that matches the schema below and contains no extra prose.\n"
                    f"Schema:\n```json\n{schema_hint}\n```"
                )
        lock = self._get_budget_lock()
        if budget_enforced:
            reservation = self._estimate_call_cost_usd(
                model=model,
                system=system,
                messages=messages,
                max_tokens=effective_max_tokens,
            )
            async with lock:
                projected = (
                    self.config.estimated_cost_usd
                    + self._reserved_budget_usd
                    + reservation
                )
                if projected > self.config.budget_limit_usd:
                    raise BudgetExceededError(
                        "Budget exhausted before call: "
                        f"used=${self.config.estimated_cost_usd:.4f}, "
                        f"reserved=${self._reserved_budget_usd:.4f}, "
                        f"next_estimate=${reservation:.4f}, "
                        f"limit=${self.config.budget_limit_usd:.4f}"
                    )
                self._reserved_budget_usd += reservation

        # Async-safe client initialization - prevent duplicate creation
        async with self._get_client_lock():
            self._get_client(provider)
        await self._ensure_fresh_token(provider)

        limiter = self._get_rate_limiter()
        if limiter is not None:
            try:
                est_tokens = self._estimate_call_tokens(
                    system=system,
                    messages=messages,
                    max_tokens=effective_max_tokens,
                )
                await limiter.acquire(estimated_tokens=est_tokens, requests=1)
            except Exception:
                # Best-effort only: never crash the pipeline due to rate limiting.
                pass

        self._call_count += 1
        call_id = self._call_count
        call_started = time.monotonic()

        logger.info(
            f"[LLM #{call_id}] provider={provider} model={model} "
            f"messages={len(messages)} budget_remaining=${self.config.budget_remaining:.2f}"
        )

        response: LLMResponse | None = None
        error_text = ""
        try:
            # Check custom providers first
            if provider in self._custom_providers:
                custom = self._custom_providers[provider]
                client = self._get_client(provider)
                try:
                    response = await self._retry_with_backoff(
                        lambda: custom.call(
                            client, model, system, messages, tools,
                            effective_max_tokens, response_json_schema,
                        )
                    )
                except TypeError:
                    response = await self._retry_with_backoff(
                        lambda: custom.call(
                            client, model, system, messages, tools,
                            effective_max_tokens,
                        )
                    )
            elif provider == "anthropic":
                response = await self._call_anthropic(
                    model,
                    effective_max_tokens,
                    system,
                    messages,
                    tools,
                    response_json_schema=response_json_schema,
                    temperature=temperature,
                )
                actual_model = model
            elif provider == "openai":
                response = await self._call_openai(
                    model,
                    effective_max_tokens,
                    system,
                    messages,
                    tools,
                    response_json_schema=response_json_schema,
                    temperature=temperature,
                )
                actual_model = str(self._last_openai_candidate or model)
            elif provider == "google":
                response = await self._call_google(
                    model,
                    effective_max_tokens,
                    system,
                    messages,
                    tools,
                    response_json_schema=response_json_schema,
                    temperature=temperature,
                )
                actual_model = model
            else:
                raise ValueError(f"Unknown provider: {provider}")
            return response
        except Exception as e:
            error_text = str(e)
            raise
        finally:
            if budget_enforced:
                async with lock:
                    self._reserved_budget_usd = max(0.0, self._reserved_budget_usd - reservation)

            if response is not None:
                self.config.record_usage(
                    actual_model,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )

            if response is not None:
                logger.info(
                    f"[LLM #{call_id}] stop_reason={response.stop_reason} "
                    f"tokens_in={response.usage.input_tokens} tokens_out={response.usage.output_tokens} "
                    f"cost_total=${self.config.estimated_cost_usd:.4f}"
                )
            if self._telemetry is not None:
                try:
                    payload: dict[str, Any] = {
                        "run_id": self.config.run_id,
                        "call_id": call_id,
                        "provider": provider,
                        "model": model,
                        "messages": len(messages),
                        "max_tokens": effective_max_tokens,
                        "schema_enforced": response_json_schema is not None,
                        "ok": response is not None,
                        "error": error_text,
                        "duration_seconds": time.monotonic() - call_started,
                        "cost_total_usd": float(getattr(self.config, "estimated_cost_usd", 0.0)),
                    }

                    capture = bool(getattr(self._telemetry, "capture_llm_content", False))
                    if capture:
                        def _jsonable(obj: Any) -> Any:
                            if obj is None or isinstance(obj, (str, int, float, bool)):
                                return obj
                            if isinstance(obj, dict):
                                return {str(k): _jsonable(v) for k, v in obj.items()}
                            if isinstance(obj, (list, tuple)):
                                return [_jsonable(v) for v in obj]
                            if isinstance(obj, ContentBlock):
                                return {
                                    "type": obj.type,
                                    "text": obj.text,
                                    "id": obj.id,
                                    "name": obj.name,
                                    "input": _jsonable(obj.input),
                                    "media_type": obj.media_type,
                                    "image_data": obj.image_data,
                                }
                            return str(obj)

                        payload["system"] = system
                        payload["messages_payload"] = _jsonable(messages)
                        if tools is not None:
                            payload["tools"] = _jsonable(tools)
                        if response is not None:
                            payload["response_content"] = [_jsonable(b) for b in response.content]
                    if response is not None:
                        payload.update(
                            {
                                "stop_reason": response.stop_reason,
                                "input_tokens": response.usage.input_tokens,
                                "output_tokens": response.usage.output_tokens,
                            }
                        )
                    self._telemetry.record_llm_call(payload)
                except Exception:
                    pass
            try:
                self._record_call_bundle(
                    call_id=call_id,
                    provider=provider,
                    requested_model=model,
                    actual_model=actual_model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    response=response,
                    error_text=error_text,
                    duration_seconds=time.monotonic() - call_started,
                    max_tokens=effective_max_tokens,
                    response_json_schema=response_json_schema,
                )
                self._record_budget_ledger(
                    call_id=call_id,
                    provider=provider,
                    actual_model=actual_model,
                    response=response,
                    budget_before=budget_before,
                )
            except Exception:
                pass

    # ── Anthropic ──────────────────────────────────────────────────

    async def _call_anthropic(
        self, model: str, max_tokens: int, system: str,
        messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
        response_json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Call Anthropic API and normalize the response."""
        client = self._get_client("anthropic")

        # Convert normalized messages to Anthropic format
        ant_messages = self._messages_to_anthropic(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": ant_messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = tools  # Already in Anthropic format

        raw = None
        schema_attempts: list[dict[str, Any]] = [{}]
        if response_json_schema is not None:
            schema_attempts = [
                {
                    "response_format": {
                        "type": "json",
                        "schema": response_json_schema,
                    }
                },
                {
                    "response_format": {
                        "type": "json",
                        "json_schema": {
                            "name": "auto_forge_schema",
                            "strict": True,
                            "schema": response_json_schema,
                        },
                    }
                },
                {},
            ]

        last_error: BaseException | None = None
        for schema_config in schema_attempts:
            attempt = dict(kwargs)
            attempt.update(schema_config)
            try:
                raw = await self._retry_with_backoff(
                    lambda: client.messages.create(**attempt)
                )
                break
            except TypeError:
                # Legacy SDK compatibility. Retry without response format.
                if schema_config == {}:
                    raise
                schema_attempts = [{}]
                last_error = None
                continue
            except Exception as exc:
                last_error = exc
                if response_json_schema is not None:
                    payload = str(exc).lower()
                    if any(token in payload for token in (
                        "response_format",
                        "json_schema",
                        "invalid_request_error",
                        "unsupported value",
                        "unrecognized field",
                    )):
                        continue
                break

        if raw is None:
            if last_error is not None:
                raise last_error
            raw = await self._retry_with_backoff(
                lambda: client.messages.create(**kwargs)
            )

        # Normalize response
        content = []
        for block in raw.content:
            if block.type == "text":
                content.append(ContentBlock(type="text", text=block.text))
            elif block.type == "tool_use":
                content.append(ContentBlock(
                    type="tool_use", id=block.id, name=block.name, input=block.input
                ))

        return LLMResponse(
            content=content,
            stop_reason=raw.stop_reason,
            usage=Usage(
                input_tokens=raw.usage.input_tokens,
                output_tokens=raw.usage.output_tokens,
            ),
        )

    def _messages_to_anthropic(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert normalized messages to Anthropic format.

        Normalized messages may contain ContentBlock dataclasses in assistant
        messages. Anthropic expects dicts.
        """
        result = []
        for msg in messages:
            content = msg.get("content")

            # If content is a list, check for ContentBlock instances
            if isinstance(content, list):
                converted = []
                for item in content:
                    if isinstance(item, ContentBlock):
                        if item.type == "text":
                            converted.append({"type": "text", "text": item.text})
                        elif item.type == "tool_use":
                            converted.append({
                                "type": "tool_use",
                                "id": item.id,
                                "name": item.name,
                                "input": item.input,
                            })
                        elif item.type == "image":
                            converted.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": item.media_type or "image/png",
                                    "data": item.image_data,
                                },
                            })
                    elif isinstance(item, dict) and item.get("type") == "image":
                        # Already in raw dict format
                        converted.append(item)
                    else:
                        converted.append(item)
                result.append({"role": msg["role"], "content": converted})
            else:
                result.append(msg)
        return result

    # ── OpenAI ─────────────────────────────────────────────────────

    async def _call_openai(
        self, model: str, max_tokens: int, system: str,
        messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
        response_json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Call OpenAI API and normalize the response with model fallback."""
        requested_key = model.lower()
        override = self._openai_model_overrides.get(requested_key)

        candidates = self._openai_model_candidates(model)
        if override:
            # Try the last known-good model first to avoid repeated 404 retries.
            candidates = [override] + [c for c in candidates if c != override]
        last_exc: Exception | None = None

        for idx, candidate in enumerate(candidates):
            try:
                resp = await self._call_openai_once(
                    candidate,
                    max_tokens,
                    system,
                    messages,
                    tools,
                    response_json_schema=response_json_schema,
                    temperature=temperature,
                )
                self._last_openai_candidate = candidate
                if candidate.lower() != requested_key and self._openai_model_overrides.get(requested_key) != candidate:
                    self._openai_model_overrides[requested_key] = candidate
                return resp
            except Exception as exc:
                last_exc = exc
                if self._is_openai_subscription_auth() and self._is_openai_codex_unsupported_model_error(exc):
                    raise RuntimeError(
                        f"OpenAI model {candidate!r} is not supported with Codex OAuth / Device Code "
                        "(ChatGPT subscription). Choose a GPT-5/Codex model (e.g. gpt-5.4, gpt-5.3-codex) "
                        "or switch OpenAI auth_method to api_key."
                    ) from exc
                if idx == len(candidates) - 1 or not self._is_openai_model_access_error(exc):
                    raise
                self._record_fallback_receipt(
                    requested_model=model,
                    failed_candidate=candidate,
                    fallback_candidate=candidates[idx + 1],
                    error=str(exc),
                )
                logger.warning(
                    "OpenAI model '%s' is unavailable (%s). Falling back to '%s'.",
                    candidate,
                    exc,
                    candidates[idx + 1],
                )

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OpenAI call failed without a concrete exception")

    async def _call_openai_once(
        self,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        response_json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        client = self._get_client("openai")
        is_reasoning = self._is_openai_reasoning_model(model)
        # The ChatGPT-subscription Codex backend requires an `instructions` field.
        # Even on standard Responses API, `instructions` is the preferred place
        # for the system prompt.
        instructions = (system or "").strip() or "You are a helpful assistant."
        input_items = self._messages_to_openai_responses_input("", messages, is_reasoning=is_reasoning)
        kwargs: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "instructions": instructions,
            "max_output_tokens": max_tokens,
        }
        if is_reasoning:
            # Responses API uses nested `reasoning` config.
            kwargs["reasoning"] = {"effort": self.config.openai_reasoning_effort}
        elif temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = self._tools_to_openai(tools)

        # Structured outputs: prefer JSON schema; fallback to json_object; then free-form.
        text_chain: list[dict[str, Any] | None] = [None]
        if response_json_schema is not None and not is_reasoning:
            text_chain = [
                {
                    "format": {
                        "type": "json_schema",
                        "name": "auto_forge_schema",
                        "schema": response_json_schema,
                        "strict": True,
                    }
                },
                {"format": {"type": "json_object"}},
                None,
            ]

        raw: Any | None = None
        last_error: BaseException | None = None
        try:
            for text_cfg in text_chain:
                if text_cfg is None:
                    kwargs.pop("text", None)
                else:
                    kwargs["text"] = text_cfg
                try:
                    raw = await self._retry_with_backoff(
                        lambda: client.responses.create(**kwargs),
                    )
                    break
                except TypeError:
                    # Legacy SDK compatibility: retry without `text` formatting.
                    if text_cfg is None:
                        raise
                    text_chain = [None]
                    last_error = None
                    continue
                except Exception as exc:  # pragma: no cover - API compatibility boundary
                    if text_cfg is not None and self._is_openai_schema_unsupported_error(exc):
                        last_error = exc
                        continue
                    raise
            else:
                if last_error is not None:
                    raise last_error
                raw = await self._retry_with_backoff(
                    lambda: client.responses.create(**kwargs),
                )

            return self._normalize_openai_responses(raw)
        except Exception as exc:
            # Some OpenAI-compatible proxies (Azure/LiteLLM) still only implement
            # Chat Completions. If `/responses` isn't supported, fall back.
            if self._is_openai_model_access_error(exc):
                raise
            if self._is_openai_responses_unsupported_error(exc):
                logger.warning(
                    "OpenAI base_url does not support /responses (%s). Falling back to /chat/completions.",
                    exc,
                )
                return await self._call_openai_chat_completions_once(
                    model,
                    max_tokens,
                    system,
                    messages,
                    tools,
                    response_json_schema=response_json_schema,
                    temperature=temperature,
                )
            raise

    async def _call_openai_chat_completions_once(
        self,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        response_json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Fallback path for OpenAI-compatible providers without `/responses`."""
        client = self._get_client("openai")
        is_reasoning = self._is_openai_reasoning_model(model)
        token_key = "max_completion_tokens" if is_reasoning else "max_tokens"

        oai_messages = self._messages_to_openai(system, messages, is_reasoning=is_reasoning)
        kwargs: dict[str, Any] = {
            "model": model,
            token_key: max_tokens,
            "messages": oai_messages,
        }
        if is_reasoning:
            kwargs["reasoning_effort"] = self.config.openai_reasoning_effort
        elif temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = self._tools_to_openai(tools)

        # Schema support: prefer JSON schema, fallback to json_object.
        response_format_chain: list[dict[str, Any] | None] = [None]
        if response_json_schema is not None and not is_reasoning:
            response_format_chain = [
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "auto_forge_schema",
                        "strict": True,
                        "schema": response_json_schema,
                    },
                },
                {"type": "json_object"},
            ]

        raw: Any | None = None
        last_error: BaseException | None = None
        for response_format in response_format_chain:
            if response_format is None:
                kwargs.pop("response_format", None)
            else:
                kwargs["response_format"] = response_format
            try:
                raw = await self._retry_with_backoff(
                    lambda: client.chat.completions.create(**kwargs),
                )
                break
            except TypeError:
                # Legacy SDK does not accept response_format parameter.
                if response_format is None:
                    raise
                response_format_chain = [None]
                last_error = None
                continue
            except Exception as exc:  # pragma: no cover - API compatibility boundary
                if response_format is not None and self._is_openai_schema_unsupported_error(exc):
                    last_error = exc
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error
            raw = await self._retry_with_backoff(
                lambda: client.chat.completions.create(**kwargs),
            )

        choice = raw.choices[0]
        message = choice.message

        content: list[ContentBlock] = []
        if message.content:
            content.append(ContentBlock(type="text", text=message.content))

        stop_reason = "end_turn"
        if message.tool_calls:
            stop_reason = "tool_use"
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                content.append(ContentBlock(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=args,
                ))

        usage = Usage()
        if raw.usage:
            usage = Usage(
                input_tokens=raw.usage.prompt_tokens or 0,
                output_tokens=raw.usage.completion_tokens or 0,
            )
        return LLMResponse(content=content, stop_reason=stop_reason, usage=usage)

    def _normalize_openai_responses(self, raw: Any) -> LLMResponse:
        """Normalize OpenAI Responses API output into ContentBlocks."""
        content: list[ContentBlock] = []
        stop_reason = "end_turn"

        outputs = getattr(raw, "output", None) or []
        for item in outputs:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                # Assistant message with structured content parts (output_text, etc.)
                for part in getattr(item, "content", None) or []:
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        text = getattr(part, "text", "") or ""
                        if text:
                            content.append(ContentBlock(type="text", text=text))
            elif item_type == "function_call":
                stop_reason = "tool_use"
                args_raw = getattr(item, "arguments", "") or ""
                try:
                    args = json.loads(args_raw) if args_raw else {}
                except Exception:
                    args = {}
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None) or ""
                name = getattr(item, "name", "") or ""
                content.append(ContentBlock(
                    type="tool_use",
                    id=str(call_id),
                    name=str(name),
                    input=args,
                ))

        # Fallback: if SDK exposes aggregated output_text and we produced no text blocks.
        if not any(b.type == "text" for b in content):
            text = getattr(raw, "output_text", "") or ""
            if text:
                content.append(ContentBlock(type="text", text=text))

        usage = Usage()
        raw_usage = getattr(raw, "usage", None)
        if raw_usage:
            if isinstance(raw_usage, dict):
                usage = Usage(
                    input_tokens=int(raw_usage.get("input_tokens", 0) or 0),
                    output_tokens=int(raw_usage.get("output_tokens", 0) or 0),
                )
            else:
                usage = Usage(
                    input_tokens=int(getattr(raw_usage, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(raw_usage, "output_tokens", 0) or 0),
                )

        return LLMResponse(content=content, stop_reason=stop_reason, usage=usage)

    def _messages_to_openai_responses_input(
        self,
        system: str,
        messages: list[dict[str, Any]],
        *,
        is_reasoning: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert normalized messages to Responses API input items.

        We intentionally support tool-use history by emitting `function_call`
        items for assistant tool invocations and `function_call_output` items
        for tool results.
        """
        items: list[dict[str, Any]] = []

        sys_role = "developer" if is_reasoning else "system"
        if system:
            items.append({"type": "message", "role": sys_role, "content": system})

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "assistant":
                blocks = content if isinstance(content, list) else [content]
                text_parts: list[str] = []
                tool_blocks: list[ContentBlock] = []
                for block in blocks:
                    if isinstance(block, ContentBlock):
                        if block.type == "text" and block.text:
                            text_parts.append(block.text)
                        elif block.type == "tool_use":
                            tool_blocks.append(block)
                        elif block.type == "image" and block.image_data:
                            # Best-effort: represent assistant images as text.
                            text_parts.append("[image]")
                    elif isinstance(block, dict):
                        btype = block.get("type")
                        if btype == "text":
                            text_parts.append(str(block.get("text", "")))
                        elif btype == "tool_use":
                            tool_blocks.append(ContentBlock(
                                type="tool_use",
                                id=str(block.get("id", "")),
                                name=str(block.get("name", "")),
                                input=block.get("input", {}) or {},
                            ))
                        elif btype == "image":
                            text_parts.append("[image]")
                        else:
                            text_parts.append(str(block))
                    elif isinstance(block, str):
                        text_parts.append(block)

                if text_parts:
                    items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": "\n".join([t for t in text_parts if t]),
                    })

                for tb in tool_blocks:
                    call_id = str(tb.id or "")
                    name = str(tb.name or "")
                    try:
                        arguments = json.dumps(tb.input or {}, ensure_ascii=False)
                    except Exception:
                        arguments = "{}"
                    items.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                        "id": call_id,
                        "status": "completed",
                    })

            elif role == "user":
                # User message may contain tool results + images + text parts.
                if isinstance(content, list):
                    msg_parts: list[dict[str, Any]] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "tool_result":
                            call_id = str(part.get("tool_use_id", "") or "")
                            items.append({
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": str(part.get("content", "")),
                                "status": "completed",
                            })
                            continue

                        if isinstance(part, ContentBlock) and part.type == "image" and part.image_data:
                            mt = part.media_type or "image/png"
                            msg_parts.append({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{mt};base64,{part.image_data}",
                            })
                            continue

                        if isinstance(part, dict) and part.get("type") == "image":
                            src = part.get("source", {})
                            mt = src.get("media_type", "image/png")
                            data = src.get("data", "")
                            msg_parts.append({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{mt};base64,{data}",
                            })
                            continue

                        text = part if isinstance(part, str) else (
                            part.text if isinstance(part, ContentBlock) else str(part)
                        )
                        if text:
                            msg_parts.append({"type": "input_text", "text": str(text)})

                    if msg_parts:
                        # Content can be a list of input_text/input_image items.
                        items.append({"type": "message", "role": "user", "content": msg_parts})
                else:
                    if content is None:
                        continue
                    items.append({"type": "message", "role": "user", "content": str(content)})

        return items

    def _messages_to_openai(
        self, system: str, messages: list[dict[str, Any]], *,
        is_reasoning: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert normalized messages to OpenAI format."""
        result: list[dict[str, Any]] = []
        if is_reasoning:
            # o1/o3/o4/codex models use "developer" role instead of "system"
            result.append({"role": "developer", "content": system})
        else:
            result.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg.get("content")

            if role == "assistant":
                # May contain ContentBlock instances
                oai_msg: dict[str, Any] = {"role": "assistant"}
                text_parts = []
                tool_calls = []

                items = content if isinstance(content, list) else [content]
                for item in items:
                    if isinstance(item, ContentBlock):
                        if item.type == "text":
                            text_parts.append(item.text)
                        elif item.type == "tool_use":
                            tool_calls.append({
                                "id": item.id,
                                "type": "function",
                                "function": {
                                    "name": item.name,
                                    "arguments": json.dumps(item.input),
                                },
                            })
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "tool_use":
                            tool_calls.append({
                                "id": item["id"],
                                "type": "function",
                                "function": {
                                    "name": item["name"],
                                    "arguments": json.dumps(item.get("input", {})),
                                },
                            })
                    elif isinstance(item, str):
                        text_parts.append(item)

                oai_msg["content"] = "\n".join(text_parts) if text_parts else None
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                result.append(oai_msg)

            elif role == "user":
                # May contain tool_result items, images, or plain text.
                # Collect tool results and user content separately to avoid
                # interleaving - OpenAI expects all tool messages grouped
                # right after the assistant tool_calls message.
                if isinstance(content, list):
                    tool_msgs: list[dict[str, Any]] = []
                    user_content_parts: list[dict[str, Any]] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            tool_msgs.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item.get("content", ""),
                            })
                        elif isinstance(item, ContentBlock) and item.type == "image":
                            # OpenAI vision: data URI format
                            mt = item.media_type or "image/png"
                            user_content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mt};base64,{item.image_data}",
                                },
                            })
                        elif isinstance(item, dict) and item.get("type") == "image":
                            src = item.get("source", {})
                            mt = src.get("media_type", "image/png")
                            data = src.get("data", "")
                            user_content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{mt};base64,{data}"},
                            })
                        else:
                            text = item if isinstance(item, str) else (
                                item.text if isinstance(item, ContentBlock) else str(item)
                            )
                            user_content_parts.append({"type": "text", "text": text})
                    # Emit tool messages first (they belong to the preceding
                    # assistant turn), then a single user message if any.
                    result.extend(tool_msgs)
                    if user_content_parts:
                        result.append({
                            "role": "user",
                            "content": user_content_parts,
                        })
                elif isinstance(content, str):
                    result.append({"role": "user", "content": content})
                else:
                    result.append({"role": "user", "content": str(content)})

        return result

    def _tools_to_openai(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal tool definitions to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    @staticmethod
    def _is_openai_schema_unsupported_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        text = str(getattr(exc, "message", msg)).lower()
        payload = f"{msg} {text}"
        detail = getattr(exc, "error", None)
        if detail is not None:
            payload += f" {detail}"
        try:
            status = getattr(exc, "status")
            status_code = getattr(exc, "status_code")
            if status in ("400", 400) or status_code in ("400", 400):
                return "response_format" in payload or "json_schema" in payload
        except Exception:
            pass

        return any(marker in payload for marker in (
            "response_format",
            "json_schema",
            "unexpected argument",
            "unexpected keyword",
            "unsupported parameter",
        ))

    @staticmethod
    def _is_openai_responses_unsupported_error(exc: BaseException) -> bool:
        """Detect when an OpenAI-compatible endpoint doesn't implement `/responses`."""
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        try:
            status_int = int(status) if status is not None else None
        except Exception:
            status_int = None

        msg = str(exc).lower()
        req = getattr(exc, "request", None)
        url = ""
        try:
            if req is not None and getattr(req, "url", None) is not None:
                url = str(req.url).lower()
        except Exception:
            url = ""
        haystack = f"{msg} {url}"

        if status_int in (404, 405):
            # Only treat as unsupported when the missing route is the responses endpoint.
            return "/responses" in haystack or " responses" in haystack

        if status_int == 400 and any(marker in haystack for marker in (
            "unknown endpoint",
            "unknown url",
            "unrecognized path",
            "not found",
        )) and "responses" in haystack:
            return True
        return False

    # ── Google Gemini ──────────────────────────────────────────────

    async def _call_google(
        self, model: str, max_tokens: int, system: str,
        messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
        response_json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Call Google Gemini API and normalize the response."""
        from google import genai
        from google.genai import types

        client = self._get_client("google")

        # Build contents
        contents = self._messages_to_gemini(messages)

        # Build config
        config: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "system_instruction": system,
        }
        if temperature is not None:
            config["temperature"] = temperature

        # Convert tools
        gemini_tools = None
        if tools:
            gemini_tools = self._tools_to_gemini(tools)
            config["tools"] = gemini_tools

        schema_attempts: list[dict[str, Any]] = [{}]
        if response_json_schema is not None:
            schema_attempts = [
                {
                    "response_mime_type": "application/json",
                    "response_schema": response_json_schema,
                },
                {
                    "response_mime_type": "application/json",
                    "response_json_schema": {
                        "name": "auto_forge_schema",
                        "schema": response_json_schema,
                        "strict": True,
                    },
                },
                {"response_mime_type": "application/json"},
                {},
            ]

        last_error: BaseException | None = None
        raw: Any | None = None
        for schema_config in schema_attempts:
            attempt = dict(config)
            attempt.update(schema_config)
            try:
                raw = await self._retry_with_backoff(
                    lambda: client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(**attempt),
                    )
                )
                break
            except TypeError:
                raw = None
                last_error = None
                continue
            except Exception as exc:
                last_error = exc
                if response_json_schema is not None:
                    payload = str(exc).lower()
                    if any(token in payload for token in (
                        "response_json_schema",
                        "response_schema",
                        "response_mime_type",
                        "unrecognized fields",
                    )):
                        continue
                break

        if raw is None:
            if last_error is not None:
                raise last_error
            raw = await self._retry_with_backoff(
                lambda: client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config),
                )
            )

        # Normalize response
        content: list[ContentBlock] = []
        stop_reason = "end_turn"

        if raw.candidates and raw.candidates[0].content:
            for part in raw.candidates[0].content.parts:
                if part.text:
                    content.append(ContentBlock(type="text", text=part.text))
                elif part.function_call:
                    stop_reason = "tool_use"
                    fc = part.function_call
                    content.append(ContentBlock(
                        type="tool_use",
                        id=uuid.uuid4().hex[:12],
                        name=fc.name,
                        input=dict(fc.args) if fc.args else {},
                    ))

        usage = Usage()
        if raw.usage_metadata:
            usage = Usage(
                input_tokens=raw.usage_metadata.prompt_token_count or 0,
                output_tokens=raw.usage_metadata.candidates_token_count or 0,
            )

        return LLMResponse(content=content, stop_reason=stop_reason, usage=usage)

    def _messages_to_gemini(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Convert normalized messages to Gemini contents format."""
        from google.genai import types

        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            content = msg.get("content")

            parts = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, ContentBlock):
                        if item.type == "text":
                            parts.append(types.Part.from_text(text=item.text))
                        elif item.type == "tool_use":
                            parts.append(types.Part(function_call=types.FunctionCall(
                                name=item.name, args=item.input
                            )))
                        elif item.type == "image":
                            import base64 as _b64
                            raw = _b64.b64decode(item.image_data)
                            mt = item.media_type or "image/png"
                            parts.append(types.Part.from_data(data=raw, mime_type=mt))
                    elif isinstance(item, dict):
                        if item.get("type") == "tool_result":
                            parts.append(types.Part(function_response=types.FunctionResponse(
                                name=item.get("tool_name", "tool"),
                                response={"result": item.get("content", "")},
                            )))
                        elif item.get("type") == "text":
                            parts.append(types.Part.from_text(text=item.get("text", "")))
                        elif item.get("type") == "image":
                            import base64 as _b64
                            src = item.get("source", {})
                            raw = _b64.b64decode(src.get("data", ""))
                            mt = src.get("media_type", "image/png")
                            parts.append(types.Part.from_data(data=raw, mime_type=mt))
                    elif isinstance(item, str):
                        parts.append(types.Part.from_text(text=item))
            elif isinstance(content, str):
                parts.append(types.Part.from_text(text=content))

            if parts:
                contents.append(types.Content(role=role, parts=parts))

        return contents

    def _tools_to_gemini(self, tools: list[dict[str, Any]]) -> list[Any]:
        """Convert internal tool definitions to Gemini format."""
        from google.genai import types

        declarations = []
        for t in tools:
            declarations.append(types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("input_schema"),
            ))
        return [types.Tool(function_declarations=declarations)]

    # ── Properties ─────────────────────────────────────────────────

    @property
    def total_cost(self) -> float:
        return self.config.estimated_cost_usd

    @property
    def call_count(self) -> int:
        return self._call_count
