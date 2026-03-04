"""LLM Router — multi-provider model selection, API calls, and budget enforcement.

Supports Anthropic (Claude), OpenAI (GPT/o-series), and Google (Gemini).
All responses are normalized to a common format so agents don't need
provider-specific code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from autoforge.engine.config import ForgeConfig

logger = logging.getLogger(__name__)


# ── Normalized response types ──────────────────────────────────────


@dataclass
class ContentBlock:
    """A block of content in an LLM response (text or tool_use)."""

    type: str  # "text" or "tool_use"
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    """Token usage for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    """Normalized LLM response — same attribute interface as Anthropic's Message."""

    content: list[ContentBlock] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" or "tool_use"
    usage: Usage = field(default_factory=Usage)


# ── Enums ──────────────────────────────────────────────────────────


class TaskComplexity(Enum):
    """Task complexity determines which model to use."""

    HIGH = "high"  # Uses model_strong (e.g. Opus, GPT-4o, Gemini Pro)
    STANDARD = "standard"  # Uses model_fast (e.g. Sonnet, GPT-4o-mini, Flash)


class BudgetExceededError(Exception):
    """Raised when the API budget limit is reached."""


# ── Retry configuration ───────────────────────────────────────────

MAX_RETRIES = 3
"""Maximum number of retry attempts for transient API errors."""

_RETRY_BASE_DELAY = 1.0  # seconds — base delay for exponential backoff


# ── Provider detection ─────────────────────────────────────────────

# Known OpenAI model prefixes/names
_OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
                  "o3", "o3-mini", "o4-mini", "o1", "o1-mini", "o1-preview"}


def detect_provider(model: str) -> str:
    """Detect LLM provider from model name.

    Returns "anthropic", "openai", or "google".
    """
    model_lower = model.lower()

    # OpenAI: check exact matches and prefixes
    if model_lower in _OPENAI_MODELS or model_lower.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"

    # Google Gemini
    if model_lower.startswith("gemini"):
        return "google"

    # Default: Anthropic (claude-* and anything else)
    return "anthropic"


# ── LLM Router ─────────────────────────────────────────────────────


class LLMRouter:
    """Routes LLM calls to appropriate providers and tracks usage.

    All agent interactions with LLM APIs go through this class.
    It handles provider detection, model selection, format conversion,
    budget enforcement, and usage logging.
    """

    def __init__(self, config: ForgeConfig) -> None:
        self.config = config
        self._call_count = 0
        self._clients: dict[str, Any] = {}
        self._auth_providers: dict[str, Any] = {}  # Cached AuthProvider per provider

    def _get_client(self, provider: str) -> Any:
        """Get or create an async client for the given provider.

        Uses the auth module to determine authentication method (API key,
        OAuth bearer, OAuth2 client_credentials, Google ADC/service account,
        AWS Bedrock, Google Vertex AI, Codex OAuth, Device Code).
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
                    "Install it with: pip install autoforge[openai]"
                ) from None
            if isinstance(auth, (CodexOAuthAuth, DeviceCodeAuth)):
                # Token-based auth: client gets a dummy key, real token
                # injected via _ensure_fresh_token before each call
                client = AsyncOpenAI(api_key="placeholder")
            else:
                client = AsyncOpenAI(**client_kwargs)

        elif provider == "google":
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "Google provider requires the 'google-genai' package. "
                    "Install it with: pip install autoforge[google]"
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
                jitter = delay * 0.25 * (2 * random.random() - 1)
                delay = max(0, delay + jitter)

                logger.warning(
                    "Transient error on attempt %d/%d (status=%s): %s — "
                    "retrying in %.1fs",
                    attempt, max_retries, status, exc, delay,
                )
                await asyncio.sleep(delay)

        # Should never reach here, but satisfy type checker
        assert last_exc is not None
        raise last_exc

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
    ) -> LLMResponse:
        """Make an LLM call with automatic provider routing.

        Args:
            complexity: Determines which model to use.
            system: System prompt.
            messages: Message history (normalized format).
            tools: Tool definitions (internal format).

        Returns:
            LLMResponse with normalized content blocks.
        """
        if not self.config.check_budget():
            raise BudgetExceededError(
                f"Budget exhausted: ${self.config.estimated_cost_usd:.2f} "
                f"of ${self.config.budget_limit_usd:.2f} used"
            )

        model, max_tokens = self._select_model(complexity)
        provider = detect_provider(model)
        await self._ensure_fresh_token(provider)
        self._call_count += 1
        call_id = self._call_count

        logger.info(
            f"[LLM #{call_id}] provider={provider} model={model} "
            f"messages={len(messages)} budget_remaining=${self.config.budget_remaining:.2f}"
        )

        if provider == "anthropic":
            response = await self._call_anthropic(model, max_tokens, system, messages, tools)
        elif provider == "openai":
            response = await self._call_openai(model, max_tokens, system, messages, tools)
        elif provider == "google":
            response = await self._call_google(model, max_tokens, system, messages, tools)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Track usage
        self.config.record_usage(model, response.usage.input_tokens, response.usage.output_tokens)

        logger.info(
            f"[LLM #{call_id}] stop_reason={response.stop_reason} "
            f"tokens_in={response.usage.input_tokens} tokens_out={response.usage.output_tokens} "
            f"cost_total=${self.config.estimated_cost_usd:.4f}"
        )

        return response

    # ── Anthropic ──────────────────────────────────────────────────

    async def _call_anthropic(
        self, model: str, max_tokens: int, system: str,
        messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
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
        if tools:
            kwargs["tools"] = tools  # Already in Anthropic format

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
    ) -> LLMResponse:
        """Call OpenAI API and normalize the response."""
        client = self._get_client("openai")

        # Convert messages to OpenAI format
        oai_messages = self._messages_to_openai(system, messages)

        # OpenAI reasoning models (o1, o3, o4) use max_completion_tokens
        model_lower = model.lower()
        is_reasoning = model_lower.startswith(("o1", "o3", "o4"))
        token_key = "max_completion_tokens" if is_reasoning else "max_tokens"

        kwargs: dict[str, Any] = {
            "model": model,
            token_key: max_tokens,
            "messages": oai_messages,
        }
        if tools:
            kwargs["tools"] = self._tools_to_openai(tools)

        raw = await self._retry_with_backoff(
            lambda: client.chat.completions.create(**kwargs)
        )
        choice = raw.choices[0]
        message = choice.message

        # Normalize response
        content: list[ContentBlock] = []

        if message.content:
            content.append(ContentBlock(type="text", text=message.content))

        # Tool calls
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

    def _messages_to_openai(
        self, system: str, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert normalized messages to OpenAI format."""
        result: list[dict[str, Any]] = [{"role": "system", "content": system}]

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
                # May contain tool_result items or plain text.
                # Collect tool results and user text separately to avoid
                # interleaving — OpenAI expects all tool messages grouped
                # right after the assistant tool_calls message.
                if isinstance(content, list):
                    tool_msgs: list[dict[str, Any]] = []
                    user_text_parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            tool_msgs.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item.get("content", ""),
                            })
                        else:
                            text = item if isinstance(item, str) else str(item)
                            user_text_parts.append(text)
                    # Emit tool messages first (they belong to the preceding
                    # assistant turn), then a single user message if any.
                    result.extend(tool_msgs)
                    if user_text_parts:
                        result.append({
                            "role": "user",
                            "content": "\n".join(user_text_parts),
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

    # ── Google Gemini ──────────────────────────────────────────────

    async def _call_google(
        self, model: str, max_tokens: int, system: str,
        messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
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

        # Convert tools
        gemini_tools = None
        if tools:
            gemini_tools = self._tools_to_gemini(tools)
            config["tools"] = gemini_tools

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
                    elif isinstance(item, dict):
                        if item.get("type") == "tool_result":
                            parts.append(types.Part(function_response=types.FunctionResponse(
                                name=item.get("tool_name", "tool"),
                                response={"result": item.get("content", "")},
                            )))
                        elif item.get("type") == "text":
                            parts.append(types.Part.from_text(text=item.get("text", "")))
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
