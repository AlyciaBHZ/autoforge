"""Authentication providers for AutoForge LLM integrations.

Supports:
- API key (passthrough, no token management)
- OAuth bearer token (static token for proxies like LiteLLM, Azure OpenAI)
- OAuth2 client_credentials flow (token exchange with auto-refresh)
- Google Application Default Credentials (ADC)
- Google service account JSON
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ── Token result ──────────────────────────────────────────────────


@dataclass
class TokenResult:
    """Result of a token acquisition."""

    access_token: str
    expires_at: float = 0.0  # Unix timestamp; 0 = never expires
    token_type: str = "Bearer"

    @property
    def is_expired(self) -> bool:
        """Check if the token is expired (with 60s safety buffer)."""
        if self.expires_at == 0.0:
            return False
        return time.time() >= (self.expires_at - 60)


# ── Auth provider base ────────────────────────────────────────────


class AuthProvider(ABC):
    """Base class for authentication providers."""

    @abstractmethod
    async def get_token(self) -> TokenResult:
        """Get a valid access token, refreshing if necessary."""
        ...

    @abstractmethod
    def get_client_kwargs(self) -> dict[str, Any]:
        """Return kwargs to pass to the LLM client constructor."""
        ...


# ── API key (default, backward-compatible) ────────────────────────


class ApiKeyAuth(AuthProvider):
    """Simple API key authentication (passthrough)."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url

    async def get_token(self) -> TokenResult:
        return TokenResult(access_token=self._api_key)

    def get_client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return kwargs


# ── Bearer token (LiteLLM proxy, Azure OpenAI, etc.) ─────────────


class OAuthBearerAuth(AuthProvider):
    """Static bearer token auth for OpenAI-compatible proxies."""

    def __init__(self, bearer_token: str, base_url: str) -> None:
        self._token = bearer_token
        self._base_url = base_url

    async def get_token(self) -> TokenResult:
        return TokenResult(access_token=self._token)

    def get_client_kwargs(self) -> dict[str, Any]:
        return {
            "api_key": self._token,
            "base_url": self._base_url,
        }


# ── OAuth2 client_credentials flow ───────────────────────────────


class OAuth2ClientCredentialsAuth(AuthProvider):
    """OAuth2 client_credentials flow with automatic token refresh.

    Exchanges client_id + client_secret at token_url for an access token.
    Caches the token and automatically refreshes when expired.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scope: str = "",
        base_url: str | None = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._scope = scope
        self._base_url = base_url
        self._cached_token: TokenResult | None = None
        self._lock = asyncio.Lock()

    async def get_token(self) -> TokenResult:
        """Get a valid token, refreshing if expired."""
        async with self._lock:
            if self._cached_token and not self._cached_token.is_expired:
                return self._cached_token

            import httpx

            async with httpx.AsyncClient(timeout=15) as client:
                data: dict[str, str] = {
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                }
                if self._scope:
                    data["scope"] = self._scope

                resp = await client.post(self._token_url, data=data)
                resp.raise_for_status()
                body = resp.json()

            expires_in = body.get("expires_in", 3600)
            self._cached_token = TokenResult(
                access_token=body["access_token"],
                expires_at=time.time() + expires_in,
                token_type=body.get("token_type", "Bearer"),
            )
            logger.info(f"OAuth2 token acquired, expires in {expires_in}s")
            return self._cached_token

    def get_client_kwargs(self) -> dict[str, Any]:
        """Return constructor kwargs (token injected separately via get_token)."""
        kwargs: dict[str, Any] = {}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        # api_key will be set dynamically from get_token() result
        return kwargs


# ── Google Application Default Credentials ────────────────────────


class GoogleADCAuth(AuthProvider):
    """Google Application Default Credentials (ADC) or service account JSON.

    Uses the google-auth library to obtain credentials. When no
    service_account_path is given, falls back to Application Default
    Credentials (respects GOOGLE_APPLICATION_CREDENTIALS env var and
    gcloud auth application-default login).
    """

    def __init__(self, service_account_path: str | None = None) -> None:
        self._sa_path = service_account_path
        self._credentials: Any = None

    async def get_token(self) -> TokenResult:
        """Get a valid Google access token."""
        import google.auth
        import google.auth.transport.requests

        if self._credentials is None:
            if self._sa_path:
                from google.oauth2 import service_account

                self._credentials = service_account.Credentials.from_service_account_file(
                    self._sa_path,
                    scopes=["https://www.googleapis.com/auth/generative-language"],
                )
            else:
                self._credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/generative-language"],
                )

        if not self._credentials.valid:
            request = google.auth.transport.requests.Request()
            self._credentials.refresh(request)

        expires_at = 0.0
        if self._credentials.expiry:
            expires_at = self._credentials.expiry.timestamp()

        return TokenResult(
            access_token=self._credentials.token or "",
            expires_at=expires_at,
        )

    def get_client_kwargs(self) -> dict[str, Any]:
        """Google genai.Client() uses ADC automatically when no api_key given."""
        return {}


# ── Factory ───────────────────────────────────────────────────────


def create_auth_provider(
    provider: str,
    api_keys: dict[str, str],
    auth_config: dict[str, dict[str, Any]],
) -> AuthProvider:
    """Create the appropriate AuthProvider for a given LLM provider.

    Falls back to ApiKeyAuth if no OAuth config is present (backward-compatible).
    """
    config = auth_config.get(provider, {})
    auth_method = config.get("auth_method", "api_key")
    api_key = api_keys.get(provider, "")

    # Default: plain API key
    if auth_method == "api_key" or (not config and api_key):
        return ApiKeyAuth(
            api_key=api_key,
            base_url=config.get("base_url"),
        )

    # Bearer token + custom base URL
    if auth_method == "oauth_bearer":
        return OAuthBearerAuth(
            bearer_token=config.get("bearer_token", api_key),
            base_url=config["base_url"],
        )

    # OAuth2 client credentials flow
    if auth_method == "oauth2_client_credentials":
        return OAuth2ClientCredentialsAuth(
            client_id=config["client_id"],
            client_secret=config["client_secret"],
            token_url=config["token_url"],
            scope=config.get("scope", ""),
            base_url=config.get("base_url"),
        )

    # Google ADC / service account
    if auth_method in ("adc", "service_account"):
        return GoogleADCAuth(
            service_account_path=config.get("service_account_path"),
        )

    # Unknown method — fallback with warning
    logger.warning(f"Unknown auth_method '{auth_method}' for {provider}, using api_key")
    return ApiKeyAuth(api_key=api_key)
