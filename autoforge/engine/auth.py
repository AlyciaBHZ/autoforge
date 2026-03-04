"""Authentication providers for AutoForge LLM integrations.

Supports:
- API key (passthrough, no token management)
- OAuth bearer token (static token for proxies like LiteLLM, Azure OpenAI)
- OAuth2 client_credentials flow (token exchange with auto-refresh)
- Google Application Default Credentials (ADC)
- Google service account JSON
- Amazon Bedrock (AWS credential chain)
- Google Vertex AI (Claude via Vertex)
- OpenAI Codex OAuth (browser-based PKCE flow)
- OpenAI Device Code flow (headless environments)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import secrets
import time
import base64
import urllib.parse
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
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy-initialize the asyncio.Lock to avoid cross-event-loop issues."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_token(self) -> TokenResult:
        """Get a valid token, refreshing if expired."""
        async with self._get_lock():
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
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy-initialize the asyncio.Lock to avoid cross-event-loop issues."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_token(self) -> TokenResult:
        """Get a valid Google access token."""
        import google.auth
        import google.auth.transport.requests

        async with self._get_lock():
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
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._credentials.refresh, request)

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


# ── Amazon Bedrock (AWS credential chain) ─────────────────────────


class AWSBedrockAuth(AuthProvider):
    """AWS Bedrock authentication using the AWS credential chain.

    Supports:
    - Static access keys (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
    - AWS profiles (AWS_PROFILE for SSO/IAM Identity Center)
    - Instance roles (automatic on EC2, ECS, Lambda)

    The Anthropic SDK's AsyncAnthropicBedrock client handles the actual
    AWS auth, so this provider mainly stores config and passes it through.
    """

    def __init__(
        self,
        aws_region: str = "",
        aws_profile: str = "",
        aws_access_key_id: str = "",
        aws_secret_access_key: str = "",
        aws_session_token: str = "",
    ) -> None:
        self._region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self._profile = aws_profile or os.getenv("AWS_PROFILE", "")
        self._access_key = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID", "")
        self._secret_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY", "")
        self._session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN", "")

    async def get_token(self) -> TokenResult:
        # Bedrock uses AWS SigV4, not bearer tokens
        return TokenResult(access_token="bedrock-sigv4")

    def get_client_kwargs(self) -> dict[str, Any]:
        """Return kwargs for AsyncAnthropicBedrock constructor."""
        kwargs: dict[str, Any] = {
            "aws_region": self._region,
        }
        if self._access_key and self._secret_key:
            kwargs["aws_access_key"] = self._access_key
            kwargs["aws_secret_key"] = self._secret_key
            if self._session_token:
                kwargs["aws_session_token"] = self._session_token
        if self._profile:
            kwargs["aws_profile"] = self._profile
        return kwargs


# ── Google Vertex AI (Claude via Vertex) ──────────────────────────


class VertexAIAuth(AuthProvider):
    """Google Vertex AI authentication for Claude models.

    The Anthropic SDK's AsyncAnthropicVertex client uses ADC automatically.
    This provider stores project_id and region config.
    """

    def __init__(
        self,
        project_id: str = "",
        region: str = "",
    ) -> None:
        self._project_id = project_id or os.getenv("ANTHROPIC_VERTEX_PROJECT_ID", "")
        self._region = region or os.getenv("CLOUD_ML_REGION", "us-east5")

    async def get_token(self) -> TokenResult:
        # Vertex uses ADC, handled by the SDK
        return TokenResult(access_token="vertex-adc")

    def get_client_kwargs(self) -> dict[str, Any]:
        """Return kwargs for AsyncAnthropicVertex constructor."""
        kwargs: dict[str, Any] = {
            "region": self._region,
        }
        if self._project_id:
            kwargs["project_id"] = self._project_id
        return kwargs


# ── OpenAI Codex OAuth (browser-based PKCE) ───────────────────────


class CodexOAuthAuth(AuthProvider):
    """OpenAI Codex OAuth using browser-based PKCE flow.

    Opens a browser for ChatGPT sign-in, receives callback on localhost,
    and exchanges the auth code for tokens. Tokens auto-refresh when
    expiring within 5 minutes.

    Requires a ChatGPT Plus/Pro/Business/Edu/Enterprise subscription.
    """

    # Public client ID from Codex CLI
    CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
    AUTH_URL = "https://auth.openai.com/oauth/authorize"
    TOKEN_URL = "https://auth.openai.com/oauth/token"
    REDIRECT_PORT = 1455

    def __init__(
        self,
        access_token: str = "",
        refresh_token: str = "",
        expires_at: float = 0.0,
    ) -> None:
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._expires_at = expires_at
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy-initialize the asyncio.Lock to avoid cross-event-loop issues."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_token(self) -> TokenResult:
        """Get a valid token, refreshing via refresh_token if expired."""
        async with self._get_lock():
            token = TokenResult(
                access_token=self._access_token,
                expires_at=self._expires_at,
            )
            if not token.is_expired and self._access_token:
                return token

            if self._refresh_token:
                return await self._refresh()

            # No tokens at all — need interactive login
            return await self._interactive_login()

    async def _refresh(self) -> TokenResult:
        """Refresh the access token using the refresh token."""
        import httpx

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                self.TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self.CLIENT_ID,
                    "refresh_token": self._refresh_token,
                },
            )
            resp.raise_for_status()
            body = resp.json()

        self._access_token = body["access_token"]
        if "refresh_token" in body:
            self._refresh_token = body["refresh_token"]
        expires_in = body.get("expires_in", 3600)
        self._expires_at = time.time() + expires_in

        logger.info(f"Codex OAuth token refreshed, expires in {expires_in}s")
        return TokenResult(
            access_token=self._access_token,
            expires_at=self._expires_at,
        )

    async def _interactive_login(self) -> TokenResult:
        """Run browser-based OAuth PKCE flow."""
        import webbrowser
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading

        # Generate PKCE verifier and challenge
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).rstrip(b"=").decode()
        state = secrets.token_urlsafe(32)

        auth_code: str | None = None
        error_msg: str | None = None

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                nonlocal auth_code, error_msg
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)

                if params.get("error"):
                    error_msg = params["error"][0]
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"<html><body><h1>Authentication failed.</h1>"
                                    b"<p>You can close this tab.</p></body></html>")
                    return

                received_state = params.get("state", [""])[0]
                if received_state != state:
                    error_msg = "State mismatch"
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"State mismatch")
                    return

                auth_code = params.get("code", [""])[0]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Login successful!</h1>"
                                b"<p>You can close this tab and return to AutoForge.</p>"
                                b"</body></html>")

            def log_message(self, format: str, *args: Any) -> None:
                pass  # Suppress server logs

        # Start callback server
        try:
            server = HTTPServer(("127.0.0.1", self.REDIRECT_PORT), CallbackHandler)
        except OSError as e:
            raise RuntimeError(
                f"OAuth callback port {self.REDIRECT_PORT} is in use. "
                "Close the other application or try again."
            ) from e

        received: dict[str, bool] = {}

        def _serve_until_callback() -> None:
            server.timeout = 1.0
            deadline = time.time() + 120
            while time.time() < deadline and not received:
                server.handle_request()
                if auth_code is not None or error_msg is not None:
                    received["done"] = True

        server_thread = threading.Thread(target=_serve_until_callback, daemon=True)
        server_thread.start()

        # Build auth URL
        params = {
            "response_type": "code",
            "client_id": self.CLIENT_ID,
            "redirect_uri": f"http://localhost:{self.REDIRECT_PORT}/auth/callback",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "scope": "openid profile email offline_access",
        }
        url = f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"

        logger.info("Opening browser for Codex OAuth login...")
        webbrowser.open(url)

        # Wait for callback (up to 120 seconds)
        server_thread.join(timeout=120)
        server.server_close()

        if error_msg:
            raise RuntimeError(f"Codex OAuth failed: {error_msg}")
        if not auth_code:
            raise RuntimeError("Codex OAuth timed out waiting for callback")

        # Exchange code for tokens
        import httpx

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                self.TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": self.CLIENT_ID,
                    "code": auth_code,
                    "redirect_uri": f"http://localhost:{self.REDIRECT_PORT}/auth/callback",
                    "code_verifier": code_verifier,
                },
            )
            resp.raise_for_status()
            body = resp.json()

        self._access_token = body["access_token"]
        self._refresh_token = body.get("refresh_token", "")
        expires_in = body.get("expires_in", 3600)
        self._expires_at = time.time() + expires_in

        logger.info(f"Codex OAuth login successful, token expires in {expires_in}s")
        return TokenResult(
            access_token=self._access_token,
            expires_at=self._expires_at,
        )

    def get_client_kwargs(self) -> dict[str, Any]:
        """Return kwargs for AsyncOpenAI constructor."""
        # The token will be injected via _ensure_fresh_token
        return {}


# ── OpenAI Device Code flow (headless) ────────────────────────────


class DeviceCodeAuth(AuthProvider):
    """OpenAI Device Code flow for headless/SSH environments.

    Posts to device endpoint, user enters code in browser on another device,
    CLI polls for completion. Good for remote/headless environments.
    """

    CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
    DEVICE_URL = "https://auth.openai.com/codex/device"
    TOKEN_URL = "https://auth.openai.com/oauth/token"

    def __init__(
        self,
        access_token: str = "",
        refresh_token: str = "",
        expires_at: float = 0.0,
    ) -> None:
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._expires_at = expires_at
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy-initialize the asyncio.Lock to avoid cross-event-loop issues."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_token(self) -> TokenResult:
        """Get a valid token, refreshing or doing device flow if needed."""
        async with self._get_lock():
            token = TokenResult(
                access_token=self._access_token,
                expires_at=self._expires_at,
            )
            if not token.is_expired and self._access_token:
                return token

            if self._refresh_token:
                return await self._refresh()

            return await self._device_flow()

    async def _refresh(self) -> TokenResult:
        """Refresh using refresh_token."""
        import httpx

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                self.TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self.CLIENT_ID,
                    "refresh_token": self._refresh_token,
                },
            )
            resp.raise_for_status()
            body = resp.json()

        self._access_token = body["access_token"]
        if "refresh_token" in body:
            self._refresh_token = body["refresh_token"]
        expires_in = body.get("expires_in", 3600)
        self._expires_at = time.time() + expires_in

        logger.info(f"Device code token refreshed, expires in {expires_in}s")
        return TokenResult(
            access_token=self._access_token,
            expires_at=self._expires_at,
        )

    async def _device_flow(self) -> TokenResult:
        """Run the device code authorization flow."""
        import httpx

        async with httpx.AsyncClient(timeout=15) as client:
            # Step 1: Request device code
            resp = await client.post(
                self.DEVICE_URL,
                data={"client_id": self.CLIENT_ID},
            )
            resp.raise_for_status()
            body = resp.json()

        device_code = body["device_code"]
        user_code = body["user_code"]
        verification_uri = body.get("verification_uri", body.get("verification_url", ""))
        interval = body.get("interval", 5)
        expires_in = body.get("expires_in", 900)

        # Display instructions to user
        from rich.console import Console
        console = Console()
        console.print()
        console.print(f"[bold cyan]Device Code Login[/bold cyan]")
        console.print(f"Open this URL: [bold]{verification_uri}[/bold]")
        console.print(f"Enter code:    [bold yellow]{user_code}[/bold yellow]")
        console.print(f"[dim]Waiting up to {expires_in // 60} minutes...[/dim]")

        # Step 2: Poll for completion
        deadline = time.time() + expires_in
        async with httpx.AsyncClient(timeout=15) as client:
            while time.time() < deadline:
                await asyncio.sleep(interval)
                resp = await client.post(
                    self.TOKEN_URL,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "client_id": self.CLIENT_ID,
                        "device_code": device_code,
                    },
                )

                if resp.status_code == 200:
                    token_body = resp.json()
                    self._access_token = token_body["access_token"]
                    self._refresh_token = token_body.get("refresh_token", "")
                    tok_expires = token_body.get("expires_in", 3600)
                    self._expires_at = time.time() + tok_expires
                    logger.info("Device code login successful")
                    console.print("[green]Login successful![/green]")
                    return TokenResult(
                        access_token=self._access_token,
                        expires_at=self._expires_at,
                    )

                try:
                    error_body = resp.json()
                except (ValueError, Exception):
                    raise RuntimeError(
                        f"Device code poll failed with HTTP {resp.status_code} "
                        f"and non-JSON response: {resp.text[:200]}"
                    )
                error = error_body.get("error", "")
                if error == "authorization_pending":
                    continue
                elif error == "slow_down":
                    interval += 5
                    continue
                elif error == "expired_token":
                    raise RuntimeError("Device code expired. Please try again.")
                elif error == "access_denied":
                    raise RuntimeError("Login was denied by the user.")
                else:
                    raise RuntimeError(f"Device code error: {error}")

        raise RuntimeError("Device code login timed out.")

    def get_client_kwargs(self) -> dict[str, Any]:
        """Return kwargs for AsyncOpenAI constructor."""
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

    # Amazon Bedrock
    if auth_method == "bedrock":
        return AWSBedrockAuth(
            aws_region=config.get("aws_region", ""),
            aws_profile=config.get("aws_profile", ""),
            aws_access_key_id=config.get("aws_access_key_id", ""),
            aws_secret_access_key=config.get("aws_secret_access_key", ""),
            aws_session_token=config.get("aws_session_token", ""),
        )

    # Google Vertex AI (Claude on Vertex)
    if auth_method == "vertex_ai":
        return VertexAIAuth(
            project_id=config.get("project_id", ""),
            region=config.get("region", ""),
        )

    # Codex OAuth (browser PKCE)
    if auth_method == "codex_oauth":
        return CodexOAuthAuth(
            access_token=config.get("access_token", ""),
            refresh_token=config.get("refresh_token", ""),
        )

    # Device Code flow
    if auth_method == "device_code":
        return DeviceCodeAuth(
            access_token=config.get("access_token", ""),
            refresh_token=config.get("refresh_token", ""),
        )

    # Unknown method — fallback with warning
    if not api_key:
        raise ValueError(
            f"Unknown auth_method '{auth_method}' for provider '{provider}' "
            f"and no API key configured. Set an API key or fix the auth_method."
        )
    logger.warning(f"Unknown auth_method '{auth_method}' for {provider}, using api_key")
    return ApiKeyAuth(api_key=api_key)
