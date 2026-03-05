"""First-run setup wizard for AutoForge."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# Global config directory
CONFIG_DIR = Path.home() / ".autoforge"
CONFIG_FILE = CONFIG_DIR / "config.toml"

# Provider metadata
PROVIDERS = {
    "anthropic": {
        "name": "Anthropic (Claude)",
        "key_hint": "Get your key at https://console.anthropic.com/settings/keys",
        "validate": lambda x: len(x.strip()) > 20,
        "invalid_msg": "API key must be at least 20 characters",
        "models_strong": [
            {"name": "Claude Opus 4.6 (best quality, higher cost)", "value": "claude-opus-4-6"},
            {"name": "Claude Sonnet 4.6 (good balance)", "value": "claude-sonnet-4-6"},
        ],
        "models_fast": [
            {"name": "Claude Sonnet 4.6 (recommended)", "value": "claude-sonnet-4-6"},
            {"name": "Claude Haiku 4.5 (faster, cheaper)", "value": "claude-haiku-4-5-20251001"},
        ],
        "auth_methods": [
            {"name": "API Key", "value": "api_key"},
            {"name": "Bearer Token + Custom URL (proxy)", "value": "oauth_bearer"},
            {"name": "OAuth2 Client Credentials", "value": "oauth2_client_credentials"},
            {"name": "Amazon Bedrock (AWS credentials)", "value": "bedrock"},
            {"name": "Google Vertex AI (ADC)", "value": "vertex_ai"},
        ],
    },
    "openai": {
        "name": "OpenAI (Codex / GPT / o-series)",
        "key_hint": "Get your key at https://platform.openai.com/api-keys",
        "validate": lambda x: x.startswith("sk-") and len(x) > 20,
        "invalid_msg": "Must start with 'sk-' and be at least 20 characters",
        "models_strong": [
            {"name": "GPT-5.3-Codex (latest code-specialized)", "value": "gpt-5.3-codex"},
            {"name": "GPT-5.2-Codex (stable code model)", "value": "gpt-5.2-codex"},
            {"name": "GPT-5.2 (latest general model)", "value": "gpt-5.2"},
            {"name": "Codex 5.3 (alias)", "value": "codex-5.3"},
            {"name": "o3 (strongest reasoning)", "value": "o3"},
            {"name": "GPT-4o (legacy fallback)", "value": "gpt-4o"},
        ],
        "models_fast": [
            {"name": "GPT-5 mini (recommended)", "value": "gpt-5-mini"},
            {"name": "GPT-5.1-Codex-mini (lightweight coding)", "value": "gpt-5.1-codex-mini"},
            {"name": "GPT-4o-mini (legacy fallback)", "value": "gpt-4o-mini"},
            {"name": "o4-mini (reasoning, cheaper)", "value": "o4-mini"},
        ],
        "auth_methods": [
            {"name": "API Key", "value": "api_key"},
            {"name": "Bearer Token + Custom URL (Azure, LiteLLM)", "value": "oauth_bearer"},
            {"name": "Codex OAuth (browser login, uses subscription)", "value": "codex_oauth"},
            {"name": "Device Code (headless/SSH)", "value": "device_code"},
            {"name": "OAuth2 Client Credentials", "value": "oauth2_client_credentials"},
        ],
    },
    "google": {
        "name": "Google (Gemini)",
        "key_hint": "Get your key at https://aistudio.google.com/apikey",
        "validate": lambda x: len(x) > 10,
        "invalid_msg": "API key seems too short",
        "models_strong": [
            {"name": "Gemini 2.5 Pro (best quality)", "value": "gemini-2.5-pro"},
            {"name": "Gemini 3 Pro Preview (latest preview)", "value": "gemini-3-pro-preview"},
            {"name": "Gemini 2.5 Flash (good balance)", "value": "gemini-2.5-flash"},
        ],
        "models_fast": [
            {"name": "Gemini 2.5 Flash (recommended)", "value": "gemini-2.5-flash"},
            {"name": "Gemini 2.5 Flash-Lite (fastest, cheapest stable)", "value": "gemini-2.5-flash-lite"},
            {"name": "Gemini 3 Flash Preview (latest preview)", "value": "gemini-3-flash-preview"},
        ],
        "auth_methods": [
            {"name": "API Key (recommended)", "value": "api_key"},
            {"name": "Application Default Credentials (ADC)", "value": "adc"},
            {"name": "Service Account JSON", "value": "service_account"},
        ],
    },
}

OPENAI_REASONING_LEVELS = [
    {"name": "medium (balanced, recommended)", "value": "medium"},
    {"name": "low (faster)", "value": "low"},
    {"name": "high (deeper reasoning)", "value": "high"},
    {"name": "minimal (very fast, shallow)", "value": "minimal"},
    {"name": "xhigh (max depth, slower)", "value": "xhigh"},
    {"name": "none (disable reasoning)", "value": "none"},
]


def needs_setup() -> bool:
    """Check if first-run setup is needed."""
    if not CONFIG_FILE.exists():
        return True

    content = CONFIG_FILE.read_text(encoding="utf-8")

    # Try proper TOML parsing first
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-redef]

        data = tomllib.loads(content)

        # Check for non-empty API keys
        api = data.get("api", {})
        has_key = any(
            bool(api.get(key_name))
            for key_name in ("anthropic_key", "openai_key", "google_key")
        )
        # Check for any auth config section
        has_auth = bool(data.get("auth"))
        return not (has_key or has_auth)
    except Exception:
        pass

    # Fallback: raw string matching if TOML parsing fails
    has_key = any(
        key_name in content and f'{key_name} = ""' not in content
        for key_name in ("anthropic_key", "openai_key", "google_key")
    )
    has_auth = "[auth." in content
    return not (has_key or has_auth)


def run_setup_wizard() -> None:
    """Run the interactive setup wizard.

    All steps are optional — users can skip everything and configure later.
    The wizard always completes with a summary of what AutoForge can do.
    """
    from InquirerPy import inquirer

    console.print()
    console.print("[bold cyan]Welcome to AutoForge[/bold cyan]")
    console.print(
        "AI-powered multi-agent platform that turns ideas into working projects.\n"
    )

    # Show what AutoForge can do
    console.print("[bold]What AutoForge can do:[/bold]")
    console.print("  [green]1.[/green] Generate complete projects from a description")
    console.print("  [green]2.[/green] Review and improve existing codebases")
    console.print("  [green]3.[/green] Build web apps, APIs, CLI tools, mobile scaffolds")
    console.print("  [green]4.[/green] Conduct research and analysis tasks")
    console.print("  [green]5.[/green] Run as a 24/7 daemon with Telegram/webhook input")
    console.print()
    console.print("[dim]All settings below are optional. Press Ctrl+C anytime to skip.[/dim]")
    console.print()

    api_keys: dict[str, str] = {}
    auth_configs: dict[str, dict[str, str]] = {}
    model_strong = "claude-sonnet-4-6"
    model_fast = "claude-haiku-4-5-20251001"
    openai_reasoning_effort = "medium"
    budget = 10.0
    max_agents = 3
    docker_enabled = False
    github_config: dict[str, Any] = {}

    try:
        # Step 1: Select providers (optional)
        configure_llm = inquirer.confirm(
            message="Configure an LLM provider now?",
            default=True,
        ).execute()

        if configure_llm:
            selected_providers = inquirer.checkbox(
                message="Which LLM providers do you want to use?",
                choices=[
                    {"name": info["name"], "value": pid, "enabled": False}
                    for pid, info in PROVIDERS.items()
                ],
                validate=lambda x: len(x) > 0,
                invalid_message="Select at least one provider",
            ).execute()

            if selected_providers:
                # Step 2: Auth method + credentials per provider
                for pid in selected_providers:
                    _collect_provider_credentials(inquirer, pid, api_keys, auth_configs)

                # Step 3: Model selection
                strong_choices = []
                fast_choices = []
                for pid in selected_providers:
                    if pid in api_keys or pid in auth_configs:
                        info = PROVIDERS[pid]
                        strong_choices.extend(info["models_strong"])
                        fast_choices.extend(info["models_fast"])

                if strong_choices:
                    model_strong = inquirer.select(
                        message="Model for complex tasks (Director, Architect):",
                        choices=strong_choices,
                        default=strong_choices[0]["value"],
                    ).execute()

                if fast_choices:
                    model_fast = inquirer.select(
                        message="Model for routine tasks (Builder, Reviewer, Tester):",
                        choices=fast_choices,
                        default=fast_choices[0]["value"],
                    ).execute()

                # Validate model → provider mapping
                _validate_model_provider(inquirer, api_keys, auth_configs, model_strong, "strong")
                _validate_model_provider(inquirer, api_keys, auth_configs, model_fast, "fast")

                if "openai" in auth_configs or "openai" in api_keys:
                    openai_reasoning_effort = inquirer.select(
                        message="OpenAI thinking level (reasoning effort):",
                        choices=OPENAI_REASONING_LEVELS,
                        default="medium",
                    ).execute()

        # Step 4: Budget
        if _should_skip_budget_prompt(model_strong, model_fast, auth_configs):
            budget = 10.0
            console.print(
                "[dim]OpenAI subscription mode detected for selected models "
                "- skipping USD budget prompt.[/dim]"
            )
        else:
            budget = float(inquirer.number(
                message="Default budget limit (USD):",
                default=10.0,
                float_allowed=True,
                min_allowed=1.0,
                max_allowed=100.0,
            ).execute())

        # Step 5: Max agents
        max_agents = int(inquirer.number(
            message="Max parallel builder agents:",
            default=3,
            min_allowed=1,
            max_allowed=8,
        ).execute())

        # Step 6: Docker
        docker_enabled = inquirer.confirm(
            message="Enable Docker sandbox for isolated builds?",
            default=False,
        ).execute()

        # Step 7: GitHub environment
        github_config = _setup_github(inquirer)

    except KeyboardInterrupt:
        console.print("\n[yellow]Setup interrupted — saving what we have.[/yellow]")

    # Always write config (even with defaults / no API keys)
    _write_config(
        api_keys, model_strong, model_fast,
        budget, max_agents, docker_enabled, openai_reasoning_effort,
        auth_configs=auth_configs,
        github_config=github_config,
    )

    console.print()
    console.print(f"[green]Configuration saved to {CONFIG_FILE}[/green]")

    if not api_keys and not auth_configs:
        console.print()
        console.print("[yellow]No API keys configured yet.[/yellow]")
        console.print("Add one anytime with: [bold]autoforgeai setup[/bold]")
        console.print("Or set environment variables: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY")
    else:
        console.print()
        console.print("[bold green]Ready to go![/bold green]")
        console.print('Try: [bold]autoforgeai[/bold] to start an interactive session')

    console.print()
    console.print("[dim]Reconfigure anytime: autoforgeai setup[/dim]")
    console.print()


def _default_auth_method(provider_id: str) -> str:
    """Return the default auth method for a provider."""
    # Most OpenAI users run with ChatGPT subscription auth, not OAuth2 client_credentials.
    if provider_id == "openai":
        return "codex_oauth"
    return "api_key"


def _is_openai_subscription_auth_method(auth_method: str | None) -> bool:
    method = (auth_method or "").strip().lower()
    return method in ("codex_oauth", "device_code")


def _should_skip_budget_prompt(
    model_strong: str,
    model_fast: str,
    auth_configs: dict[str, dict[str, str]],
) -> bool:
    """Skip USD budget prompt when both selected models run on OpenAI subscription auth."""
    from autoforge.engine.llm_router import detect_provider

    if detect_provider(model_strong) != "openai":
        return False
    if detect_provider(model_fast) != "openai":
        return False
    return _is_openai_subscription_auth_method(
        auth_configs.get("openai", {}).get("auth_method")
    )


def _maybe_run_openai_subscription_login(
    inquirer: Any,
    provider_id: str,
    auth_method: str,
    auth_configs: dict[str, dict[str, str]],
) -> None:
    """Run Codex OAuth / Device Code login during setup."""
    if provider_id != "openai" or auth_method not in ("codex_oauth", "device_code"):
        return

    if auth_method == "codex_oauth":
        console.print("[dim]Starting Codex OAuth login (browser)...[/dim]")
    else:
        run_now = inquirer.confirm(message="Run Device Code login now?", default=True).execute()
        if not run_now:
            return

    try:
        import asyncio
        from autoforge.engine.auth import CodexOAuthAuth, DeviceCodeAuth

        provider = CodexOAuthAuth() if auth_method == "codex_oauth" else DeviceCodeAuth()
        asyncio.run(provider.get_token())

        # Persist tokens to reduce repeated login prompts.
        access_token = getattr(provider, "_access_token", "")
        refresh_token = getattr(provider, "_refresh_token", "")
        expires_at = getattr(provider, "_expires_at", 0.0)
        if access_token:
            auth_configs[provider_id]["access_token"] = str(access_token)
        if refresh_token:
            auth_configs[provider_id]["refresh_token"] = str(refresh_token)
        if expires_at:
            auth_configs[provider_id]["expires_at"] = str(expires_at)

        console.print("[green]OpenAI subscription login successful.[/green]")
    except Exception as exc:
        console.print(f"[yellow]OpenAI subscription login not completed:[/yellow] {exc}")
        console.print("[dim]You can retry anytime with: autoforgeai setup[/dim]")


def _collect_provider_credentials(
    inquirer: Any,
    pid: str,
    api_keys: dict[str, str],
    auth_configs: dict[str, dict[str, str]],
) -> None:
    """Collect credentials for a single provider."""
    info = PROVIDERS[pid]
    auth_methods = info["auth_methods"]

    # Ask auth method (skip if only one option)
    if len(auth_methods) > 1:
        default_auth = _default_auth_method(pid)
        auth_method = inquirer.select(
            message=f"{info['name']} — authentication method:",
            choices=auth_methods,
            default=default_auth,
        ).execute()
    else:
        auth_method = _default_auth_method(pid)

    if pid == "openai" and auth_method == "oauth2_client_credentials":
        switch_to_codex = inquirer.confirm(
            message=(
                "OAuth2 Client Credentials requires your own enterprise token endpoint. "
                "Switch to Codex OAuth browser login instead?"
            ),
            default=True,
        ).execute()
        if switch_to_codex:
            auth_method = "codex_oauth"

    if auth_method == "api_key":
        key = inquirer.secret(
            message=f"{info['name']} API key:",
            validate=info["validate"],
            invalid_message=info["invalid_msg"],
            long_instruction=info["key_hint"],
        ).execute()
        if key:
            api_keys[pid] = key

    elif auth_method == "oauth_bearer":
        base_url = inquirer.text(
            message="Base URL (e.g. https://your-proxy.com/v1):",
            validate=lambda x: x.startswith("http"),
            invalid_message="Must be a valid URL starting with http",
        ).execute()
        bearer = inquirer.secret(
            message="Bearer token:",
            validate=lambda x: len(x) > 5,
            invalid_message="Token seems too short",
        ).execute()
        auth_configs[pid] = {
            "auth_method": "oauth_bearer",
            "base_url": base_url,
            "bearer_token": bearer,
        }
        api_keys[pid] = bearer

    elif auth_method == "oauth2_client_credentials":
        token_url = inquirer.text(
            message="Token URL:",
            validate=lambda x: x.startswith("http"),
            invalid_message="Must be a valid URL",
        ).execute()
        client_id = inquirer.text(message="Client ID:").execute()
        client_secret = inquirer.secret(
            message="Client Secret:",
            validate=lambda x: len(x) > 5,
            invalid_message="Secret seems too short",
        ).execute()
        scope = inquirer.text(
            message="Scope (optional, press Enter to skip):",
            default="",
        ).execute()
        base_url = inquirer.text(
            message="Base URL (optional, for proxy — press Enter to skip):",
            default="",
        ).execute()
        auth_configs[pid] = {
            "auth_method": "oauth2_client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "token_url": token_url,
        }
        if scope:
            auth_configs[pid]["scope"] = scope
        if base_url:
            auth_configs[pid]["base_url"] = base_url

    elif auth_method == "adc":
        auth_configs[pid] = {"auth_method": "adc"}
        console.print(
            "[dim]Ensure GOOGLE_APPLICATION_CREDENTIALS is set or "
            "run 'gcloud auth application-default login'.[/dim]"
        )

    elif auth_method == "service_account":
        sa_path = inquirer.filepath(
            message="Path to service account JSON:",
            validate=lambda x: Path(x).is_file() and x.endswith(".json"),
            invalid_message="Must be a valid .json file",
        ).execute()
        auth_configs[pid] = {
            "auth_method": "service_account",
            "service_account_path": sa_path,
        }

    elif auth_method == "bedrock":
        console.print("[dim]Amazon Bedrock — Claude via AWS[/dim]")
        aws_region = inquirer.text(
            message="AWS Region:",
            default="us-east-1",
        ).execute()
        auth_type = inquirer.select(
            message="AWS authentication:",
            choices=[
                {"name": "AWS Profile (SSO/IAM Identity Center)", "value": "profile"},
                {"name": "Access Keys (static)", "value": "keys"},
                {"name": "Instance Role (automatic on EC2/ECS/Lambda)", "value": "instance"},
            ],
        ).execute()
        auth_configs[pid] = {
            "auth_method": "bedrock",
            "aws_region": aws_region,
        }
        if auth_type == "profile":
            profile = inquirer.text(
                message="AWS Profile name:",
                default="default",
            ).execute()
            auth_configs[pid]["aws_profile"] = profile
            console.print(
                f"[dim]Run 'aws sso login --profile {profile}' if using SSO.[/dim]"
            )
        elif auth_type == "keys":
            access_key = inquirer.secret(
                message="AWS Access Key ID:",
                validate=lambda x: len(x) >= 16,
                invalid_message="Access key too short",
            ).execute()
            secret_key = inquirer.secret(
                message="AWS Secret Access Key:",
                validate=lambda x: len(x) >= 20,
                invalid_message="Secret key too short",
            ).execute()
            session_token = inquirer.secret(
                message="AWS Session Token (optional, press Enter to skip):",
                default="",
            ).execute()
            auth_configs[pid]["aws_access_key_id"] = access_key
            auth_configs[pid]["aws_secret_access_key"] = secret_key
            if session_token:
                auth_configs[pid]["aws_session_token"] = session_token
        else:
            console.print(
                "[dim]Instance role will be auto-detected on AWS compute.[/dim]"
            )

    elif auth_method == "vertex_ai":
        console.print("[dim]Google Vertex AI — Claude via Google Cloud[/dim]")
        project_id = inquirer.text(
            message="GCP Project ID:",
            validate=lambda x: len(x) > 0,
            invalid_message="Project ID is required",
        ).execute()
        region = inquirer.text(
            message="GCP Region:",
            default="us-east5",
        ).execute()
        auth_configs[pid] = {
            "auth_method": "vertex_ai",
            "project_id": project_id,
            "region": region,
        }
        console.print(
            "[dim]Ensure you've run 'gcloud auth application-default login' "
            "or set GOOGLE_APPLICATION_CREDENTIALS.[/dim]"
        )

    elif auth_method == "codex_oauth":
        console.print(
            "[dim]Codex OAuth — will open browser for ChatGPT login.\n"
            "Requires ChatGPT Plus/Pro/Business/Edu/Enterprise subscription.\n"
            "Uses subscription quota (not API billing).[/dim]"
        )
        auth_configs[pid] = {"auth_method": "codex_oauth"}
        _maybe_run_openai_subscription_login(inquirer, pid, auth_method, auth_configs)

    elif auth_method == "device_code":
        console.print(
            "[dim]Device Code Flow — for headless/SSH environments.\n"
            "Will display a URL and code to enter on another device.\n"
            "Requires ChatGPT Plus/Pro/Business/Edu/Enterprise subscription.[/dim]"
        )
        auth_configs[pid] = {"auth_method": "device_code"}
        _maybe_run_openai_subscription_login(inquirer, pid, auth_method, auth_configs)


def _setup_github(inquirer: Any) -> dict[str, Any]:
    """Detect and optionally configure GitHub environment.

    Checks for git and gh CLI availability, guides the user through
    configuration if desired.

    Returns a dict with github config to persist (may be empty).
    """
    result: dict[str, Any] = {}

    console.print()
    console.print("[bold cyan]GitHub Environment[/bold cyan]")

    # Check git
    git_path = shutil.which("git")
    if git_path:
        try:
            version = subprocess.run(
                ["git", "--version"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            console.print(f"  [green]Git detected:[/green] {version}")
        except (subprocess.SubprocessError, OSError):
            console.print("  [green]Git detected[/green]")
        result["git_available"] = True
    else:
        console.print("  [yellow]Git not found.[/yellow]")
        console.print("  [dim]Install git for version control: https://git-scm.com/downloads[/dim]")
        result["git_available"] = False

    # Check gh CLI
    gh_path = shutil.which("gh")
    if gh_path:
        try:
            auth_status = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True, text=True, timeout=10,
            )
            if auth_status.returncode == 0:
                console.print("  [green]GitHub CLI detected and authenticated[/green]")
                result["gh_authenticated"] = True
            else:
                console.print("  [yellow]GitHub CLI detected but not authenticated[/yellow]")
                login = inquirer.confirm(
                    message="Authenticate GitHub CLI now? (opens browser)",
                    default=True,
                ).execute()
                if login:
                    console.print("  [dim]Running: gh auth login[/dim]")
                    login_result = subprocess.run(
                        ["gh", "auth", "login", "--web"],
                        timeout=120,
                    )
                    result["gh_authenticated"] = login_result.returncode == 0
                    if result["gh_authenticated"]:
                        console.print("  [green]GitHub authentication successful![/green]")
                    else:
                        console.print("  [yellow]Authentication skipped or failed.[/yellow]")
                else:
                    result["gh_authenticated"] = False
                    console.print(
                        "  [dim]Run 'gh auth login' later to enable GitHub integration.[/dim]"
                    )
        except (subprocess.SubprocessError, OSError):
            console.print("  [yellow]GitHub CLI detected but status check failed[/yellow]")
            result["gh_authenticated"] = False
    else:
        console.print("  [dim]GitHub CLI (gh) not found — optional but enables auto-push.[/dim]")
        console.print("  [dim]Install: https://cli.github.com[/dim]")
        result["gh_authenticated"] = False

    # Auto-push preference
    if result.get("git_available"):
        auto_push = inquirer.confirm(
            message="Auto-push generated projects to GitHub?",
            default=result.get("gh_authenticated", False),
        ).execute()
        result["auto_push"] = auto_push
        if auto_push:
            default_org = inquirer.text(
                message="Default GitHub org/user (press Enter to skip):",
                default="",
            ).execute()
            if default_org:
                result["default_org"] = default_org

    return result


def _validate_model_provider(
    inquirer: Any,
    api_keys: dict[str, str],
    auth_configs: dict[str, dict[str, str]],
    model: str,
    label: str,
) -> None:
    """Validate that a selected model has a configured provider."""
    from autoforge.engine.llm_router import detect_provider

    required_provider = detect_provider(model)
    if required_provider in api_keys or required_provider in auth_configs:
        return

    info = PROVIDERS.get(required_provider)
    if not info:
        return

    console.print(
        f"[yellow]Warning:[/yellow] {label} model '{model}' requires "
        f"{info['name']} but no credentials were provided."
    )
    fix = inquirer.confirm(
        message=f"Add credentials for {info['name']}?",
        default=True,
    ).execute()
    if fix:
        auth_methods = info.get("auth_methods", [{"name": "API Key", "value": "api_key"}])

        # Offer full auth method selection (same as initial setup)
        if len(auth_methods) > 1:
            default_auth = _default_auth_method(required_provider)
            auth_method = inquirer.select(
                message=f"{info['name']} — authentication method:",
                choices=auth_methods,
                default=default_auth,
            ).execute()
        else:
            auth_method = _default_auth_method(required_provider)

        if required_provider == "openai" and auth_method == "oauth2_client_credentials":
            switch_to_codex = inquirer.confirm(
                message=(
                    "OAuth2 Client Credentials requires your own enterprise token endpoint. "
                    "Switch to Codex OAuth browser login instead?"
                ),
                default=True,
            ).execute()
            if switch_to_codex:
                auth_method = "codex_oauth"

        if auth_method == "api_key":
            key = inquirer.secret(
                message=f"{info['name']} API key:",
                validate=info["validate"],
                invalid_message=info["invalid_msg"],
                long_instruction=info["key_hint"],
            ).execute()
            if key:
                api_keys[required_provider] = key

        elif auth_method == "oauth_bearer":
            base_url = inquirer.text(
                message="Base URL (e.g. https://your-proxy.com/v1):",
                validate=lambda x: x.startswith("http"),
                invalid_message="Must be a valid URL starting with http",
            ).execute()
            bearer = inquirer.secret(
                message="Bearer token:",
                validate=lambda x: len(x) > 5,
                invalid_message="Token seems too short",
            ).execute()
            auth_configs[required_provider] = {
                "auth_method": "oauth_bearer",
                "base_url": base_url,
                "bearer_token": bearer,
            }
            api_keys[required_provider] = bearer

        elif auth_method == "oauth2_client_credentials":
            token_url = inquirer.text(
                message="Token URL:",
                validate=lambda x: x.startswith("http"),
                invalid_message="Must be a valid URL",
            ).execute()
            client_id = inquirer.text(message="Client ID:").execute()
            client_secret = inquirer.secret(
                message="Client Secret:",
                validate=lambda x: len(x) > 5,
                invalid_message="Secret seems too short",
            ).execute()
            auth_configs[required_provider] = {
                "auth_method": "oauth2_client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "token_url": token_url,
            }

        elif auth_method == "bedrock":
            aws_region = inquirer.text(
                message="AWS Region:", default="us-east-1",
            ).execute()
            auth_configs[required_provider] = {
                "auth_method": "bedrock",
                "aws_region": aws_region,
            }

        elif auth_method == "vertex_ai":
            project_id = inquirer.text(
                message="GCP Project ID:",
                validate=lambda x: len(x) > 0,
                invalid_message="Project ID is required",
            ).execute()
            region = inquirer.text(
                message="GCP Region:", default="us-east5",
            ).execute()
            auth_configs[required_provider] = {
                "auth_method": "vertex_ai",
                "project_id": project_id,
                "region": region,
            }

        elif auth_method in ("adc", "service_account", "codex_oauth", "device_code"):
            auth_configs[required_provider] = {"auth_method": auth_method}
            _maybe_run_openai_subscription_login(
                inquirer,
                required_provider,
                auth_method,
                auth_configs,
            )


def _escape_toml_value(value: str) -> str:
    """Escape a string value for safe inclusion in a TOML quoted string.

    Handles backslashes, double quotes, and control characters that would
    otherwise allow TOML injection via user-supplied input.
    """
    value = value.replace("\\", "\\\\")  # Backslashes first
    value = value.replace('"', '\\"')
    value = value.replace("\n", "\\n")
    value = value.replace("\r", "\\r")
    value = value.replace("\t", "\\t")
    return value


def _write_config(
    api_keys: dict[str, str],
    model_strong: str,
    model_fast: str,
    budget: float,
    max_agents: int,
    docker_enabled: bool,
    openai_reasoning_effort: str = "medium",
    auth_configs: dict[str, dict[str, str]] | None = None,
    github_config: dict[str, Any] | None = None,
) -> None:
    """Write configuration to ~/.autoforge/config.toml."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Set restrictive permissions on config directory (Unix only)
    if sys.platform != "win32":
        os.chmod(CONFIG_DIR, 0o700)

    # Build github section
    gh_section: dict[str, Any] = {}
    if github_config:
        if github_config.get("auto_push"):
            gh_section["auto_push"] = True
        if github_config.get("default_org"):
            gh_section["default_org"] = github_config["default_org"]
        gh_section["git_available"] = github_config.get("git_available", False)
        gh_section["gh_authenticated"] = github_config.get("gh_authenticated", False)

    try:
        import tomli_w

        api_section: dict[str, str] = {}
        if "anthropic" in api_keys:
            api_section["anthropic_key"] = api_keys["anthropic"]
        if "openai" in api_keys:
            api_section["openai_key"] = api_keys["openai"]
        if "google" in api_keys:
            api_section["google_key"] = api_keys["google"]

        data: dict[str, Any] = {
            "api": api_section,
            "models": {"strong": model_strong, "fast": model_fast},
            "defaults": {
                "budget": budget,
                "max_agents": max_agents,
                "docker": docker_enabled,
                "mode": "developer",
                "openai_reasoning_effort": openai_reasoning_effort,
            },
        }

        # Add auth config sections
        if auth_configs:
            auth_section: dict[str, Any] = {}
            for provider, auth_data in auth_configs.items():
                auth_section[provider] = dict(auth_data)
            data["auth"] = auth_section

        # Add github section
        if gh_section:
            data["github"] = gh_section

        CONFIG_FILE.write_bytes(tomli_w.dumps(data).encode("utf-8"))
    except ImportError:
        # Fallback: write TOML manually (with proper escaping to prevent injection)
        lines = ["[api]"]
        if "anthropic" in api_keys:
            lines.append(f'anthropic_key = "{_escape_toml_value(api_keys["anthropic"])}"')
        if "openai" in api_keys:
            lines.append(f'openai_key = "{_escape_toml_value(api_keys["openai"])}"')
        if "google" in api_keys:
            lines.append(f'google_key = "{_escape_toml_value(api_keys["google"])}"')

        lines.append("")
        lines.append("[models]")
        lines.append(f'strong = "{_escape_toml_value(model_strong)}"')
        lines.append(f'fast = "{_escape_toml_value(model_fast)}"')
        lines.append("")
        lines.append("[defaults]")
        lines.append(f"budget = {budget}")
        lines.append(f"max_agents = {max_agents}")
        lines.append(f'docker = {"true" if docker_enabled else "false"}')
        lines.append('mode = "developer"')
        lines.append(f'openai_reasoning_effort = "{_escape_toml_value(openai_reasoning_effort)}"')
        lines.append("")

        # Write auth configs
        if auth_configs:
            for provider, auth_data in auth_configs.items():
                lines.append(f"[auth.{_escape_toml_value(provider)}]")
                for k, v in auth_data.items():
                    if v:
                        lines.append(f'{_escape_toml_value(k)} = "{_escape_toml_value(v)}"')
                lines.append("")

        # Write github config
        if gh_section:
            lines.append("[github]")
            for k, v in gh_section.items():
                if isinstance(v, bool):
                    lines.append(f'{k} = {"true" if v else "false"}')
                elif isinstance(v, str):
                    lines.append(f'{k} = "{_escape_toml_value(v)}"')
            lines.append("")

        CONFIG_FILE.write_text("\n".join(lines), encoding="utf-8")

    # Set restrictive permissions on config file (Unix only).
    if sys.platform != "win32":
        os.chmod(CONFIG_FILE, 0o600)


def load_global_config() -> dict:
    """Load global config from ~/.autoforge/config.toml.

    Returns a flat dict with keys matching ForgeConfig fields.
    """
    if not CONFIG_FILE.exists():
        return {}

    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        data = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (ImportError, ValueError, OSError):
        # Manual parse fallback for simple TOML
        return _parse_toml_simple(CONFIG_FILE)

    result: dict[str, Any] = {}
    api = data.get("api", {})
    if api.get("anthropic_key"):
        result["anthropic_api_key"] = api["anthropic_key"]
    if api.get("openai_key"):
        result["openai_api_key"] = api["openai_key"]
    if api.get("google_key"):
        result["google_api_key"] = api["google_key"]

    models = data.get("models", {})
    if models.get("strong"):
        result["model_strong"] = models["strong"]
    if models.get("fast"):
        result["model_fast"] = models["fast"]

    defaults = data.get("defaults", {})
    if "budget" in defaults:
        try:
            result["budget_limit_usd"] = float(defaults["budget"])
        except (ValueError, TypeError):
            pass  # Skip malformed budget value
    if "max_agents" in defaults:
        try:
            result["max_agents"] = int(defaults["max_agents"])
        except (ValueError, TypeError):
            pass  # Skip malformed max_agents value
    if "docker" in defaults:
        result["docker_enabled"] = bool(defaults["docker"])
    if "mode" in defaults:
        result["mode"] = defaults["mode"]
    if "openai_reasoning_effort" in defaults:
        result["openai_reasoning_effort"] = str(defaults["openai_reasoning_effort"])

    # Load per-provider auth config
    auth = data.get("auth", {})
    for provider in ("anthropic", "openai", "google"):
        auth_section = auth.get(provider, {})
        if auth_section:
            result[f"auth_{provider}"] = dict(auth_section)

    # Load github config
    github = data.get("github", {})
    if github:
        result["github_auto_push"] = github.get("auto_push", False)
        result["github_default_org"] = github.get("default_org", "")
        result["github_git_available"] = github.get("git_available", False)
        result["github_gh_authenticated"] = github.get("gh_authenticated", False)

    return result


def _parse_toml_simple(path: Path) -> dict:
    """Simple TOML parser fallback for basic key=value pairs.

    Handles nested sections like [auth.openai] by tracking the current
    section path and mapping dotted keys to auth_{provider} config keys.
    """
    result: dict[str, Any] = {}
    current_section = ""

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Section header
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            continue

        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"')

        # Auth sections: [auth.openai], [auth.google], etc.
        if current_section.startswith("auth."):
            provider = current_section.split(".", 1)[1]
            auth_key = f"auth_{provider}"
            if auth_key not in result:
                result[auth_key] = {}
            result[auth_key][key] = value
            continue

        # Standard key mapping
        key_map = {
            "anthropic_key": "anthropic_api_key",
            "openai_key": "openai_api_key",
            "google_key": "google_api_key",
            "strong": "model_strong",
            "fast": "model_fast",
            "budget": "budget_limit_usd",
            "max_agents": "max_agents",
            "docker": "docker_enabled",
            "mode": "mode",
            "openai_reasoning_effort": "openai_reasoning_effort",
        }
        config_key = key_map.get(key)
        if config_key:
            if config_key in ("budget_limit_usd",):
                try:
                    result[config_key] = float(value)
                except (ValueError, TypeError):
                    continue  # Skip malformed value
            elif config_key in ("max_agents",):
                try:
                    result[config_key] = int(value)
                except (ValueError, TypeError):
                    continue  # Skip malformed value
            elif config_key in ("docker_enabled",):
                result[config_key] = value.lower() in ("true", "1", "yes")
            else:
                result[config_key] = value
    return result
