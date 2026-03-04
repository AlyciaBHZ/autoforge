"""First-run setup wizard for AutoForge."""

from __future__ import annotations

import sys
from pathlib import Path

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
        "validate": lambda x: x.startswith("sk-ant-") and len(x) > 20,
        "invalid_msg": "Must start with 'sk-ant-' and be at least 20 characters",
        "models_strong": [
            {"name": "Claude Opus 4.6 (best quality, higher cost)", "value": "claude-opus-4-6"},
            {"name": "Claude Sonnet 4.5 (good balance)", "value": "claude-sonnet-4-5-20250929"},
        ],
        "models_fast": [
            {"name": "Claude Sonnet 4.5 (recommended)", "value": "claude-sonnet-4-5-20250929"},
            {"name": "Claude Haiku 4.5 (faster, cheaper)", "value": "claude-haiku-4-5-20251001"},
        ],
    },
    "openai": {
        "name": "OpenAI (GPT / o-series)",
        "key_hint": "Get your key at https://platform.openai.com/api-keys",
        "validate": lambda x: x.startswith("sk-") and len(x) > 20,
        "invalid_msg": "Must start with 'sk-' and be at least 20 characters",
        "models_strong": [
            {"name": "o3 (strongest reasoning)", "value": "o3"},
            {"name": "GPT-4o (fast + capable)", "value": "gpt-4o"},
        ],
        "models_fast": [
            {"name": "GPT-4o-mini (recommended)", "value": "gpt-4o-mini"},
            {"name": "o4-mini (reasoning, cheaper)", "value": "o4-mini"},
        ],
    },
    "google": {
        "name": "Google (Gemini)",
        "key_hint": "Get your key at https://aistudio.google.com/apikey",
        "validate": lambda x: len(x) > 10,
        "invalid_msg": "API key seems too short",
        "models_strong": [
            {"name": "Gemini 2.5 Pro (best quality)", "value": "gemini-2.5-pro"},
            {"name": "Gemini 2.5 Flash (good balance)", "value": "gemini-2.5-flash"},
        ],
        "models_fast": [
            {"name": "Gemini 2.5 Flash (recommended)", "value": "gemini-2.5-flash"},
            {"name": "Gemini 2.0 Flash (fastest, cheapest)", "value": "gemini-2.0-flash"},
        ],
    },
}


def needs_setup() -> bool:
    """Check if first-run setup is needed."""
    if not CONFIG_FILE.exists():
        return True
    content = CONFIG_FILE.read_text(encoding="utf-8")
    # Need at least one API key configured
    has_key = any(
        key_name in content and f'{key_name} = ""' not in content
        for key_name in ("anthropic_key", "openai_key", "google_key")
    )
    return not has_key


def run_setup_wizard() -> None:
    """Run the interactive setup wizard."""
    from InquirerPy import inquirer

    console.print()
    console.print("[bold cyan]AutoForge Setup Wizard[/bold cyan]")
    console.print("Let's configure your environment.\n")

    # Step 1: Select providers
    selected_providers = inquirer.checkbox(
        message="Which LLM providers do you want to use?",
        choices=[
            {"name": info["name"], "value": pid, "enabled": pid == "anthropic"}
            for pid, info in PROVIDERS.items()
        ],
        validate=lambda x: len(x) > 0,
        invalid_message="Select at least one provider",
    ).execute()

    if not selected_providers:
        console.print("[red]Setup cancelled.[/red]")
        return

    # Step 2: API keys for each provider
    api_keys: dict[str, str] = {}
    for pid in selected_providers:
        info = PROVIDERS[pid]
        key = inquirer.secret(
            message=f"{info['name']} API key:",
            validate=info["validate"],
            invalid_message=info["invalid_msg"],
            long_instruction=info["key_hint"],
        ).execute()
        if key:
            api_keys[pid] = key

    if not api_keys:
        console.print("[red]No API keys provided. Setup cancelled.[/red]")
        return

    # Step 3: Model selection — choices from selected providers only
    strong_choices = []
    fast_choices = []
    for pid in selected_providers:
        if pid in api_keys:
            info = PROVIDERS[pid]
            strong_choices.extend(info["models_strong"])
            fast_choices.extend(info["models_fast"])

    model_strong = inquirer.select(
        message="Model for complex tasks (Director, Architect):",
        choices=strong_choices,
        default=strong_choices[0]["value"] if strong_choices else None,
    ).execute()

    model_fast = inquirer.select(
        message="Model for routine tasks (Builder, Reviewer, Tester):",
        choices=fast_choices,
        default=fast_choices[0]["value"] if fast_choices else None,
    ).execute()

    # Step 4: Budget
    budget = inquirer.number(
        message="Default budget limit (USD):",
        default=10.0,
        float_allowed=True,
        min_allowed=1.0,
        max_allowed=100.0,
    ).execute()

    # Step 5: Max agents
    max_agents = inquirer.number(
        message="Max parallel builder agents:",
        default=3,
        min_allowed=1,
        max_allowed=8,
    ).execute()

    # Step 6: Docker
    docker_enabled = inquirer.confirm(
        message="Enable Docker sandbox for isolated builds?",
        default=False,
    ).execute()

    # Write config
    _write_config(api_keys, model_strong, model_fast, float(budget), int(max_agents), docker_enabled)

    console.print()
    console.print(f"[green]Configuration saved to {CONFIG_FILE}[/green]")
    console.print("You can re-run setup anytime with: [bold]autoforge setup[/bold]")
    console.print()


def _write_config(
    api_keys: dict[str, str],
    model_strong: str,
    model_fast: str,
    budget: float,
    max_agents: int,
    docker_enabled: bool,
) -> None:
    """Write configuration to ~/.autoforge/config.toml."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import tomli_w

        api_section: dict[str, str] = {}
        if "anthropic" in api_keys:
            api_section["anthropic_key"] = api_keys["anthropic"]
        if "openai" in api_keys:
            api_section["openai_key"] = api_keys["openai"]
        if "google" in api_keys:
            api_section["google_key"] = api_keys["google"]

        data = {
            "api": api_section,
            "models": {"strong": model_strong, "fast": model_fast},
            "defaults": {
                "budget": budget,
                "max_agents": max_agents,
                "docker": docker_enabled,
                "mode": "developer",
            },
        }
        CONFIG_FILE.write_bytes(tomli_w.dumps(data).encode("utf-8"))
    except ImportError:
        # Fallback: write TOML manually
        lines = ["[api]"]
        if "anthropic" in api_keys:
            lines.append(f'anthropic_key = "{api_keys["anthropic"]}"')
        if "openai" in api_keys:
            lines.append(f'openai_key = "{api_keys["openai"]}"')
        if "google" in api_keys:
            lines.append(f'google_key = "{api_keys["google"]}"')

        lines.append("")
        lines.append("[models]")
        lines.append(f'strong = "{model_strong}"')
        lines.append(f'fast = "{model_fast}"')
        lines.append("")
        lines.append("[defaults]")
        lines.append(f"budget = {budget}")
        lines.append(f"max_agents = {max_agents}")
        lines.append(f'docker = {"true" if docker_enabled else "false"}')
        lines.append('mode = "developer"')
        lines.append("")

        CONFIG_FILE.write_text("\n".join(lines), encoding="utf-8")


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
    except (ImportError, Exception):
        # Manual parse fallback for simple TOML
        return _parse_toml_simple(CONFIG_FILE)

    result = {}
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
        result["budget_limit_usd"] = float(defaults["budget"])
    if "max_agents" in defaults:
        result["max_agents"] = int(defaults["max_agents"])
    if "docker" in defaults:
        result["docker_enabled"] = bool(defaults["docker"])
    if "mode" in defaults:
        result["mode"] = defaults["mode"]

    return result


def _parse_toml_simple(path: Path) -> dict:
    """Simple TOML parser fallback for basic key=value pairs."""
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"')
            # Map TOML keys to config fields
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
            }
            config_key = key_map.get(key)
            if config_key:
                if config_key in ("budget_limit_usd",):
                    result[config_key] = float(value)
                elif config_key in ("max_agents",):
                    result[config_key] = int(value)
                elif config_key in ("docker_enabled",):
                    result[config_key] = value.lower() in ("true", "1", "yes")
                else:
                    result[config_key] = value
    return result
