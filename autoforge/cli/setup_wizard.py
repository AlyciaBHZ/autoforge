"""First-run setup wizard for AutoForge."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

console = Console()

# Global config directory
CONFIG_DIR = Path.home() / ".autoforge"
CONFIG_FILE = CONFIG_DIR / "config.toml"


def needs_setup() -> bool:
    """Check if first-run setup is needed."""
    if not CONFIG_FILE.exists():
        return True
    content = CONFIG_FILE.read_text(encoding="utf-8")
    return "anthropic_key" not in content or 'anthropic_key = ""' in content


def run_setup_wizard() -> None:
    """Run the interactive setup wizard."""
    from InquirerPy import inquirer

    console.print()
    console.print("[bold cyan]AutoForge Setup Wizard[/bold cyan]")
    console.print("Let's configure your environment.\n")

    # Step 1: API Key
    api_key = inquirer.secret(
        message="Anthropic API key (sk-ant-...):",
        validate=lambda x: x.startswith("sk-ant-") and len(x) > 20,
        invalid_message="Must start with 'sk-ant-' and be at least 20 characters",
        long_instruction="Get your key at https://console.anthropic.com/settings/keys",
    ).execute()

    if not api_key:
        console.print("[red]Setup cancelled.[/red]")
        return

    # Step 2: Model selection
    model_strong = inquirer.select(
        message="Model for complex tasks (Director, Architect):",
        choices=[
            {"name": "Claude Opus 4.6 (best quality, higher cost)", "value": "claude-opus-4-6"},
            {"name": "Claude Sonnet 4.5 (good balance)", "value": "claude-sonnet-4-5-20250929"},
        ],
        default="claude-opus-4-6",
    ).execute()

    model_fast = inquirer.select(
        message="Model for routine tasks (Builder, Reviewer, Tester):",
        choices=[
            {"name": "Claude Sonnet 4.5 (recommended)", "value": "claude-sonnet-4-5-20250929"},
            {"name": "Claude Haiku 4.5 (faster, cheaper)", "value": "claude-haiku-4-5-20251001"},
        ],
        default="claude-sonnet-4-5-20250929",
    ).execute()

    # Step 3: Budget
    budget = inquirer.number(
        message="Default budget limit (USD):",
        default=10.0,
        float_allowed=True,
        min_allowed=1.0,
        max_allowed=100.0,
    ).execute()

    # Step 4: Max agents
    max_agents = inquirer.number(
        message="Max parallel builder agents:",
        default=3,
        min_allowed=1,
        max_allowed=8,
    ).execute()

    # Step 5: Docker
    docker_enabled = inquirer.confirm(
        message="Enable Docker sandbox for isolated builds?",
        default=False,
    ).execute()

    # Write config
    _write_config(api_key, model_strong, model_fast, float(budget), int(max_agents), docker_enabled)

    console.print()
    console.print(f"[green]Configuration saved to {CONFIG_FILE}[/green]")
    console.print("You can re-run setup anytime with: [bold]autoforge setup[/bold]")
    console.print()


def _write_config(
    api_key: str,
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

        data = {
            "api": {"anthropic_key": api_key},
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
        content = f"""[api]
anthropic_key = "{api_key}"

[models]
strong = "{model_strong}"
fast = "{model_fast}"

[defaults]
budget = {budget}
max_agents = {max_agents}
docker = {"true" if docker_enabled else "false"}
mode = "developer"
"""
        CONFIG_FILE.write_text(content, encoding="utf-8")


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
