"""AutoForge configuration.

Config priority chain (highest -> lowest):
    1. CLI arguments
    2. Project .env file (in repo root)
    3. Global ~/.autoforge/config.toml (user-level defaults)
    4. Built-in defaults
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

import autoforge


# Approximate pricing per million tokens (USD)
MODEL_PRICING = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
}

# Default fallback pricing
DEFAULT_PRICING = {"input": 5.0, "output": 25.0}


def _safe_float(key: str, default: float) -> float:
    raw = os.getenv(key, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            f"Invalid value for {key}={raw!r}, using default {default}"
        )
        return default


def _safe_int(key: str, default: int) -> int:
    raw = os.getenv(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            f"Invalid value for {key}={raw!r}, using default {default}"
        )
        return default


@dataclass
class ForgeConfig:
    """Central configuration for an AutoForge run."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    workspace_dir: Path | None = None
    constitution_dir: Path | None = None

    # LLM settings
    anthropic_api_key: str = ""
    model_strong: str = "claude-opus-4-6"
    model_fast: str = "claude-sonnet-4-5-20250929"
    max_tokens_strong: int = 16384
    max_tokens_fast: int = 8192

    # Budget
    budget_limit_usd: float = 10.0

    # Token tracking -- per-model accumulators
    token_usage: dict[str, dict[str, int]] = field(default_factory=dict)

    # Execution
    max_agents: int = 3
    max_retries: int = 3
    quality_threshold: float = 0.7
    verbose: bool = False
    log_level: str = "INFO"

    # Operating mode
    mode: str = "developer"  # "developer" or "research"

    # Mobile target
    mobile_target: str = "none"  # "none", "ios", "android", "both"
    mobile_framework: str = "react-native"  # "react-native" or "flutter"

    # Run identity
    run_id: str = ""

    # Docker
    sandbox_image: str = "autoforge-sandbox:latest"
    docker_enabled: bool = False  # Default off; setup.sh enables if available

    # Daemon mode
    daemon_enabled: bool = False
    daemon_poll_interval: int = 10  # seconds between queue checks
    db_path: Path | None = None  # SQLite database for project registry

    # Telegram bot
    telegram_token: str = ""
    telegram_allowed_users: list[str] = field(default_factory=list)

    # Webhook API
    webhook_enabled: bool = False
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 8420
    webhook_secret: str = ""  # Bearer token for webhook auth

    def __post_init__(self) -> None:
        if self.workspace_dir is None:
            self.workspace_dir = self.project_root / "workspace"
        if self.constitution_dir is None:
            self.constitution_dir = autoforge.DATA_DIR / "constitution"
        if self.db_path is None:
            self.db_path = self.project_root / "autoforge.db"
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:12]

    @classmethod
    def from_env(cls, **overrides) -> ForgeConfig:
        """Create config from environment variables and optional overrides.

        Priority chain:
            1. overrides (CLI args)
            2. .env file (project-level)
            3. ~/.autoforge/config.toml (global user config)
            4. Built-in defaults
        """
        # Layer 3: Load global config (lowest priority)
        global_config = _load_global_config()

        # Layer 2: Load .env (overrides global)
        load_dotenv()

        # Parse allowed Telegram users (comma-separated)
        allowed_raw = os.getenv("FORGE_TELEGRAM_ALLOWED_USERS", "")
        allowed_users = [u.strip() for u in allowed_raw.split(",") if u.strip()]

        # Validate log level
        log_level = (
            os.getenv("FORGE_LOG_LEVEL")
            or global_config.get("log_level", "INFO")
        ).upper()
        if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            logging.getLogger(__name__).warning(
                f"Invalid FORGE_LOG_LEVEL={log_level!r}, using INFO"
            )
            log_level = "INFO"

        budget = _safe_float(
            "FORGE_BUDGET_LIMIT",
            float(global_config.get("budget_limit_usd", 10.0)),
        )
        if budget <= 0:
            budget = 10.0

        max_agents = _safe_int(
            "FORGE_MAX_AGENTS",
            int(global_config.get("max_agents", 3)),
        )
        if max_agents < 1:
            max_agents = 1
        elif max_agents > 50:
            max_agents = 50

        # Start with global config values, then override with .env, then CLI
        config = cls(
            anthropic_api_key=(
                os.getenv("ANTHROPIC_API_KEY")
                or global_config.get("anthropic_api_key", "")
            ),
            model_strong=(
                os.getenv("FORGE_MODEL_STRONG")
                or global_config.get("model_strong", "claude-opus-4-6")
            ),
            model_fast=(
                os.getenv("FORGE_MODEL_FAST")
                or global_config.get("model_fast", "claude-sonnet-4-5-20250929")
            ),
            budget_limit_usd=budget,
            max_agents=max_agents,
            log_level=log_level,
            docker_enabled=(
                os.getenv("FORGE_DOCKER_ENABLED", "").lower() in ("true", "1", "yes")
                or global_config.get("docker_enabled", False)
            ),
            mode=global_config.get("mode", "developer"),
            mobile_target=global_config.get("mobile_target", "none"),
            # Daemon
            daemon_poll_interval=_safe_int("FORGE_DAEMON_POLL_INTERVAL", 10),
            # Telegram
            telegram_token=os.getenv("FORGE_TELEGRAM_TOKEN", ""),
            telegram_allowed_users=allowed_users,
            # Webhook
            webhook_enabled=os.getenv("FORGE_WEBHOOK_ENABLED", "").lower() in ("true", "1", "yes"),
            webhook_host=os.getenv("FORGE_WEBHOOK_HOST", "127.0.0.1"),
            webhook_port=_safe_int("FORGE_WEBHOOK_PORT", 8420),
            webhook_secret=os.getenv("FORGE_WEBHOOK_SECRET", ""),
        )

        # Layer 1: CLI overrides (highest priority)
        for key, value in overrides.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        return config

    def record_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for a model."""
        if model not in self.token_usage:
            self.token_usage[model] = {"input": 0, "output": 0}
        self.token_usage[model]["input"] += input_tokens
        self.token_usage[model]["output"] += output_tokens

    @property
    def total_input_tokens(self) -> int:
        return sum(u["input"] for u in self.token_usage.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(u["output"] for u in self.token_usage.values())

    @property
    def estimated_cost_usd(self) -> float:
        """Calculate cost based on per-model pricing."""
        total = 0.0
        for model, usage in self.token_usage.items():
            pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
            total += usage["input"] * pricing["input"] / 1_000_000
            total += usage["output"] * pricing["output"] / 1_000_000
        return total

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.budget_limit_usd - self.estimated_cost_usd)

    def check_budget(self) -> bool:
        """Return True if there is budget remaining."""
        return self.estimated_cost_usd < self.budget_limit_usd


def _load_global_config() -> dict:
    """Load global config from ~/.autoforge/config.toml."""
    try:
        from autoforge.cli.setup_wizard import load_global_config
        return load_global_config()
    except ImportError:
        return {}
