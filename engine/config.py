"""AutoForge configuration."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# Approximate pricing per million tokens (USD)
MODEL_PRICING = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
}

# Default fallback pricing
DEFAULT_PRICING = {"input": 5.0, "output": 25.0}


@dataclass
class ForgeConfig:
    """Central configuration for an AutoForge run."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
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

    # Token tracking — per-model accumulators
    token_usage: dict[str, dict[str, int]] = field(default_factory=dict)

    # Execution
    max_agents: int = 3
    max_retries: int = 3
    quality_threshold: float = 0.7
    verbose: bool = False
    log_level: str = "INFO"

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
            self.constitution_dir = self.project_root / "constitution"
        if self.db_path is None:
            self.db_path = self.project_root / "autoforge.db"
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:12]

    @classmethod
    def from_env(cls, **overrides) -> ForgeConfig:
        """Create config from environment variables and optional overrides."""
        load_dotenv()

        # Parse allowed Telegram users (comma-separated)
        allowed_raw = os.getenv("FORGE_TELEGRAM_ALLOWED_USERS", "")
        allowed_users = [u.strip() for u in allowed_raw.split(",") if u.strip()]

        config = cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model_strong=os.getenv("FORGE_MODEL_STRONG", "claude-opus-4-6"),
            model_fast=os.getenv("FORGE_MODEL_FAST", "claude-sonnet-4-5-20250929"),
            budget_limit_usd=float(os.getenv("FORGE_BUDGET_LIMIT", "10.0")),
            max_agents=int(os.getenv("FORGE_MAX_AGENTS", "3")),
            log_level=os.getenv("FORGE_LOG_LEVEL", "INFO"),
            docker_enabled=os.getenv("FORGE_DOCKER_ENABLED", "").lower() in ("true", "1", "yes"),
            # Daemon
            daemon_poll_interval=int(os.getenv("FORGE_DAEMON_POLL_INTERVAL", "10")),
            # Telegram
            telegram_token=os.getenv("FORGE_TELEGRAM_TOKEN", ""),
            telegram_allowed_users=allowed_users,
            # Webhook
            webhook_enabled=os.getenv("FORGE_WEBHOOK_ENABLED", "").lower() in ("true", "1", "yes"),
            webhook_host=os.getenv("FORGE_WEBHOOK_HOST", "127.0.0.1"),
            webhook_port=int(os.getenv("FORGE_WEBHOOK_PORT", "8420")),
            webhook_secret=os.getenv("FORGE_WEBHOOK_SECRET", ""),
        )
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
