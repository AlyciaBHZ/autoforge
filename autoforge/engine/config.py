"""AutoForge configuration.

Config priority chain (highest → lowest):
    1. CLI arguments
    2. Project .env file (in repo root)
    3. Global ~/.autoforge/config.toml (user-level defaults)
    4. Built-in defaults
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

import autoforge


# Approximate pricing per million tokens (USD)
MODEL_PRICING = {
    # Anthropic
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    # OpenAI
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "o3": {"input": 10.0, "output": 40.0},
    "o3-mini": {"input": 1.1, "output": 4.4},
    "o4-mini": {"input": 1.1, "output": 4.4},
    # Google Gemini
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.6},
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
}

# Default fallback pricing
DEFAULT_PRICING = {"input": 5.0, "output": 25.0}


@dataclass
class ForgeConfig:
    """Central configuration for an AutoForge run."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    workspace_dir: Path | None = None
    constitution_dir: Path | None = None

    # LLM settings — multi-provider API keys
    api_keys: dict[str, str] = field(default_factory=dict)
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

    def __post_init__(self) -> None:
        if self.workspace_dir is None:
            self.workspace_dir = self.project_root / "workspace"
        if self.constitution_dir is None:
            self.constitution_dir = autoforge.DATA_DIR / "constitution"
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:12]

    @property
    def anthropic_api_key(self) -> str:
        """Backward-compatible access to Anthropic API key."""
        return self.api_keys.get("anthropic", "")

    @anthropic_api_key.setter
    def anthropic_api_key(self, value: str) -> None:
        """Backward-compatible setter for Anthropic API key."""
        if value:
            self.api_keys["anthropic"] = value

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

        # Build API keys dict from all sources
        api_keys: dict[str, str] = {}

        # From global config
        if global_config.get("anthropic_api_key"):
            api_keys["anthropic"] = global_config["anthropic_api_key"]
        if global_config.get("openai_api_key"):
            api_keys["openai"] = global_config["openai_api_key"]
        if global_config.get("google_api_key"):
            api_keys["google"] = global_config["google_api_key"]

        # From env vars (override global)
        if os.getenv("ANTHROPIC_API_KEY"):
            api_keys["anthropic"] = os.getenv("ANTHROPIC_API_KEY", "")
        if os.getenv("OPENAI_API_KEY"):
            api_keys["openai"] = os.getenv("OPENAI_API_KEY", "")
        if os.getenv("GOOGLE_API_KEY"):
            api_keys["google"] = os.getenv("GOOGLE_API_KEY", "")

        config = cls(
            api_keys=api_keys,
            model_strong=(
                os.getenv("FORGE_MODEL_STRONG")
                or global_config.get("model_strong", "claude-opus-4-6")
            ),
            model_fast=(
                os.getenv("FORGE_MODEL_FAST")
                or global_config.get("model_fast", "claude-sonnet-4-5-20250929")
            ),
            budget_limit_usd=float(
                os.getenv("FORGE_BUDGET_LIMIT")
                or global_config.get("budget_limit_usd", 10.0)
            ),
            max_agents=int(
                os.getenv("FORGE_MAX_AGENTS")
                or global_config.get("max_agents", 3)
            ),
            log_level=(
                os.getenv("FORGE_LOG_LEVEL")
                or global_config.get("log_level", "INFO")
            ),
            docker_enabled=(
                os.getenv("FORGE_DOCKER_ENABLED", "").lower() in ("true", "1", "yes")
                or global_config.get("docker_enabled", False)
            ),
            mode=global_config.get("mode", "developer"),
            mobile_target=global_config.get("mobile_target", "none"),
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
