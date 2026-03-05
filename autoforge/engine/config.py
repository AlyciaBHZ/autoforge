"""AutoForge configuration.

Config priority chain (highest -> lowest):
    1. CLI arguments
    2. Project .env file (in repo root)
    3. Global ~/.autoforge/config.toml (user-level defaults)
    4. Built-in defaults

ForgeConfig is the central config object. Advanced sub-systems use
dedicated sub-config dataclasses to keep the top-level clean.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    except (ValueError, TypeError):
        logging.getLogger(__name__).warning(
            f"Invalid value for {key}={raw!r}, using default {default}"
        )
        return default


def _to_int(value: Any, default: int) -> int:
    """Safely convert a value to int with a fallback default.

    Handles strings from TOML config, None, and other non-int types.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        logging.getLogger(__name__).warning(
            f"Cannot convert {value!r} to int, using default {default}"
        )
        return default


def _load_clamped_int(
    env_key: str, global_key: str, global_config: dict,
    default: int, minimum: int = 1,
) -> int:
    """Load an int from env / global config with a floor clamp."""
    val = _safe_int(env_key, _to_int(global_config.get(global_key, default), default))
    return max(minimum, val)


# ── Sub-config dataclasses ────────────────────────────────────────


@dataclass
class DaemonConfig:
    """Settings for the 24/7 background daemon and input channels."""

    daemon_enabled: bool = False
    daemon_poll_interval: int = 10
    daemon_max_concurrent_projects: int = 1
    daemon_pid_file: Path | None = None
    db_path: Path | None = None

    # Intake policy (CLI queue, Telegram, Webhook)
    queue_max_size: int = 200
    requester_queue_limit: int = 3
    requester_daily_limit: int = 20
    requester_rate_limit: int = 5
    requester_rate_window_seconds: int = 60
    request_max_budget_usd: float = 1000.0
    request_max_description_chars: int = 10000

    # Telegram channel
    telegram_token: str = ""
    telegram_allowed_users: list[str] = field(default_factory=list)
    telegram_allow_public: bool = False

    # Webhook channel
    webhook_enabled: bool = False
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 8420
    webhook_secret: str = ""
    webhook_require_auth: bool = True
    webhook_allow_unauthenticated_local: bool = False
    webhook_admin_secret: str = ""
    webhook_requester_header: str = "X-Autoforge-Requester"
    webhook_trust_requester_header: bool = False
    webhook_idempotency_header: str = "Idempotency-Key"


@dataclass
class ToolsConfig:
    """Settings for agent tools (web search, code search, GitHub)."""

    web_tools_enabled: bool = True
    search_backend: str = "duckduckgo"
    search_api_key: str = ""
    github_token: str = ""


@dataclass
class PipelineConfig:
    """Settings for the core build pipeline behaviour."""

    search_tree_enabled: bool = True
    search_tree_max_candidates: int = 3
    checkpoints_enabled: bool = True
    checkpoint_interval: int = 8
    confirm_phases: list[str] = field(default_factory=list)
    build_test_loops: int = 0
    speculative_enabled: bool = True
    hierarchical_decomp_enabled: bool = True


@dataclass
class AdvancedConfig:
    """Feature flags and settings for advanced ML/AI sub-systems.

    All flags default to True (enabled). Set to False to skip the engine.
    """

    # Core ML engines
    evolution_enabled: bool = True
    prompt_optimization_enabled: bool = True
    process_reward_enabled: bool = True
    mcts_enabled: bool = True
    mcts_max_iterations: int = 9
    evomac_enabled: bool = True
    sica_enabled: bool = True
    rag_enabled: bool = True
    formal_verify_enabled: bool = True
    debate_enabled: bool = True
    security_scan_enabled: bool = True
    reflexion_enabled: bool = True
    adaptive_compute_enabled: bool = True
    ldb_debugger_enabled: bool = True

    # Knowledge systems
    lean_prover_enabled: bool = True
    capability_dag_enabled: bool = True
    theoretical_reasoning_enabled: bool = True
    autonomous_discovery_enabled: bool = True
    paper_formalizer_enabled: bool = True
    cloud_prover_enabled: bool = False

    # Context budget
    context_budget_tokens: int = 4000
    dag_ingest_confidence_threshold: float = 0.4
    dag_ingest_relevance_threshold: float = 0.3

    # Lean prover deep settings
    lean_mcts_iterations: int = 200
    lean_decomposition_depth: int = 5
    lean_auto_repair_passes: int = 3
    lean_mathlib_search_enabled: bool = True
    lean_pantograph_repl: bool = True

    # Multi-prover formal verification
    coq_enabled: bool = False
    isabelle_enabled: bool = False
    tlaplus_enabled: bool = False
    z3_smt_enabled: bool = False
    dafny_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate bounds on critical settings (D6)."""
        import warnings

        if self.lean_mcts_iterations <= 0:
            raise ValueError(
                f"lean_mcts_iterations must be > 0 (got {self.lean_mcts_iterations})"
            )
        if not (1 <= self.lean_decomposition_depth <= 20):
            raise ValueError(
                f"lean_decomposition_depth must be in [1, 20] "
                f"(got {self.lean_decomposition_depth})"
            )
        if not (0 <= self.lean_auto_repair_passes <= 10):
            raise ValueError(
                f"lean_auto_repair_passes must be in [0, 10] "
                f"(got {self.lean_auto_repair_passes})"
            )
        if not (0.0 <= self.dag_ingest_confidence_threshold <= 1.0):
            raise ValueError(
                f"dag_ingest_confidence_threshold must be in [0, 1] "
                f"(got {self.dag_ingest_confidence_threshold})"
            )

        # Warn if cloud prover enabled but no backend configured
        if self.cloud_prover_enabled:
            import os
            has_docker = bool(os.environ.get("DOCKER_HOST"))
            has_ssh = False  # Would need SSH config, but we can't check here
            if not has_docker:
                warnings.warn(
                    "cloud_prover_enabled=True but no Docker host detected "
                    "(set DOCKER_HOST or configure SSH in ForgeConfig)",
                    UserWarning,
                    stacklevel=2,
                )


@dataclass
class ProjectGoalConfig:
    """User-declared workspace intent (guides engine resource allocation)."""

    project_goal_type: str = "general"
    project_goal_description: str = ""
    project_goal_disciplines: list[str] = field(default_factory=list)


# ── Main config ──────────────────────────────────────────────────


@dataclass
class ForgeConfig:
    """Central configuration for an AutoForge run.

    Core settings live directly on ForgeConfig. Advanced sub-systems
    are grouped into dedicated sub-config objects for maintainability.
    """

    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    workspace_dir: Path | None = None
    constitution_dir: Path | None = None

    # LLM settings — multi-provider API keys
    api_keys: dict[str, str] = field(default_factory=dict)
    auth_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_strong: str = "claude-opus-4-6"
    model_fast: str = "claude-sonnet-4-5-20250929"
    max_tokens_strong: int = 16384
    max_tokens_fast: int = 8192

    # Budget
    budget_limit_usd: float = 10.0
    token_usage: dict[str, dict[str, int]] = field(default_factory=dict)

    # Execution
    max_agents: int = 3
    max_retries: int = 3
    quality_threshold: float = 0.7
    verbose: bool = False
    log_level: str = "INFO"

    # Operating mode
    mode: str = "developer"
    mobile_target: str = "none"
    mobile_framework: str = "react-native"

    # Run identity
    run_id: str = ""

    # Docker
    sandbox_image: str = "autoforge-sandbox:latest"
    docker_enabled: bool = False

    # ── Sub-configs ──
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    goal: ProjectGoalConfig = field(default_factory=ProjectGoalConfig)

    # Known model name patterns for validation (prefix-based)
    _KNOWN_MODEL_PATTERNS: tuple[str, ...] = (
        "claude-",
        "gpt-",
        "o3",
        "o4",
        "gemini-",
    )

    # ── Backward-compatible sub-config delegation ──
    # These names used to live on sub-config objects (daemon, tools, pipeline,
    # advanced, goal).  __getattr__ / __setattr__ delegate transparently so
    # existing code can keep using  config.xxx_enabled  without changes.

    _SUB_CONFIG_NAMES: tuple[str, ...] = (
        "daemon", "tools", "pipeline", "advanced", "goal",
    )

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to sub-configs for backward compat."""
        for sub_name in ForgeConfig._SUB_CONFIG_NAMES:
            try:
                sub = object.__getattribute__(self, sub_name)
            except AttributeError:
                continue
            if hasattr(type(sub), name) or name in sub.__dict__:
                return getattr(sub, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Route attribute sets to sub-configs when they own the field."""
        # Always allow setting ForgeConfig's own dataclass fields and private attrs
        own_fields = ForgeConfig.__dataclass_fields__
        if name in own_fields or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        # Check sub-configs for the attribute
        for sub_name in ForgeConfig._SUB_CONFIG_NAMES:
            try:
                sub = object.__getattribute__(self, sub_name)
            except AttributeError:
                continue
            if hasattr(type(sub), name) or name in sub.__dict__:
                setattr(sub, name, value)
                return
        # Fall back to normal set (e.g. for dynamically added attrs)
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        if self.workspace_dir is None:
            self.workspace_dir = self.project_root / "workspace"
        if self.constitution_dir is None:
            self.constitution_dir = autoforge.DATA_DIR / "constitution"
        if self.daemon.daemon_pid_file is None:
            self.daemon.daemon_pid_file = self.project_root / ".autoforge" / "daemon.pid"
        if self.daemon.db_path is None:
            self.daemon.db_path = self.project_root / "autoforge.db"
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:12]

        # Warn (but don't block) if model names don't match known patterns
        logger = logging.getLogger(__name__)
        for label, model_name in (
            ("model_strong", self.model_strong),
            ("model_fast", self.model_fast),
        ):
            if not any(model_name.startswith(p) for p in self._KNOWN_MODEL_PATTERNS):
                logger.warning(
                    "%s=%r does not match any known model pattern (%s). "
                    "Proceeding anyway — verify this is correct.",
                    label,
                    model_name,
                    ", ".join(self._KNOWN_MODEL_PATTERNS),
                )

    @property
    def has_api_key(self) -> bool:
        """Check if at least one LLM provider is configured (API key or OAuth)."""
        return any(v for v in self.api_keys.values()) or bool(self.auth_config)

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

        # Load per-provider auth config (OAuth bearer, client_credentials, ADC)
        auth_config: dict[str, dict[str, Any]] = {}
        for provider in ("anthropic", "openai", "google"):
            provider_auth = global_config.get(f"auth_{provider}", {})
            if provider_auth:
                auth_config[provider] = dict(provider_auth)

        # Env var overrides for auth
        if os.getenv("OPENAI_BASE_URL"):
            auth_config.setdefault("openai", {})
            auth_config["openai"]["base_url"] = os.getenv("OPENAI_BASE_URL", "")
            if "auth_method" not in auth_config["openai"]:
                auth_config["openai"]["auth_method"] = "oauth_bearer"
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            auth_config.setdefault("google", {})
            if "auth_method" not in auth_config["google"]:
                auth_config["google"]["auth_method"] = "adc"

        # AWS Bedrock env overrides
        if os.getenv("CLAUDE_CODE_USE_BEDROCK", "").lower() in ("1", "true", "yes"):
            auth_config.setdefault("anthropic", {})
            if "auth_method" not in auth_config["anthropic"]:
                auth_config["anthropic"]["auth_method"] = "bedrock"
            if os.getenv("AWS_REGION"):
                auth_config["anthropic"]["aws_region"] = os.getenv("AWS_REGION", "")

        # Google Vertex AI env overrides
        if os.getenv("CLAUDE_CODE_USE_VERTEX", "").lower() in ("1", "true", "yes"):
            auth_config.setdefault("anthropic", {})
            if "auth_method" not in auth_config["anthropic"]:
                auth_config["anthropic"]["auth_method"] = "vertex_ai"
            if os.getenv("ANTHROPIC_VERTEX_PROJECT_ID"):
                auth_config["anthropic"]["project_id"] = os.getenv("ANTHROPIC_VERTEX_PROJECT_ID", "")
            if os.getenv("CLOUD_ML_REGION"):
                auth_config["anthropic"]["region"] = os.getenv("CLOUD_ML_REGION", "")

        # Parse allowed Telegram users (comma-separated)
        allowed_raw = os.getenv("FORGE_TELEGRAM_ALLOWED_USERS", "")
        allowed_users = [u.strip() for u in allowed_raw.split(",") if u.strip()]

        daemon_pid_raw = (
            os.getenv("FORGE_DAEMON_PID_FILE")
            or str(global_config.get("daemon_pid_file", "")).strip()
        )
        daemon_pid_file = Path(daemon_pid_raw).expanduser() if daemon_pid_raw else None

        _cli = lambda env, key, default, minimum=1: _load_clamped_int(
            env, key, global_config, default, minimum,
        )
        daemon_max_concurrent_projects = _cli("FORGE_DAEMON_MAX_CONCURRENT_PROJECTS", "daemon_max_concurrent_projects", 1)
        queue_max_size = _cli("FORGE_QUEUE_MAX_SIZE", "queue_max_size", 200)
        requester_queue_limit = _cli("FORGE_REQUESTER_QUEUE_LIMIT", "requester_queue_limit", 3)
        requester_daily_limit = _cli("FORGE_REQUESTER_DAILY_LIMIT", "requester_daily_limit", 20)
        requester_rate_limit = _cli("FORGE_REQUESTER_RATE_LIMIT", "requester_rate_limit", 5)
        requester_rate_window_seconds = _cli("FORGE_REQUESTER_RATE_WINDOW_SECONDS", "requester_rate_window_seconds", 60)
        request_max_description_chars = _cli("FORGE_REQUEST_MAX_DESCRIPTION_CHARS", "request_max_description_chars", 10000, minimum=100)

        request_max_budget_usd = _safe_float(
            "FORGE_REQUEST_MAX_BUDGET_USD",
            float(global_config.get("request_max_budget_usd", 1000.0)),
        )
        if request_max_budget_usd <= 0:
            request_max_budget_usd = 1000.0

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
            _to_int(global_config.get("max_agents", 3), 3),
        )
        if max_agents < 1:
            max_agents = 1
        elif max_agents > 50:
            max_agents = 50

        config = cls(
            api_keys=api_keys,
            auth_config=auth_config,
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
            # Sub-configs
            daemon=DaemonConfig(
                daemon_poll_interval=_safe_int("FORGE_DAEMON_POLL_INTERVAL", 10),
                daemon_max_concurrent_projects=daemon_max_concurrent_projects,
                daemon_pid_file=daemon_pid_file,
                queue_max_size=queue_max_size,
                requester_queue_limit=requester_queue_limit,
                requester_daily_limit=requester_daily_limit,
                requester_rate_limit=requester_rate_limit,
                requester_rate_window_seconds=requester_rate_window_seconds,
                request_max_budget_usd=request_max_budget_usd,
                request_max_description_chars=request_max_description_chars,
                telegram_token=os.getenv("FORGE_TELEGRAM_TOKEN", ""),
                telegram_allowed_users=allowed_users,
                telegram_allow_public=os.getenv("FORGE_TELEGRAM_ALLOW_PUBLIC", "").lower() in ("true", "1", "yes"),
                webhook_enabled=os.getenv("FORGE_WEBHOOK_ENABLED", "").lower() in ("true", "1", "yes"),
                webhook_host=os.getenv("FORGE_WEBHOOK_HOST", "127.0.0.1"),
                webhook_port=_safe_int("FORGE_WEBHOOK_PORT", 8420),
                webhook_secret=os.getenv("FORGE_WEBHOOK_SECRET", ""),
                webhook_require_auth=os.getenv("FORGE_WEBHOOK_REQUIRE_AUTH", "true").lower() not in ("false", "0", "no"),
                webhook_allow_unauthenticated_local=os.getenv("FORGE_WEBHOOK_ALLOW_UNAUTH_LOCAL", "").lower() in ("true", "1", "yes"),
                webhook_admin_secret=os.getenv("FORGE_WEBHOOK_ADMIN_SECRET", ""),
                webhook_requester_header=os.getenv("FORGE_WEBHOOK_REQUESTER_HEADER", "X-Autoforge-Requester"),
                webhook_trust_requester_header=os.getenv("FORGE_WEBHOOK_TRUST_REQUESTER_HEADER", "").lower() in ("true", "1", "yes"),
                webhook_idempotency_header=os.getenv("FORGE_WEBHOOK_IDEMPOTENCY_HEADER", "Idempotency-Key"),
            ),
            tools=ToolsConfig(
                web_tools_enabled=os.getenv("FORGE_WEB_TOOLS", "true").lower() not in ("false", "0", "no"),
                search_backend=os.getenv("FORGE_SEARCH_BACKEND", global_config.get("search_backend", "duckduckgo")),
                search_api_key=os.getenv("FORGE_SEARCH_API_KEY", global_config.get("search_api_key", "")),
                github_token=os.getenv("GITHUB_TOKEN", global_config.get("github_token", "")),
            ),
            pipeline=PipelineConfig(
                search_tree_enabled=os.getenv("FORGE_SEARCH_TREE", "true").lower() not in ("false", "0", "no"),
                search_tree_max_candidates=_safe_int("FORGE_SEARCH_CANDIDATES", _to_int(global_config.get("search_tree_max_candidates", 3), 3)),
                checkpoints_enabled=os.getenv("FORGE_CHECKPOINTS", "true").lower() not in ("false", "0", "no"),
                checkpoint_interval=_safe_int("FORGE_CHECKPOINT_INTERVAL", _to_int(global_config.get("checkpoint_interval", 8), 8)),
                build_test_loops=_safe_int("FORGE_BUILD_TEST_LOOPS", _to_int(global_config.get("build_test_loops", 0), 0)),
            ),
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


# ── Backward-compatible constructor ──────────────────────────────
# Wrap the dataclass-generated __init__ so callers can pass legacy
# field names (e.g. web_tools_enabled=False) that are now property
# proxies routed to sub-config objects.

_ForgeConfig_dc_init = ForgeConfig.__init__


def _ForgeConfig_compat_init(self: ForgeConfig, **kwargs: Any) -> None:
    real_fields = {f.name for f in ForgeConfig.__dataclass_fields__.values()}
    extra = {k: v for k, v in kwargs.items() if k not in real_fields}
    clean = {k: v for k, v in kwargs.items() if k in real_fields}
    _ForgeConfig_dc_init(self, **clean)
    for k, v in extra.items():
        if v is not None:
            try:
                setattr(self, k, v)
            except AttributeError:
                pass


ForgeConfig.__init__ = _ForgeConfig_compat_init  # type: ignore[assignment]


def _load_global_config() -> dict:
    """Load global config from ~/.autoforge/config.toml."""
    try:
        from autoforge.cli.setup_wizard import load_global_config
        return load_global_config()
    except ImportError:
        return {}
