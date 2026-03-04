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


@dataclass
class ForgeConfig:
    """Central configuration for an AutoForge run."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    workspace_dir: Path | None = None
    constitution_dir: Path | None = None

    # LLM settings — multi-provider API keys
    api_keys: dict[str, str] = field(default_factory=dict)
    # Per-provider auth config (OAuth bearer, client_credentials, Google ADC, etc.)
    auth_config: dict[str, dict[str, Any]] = field(default_factory=dict)
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
    daemon_max_concurrent_projects: int = 1
    daemon_pid_file: Path | None = None
    db_path: Path | None = None  # SQLite database for project registry

    # Intake policy (applies to CLI queue, Telegram, Webhook)
    queue_max_size: int = 200
    requester_queue_limit: int = 3
    requester_daily_limit: int = 20
    requester_rate_limit: int = 5
    requester_rate_window_seconds: int = 60
    request_max_budget_usd: float = 1000.0
    request_max_description_chars: int = 10000

    # Telegram bot
    telegram_token: str = ""
    telegram_allowed_users: list[str] = field(default_factory=list)
    telegram_allow_public: bool = False

    # Webhook API
    webhook_enabled: bool = False
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 8420
    webhook_secret: str = ""  # Bearer token for webhook auth
    webhook_require_auth: bool = True
    webhook_allow_unauthenticated_local: bool = False
    webhook_admin_secret: str = ""
    webhook_requester_header: str = "X-Autoforge-Requester"
    webhook_trust_requester_header: bool = False
    webhook_idempotency_header: str = "Idempotency-Key"

    # Web tools (search + fetch)
    web_tools_enabled: bool = True  # Kill switch for all web tools
    search_backend: str = "duckduckgo"  # "duckduckgo" | "google" | "bing"
    search_api_key: str = ""  # API key for Google/Bing search backends

    # GitHub integration
    github_token: str = ""  # Personal access token for GitHub API (optional, increases rate limit)

    # Search tree (branching/backtracking)
    search_tree_enabled: bool = True  # Enable tree search for architecture exploration
    search_tree_max_candidates: int = 3  # Candidates per branch point

    # Mid-task checkpoints
    checkpoints_enabled: bool = True  # Enable mid-task direction checking
    checkpoint_interval: int = 8  # Check direction every N turns

    # Human-in-the-loop checkpoints
    confirm_phases: list[str] = field(default_factory=list)  # e.g. ["spec", "build"] or ["all"]

    # Build-phase TDD loops
    build_test_loops: int = 0  # 0=disabled, 1-3=test-fix iterations per task

    # Evolution engine
    evolution_enabled: bool = True  # Enable cross-project workflow evolution

    # Prompt optimization (DSPy/OPRO-style)
    prompt_optimization_enabled: bool = True  # Enable automatic prompt self-improvement

    # Process reward model (CodePRM)
    process_reward_enabled: bool = True  # Enable step-level code generation evaluation

    # RethinkMCTS
    mcts_enabled: bool = True        # Enable MCTS-enhanced search during BUILD
    mcts_max_iterations: int = 9     # Max MCTS iterations per decision point

    # EvoMAC text backpropagation
    evomac_enabled: bool = True      # Enable text gradient feedback between agents

    # SICA self-improvement
    sica_enabled: bool = True        # Enable self-improving constitution edits

    # Library-level RAG retrieval
    rag_enabled: bool = True         # Enable cross-project code knowledge base

    # Formal verification
    formal_verify_enabled: bool = True  # Enable static analysis + LLM formal checks

    # Conditional multi-agent debate
    debate_enabled: bool = True      # Enable reward-guided architecture debates

    # RedCode security scanning
    security_scan_enabled: bool = True  # Enable security vulnerability scanning

    # Reflexion episodic memory (NeurIPS 2023)
    reflexion_enabled: bool = True   # Enable verbal RL with failure reflections

    # Adaptive test-time compute (ICLR 2025)
    adaptive_compute_enabled: bool = True  # Enable difficulty-aware resource allocation

    # LDB block-level debugger (ACL 2024)
    ldb_debugger_enabled: bool = True  # Enable block-level fault localization

    # Speculative pipeline execution
    speculative_enabled: bool = True  # Enable parallel speculative pre-execution

    # Hierarchical task decomposition (Parsel / CodePlan)
    hierarchical_decomp_enabled: bool = True  # Enable function-level decomposition for complex tasks
    lean_prover_enabled: bool = True           # Enable Lean 4 formal theorem proving engine
    capability_dag_enabled: bool = True        # Enable universal self-growing knowledge graph
    theoretical_reasoning_enabled: bool = True # Enable cross-domain scientific reasoning & theory evolution

    # Project goal declaration (v2.8) — users declare workspace intent during setup
    # so the engine can allocate resources, context, and agent strategies to match.
    # Valid goal_type values: "general", "formal_verification", "web_app", "api_service",
    #   "data_pipeline", "mobile_app", "cli_tool", "library", "research"
    project_goal_type: str = "general"       # Workspace development goal category
    project_goal_description: str = ""       # Free-text description of what the project aims to achieve
    project_goal_disciplines: list[str] = field(default_factory=list)
    # e.g. ["mathematics", "physics", "cs_theory"] — guides which formal tools to load

    # Context budget management (v2.7)
    context_budget_tokens: int = 4000          # Max total supplementary context injected per agent call
    dag_ingest_confidence_threshold: float = 0.4  # Min confidence to ingest a concept into CapabilityDAG
    dag_ingest_relevance_threshold: float = 0.3   # Min relevance score for DAG retrieval results

    # Lean prover deep settings (v2.8)
    lean_mcts_iterations: int = 200           # Max MCTS iterations for proof search
    lean_decomposition_depth: int = 5         # Max recursive decomposition depth
    lean_auto_repair_passes: int = 3          # Max sorry-elimination repair passes
    lean_mathlib_search_enabled: bool = True   # Enable Mathlib-aware premise selection
    lean_pantograph_repl: bool = True          # Enable Pantograph interactive REPL mode

    # Multi-prover formal verification (v2.8)
    coq_enabled: bool = False                  # Coq theorem prover
    isabelle_enabled: bool = False             # Isabelle/HOL prover
    tlaplus_enabled: bool = False              # TLA+ model checker (distributed systems)
    z3_smt_enabled: bool = False               # Z3 SMT solver (program verification)
    dafny_enabled: bool = False                # Dafny (verified programming)
    # Known model name patterns for validation (prefix-based)
    _KNOWN_MODEL_PATTERNS: tuple[str, ...] = (
        "claude-",
        "gpt-",
        "o3",
        "o4",
        "gemini-",
    )

    def __post_init__(self) -> None:
        if self.workspace_dir is None:
            self.workspace_dir = self.project_root / "workspace"
        if self.constitution_dir is None:
            self.constitution_dir = autoforge.DATA_DIR / "constitution"
        if self.daemon_pid_file is None:
            self.daemon_pid_file = self.project_root / ".autoforge" / "daemon.pid"
        if self.db_path is None:
            self.db_path = self.project_root / "autoforge.db"
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

        daemon_max_concurrent_projects = _safe_int(
            "FORGE_DAEMON_MAX_CONCURRENT_PROJECTS",
            _to_int(global_config.get("daemon_max_concurrent_projects", 1), 1),
        )
        if daemon_max_concurrent_projects < 1:
            daemon_max_concurrent_projects = 1

        queue_max_size = _safe_int(
            "FORGE_QUEUE_MAX_SIZE",
            _to_int(global_config.get("queue_max_size", 200), 200),
        )
        if queue_max_size < 1:
            queue_max_size = 1

        requester_queue_limit = _safe_int(
            "FORGE_REQUESTER_QUEUE_LIMIT",
            _to_int(global_config.get("requester_queue_limit", 3), 3),
        )
        if requester_queue_limit < 1:
            requester_queue_limit = 1

        requester_daily_limit = _safe_int(
            "FORGE_REQUESTER_DAILY_LIMIT",
            _to_int(global_config.get("requester_daily_limit", 20), 20),
        )
        if requester_daily_limit < 1:
            requester_daily_limit = 1

        requester_rate_limit = _safe_int(
            "FORGE_REQUESTER_RATE_LIMIT",
            _to_int(global_config.get("requester_rate_limit", 5), 5),
        )
        if requester_rate_limit < 1:
            requester_rate_limit = 1

        requester_rate_window_seconds = _safe_int(
            "FORGE_REQUESTER_RATE_WINDOW_SECONDS",
            _to_int(global_config.get("requester_rate_window_seconds", 60), 60),
        )
        if requester_rate_window_seconds < 1:
            requester_rate_window_seconds = 1

        request_max_description_chars = _safe_int(
            "FORGE_REQUEST_MAX_DESCRIPTION_CHARS",
            _to_int(global_config.get("request_max_description_chars", 10000), 10000),
        )
        if request_max_description_chars < 100:
            request_max_description_chars = 100

        request_max_budget_usd = _safe_float(
            "FORGE_REQUEST_MAX_BUDGET_USD",
            float(global_config.get("request_max_budget_usd", 1000.0)),
        )
        if request_max_budget_usd <= 0:
            request_max_budget_usd = 1000.0

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
            # Daemon
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
            # Telegram
            telegram_token=os.getenv("FORGE_TELEGRAM_TOKEN", ""),
            telegram_allowed_users=allowed_users,
            telegram_allow_public=os.getenv("FORGE_TELEGRAM_ALLOW_PUBLIC", "").lower() in ("true", "1", "yes"),
            # Webhook
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
            # Web tools
            web_tools_enabled=os.getenv("FORGE_WEB_TOOLS", "true").lower() not in ("false", "0", "no"),
            search_backend=os.getenv("FORGE_SEARCH_BACKEND", global_config.get("search_backend", "duckduckgo")),
            search_api_key=os.getenv("FORGE_SEARCH_API_KEY", global_config.get("search_api_key", "")),
            # Build TDD
            build_test_loops=_safe_int("FORGE_BUILD_TEST_LOOPS", _to_int(global_config.get("build_test_loops", 0), 0)),
            # GitHub
            github_token=os.getenv("GITHUB_TOKEN", global_config.get("github_token", "")),
            # Search tree
            search_tree_enabled=os.getenv("FORGE_SEARCH_TREE", "true").lower() not in ("false", "0", "no"),
            search_tree_max_candidates=_safe_int("FORGE_SEARCH_CANDIDATES", _to_int(global_config.get("search_tree_max_candidates", 3), 3)),
            # Checkpoints
            checkpoints_enabled=os.getenv("FORGE_CHECKPOINTS", "true").lower() not in ("false", "0", "no"),
            checkpoint_interval=_safe_int("FORGE_CHECKPOINT_INTERVAL", _to_int(global_config.get("checkpoint_interval", 8), 8)),
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
