"""Environment policy for sandboxed command execution.

AutoForge runs user code in multiple backends (subprocess/docker/slurm).
We want a consistent, minimal set of environment variables that:
  - avoids interactive prompts (CI=1, GIT_TERMINAL_PROMPT=0, PIP_NO_INPUT=1)
  - supports deterministic evaluation mode (seed/time-related knobs)
  - supports HPC dependency proxies/caches (PIP_INDEX_URL / NPM registry)

Call-sites should treat the returned mapping as an overlay onto os.environ.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Mapping

from autoforge.engine.config import ForgeConfig


def _is_docker_safe_cache_dir(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False
    # If it's explicitly a container path, allow.
    if v.replace("\\", "/").startswith("/workspace/"):
        return True
    p = Path(v)
    # Avoid injecting host-absolute paths into containers (they usually don't exist).
    return not p.is_absolute()


def build_env_overrides(
    config: ForgeConfig,
    *,
    backend: str,
) -> dict[str, str]:
    """Build environment overrides for a given execution backend."""
    b = (backend or "auto").strip().lower()

    env: dict[str, str] = {
        "CI": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        # Prevent accidental global/site-packages installs when running in SubprocessSandbox.
        # Project-local venv installs should use the venv's python/pip explicitly.
        "PIP_REQUIRE_VIRTUALENV": "1",
    }

    # Determinism: keep this conservative. Full determinism across arbitrary
    # user code is impossible, but we can stabilize key entropy sources.
    if bool(getattr(config, "deterministic", False)):
        try:
            seed = int(getattr(config, "deterministic_seed", 0) or 0)
        except (ValueError, TypeError):
            seed = 0
        env["PYTHONHASHSEED"] = str(seed)
        env["TZ"] = "UTC"
        try:
            sde = int(getattr(config, "deterministic_source_date_epoch", 0) or 0)
        except (ValueError, TypeError):
            sde = 0
        if sde > 0:
            env["SOURCE_DATE_EPOCH"] = str(sde)

    pip_index_url = str(getattr(config, "pip_index_url", "") or "").strip()
    if pip_index_url:
        env["PIP_INDEX_URL"] = pip_index_url

    npm_registry = str(getattr(config, "npm_registry", "") or "").strip()
    if npm_registry:
        env["NPM_CONFIG_REGISTRY"] = npm_registry

    # Cache dirs: safe on subprocess/slurm where paths refer to host/shared FS.
    # For docker, only allow relative or /workspace paths to avoid breaking installs.
    pip_cache_dir = str(getattr(config, "pip_cache_dir", "") or "").strip()
    if pip_cache_dir:
        if b != "docker" or _is_docker_safe_cache_dir(pip_cache_dir):
            env["PIP_CACHE_DIR"] = pip_cache_dir

    npm_cache_dir = str(getattr(config, "npm_cache_dir", "") or "").strip()
    if npm_cache_dir:
        if b != "docker" or _is_docker_safe_cache_dir(npm_cache_dir):
            env["NPM_CONFIG_CACHE"] = npm_cache_dir

    return env


def shell_export_block(env: Mapping[str, str]) -> str:
    """Render env as a stable `export KEY='value'` block for POSIX shells."""
    lines: list[str] = []
    for k, v in sorted(env.items(), key=lambda kv: kv[0]):
        key = str(k).strip()
        if not key:
            continue
        if any(ch.isspace() for ch in key) or "=" in key:
            continue
        lines.append(f"export {key}={shlex.quote(str(v))}")
    return "".join(f"{line}\n" for line in lines)
