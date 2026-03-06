"""Harness environment determinism helpers (Docker images/snapshots)."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from autoforge.engine.runtime.commands import run_args

logger = logging.getLogger(__name__)


def compute_env_fingerprint(spec: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(spec), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ImageBuildPlan:
    tag: str
    dockerfile_text: str
    context_dir: Path


async def docker_image_exists(image: str, *, cwd: Path) -> bool:
    res = await run_args(
        ["docker", "image", "inspect", str(image)],
        cwd=cwd,
        timeout_s=15.0,
        max_stdout_chars=2000,
        max_stderr_chars=2000,
        label="harness.docker.inspect",
    )
    return res.exit_code == 0


async def docker_build_image(plan: ImageBuildPlan) -> bool:
    plan.context_dir.mkdir(parents=True, exist_ok=True)
    dockerfile = plan.context_dir / "Dockerfile"
    dockerfile.write_text(plan.dockerfile_text, encoding="utf-8")
    res = await run_args(
        ["docker", "build", "-t", plan.tag, str(plan.context_dir)],
        cwd=plan.context_dir,
        timeout_s=3600.0,
        max_stdout_chars=20000,
        max_stderr_chars=5000,
        label="harness.docker.build",
    )
    if res.exit_code != 0:
        logger.warning(
            "Docker build failed for %s: %s",
            plan.tag,
            (res.stderr or res.stdout or "").strip()[:500],
        )
        return False
    return True


def build_plan_from_dockerfile(
    dockerfile_text: str,
    *,
    base_dir: Path,
    tag_prefix: str = "autoforge-harness",
) -> ImageBuildPlan:
    fingerprint = compute_env_fingerprint({"dockerfile": dockerfile_text})
    tag = f"{tag_prefix}:{fingerprint}"
    ctx = base_dir / "images" / fingerprint
    return ImageBuildPlan(tag=tag, dockerfile_text=dockerfile_text, context_dir=ctx)

