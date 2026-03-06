"""Harness dataset schema + loader (JSONL).

Each line in a dataset is a JSON object describing a single evaluation case.
The schema is intentionally minimal but extensible:

Required:
  - id: str
  - mode: "generate" | "import" | "review"  (default: "generate")

Mode-specific:
  - generate: description (str)
  - import/review: project_path (str), optional enhance (str for import)

Optional:
  - budget_usd: float
  - max_agents: int
  - env: { sandbox_image, docker_memory_limit, docker_cpu_limit, docker_pids_limit, docker_network_mode, docker_required, dockerfile }
  - judge: { visible_test_command, hidden_test_command, hide_paths, golden_patch_path }
  - trace: { enabled, llm_content, command_output, fs_snapshots }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


CaseMode = Literal["generate", "import", "review"]


def _as_str(v: Any) -> str:
    return str(v) if v is not None else ""


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_int(v: Any, default: int | None = None) -> int | None:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_float(v: Any, default: float | None = None) -> float | None:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_list_str(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        out: list[str] = []
        for item in v:
            s = _as_str(item).strip()
            if s:
                out.append(s)
        return out
    s = _as_str(v).strip()
    return [s] if s else []


def _safe_relpath(rel: str) -> str:
    rel = (rel or "").replace("\\", "/").lstrip("/")
    if ".." in Path(rel).parts:
        raise ValueError("Path traversal not allowed")
    return rel


@dataclass(frozen=True)
class CaseEnv:
    sandbox_image: str | None = None
    docker_memory_limit: str | None = None
    docker_cpu_limit: str | None = None
    docker_pids_limit: int | None = None
    docker_network_mode: str | None = None
    docker_required: bool | None = None
    dockerfile: str | None = None  # optional inline dockerfile for prewarm/build


@dataclass(frozen=True)
class CaseTrace:
    enabled: bool = True
    llm_content: bool = True
    command_output: bool = True
    fs_snapshots: bool = True


@dataclass(frozen=True)
class CaseJudge:
    visible_test_command: str | None = None
    hidden_test_command: str | None = None
    hide_paths: list[str] = field(default_factory=list)
    golden_patch_path: str | None = None


@dataclass(frozen=True)
class HarnessCase:
    id: str
    mode: CaseMode = "generate"
    description: str = ""
    project_path: str = ""
    enhance: str = ""
    budget_usd: float | None = None
    max_agents: int | None = None
    env: CaseEnv = field(default_factory=CaseEnv)
    judge: CaseJudge = field(default_factory=CaseJudge)
    trace: CaseTrace = field(default_factory=CaseTrace)
    raw: dict[str, Any] = field(default_factory=dict)


def load_dataset(path: Path) -> list[HarnessCase]:
    cases: list[HarnessCase] = []
    base_dir = path.resolve().parent
    for idx, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        obj = json.loads(stripped)
        if not isinstance(obj, dict):
            raise ValueError(f"Dataset line {idx}: expected JSON object")
        cid = _as_str(obj.get("id") or f"case_{idx}").strip()
        if not cid:
            cid = f"case_{idx}"

        mode_raw = _as_str(obj.get("mode") or "generate").strip().lower()
        mode: CaseMode = "generate"
        if mode_raw in {"import", "review", "generate"}:
            mode = mode_raw  # type: ignore[assignment]

        description = _as_str(obj.get("description")).strip()
        project_path = _as_str(obj.get("project_path") or obj.get("project")).strip()
        enhance = _as_str(obj.get("enhance") or obj.get("enhancement")).strip()

        # Resolve project_path relative to dataset file.
        if project_path and not Path(project_path).is_absolute():
            project_path = str((base_dir / project_path).resolve())
        if mode == "generate" and not description:
            raise ValueError(
                f"Dataset line {idx} ({cid}): missing description for mode=generate"
            )
        if mode in {"import", "review"} and not project_path:
            raise ValueError(
                f"Dataset line {idx} ({cid}): missing project_path for mode={mode}"
            )

        env_obj = obj.get("env") if isinstance(obj.get("env"), dict) else {}
        env = CaseEnv(
            sandbox_image=_as_str(env_obj.get("sandbox_image") or env_obj.get("image")).strip() or None,
            docker_memory_limit=_as_str(env_obj.get("docker_memory_limit") or env_obj.get("memory_limit")).strip() or None,
            docker_cpu_limit=_as_str(env_obj.get("docker_cpu_limit") or env_obj.get("cpu_limit")).strip() or None,
            docker_pids_limit=_as_int(env_obj.get("docker_pids_limit") or env_obj.get("pids_limit")),
            docker_network_mode=_as_str(env_obj.get("docker_network_mode") or env_obj.get("network_mode")).strip() or None,
            docker_required=(
                _as_bool(env_obj.get("docker_required"), default=False)
                if "docker_required" in env_obj
                else None
            ),
            dockerfile=_as_str(env_obj.get("dockerfile")).strip() or None,
        )

        judge_obj = obj.get("judge") if isinstance(obj.get("judge"), dict) else {}
        golden_patch_path = _as_str(
            judge_obj.get("golden_patch_path") or judge_obj.get("golden_patch")
        ).strip()
        if golden_patch_path and not Path(golden_patch_path).is_absolute():
            golden_patch_path = str((base_dir / golden_patch_path).resolve())
        judge = CaseJudge(
            visible_test_command=_as_str(
                judge_obj.get("visible_test_command") or judge_obj.get("test")
            ).strip() or None,
            hidden_test_command=_as_str(
                judge_obj.get("hidden_test_command") or judge_obj.get("holdout_test")
            ).strip() or None,
            hide_paths=[
                _safe_relpath(p)
                for p in _as_list_str(judge_obj.get("hide_paths") or judge_obj.get("hide"))
            ],
            golden_patch_path=golden_patch_path or None,
        )

        trace_obj = obj.get("trace") if isinstance(obj.get("trace"), dict) else {}
        trace = CaseTrace(
            enabled=_as_bool(trace_obj.get("enabled"), default=True),
            llm_content=_as_bool(trace_obj.get("llm_content"), default=True),
            command_output=_as_bool(trace_obj.get("command_output"), default=True),
            fs_snapshots=_as_bool(trace_obj.get("fs_snapshots"), default=True),
        )

        cases.append(
            HarnessCase(
                id=cid,
                mode=mode,
                description=description,
                project_path=project_path,
                enhance=enhance,
                budget_usd=_as_float(obj.get("budget_usd") or obj.get("budget")),
                max_agents=_as_int(obj.get("max_agents") or obj.get("agents")),
                env=env,
                judge=judge,
                trace=trace,
                raw=obj,
            )
        )
    return cases

