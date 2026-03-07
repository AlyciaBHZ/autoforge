from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

DEVELOPMENT_HARNESS_SCHEMA_VERSION = 1


def normalize_development_payload(
    payload: dict[str, Any] | None,
    *,
    artifact_type: str,
) -> dict[str, Any]:
    data = dict(payload or {})
    data.setdefault("schema_version", DEVELOPMENT_HARNESS_SCHEMA_VERSION)
    data.setdefault("artifact_type", str(artifact_type or "development_artifact"))
    return data


def serialize_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serialize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        try:
            return serialize_jsonable(vars(value))
        except Exception:
            return str(value)
    return str(value)


def resolve_development_harness_root(
    *,
    config: Any | None = None,
    project_dir: Path | None = None,
    working_dir: Path | None = None,
) -> Path:
    if project_dir is not None:
        base_dir = Path(project_dir)
    elif working_dir is not None:
        base_dir = Path(working_dir)
    else:
        project_root = getattr(config, "project_root", None)
        base_dir = Path(project_root) if project_root is not None else Path.cwd()
    root = base_dir / ".autoforge" / "development_harness"
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_development_json(path: Path, payload: dict[str, Any], *, artifact_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(
        json.dumps(
            normalize_development_payload(payload, artifact_type=artifact_type),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def append_development_jsonl(path: Path, payload: dict[str, Any], *, event_type: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload or {})
    record.setdefault("ts", time.time())
    if event_type:
        record.setdefault("event_type", str(event_type))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialize_jsonable(record), ensure_ascii=False) + "\n")
