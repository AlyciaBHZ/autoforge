from __future__ import annotations

import json
from pathlib import Path
from typing import Any

KERNEL_SCHEMA_VERSION = 1


def normalize_kernel_payload(
    payload: dict[str, Any] | None,
    *,
    artifact_type: str,
) -> dict[str, Any]:
    data = dict(payload or {})
    data.setdefault("schema_version", KERNEL_SCHEMA_VERSION)
    data.setdefault("artifact_type", str(artifact_type or "kernel_artifact"))
    return data


def read_kernel_json(path: Path, *, artifact_type: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    kind = str(artifact_type or payload.get("artifact_type", "") or path.stem)
    return normalize_kernel_payload(payload, artifact_type=kind)


def write_kernel_json(path: Path, payload: dict[str, Any], *, artifact_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(
        json.dumps(
            normalize_kernel_payload(payload, artifact_type=artifact_type),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)
