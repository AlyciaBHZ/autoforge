"""Artifact store for run outputs.

Artifacts are run-local files written under:
  <project_dir>/.autoforge/artifacts/<run_id>/
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _safe_relpath(rel_path: str) -> str:
    rel = (rel_path or "").replace("\\", "/").lstrip("/")
    # Keep it simple; forbid parent traversal.
    if ".." in Path(rel).parts:
        raise ValueError("Path traversal not allowed for artifact path")
    return rel


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    bytes: int
    sha256: str
    media_type: str = "application/octet-stream"


class ArtifactStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel_path: str) -> Path:
        rel = _safe_relpath(rel_path)
        full = (self.base_dir / rel).resolve()
        base = self.base_dir.resolve()
        if not full.is_relative_to(base):
            raise ValueError("Artifact path escapes base_dir")
        return full

    def write_text(self, rel_path: str, content: str, *, encoding: str = "utf-8") -> ArtifactRef:
        path = self._resolve(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode(encoding, errors="replace")
        path.write_bytes(data)
        return ArtifactRef(
            path=str(path.relative_to(self.base_dir)),
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            media_type="text/plain",
        )

    def write_json(self, rel_path: str, obj: Any) -> ArtifactRef:
        payload = json.dumps(obj, indent=2, ensure_ascii=False)
        return self.write_text(rel_path, payload, encoding="utf-8")

    def write_bytes(self, rel_path: str, data: bytes, *, media_type: str = "application/octet-stream") -> ArtifactRef:
        path = self._resolve(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return ArtifactRef(
            path=str(path.relative_to(self.base_dir)),
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            media_type=media_type,
        )


class NullArtifactStore:
    """No-op artifact store used when artifacts are disabled."""

    _ZERO_SHA256 = "0" * 64

    def write_text(self, rel_path: str, content: str, *, encoding: str = "utf-8") -> ArtifactRef:  # noqa: ARG002
        return ArtifactRef(path="", bytes=0, sha256=self._ZERO_SHA256, media_type="text/plain")

    def write_json(self, rel_path: str, obj: Any) -> ArtifactRef:  # noqa: ARG002
        return ArtifactRef(path="", bytes=0, sha256=self._ZERO_SHA256, media_type="application/json")

    def write_bytes(self, rel_path: str, data: bytes, *, media_type: str = "application/octet-stream") -> ArtifactRef:  # noqa: ARG002
        return ArtifactRef(path="", bytes=0, sha256=self._ZERO_SHA256, media_type=media_type)
