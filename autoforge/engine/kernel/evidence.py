from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoforge.engine.kernel.inspector import inspect_kernel_run, resolve_kernel_run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path, *, exclude_roots: tuple[Path, ...] = ()) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        resolved_excludes = tuple(root.resolve() for root in exclude_roots)

        def _ignore(directory: str, names: list[str]) -> list[str]:
            if not resolved_excludes:
                return []
            current = Path(directory).resolve()
            ignored: list[str] = []
            for name in names:
                child = (current / name).resolve()
                for root in resolved_excludes:
                    try:
                        child.relative_to(root)
                        ignored.append(name)
                        break
                    except ValueError:
                        continue
            return ignored

        shutil.copytree(src, dst, ignore=_ignore)
    else:
        shutil.copy2(src, dst)
    return True


@dataclass(frozen=True)
class EvidencePack:
    run_dir: Path
    output_dir: Path
    manifest_path: Path
    summary_path: Path
    copied_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "summary_path": str(self.summary_path),
            "copied_files": list(self.copied_files),
        }


def export_evidence_pack(
    path: Path,
    *,
    run_id: str | None = None,
    out_dir: Path | None = None,
    include_workspace_artifacts: bool = True,
) -> EvidencePack:
    inspection = inspect_kernel_run(path, run_id=run_id, tail=50)
    run_dir = inspection.run_dir
    manifest = inspection.manifest
    profile = str(manifest.get("profile", "") or "")
    default_root = run_dir.parent.parent.parent / "research" / "evidence_pack" / run_dir.name
    output_dir = (out_dir or default_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    kernel_files = [
        "manifest.json",
        "execution_plan.json",
        "contract_verdict.json",
        "harness_judge.json",
        "profile.json",
        "contracts.json",
        "environment.lock.json",
        "artifact_manifest.json",
        "events.jsonl",
        "inbox.json",
    ]
    for name in kernel_files:
        if _copy_if_exists(run_dir / name, output_dir / "kernel" / name):
            copied.append(f"kernel/{name}")
    run_store = run_dir.parent.parent / "run_store.sqlite3"
    if _copy_if_exists(run_store, output_dir / "kernel" / "run_store.sqlite3"):
        copied.append("kernel/run_store.sqlite3")

    if include_workspace_artifacts:
        for artifact in inspection.artifact_manifest.get("artifacts", []):
            absolute = Path(str(artifact.get("absolute_path", "") or "")).resolve()
            kind = str(artifact.get("kind", "") or "artifact")
            label = absolute.name or kind
            dst = output_dir / "artifacts" / kind / label
            try:
                if _copy_if_exists(absolute, dst, exclude_roots=(output_dir,)):
                    copied.append(f"artifacts/{kind}/{label}")
            except Exception:
                continue

    summary_path = output_dir / "summary.json"
    summary_payload = inspection.to_dict()
    summary_payload["profile"] = profile
    summary_payload["evidence_pack_dir"] = str(output_dir)
    _write_json(summary_path, summary_payload)

    manifest_path = output_dir / "evidence_pack_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_id": manifest.get("run_id", ""),
            "thread_id": manifest.get("thread_id", ""),
            "profile": profile,
            "status": manifest.get("status", ""),
            "source_run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "copied_files": copied,
            "include_workspace_artifacts": bool(include_workspace_artifacts),
        },
    )

    return EvidencePack(
        run_dir=run_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        copied_files=tuple(copied),
    )


def ensure_research_evidence_pack(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
) -> EvidencePack:
    return export_evidence_pack(
        run_dir,
        out_dir=out_dir,
        include_workspace_artifacts=True,
    )
