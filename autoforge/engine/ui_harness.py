from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoforge.engine.kernel.schema import write_kernel_json


_UI_KEYWORDS = (
    "ui",
    "ux",
    "web",
    "website",
    "web app",
    "landing page",
    "dashboard",
    "frontend",
    "admin panel",
    "mobile app",
    "screen",
    "page",
)

_FRONTEND_FILE_SUFFIXES = {
    ".tsx",
    ".jsx",
    ".ts",
    ".js",
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",
    ".svelte",
}


@dataclass(frozen=True)
class UIHarnessArtifacts:
    active: bool
    design_brief_path: Path | None = None
    style_guide_path: Path | None = None
    design_tokens_path: Path | None = None
    component_inventory_path: Path | None = None
    ui_judge_report_path: Path | None = None
    ui_handoff_path: Path | None = None
    metadata: dict[str, Any] | None = None


def normalize_design_context_refs(raw: Any) -> list[str]:
    if raw is None:
        return []
    values: list[str] = []
    if isinstance(raw, str):
        chunks = re.split(r"[\n,;]+", raw)
        values.extend(chunks)
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            if isinstance(item, str):
                values.extend(re.split(r"[\n,;]+", item))
            elif item is not None:
                values.append(str(item))
    else:
        values.append(str(raw))
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        token = str(item or "").strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(token)
    return normalized


def _extract_project_type(spec: dict[str, Any]) -> str:
    return str(spec.get("project_type", "") or "").strip().lower()


def _extract_frontend_stack(spec: dict[str, Any]) -> str:
    tech_stack = spec.get("tech_stack", {})
    if isinstance(tech_stack, dict):
        return str(tech_stack.get("frontend", "") or tech_stack.get("mobile", "") or "").strip()
    return ""


def is_frontend_target(*, requirement: str, spec: dict[str, Any], mobile_target: str = "none") -> bool:
    if str(mobile_target or "").strip().lower() not in {"", "none"}:
        return True
    project_type = _extract_project_type(spec)
    if project_type in {"web-app", "static-site", "mobile-scaffold", "desktop-scaffold"}:
        return True
    frontend_stack = _extract_frontend_stack(spec)
    if frontend_stack:
        return True
    haystack = " ".join(
        [
            str(requirement or ""),
            str(spec.get("project_name", "") or ""),
            json.dumps(spec.get("tech_stack", {}), ensure_ascii=False),
            json.dumps(spec.get("modules", []), ensure_ascii=False),
        ]
    ).lower()
    return any(keyword in haystack for keyword in _UI_KEYWORDS)


def ui_harness_enabled(*, config: Any, requirement: str, spec: dict[str, Any]) -> bool:
    if bool(getattr(config, "ui_harness_enabled", False)):
        return True
    if normalize_design_context_refs(getattr(config, "design_context_refs", [])):
        return True
    return is_frontend_target(
        requirement=requirement,
        spec=spec,
        mobile_target=str(getattr(config, "mobile_target", "none") or "none"),
    )


def _ui_root(project_dir: Path) -> Path:
    return project_dir / ".autoforge" / "ui"


def _style_direction(requirement: str, spec: dict[str, Any], mobile_target: str) -> dict[str, Any]:
    text = " ".join(
        [
            str(requirement or ""),
            json.dumps(spec.get("tech_stack", {}), ensure_ascii=False),
            json.dumps(spec.get("modules", []), ensure_ascii=False),
        ]
    ).lower()
    if mobile_target not in {"", "none"} or "mobile" in text:
        return {
            "preset": "touch_product",
            "tone": "calm, tactile, high-clarity",
            "density": "comfortable",
            "palette": "deep-ink base with bright accent",
            "typography": {"heading": "Space Grotesk", "body": "Inter"},
            "motion": "short and direct transitions",
        }
    if "dashboard" in text or "admin" in text or "analytics" in text:
        return {
            "preset": "signal_dashboard",
            "tone": "precise, restrained, data-first",
            "density": "compact",
            "palette": "neutral slate base with one strong signal accent",
            "typography": {"heading": "IBM Plex Sans", "body": "IBM Plex Sans"},
            "motion": "minimal, stateful transitions",
        }
    if "landing" in text or "marketing" in text or "brand" in text:
        return {
            "preset": "expressive_brand",
            "tone": "bold, editorial, memorable",
            "density": "airy",
            "palette": "warm neutrals with one saturated accent",
            "typography": {"heading": "Fraunces", "body": "Manrope"},
            "motion": "staggered reveals and richer transitions",
        }
    return {
        "preset": "product_default",
        "tone": "confident, clean, modern",
        "density": "balanced",
        "palette": "soft neutral base with disciplined accent use",
        "typography": {"heading": "Sora", "body": "Inter"},
        "motion": "subtle state transitions",
    }


def _design_tokens(style_guide: dict[str, Any]) -> dict[str, Any]:
    preset = str(style_guide.get("preset", "") or "product_default")
    palette_map = {
        "touch_product": {
            "--color-bg-canvas": "#0F172A",
            "--color-bg-surface": "#111827",
            "--color-fg-primary": "#F8FAFC",
            "--color-fg-muted": "#CBD5E1",
            "--color-accent": "#22C55E",
            "--color-accent-strong": "#16A34A",
        },
        "signal_dashboard": {
            "--color-bg-canvas": "#F8FAFC",
            "--color-bg-surface": "#FFFFFF",
            "--color-fg-primary": "#0F172A",
            "--color-fg-muted": "#475569",
            "--color-accent": "#0EA5E9",
            "--color-accent-strong": "#0284C7",
        },
        "expressive_brand": {
            "--color-bg-canvas": "#FFF7ED",
            "--color-bg-surface": "#FFFFFF",
            "--color-fg-primary": "#1C1917",
            "--color-fg-muted": "#57534E",
            "--color-accent": "#EA580C",
            "--color-accent-strong": "#C2410C",
        },
    }
    css_variables = palette_map.get(
        preset,
        {
            "--color-bg-canvas": "#F8FAFC",
            "--color-bg-surface": "#FFFFFF",
            "--color-fg-primary": "#0F172A",
            "--color-fg-muted": "#475569",
            "--color-accent": "#2563EB",
            "--color-accent-strong": "#1D4ED8",
        },
    )
    css_variables.update(
        {
            "--space-2": "0.5rem",
            "--space-4": "1rem",
            "--space-6": "1.5rem",
            "--space-8": "2rem",
            "--radius-sm": "0.5rem",
            "--radius-md": "1rem",
            "--radius-lg": "1.5rem",
            "--shadow-soft": "0 12px 32px rgba(15, 23, 42, 0.12)",
            "--font-heading": str(style_guide.get("typography", {}).get("heading", "Sora")),
            "--font-body": str(style_guide.get("typography", {}).get("body", "Inter")),
        }
    )
    return {
        "schema_version": 1,
        "artifact_type": "design_tokens",
        "css_variables": css_variables,
        "token_groups": {
            "color": {k: v for k, v in css_variables.items() if k.startswith("--color-")},
            "space": {k: v for k, v in css_variables.items() if k.startswith("--space-")},
            "radius": {k: v for k, v in css_variables.items() if k.startswith("--radius-")},
            "shadow": {k: v for k, v in css_variables.items() if k.startswith("--shadow-")},
            "font": {k: v for k, v in css_variables.items() if k.startswith("--font-")},
        },
    }


def _component_inventory(spec: dict[str, Any]) -> dict[str, Any]:
    modules = list(spec.get("modules", []) or [])
    pages: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    for index, module in enumerate(modules, start=1):
        if isinstance(module, dict):
            name = str(module.get("name", "") or f"module_{index}")
            description = str(module.get("description", "") or "")
            files = list(module.get("files", []) or [])
        else:
            name = str(module or f"module_{index}")
            description = ""
            files = []
        entry = {
            "id": f"module_{index:02d}",
            "name": name,
            "description": description,
            "files": files,
        }
        lowered = name.lower()
        if any(token in lowered for token in ("page", "screen", "dashboard", "home", "landing", "profile")):
            pages.append(entry)
        else:
            components.append(entry)
    return {
        "schema_version": 1,
        "artifact_type": "component_inventory",
        "page_count": len(pages),
        "component_count": len(components),
        "pages": pages,
        "components": components,
    }


def write_ui_harness_artifacts(
    *,
    project_dir: Path,
    config: Any,
    requirement: str,
    spec: dict[str, Any],
    kernel_session: Any | None = None,
) -> UIHarnessArtifacts:
    active = ui_harness_enabled(config=config, requirement=requirement, spec=spec)
    if not active:
        return UIHarnessArtifacts(active=False, metadata={"active": False})

    ui_root = _ui_root(project_dir)
    ui_root.mkdir(parents=True, exist_ok=True)
    refs = normalize_design_context_refs(getattr(config, "design_context_refs", []))
    style_guide = _style_direction(
        requirement=requirement,
        spec=spec,
        mobile_target=str(getattr(config, "mobile_target", "none") or "none"),
    )
    frontend_target = is_frontend_target(
        requirement=requirement,
        spec=spec,
        mobile_target=str(getattr(config, "mobile_target", "none") or "none"),
    )
    design_brief = {
        "schema_version": 1,
        "artifact_type": "design_brief",
        "run_id": str(getattr(config, "run_id", "") or ""),
        "project_name": str(spec.get("project_name", project_dir.name) or project_dir.name),
        "frontend_target": frontend_target,
        "requirement": str(requirement or ""),
        "design_context_refs": refs,
        "mobile_target": str(getattr(config, "mobile_target", "none") or "none"),
        "style_direction": style_guide,
        "generated_at": time.time(),
    }
    style_payload = {
        "schema_version": 1,
        "artifact_type": "style_guide",
        "run_id": str(getattr(config, "run_id", "") or ""),
        "project_name": design_brief["project_name"],
        "preset": style_guide["preset"],
        "tone": style_guide["tone"],
        "density": style_guide["density"],
        "palette": style_guide["palette"],
        "typography": dict(style_guide["typography"]),
        "motion": style_guide["motion"],
        "layout_rules": [
            "Prefer strong section hierarchy and visible whitespace rhythm",
            "Use one accent color deliberately instead of many competing accents",
            "Favor component reuse over one-off decorative divergence",
        ],
        "generated_at": time.time(),
    }
    tokens_payload = _design_tokens(style_payload)
    tokens_payload.update(
        {
            "run_id": str(getattr(config, "run_id", "") or ""),
            "project_name": design_brief["project_name"],
            "generated_at": time.time(),
        }
    )
    inventory_payload = _component_inventory(spec)
    inventory_payload.update(
        {
            "run_id": str(getattr(config, "run_id", "") or ""),
            "project_name": design_brief["project_name"],
            "generated_at": time.time(),
        }
    )

    design_brief_path = ui_root / "design_brief.json"
    style_guide_path = ui_root / "style_guide.json"
    design_tokens_path = ui_root / "design_tokens.json"
    component_inventory_path = ui_root / "component_inventory.json"

    write_kernel_json(design_brief_path, design_brief, artifact_type="design_brief")
    write_kernel_json(style_guide_path, style_payload, artifact_type="style_guide")
    write_kernel_json(design_tokens_path, tokens_payload, artifact_type="design_tokens")
    write_kernel_json(component_inventory_path, inventory_payload, artifact_type="component_inventory")

    if kernel_session is not None:
        kernel_session.register_artifact("design_brief", design_brief_path, required=False)
        kernel_session.register_artifact("style_guide", style_guide_path, required=False)
        kernel_session.register_artifact("design_tokens", design_tokens_path, required=False)
        kernel_session.register_artifact("component_inventory", component_inventory_path, required=False)

    metadata = {
        "active": True,
        "frontend_target": frontend_target,
        "design_context_ref_count": len(refs),
        "style_preset": style_guide["preset"],
        "component_count": int(inventory_payload["component_count"]),
        "page_count": int(inventory_payload["page_count"]),
    }
    return UIHarnessArtifacts(
        active=True,
        design_brief_path=design_brief_path,
        style_guide_path=style_guide_path,
        design_tokens_path=design_tokens_path,
        component_inventory_path=component_inventory_path,
        metadata=metadata,
    )


def _iter_frontend_files(project_dir: Path) -> list[Path]:
    results: list[Path] = []
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {".autoforge", ".git", "node_modules", "__pycache__", ".venv", "venv"} for part in path.parts):
            continue
        if path.suffix.lower() in _FRONTEND_FILE_SUFFIXES:
            results.append(path)
    return results


def write_ui_judge_report(
    *,
    project_dir: Path,
    config: Any,
    spec: dict[str, Any],
    requirement: str,
    verify_passed: bool,
    kernel_session: Any | None = None,
) -> tuple[UIHarnessArtifacts, tuple[str, ...]]:
    ui_root = _ui_root(project_dir)
    ui_root.mkdir(parents=True, exist_ok=True)
    frontend_target = ui_harness_enabled(config=config, requirement=requirement, spec=spec)
    frontend_files = _iter_frontend_files(project_dir)
    has_frontend_source = bool(frontend_files)
    sample = ""
    for path in frontend_files[:12]:
        try:
            sample += "\n" + path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

    design_tokens_path = ui_root / "design_tokens.json"
    has_design_tokens = design_tokens_path.is_file()
    css_variables_present = "var(--" in sample or "--color-" in sample or "--space-" in sample
    accessibility_hints = any(token in sample for token in ("aria-", "role=", "alt=", "<label", "htmlFor"))
    responsive_hints = any(token in sample for token in ("@media", "sm:", "md:", "lg:", "xl:", "clamp(", "minmax(", "grid-template-columns"))

    checks = {
        "tests_passed": bool(verify_passed),
        "frontend_source_present": has_frontend_source,
        "design_tokens_present": has_design_tokens,
        "design_tokens_consumed": css_variables_present,
        "accessibility_hints_present": accessibility_hints,
        "responsive_hints_present": responsive_hints,
    }
    score = sum(1 for value in checks.values() if value) / max(len(checks), 1)
    outcomes: list[str] = []
    if responsive_hints:
        outcomes.append("responsive_layout_pass")
    if has_design_tokens and css_variables_present:
        outcomes.append("design_consistency_pass")
    if accessibility_hints:
        outcomes.append("a11y_contract_pass")

    report = {
        "schema_version": 1,
        "artifact_type": "ui_judge_report",
        "run_id": str(getattr(config, "run_id", "") or ""),
        "project_name": str(spec.get("project_name", project_dir.name) or project_dir.name),
        "ui_harness_active": frontend_target,
        "passed": bool(frontend_target and has_frontend_source and verify_passed and score >= 0.5),
        "score": round(score, 4),
        "frontend_file_count": len(frontend_files),
        "sample_files": [str(path.relative_to(project_dir)).replace("\\", "/") for path in frontend_files[:25]],
        "checks": checks,
        "declared_outcomes": outcomes,
        "generated_at": time.time(),
    }
    report_path = ui_root / "ui_judge_report.json"
    write_kernel_json(report_path, report, artifact_type="ui_judge_report")
    if kernel_session is not None:
        kernel_session.register_artifact("ui_judge_report", report_path, required=False)
    artifacts = UIHarnessArtifacts(
        active=frontend_target,
        ui_judge_report_path=report_path,
        metadata={"judge_score": report["score"], "frontend_file_count": len(frontend_files)},
    )
    return artifacts, tuple(outcomes)


def write_ui_handoff(
    *,
    project_dir: Path,
    config: Any,
    spec: dict[str, Any],
    kernel_session: Any | None = None,
) -> UIHarnessArtifacts:
    ui_root = _ui_root(project_dir)
    ui_root.mkdir(parents=True, exist_ok=True)
    handoff_path = ui_root / "ui_handoff.md"
    refs = normalize_design_context_refs(getattr(config, "design_context_refs", []))
    lines = [
        "# UI Handoff",
        "",
        f"- Run ID: {str(getattr(config, 'run_id', '') or '')}",
        f"- Project: {str(spec.get('project_name', project_dir.name) or project_dir.name)}",
        f"- Design refs: {len(refs)}",
        "",
        "## Repo-local UI artifacts",
        "- design_brief.json",
        "- style_guide.json",
        "- design_tokens.json",
        "- component_inventory.json",
        "- ui_judge_report.json",
        "",
        "## Follow-up",
        "- Add Storybook stories for key components/pages",
        "- Add Playwright or Chromatic visual baselines",
        "- Promote accessibility checks from heuristic hints to automated browser assertions",
        "",
    ]
    handoff_path.write_text("\n".join(lines), encoding="utf-8")
    if kernel_session is not None:
        kernel_session.register_artifact("ui_handoff", handoff_path, required=False)
    return UIHarnessArtifacts(
        active=True,
        ui_handoff_path=handoff_path,
        metadata={"design_context_ref_count": len(refs)},
    )
