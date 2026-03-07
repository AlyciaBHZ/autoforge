from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

from autoforge.cli.app import _build_config_overrides, build_parser
from autoforge.engine.config import ForgeConfig
from autoforge.engine.ui_harness import (
    normalize_design_context_refs,
    write_ui_handoff,
    write_ui_harness_artifacts,
    write_ui_judge_report,
)


def _make_local_tmp_dir() -> Path:
    path = Path.cwd() / ".tmp_ui_harness_tests" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_ui_harness_cli_and_config_normalization():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--ui-harness",
            "--design-ref",
            "https://figma.example/file/1",
            "--design-ref",
            "brand-guide.pdf",
            "generate",
            "Build a dashboard",
        ]
    )
    overrides = _build_config_overrides(args)

    assert overrides["ui_harness_enabled"] is True
    assert overrides["design_context_refs"] == [
        "https://figma.example/file/1",
        "brand-guide.pdf",
    ]

    config = ForgeConfig(
        ui_harness_enabled=True,
        design_context_refs=["brand-guide.pdf", "BRAND-GUIDE.PDF", "shots.zip"],
    )
    assert config.ui_harness_enabled is True
    assert config.design_context_refs == ["brand-guide.pdf", "shots.zip"]
    assert normalize_design_context_refs("figma, screenshots.zip; brand.pdf") == [
        "figma",
        "screenshots.zip",
        "brand.pdf",
    ]


def test_ui_harness_artifacts_materialize_repo_local_files():
    tmp_path = _make_local_tmp_dir()
    try:
        project_dir = tmp_path / "workspace" / "demo-ui"
        project_dir.mkdir(parents=True, exist_ok=True)
        registered: list[str] = []
        kernel_session = SimpleNamespace(register_artifact=lambda kind, path, required=False: registered.append(kind))
        config = ForgeConfig(
            project_root=tmp_path,
            workspace_dir=project_dir.parent,
            run_id="ui-run-1",
            ui_harness_enabled=True,
            design_context_refs=["https://figma.example/file/1"],
        )
        spec = {
            "project_name": "demo-ui",
            "project_type": "web-app",
            "tech_stack": {"frontend": "React"},
            "modules": [
                {"name": "LandingPage", "description": "Marketing homepage"},
                {"name": "HeroBanner", "description": "Top section"},
            ],
        }

        artifacts = write_ui_harness_artifacts(
            project_dir=project_dir,
            config=config,
            requirement="Build a marketing landing page",
            spec=spec,
            kernel_session=kernel_session,
        )

        assert artifacts.active is True
        assert artifacts.design_brief_path is not None and artifacts.design_brief_path.is_file()
        assert artifacts.style_guide_path is not None and artifacts.style_guide_path.is_file()
        assert artifacts.design_tokens_path is not None and artifacts.design_tokens_path.is_file()
        assert artifacts.component_inventory_path is not None and artifacts.component_inventory_path.is_file()

        design_brief = json.loads(artifacts.design_brief_path.read_text(encoding="utf-8"))
        style_guide = json.loads(artifacts.style_guide_path.read_text(encoding="utf-8"))
        design_tokens = json.loads(artifacts.design_tokens_path.read_text(encoding="utf-8"))
        inventory = json.loads(artifacts.component_inventory_path.read_text(encoding="utf-8"))

        assert design_brief["frontend_target"] is True
        assert design_brief["design_context_refs"] == ["https://figma.example/file/1"]
        assert style_guide["artifact_type"] == "style_guide"
        assert design_tokens["artifact_type"] == "design_tokens"
        assert "--color-accent" in design_tokens["css_variables"]
        assert inventory["page_count"] == 1
        assert set(registered) == {"design_brief", "style_guide", "design_tokens", "component_inventory"}
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_ui_harness_judge_emits_design_outcomes():
    tmp_path = _make_local_tmp_dir()
    try:
        project_dir = tmp_path / "workspace" / "demo-ui"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "src").mkdir(parents=True, exist_ok=True)
        (project_dir / "src" / "app.css").write_text(
            ":root { --color-accent: #2563EB; --space-4: 1rem; }\n"
            ".shell { color: var(--color-accent); }\n"
            "@media (min-width: 768px) { .shell { display: grid; } }\n",
            encoding="utf-8",
        )
        (project_dir / "src" / "App.tsx").write_text(
            "export function App() {\n"
            "  return <main aria-label=\"Dashboard\"><img alt=\"logo\" src=\"/logo.png\" /></main>;\n"
            "}\n",
            encoding="utf-8",
        )
        config = ForgeConfig(
            project_root=tmp_path,
            workspace_dir=project_dir.parent,
            run_id="ui-run-2",
            ui_harness_enabled=True,
        )
        spec = {
            "project_name": "demo-ui",
            "project_type": "web-app",
            "tech_stack": {"frontend": "React"},
            "modules": [{"name": "DashboardPage", "description": "Analytics dashboard"}],
        }

        write_ui_harness_artifacts(
            project_dir=project_dir,
            config=config,
            requirement="Build a responsive analytics dashboard",
            spec=spec,
            kernel_session=None,
        )
        artifacts, outcomes = write_ui_judge_report(
            project_dir=project_dir,
            config=config,
            spec=spec,
            requirement="Build a responsive analytics dashboard",
            verify_passed=True,
            kernel_session=None,
        )

        assert artifacts.ui_judge_report_path is not None and artifacts.ui_judge_report_path.is_file()
        report = json.loads(artifacts.ui_judge_report_path.read_text(encoding="utf-8"))

        assert report["passed"] is True
        assert "responsive_layout_pass" in outcomes
        assert "design_consistency_pass" in outcomes
        assert "a11y_contract_pass" in outcomes
        assert report["checks"]["frontend_source_present"] is True
        assert report["checks"]["responsive_hints_present"] is True

        handoff = write_ui_handoff(
            project_dir=project_dir,
            config=config,
            spec=spec,
            kernel_session=None,
        )
        assert handoff.ui_handoff_path is not None and handoff.ui_handoff_path.is_file()
        assert "UI Handoff" in handoff.ui_handoff_path.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
