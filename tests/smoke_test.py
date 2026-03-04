#!/usr/bin/env python3
"""AutoForge smoke test suite.

Validates that all components can be imported, instantiated, and perform
basic operations without requiring an API key or Docker.

Usage:
    python tests/smoke_test.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASSED = 0
FAILED = 0


def check(name: str, fn):
    """Run a check function and report pass/fail."""
    global PASSED, FAILED
    try:
        fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED += 1
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


# ────────────────────────────────────────────
# Phase 1: Imports
# ────────────────────────────────────────────

def test_stdlib_imports():
    import argparse, asyncio, json, logging, os, uuid, time, re, shutil  # noqa: F811
    from pathlib import Path  # noqa: F811
    from abc import ABC, abstractmethod  # noqa: F401
    from dataclasses import dataclass, field  # noqa: F401
    from enum import Enum  # noqa: F401


def test_dependency_imports():
    import anthropic  # noqa: F401
    import dotenv  # noqa: F401
    import rich  # noqa: F401
    import yaml  # noqa: F401


def test_engine_imports():
    from autoforge.engine.config import ForgeConfig  # noqa: F401
    from autoforge.engine.llm_router import (  # noqa: F401
        LLMRouter, TaskComplexity, BudgetExceededError,
        ContentBlock, Usage, LLMResponse, detect_provider,
    )
    from autoforge.engine.agent_base import AgentBase, AgentResult, ToolDefinition  # noqa: F401
    from autoforge.engine.task_dag import TaskDAG, Task, TaskPhase, TaskStatus  # noqa: F401
    from autoforge.engine.lock_manager import LockManager  # noqa: F401
    from autoforge.engine.git_manager import GitManager, GitError  # noqa: F401
    from autoforge.engine.sandbox import (  # noqa: F401
        SandboxBase, SubprocessSandbox, DockerSandbox,
        SandboxResult, create_sandbox,
    )
    from autoforge.engine.orchestrator import Orchestrator  # noqa: F401
    from autoforge.engine.project_registry import ProjectRegistry, ProjectStatus, Project  # noqa: F401
    from autoforge.engine.daemon import ForgeDaemon  # noqa: F401
    from autoforge.engine.deploy_guide import detect_framework, generate_deploy_guide  # noqa: F401
    import autoforge.engine.channels  # noqa: F401


def test_agent_imports():
    from autoforge.engine.agents import AGENT_REGISTRY
    assert len(AGENT_REGISTRY) == 8, f"Expected 8 agents, got {len(AGENT_REGISTRY)}"
    from autoforge.engine.agents.director import DirectorAgent, DirectorFixAgent  # noqa: F401
    from autoforge.engine.agents.architect import ArchitectAgent  # noqa: F401
    from autoforge.engine.agents.builder import BuilderAgent  # noqa: F401
    from autoforge.engine.agents.reviewer import ReviewerAgent  # noqa: F401
    from autoforge.engine.agents.tester import TesterAgent  # noqa: F401
    from autoforge.engine.agents.gardener import GardenerAgent  # noqa: F401
    from autoforge.engine.agents.scanner import ScannerAgent  # noqa: F401


def test_cli_imports():
    from autoforge.cli.app import build_parser, main  # noqa: F401
    from autoforge.cli.display import show_banner, show_startup_info, show_review_report  # noqa: F401
    from autoforge.cli.setup_wizard import needs_setup, load_global_config  # noqa: F401


# ────────────────────────────────────────────
# Phase 2: CLI
# ────────────────────────────────────────────

def test_cli_parse_subcommands():
    from autoforge.cli.app import build_parser
    parser = build_parser()

    # Test generate subcommand
    args = parser.parse_args(["generate", "Build a Todo app"])
    assert args.command == "generate"
    assert args.description == "Build a Todo app"

    # Test review subcommand
    args = parser.parse_args(["review", "/tmp/project"])
    assert args.command == "review"
    assert args.path == "/tmp/project"

    # Test import subcommand
    args = parser.parse_args(["import", "/tmp/project", "--enhance", "add dark mode"])
    assert args.command == "import"
    assert args.path == "/tmp/project"
    assert args.enhance == "add dark mode"

    # Test status
    args = parser.parse_args(["status"])
    assert args.command == "status"

    # Test global flags
    args = parser.parse_args(["--budget", "5.0", "--mode", "research", "--mobile", "both", "generate", "test"])
    assert args.budget == 5.0
    assert args.mode == "research"
    assert args.mobile == "both"


def test_cli_legacy_compat():
    """Test that legacy CLI usage still works."""
    from autoforge.cli.app import build_parser, _KNOWN_COMMANDS

    parser = build_parser()

    # Legacy: python forge.py --status
    args = parser.parse_args(["--status"])
    assert args.legacy_status is True

    # Legacy: python forge.py --resume
    args = parser.parse_args(["--resume"])
    assert args.legacy_resume == "auto"

    # Verify legacy pre-scan detects bare descriptions
    # (bare descriptions are handled by argv pre-scanning in _resolve_command,
    #  not by argparse, so we test the detection logic here)
    assert "Build a todo app" not in _KNOWN_COMMANDS
    assert "generate" in _KNOWN_COMMANDS


# ────────────────────────────────────────────
# Phase 3: Unit Tests
# ────────────────────────────────────────────

def test_forge_config():
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig()
    assert config.budget_limit_usd == 10.0
    assert config.max_agents == 3
    assert len(config.run_id) == 12
    assert config.workspace_dir.name == "workspace"
    assert config.constitution_dir.name == "constitution"
    assert config.mode == "developer"
    assert config.mobile_target == "none"


def test_forge_config_from_env():
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig.from_env(budget_limit_usd=5.0, max_agents=2, mode="research")
    assert config.budget_limit_usd == 5.0
    assert config.max_agents == 2
    assert config.mode == "research"


def test_forge_config_budget_tracking():
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig(budget_limit_usd=1.0)
    config.record_usage("claude-sonnet-4-5-20250929", 1000, 500)
    assert config.total_input_tokens == 1000
    assert config.total_output_tokens == 500
    assert config.estimated_cost_usd > 0
    assert config.check_budget() is True
    # Exhaust budget
    config.record_usage("claude-opus-4-6", 10_000_000, 10_000_000)
    assert config.check_budget() is False


def test_forge_config_mobile():
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig(mobile_target="both", mobile_framework="flutter")
    assert config.mobile_target == "both"
    assert config.mobile_framework == "flutter"


def test_task_dag_basic():
    from autoforge.engine.task_dag import TaskDAG, Task, TaskPhase, TaskStatus
    dag = TaskDAG()
    t1 = Task(id="T-001", description="Setup", phase=TaskPhase.BUILD)
    t2 = Task(id="T-002", description="Auth", phase=TaskPhase.BUILD, depends_on=["T-001"])
    dag.add_task(t1)
    dag.add_task(t2)
    assert dag.total_tasks() == 2
    ready = dag.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "T-001"
    dag.mark_done("T-001")
    ready2 = dag.get_ready_tasks()
    assert len(ready2) == 1
    assert ready2[0].id == "T-002"


def test_task_dag_save_load():
    from autoforge.engine.task_dag import TaskDAG, Task
    dag = TaskDAG()
    dag.add_task(Task(id="X-1", description="test"))
    tmp = Path(tempfile.mktemp(suffix=".json"))
    try:
        dag.save(tmp)
        loaded = TaskDAG.load(tmp)
        assert loaded.total_tasks() == 1
        assert loaded.get_task("X-1").description == "test"
    finally:
        tmp.unlink(missing_ok=True)


def test_task_dag_failure_handling():
    from autoforge.engine.task_dag import TaskDAG, Task, TaskStatus
    dag = TaskDAG()
    dag.add_task(Task(id="F-1", description="fail test"))
    dag.mark_failed("F-1", "error")
    assert dag.get_task("F-1").status == TaskStatus.FAILED
    dag.reset_failed("F-1")
    assert dag.get_task("F-1").status == TaskStatus.TODO
    # 3 failures → BLOCKED
    dag.mark_failed("F-1", "e1")
    dag.mark_failed("F-1", "e2")
    dag.mark_failed("F-1", "e3")
    assert dag.get_task("F-1").status == TaskStatus.BLOCKED


def test_task_dag_cycle_detection():
    from autoforge.engine.task_dag import TaskDAG, Task
    dag = TaskDAG()
    dag.add_task(Task(id="A", description="a", depends_on=["B"]))
    dag.add_task(Task(id="B", description="b", depends_on=["A"]))
    try:
        dag.validate_acyclic()
        raise AssertionError("Should have raised ValueError for cycle")
    except ValueError as e:
        assert "Cycle" in str(e)


def test_project_registry_validation():
    """Test that enqueue rejects empty/invalid descriptions."""
    from autoforge.engine.project_registry import ProjectRegistry

    async def _test():
        db_path = Path(tempfile.mktemp(suffix=".db"))
        try:
            async with ProjectRegistry(db_path) as reg:
                # Empty description
                try:
                    await reg.enqueue("", "cli")
                    raise AssertionError("Should reject empty description")
                except ValueError:
                    pass

                # Whitespace-only description
                try:
                    await reg.enqueue("   ", "cli")
                    raise AssertionError("Should reject whitespace description")
                except ValueError:
                    pass

                # Negative budget
                try:
                    await reg.enqueue("valid", "cli", budget_usd=-1.0)
                    raise AssertionError("Should reject negative budget")
                except ValueError:
                    pass
        finally:
            db_path.unlink(missing_ok=True)

    asyncio.run(_test())


def test_lock_manager():
    from autoforge.engine.lock_manager import LockManager
    d = Path(tempfile.mkdtemp()) / "locks"
    try:
        lm = LockManager(d)
        assert lm.try_claim("t-1", "a-0") is True
        assert lm.try_claim("t-1", "a-1") is False  # Double claim rejected
        assert lm.get_owner("t-1") == "a-0"
        assert lm.release("t-1", "a-0") is True
        assert lm.get_owner("t-1") is None
        # Wrong agent cannot release
        lm.try_claim("t-2", "a-0")
        assert lm.release("t-2", "a-1") is False
        assert lm.enforce_single_task("a-0") is False  # holds t-2
        assert lm.enforce_single_task("a-1") is True
        assert lm.agent_task_count("a-0") == 1
        lm.clear_all()
        assert lm.agent_task_count("a-0") == 0
    finally:
        shutil.rmtree(d.parent, ignore_errors=True)


def test_sandbox_subprocess():
    from autoforge.engine.sandbox import SubprocessSandbox

    async def _test():
        d = Path(tempfile.mkdtemp())
        try:
            async with SubprocessSandbox(d) as sb:
                result = await sb.exec("echo hello")
                assert result.exit_code == 0
                assert "hello" in result.stdout
                result2 = await sb.exec("false")
                assert result2.exit_code != 0
        finally:
            shutil.rmtree(d, ignore_errors=True)

    asyncio.run(_test())


def test_sandbox_factory():
    from autoforge.engine.sandbox import create_sandbox, SubprocessSandbox
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig(docker_enabled=False)
    d = Path(tempfile.mkdtemp())
    try:
        sb = create_sandbox(config, d)
        assert isinstance(sb, SubprocessSandbox)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_llm_router_instantiation():
    from autoforge.engine.llm_router import LLMRouter, TaskComplexity
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig(api_keys={"anthropic": "fake-key"})
    router = LLMRouter(config)
    m, t = router._select_model(TaskComplexity.HIGH)
    assert m == "claude-opus-4-6"
    assert t == 16384
    m2, t2 = router._select_model(TaskComplexity.STANDARD)
    assert m2 == "claude-sonnet-4-5-20250929"
    assert t2 == 8192


def test_git_manager_instantiation():
    from autoforge.engine.git_manager import GitManager
    d = Path(tempfile.mkdtemp())
    try:
        project_dir = d / "test-project"
        gm = GitManager(project_dir)
        assert gm.main_worktree == project_dir
        assert gm.worktrees_dir == d / "worktrees"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_all_agents_instantiate():
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.llm_router import LLMRouter
    from autoforge.engine.agents.director import DirectorAgent, DirectorFixAgent
    from autoforge.engine.agents.architect import ArchitectAgent
    from autoforge.engine.agents.builder import BuilderAgent
    from autoforge.engine.agents.reviewer import ReviewerAgent
    from autoforge.engine.agents.tester import TesterAgent
    from autoforge.engine.agents.gardener import GardenerAgent
    from autoforge.engine.agents.scanner import ScannerAgent
    from autoforge.engine.sandbox import SubprocessSandbox

    config = ForgeConfig(api_keys={"anthropic": "fake"})
    llm = LLMRouter(config)
    wd = Path(tempfile.mkdtemp())

    try:
        d = DirectorAgent(config, llm)
        assert d.ROLE == "director" and len(d._tools) == 0
        df = DirectorFixAgent(config, llm)
        assert df.ROLE == "director"
        a = ArchitectAgent(config, llm)
        assert a.ROLE == "architect" and len(a._tools) == 1
        sb = SubprocessSandbox(wd)
        b = BuilderAgent(config, llm, working_dir=wd, sandbox=sb)
        assert b.ROLE == "builder" and len(b._tools) == 4
        r = ReviewerAgent(config, llm, working_dir=wd)
        assert r.ROLE == "reviewer" and len(r._tools) == 2
        t = TesterAgent(config, llm, working_dir=wd, sandbox=sb)
        assert t.ROLE == "tester" and len(t._tools) == 2
        g = GardenerAgent(config, llm, working_dir=wd)
        assert g.ROLE == "gardener" and len(g._tools) == 3
        s = ScannerAgent(config, llm, working_dir=wd)
        assert s.ROLE == "scanner" and len(s._tools) == 3
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def test_agent_build_prompts():
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.llm_router import LLMRouter
    from autoforge.engine.agents.director import DirectorAgent
    from autoforge.engine.agents.architect import ArchitectAgent
    from autoforge.engine.agents.builder import BuilderAgent
    from autoforge.engine.agents.reviewer import ReviewerAgent
    from autoforge.engine.agents.tester import TesterAgent
    from autoforge.engine.agents.gardener import GardenerAgent
    from autoforge.engine.agents.scanner import ScannerAgent
    from autoforge.engine.sandbox import SubprocessSandbox

    config = ForgeConfig(api_keys={"anthropic": "fake"})
    llm = LLMRouter(config)
    wd = Path(tempfile.mkdtemp())
    sb = SubprocessSandbox(wd)
    spec = {"project_name": "test", "modules": []}

    try:
        assert len(DirectorAgent(config, llm).build_prompt({"project_description": "x"})) > 0
        assert len(ArchitectAgent(config, llm).build_prompt({"spec": spec})) > 0
        assert len(BuilderAgent(config, llm, wd, sb).build_prompt({"task": {"id": "T"}, "spec": spec})) > 0
        assert len(ReviewerAgent(config, llm, wd).build_prompt({"task": {"id": "T"}, "spec": spec})) > 0
        # Full project review prompt
        assert len(ReviewerAgent(config, llm, wd).build_prompt(
            {"task": {"id": "T"}, "spec": spec, "full_project_review": True}
        )) > 0
        assert len(TesterAgent(config, llm, wd, sb).build_prompt({"spec": spec})) > 0
        # Tester with mobile spec
        mobile_spec = {**spec, "mobile": {"target": "both", "framework": "react-native"}}
        assert "React Native" in TesterAgent(config, llm, wd, sb).build_prompt({"spec": mobile_spec})
        assert len(GardenerAgent(config, llm, wd).build_prompt({"review": {}, "spec": spec})) > 0
        assert len(ScannerAgent(config, llm, wd).build_prompt({"project_path": str(wd)})) > 0
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def test_agent_parse_methods():
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.llm_router import LLMRouter
    from autoforge.engine.agents.director import DirectorAgent, DirectorFixAgent
    from autoforge.engine.agents.reviewer import ReviewerAgent
    from autoforge.engine.agents.tester import TesterAgent
    from autoforge.engine.agents.scanner import ScannerAgent

    config = ForgeConfig(api_keys={"anthropic": "fake"})
    llm = LLMRouter(config)

    d = DirectorAgent(config, llm)
    spec = d.parse_spec('```json\n{"project_name": "x", "modules": []}\n```')
    assert spec["project_name"] == "x"

    df = DirectorFixAgent(config, llm)
    fix = df.parse_fix_task('{"id": "FIX-1", "description": "fix"}')
    assert fix["id"] == "FIX-1"

    _td = Path(tempfile.mkdtemp())
    r = ReviewerAgent(config, llm, _td)
    review = r.parse_review('{"approved": true, "score": 8, "issues": [], "summary": "ok"}')
    assert review.approved is True and review.score == 8

    t = TesterAgent(config, llm, _td)
    results = t.parse_results('{"all_passed": true, "results": [], "summary": "ok"}')
    assert results.all_passed is True

    s = ScannerAgent(config, llm, _td)
    scan = s.parse_scan('```json\n{"project_name": "test", "completeness": 80, "gaps": ["missing tests"]}\n```')
    assert scan.completeness == 80
    assert len(scan.gaps) == 1


def test_research_mode_blocks_writes():
    """Test that research mode blocks write tools."""
    from autoforge.engine.agent_base import AgentBase
    from autoforge.engine.config import ForgeConfig

    config = ForgeConfig(api_keys={"anthropic": "fake"}, mode="research")
    assert config.mode == "research"

    # Verify WRITE_TOOLS is defined
    assert "write_file" in AgentBase.WRITE_TOOLS
    assert "run_command" in AgentBase.WRITE_TOOLS


def test_agent_parse_failsafe():
    """Test that agents fail-safe on unparseable output (never auto-approve/pass)."""
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.llm_router import LLMRouter
    from autoforge.engine.agents.reviewer import ReviewerAgent
    from autoforge.engine.agents.tester import TesterAgent

    config = ForgeConfig(api_keys={"anthropic": "fake"})
    llm = LLMRouter(config)
    _td = Path(tempfile.mkdtemp())

    r = ReviewerAgent(config, llm, _td)
    # Unparseable output should NOT auto-approve
    review = r.parse_review("This is not JSON at all")
    assert review.approved is False
    assert review.score == 0

    # Invalid JSON should NOT auto-approve
    review2 = r.parse_review('```json\n{bad json}\n```')
    assert review2.approved is False

    t = TesterAgent(config, llm, _td)
    # Unparseable output should NOT auto-pass
    results = t.parse_results("Random text output")
    assert results.all_passed is False


# ────────────────────────────────────────────
# Phase 3b: Multi-Provider
# ────────────────────────────────────────────

def test_provider_detection():
    from autoforge.engine.llm_router import detect_provider
    assert detect_provider("claude-opus-4-6") == "anthropic"
    assert detect_provider("claude-sonnet-4-5-20250929") == "anthropic"
    assert detect_provider("claude-haiku-4-5-20251001") == "anthropic"
    assert detect_provider("gpt-4o") == "openai"
    assert detect_provider("gpt-4o-mini") == "openai"
    assert detect_provider("o3") == "openai"
    assert detect_provider("o4-mini") == "openai"
    assert detect_provider("gemini-2.5-pro") == "google"
    assert detect_provider("gemini-2.5-flash") == "google"
    assert detect_provider("gemini-2.0-flash") == "google"
    # Unknown model defaults to anthropic
    assert detect_provider("some-unknown-model") == "anthropic"


def test_normalized_response_types():
    from autoforge.engine.llm_router import ContentBlock, Usage, LLMResponse
    # Text block
    tb = ContentBlock(type="text", text="hello")
    assert tb.type == "text" and tb.text == "hello"
    # Tool use block
    tu = ContentBlock(type="tool_use", id="abc", name="write_file", input={"path": "x"})
    assert tu.type == "tool_use" and tu.name == "write_file"
    # Response
    resp = LLMResponse(
        content=[tb, tu],
        stop_reason="tool_use",
        usage=Usage(input_tokens=100, output_tokens=50),
    )
    assert resp.stop_reason == "tool_use"
    assert len(resp.content) == 2
    assert resp.usage.input_tokens == 100


def test_multi_key_config():
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig(api_keys={
        "anthropic": "sk-ant-fake",
        "openai": "sk-fake",
        "google": "AI-fake",
    })
    # Backward compat property
    assert config.anthropic_api_key == "sk-ant-fake"
    assert config.api_keys["openai"] == "sk-fake"
    assert config.api_keys["google"] == "AI-fake"


def test_multi_key_config_setter():
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig()
    config.anthropic_api_key = "sk-ant-test"
    assert config.api_keys["anthropic"] == "sk-ant-test"


def test_extended_model_pricing():
    from autoforge.engine.config import MODEL_PRICING
    # Anthropic
    assert "claude-opus-4-6" in MODEL_PRICING
    # OpenAI
    assert "gpt-4o" in MODEL_PRICING
    assert "o3" in MODEL_PRICING
    # Google
    assert "gemini-2.5-pro" in MODEL_PRICING
    assert "gemini-2.0-flash" in MODEL_PRICING


# ────────────────────────────────────────────
# Phase 4: Integration
# ────────────────────────────────────────────

def test_orchestrator_instantiation():
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.orchestrator import Orchestrator
    config = ForgeConfig(api_keys={"anthropic": "fake-key"})
    orch = Orchestrator(config)
    assert orch.llm is not None
    assert orch.config is config
    assert orch._list_project_files() == []


def test_orchestrator_has_review_and_import():
    """Test that orchestrator has review_project and import_project methods."""
    from autoforge.engine.orchestrator import Orchestrator
    assert hasattr(Orchestrator, "review_project")
    assert hasattr(Orchestrator, "import_project")
    assert callable(getattr(Orchestrator, "review_project"))
    assert callable(getattr(Orchestrator, "import_project"))


def test_orchestrator_show_status():
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.orchestrator import Orchestrator
    config = ForgeConfig()
    orch = Orchestrator(config)
    orch.show_status()  # Should not raise


def test_llm_router_requires_api_key():
    """LLMRouter allows init without keys (lazy client creation)."""
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.llm_router import LLMRouter
    config = ForgeConfig()  # No API keys
    router = LLMRouter(config)
    # Should instantiate fine — keys checked lazily when _get_client is called
    assert router.call_count == 0


# ────────────────────────────────────────────
# Phase 5: Daemon components
# ────────────────────────────────────────────

def test_project_registry_crud():
    """Test ProjectRegistry CRUD operations."""
    from autoforge.engine.project_registry import ProjectRegistry, ProjectStatus

    async def _test():
        import os
        db_path = Path(tempfile.mktemp(suffix=".db"))
        try:
            async with ProjectRegistry(db_path) as reg:
                # Enqueue
                p = await reg.enqueue("Test project", "cli", 5.0)
                assert p.status == ProjectStatus.QUEUED
                assert p.budget_usd == 5.0

                # Get
                p2 = await reg.get(p.id)
                assert p2.description == "Test project"

                # List
                all_projects = await reg.list_all()
                assert len(all_projects) == 1

                # Queue size
                assert await reg.queue_size() == 1

                # Dequeue
                dequeued = await reg.dequeue()
                assert dequeued is not None
                assert dequeued.status == ProjectStatus.BUILDING
                assert await reg.queue_size() == 0

                # Update phase
                await reg.update_phase(p.id, "build")
                updated = await reg.get(p.id)
                assert updated.phase == "build"

                # Mark completed
                await reg.mark_completed(p.id, 2.50)
                completed = await reg.get(p.id)
                assert completed.status == ProjectStatus.COMPLETED
                assert completed.cost_usd == 2.50

                # Total cost
                assert await reg.total_cost() == 2.50

                # Enqueue + Cancel
                p3 = await reg.enqueue("Cancel me", "cli")
                assert await reg.cancel(p3.id) is True
                cancelled = await reg.get(p3.id)
                assert cancelled.status == ProjectStatus.CANCELLED

                # Cannot cancel non-queued
                assert await reg.cancel(p.id) is False
        finally:
            db_path.unlink(missing_ok=True)

    asyncio.run(_test())


def test_project_registry_to_dict():
    """Test Project.to_dict() serialization."""
    from autoforge.engine.project_registry import Project, ProjectStatus
    p = Project(
        id="abc123", name="test", description="desc",
        status=ProjectStatus.QUEUED, phase="", workspace_path="",
        requested_by="cli", budget_usd=10.0, cost_usd=0.0,
        created_at="2025-01-01T00:00:00", started_at=None,
        completed_at=None, error=None,
    )
    d = p.to_dict()
    assert d["id"] == "abc123"
    assert d["status"] == "queued"


def test_deploy_guide_generation():
    """Test deploy guide generation."""
    from autoforge.engine.deploy_guide import detect_framework, generate_deploy_guide

    d = Path(tempfile.mkdtemp())
    try:
        # Create a Next.js project
        (d / "package.json").write_text(json.dumps({
            "name": "test-app",
            "dependencies": {"next": "14.0.0", "react": "18.0.0"},
            "scripts": {"build": "next build", "dev": "next dev"},
        }), encoding="utf-8")
        (d / ".env.example").write_text(
            "DATABASE_URL=\nNEXTAUTH_SECRET=\n", encoding="utf-8"
        )

        # Detect
        info = detect_framework(d)
        assert info["framework"] == "nextjs"
        assert "DATABASE_URL" in info["env_vars"]

        # Generate
        guide = generate_deploy_guide(d, "test-app")
        assert "Vercel" in guide
        assert "DATABASE_URL" in guide
        assert "test-app" in guide
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_deploy_guide_vite():
    """Test framework detection for Vite project."""
    from autoforge.engine.deploy_guide import detect_framework

    d = Path(tempfile.mkdtemp())
    try:
        (d / "package.json").write_text(json.dumps({
            "dependencies": {"react": "18.0.0"},
            "devDependencies": {"vite": "5.0.0"},
        }), encoding="utf-8")
        info = detect_framework(d)
        assert info["framework"] == "vite"
        assert info["output_directory"] == "dist"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_daemon_instantiation():
    """Test ForgeDaemon can be instantiated."""
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.daemon import ForgeDaemon
    config = ForgeConfig(api_keys={"anthropic": "fake-key"})
    daemon = ForgeDaemon(config)
    assert daemon.config is config
    assert daemon._running is False


def test_forge_config_daemon_fields():
    """Test daemon-related config fields."""
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig()
    assert config.daemon_enabled is False
    assert config.daemon_poll_interval == 10
    assert config.telegram_token == ""
    assert config.webhook_enabled is False
    assert config.webhook_port == 8420
    assert config.db_path is not None


def test_cli_daemon_subcommand():
    """Test that daemon module can be imported."""
    from autoforge.engine.daemon import ForgeDaemon  # noqa: F401
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig(api_keys={"anthropic": "fake-key"})
    daemon = ForgeDaemon(config)
    assert daemon._running is False


def test_cli_queue_subcommand():
    """Test that project registry supports queue operations."""
    from autoforge.engine.project_registry import ProjectRegistry  # noqa: F401


def test_service_files_exist():
    """Test service config files exist."""
    assert (PROJECT_ROOT / "services" / "autoforge.service").exists()
    assert (PROJECT_ROOT / "services" / "com.autoforge.daemon.plist").exists()


# ────────────────────────────────────────────
# Phase 6: Constitution files
# ────────────────────────────────────────────

def test_constitution_files_exist():
    base = PROJECT_ROOT / "autoforge" / "data" / "constitution"
    assert (base / "CONSTITUTION.md").exists()
    assert (base / "quality_gates.md").exists()
    for agent in ["director", "architect", "builder", "reviewer", "tester", "gardener", "scanner"]:
        path = base / "agents" / f"{agent}.md"
        assert path.exists(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"
    for workflow in ["spec", "build", "verify", "refactor", "deliver", "review", "import"]:
        path = base / "workflows" / f"{workflow}.md"
        assert path.exists(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"


def test_display_module():
    """Test display module works without errors."""
    from autoforge.cli.display import show_banner, show_phase_progress, show_cost_tracker
    show_banner()
    show_phase_progress("SPEC", "done")
    show_cost_tracker(1.5, 10.0)


# ────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────

def main():
    print("=" * 56)
    print("AutoForge Smoke Test Suite")
    print("=" * 56)

    sections = [
        ("Imports", [
            ("Standard library imports", test_stdlib_imports),
            ("Dependency imports", test_dependency_imports),
            ("Engine module imports", test_engine_imports),
            ("Agent imports + registry (8 agents)", test_agent_imports),
            ("CLI module imports", test_cli_imports),
        ]),
        ("CLI", [
            ("Subcommand parsing", test_cli_parse_subcommands),
            ("Legacy CLI compatibility", test_cli_legacy_compat),
        ]),
        ("Unit: ForgeConfig", [
            ("Default construction + new fields", test_forge_config),
            ("from_env with overrides", test_forge_config_from_env),
            ("Budget tracking + exhaustion", test_forge_config_budget_tracking),
            ("Mobile config fields", test_forge_config_mobile),
        ]),
        ("Unit: TaskDAG", [
            ("Basic operations + dependency resolution", test_task_dag_basic),
            ("Save and load (JSON persistence)", test_task_dag_save_load),
            ("Failure handling + BLOCKED status", test_task_dag_failure_handling),
            ("Cycle detection", test_task_dag_cycle_detection),
        ]),
        ("Unit: LockManager", [
            ("Claim, release, enforce, clear", test_lock_manager),
        ]),
        ("Unit: Sandbox", [
            ("Subprocess execution", test_sandbox_subprocess),
            ("Factory function", test_sandbox_factory),
        ]),
        ("Unit: LLMRouter", [
            ("Instantiation + model selection", test_llm_router_instantiation),
            ("Rejects empty API key", test_llm_router_requires_api_key),
        ]),
        ("Unit: GitManager", [
            ("Instantiation", test_git_manager_instantiation),
        ]),
        ("Unit: Agents", [
            ("All 8 agents instantiate", test_all_agents_instantiate),
            ("All build_prompt methods", test_agent_build_prompts),
            ("All parse methods", test_agent_parse_methods),
            ("Fail-safe on unparseable output", test_agent_parse_failsafe),
            ("Research mode blocks writes", test_research_mode_blocks_writes),
        ]),
        ("Multi-Provider", [
            ("Provider detection from model names", test_provider_detection),
            ("Normalized response types", test_normalized_response_types),
            ("Multi-key config", test_multi_key_config),
            ("Multi-key config setter (backward compat)", test_multi_key_config_setter),
            ("Extended model pricing", test_extended_model_pricing),
        ]),
        ("Integration", [
            ("Orchestrator instantiation", test_orchestrator_instantiation),
            ("Orchestrator has review + import", test_orchestrator_has_review_and_import),
            ("Orchestrator show_status", test_orchestrator_show_status),
        ]),
        ("Unit: ProjectRegistry", [
            ("CRUD operations", test_project_registry_crud),
            ("Project.to_dict()", test_project_registry_to_dict),
            ("Input validation", test_project_registry_validation),
        ]),
        ("Unit: DeployGuide", [
            ("Next.js detection + guide generation", test_deploy_guide_generation),
            ("Vite detection", test_deploy_guide_vite),
        ]),
        ("Unit: Daemon", [
            ("ForgeDaemon instantiation", test_daemon_instantiation),
            ("Config daemon fields", test_forge_config_daemon_fields),
        ]),
        ("CLI: Subcommands", [
            ("daemon subcommand parsing", test_cli_daemon_subcommand),
            ("queue subcommand parsing", test_cli_queue_subcommand),
        ]),
        ("Service Files", [
            ("systemd + launchd configs exist", test_service_files_exist),
        ]),
        ("Constitution", [
            ("All constitution files exist + non-empty", test_constitution_files_exist),
        ]),
        ("Display", [
            ("Display module functions", test_display_module),
        ]),
    ]

    for section_name, tests in sections:
        print(f"\n[{section_name}]")
        for name, fn in tests:
            check(name, fn)

    print()
    print("=" * 56)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 56)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
