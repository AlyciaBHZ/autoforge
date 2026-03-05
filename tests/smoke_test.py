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

    # Test daemon
    args = parser.parse_args(["daemon", "start"])
    assert args.command == "daemon"
    assert args.daemon_action == "start"

    # Test queue
    args = parser.parse_args(["queue", "Build a blog", "--budget", "5", "--idempotency-key", "abc"])
    assert args.command == "queue"
    assert args.description == "Build a blog"
    assert args.idempotency_key == "abc"

    # Test projects
    args = parser.parse_args(["projects", "--limit", "10"])
    assert args.command == "projects"
    assert args.limit == 10

    # Test deploy
    args = parser.parse_args(["deploy", "abc123"])
    assert args.command == "deploy"
    assert args.project_id == "abc123"

    # Test paper infer
    args = parser.parse_args(["paper", "infer", "improve long-context reasoning", "--year", "2025"])
    assert args.command == "paper"
    assert args.paper_action == "infer"
    assert args.year == 2025

    # Test paper benchmark
    args = parser.parse_args(["paper", "benchmark", "--sample-size", "4"])
    assert args.command == "paper"
    assert args.paper_action == "benchmark"
    assert args.sample_size == 4

    # Test paper reproduce
    args = parser.parse_args(
        ["paper", "reproduce", "robust graph learning", "--pick", "2", "--strict-contract"]
    )
    assert args.command == "paper"
    assert args.paper_action == "reproduce"
    assert args.pick == 2
    assert args.strict_contract is True

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
    assert "daemon" in _KNOWN_COMMANDS
    assert "queue" in _KNOWN_COMMANDS
    assert "paper" in _KNOWN_COMMANDS


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
    assert config.daemon_max_concurrent_projects >= 1
    assert config.webhook_require_auth is True
    assert config.webhook_trust_requester_header is False


def test_forge_config_from_env():
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig.from_env(budget_limit_usd=5.0, max_agents=2, mode="research")
    assert config.budget_limit_usd == 5.0
    assert config.max_agents == 2
    assert config.mode == "research"


def test_forge_config_budget_tracking():
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig(budget_limit_usd=1.0)
    config.record_usage("claude-sonnet-4-6", 1000, 500)
    assert config.total_input_tokens == 1000
    assert config.total_output_tokens == 500
    assert config.estimated_cost_usd > 0
    assert config.check_budget() is True
    # Exhaust budget
    config.record_usage("claude-opus-4-6", 10_000_000, 10_000_000)
    assert config.check_budget() is False


def test_request_intake_service_limits_and_idempotency():
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.project_registry import ProjectRegistry
    from autoforge.engine.request_intake import RequestIntakeService

    async def _test():
        db_path = Path(tempfile.mktemp(suffix=".db"))
        try:
            async with ProjectRegistry(db_path) as reg:
                config = ForgeConfig(
                    queue_max_size=10,
                    requester_queue_limit=3,
                    requester_daily_limit=20,
                    requester_rate_limit=5,
                    requester_rate_window_seconds=60,
                )
                intake = RequestIntakeService(config, reg)
                first = await intake.enqueue(
                    channel="webhook",
                    requester_hint="user-1",
                    description="Build a small test app",
                    budget=5.0,
                    idempotency_key="idem-1",
                )
                assert first.project.requested_by.startswith("webhook:")

                second = await intake.enqueue(
                    channel="webhook",
                    requester_hint="user-1",
                    description="Build a small test app",
                    budget=5.0,
                    idempotency_key="idem-1",
                )
                assert second.deduplicated is True
                assert second.project.id == first.project.id
        finally:
            db_path.unlink(missing_ok=True)

    asyncio.run(_test())


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
    assert m2 == "claude-sonnet-4-6"
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


def test_agent_result_metrics_fields():
    """Test AgentResult has files_written and tool_calls fields."""
    from autoforge.engine.agent_base import AgentResult
    r = AgentResult(agent_name="builder", success=True)
    assert r.files_written == 0
    assert r.tool_calls == {}
    r2 = AgentResult(
        agent_name="builder", success=True,
        files_written=3, tool_calls={"write_file": 3, "read_file": 5},
    )
    assert r2.files_written == 3
    assert r2.tool_calls["write_file"] == 3


def test_anti_spin_constants():
    """Test anti-spin detection constants on AgentBase."""
    from autoforge.engine.agent_base import AgentBase
    assert hasattr(AgentBase, "SPIN_WARN_TURNS")
    assert hasattr(AgentBase, "SPIN_FAIL_TURNS")
    assert AgentBase.SPIN_WARN_TURNS == 10
    assert AgentBase.SPIN_FAIL_TURNS == 20
    assert AgentBase.SPIN_FAIL_TURNS > AgentBase.SPIN_WARN_TURNS


def test_build_gate_enforcement():
    """Test _enforce_build_gate catches missing/failed tasks."""
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.orchestrator import Orchestrator
    from autoforge.engine.task_dag import TaskDAG, Task, TaskPhase, TaskStatus

    config = ForgeConfig(api_keys={"anthropic": "fake"})
    orch = Orchestrator(config)

    # All done → no error
    dag = TaskDAG()
    dag.add_task(Task(id="T1", description="test", phase=TaskPhase.BUILD, status=TaskStatus.DONE, files=[]))
    try:
        asyncio.run(orch._enforce_build_gate(dag, Path(tempfile.mkdtemp())))
    except RuntimeError:
        raise AssertionError("Should not raise for completed tasks")

    # Zero completed → RuntimeError
    dag2 = TaskDAG()
    dag2.add_task(Task(id="T2", description="fail", phase=TaskPhase.BUILD, status=TaskStatus.FAILED))
    dag2.add_task(Task(id="T3", description="fail2", phase=TaskPhase.BUILD, status=TaskStatus.FAILED))
    try:
        asyncio.run(orch._enforce_build_gate(dag2, Path(tempfile.mkdtemp())))
        raise AssertionError("Should raise for all-failed tasks")
    except RuntimeError as e:
        assert "failed" in str(e).lower()


def test_file_overlap_detection():
    """Test _detect_file_overlaps finds overlapping files."""
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.orchestrator import Orchestrator
    from autoforge.engine.task_dag import TaskDAG, Task, TaskPhase

    config = ForgeConfig(api_keys={"anthropic": "fake"})
    orch = Orchestrator(config)

    dag = TaskDAG()
    dag.add_task(Task(id="A", description="task A", phase=TaskPhase.BUILD, files=["shared.py", "a.py"]))
    dag.add_task(Task(id="B", description="task B", phase=TaskPhase.BUILD, files=["shared.py", "b.py"]))
    dag.add_task(Task(id="C", description="task C", phase=TaskPhase.BUILD, files=["c.py"]))

    overlaps = orch._detect_file_overlaps(dag)
    assert "shared.py" in overlaps
    assert set(overlaps["shared.py"]) == {"A", "B"}
    assert "a.py" not in overlaps
    assert "c.py" not in overlaps


def test_smoke_check_missing_files():
    """Test _smoke_check catches missing files."""
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.orchestrator import Orchestrator
    from autoforge.engine.task_dag import Task, TaskPhase

    config = ForgeConfig(api_keys={"anthropic": "fake"})
    orch = Orchestrator(config)

    d = Path(tempfile.mkdtemp())
    try:
        task = Task(id="T1", description="test", phase=TaskPhase.BUILD, files=["missing.py"])
        ok, msg = asyncio.run(orch._smoke_check(task, d))
        assert ok is False
        assert "Missing files" in msg
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_smoke_check_syntax_error():
    """Test _smoke_check catches Python syntax errors."""
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.orchestrator import Orchestrator
    from autoforge.engine.task_dag import Task, TaskPhase

    config = ForgeConfig(api_keys={"anthropic": "fake"})
    orch = Orchestrator(config)

    d = Path(tempfile.mkdtemp())
    try:
        # Write a file with a syntax error
        (d / "bad.py").write_text("def f(\n  pass", encoding="utf-8")
        task = Task(id="T1", description="test", phase=TaskPhase.BUILD, files=["bad.py"])
        ok, msg = asyncio.run(orch._smoke_check(task, d))
        assert ok is False
        assert "Syntax errors" in msg
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_smoke_check_valid_file():
    """Test _smoke_check passes for valid Python files."""
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.orchestrator import Orchestrator
    from autoforge.engine.task_dag import Task, TaskPhase

    config = ForgeConfig(api_keys={"anthropic": "fake"})
    orch = Orchestrator(config)

    d = Path(tempfile.mkdtemp())
    try:
        (d / "good.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
        task = Task(id="T1", description="test", phase=TaskPhase.BUILD, files=["good.py"])
        ok, msg = asyncio.run(orch._smoke_check(task, d))
        assert ok is True
        assert msg == "OK"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_constitution_builder_mandatory_rules():
    """Test builder.md contains mandatory rules section."""
    builder_path = PROJECT_ROOT / "autoforge" / "data" / "constitution" / "agents" / "builder.md"
    content = builder_path.read_text(encoding="utf-8")
    assert "MANDATORY" in content
    assert "write_file" in content
    assert "read_file" in content
    assert "py_compile" in content


def test_constitution_architect_composition():
    """Test architect.md contains composition principle."""
    arch_path = PROJECT_ROOT / "autoforge" / "data" / "constitution" / "agents" / "architect.md"
    content = arch_path.read_text(encoding="utf-8")
    assert "Composition" in content
    assert "library" in content.lower() or "libraries" in content.lower()


def test_constitution_quality_gates_enforcement():
    """Test quality_gates.md documents enforcement status."""
    qg_path = PROJECT_ROOT / "autoforge" / "data" / "constitution" / "quality_gates.md"
    content = qg_path.read_text(encoding="utf-8")
    assert "Enforcement" in content
    assert "_enforce_build_gate" in content
    assert "Anti-Spin" in content or "anti-spin" in content.lower()


def test_git_available_detection():
    from autoforge.engine.git_manager import is_git_available, _git_available
    import autoforge.engine.git_manager as gm_mod
    # Reset cache so we test fresh
    old = gm_mod._git_available
    gm_mod._git_available = None
    try:
        result = is_git_available()
        assert isinstance(result, bool)
        # On most CI/dev systems git is installed; just verify it returns bool
        # Also verify caching works
        assert gm_mod._git_available is result
        assert is_git_available() is result  # cached
    finally:
        gm_mod._git_available = old


def test_git_version_async():
    from autoforge.engine.git_manager import get_git_version, is_git_available
    version = asyncio.run(get_git_version())
    if is_git_available():
        assert version is not None
        assert "git version" in version.lower()
    else:
        assert version is None


def test_config_has_api_key():
    from autoforge.engine.config import ForgeConfig
    # No keys → has_api_key is False
    config = ForgeConfig(api_keys={})
    assert config.has_api_key is False
    # With a key → has_api_key is True
    config2 = ForgeConfig(api_keys={"openai": "sk-test123"})
    assert config2.has_api_key is True
    # Empty string key → has_api_key is False
    config3 = ForgeConfig(api_keys={"anthropic": ""})
    assert config3.has_api_key is False


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
        assert d.ROLE == "director" and len(d._tools) == 4  # search_web + fetch_url + search_github + inspect_repo
        df = DirectorFixAgent(config, llm)
        assert df.ROLE == "director_fix"
        a = ArchitectAgent(config, llm)
        assert a.ROLE == "architect" and len(a._tools) == 5  # read_template + search_web + fetch_url + search_github + inspect_repo
        sb = SubprocessSandbox(wd)
        b = BuilderAgent(config, llm, working_dir=wd, sandbox=sb)
        assert b.ROLE == "builder" and len(b._tools) == 7  # write/read/list/run + grep_search + fetch_url + search_github
        r = ReviewerAgent(config, llm, working_dir=wd)
        assert r.ROLE == "reviewer" and len(r._tools) == 4  # read/list/run_check + grep_search
        t = TesterAgent(config, llm, working_dir=wd, sandbox=sb)
        assert t.ROLE == "tester" and len(t._tools) == 2
        g = GardenerAgent(config, llm, working_dir=wd)
        assert g.ROLE == "gardener" and len(g._tools) == 5  # write/read/list + grep_search + fetch_url
        s = ScannerAgent(config, llm, working_dir=wd)
        assert s.ROLE == "scanner" and len(s._tools) == 4  # read/list/run + grep_search
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
    assert detect_provider("claude-sonnet-4-6") == "anthropic"
    assert detect_provider("claude-haiku-4-5-20251001") == "anthropic"
    assert detect_provider("gpt-4o") == "openai"
    assert detect_provider("gpt-5-mini") == "openai"
    assert detect_provider("gpt-4o-mini") == "openai"
    assert detect_provider("o3") == "openai"
    assert detect_provider("o4-mini") == "openai"
    assert detect_provider("gemini-2.5-pro") == "google"
    assert detect_provider("gemini-3-pro-preview") == "google"
    assert detect_provider("gemini-2.5-flash") == "google"
    assert detect_provider("gemini-2.5-flash-lite") == "google"
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
    assert "claude-sonnet-4-6" in MODEL_PRICING
    # OpenAI
    assert "gpt-5.2" in MODEL_PRICING
    assert "gpt-5-mini" in MODEL_PRICING
    assert "gpt-4o" in MODEL_PRICING
    assert "o3" in MODEL_PRICING
    # Google
    assert "gemini-2.5-pro" in MODEL_PRICING
    assert "gemini-2.5-flash-lite" in MODEL_PRICING


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

                # Requester isolation + idempotency
                p4 = await reg.enqueue("Owned by A", "telegram:100", idempotency_key="idem-a")
                p5 = await reg.enqueue("Owned by B", "telegram:200", idempotency_key="idem-b")
                a_projects = await reg.list_for_requester("telegram:100")
                b_projects = await reg.list_for_requester("telegram:200")
                assert any(x.id == p4.id for x in a_projects)
                assert any(x.id == p5.id for x in b_projects)
                found = await reg.get_by_idempotency("telegram:100", "idem-a")
                assert found is not None and found.id == p4.id
                with_raises = False
                try:
                    await reg.get_for_requester(p5.id, "telegram:100")
                except KeyError:
                    with_raises = True
                assert with_raises is True
                assert await reg.cancel_for_requester(p5.id, "telegram:100") is False
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


def test_cli_paper_subcommand():
    """Test that paper inference module imports."""
    from autoforge.engine.paper_repro import fetch_iclr_papers  # noqa: F401


def test_paper_repro_signals_and_simulation():
    """Test local paper signal extraction + no-key simulation feedback."""
    from autoforge.engine.paper_repro import (
        PaperRecord,
        build_environment_spec,
        build_verification_plan,
        extract_paper_signals,
        simulate_pipeline_feedback,
    )
    from autoforge.engine.repro_contract import build_repro_report, validate_report_schema

    paper = PaperRecord(
        note_id="test123",
        title="Fast Sparse Attention for Long-Context LLM Decoding",
        abstract=(
            "We propose a sparse attention method that reduces latency by 2.0x "
            "while maintaining accuracy on MMLU."
        ),
        keywords=["sparse attention", "long context", "llm inference"],
        year=2025,
        openreview_url="https://openreview.net/forum?id=test123",
        pdf_url="https://openreview.net/pdf?id=test123",
    )
    signals = extract_paper_signals(paper, include_pdf=False)
    assert "latency" in signals.metrics

    plan = build_verification_plan(signals)
    assert "checklist" in plan and len(plan["checklist"]) >= 1

    env = build_environment_spec(paper, signals)
    assert env["python"] == "3.10"
    dep_names = {d["name"] for d in env["dependencies"]}
    assert "torch" not in dep_names
    assert env["profile"] == "theory-first"

    fb = simulate_pipeline_feedback(
        goal="speed up long-context sparse attention decoding",
        paper=paper,
        signals=signals,
        inference_score=15.0,
    )
    assert fb["mode"] == "simulated_no_api_key"
    assert "p0_p4_status" in fb

    report = build_repro_report(
        run_id="run-test123",
        paper_id=paper.note_id,
        goal="speed up long-context sparse attention decoding",
        mode="simulated_no_api_key",
        profile="theory-first",
        output_dir=Path("."),
        strict_contract=True,
        p0_p4_status=fb["p0_p4_status"],
        artifacts_complete=True,
        failure_reasons=[],
    )
    assert validate_report_schema(report) == []


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
# Phase 7: Web Tools, Code Search, Checkpoints, TDD
# ────────────────────────────────────────────

def test_tools_package_imports():
    """Test that the tools package can be imported."""
    from autoforge.engine.tools import web  # noqa: F401
    from autoforge.engine.tools import search  # noqa: F401
    from autoforge.engine.tools.web import (
        handle_fetch_url,  # noqa: F401
        handle_search_web,  # noqa: F401
        FETCH_URL_TOOL_SCHEMA,
        SEARCH_WEB_TOOL_SCHEMA,
    )
    from autoforge.engine.tools.search import (
        handle_grep_search,  # noqa: F401
        GREP_SEARCH_TOOL_SCHEMA,
    )
    # Validate schemas have required fields
    assert "properties" in FETCH_URL_TOOL_SCHEMA
    assert "url" in FETCH_URL_TOOL_SCHEMA["properties"]
    assert "properties" in SEARCH_WEB_TOOL_SCHEMA
    assert "query" in SEARCH_WEB_TOOL_SCHEMA["properties"]
    assert "properties" in GREP_SEARCH_TOOL_SCHEMA
    assert "pattern" in GREP_SEARCH_TOOL_SCHEMA["properties"]


def test_config_new_fields():
    """Test new config fields for web tools, checkpoints, TDD."""
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig()
    # Web tools
    assert config.web_tools_enabled is True
    assert config.search_backend == "duckduckgo"
    assert config.search_api_key == ""
    # Checkpoints
    assert config.confirm_phases == []
    # TDD
    assert config.build_test_loops == 0


def test_config_with_overrides():
    """Test new config fields can be overridden."""
    from autoforge.engine.config import ForgeConfig
    config = ForgeConfig(
        web_tools_enabled=False,
        search_backend="google",
        search_api_key="test-key:cx123",
        confirm_phases=["spec", "build"],
        build_test_loops=2,
    )
    assert config.web_tools_enabled is False
    assert config.search_backend == "google"
    assert config.search_api_key == "test-key:cx123"
    assert config.confirm_phases == ["spec", "build"]
    assert config.build_test_loops == 2


def test_cli_confirm_flag():
    """Test --confirm flag parsing."""
    from autoforge.cli.app import build_parser
    parser = build_parser()
    args = parser.parse_args(["--confirm", "spec,build", "generate", "test"])
    assert args.confirm == "spec,build"


def test_cli_tdd_flag():
    """Test --tdd flag parsing."""
    from autoforge.cli.app import build_parser
    parser = build_parser()
    args = parser.parse_args(["--tdd", "2", "generate", "test"])
    assert args.tdd == 2


def test_cli_config_overrides_confirm():
    """Test --confirm wiring to config overrides."""
    from autoforge.cli.app import build_parser, _build_config_overrides
    parser = build_parser()
    args = parser.parse_args(["--confirm", "spec,build,verify", "generate", "test"])
    overrides = _build_config_overrides(args)
    assert overrides["confirm_phases"] == ["spec", "build", "verify"]


def test_cli_config_overrides_tdd():
    """Test --tdd wiring to config overrides."""
    from autoforge.cli.app import build_parser, _build_config_overrides
    parser = build_parser()
    args = parser.parse_args(["--tdd", "3", "generate", "test"])
    overrides = _build_config_overrides(args)
    assert overrides["build_test_loops"] == 3


def test_grep_search_basic():
    """Test grep_search on a temporary directory."""
    from autoforge.engine.tools.search import handle_grep_search
    d = Path(tempfile.mkdtemp())
    try:
        # Create test files
        (d / "hello.py").write_text("def hello_world():\n    print('hello')\n", encoding="utf-8")
        (d / "utils.py").write_text("def helper_func():\n    pass\n", encoding="utf-8")

        result = asyncio.run(handle_grep_search({"pattern": "def.*hello"}, d))
        data = json.loads(result)
        assert data["total_matches"] >= 1
        assert any("hello.py" in m["file"] for m in data["matches"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_grep_search_file_glob():
    """Test grep_search with file_glob filter."""
    from autoforge.engine.tools.search import handle_grep_search
    d = Path(tempfile.mkdtemp())
    try:
        (d / "app.py").write_text("import flask\n", encoding="utf-8")
        (d / "style.css").write_text("/* import reset */\n", encoding="utf-8")

        result = asyncio.run(handle_grep_search({"pattern": "import", "file_glob": "*.py"}, d))
        data = json.loads(result)
        # Should only find .py files
        for m in data["matches"]:
            assert m["file"].endswith(".py"), f"Expected .py file but got {m['file']}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_grep_search_empty_pattern():
    """Test grep_search rejects empty pattern."""
    from autoforge.engine.tools.search import handle_grep_search
    d = Path(tempfile.mkdtemp())
    try:
        result = asyncio.run(handle_grep_search({"pattern": ""}, d))
        data = json.loads(result)
        assert "error" in data
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_web_tool_schemas():
    """Test web tool schema structures are valid."""
    from autoforge.engine.tools.web import FETCH_URL_TOOL_SCHEMA, SEARCH_WEB_TOOL_SCHEMA
    # fetch_url schema
    assert FETCH_URL_TOOL_SCHEMA["type"] == "object"
    assert "url" in FETCH_URL_TOOL_SCHEMA["properties"]
    assert FETCH_URL_TOOL_SCHEMA["required"] == ["url"]
    # search_web schema
    assert SEARCH_WEB_TOOL_SCHEMA["type"] == "object"
    assert "query" in SEARCH_WEB_TOOL_SCHEMA["properties"]
    assert SEARCH_WEB_TOOL_SCHEMA["required"] == ["query"]


def test_html_strip_fallback():
    """Test the HTML stripping fallback (no html2text dependency needed)."""
    from autoforge.engine.tools.web import _strip_html_tags
    html = "<html><body><h1>Title</h1><p>Hello <b>world</b></p><script>evil()</script></body></html>"
    text = _strip_html_tags(html)
    assert "Title" in text
    assert "Hello" in text
    assert "world" in text
    assert "evil()" not in text
    assert "<script>" not in text


def test_orchestrator_checkpoint_method():
    """Test that orchestrator has _checkpoint method."""
    from autoforge.engine.orchestrator import Orchestrator, UserPausedError
    assert hasattr(Orchestrator, "_checkpoint")
    assert callable(getattr(Orchestrator, "_checkpoint"))
    # UserPausedError should be an Exception subclass
    assert issubclass(UserPausedError, Exception)


def test_orchestrator_tdd_loop_method():
    """Test that orchestrator has _tdd_loop and _detect_test_command methods."""
    from autoforge.engine.orchestrator import Orchestrator
    assert hasattr(Orchestrator, "_tdd_loop")
    assert hasattr(Orchestrator, "_detect_test_command")


def test_detect_test_command_npm():
    """Test _detect_test_command detects npm test."""
    from autoforge.engine.orchestrator import Orchestrator
    d = Path(tempfile.mkdtemp())
    try:
        (d / "package.json").write_text(json.dumps({
            "scripts": {"test": "jest"}
        }), encoding="utf-8")
        cmd = Orchestrator._detect_test_command(d)
        assert cmd is not None
        assert "npm test" in cmd
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_detect_test_command_pytest():
    """Test _detect_test_command detects pytest."""
    from autoforge.engine.orchestrator import Orchestrator
    d = Path(tempfile.mkdtemp())
    try:
        (d / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        cmd = Orchestrator._detect_test_command(d)
        assert cmd is not None
        assert "pytest" in cmd
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_detect_test_command_none():
    """Test _detect_test_command returns None for unknown project type."""
    from autoforge.engine.orchestrator import Orchestrator
    d = Path(tempfile.mkdtemp())
    try:
        cmd = Orchestrator._detect_test_command(d)
        assert cmd is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_agents_have_new_tools():
    """Test that agents have the new tools registered."""
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.llm_router import LLMRouter
    config = ForgeConfig(api_keys={"anthropic": "fake-key"})
    llm = LLMRouter(config)
    _td = Path(tempfile.mkdtemp())

    try:
        # Builder should have grep_search and fetch_url
        from autoforge.engine.agents.builder import BuilderAgent
        builder = BuilderAgent(config, llm, _td)
        builder_tool_names = {t.name for t in builder._tools}
        assert "grep_search" in builder_tool_names, f"Builder tools: {builder_tool_names}"
        assert "fetch_url" in builder_tool_names, f"Builder tools: {builder_tool_names}"

        # Scanner should have grep_search
        from autoforge.engine.agents.scanner import ScannerAgent
        scanner = ScannerAgent(config, llm, _td)
        scanner_tool_names = {t.name for t in scanner._tools}
        assert "grep_search" in scanner_tool_names, f"Scanner tools: {scanner_tool_names}"

        # Gardener should have grep_search and fetch_url
        from autoforge.engine.agents.gardener import GardenerAgent
        gardener = GardenerAgent(config, llm, _td)
        gardener_tool_names = {t.name for t in gardener._tools}
        assert "grep_search" in gardener_tool_names, f"Gardener tools: {gardener_tool_names}"
        assert "fetch_url" in gardener_tool_names, f"Gardener tools: {gardener_tool_names}"

        # Director should have search_web and fetch_url
        from autoforge.engine.agents.director import DirectorAgent
        director = DirectorAgent(config, llm)
        director_tool_names = {t.name for t in director._tools}
        assert "search_web" in director_tool_names, f"Director tools: {director_tool_names}"
        assert "fetch_url" in director_tool_names, f"Director tools: {director_tool_names}"

        # Architect should have search_web and fetch_url
        from autoforge.engine.agents.architect import ArchitectAgent
        architect = ArchitectAgent(config, llm)
        architect_tool_names = {t.name for t in architect._tools}
        assert "search_web" in architect_tool_names, f"Architect tools: {architect_tool_names}"
        assert "fetch_url" in architect_tool_names, f"Architect tools: {architect_tool_names}"
    finally:
        shutil.rmtree(_td, ignore_errors=True)


def test_agents_web_tools_disabled():
    """Test that web tools are not registered when disabled."""
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.llm_router import LLMRouter
    config = ForgeConfig(api_keys={"anthropic": "fake-key"}, web_tools_enabled=False)
    llm = LLMRouter(config)

    # Director should NOT have search_web or fetch_url when disabled
    from autoforge.engine.agents.director import DirectorAgent
    director = DirectorAgent(config, llm)
    director_tool_names = {t.name for t in director._tools}
    assert "search_web" not in director_tool_names
    assert "fetch_url" not in director_tool_names


def test_constitution_builder_has_new_tools():
    """Test that builder constitution documents grep_search and fetch_url."""
    builder_md = PROJECT_ROOT / "autoforge" / "data" / "constitution" / "agents" / "builder.md"
    content = builder_md.read_text(encoding="utf-8")
    assert "grep_search" in content
    assert "fetch_url" in content


def test_constitution_architect_has_web_tools():
    """Test that architect constitution documents search_web and fetch_url."""
    architect_md = PROJECT_ROOT / "autoforge" / "data" / "constitution" / "agents" / "architect.md"
    content = architect_md.read_text(encoding="utf-8")
    assert "search_web" in content
    assert "fetch_url" in content


def test_constitution_quality_gates_has_agent_capabilities():
    """Test that quality_gates documents agent capabilities table."""
    qg_md = PROJECT_ROOT / "autoforge" / "data" / "constitution" / "quality_gates.md"
    content = qg_md.read_text(encoding="utf-8")
    assert "Agent Capabilities" in content
    assert "grep_search" in content
    assert "TDD Loop" in content
    assert "Human Checkpoints" in content


def test_pyproject_has_new_dependencies():
    """Test that pyproject.toml includes httpx, duckduckgo-search, html2text."""
    toml_path = PROJECT_ROOT / "pyproject.toml"
    content = toml_path.read_text(encoding="utf-8")
    assert "httpx" in content
    assert "duckduckgo-search" in content
    assert "html2text" in content


# ────────────────────────────────────────────
# ────────────────────────────────────────────
# Auth Module
# ────────────────────────────────────────────


def test_auth_module_imports():
    """Auth module imports and classes are accessible."""
    from autoforge.engine.auth import (  # noqa: F401
        AuthProvider,
        ApiKeyAuth,
        OAuthBearerAuth,
        OAuth2ClientCredentialsAuth,
        GoogleADCAuth,
        AWSBedrockAuth,
        VertexAIAuth,
        CodexOAuthAuth,
        DeviceCodeAuth,
        TokenResult,
        create_auth_provider,
    )


def test_auth_api_key_provider():
    """ApiKeyAuth returns key and correct client kwargs."""
    from autoforge.engine.auth import ApiKeyAuth

    auth = ApiKeyAuth(api_key="sk-test123")
    kwargs = auth.get_client_kwargs()
    assert kwargs == {"api_key": "sk-test123"}

    # With base_url
    auth2 = ApiKeyAuth(api_key="sk-test", base_url="https://proxy.com/v1")
    kwargs2 = auth2.get_client_kwargs()
    assert kwargs2["api_key"] == "sk-test"
    assert kwargs2["base_url"] == "https://proxy.com/v1"


def test_auth_bearer_provider():
    """OAuthBearerAuth returns bearer token as api_key."""
    from autoforge.engine.auth import OAuthBearerAuth

    auth = OAuthBearerAuth(bearer_token="tok123", base_url="https://proxy.com/v1")
    kwargs = auth.get_client_kwargs()
    assert kwargs["api_key"] == "tok123"
    assert kwargs["base_url"] == "https://proxy.com/v1"


def test_auth_token_result_expiry():
    """TokenResult.is_expired works correctly."""
    import time
    from autoforge.engine.auth import TokenResult

    # Never expires
    t1 = TokenResult(access_token="tok", expires_at=0.0)
    assert t1.is_expired is False

    # Expired
    t2 = TokenResult(access_token="tok", expires_at=time.time() - 100)
    assert t2.is_expired is True

    # Not expired (far future)
    t3 = TokenResult(access_token="tok", expires_at=time.time() + 3600)
    assert t3.is_expired is False


def test_auth_factory_fallback():
    """create_auth_provider falls back to ApiKeyAuth."""
    from autoforge.engine.auth import ApiKeyAuth, create_auth_provider

    auth = create_auth_provider("anthropic", {"anthropic": "sk-test"}, {})
    assert isinstance(auth, ApiKeyAuth)


def test_auth_factory_bearer():
    """create_auth_provider creates OAuthBearerAuth when configured."""
    from autoforge.engine.auth import OAuthBearerAuth, create_auth_provider

    auth = create_auth_provider("openai", {}, {
        "openai": {
            "auth_method": "oauth_bearer",
            "base_url": "https://proxy.com/v1",
            "bearer_token": "tok",
        }
    })
    assert isinstance(auth, OAuthBearerAuth)


def test_auth_factory_oauth2():
    """create_auth_provider creates OAuth2ClientCredentialsAuth."""
    from autoforge.engine.auth import OAuth2ClientCredentialsAuth, create_auth_provider

    auth = create_auth_provider("openai", {}, {
        "openai": {
            "auth_method": "oauth2_client_credentials",
            "client_id": "cid",
            "client_secret": "csec",
            "token_url": "https://token.example.com/token",
        }
    })
    assert isinstance(auth, OAuth2ClientCredentialsAuth)


def test_auth_factory_adc():
    """create_auth_provider creates GoogleADCAuth."""
    from autoforge.engine.auth import GoogleADCAuth, create_auth_provider

    auth = create_auth_provider("google", {}, {
        "google": {"auth_method": "adc"}
    })
    assert isinstance(auth, GoogleADCAuth)


def test_config_auth_config_field():
    """ForgeConfig has auth_config field with correct default."""
    from autoforge.engine.config import ForgeConfig

    config = ForgeConfig()
    assert config.auth_config == {}

    config2 = ForgeConfig(auth_config={
        "openai": {"auth_method": "oauth_bearer", "base_url": "https://x"}
    })
    assert config2.auth_config["openai"]["auth_method"] == "oauth_bearer"


def test_config_has_api_key_with_auth_config():
    """has_api_key returns True when only auth_config is set (no API keys)."""
    from autoforge.engine.config import ForgeConfig

    # No keys, no auth → False
    config1 = ForgeConfig()
    assert config1.has_api_key is False

    # No keys but auth_config → True
    config2 = ForgeConfig(auth_config={"openai": {"auth_method": "adc"}})
    assert config2.has_api_key is True


def test_auth_bedrock_provider():
    """AWSBedrockAuth stores region and returns correct kwargs."""
    from autoforge.engine.auth import AWSBedrockAuth

    auth = AWSBedrockAuth(aws_region="us-west-2", aws_profile="myprofile")
    kwargs = auth.get_client_kwargs()
    assert kwargs["aws_region"] == "us-west-2"
    assert kwargs["aws_profile"] == "myprofile"

    # With static keys
    auth2 = AWSBedrockAuth(
        aws_region="eu-west-1",
        aws_access_key_id="AKID",
        aws_secret_access_key="SECRET",
    )
    kwargs2 = auth2.get_client_kwargs()
    assert kwargs2["aws_access_key"] == "AKID"
    assert kwargs2["aws_secret_key"] == "SECRET"


def test_auth_vertex_provider():
    """VertexAIAuth stores project_id and region."""
    from autoforge.engine.auth import VertexAIAuth

    auth = VertexAIAuth(project_id="my-project", region="us-east5")
    kwargs = auth.get_client_kwargs()
    assert kwargs["region"] == "us-east5"
    assert kwargs["project_id"] == "my-project"


def test_auth_codex_oauth_provider():
    """CodexOAuthAuth initializes correctly."""
    from autoforge.engine.auth import CodexOAuthAuth

    auth = CodexOAuthAuth()
    # No token yet — get_client_kwargs returns empty (token injected dynamically)
    kwargs = auth.get_client_kwargs()
    assert kwargs == {}
    # Has correct endpoints
    assert "openai.com" in auth.AUTH_URL
    assert "openai.com" in auth.TOKEN_URL


def test_auth_device_code_provider():
    """DeviceCodeAuth initializes correctly."""
    from autoforge.engine.auth import DeviceCodeAuth

    auth = DeviceCodeAuth()
    kwargs = auth.get_client_kwargs()
    assert kwargs == {}
    assert "openai.com" in auth.DEVICE_URL
    assert "openai.com" in auth.TOKEN_URL


def test_auth_factory_bedrock():
    """create_auth_provider creates AWSBedrockAuth."""
    from autoforge.engine.auth import AWSBedrockAuth, create_auth_provider

    auth = create_auth_provider("anthropic", {}, {
        "anthropic": {
            "auth_method": "bedrock",
            "aws_region": "us-west-2",
        }
    })
    assert isinstance(auth, AWSBedrockAuth)


def test_auth_factory_vertex():
    """create_auth_provider creates VertexAIAuth."""
    from autoforge.engine.auth import VertexAIAuth, create_auth_provider

    auth = create_auth_provider("anthropic", {}, {
        "anthropic": {
            "auth_method": "vertex_ai",
            "project_id": "proj-123",
            "region": "us-east5",
        }
    })
    assert isinstance(auth, VertexAIAuth)


def test_auth_factory_codex_oauth():
    """create_auth_provider creates CodexOAuthAuth."""
    from autoforge.engine.auth import CodexOAuthAuth, create_auth_provider

    auth = create_auth_provider("openai", {}, {
        "openai": {"auth_method": "codex_oauth"}
    })
    assert isinstance(auth, CodexOAuthAuth)


def test_auth_factory_device_code():
    """create_auth_provider creates DeviceCodeAuth."""
    from autoforge.engine.auth import DeviceCodeAuth, create_auth_provider

    auth = create_auth_provider("openai", {}, {
        "openai": {"auth_method": "device_code"}
    })
    assert isinstance(auth, DeviceCodeAuth)


def test_setup_wizard_no_precheck():
    """PROVIDERS dict does not pre-enable any provider."""
    from autoforge.cli.setup_wizard import PROVIDERS

    assert "anthropic" in PROVIDERS
    assert "openai" in PROVIDERS
    assert "google" in PROVIDERS
    # Verify auth_methods are defined
    for pid, info in PROVIDERS.items():
        assert "auth_methods" in info, f"{pid} missing auth_methods"


def test_setup_wizard_auth_methods():
    """Each provider has valid auth method options."""
    from autoforge.cli.setup_wizard import PROVIDERS

    # Anthropic: api_key + bearer + oauth2 + bedrock + vertex_ai
    ant_methods = [m["value"] for m in PROVIDERS["anthropic"]["auth_methods"]]
    assert "api_key" in ant_methods
    assert "oauth_bearer" in ant_methods
    assert "oauth2_client_credentials" in ant_methods
    assert "bedrock" in ant_methods
    assert "vertex_ai" in ant_methods

    # OpenAI: api_key + bearer + codex_oauth + device_code + oauth2
    oai_methods = [m["value"] for m in PROVIDERS["openai"]["auth_methods"]]
    assert "api_key" in oai_methods
    assert "oauth_bearer" in oai_methods
    assert "codex_oauth" in oai_methods
    assert "device_code" in oai_methods
    assert "oauth2_client_credentials" in oai_methods

    # Google: api_key + adc + service_account (no OAuth)
    ggl_methods = [m["value"] for m in PROVIDERS["google"]["auth_methods"]]
    assert "api_key" in ggl_methods
    assert "adc" in ggl_methods
    assert "service_account" in ggl_methods


def test_llm_router_has_auth_providers():
    """LLMRouter has _auth_providers dict."""
    from autoforge.engine.config import ForgeConfig
    from autoforge.engine.llm_router import LLMRouter

    config = ForgeConfig(api_keys={"anthropic": "fake"})
    router = LLMRouter(config)
    assert hasattr(router, "_auth_providers")
    assert isinstance(router._auth_providers, dict)


def test_env_example_no_required_anthropic():
    """.env.example does not say 'Required: Anthropic'."""
    env_path = PROJECT_ROOT / ".env.example"
    assert env_path.exists()
    content = env_path.read_text(encoding="utf-8")
    assert "Required: Anthropic" not in content
    # All three providers should be present
    assert "ANTHROPIC_API_KEY" in content
    assert "OPENAI_API_KEY" in content
    assert "GOOGLE_API_KEY" in content
    # OAuth/proxy env vars
    assert "OPENAI_BASE_URL" in content
    assert "GOOGLE_APPLICATION_CREDENTIALS" in content
    # Bedrock env vars
    assert "CLAUDE_CODE_USE_BEDROCK" in content
    assert "AWS_REGION" in content
    # Vertex AI env vars
    assert "CLAUDE_CODE_USE_VERTEX" in content
    assert "ANTHROPIC_VERTEX_PROJECT_ID" in content
    assert "CLOUD_ML_REGION" in content


def test_pyproject_has_google_auth():
    """pyproject.toml includes google-auth dependency."""
    toml_path = PROJECT_ROOT / "pyproject.toml"
    content = toml_path.read_text(encoding="utf-8")
    assert "google-auth" in content


# ────────────────────────────────────────────
# Phase 11: Reasoning Extension & Article Verification
# ────────────────────────────────────────────


def test_reasoning_extension_imports():
    """Reasoning extension module imports correctly."""
    from autoforge.engine.reasoning_extension import (  # noqa: F401
        ReasoningExtensionEngine,
        MinimalKernel,
        NumberedConclusion,
        GrowthOperator,
        ConclusionType,
        PublicationWorthiness,
        PublicationGate,
        FormalVerificationBridge,
        ReasoningRound,
    )


def test_minimal_kernel_creation():
    """MinimalKernel creates with default Φ-Kernel axioms."""
    from autoforge.engine.reasoning_extension import MinimalKernel
    kernel = MinimalKernel.create_default()
    assert kernel.name == "Φ-Kernel"
    assert len(kernel.axioms) == 5
    assert "Axiom (Φ1)" in kernel.axioms[0]
    assert len(kernel.domain_seeds) >= 5
    assert kernel.counter == 0
    # Test counter increment
    n = kernel.next_conclusion_number()
    assert n == 1
    assert kernel.counter == 1


def test_numbered_conclusion():
    """NumberedConclusion formats correctly and serializes."""
    from autoforge.engine.reasoning_extension import (
        NumberedConclusion, ConclusionType, GrowthOperator,
        PublicationWorthiness,
    )
    c = NumberedConclusion(
        number=42,
        conclusion_type=ConclusionType.THEOREM,
        statement="For any compact Riemannian manifold M, the spectral zeta function ζ_M(s) admits meromorphic continuation to ℂ.",
        proof_sketch="By standard heat kernel asymptotics.",
        growth_operator=GrowthOperator.SPECTRAL_DECOMPOSE,
        domain="spectral geometry",
        worthiness=PublicationWorthiness.PUBLISHABLE,
    )
    # Format
    formatted = c.format_academic()
    assert "Theorem 42" in formatted
    assert "spectral zeta" in formatted
    assert "□" in formatted
    # Serialize
    d = c.to_dict()
    assert d["number"] == 42
    assert d["type"] == "theorem"
    assert d["worthiness"] == "publishable"
    # Deserialize
    c2 = NumberedConclusion.from_dict(d)
    assert c2.number == 42
    assert c2.conclusion_type == ConclusionType.THEOREM


def test_reasoning_extension_engine_instantiation():
    """ReasoningExtensionEngine instantiates with default kernel."""
    from autoforge.engine.reasoning_extension import ReasoningExtensionEngine
    engine = ReasoningExtensionEngine()
    assert engine.kernel.name == "Φ-Kernel"
    assert engine.conclusion_count == 0
    stats = engine.get_stats()
    assert stats["total_conclusions"] == 0
    assert stats["kernel_name"] == "Φ-Kernel"


def test_reasoning_extension_persistence():
    """ReasoningExtensionEngine saves and loads state."""
    import tempfile
    from autoforge.engine.reasoning_extension import (
        ReasoningExtensionEngine, NumberedConclusion, ConclusionType,
        GrowthOperator, PublicationWorthiness,
    )
    engine = ReasoningExtensionEngine()
    # Add a conclusion manually for testing
    c = NumberedConclusion(
        number=1,
        conclusion_type=ConclusionType.THEOREM,
        statement="Test theorem statement.",
        domain="test",
        worthiness=PublicationWorthiness.PUBLISHABLE,
    )
    engine._conclusions.append(c)
    engine._kernel._conclusion_counter = 1

    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "ext_state"
        engine.save(save_dir)

        # Load into new engine
        engine2 = ReasoningExtensionEngine()
        engine2.load(save_dir)
        assert engine2.conclusion_count == 1
        assert engine2._conclusions[0].statement == "Test theorem statement."
        assert engine2.kernel.counter == 1


def test_reasoning_extension_report_generation():
    """ReasoningExtensionEngine generates markdown and latex reports."""
    from autoforge.engine.reasoning_extension import (
        ReasoningExtensionEngine, NumberedConclusion, ConclusionType,
        GrowthOperator, PublicationWorthiness,
    )
    engine = ReasoningExtensionEngine()
    engine._conclusions = [
        NumberedConclusion(
            number=1,
            conclusion_type=ConclusionType.THEOREM,
            statement="The spectral gap of M is bounded below.",
            domain="spectral theory",
            worthiness=PublicationWorthiness.PUBLISHABLE,
        ),
        NumberedConclusion(
            number=2,
            conclusion_type=ConclusionType.CONJECTURE,
            statement="The partition function converges for Re(β) > 0.",
            domain="statistical mechanics",
            worthiness=PublicationWorthiness.EXCEPTIONAL,
        ),
    ]
    md = engine.generate_report(latex=False)
    assert "Autonomous Reasoning Extension" in md
    assert "Theorem 1" in md or "spectral gap" in md
    latex = engine.generate_report(latex=True)
    assert r"\begin{theorem}" in latex
    assert r"\begin{conjecture}" in latex


def test_publication_gate_dedup():
    """PublicationGate detects duplicate conclusions."""
    from autoforge.engine.reasoning_extension import (
        PublicationGate, NumberedConclusion, ConclusionType,
        PublicationWorthiness,
    )
    gate = PublicationGate()
    c1 = NumberedConclusion(
        number=1,
        conclusion_type=ConclusionType.THEOREM,
        statement="The eigenvalues of L are non-negative.",
    )
    gate.register_existing([c1])

    c2 = NumberedConclusion(
        number=2,
        conclusion_type=ConclusionType.THEOREM,
        statement="The eigenvalues of L are non-negative.",  # Same
    )
    # Same hash → should be redundant
    assert c2.content_hash == c1.content_hash


def test_growth_operators():
    """All GrowthOperator values are accessible."""
    from autoforge.engine.reasoning_extension import GrowthOperator
    assert len(GrowthOperator) == 12
    assert GrowthOperator.LIFT.value == "lift"
    assert GrowthOperator.ERGODIC_LIMIT.value == "ergodic_limit"


def test_article_verifier_imports():
    """Article verifier module imports correctly."""
    from autoforge.engine.article_verifier import (  # noqa: F401
        ArticleVerifier,
        ArticleParser,
        FormalizationPipeline,
        VerifiableClaim,
        ArticleVerificationReport,
        ClaimType,
        VerificationStatus,
    )


def test_article_parser_latex():
    """ArticleParser extracts claims from LaTeX environments."""
    from autoforge.engine.article_verifier import ArticleParser, ClaimType
    parser = ArticleParser()
    text = r"""
\begin{theorem}[Main]
For all $n \geq 1$, $\sum_{k=1}^n k = n(n+1)/2$.
\end{theorem}
\begin{proof}
By induction on $n$.
\end{proof}
\begin{lemma}
The sequence is monotone increasing.
\end{lemma}
"""
    claims = parser._extract_latex(text)
    assert len(claims) >= 2
    theorem_claims = [c for c in claims if c.claim_type == ClaimType.THEOREM]
    assert len(theorem_claims) >= 1
    assert "sum" in theorem_claims[0].statement.lower() or "\\sum" in theorem_claims[0].statement


def test_article_parser_markdown():
    """ArticleParser extracts claims from Markdown format."""
    from autoforge.engine.article_verifier import ArticleParser, ClaimType
    parser = ArticleParser()
    text = """
## Theorem 1. The main result

For all primes p, the Legendre symbol satisfies quadratic reciprocity.

**Proof.** By Gauss's lemma.

## Lemma 2

The Jacobi symbol extends the Legendre symbol to composite moduli.
"""
    claims = parser._extract_markdown(text)
    assert len(claims) >= 1


def test_verification_report_formatting():
    """ArticleVerificationReport formats summary correctly."""
    from autoforge.engine.article_verifier import (
        ArticleVerificationReport, VerifiableClaim,
        ClaimType, VerificationStatus,
    )
    report = ArticleVerificationReport(
        title="Test Article",
        total_claims=3,
        verified=2,
        failed=1,
        overall_confidence=0.67,
        assessment="Good: Majority verified.",
    )
    summary = report.format_summary()
    assert "Test Article" in summary
    assert "3" in summary


def test_orchestrator_has_reasoning_extension():
    """Orchestrator has reasoning extension and article verifier attributes."""
    from autoforge.engine.orchestrator import Orchestrator
    from autoforge.engine.config import ForgeConfig
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-000")
    config = ForgeConfig.from_env()
    orch = Orchestrator(config)
    assert hasattr(orch, '_reasoning_extension')
    assert hasattr(orch, '_article_verifier')
    assert hasattr(orch, '_run_reasoning_extension')
    assert hasattr(orch, '_run_article_verification')


def test_theoretical_reasoning_has_extension_methods():
    """TheoreticalReasoningEngine has the new integration methods."""
    from autoforge.engine.theoretical_reasoning import TheoreticalReasoningEngine
    engine = TheoreticalReasoningEngine()
    assert hasattr(engine, 'run_reasoning_extension')
    assert hasattr(engine, 'verify_article_claims')
    assert callable(engine.run_reasoning_extension)
    assert callable(engine.verify_article_claims)


# ────────────────────────────────────────────
# Academic Stack (D5 — 6 new checks)
# ────────────────────────────────────────────


def test_autonomous_discovery_imports():
    """autonomous_discovery module imports with DomainContext."""
    from autoforge.engine.autonomous_discovery import (  # noqa: F401
        DiscoveryOrchestrator, DomainContext, SUPERSPACE_MODEL_SETS,
        ALGEBRAIC_GEOMETRY, DYNAMICAL_SYSTEMS, detect_domain_context,
    )


def test_paper_formalizer_imports():
    """paper_formalizer module imports with report generator."""
    from autoforge.engine.paper_formalizer import (  # noqa: F401
        PaperFormalizer, FormalizationReport, FormalizationUnit,
        LeanCodeGenerator, FormalizationStatus,
    )
    pf = PaperFormalizer()
    tmpl = pf.get_lean_project_template()
    assert "lakefile.lean" in tmpl


def test_cloud_prover_imports():
    """cloud_prover module imports."""
    from autoforge.engine.cloud_prover import (  # noqa: F401
        CloudProver, CloudProverConfig, ProofCache, ProofJob,
        JobStatus, CloudBackend,
    )


def test_discovery_config_sane_defaults():
    """DiscoveryConfig has sane defaults (max_rounds>0, min_confidence in [0,1])."""
    from autoforge.engine.autonomous_discovery import DiscoveryConfig
    dc = DiscoveryConfig()
    assert dc.max_rounds > 0
    assert 0 <= dc.min_confidence <= 1


def test_formalization_report_compute_score():
    """FormalizationReport.compute_score() returns correct value for known inputs."""
    from autoforge.engine.paper_formalizer import FormalizationReport
    report = FormalizationReport(
        paper_title="Test", paper_source="t.pdf",
        total_statements=10, lean_proved=3, lean_sorry=2,
        numerically_verified=2, computationally_reproduced=1,
    )
    score = report.compute_score()
    expected = (3*1.0 + 2*0.5 + 2*0.7 + 1*0.8) / 10  # 0.62
    assert abs(score - expected) < 1e-6, f"Expected {expected}, got {score}"


def test_proof_cache_empty_get():
    """ProofCache get on empty cache returns None."""
    from autoforge.engine.cloud_prover import ProofCache
    cache = ProofCache()
    assert cache.get("nonexistent code") is None


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
            ("has_api_key property", test_config_has_api_key),
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
            ("Git available detection", test_git_available_detection),
            ("Git version async", test_git_version_async),
        ]),
        ("Pipeline Hardening", [
            ("AgentResult metrics fields", test_agent_result_metrics_fields),
            ("Anti-spin detection constants", test_anti_spin_constants),
            ("Build gate enforcement", test_build_gate_enforcement),
            ("File overlap detection", test_file_overlap_detection),
            ("Smoke check: missing files", test_smoke_check_missing_files),
            ("Smoke check: syntax error", test_smoke_check_syntax_error),
            ("Smoke check: valid file", test_smoke_check_valid_file),
            ("Constitution: builder mandatory rules", test_constitution_builder_mandatory_rules),
            ("Constitution: architect composition", test_constitution_architect_composition),
            ("Constitution: quality gates enforcement", test_constitution_quality_gates_enforcement),
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
            ("paper subcommand parsing", test_cli_paper_subcommand),
        ]),
        ("Paper Repro", [
            ("Signal extraction + no-key simulation", test_paper_repro_signals_and_simulation),
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
        ("Web Tools & Code Search", [
            ("Tools package imports", test_tools_package_imports),
            ("Config new fields (web, checkpoints, TDD)", test_config_new_fields),
            ("Config with overrides", test_config_with_overrides),
            ("CLI --confirm flag parsing", test_cli_confirm_flag),
            ("CLI --tdd flag parsing", test_cli_tdd_flag),
            ("CLI config overrides: confirm", test_cli_config_overrides_confirm),
            ("CLI config overrides: tdd", test_cli_config_overrides_tdd),
            ("grep_search: basic pattern match", test_grep_search_basic),
            ("grep_search: file_glob filter", test_grep_search_file_glob),
            ("grep_search: empty pattern rejected", test_grep_search_empty_pattern),
            ("Web tool schemas valid", test_web_tool_schemas),
            ("HTML strip fallback", test_html_strip_fallback),
            ("Orchestrator _checkpoint method", test_orchestrator_checkpoint_method),
            ("Orchestrator _tdd_loop method", test_orchestrator_tdd_loop_method),
            ("Test command detection: npm", test_detect_test_command_npm),
            ("Test command detection: pytest", test_detect_test_command_pytest),
            ("Test command detection: none", test_detect_test_command_none),
            ("Agents have new tools", test_agents_have_new_tools),
            ("Agents web tools disabled", test_agents_web_tools_disabled),
            ("Constitution: builder has new tools", test_constitution_builder_has_new_tools),
            ("Constitution: architect has web tools", test_constitution_architect_has_web_tools),
            ("Constitution: quality gates agent capabilities", test_constitution_quality_gates_has_agent_capabilities),
            ("pyproject.toml has new dependencies", test_pyproject_has_new_dependencies),
        ]),
        ("Auth Module & OAuth", [
            ("Auth module imports", test_auth_module_imports),
            ("ApiKeyAuth provider", test_auth_api_key_provider),
            ("OAuthBearerAuth provider", test_auth_bearer_provider),
            ("TokenResult expiry", test_auth_token_result_expiry),
            ("Auth factory fallback to ApiKeyAuth", test_auth_factory_fallback),
            ("Auth factory bearer", test_auth_factory_bearer),
            ("Auth factory OAuth2", test_auth_factory_oauth2),
            ("Auth factory ADC", test_auth_factory_adc),
            ("AWSBedrockAuth provider", test_auth_bedrock_provider),
            ("VertexAIAuth provider", test_auth_vertex_provider),
            ("CodexOAuthAuth provider", test_auth_codex_oauth_provider),
            ("DeviceCodeAuth provider", test_auth_device_code_provider),
            ("Auth factory Bedrock", test_auth_factory_bedrock),
            ("Auth factory Vertex AI", test_auth_factory_vertex),
            ("Auth factory Codex OAuth", test_auth_factory_codex_oauth),
            ("Auth factory Device Code", test_auth_factory_device_code),
            ("Config auth_config field", test_config_auth_config_field),
            ("Config has_api_key with auth_config", test_config_has_api_key_with_auth_config),
            ("Setup wizard no pre-check", test_setup_wizard_no_precheck),
            ("Setup wizard auth methods", test_setup_wizard_auth_methods),
            ("LLM router has _auth_providers", test_llm_router_has_auth_providers),
            (".env.example no required Anthropic", test_env_example_no_required_anthropic),
            ("pyproject.toml has google-auth", test_pyproject_has_google_auth),
        ]),
        ("Reasoning Extension & Article Verification", [
            ("Reasoning extension imports", test_reasoning_extension_imports),
            ("MinimalKernel creation", test_minimal_kernel_creation),
            ("NumberedConclusion format + serialize", test_numbered_conclusion),
            ("ReasoningExtensionEngine instantiation", test_reasoning_extension_engine_instantiation),
            ("ReasoningExtensionEngine persistence", test_reasoning_extension_persistence),
            ("Report generation (markdown + latex)", test_reasoning_extension_report_generation),
            ("PublicationGate dedup", test_publication_gate_dedup),
            ("GrowthOperator completeness", test_growth_operators),
            ("Article verifier imports", test_article_verifier_imports),
            ("ArticleParser LaTeX extraction", test_article_parser_latex),
            ("ArticleParser Markdown extraction", test_article_parser_markdown),
            ("VerificationReport formatting", test_verification_report_formatting),
            ("Orchestrator has reasoning extension", test_orchestrator_has_reasoning_extension),
            ("TheoreticalReasoning has extension methods", test_theoretical_reasoning_has_extension_methods),
        ]),
        ("Academic Stack", [
            ("autonomous_discovery imports", test_autonomous_discovery_imports),
            ("paper_formalizer imports", test_paper_formalizer_imports),
            ("cloud_prover imports", test_cloud_prover_imports),
            ("DiscoveryConfig sane defaults", test_discovery_config_sane_defaults),
            ("FormalizationReport compute_score", test_formalization_report_compute_score),
            ("ProofCache empty get -> None", test_proof_cache_empty_get),
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
