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
    from engine.config import ForgeConfig  # noqa: F401
    from engine.llm_router import LLMRouter, TaskComplexity, BudgetExceededError  # noqa: F401
    from engine.agent_base import AgentBase, AgentResult, ToolDefinition  # noqa: F401
    from engine.task_dag import TaskDAG, Task, TaskPhase, TaskStatus  # noqa: F401
    from engine.lock_manager import LockManager  # noqa: F401
    from engine.git_manager import GitManager, GitError  # noqa: F401
    from engine.sandbox import (  # noqa: F401
        SandboxBase, SubprocessSandbox, DockerSandbox,
        SandboxResult, create_sandbox,
    )
    from engine.orchestrator import Orchestrator  # noqa: F401


def test_agent_imports():
    from engine.agents import AGENT_REGISTRY
    assert len(AGENT_REGISTRY) == 7, f"Expected 7 agents, got {len(AGENT_REGISTRY)}"
    from engine.agents.director import DirectorAgent, DirectorFixAgent  # noqa: F401
    from engine.agents.architect import ArchitectAgent  # noqa: F401
    from engine.agents.builder import BuilderAgent  # noqa: F401
    from engine.agents.reviewer import ReviewerAgent  # noqa: F401
    from engine.agents.tester import TesterAgent  # noqa: F401
    from engine.agents.gardener import GardenerAgent  # noqa: F401


# ────────────────────────────────────────────
# Phase 2: CLI
# ────────────────────────────────────────────

def test_cli_parse_args():
    saved = sys.argv
    try:
        sys.argv = [
            "forge.py", "Build a Todo app",
            "--budget", "5.0", "--agents", "2", "--verbose",
        ]
        from forge import parse_args
        args = parse_args()
        assert args.description == "Build a Todo app"
        assert args.budget == 5.0
        assert args.agents == 2
        assert args.verbose is True
    finally:
        sys.argv = saved


# ────────────────────────────────────────────
# Phase 3: Unit Tests
# ────────────────────────────────────────────

def test_forge_config():
    from engine.config import ForgeConfig
    config = ForgeConfig()
    assert config.budget_limit_usd == 10.0
    assert config.max_agents == 3
    assert len(config.run_id) == 12
    assert config.workspace_dir.name == "workspace"
    assert config.constitution_dir.name == "constitution"


def test_forge_config_from_env():
    from engine.config import ForgeConfig
    config = ForgeConfig.from_env(budget_limit_usd=5.0, max_agents=2)
    assert config.budget_limit_usd == 5.0
    assert config.max_agents == 2


def test_forge_config_budget_tracking():
    from engine.config import ForgeConfig
    config = ForgeConfig(budget_limit_usd=1.0)
    config.record_usage("claude-sonnet-4-5-20250929", 1000, 500)
    assert config.total_input_tokens == 1000
    assert config.total_output_tokens == 500
    assert config.estimated_cost_usd > 0
    assert config.check_budget() is True
    # Exhaust budget
    config.record_usage("claude-opus-4-6", 10_000_000, 10_000_000)
    assert config.check_budget() is False


def test_task_dag_basic():
    from engine.task_dag import TaskDAG, Task, TaskPhase, TaskStatus
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
    from engine.task_dag import TaskDAG, Task
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
    from engine.task_dag import TaskDAG, Task, TaskStatus
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


def test_lock_manager():
    from engine.lock_manager import LockManager
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
    from engine.sandbox import SubprocessSandbox

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
    from engine.sandbox import create_sandbox, SubprocessSandbox
    from engine.config import ForgeConfig
    config = ForgeConfig(docker_enabled=False)
    d = Path(tempfile.mkdtemp())
    try:
        sb = create_sandbox(config, d)
        assert isinstance(sb, SubprocessSandbox)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_llm_router_instantiation():
    from engine.llm_router import LLMRouter, TaskComplexity
    from engine.config import ForgeConfig
    config = ForgeConfig(anthropic_api_key="fake-key")
    router = LLMRouter(config)
    m, t = router._select_model(TaskComplexity.HIGH)
    assert m == "claude-opus-4-6"
    assert t == 16384
    m2, t2 = router._select_model(TaskComplexity.STANDARD)
    assert m2 == "claude-sonnet-4-5-20250929"
    assert t2 == 8192


def test_git_manager_instantiation():
    from engine.git_manager import GitManager
    gm = GitManager(Path("/tmp/test-project"))
    assert gm.main_worktree == Path("/tmp/test-project")
    assert gm.worktrees_dir == Path("/tmp/worktrees")


def test_all_agents_instantiate():
    from engine.config import ForgeConfig
    from engine.llm_router import LLMRouter
    from engine.agents.director import DirectorAgent, DirectorFixAgent
    from engine.agents.architect import ArchitectAgent
    from engine.agents.builder import BuilderAgent
    from engine.agents.reviewer import ReviewerAgent
    from engine.agents.tester import TesterAgent
    from engine.agents.gardener import GardenerAgent
    from engine.sandbox import SubprocessSandbox

    config = ForgeConfig(anthropic_api_key="fake")
    llm = LLMRouter(config)
    wd = Path("/tmp/test")

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


def test_agent_build_prompts():
    from engine.config import ForgeConfig
    from engine.llm_router import LLMRouter
    from engine.agents.director import DirectorAgent
    from engine.agents.architect import ArchitectAgent
    from engine.agents.builder import BuilderAgent
    from engine.agents.reviewer import ReviewerAgent
    from engine.agents.tester import TesterAgent
    from engine.agents.gardener import GardenerAgent
    from engine.sandbox import SubprocessSandbox

    config = ForgeConfig(anthropic_api_key="fake")
    llm = LLMRouter(config)
    wd = Path("/tmp/test")
    sb = SubprocessSandbox(wd)
    spec = {"project_name": "test", "modules": []}

    assert len(DirectorAgent(config, llm).build_prompt({"project_description": "x"})) > 0
    assert len(ArchitectAgent(config, llm).build_prompt({"spec": spec})) > 0
    assert len(BuilderAgent(config, llm, wd, sb).build_prompt({"task": {"id": "T"}, "spec": spec})) > 0
    assert len(ReviewerAgent(config, llm, wd).build_prompt({"task": {"id": "T"}, "spec": spec})) > 0
    assert len(TesterAgent(config, llm, wd, sb).build_prompt({"spec": spec})) > 0
    assert len(GardenerAgent(config, llm, wd).build_prompt({"review": {}, "spec": spec})) > 0


def test_agent_parse_methods():
    from engine.config import ForgeConfig
    from engine.llm_router import LLMRouter
    from engine.agents.director import DirectorAgent, DirectorFixAgent
    from engine.agents.reviewer import ReviewerAgent
    from engine.agents.tester import TesterAgent

    config = ForgeConfig(anthropic_api_key="fake")
    llm = LLMRouter(config)

    d = DirectorAgent(config, llm)
    spec = d.parse_spec('```json\n{"project_name": "x", "modules": []}\n```')
    assert spec["project_name"] == "x"

    df = DirectorFixAgent(config, llm)
    fix = df.parse_fix_task('{"id": "FIX-1", "description": "fix"}')
    assert fix["id"] == "FIX-1"

    r = ReviewerAgent(config, llm, Path("/tmp"))
    review = r.parse_review('{"approved": true, "score": 8, "issues": [], "summary": "ok"}')
    assert review.approved is True and review.score == 8

    t = TesterAgent(config, llm, Path("/tmp"))
    results = t.parse_results('{"all_passed": true, "results": [], "summary": "ok"}')
    assert results.all_passed is True


# ────────────────────────────────────────────
# Phase 4: Integration
# ────────────────────────────────────────────

def test_orchestrator_instantiation():
    from engine.config import ForgeConfig
    from engine.orchestrator import Orchestrator
    config = ForgeConfig(anthropic_api_key="fake-key")
    orch = Orchestrator(config)
    assert orch.llm is not None
    assert orch.config is config
    assert orch._list_project_files() == []


def test_orchestrator_show_status():
    from engine.config import ForgeConfig
    from engine.orchestrator import Orchestrator
    config = ForgeConfig()
    orch = Orchestrator(config)
    orch.show_status()  # Should not raise


# ────────────────────────────────────────────
# Phase 5: Constitution files
# ────────────────────────────────────────────

def test_constitution_files_exist():
    base = PROJECT_ROOT / "constitution"
    assert (base / "CONSTITUTION.md").exists()
    assert (base / "quality_gates.md").exists()
    for agent in ["director", "architect", "builder", "reviewer", "tester", "gardener"]:
        path = base / "agents" / f"{agent}.md"
        assert path.exists(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"
    for workflow in ["spec", "build", "verify", "refactor", "deliver"]:
        path = base / "workflows" / f"{workflow}.md"
        assert path.exists(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty: {path}"


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
            ("Agent imports + registry", test_agent_imports),
        ]),
        ("CLI", [
            ("Argument parsing", test_cli_parse_args),
        ]),
        ("Unit: ForgeConfig", [
            ("Default construction", test_forge_config),
            ("from_env with overrides", test_forge_config_from_env),
            ("Budget tracking + exhaustion", test_forge_config_budget_tracking),
        ]),
        ("Unit: TaskDAG", [
            ("Basic operations + dependency resolution", test_task_dag_basic),
            ("Save and load (JSON persistence)", test_task_dag_save_load),
            ("Failure handling + BLOCKED status", test_task_dag_failure_handling),
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
        ]),
        ("Unit: GitManager", [
            ("Instantiation", test_git_manager_instantiation),
        ]),
        ("Unit: Agents", [
            ("All 7 agents instantiate", test_all_agents_instantiate),
            ("All build_prompt methods", test_agent_build_prompts),
            ("All parse methods", test_agent_parse_methods),
        ]),
        ("Integration", [
            ("Orchestrator instantiation", test_orchestrator_instantiation),
            ("Orchestrator show_status", test_orchestrator_show_status),
        ]),
        ("Constitution", [
            ("All constitution files exist + non-empty", test_constitution_files_exist),
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
