from __future__ import annotations

import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path

from autoforge.engine.agent_base import AgentBase, FileToolsMixin, ToolDefinition
from autoforge.engine.config import ForgeConfig
from autoforge.engine.deploy_guide import generate_deploy_guide, write_delivery_harness_artifacts
from autoforge.engine.git_manager import GitError, GitManager
from autoforge.engine.llm_router import ContentBlock, LLMResponse, LLMRouter, TaskComplexity, Usage
from autoforge.engine.lock_manager import LockManager
from autoforge.engine.sandbox import create_sandbox
from autoforge.engine.task_dag import Task, TaskDAG, TaskPhase


def _make_local_tmp_dir() -> Path:
    path = Path.cwd() / ".tmp_development_harness_tests" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


class _DummyLLM:
    def __init__(self) -> None:
        self._calls = 0

    async def call(self, *args, **kwargs) -> LLMResponse:
        self._calls += 1
        if self._calls == 1:
            return LLMResponse(
                content=[
                    ContentBlock(
                        type="tool_use",
                        id="tool-1",
                        name="write_file",
                        input={"path": "hello.txt", "content": "hello harness"},
                    )
                ],
                stop_reason="tool_use",
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        return LLMResponse(
            content=[ContentBlock(type="text", text="done")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=8, output_tokens=3),
        )


class _HarnessAgent(FileToolsMixin, AgentBase):
    ROLE = "builder"
    MAX_TURNS = 4

    def __init__(self, config: ForgeConfig, llm: _DummyLLM, *, working_dir: Path) -> None:
        self.working_dir = working_dir
        super().__init__(config, llm)  # type: ignore[arg-type]

    def _register_tools(self) -> None:
        self._tools = [
            ToolDefinition(
                name="write_file",
                description="Write file",
                input_schema={"type": "object"},
                handler=self._handle_write_file,
            )
        ]

    def build_prompt(self, context: dict[str, object]) -> str:
        return f"Build: {context.get('goal', 'task')}"


def test_agent_harness_writes_manifest_trace_and_verdict():
    tmp_path = _make_local_tmp_dir()
    try:
        project_dir = tmp_path / "workspace" / "agent-demo"
        project_dir.mkdir(parents=True, exist_ok=True)
        config = ForgeConfig(project_root=tmp_path, workspace_dir=project_dir.parent, run_id="agent-run-1")
        agent = _HarnessAgent(config, _DummyLLM(), working_dir=project_dir)

        result = asyncio.run(agent.run({"goal": "write a file"}))

        assert result.success is True
        harness_root = project_dir / ".autoforge" / "development_harness" / "agent_harness" / "builder"
        runs = list(harness_root.iterdir())
        assert len(runs) == 1
        run_dir = runs[0]
        manifest = json.loads((run_dir / "agent_run.json").read_text(encoding="utf-8"))
        verdict = json.loads((run_dir / "agent_verdict.json").read_text(encoding="utf-8"))
        trace_lines = (run_dir / "tool_call_trace.jsonl").read_text(encoding="utf-8").splitlines()

        assert manifest["artifact_type"] == "agent_run"
        assert manifest["agent_name"] == "builder"
        assert verdict["artifact_type"] == "agent_verdict"
        assert verdict["success"] is True
        assert verdict["files_written"] == 1
        assert len(trace_lines) == 1
        trace = json.loads(trace_lines[0])
        assert trace["tool_name"] == "write_file"
        assert (project_dir / "hello.txt").read_text(encoding="utf-8") == "hello harness"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_llm_harness_writes_call_bundle_budget_ledger_and_fallback_receipt():
    async def _run() -> None:
        tmp_path = _make_local_tmp_dir()
        try:
            project_dir = tmp_path / "workspace" / "llm-demo"
            project_dir.mkdir(parents=True, exist_ok=True)
            config = ForgeConfig(
                project_root=tmp_path,
                workspace_dir=project_dir.parent,
                run_id="llm-run-1",
                model_fast="claude-3-5-sonnet",
            )
            router = LLMRouter(config)
            router.set_harness_project_dir(project_dir)
            router._get_client = lambda provider: object()  # type: ignore[method-assign]

            async def _noop(*args, **kwargs) -> None:
                return None

            async def _fake_call_anthropic(*args, **kwargs) -> LLMResponse:
                return LLMResponse(
                    content=[ContentBlock(type="text", text="ok")],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=12, output_tokens=7),
                )

            router._ensure_fresh_token = _noop  # type: ignore[method-assign]
            router._call_anthropic = _fake_call_anthropic  # type: ignore[method-assign]

            response = await router.call("hello", complexity=TaskComplexity.STANDARD)
            assert response.stop_reason == "end_turn"

            class _ModelNotFound(Exception):
                status_code = 404
                body = {"error": {"code": "model_not_found"}}

            attempts = {"count": 0}

            async def _fake_openai_once(model: str, *args, **kwargs) -> LLMResponse:
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise _ModelNotFound("missing model")
                return LLMResponse(
                    content=[ContentBlock(type="text", text=f"fallback:{model}")],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=5, output_tokens=2),
                )

            router._call_openai_once = _fake_openai_once  # type: ignore[method-assign]
            await router._call_openai(
                "gpt-5.2-codex",
                64,
                "",
                [{"role": "user", "content": "fallback"}],
                None,
            )

            llm_root = project_dir / ".autoforge" / "development_harness" / "llm_harness"
            call_bundle = json.loads((llm_root / "calls" / "call_00001.json").read_text(encoding="utf-8"))
            budget_entries = (llm_root / "budget_ledger.jsonl").read_text(encoding="utf-8").splitlines()
            budget_summary = json.loads((llm_root / "budget_summary.json").read_text(encoding="utf-8"))
            fallback_entries = (llm_root / "fallback_receipts.jsonl").read_text(encoding="utf-8").splitlines()

            assert call_bundle["artifact_type"] == "llm_call_bundle"
            assert call_bundle["requested_model"] == "claude-3-5-sonnet"
            assert call_bundle["ok"] is True
            assert len(budget_entries) == 1
            assert budget_summary["artifact_type"] == "llm_budget_summary"
            assert len(fallback_entries) == 1
            fallback = json.loads(fallback_entries[0])
            assert fallback["fallback_candidate"]
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    asyncio.run(_run())


def test_task_harness_writes_graph_attempts_and_verdict():
    tmp_path = _make_local_tmp_dir()
    try:
        harness_root = tmp_path / "task_harness"
        dag = TaskDAG()
        dag.set_harness_root(harness_root)
        dag.add_task(Task(id="task-1", description="Build header", phase=TaskPhase.BUILD, acceptance_criteria="Header renders"))
        dag.mark_in_progress("task-1", "builder-1")
        dag.mark_done("task-1", "implemented")

        graph = json.loads((harness_root / "task_graph.json").read_text(encoding="utf-8"))
        verdict = json.loads((harness_root / "task_verdicts.json").read_text(encoding="utf-8"))
        attempts = (harness_root / "task_attempts.jsonl").read_text(encoding="utf-8").splitlines()

        assert graph["artifact_type"] == "task_graph"
        assert graph["tasks"][0]["id"] == "task-1"
        assert verdict["artifact_type"] == "task_verdicts"
        assert verdict["all_done"] is True
        assert len(attempts) == 2
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_execution_harness_writes_command_receipts_and_lease_artifacts():
    async def _run() -> None:
        tmp_path = _make_local_tmp_dir()
        try:
            project_dir = tmp_path / "workspace" / "exec-demo"
            project_dir.mkdir(parents=True, exist_ok=True)
            config = ForgeConfig(
                project_root=tmp_path,
                workspace_dir=project_dir.parent,
                execution_backend="subprocess",
                run_id="exec-run-1",
            )
            sandbox = create_sandbox(config, project_dir)
            async with sandbox:
                result = await sandbox.exec_args([sys.executable, "-c", "print('hello')"], capability="test")
            assert result.exit_code == 0

            harness_root = project_dir / ".autoforge" / "development_harness" / "execution_harness"
            receipts = (harness_root / "command_receipts.jsonl").read_text(encoding="utf-8").splitlines()
            env_payload = json.loads((harness_root / "execution_environment.json").read_text(encoding="utf-8"))
            policy_payload = json.loads((harness_root / "sandbox_policy.json").read_text(encoding="utf-8"))

            lock_dir = tmp_path / ".locks"
            lock_manager = LockManager(lock_dir, harness_root=harness_root)
            assert lock_manager.try_claim("task-1", "agent-a") is True
            assert lock_manager.release("task-1", "agent-a") is True

            lease_artifact = json.loads((harness_root / "lease_artifact.json").read_text(encoding="utf-8"))
            lease_events = (harness_root / "lease_events.jsonl").read_text(encoding="utf-8").splitlines()

            assert len(receipts) >= 1
            assert env_payload["artifact_type"] == "execution_environment"
            assert policy_payload["artifact_type"] == "sandbox_policy"
            assert lease_artifact["artifact_type"] == "lease_artifact"
            assert len(lease_events) >= 2
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    asyncio.run(_run())


def test_git_manager_writes_worktree_manifest_and_merge_verdict():
    async def _run() -> None:
        tmp_path = _make_local_tmp_dir()
        try:
            project_dir = tmp_path / "workspace" / "git-demo"
            project_dir.mkdir(parents=True, exist_ok=True)
            manager = GitManager(project_dir)

            async def _fake_run_git(*args: str, cwd: Path | None = None) -> str:
                if args[:1] == ("init",):
                    (project_dir / ".git").mkdir(parents=True, exist_ok=True)
                    return ""
                if args[:3] == ("checkout", "-b", "main"):
                    return ""
                if args[:2] == ("worktree", "add"):
                    Path(args[4]).mkdir(parents=True, exist_ok=True)
                    return ""
                if args[:3] == ("diff", "--cached", "--quiet"):
                    raise GitError("changes present")
                if args[:2] == ("rev-parse", "--short"):
                    return "abc1234"
                if args[:2] == ("worktree", "remove"):
                    return ""
                if args[:2] == ("branch", "-d"):
                    return ""
                if args[:2] == ("merge", "--abort"):
                    return ""
                return ""

            manager._run_git = _fake_run_git  # type: ignore[method-assign]
            await manager.init_repo()
            worktree = await manager.create_worktree("feature/header")
            (worktree / "index.ts").write_text("export const x = 1;\n", encoding="utf-8")
            commit_hash = await manager.commit_worktree("feature/header", "Add header", "task-1")
            await manager.merge_branch("feature/header")

            harness_root = project_dir / ".autoforge" / "development_harness" / "execution_harness"
            worktree_manifest = json.loads((harness_root / "worktree_manifest.json").read_text(encoding="utf-8"))
            merge_verdict = json.loads((harness_root / "merge_verdict.json").read_text(encoding="utf-8"))

            assert commit_hash == "abc1234"
            assert worktree_manifest["artifact_type"] == "worktree_manifest"
            assert worktree_manifest["worktree_count"] >= 1
            assert merge_verdict["artifact_type"] == "merge_verdict"
            assert merge_verdict["success"] is True
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    asyncio.run(_run())


def test_delivery_harness_writes_manifest_requirements_and_verdict():
    tmp_path = _make_local_tmp_dir()
    try:
        project_dir = tmp_path / "workspace" / "delivery-demo"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "README.md").write_text("# demo\n", encoding="utf-8")
        (project_dir / "package.json").write_text(
            json.dumps(
                {
                    "dependencies": {"next": "15.0.0"},
                    "scripts": {"build": "next build", "dev": "next dev"},
                }
            ),
            encoding="utf-8",
        )
        (project_dir / ".env.example").write_text("OPENAI_API_KEY=\nNEXTAUTH_SECRET=\n", encoding="utf-8")

        guide = generate_deploy_guide(project_dir, "delivery-demo")
        artifacts = write_delivery_harness_artifacts(project_dir, "delivery-demo")

        deploy_manifest = json.loads(artifacts["deploy_manifest"].read_text(encoding="utf-8"))
        env_requirements = json.loads(artifacts["environment_requirements"].read_text(encoding="utf-8"))
        publish_verdict = json.loads(artifacts["publish_verdict"].read_text(encoding="utf-8"))

        assert "Vercel Deployment Guide" in guide
        assert deploy_manifest["artifact_type"] == "deploy_manifest"
        assert deploy_manifest["framework"] == "nextjs"
        assert env_requirements["artifact_type"] == "environment_requirements"
        assert len(env_requirements["environment_variables"]) == 2
        assert publish_verdict["artifact_type"] == "publish_verdict"
        assert publish_verdict["publish_ready"] is True
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
