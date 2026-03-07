from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any

from autoforge.engine.config import ForgeConfig
from autoforge.engine.kernel.checkpoint import write_kernel_checkpoint
from autoforge.engine.kernel.evidence import ensure_research_evidence_pack
from autoforge.engine.kernel.plan import ExecutionPlanArtifact, write_execution_plan
from autoforge.engine.kernel.profiles import resolve_profile
from autoforge.engine.kernel.protocol import KernelEvent, KernelItem, KernelThread, KernelTurn
from autoforge.engine.kernel.run_store import KernelRunStore
from autoforge.engine.kernel.schema import write_kernel_json
from autoforge.engine.kernel.verdict import ContractVerdict, evaluate_contract_verdict, write_contract_verdict
from autoforge.engine.kernel.workspace import WorkspaceLock
from autoforge.engine.runtime.env import build_env_overrides
from autoforge.engine.runtime.runtime import ForgeRuntime


def _safe_relpath(base: Path, target: Path) -> str:
    try:
        return str(target.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(target)


def _trace_secrets_from_config(config: ForgeConfig) -> list[str]:
    secrets = list(getattr(config, "api_keys", {}).values())
    secrets.extend(
        [
            getattr(config, "search_api_key", ""),
            getattr(config, "github_token", ""),
            getattr(config, "webhook_secret", ""),
            getattr(config, "webhook_admin_secret", ""),
            getattr(config, "dag_federation_api_key", ""),
        ]
    )
    return [s for s in secrets if isinstance(s, str) and s]


def create_runtime_from_config(config: ForgeConfig, project_dir: Path) -> ForgeRuntime:
    return ForgeRuntime.create(
        project_dir,
        str(getattr(config, "run_id", "")),
        artifacts_enabled=bool(getattr(config, "artifacts_enabled", True)),
        trace_enabled=bool(getattr(config, "trace_enabled", False)),
        trace_capture_llm_content=bool(getattr(config, "trace_capture_llm_content", False)),
        trace_capture_command_output=bool(getattr(config, "trace_capture_command_output", False)),
        trace_capture_fs_snapshots=bool(getattr(config, "trace_capture_fs_snapshots", False)),
        trace_max_inline_chars=int(getattr(config, "trace_max_inline_chars", 20000) or 20000),
        trace_redact_secrets=bool(getattr(config, "trace_redact_secrets", True)),
        trace_secrets=_trace_secrets_from_config(config),
    )


class KernelSession:
    """Repository-local kernel runtime for one run/profile."""

    def __init__(
        self,
        *,
        config: ForgeConfig,
        project_dir: Path,
        runtime: ForgeRuntime,
        profile_name: str,
        operation: str,
        surface: str = "cli",
        metadata: dict[str, Any] | None = None,
        lock_ttl_seconds: int = 900,
    ) -> None:
        self.config = config
        self.project_dir = project_dir
        self.runtime = runtime
        self.profile = resolve_profile(profile_name)
        self.operation = str(operation or "run")
        self.surface = str(surface or "cli")
        self._lock_ttl_seconds = max(60, int(lock_ttl_seconds or 0))
        self.thread = KernelThread(
            run_id=str(getattr(config, "run_id", "") or runtime.run_id),
            lineage_id=str(getattr(config, "lineage_id", "") or getattr(config, "run_id", "") or runtime.run_id),
            parent_run_id=str(getattr(config, "parent_run_id", "") or ""),
            project_id=str(getattr(config, "project_id", "") or ""),
            profile=self.profile.name,
            surface=self.surface,
            metadata=dict(metadata or {}),
        )
        self._holder = f"{self.surface}:{os.getpid()}"
        self._lock = threading.RLock()
        self._seq = 0
        self._turns: dict[str, KernelTurn] = {}
        self._phase_history: list[dict[str, Any]] = []
        self._artifacts: list[dict[str, Any]] = []
        self._inbox: list[dict[str, Any]] = []
        self._closed = False

        self._workspace_lock = WorkspaceLock(project_dir)
        self._kernel_root = project_dir / ".autoforge" / "kernel"
        self._run_dir = self._kernel_root / "runs" / self.thread.run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self._run_dir / "events.jsonl"
        self._manifest_path = self._run_dir / "manifest.json"
        self._plan_path = self._run_dir / "execution_plan.json"
        self._checkpoint_path = self._run_dir / "checkpoint.json"
        self._verdict_path = self._run_dir / "contract_verdict.json"
        self._profile_path = self._run_dir / "profile.json"
        self._contracts_path = self._run_dir / "contracts.json"
        self._env_lock_path = self._run_dir / "environment.lock.json"
        self._artifact_manifest_path = self._run_dir / "artifact_manifest.json"
        self._inbox_path = self._run_dir / "inbox.json"
        self._research_root = self.project_dir / ".autoforge" / "research"
        self._research_brief_path = self._research_root / "brief.md"
        self._research_metrics_path = self._research_root / "metrics.json"
        self._plan: ExecutionPlanArtifact | None = None
        self._declared_outcomes: list[str] = []
        self._contract_verdict: ContractVerdict | None = None
        self._run_store = KernelRunStore(self._kernel_root / "run_store.sqlite3")

    @classmethod
    def open(
        cls,
        *,
        config: ForgeConfig,
        project_dir: Path,
        runtime: ForgeRuntime,
        profile_name: str,
        operation: str,
        surface: str = "cli",
        metadata: dict[str, Any] | None = None,
        lock_ttl_seconds: int = 900,
    ) -> "KernelSession":
        session = cls(
            config=config,
            project_dir=project_dir,
            runtime=runtime,
            profile_name=profile_name,
            operation=operation,
            surface=surface,
            metadata=metadata,
            lock_ttl_seconds=lock_ttl_seconds,
        )
        session._bootstrap()
        return session

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def declare_outcomes(self, *outcomes: str) -> None:
        for outcome in outcomes:
            token = str(outcome or "").strip()
            if token and token not in self._declared_outcomes:
                self._declared_outcomes.append(token)
        if self._declared_outcomes:
            self.thread.metadata["declared_outcomes"] = list(self._declared_outcomes)
            self._sync_run_store(status="active")

    def _bootstrap(self) -> None:
        lock_ok = self._workspace_lock.acquire(
            holder=self._holder,
            run_id=self.thread.run_id,
            ttl_seconds=self._lock_ttl_seconds,
            metadata={
                "thread_id": self.thread.thread_id,
                "profile": self.profile.name,
                "surface": self.surface,
            },
        )
        if not lock_ok:
            existing = self._workspace_lock.inspect()
            raise RuntimeError(
                "Workspace is already locked by another active run: "
                f"{existing.holder if existing is not None else 'unknown'}"
            )

        write_kernel_json(self._profile_path, self.profile.to_dict(), artifact_type="kernel_profile")
        write_kernel_json(
            self._contracts_path,
            {
                "profile": self.profile.name,
                "phase_graph": self.profile.phase_graph.to_dict(),
                "success_contract": self.profile.success_contract.to_dict(),
                "artifact_contract": self.profile.artifact_contract.to_dict(),
            },
            artifact_type="kernel_contracts",
        )
        write_kernel_json(self._env_lock_path, self._build_environment_lock(), artifact_type="environment_lock")
        self._initialize_execution_plan()
        self.register_artifact("kernel_profile", self._profile_path, required=True)
        self.register_artifact("kernel_contracts", self._contracts_path, required=True)
        self.register_artifact("env_lock", self._env_lock_path, required=True)
        if getattr(self.runtime, "trace", None) is not None:
            trace_path = getattr(self.runtime.trace, "trace_path", None)
            if isinstance(trace_path, Path):
                self.register_artifact("traces", trace_path, required=False)
        if self.profile.name == "development":
            self.register_artifact("code", self.project_dir, required=True)
        elif self.profile.name == "research":
            self.register_artifact("experiment_code", self.project_dir, required=True)
            self._write_research_brief()
        self._write_artifact_manifest()
        self._write_inbox()
        self._write_manifest(status="active")
        self._sync_run_store(status="active")
        self.register_artifact("kernel_manifest", self._manifest_path, required=True)
        self.register_artifact("execution_plan", self._plan_path, required=True)
        self._record("thread_started", self.thread.to_dict())

    def _initial_objective(self) -> str:
        objective = str(self.thread.metadata.get("objective", "") or "").strip()
        if objective:
            return objective
        if self.operation == "generate":
            name = str(self.thread.metadata.get("project_name", "") or "").strip()
            if name:
                return f"Build project {name}"
            return "Build a runnable project"
        if self.operation == "review":
            return f"Review project {self.project_dir}"
        if self.operation == "import":
            return f"Import and improve project {self.project_dir}"
        if self.operation == "resume":
            return f"Resume run in {self.project_dir}"
        return f"Execute {self.operation} run"

    def _plan_constraints(self) -> dict[str, Any]:
        return {
            "budget_limit_usd": float(getattr(self.config, "budget_limit_usd", 0.0) or 0.0),
            "max_agents": int(getattr(self.config, "max_agents", 1) or 1),
            "execution_backend": str(getattr(self.config, "execution_backend", "auto") or "auto"),
            "deterministic": bool(getattr(self.config, "deterministic", False)),
            "confirm_phases": list(getattr(self.config, "confirm_phases", []) or []),
            "ui_harness_enabled": bool(getattr(self.config, "ui_harness_enabled", False)),
            "design_context_refs": list(getattr(self.config, "design_context_refs", []) or []),
        }

    def _initialize_execution_plan(self) -> None:
        objective = self._initial_objective()
        summary = str(self.thread.metadata.get("summary", "") or "").strip()
        if not summary:
            summary = f"{self.operation} run under profile {self.profile.name}"
        self._plan = ExecutionPlanArtifact.create(
            run_id=self.thread.run_id,
            lineage_id=self.thread.lineage_id,
            parent_run_id=self.thread.parent_run_id,
            project_id=self.thread.project_id,
            thread_id=self.thread.thread_id,
            profile=self.profile,
            operation=self.operation,
            surface=self.surface,
            objective=objective,
            summary=summary,
            constraints=self._plan_constraints(),
            inputs={"project_dir": str(self.project_dir)},
            metadata=dict(self.thread.metadata),
            checkpoints=list(getattr(self.config, "confirm_phases", []) or []),
        )
        self._write_execution_plan()

    def _write_execution_plan(self) -> None:
        if self._plan is None:
            return
        write_execution_plan(self._plan_path, self._plan)
        self._sync_run_store(status=self._plan.status or "active")

    def update_execution_plan(
        self,
        *,
        objective: str | None = None,
        summary: str | None = None,
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
        current_phase: str | None = None,
    ) -> None:
        if self._plan is None:
            return
        self._plan.update(
            objective=objective,
            summary=summary,
            inputs=inputs,
            metadata=metadata,
            status=status,
            current_phase=current_phase,
        )
        self._write_execution_plan()
        if self.profile.name == "research":
            self._write_research_brief()

    def _write_research_brief(self) -> None:
        if self.profile.name != "research" or self._plan is None:
            return
        self._research_root.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Research Brief",
            "",
            f"- Run ID: {self.thread.run_id}",
            f"- Profile: {self.profile.name}",
            f"- Operation: {self.operation}",
            f"- Objective: {self._plan.objective}",
            f"- Summary: {self._plan.summary}",
            "",
            "## Phase Graph",
        ]
        for phase in self.profile.phase_graph.phase_ids():
            state = self._plan.phase_states.get(phase, "pending")
            lines.append(f"- {phase}: {state}")
        lines.extend(
            [
                "",
                "## Success Contract",
                f"- {self.profile.success_contract.description}",
                "",
                "## Artifact Contract",
            ]
        )
        for kind in self.profile.artifact_contract.required_kinds():
            lines.append(f"- {kind}")
        self._research_brief_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.register_artifact("brief", self._research_brief_path, required=True)

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _append_event(self, event: KernelEvent) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
        with self._lock:
            with self._events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        event = KernelEvent(
            seq=self._next_seq(),
            kind=str(kind),
            payload=dict(payload),
            run_id=self.thread.run_id,
            thread_id=self.thread.thread_id,
            profile=self.profile.name,
        )
        self._append_event(event)
        try:
            self.runtime.telemetry.record(f"kernel.{kind}", payload)
        except Exception:
            pass

    def _build_environment_lock(self) -> dict[str, Any]:
        backend = str(getattr(self.config, "execution_backend", "auto") or "auto")
        return {
            "schema_version": 1,
            "artifact_type": "environment_lock",
            "run_id": self.thread.run_id,
            "lineage_id": self.thread.lineage_id,
            "parent_run_id": self.thread.parent_run_id,
            "project_id": self.thread.project_id,
            "thread_id": self.thread.thread_id,
            "profile": self.profile.name,
            "surface": self.surface,
            "project_dir": str(self.project_dir),
            "workspace_dir": str(getattr(self.config, "workspace_dir", self.project_dir.parent)),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "runtime": {
                "backend": backend,
                "docker_enabled": bool(getattr(self.config, "docker_enabled", False)),
                "docker_required": bool(getattr(self.config, "docker_required", False)),
                "sandbox_image": str(getattr(self.config, "sandbox_image", "")),
                "docker_network_mode": str(getattr(self.config, "docker_network_mode", "")),
                "deterministic": bool(getattr(self.config, "deterministic", False)),
                "deterministic_seed": int(getattr(self.config, "deterministic_seed", 0) or 0),
                "subprocess_security_mode": str(
                    getattr(self.config, "subprocess_security_mode", "blacklist")
                ),
                "env_overrides": build_env_overrides(self.config, backend=backend),
            },
            "ui_harness": {
                "enabled": bool(getattr(self.config, "ui_harness_enabled", False)),
                "design_context_refs": list(getattr(self.config, "design_context_refs", []) or []),
            },
            "observability": {
                "trace_enabled": bool(getattr(self.config, "trace_enabled", False)),
                "trace_capture_llm_content": bool(
                    getattr(self.config, "trace_capture_llm_content", False)
                ),
                "trace_capture_command_output": bool(
                    getattr(self.config, "trace_capture_command_output", False)
                ),
                "trace_capture_fs_snapshots": bool(
                    getattr(self.config, "trace_capture_fs_snapshots", False)
                ),
                "artifacts_enabled": bool(getattr(self.config, "artifacts_enabled", True)),
            },
            "sys_path": list(sys.path[:20]),
            "created_at": time.time(),
        }

    def _write_manifest(self, *, status: str, extra: dict[str, Any] | None = None) -> None:
        manifest = {
            "schema_version": 1,
            "artifact_type": "kernel_manifest",
            "run_id": self.thread.run_id,
            "lineage_id": self.thread.lineage_id,
            "parent_run_id": self.thread.parent_run_id,
            "project_id": self.thread.project_id,
            "thread_id": self.thread.thread_id,
            "profile": self.profile.name,
            "operation": self.operation,
            "surface": self.surface,
            "status": status,
            "project_dir": str(self.project_dir),
            "phase_history": list(self._phase_history),
            "turn_count": len(self._turns),
            "artifacts": str(self._artifact_manifest_path),
            "events": str(self._events_path),
            "inbox": str(self._inbox_path),
            "contract_verdict": str(self._verdict_path) if self._contract_verdict is not None else "",
            "updated_at": time.time(),
            "metadata": dict(self.thread.metadata),
        }
        if extra:
            manifest.update(extra)
        write_kernel_json(self._manifest_path, manifest, artifact_type="kernel_manifest")
        self._sync_run_store(status=status)

    def _write_artifact_manifest(self) -> None:
        payload = {
            "schema_version": 1,
            "artifact_type": "artifact_manifest",
            "run_id": self.thread.run_id,
            "lineage_id": self.thread.lineage_id,
            "project_id": self.thread.project_id,
            "profile": self.profile.name,
            "artifacts": list(self._artifacts),
            "contract": self.profile.artifact_contract.to_dict(),
            "updated_at": time.time(),
        }
        write_kernel_json(self._artifact_manifest_path, payload, artifact_type="artifact_manifest")

    def _write_inbox(self) -> None:
        payload = {
            "schema_version": 1,
            "artifact_type": "kernel_inbox",
            "run_id": self.thread.run_id,
            "messages": list(self._inbox),
            "updated_at": time.time(),
        }
        write_kernel_json(self._inbox_path, payload, artifact_type="kernel_inbox")

    def heartbeat(self) -> bool:
        ok = self._workspace_lock.heartbeat(
            holder=self._holder,
            run_id=self.thread.run_id,
            ttl_seconds=self._lock_ttl_seconds,
        )
        if ok:
            self._record("workspace_lock_heartbeat", {"holder": self._holder})
        return ok

    def register_artifact(
        self,
        kind: str,
        path: str | Path,
        *,
        required: bool | None = None,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        target = Path(path) if not isinstance(path, Path) else path
        resolved = target.resolve() if target.is_absolute() else (self.project_dir / target).resolve()
        entry = {
            "kind": str(kind),
            "path": _safe_relpath(self.project_dir, resolved),
            "absolute_path": str(resolved),
            "exists": resolved.exists(),
            "required": bool(required) if required is not None else None,
            "description": description,
            "metadata": dict(metadata or {}),
            "recorded_at": time.time(),
        }
        self._artifacts = [
            item for item in self._artifacts
            if not (item.get("kind") == entry["kind"] and item.get("absolute_path") == entry["absolute_path"])
        ]
        self._artifacts.append(entry)
        self._write_artifact_manifest()
        self._run_store.record_artifact(run_id=self.thread.run_id, artifact=entry)
        self._record("artifact_registered", entry)
        self._sync_run_store(status="active")

    def write_checkpoint(
        self,
        *,
        state_marker: str,
        state_version: int,
        state: dict[str, Any],
    ) -> None:
        checkpoint = write_kernel_checkpoint(
            self._checkpoint_path,
            run_id=self.thread.run_id,
            lineage_id=self.thread.lineage_id,
            parent_run_id=self.thread.parent_run_id,
            project_id=self.thread.project_id,
            thread_id=self.thread.thread_id,
            profile=self.profile.name,
            operation=self.operation,
            state_marker=state_marker,
            state_version=state_version,
            state=state,
        )
        self.register_artifact(
            "kernel_checkpoint",
            self._checkpoint_path,
            required=False,
            metadata={
                "state_marker": checkpoint.state_marker,
                "state_version": checkpoint.state_version,
            },
        )
        self._record(
            "checkpoint_snapshot",
            {
                "state_marker": checkpoint.state_marker,
                "state_version": checkpoint.state_version,
                "path": str(self._checkpoint_path),
            },
        )

    def record_phase(
        self,
        phase: str,
        *,
        state: str,
        summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        canonical = self.profile.normalize_phase(phase)
        entry = {
            "phase": canonical,
            "state": str(state),
            "summary": summary,
            "metadata": dict(metadata or {}),
            "ts": time.time(),
        }
        self._phase_history.append(entry)
        if self._plan is not None:
            self._plan.mark_phase(canonical, str(state))
            if str(state) == "completed":
                for successor in self.profile.phase_graph.successors(canonical):
                    if self._plan.phase_states.get(successor) == "pending":
                        self._plan.phase_states[successor] = "ready"
            self._write_execution_plan()
            if self.profile.name == "research":
                self._write_research_brief()
        self._write_manifest(status="active")
        self._run_store.record_phase(
            run_id=self.thread.run_id,
            phase=canonical,
            state=str(state),
            summary=summary,
            metadata=dict(metadata or {}),
            ts=float(entry["ts"]),
        )
        self._record("phase", entry)

    def start_turn(
        self,
        *,
        kind: str,
        phase: str,
        input_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> KernelTurn:
        turn = KernelTurn(
            kind=kind,
            phase=self.profile.normalize_phase(phase),
            input_summary=input_summary,
            metadata=dict(metadata or {}),
        )
        self._turns[turn.turn_id] = turn
        self._record("turn_started", turn.to_dict())
        self._write_manifest(status="active")
        return turn

    def complete_turn(
        self,
        turn: KernelTurn,
        *,
        status: str = "completed",
        output_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        turn.complete(status=status, output_summary=output_summary, metadata=metadata)
        self._record("turn_completed", turn.to_dict())
        self._write_manifest(status="active")

    def emit_item(
        self,
        *,
        turn: KernelTurn,
        item_type: str,
        summary: str = "",
        payload: dict[str, Any] | None = None,
        status: str = "completed",
    ) -> KernelItem:
        item = KernelItem(
            turn_id=turn.turn_id,
            item_type=item_type,
            summary=summary,
            payload=dict(payload or {}),
            status=status,
            completed_at=time.time() if status != "started" else None,
        )
        self._record("item", item.to_dict())
        return item

    def record_checkpoint(
        self,
        *,
        phase: str,
        summary: str,
        proceed: bool | None,
        source: str = "checkpoint",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._record(
            "checkpoint",
            {
                "phase": self.profile.normalize_phase(phase),
                "summary": summary,
                "proceed": proceed,
                "source": source,
                "metadata": dict(metadata or {}),
            },
        )

    def record_inbox_message(
        self,
        *,
        source: str,
        kind: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "source": source,
            "kind": kind,
            "text": text,
            "metadata": dict(metadata or {}),
            "ts": time.time(),
        }
        self._inbox.append(entry)
        if len(self._inbox) > 200:
            self._inbox = self._inbox[-200:]
        self._write_inbox()
        self._record("inbox_message", entry)

    def close(self, *, status: str = "completed", metadata: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        extra = dict(metadata or {})
        declared_from_close = extra.get("declared_outcomes")
        if isinstance(declared_from_close, str):
            self.declare_outcomes(declared_from_close)
        elif isinstance(declared_from_close, (list, tuple, set)):
            self.declare_outcomes(*[str(item) for item in declared_from_close])
        try:
            self.heartbeat()
        except Exception:
            pass
        self._record(
            "thread_completed",
            {
                "status": status,
                "metadata": extra,
                "phase_count": len(self._phase_history),
                "turn_count": len(self._turns),
            },
        )
        self.update_execution_plan(
            status=status,
            metadata={"close_metadata": extra},
        )
        if self.profile.name == "research":
            self._write_research_metrics(status=status)
            self._write_research_evidence_pack()
        self._contract_verdict = evaluate_contract_verdict(
            profile=self.profile,
            run_id=self.thread.run_id,
            operation=self.operation,
            surface=self.surface,
            status=status,
            phase_history=list(self._phase_history),
            artifacts=list(self._artifacts),
            declared_outcomes=list(self._declared_outcomes),
            metadata={
                "lineage_id": self.thread.lineage_id,
                "parent_run_id": self.thread.parent_run_id,
                "project_id": self.thread.project_id,
                "close_metadata": extra,
            },
        )
        write_contract_verdict(self._verdict_path, self._contract_verdict)
        self.register_artifact(
            "contract_verdict",
            self._verdict_path,
            required=True,
            metadata={
                "contract_satisfied": self._contract_verdict.contract_satisfied,
                "success_outcome_satisfied": self._contract_verdict.success_outcome_satisfied,
            },
        )
        self._write_manifest(
            status=status,
            extra={
                "close_metadata": extra,
                "contract_verdict": str(self._verdict_path),
                "contract_satisfied": self._contract_verdict.contract_satisfied,
            },
        )
        self._sync_run_store(status=status, completed_at=time.time())
        self._write_artifact_manifest()
        self._write_inbox()
        self._workspace_lock.release(holder=self._holder, run_id=self.thread.run_id)
        self._run_store.close()

    def _write_research_metrics(self, *, status: str) -> None:
        if self.profile.name != "research":
            return
        self._research_root.mkdir(parents=True, exist_ok=True)
        metrics = {
            "schema_version": 1,
            "artifact_type": "research_metrics",
            "run_id": self.thread.run_id,
            "profile": self.profile.name,
            "status": status,
            "event_count": int(self._seq),
            "turn_count": len(self._turns),
            "phase_count": len(self._phase_history),
            "total_input_tokens": int(getattr(self.config, "total_input_tokens", 0) or 0),
            "total_output_tokens": int(getattr(self.config, "total_output_tokens", 0) or 0),
            "estimated_cost_usd": float(getattr(self.config, "estimated_cost_usd", 0.0) or 0.0),
            "updated_at": time.time(),
        }
        write_kernel_json(self._research_metrics_path, metrics, artifact_type="research_metrics")
        self.register_artifact("metrics", self._research_metrics_path, required=True)

    def _write_research_evidence_pack(self) -> None:
        if self.profile.name != "research":
            return
        evidence_dir = self._research_root / "evidence_pack" / self.thread.run_id
        pack = ensure_research_evidence_pack(self._run_dir, out_dir=evidence_dir)
        self.register_artifact(
            "evidence_pack",
            pack.output_dir,
            required=True,
            metadata={"manifest_path": str(pack.manifest_path), "copied_files": len(pack.copied_files)},
        )

    def _sync_run_store(self, *, status: str, completed_at: float | None = None) -> None:
        objective = self._plan.objective if self._plan is not None else self._initial_objective()
        summary = self._plan.summary if self._plan is not None else ""
        current_phase = self._plan.current_phase if self._plan is not None else ""
        metadata = dict(self.thread.metadata)
        if self._declared_outcomes:
            metadata["declared_outcomes"] = list(self._declared_outcomes)
        self._run_store.sync_run(
            run_id=self.thread.run_id,
            lineage_id=self.thread.lineage_id,
            parent_run_id=self.thread.parent_run_id,
            project_id=self.thread.project_id,
            thread_id=self.thread.thread_id,
            profile=self.profile.name,
            operation=self.operation,
            surface=self.surface,
            status=str(status or "active"),
            project_dir=str(self.project_dir),
            current_phase=current_phase,
            objective=objective,
            summary=summary,
            metadata=metadata,
            verdict=self._contract_verdict.to_dict() if self._contract_verdict is not None else {},
            completed_at=completed_at,
        )
