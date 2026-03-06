"""ForgeRuntime: wires kernel services for a single project run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autoforge.engine.runtime.artifacts import ArtifactStore, NullArtifactStore
from autoforge.engine.runtime.journal import RunJournal
from autoforge.engine.runtime.telemetry import TelemetrySink
from autoforge.engine.runtime.trace import TraceRecorder


@dataclass
class ForgeRuntime:
    project_dir: Path
    run_id: str
    journal: RunJournal
    artifacts: ArtifactStore | NullArtifactStore
    telemetry: TelemetrySink
    trace: TraceRecorder | None = None

    @classmethod
    def create(
        cls,
        project_dir: Path,
        run_id: str,
        *,
        artifacts_enabled: bool = True,
        trace_enabled: bool = False,
        trace_write_header: bool = True,
        trace_capture_llm_content: bool = False,
        trace_capture_command_output: bool = False,
        trace_capture_fs_snapshots: bool = False,
        trace_max_inline_chars: int = 20000,
        trace_redact_secrets: bool = True,
        trace_secrets: list[str] | None = None,
    ) -> "ForgeRuntime":
        journal = RunJournal(project_dir, run_id=run_id)
        artifacts = (
            ArtifactStore(project_dir / ".autoforge" / "artifacts" / run_id)
            if artifacts_enabled
            else NullArtifactStore()
        )
        telemetry: TelemetrySink = journal
        trace: TraceRecorder | None = None
        if trace_enabled:
            trace_path = project_dir / ".autoforge" / "traces" / f"{run_id}.jsonl"
            trace = TraceRecorder(
                run_id=run_id,
                trace_path=trace_path,
                delegate=journal,
                artifacts=artifacts,
                write_header=trace_write_header,
                capture_llm_content=trace_capture_llm_content,
                capture_command_output=trace_capture_command_output,
                capture_fs_snapshots=trace_capture_fs_snapshots,
                max_inline_chars=trace_max_inline_chars,
                redact_secrets=trace_redact_secrets,
                secrets=trace_secrets,
            )
            telemetry = trace
        return cls(
            project_dir=project_dir,
            run_id=run_id,
            journal=journal,
            artifacts=artifacts,
            telemetry=telemetry,
            trace=trace,
        )

    def close(self) -> None:
        if self.trace is not None:
            try:
                self.trace.close()
            except Exception:
                pass
        self.journal.close()
