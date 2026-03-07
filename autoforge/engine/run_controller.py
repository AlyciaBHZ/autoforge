from __future__ import annotations

from autoforge.engine.kernel import default_profile_for_command
from pathlib import Path
from typing import Any

from autoforge.engine.profile_runner import (
    DevelopmentProfileRunner,
    ResearchProfileRunner,
    ResumeProfileRunner,
    VerificationProfileRunner,
)


class RunController:
    """Dispatch top-level operations onto profile-specific runners."""

    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator

    @property
    def _config(self) -> Any:
        return getattr(self.orchestrator, "config", None)

    async def run_generate(self, requirement: str) -> Path:
        profile_name = default_profile_for_command(
            "generate",
            mode=str(getattr(self._config, "mode", "") or ""),
            explicit_profile=str(getattr(self._config, "profile", "") or ""),
        )
        if profile_name == "research":
            return await ResearchProfileRunner(self.orchestrator).run_generate(requirement)
        return await DevelopmentProfileRunner(self.orchestrator).run_generate(requirement)

    async def run_review(self, project_path: str) -> dict[str, Any]:
        return await VerificationProfileRunner(self.orchestrator).run_review(project_path)

    async def run_import(self, project_path: str, enhancement: str = "") -> Path:
        return await DevelopmentProfileRunner(self.orchestrator).run_import(project_path, enhancement)

    async def run_resume(self, workspace_path: Path | None = None) -> Path:
        return await ResumeProfileRunner(self.orchestrator).run_resume(workspace_path)
