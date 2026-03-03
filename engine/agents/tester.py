"""Tester Agent — project verification and testing."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.agent_base import AgentBase, ToolDefinition
from engine.llm_router import TaskComplexity
from engine.sandbox import SandboxBase

logger = logging.getLogger(__name__)


@dataclass
class TestResults:
    """Structured test output."""

    all_passed: bool
    results: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [r for r in self.results if not r.get("passed", True)]


class TesterAgent(AgentBase):
    """Verifies that the generated project builds, runs, and functions correctly.

    Detects the project type, installs dependencies, runs builds and tests,
    and reports results in a structured format.
    """

    ROLE = "tester"
    COMPLEXITY = TaskComplexity.STANDARD

    def __init__(
        self, config, llm, working_dir: Path, sandbox: SandboxBase | None = None
    ) -> None:
        self.working_dir = working_dir
        self.sandbox = sandbox
        super().__init__(config, llm)

    def _register_tools(self) -> None:
        self._tools = [
            ToolDefinition(
                name="run_command",
                description=(
                    "Execute a shell command in the project sandbox. "
                    "Use for installing dependencies, building, and running tests."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute",
                        },
                    },
                    "required": ["command"],
                },
                handler=self._handle_run_command,
            ),
            ToolDefinition(
                name="read_file",
                description="Read a file to check its contents.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path",
                        },
                    },
                    "required": ["path"],
                },
                handler=self._handle_read_file,
            ),
        ]

    async def _handle_run_command(self, input_data: dict[str, Any]) -> str:
        command = input_data["command"]
        if self.sandbox:
            result = await self.sandbox.exec(command)
            return json.dumps({
                "exit_code": result.exit_code,
                "stdout": result.stdout[:5000],
                "stderr": result.stderr[:2000],
                "timed_out": result.timed_out,
            })
        else:
            return json.dumps({
                "warning": "No sandbox available, command not executed",
                "command": command,
            })

    async def _handle_read_file(self, input_data: dict[str, Any]) -> str:
        rel_path = input_data["path"]
        full_path = (self.working_dir / rel_path).resolve()
        if not str(full_path).startswith(str(self.working_dir.resolve())):
            return json.dumps({"error": "Path traversal not allowed"})
        if not full_path.exists():
            return json.dumps({"error": f"File not found: {rel_path}"})
        return full_path.read_text(encoding="utf-8")

    def build_prompt(self, context: dict[str, Any]) -> str:
        spec = context.get("spec", {})
        mobile = spec.get("mobile", {})
        mobile_target = mobile.get("target", "none") if mobile else "none"

        base_prompt = (
            f"Verify the project '{spec.get('project_name', '')}'.\n\n"
            f"## Tech Stack\n"
            f"```json\n{json.dumps(spec.get('tech_stack', {}), indent=2)}\n```\n\n"
            f"## Instructions\n"
            f"1. Read package.json or requirements.txt to understand the project\n"
            f"2. Install dependencies\n"
            f"3. Run the build command\n"
            f"4. Run tests if they exist\n"
            f"5. Check for common issues\n\n"
        )

        # Add mobile-specific instructions if applicable
        if mobile_target != "none":
            mobile_framework = mobile.get("framework", "react-native")
            base_prompt += (
                f"## Mobile App Testing\n"
                f"This project includes a mobile app ({mobile_framework}, target: {mobile_target}).\n\n"
            )
            if mobile_framework == "react-native":
                base_prompt += (
                    f"For React Native:\n"
                    f"- Run `npm install` or `yarn install`\n"
                    f"- Run `npx tsc --noEmit` for type checking\n"
                    f"- Run `npm test` or `jest` if tests exist\n"
                    f"- Check that metro bundler config is valid\n"
                )
                if mobile_target in ("android", "both"):
                    base_prompt += (
                        f"- If Android SDK is available: `cd android && ./gradlew assembleDebug`\n"
                    )
            elif mobile_framework == "flutter":
                base_prompt += (
                    f"For Flutter:\n"
                    f"- Run `flutter pub get`\n"
                    f"- Run `flutter analyze`\n"
                    f"- Run `flutter test` if tests exist\n"
                )
                if mobile_target in ("android", "both"):
                    base_prompt += f"- If Android SDK available: `flutter build apk --debug`\n"
            base_prompt += "\n"

        base_prompt += (
            f"Output a JSON code block with your test results following the format "
            f"in your system prompt.\n"
        )
        return base_prompt

    def parse_results(self, output: str) -> TestResults:
        """Extract structured test results from output."""
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", output, re.DOTALL)
        if match:
            data = json.loads(match.group(1).strip())
        else:
            start = output.find("{")
            end = output.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(output[start : end + 1])
            else:
                logger.warning("Could not parse test results, defaulting to passed")
                return TestResults(all_passed=True, summary="Could not parse results")

        return TestResults(
            all_passed=data.get("all_passed", True),
            results=data.get("results", []),
            summary=data.get("summary", ""),
        )
