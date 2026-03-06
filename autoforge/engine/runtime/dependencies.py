"""Dependency detection and resolution helpers.

This module is intentionally deterministic and non-LLM:
  - detect project package managers
  - generate install steps
  - generate test commands

The orchestrator can treat this as an explicit PREPARE_ENV phase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


ExecutionPlatform = Literal["posix", "windows"]


def normalize_execution_platform(raw: str | None) -> ExecutionPlatform:
    s = (raw or "").strip().lower()
    if s in {"windows", "win32", "win"} or s.startswith("win"):
        return "windows"
    return "posix"


def venv_dir() -> str:
    """Project-local venv dir used for deterministic, isolated installs."""
    return ".autoforge/venv"


def venv_python_relpath(platform: ExecutionPlatform) -> str:
    # Use forward slashes: they work for POSIX and are generally accepted on Windows.
    if platform == "windows":
        return f"{venv_dir()}/Scripts/python.exe"
    return f"{venv_dir()}/bin/python"


@dataclass(frozen=True)
class DependencyStep:
    command: str
    reason: str
    required: bool = True
    timeout_s: int = 900
    fallback_command: str | None = None
    fallback_reason: str | None = None
    fallback_timeout_s: int | None = None

    @staticmethod
    def _probe_for(command: str) -> str:
        cmd = (command or "").strip()
        if not cmd:
            return ""
        if cmd.startswith("python -m pip"):
            return "python -m pip --version"
        if cmd.startswith("python -m venv"):
            return "python --version"
        if cmd.startswith("python "):
            return "python --version"
        first = cmd.split(" ", 1)[0]
        return f"{first} --version"

    def probe_command(self) -> str:
        return self._probe_for(self.command)

    def fallback_probe_command(self) -> str:
        return self._probe_for(self.fallback_command or "")


def normalize_package_manager(raw: str) -> str | None:
    raw = (raw or "").strip().lower()
    if raw in {"npm", "pnpm", "yarn", "bun"}:
        return raw
    if raw.startswith(("npm@", "pnpm@", "yarn@", "bun@")):
        return raw.split("@", 1)[0]
    return None


def detect_package_manager(project_dir: Path) -> str:
    """Detect Node.js package manager for a project dir."""
    lock_to_pm = [
        ("pnpm-lock.yaml", "pnpm"),
        ("yarn.lock", "yarn"),
        ("bun.lockb", "bun"),
        ("package-lock.json", "npm"),
        ("npm-shrinkwrap.json", "npm"),
    ]
    for lock, pm in lock_to_pm:
        if (project_dir / lock).exists():
            return pm
    # Fallback: look in package.json for a packageManager field
    pkg = project_dir / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict):
                raw = data.get("packageManager", "")
                if isinstance(raw, str) and raw:
                    normalized = normalize_package_manager(raw)
                    if normalized:
                        return normalized
        except Exception:
            pass
    return "npm"


def select_node_test_script(scripts: dict[str, Any]) -> str | None:
    """Pick the most appropriate test script name from package.json scripts."""
    if not scripts:
        return None
    candidates = ("test", "test:ci", "test:unit", "test:all", "ci")
    for name in candidates:
        body = scripts.get(name)
        if isinstance(body, str) and body.strip():
            return name
    return None


def build_node_test_command(package_manager: str, script_name: str | None) -> str:
    pm = normalize_package_manager(package_manager) or "npm"
    if script_name:
        if pm == "npm" and script_name == "test":
            return "npm test -- --passWithNoTests"
        if pm == "pnpm" and script_name == "test":
            return "pnpm test -- --passWithNoTests"
        return f"{pm} run {script_name}"

    if pm == "npm":
        return "npm test -- --passWithNoTests"
    if pm == "pnpm":
        return "pnpm test -- --passWithNoTests"
    return f"{pm} test"


def makefile_has_test_target(makefile: Path) -> bool:
    try:
        content = makefile.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("test:"):
            return True
        # allow indentation
        if stripped.startswith("test") and stripped.replace("\t", " ").startswith("test"):
            if ":" in stripped and stripped.split(":", 1)[0].strip() == "test":
                return True
    return False


def detect_test_command(work_dir: Path, *, execution_platform: str | None = None) -> str | None:
    """Detect the best test command for a project directory."""
    platform = normalize_execution_platform(execution_platform)
    py = venv_python_relpath(platform)

    # Node
    pkg = work_dir / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            scripts = data.get("scripts", {})
            if isinstance(scripts, dict):
                pm = detect_package_manager(work_dir)
                script_name = select_node_test_script(scripts)
                return build_node_test_command(pm, script_name)
        # If package.json exists but scripts are missing, try default
        pm = detect_package_manager(work_dir)
        return build_node_test_command(pm, "test")

    # Python / pytest
    if (
        (work_dir / "pytest.ini").exists()
        or (work_dir / "setup.cfg").exists()
        or (work_dir / "tox.ini").exists()
    ):
        return f"{py} -m pytest --tb=short -q"

    pyproject = work_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text(encoding="utf-8", errors="replace")
            if "tool.pytest" in content or "pytest" in content:
                return f"{py} -m pytest --tb=short -q"
        except OSError:
            pass

    # Rust / cargo
    if (work_dir / "Cargo.toml").exists():
        return "cargo test"

    # Go
    if (work_dir / "go.mod").exists():
        return "go test ./..."

    # Dart / Flutter
    if (work_dir / "pubspec.yaml").exists():
        return "dart test"

    # Java
    if (work_dir / "pom.xml").exists():
        return "./mvnw test" if (work_dir / "mvnw").exists() else "mvn test"
    if (work_dir / "build.gradle").exists() or (work_dir / "build.gradle.kts").exists():
        return "./gradlew test" if (work_dir / "gradlew").exists() else "gradle test"

    # .NET
    if any(work_dir.glob("*.sln")) or any(work_dir.glob("*.csproj")):
        return "dotnet test"

    # Makefile
    makefile = work_dir / "Makefile"
    if makefile.exists() and makefile_has_test_target(makefile):
        return "make test"
    lowercase = work_dir / "makefile"
    if lowercase.exists() and makefile_has_test_target(lowercase):
        return "make test"

    return None


def dependency_steps(project_dir: Path, *, execution_platform: str | None = None) -> list[DependencyStep]:
    """Generate dependency installation steps for common ecosystems."""
    steps: list[DependencyStep] = []
    platform = normalize_execution_platform(execution_platform)
    venv_python = venv_python_relpath(platform)

    if (project_dir / "package.json").exists():
        pm = detect_package_manager(project_dir)
        npm_fallback = "npm install --no-audit --no-fund"
        if pm == "pnpm":
            if (project_dir / "pnpm-lock.yaml").exists():
                steps.append(DependencyStep(
                    "pnpm install --frozen-lockfile",
                    "pnpm dependencies (frozen)",
                    True,
                    fallback_command=npm_fallback,
                    fallback_reason="npm fallback (pnpm unavailable)",
                ))
            else:
                steps.append(DependencyStep(
                    "pnpm install",
                    "pnpm dependencies",
                    True,
                    fallback_command=npm_fallback,
                    fallback_reason="npm fallback (pnpm unavailable)",
                ))
        elif pm == "yarn":
            if (project_dir / "yarn.lock").exists():
                steps.append(DependencyStep(
                    "yarn install --frozen-lockfile",
                    "yarn dependencies (frozen)",
                    True,
                    fallback_command=npm_fallback,
                    fallback_reason="npm fallback (yarn unavailable)",
                ))
            else:
                steps.append(DependencyStep(
                    "yarn install",
                    "yarn dependencies",
                    True,
                    fallback_command=npm_fallback,
                    fallback_reason="npm fallback (yarn unavailable)",
                ))
        elif pm == "bun":
            steps.append(DependencyStep(
                "bun install",
                "bun dependencies",
                True,
                fallback_command=npm_fallback,
                fallback_reason="npm fallback (bun unavailable)",
            ))
        else:
            if (project_dir / "package-lock.json").exists():
                steps.append(DependencyStep("npm ci --no-audit --no-fund", "npm dependencies (ci)", True))
            else:
                steps.append(DependencyStep("npm install --no-audit --no-fund", "npm dependencies", True))

    pyproject = project_dir / "pyproject.toml"
    has_pyproject = pyproject.exists()
    has_python_setup = (project_dir / "setup.py").exists() or (project_dir / "setup.cfg").exists()

    requirements = project_dir / "requirements.txt"
    requirements_dev = project_dir / "requirements-dev.txt"

    pyproject_text = ""
    if has_pyproject:
        try:
            pyproject_text = pyproject.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pyproject_text = ""

    pyproject_looks_packaged = bool(
        pyproject_text
        and ("[project]" in pyproject_text or "tool.poetry" in pyproject_text or "[build-system]" in pyproject_text)
    )
    pyproject_mentions_pytest = bool(
        pyproject_text
        and ("tool.pytest" in pyproject_text or "pytest.ini_options" in pyproject_text)
    )
    has_pytest_config = (
        (project_dir / "pytest.ini").exists()
        or (project_dir / "tox.ini").exists()
    )

    needs_python_env = (
        requirements.exists()
        or requirements_dev.exists()
        or has_python_setup
        or pyproject_looks_packaged
        or pyproject_mentions_pytest
        or has_pytest_config
    )

    if needs_python_env:
        steps.append(DependencyStep(
            f"python -m venv {venv_dir()}",
            "python venv (project-local)",
            True,
            timeout_s=300,
        ))
        if requirements.exists():
            steps.append(DependencyStep(
                f"{venv_python} -m pip install -r {requirements.name}",
                "python requirements",
                True,
            ))
        if requirements_dev.exists():
            steps.append(DependencyStep(
                f"{venv_python} -m pip install -r {requirements_dev.name}",
                "python dev requirements",
                False,
            ))
        if has_python_setup or pyproject_looks_packaged:
            steps.append(DependencyStep(
                f"{venv_python} -m pip install -e .",
                "python package",
                True,
            ))

    # Dedup by command
    deduped: list[DependencyStep] = []
    seen: set[tuple[str, str]] = set()
    for s in steps:
        key = (s.command, s.fallback_command or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped
