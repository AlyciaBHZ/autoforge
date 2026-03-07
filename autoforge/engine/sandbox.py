"""Sandbox — safe command execution environment.

Supports two modes:
- SubprocessSandbox: Direct subprocess execution (default, no Docker required)
- DockerSandbox: Isolated Docker container execution (optional)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoforge.engine.config import ForgeConfig
from autoforge.engine.development_harness import (
    append_development_jsonl,
    resolve_development_harness_root,
    write_development_json,
)
from autoforge.engine.runtime.env import build_env_overrides, shell_export_block
from autoforge.engine.runtime.commands import run_args, run_shell
from autoforge.engine.runtime.telemetry import TelemetrySink

logger = logging.getLogger(__name__)


def _shell_quote(s: str) -> str:
    """Quote a string for shell use, cross-platform."""
    if sys.platform == "win32":
        # Use subprocess.list2cmdline for proper Windows escaping
        return subprocess.list2cmdline([s])
    return shlex.quote(s)


@dataclass
class SandboxResult:
    """Result of a command execution in the sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_seconds: float = 0.0
    used_fallback: bool = False


class SandboxBase(ABC):
    """Abstract sandbox for command execution."""

    telemetry: TelemetrySink | None = None
    _development_harness_root: Path | None = None
    _sandbox_backend: str = ""

    @property
    def execution_platform(self) -> str:
        """Execution platform inside this sandbox ("windows" | "posix")."""
        return "windows" if sys.platform == "win32" else "posix"

    def configure_execution_harness(
        self,
        *,
        root: Path,
        backend: str,
        config: ForgeConfig,
        working_dir: Path,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        self._development_harness_root = root
        self._sandbox_backend = str(backend or "")
        root.mkdir(parents=True, exist_ok=True)
        write_development_json(
            root / "execution_environment.json",
            {
                "run_id": str(getattr(config, "run_id", "") or ""),
                "lineage_id": str(getattr(config, "lineage_id", "") or ""),
                "project_id": str(getattr(config, "project_id", "") or ""),
                "backend": str(backend or ""),
                "working_dir": str(working_dir),
                "execution_platform": self.execution_platform,
                "env_keys": sorted(str(key) for key in (env_overrides or {}).keys()),
            },
            artifact_type="execution_environment",
        )
        write_development_json(
            root / "sandbox_policy.json",
            {
                "backend": str(backend or ""),
                "security_mode": str(getattr(self, "_security_mode", "") or ""),
                "docker_enabled": bool(getattr(config, "docker_enabled", False)),
                "execution_backend": str(getattr(config, "execution_backend", "") or ""),
            },
            artifact_type="sandbox_policy",
        )

    def _record_command_receipt(
        self,
        *,
        command: str = "",
        args: list[str] | None = None,
        capability: str | None = None,
        result: SandboxResult,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._development_harness_root is None:
            return
        payload = {
            "cwd": str(getattr(self, "working_dir", "")),
            "backend": str(self._sandbox_backend or ""),
            "execution_platform": self.execution_platform,
            "command": str(command or ""),
            "args": [str(item) for item in (args or [])],
            "capability": str(capability or ""),
            "exit_code": int(result.exit_code),
            "timed_out": bool(result.timed_out),
            "duration_seconds": float(result.duration_seconds),
            "used_fallback": bool(result.used_fallback),
            "stderr_preview": str(result.stderr or "")[:600],
            "stdout_preview": str(result.stdout or "")[:600],
        }
        if extra:
            payload.update(extra)
        append_development_jsonl(
            self._development_harness_root / "command_receipts.jsonl",
            payload,
            event_type="command_receipt",
        )

    @abstractmethod
    async def start(self) -> None:
        """Initialize the sandbox."""

    @abstractmethod
    async def exec(
        self,
        command: str,
        timeout: int = 120,
        *,
        capability: str | None = None,
    ) -> SandboxResult:
        """Execute a command in the sandbox."""

    async def exec_args(
        self,
        args: list[str],
        timeout: int = 120,
        *,
        capability: str | None = None,
    ) -> SandboxResult:
        """Execute an argument-vector command.

        Subclasses should override this for a true non-shell execution path.
        The base implementation quotes args and delegates to exec().
        """
        if not args:
            return SandboxResult(exit_code=-1, stdout="", stderr="No command provided")
        command = " ".join(_shell_quote(str(a)) for a in args)
        return await self.exec(command, timeout=timeout, capability=capability)

    @abstractmethod
    async def stop(self) -> None:
        """Clean up the sandbox."""

    async def __aenter__(self) -> SandboxBase:
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()


class SubprocessSandbox(SandboxBase):
    """Sandbox that runs commands as local subprocesses.

    Used when Docker is not available. Commands run in a specified
    working directory with timeout enforcement.
    """

    # Dangerous command patterns that must never be executed.
    BLOCKED_PATTERNS: list[str] = [
        r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+(-[a-zA-Z]*\s+)*|(-[a-zA-Z]*\s+)*-[a-zA-Z]*f[a-zA-Z]*\s+)/",  # rm with -f and /
        r"\brm\s+-rf\s+/",              # rm -rf /
        r"\brm\s+-fr\s+/",              # rm -fr /
        r"\bmkfs\b",                     # mkfs (format disk)
        r"\bdd\s+if=",                   # dd if= (raw disk write)
        r"\bchmod\s+777\s+/",           # chmod 777 / (recursive perm open)
        r"\bcurl\b.*\|.*\bsh\b",        # curl | sh (remote code exec)
        r"\bcurl\b.*\|.*\bbash\b",      # curl | bash
        r"\bwget\b.*\|.*\bsh\b",        # wget | sh
        r"\bwget\b.*\|.*\bbash\b",      # wget | bash
        r">\s*/dev/sd[a-z]",            # > /dev/sda (overwrite disk)
        r":\(\)\s*\{\s*:\|\s*:&\s*\}\s*;", # fork bomb
        r"\bshutdown\b",                # shutdown
        r"\breboot\b",                  # reboot
        r"\binit\s+0\b",               # init 0
        r"\bhalt\b",                    # halt
        r"\bsystemctl\s+(start|stop|restart|disable|enable)\b",
        r"\biptables\b",               # firewall manipulation
        r">\s*/etc/passwd",             # overwrite passwd
        r">\s*/etc/shadow",             # overwrite shadow
        r"\bnc\s+-[elp]",              # netcat listener/exec
        # Additional patterns to harden against bypass
        r"\bfind\b.*-delete\s+/",       # find / -delete
        r"\bfind\b.*-exec\s+rm\b",     # find -exec rm
        r"\bshutil\.rmtree\s*\(\s*['\"]\/", # Python shutil.rmtree('/')
        r"\bos\.remove\s*\(\s*['\"]\/",     # Python os.remove('/')
        r"\bos\.system\s*\(",           # Python os.system()
        r"\b__import__\s*\(",           # Python __import__()
        # Supply-chain hardening for dynamic dependency installation
        r"\bpip\s+install\b[^\n]*(--extra-index-url|--index-url|git\+|https?://|file://)",
        r"\bnpm\s+install\b[^\n]*(https?://|git\+|github:|file:)",
        r"\byarn\s+add\b[^\n]*(https?://|git\+|github:|file:)",
        r"\bpnpm\s+add\b[^\n]*(https?://|git\+|github:|file:)",
    ]

    DEFAULT_ALLOWLIST: frozenset[str] = frozenset({
        # Python
        "python", "python3", "pip", "pip3", "pytest",
        # Node
        "node", "npm", "npx", "pnpm", "yarn", "bun",
        # Common build tools
        "go", "cargo", "rustc", "dotnet", "java", "javac", "mvn", "gradle", "make",
        # TypeScript toolchain
        "tsc",
        # Lean tooling (formal verification)
        "lean", "lake", "elan",
    })

    DEFAULT_ALLOWLIST_BY_CAPABILITY: dict[str, frozenset[str]] = {
        # The default is intentionally conservative and executable-name based.
        # Users can extend per-capability via config/env:
        #   FORGE_SUBPROCESS_ALLOWLIST_MAP='{"deps":["uv"],"lint":["ruff"]}'
        "general": DEFAULT_ALLOWLIST,
        "deps": frozenset({
            "python", "python3", "pip", "pip3",
            "node", "npm", "npx", "pnpm", "yarn", "bun",
            "go", "cargo", "rustc", "dotnet", "java", "javac", "mvn", "gradle", "make",
            "tsc",
            # optional Python packaging tools commonly seen in repos
            "uv", "poetry", "pipenv",
        }),
        "test": frozenset({
            "python", "python3", "pytest",
            "node", "npm", "npx", "pnpm", "yarn", "bun",
            "go", "cargo", "rustc", "dotnet", "java", "javac", "mvn", "gradle", "make",
            "tsc",
        }),
        "typecheck": frozenset({
            "python", "python3",
            "node", "npm", "npx", "pnpm", "yarn", "bun",
            "go", "cargo", "rustc", "dotnet", "java", "javac", "mvn", "gradle", "make",
            "tsc",
            # common typecheck tools
            "mypy", "pyright", "ruff",
        }),
        "lint": frozenset({
            "python", "python3",
            "node", "npm", "npx", "pnpm", "yarn", "bun",
            "go", "cargo", "rustc", "dotnet", "java", "javac", "mvn", "gradle", "make",
            "tsc",
            # common linters
            "ruff", "flake8", "pylint", "mypy", "pyright", "eslint", "stylelint",
        }),
        "format": frozenset({
            "python", "python3",
            "node", "npm", "npx", "pnpm", "yarn", "bun",
            "go", "cargo", "rustc", "dotnet", "java", "javac", "mvn", "gradle", "make",
            "tsc",
            # common formatters
            "ruff", "black", "isort", "prettier",
        }),
        "prove": frozenset({
            "lean", "lake", "elan",
        }),
    }

    def __init__(
        self,
        working_dir: Path,
        env: dict[str, str] | None = None,
        *,
        config: ForgeConfig | None = None,
    ) -> None:
        self.working_dir = working_dir
        self.env = env
        self.telemetry = None
        mode = "blacklist"
        extra_allow: list[str] = []
        allow_map_raw: Any = {}
        if config is not None:
            mode = str(getattr(config, "subprocess_security_mode", "blacklist") or "blacklist").strip().lower()
            raw = getattr(config, "subprocess_allowlist", [])
            if isinstance(raw, list):
                extra_allow = [str(s) for s in raw]
            elif isinstance(raw, str):
                extra_allow = [s.strip() for s in raw.split(",") if s.strip()]
            allow_map_raw = getattr(config, "subprocess_allowlist_by_capability", {})
        if mode not in ("blacklist", "allowlist", "disabled"):
            mode = "blacklist"
        self._security_mode = mode

        allow_global = {
            self._normalize_executable_name(s)
            for s in extra_allow
            if self._normalize_executable_name(s)
        }
        allow_map: dict[str, list[str]] = {}
        if isinstance(allow_map_raw, dict):
            for k, v in allow_map_raw.items():
                cap = str(k).strip().lower()
                if not cap:
                    continue
                if isinstance(v, list):
                    items = [str(s) for s in v]
                elif isinstance(v, str):
                    items = [s.strip() for s in v.split(",") if s.strip()]
                else:
                    continue
                allow_map[cap] = items

        allow_by_cap: dict[str, set[str]] = {}
        for cap, base in SubprocessSandbox.DEFAULT_ALLOWLIST_BY_CAPABILITY.items():
            allow_by_cap[cap] = set(base) | set(allow_global)
        for cap, items in allow_map.items():
            base = set(SubprocessSandbox.DEFAULT_ALLOWLIST)
            allow_by_cap.setdefault(cap, base | set(allow_global))
            allow_by_cap[cap].update(
                self._normalize_executable_name(s)
                for s in items
                if self._normalize_executable_name(s)
            )
        self._allowlist_by_capability = allow_by_cap
        self._default_allowlist = allow_by_cap.get("general", set(SubprocessSandbox.DEFAULT_ALLOWLIST) | set(allow_global))

    @staticmethod
    def _normalize_capability(raw: str | None) -> str:
        cap = (raw or "").strip().lower()
        if not cap:
            return "general"
        # Allow hierarchical names, but treat the prefix as the policy key.
        for sep in (":", ".", "/", "\\"):
            if sep in cap:
                cap = cap.split(sep, 1)[0].strip()
                break
        return cap or "general"

    def _effective_allowlist(self, capability: str | None) -> set[str]:
        cap = self._normalize_capability(capability)
        return self._allowlist_by_capability.get(cap) or self._default_allowlist

    @classmethod
    def _sanitize_command(cls, command: str) -> None:
        """Check command against blocked patterns and raise ValueError if matched."""
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                raise ValueError(
                    f"Blocked dangerous command pattern: {pattern!r}"
                )

    @staticmethod
    def _normalize_executable_name(raw: str) -> str:
        s = (raw or "").strip().strip("\"' ")
        if not s:
            return ""
        name = Path(s).name
        lower = name.lower()
        if lower.endswith(".exe"):
            lower = lower[:-4]
        return lower

    @classmethod
    def _extract_executable_from_command(cls, command: str) -> str:
        s = (command or "").strip()
        if not s:
            return ""
        try:
            parts = shlex.split(s, posix=(sys.platform != "win32"))
        except Exception:
            parts = s.split()
        if not parts:
            return ""
        return cls._normalize_executable_name(parts[0])

    def _allowlist_reject_reason(
        self,
        *,
        command: str = "",
        args: list[str] | None = None,
        capability: str | None = None,
    ) -> str:
        if self._security_mode != "allowlist":
            return ""
        joined = command or (" ".join(str(a) for a in (args or [])))
        # Prevent simple command chaining in strict allowlist mode. Prefer exec_args.
        if "|" in joined or "&&" in joined or "||" in joined or ";" in joined or "\n" in joined or "\r" in joined:
            return "shell metacharacters are not allowed in allowlist mode; use exec_args"
        exe = (
            self._normalize_executable_name(args[0]) if args else self._extract_executable_from_command(joined)
        )
        if not exe:
            return "could not determine executable for allowlist check"
        allow = self._effective_allowlist(capability)
        if exe not in allow:
            return f"executable {exe!r} not in allowlist"
        return ""

    async def start(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SubprocessSandbox ready at {self.working_dir}")

    async def exec(
        self,
        command: str,
        timeout: int = 120,
        *,
        capability: str | None = None,
    ) -> SandboxResult:
        """Execute a command as a subprocess."""
        logger.debug("[sandbox] exec: %s", command[:100])

        if self._security_mode == "allowlist":
            reason = self._allowlist_reject_reason(command=command, args=None, capability=capability)
            if reason:
                logger.warning("[sandbox] Command blocked (allowlist): %s", reason)
                blocked = SandboxResult(exit_code=-1, stdout="", stderr=f"Command blocked by allowlist policy: {reason}")
                self._record_command_receipt(command=command, capability=capability, result=blocked, extra={"blocked_reason": reason})
                return blocked

        # Validate command against blocked patterns
        if self._security_mode != "disabled":
            try:
                self._sanitize_command(command)
            except ValueError as e:
                logger.warning("[sandbox] Command blocked: %s", e)
                blocked = SandboxResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command blocked by security filter: {e}",
                )
                self._record_command_receipt(command=command, capability=capability, result=blocked, extra={"blocked_reason": str(e)})
                return blocked

        # On Windows, wrap Unix-style commands to run via bash/sh if available
        shell_command = command
        if sys.platform == "win32" and not command.startswith("cmd") and self._security_mode != "allowlist":
            sh = shutil.which("bash") or shutil.which("sh")
            if sh:
                sh_norm = sh.replace("/", "\\").lower()
                # Avoid WSL's bash.exe shim (often present on PATH but not usable in restricted environments).
                if "\\windowsapps\\bash.exe" in sh_norm or sh_norm.endswith("\\system32\\bash.exe"):
                    sh = None

                    # Prefer a non-WSL bash/sh if one exists on PATH (for example Git Bash).
                    for entry in os.environ.get("PATH", "").split(os.pathsep):
                        entry = entry.strip().strip('"')
                        if not entry:
                            continue
                        low = entry.replace("/", "\\").lower().rstrip("\\")
                        if "windowsapps" in low or low.endswith("\\system32"):
                            continue
                        for name in ("bash.exe", "sh.exe"):
                            candidate = Path(entry) / name
                            if candidate.is_file():
                                sh = str(candidate)
                                break
                        if sh:
                            break

            if sh:
                shell_command = f'{_shell_quote(sh)} -c {_shell_quote(command)}'

        res = await run_shell(
            shell_command,
            cwd=self.working_dir,
            env=self.env,
            timeout_s=float(timeout),
            label="sandbox.exec",
        )
        result = SandboxResult(
            exit_code=res.exit_code,
            stdout=res.stdout,
            stderr=res.stderr,
            timed_out=res.timed_out,
            duration_seconds=res.duration_s,
            used_fallback=res.used_fallback,
        )
        if self.telemetry is not None:
            try:
                payload: dict[str, Any] = {
                    "cwd": str(self.working_dir),
                    "command": str(command),
                    "capability": str(capability or ""),
                    "security_mode": str(self._security_mode),
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "duration_seconds": result.duration_seconds,
                    "used_fallback": result.used_fallback,
                }
                if bool(getattr(self.telemetry, "capture_command_output", False)):
                    payload["stdout"] = result.stdout
                    payload["stderr"] = result.stderr
                self.telemetry.record_command(payload)
            except Exception:
                pass
        self._record_command_receipt(command=command, capability=capability, result=result)
        return result

    async def exec_args(
        self,
        args: list[str],
        timeout: int = 120,
        *,
        capability: str | None = None,
    ) -> SandboxResult:
        """Execute an argument-vector command without invoking a shell."""
        if not args:
            return SandboxResult(exit_code=-1, stdout="", stderr="No command provided")

        joined = " ".join(str(a) for a in args)
        logger.debug("[sandbox] exec_args: %s", joined[:100])

        if self._security_mode == "allowlist":
            reason = self._allowlist_reject_reason(command="", args=args, capability=capability)
            if reason:
                logger.warning("[sandbox] Command blocked (allowlist): %s", reason)
                blocked = SandboxResult(exit_code=-1, stdout="", stderr=f"Command blocked by allowlist policy: {reason}")
                self._record_command_receipt(args=args, capability=capability, result=blocked, extra={"blocked_reason": reason})
                return blocked

        # Sanitize the joined command too; patterns like "pip install" span args.
        if self._security_mode != "disabled":
            try:
                self._sanitize_command(joined)
            except ValueError as e:
                logger.warning("[sandbox] Command blocked: %s", e)
                blocked = SandboxResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command blocked by security filter: {e}",
                )
                self._record_command_receipt(args=args, capability=capability, result=blocked, extra={"blocked_reason": str(e)})
                return blocked

        # Sanitize args individually — don't join into a shell string
        if self._security_mode != "disabled":
            for arg in args:
                try:
                    self._sanitize_command(arg)
                except ValueError as e:
                    logger.warning("[sandbox] Argument blocked: %s", e)
                    blocked = SandboxResult(
                        exit_code=-1,
                        stdout="",
                        stderr=f"Command blocked by security filter: {e}",
                    )
                    self._record_command_receipt(args=args, capability=capability, result=blocked, extra={"blocked_reason": str(e)})
                    return blocked

        res = await run_args(
            args,
            cwd=self.working_dir,
            env=self.env,
            timeout_s=float(timeout),
            label="sandbox.exec_args",
        )
        result = SandboxResult(
            exit_code=res.exit_code,
            stdout=res.stdout,
            stderr=res.stderr,
            timed_out=res.timed_out,
            duration_seconds=res.duration_s,
            used_fallback=res.used_fallback,
        )
        if self.telemetry is not None:
            try:
                payload: dict[str, Any] = {
                    "cwd": str(self.working_dir),
                    "args": [str(a) for a in args],
                    "capability": str(capability or ""),
                    "security_mode": str(self._security_mode),
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "duration_seconds": result.duration_seconds,
                    "used_fallback": result.used_fallback,
                }
                if bool(getattr(self.telemetry, "capture_command_output", False)):
                    payload["stdout"] = result.stdout
                    payload["stderr"] = result.stderr
                self.telemetry.record_command(payload)
            except Exception:
                pass
        self._record_command_receipt(args=args, capability=capability, result=result)
        return result

    async def stop(self) -> None:
        logger.debug("SubprocessSandbox stopped")


class DockerSandbox(SandboxBase):
    """Sandbox using a Docker container for isolation.

    Features:
    - No network access (--network none)
    - Memory and CPU limits
    - Workspace bind-mounted
    """

    def __init__(
        self,
        image: str,
        working_dir: Path,
        memory_limit: str = "2g",
        cpu_limit: str = "2",
        network_mode: str = "none",
        pids_limit: int = 512,
        env: dict[str, str] | None = None,
    ) -> None:
        self.image = image
        self.working_dir = working_dir
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_mode = (network_mode or "none").strip() or "none"
        self.pids_limit = max(0, int(pids_limit or 0))
        self.env = env
        self._container_id: str | None = None
        self._container_name: str = f"autoforge-sandbox-{uuid.uuid4().hex[:8]}"
        self.telemetry = None

    @property
    def execution_platform(self) -> str:
        # The container is always POSIX, even if the host is Windows.
        return "posix"

    async def _restart_container_best_effort(self) -> None:
        """Recover after a timed-out exec by recreating the container.

        The workspace is bind-mounted, so state under /workspace persists.
        """
        target = self._container_id or self._container_name
        try:
            await run_args(
                ["docker", "rm", "-f", str(target)],
                cwd=self.working_dir,
                timeout_s=30.0,
                label="docker.restart",
            )
        except Exception:
            pass
        self._container_id = None
        try:
            await self.start()
        except Exception:
            pass

    async def start(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        container_name = self._container_name

        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-v", f"{self.working_dir.resolve()}:/workspace",
            "-w", "/workspace",
            "--network", self.network_mode,
            "--memory", self.memory_limit,
            "--cpus", self.cpu_limit,
        ]
        if self.env:
            for k, v in self.env.items():
                key = str(k).strip()
                if not key or any(ch.isspace() for ch in key) or "=" in key:
                    continue
                cmd.extend(["-e", f"{key}={v}"])
        if self.pids_limit > 0:
            cmd.extend(["--pids-limit", str(self.pids_limit)])
        cmd.extend([self.image, "tail", "-f", "/dev/null"])

        res = await run_args(cmd, cwd=self.working_dir, timeout_s=60.0, label="docker.start")

        if res.exit_code == 0:
            raw_id = (res.stdout or "").strip()
            # Validate container ID format (hex string, typically 64 chars)
            if re.fullmatch(r"[0-9a-f]+", raw_id) and len(raw_id) >= 12:
                self._container_id = raw_id
            else:
                raise RuntimeError(
                    f"Docker returned unexpected container ID: {raw_id[:64]!r}"
                )
            logger.info(f"DockerSandbox started: {self._container_id[:12]}")
        else:
            raise RuntimeError(f"Failed to start Docker sandbox: {res.stderr}")

    async def exec(
        self,
        command: str,
        timeout: int = 120,
        *,
        capability: str | None = None,
    ) -> SandboxResult:
        if not self._container_id:
            raise RuntimeError("Docker sandbox not started")

        # Apply command sanitization even inside Docker for defense-in-depth
        try:
            SubprocessSandbox._sanitize_command(command)
        except ValueError as e:
            logger.warning("[docker-sandbox] Command blocked: %s", e)
            blocked = SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command blocked by security filter: {e}",
            )
            self._record_command_receipt(command=command, capability=capability, result=blocked, extra={"blocked_reason": str(e), "docker": True})
            return blocked

        cmd = ["docker", "exec", "-w", "/workspace", self._container_id, "bash", "-c", command]
        res = await run_args(cmd, cwd=self.working_dir, timeout_s=float(timeout), label="docker.exec")
        if res.timed_out:
            await self._restart_container_best_effort()
        result = SandboxResult(
            exit_code=res.exit_code,
            stdout=res.stdout,
            stderr=res.stderr,
            timed_out=res.timed_out,
            duration_seconds=res.duration_s,
            used_fallback=res.used_fallback,
        )
        if self.telemetry is not None:
            try:
                payload: dict[str, Any] = {
                    "cwd": str(self.working_dir),
                    "command": str(command),
                    "capability": str(capability or ""),
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "duration_seconds": result.duration_seconds,
                    "used_fallback": result.used_fallback,
                    "docker": True,
                }
                if bool(getattr(self.telemetry, "capture_command_output", False)):
                    payload["stdout"] = result.stdout
                    payload["stderr"] = result.stderr
                self.telemetry.record_command(payload)
            except Exception:
                pass
        self._record_command_receipt(command=command, capability=capability, result=result, extra={"docker": True})
        return result

    async def exec_args(
        self,
        args: list[str],
        timeout: int = 120,
        *,
        capability: str | None = None,
    ) -> SandboxResult:
        """Execute an argument-vector command inside the container."""
        if not self._container_id:
            raise RuntimeError("Docker sandbox not started")
        if not args:
            return SandboxResult(exit_code=-1, stdout="", stderr="No command provided")

        joined = " ".join(str(a) for a in args)
        logger.debug("[docker-sandbox] exec_args: %s", joined[:100])

        try:
            SubprocessSandbox._sanitize_command(joined)
        except ValueError as e:
            logger.warning("[docker-sandbox] Command blocked: %s", e)
            blocked = SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command blocked by security filter: {e}",
            )
            self._record_command_receipt(args=args, capability=capability, result=blocked, extra={"blocked_reason": str(e), "docker": True})
            return blocked

        cmd = ["docker", "exec", "-w", "/workspace", self._container_id, *[str(a) for a in args]]
        res = await run_args(cmd, cwd=self.working_dir, timeout_s=float(timeout), label="docker.exec_args")
        if res.timed_out:
            await self._restart_container_best_effort()

        result = SandboxResult(
            exit_code=res.exit_code,
            stdout=res.stdout,
            stderr=res.stderr,
            timed_out=res.timed_out,
            duration_seconds=res.duration_s,
            used_fallback=res.used_fallback,
        )
        if self.telemetry is not None:
            try:
                payload: dict[str, Any] = {
                    "cwd": str(self.working_dir),
                    "args": [str(a) for a in args],
                    "capability": str(capability or ""),
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "duration_seconds": result.duration_seconds,
                    "used_fallback": result.used_fallback,
                    "docker": True,
                }
                if bool(getattr(self.telemetry, "capture_command_output", False)):
                    payload["stdout"] = result.stdout
                    payload["stderr"] = result.stderr
                self.telemetry.record_command(payload)
            except Exception:
                pass
        self._record_command_receipt(args=args, capability=capability, result=result, extra={"docker": True})
        return result

    async def stop(self) -> None:
        target = self._container_id or self._container_name
        if target:
            await run_args(
                ["docker", "rm", "-f", str(target)],
                cwd=self.working_dir,
                timeout_s=30.0,
                label="docker.stop",
            )
        if self._container_id:
            logger.info(f"DockerSandbox stopped: {self._container_id[:12]}")
        self._container_id = None


def _slurm_available() -> bool:
    """Check if Slurm CLI tooling is available on this host."""
    return shutil.which("sbatch") is not None


def _format_slurm_time(seconds: int) -> str:
    """Format seconds as a Slurm --time string (D-HH:MM:SS or HH:MM:SS)."""
    sec = max(1, int(seconds or 1))
    minutes, sec = divmod(sec, 60)
    hours, minutes = divmod(minutes, 60)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days}-{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _read_file_tail(path: Path, *, max_bytes: int = 200_000) -> str:
    """Read the tail of a file to avoid unbounded memory usage."""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - int(max_bytes))
            f.seek(start)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    except Exception as e:
        return f"[autoforge] failed to read {path.name}: {e}"


class SlurmSandbox(SandboxBase):
    """Sandbox that executes commands by submitting Slurm batch jobs.

    Intended for HPC clusters where Docker may be unavailable and running
    subprocesses on a login node is undesirable.
    """

    def __init__(self, *, working_dir: Path, config: ForgeConfig) -> None:
        self.working_dir = working_dir
        self.config = config
        self._jobs_dir = self.working_dir / ".autoforge" / "slurm" / "jobs"
        self._logs_dir = self.working_dir / ".autoforge" / "slurm" / "logs"
        self._active_job_ids: set[str] = set()

    @property
    def execution_platform(self) -> str:
        # Slurm jobs run on Linux compute nodes.
        return "posix"

    async def start(self) -> None:
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)

    def _build_sbatch_args(
        self,
        *,
        script_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        timeout_s: int,
        job_name: str,
    ) -> list[str]:
        args: list[str] = [
            "sbatch",
            "--parsable",
            "--chdir",
            str(self.working_dir),
            "--output",
            str(stdout_path),
            "--error",
            str(stderr_path),
            "--job-name",
            str(job_name)[:128],
            "--time",
            _format_slurm_time(timeout_s),
        ]

        partition = str(getattr(self.config, "slurm_partition", "") or "").strip()
        if partition:
            args += ["--partition", partition]
        account = str(getattr(self.config, "slurm_account", "") or "").strip()
        if account:
            args += ["--account", account]
        qos = str(getattr(self.config, "slurm_qos", "") or "").strip()
        if qos:
            args += ["--qos", qos]

        cpus = int(getattr(self.config, "slurm_cpus_per_task", 0) or 0)
        if cpus > 0:
            args += ["--cpus-per-task", str(cpus)]
        mem = str(getattr(self.config, "slurm_mem", "") or "").strip()
        if mem:
            args += ["--mem", mem]
        gres = str(getattr(self.config, "slurm_gres", "") or "").strip()
        if gres:
            args += ["--gres", gres]

        args.append(str(script_path))
        return args

    def _write_job_script(self, *, command: str, exit_path: Path, script_path: Path) -> None:
        # Run command inside a login-style shell for consistency with typical HPC env modules.
        cmd_quoted = shlex.quote(str(command))
        exit_quoted = shlex.quote(str(exit_path))
        env = build_env_overrides(self.config, backend="slurm")
        export_block = shell_export_block(env)
        script = (
            "#!/usr/bin/env bash\n"
            "set -uo pipefail\n"
            + export_block
            + f"bash -lc {cmd_quoted}\n"
            + "exit_code=$?\n"
            + f'printf "%s" "$exit_code" > {exit_quoted}\n'
            + 'exit "$exit_code"\n'
        )
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with script_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(script)

    async def _try_cancel(self, job_id: str) -> None:
        try:
            await run_args(
                ["scancel", str(job_id)],
                cwd=self.working_dir,
                timeout_s=30.0,
                label="slurm.scancel",
            )
        except Exception:
            pass

    async def _job_in_queue(self, job_id: str) -> bool:
        res = await run_args(
            ["squeue", "-h", "-j", str(job_id)],
            cwd=self.working_dir,
            timeout_s=20.0,
            label="slurm.squeue",
        )
        # If squeue itself fails, keep polling via the file-based exit code path.
        if res.exit_code != 0:
            return True
        return bool(res.stdout.strip())

    async def _try_sacct_exit_code(self, job_id: str) -> int | None:
        if shutil.which("sacct") is None:
            return None
        res = await run_args(
            ["sacct", "-n", "-P", "-j", str(job_id), "--format=ExitCode"],
            cwd=self.working_dir,
            timeout_s=20.0,
            label="slurm.sacct",
        )
        if res.exit_code != 0:
            return None
        for line in (res.stdout or "").splitlines():
            s = line.strip()
            if not s:
                continue
            # Common format: "0:0"
            token = s.split("|", 1)[0].strip()
            if not token:
                continue
            left = token.split(":", 1)[0].strip()
            if not left:
                continue
            try:
                return int(left)
            except ValueError:
                continue
        return None

    async def exec(
        self,
        command: str,
        timeout: int = 120,
        *,
        capability: str | None = None,
    ) -> SandboxResult:
        logger.debug("[slurm-sandbox] exec: %s", str(command)[:100])

        try:
            SubprocessSandbox._sanitize_command(command)
        except ValueError as e:
            logger.warning("[slurm-sandbox] Command blocked: %s", e)
            blocked = SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command blocked by security filter: {e}",
            )
            self._record_command_receipt(command=command, capability=capability, result=blocked, extra={"blocked_reason": str(e), "slurm": True})
            return blocked

        use_local_in_alloc = bool(getattr(self.config, "slurm_use_local_in_allocation", True))
        if use_local_in_alloc and os.getenv("SLURM_JOB_ID"):
            res = await run_shell(
                command,
                cwd=self.working_dir,
                timeout_s=float(timeout),
                label="slurm.in_alloc",
            )
            result = SandboxResult(
                exit_code=res.exit_code,
                stdout=res.stdout,
                stderr=res.stderr,
                timed_out=res.timed_out,
                duration_seconds=res.duration_s,
                used_fallback=res.used_fallback,
            )
            if self.telemetry is not None:
                try:
                    payload: dict[str, Any] = {
                        "cwd": str(self.working_dir),
                        "command": str(command),
                        "capability": str(capability or ""),
                        "exit_code": result.exit_code,
                        "timed_out": result.timed_out,
                        "duration_seconds": result.duration_seconds,
                        "used_fallback": result.used_fallback,
                        "slurm": True,
                        "slurm_local_in_allocation": True,
                    }
                    if bool(getattr(self.telemetry, "capture_command_output", False)):
                        payload["stdout"] = result.stdout
                        payload["stderr"] = result.stderr
                    self.telemetry.record_command(payload)
                except Exception:
                    pass
            self._record_command_receipt(command=command, capability=capability, result=result, extra={"slurm": True, "slurm_local_in_allocation": True})
            return result

        if not _slurm_available():
            blocked = SandboxResult(
                exit_code=-1,
                stdout="",
                stderr="Slurm requested but sbatch was not found on PATH",
            )
            self._record_command_receipt(command=command, capability=capability, result=blocked, extra={"slurm": True, "blocked_reason": "sbatch_not_found"})
            return blocked

        token = uuid.uuid4().hex[:12]
        script_path = self._jobs_dir / f"job_{token}.sh"
        stdout_path = self._logs_dir / f"{token}.out"
        stderr_path = self._logs_dir / f"{token}.err"
        exit_path = self._logs_dir / f"{token}.exit"
        job_name = f"autoforge-{getattr(self.config, 'run_id', 'run')}-{token[:6]}"

        self._write_job_script(command=command, exit_path=exit_path, script_path=script_path)
        sbatch_args = self._build_sbatch_args(
            script_path=script_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_s=int(timeout),
            job_name=job_name,
        )

        submit = await run_args(
            sbatch_args,
            cwd=self.working_dir,
            timeout_s=30.0,
            label="slurm.sbatch",
        )
        if submit.exit_code != 0:
            failed = SandboxResult(
                exit_code=submit.exit_code,
                stdout=submit.stdout,
                stderr=submit.stderr,
                timed_out=submit.timed_out,
                duration_seconds=submit.duration_s,
                used_fallback=submit.used_fallback,
            )
            self._record_command_receipt(command=command, capability=capability, result=failed, extra={"slurm": True, "stage": "submit"})
            return failed

        raw_out = (submit.stdout or "").strip()
        lines = raw_out.splitlines()
        job_id_raw = (lines[0].strip() if lines else raw_out).strip()
        job_id = job_id_raw.split(";", 1)[0].strip()
        if not job_id:
            failed = SandboxResult(
                exit_code=-1,
                stdout=submit.stdout,
                stderr=f"sbatch did not return a job id (stdout={submit.stdout!r} stderr={submit.stderr!r})",
                duration_seconds=submit.duration_s,
            )
            self._record_command_receipt(command=command, capability=capability, result=failed, extra={"slurm": True, "stage": "submit"})
            return failed

        self._active_job_ids.add(job_id)
        start = time.monotonic()
        queue_timeout = int(getattr(self.config, "slurm_queue_timeout_seconds", 600) or 0)
        poll_interval = float(getattr(self.config, "slurm_poll_interval_seconds", 1.0) or 1.0)
        deadline = start + float(max(0, queue_timeout)) + float(max(1, int(timeout)))

        exit_code: int | None = None
        timed_out = False
        try:
            while True:
                # Fast path: script wrote exit code file
                try:
                    if exit_path.is_file():
                        raw = exit_path.read_text(encoding="utf-8", errors="replace").strip()
                        if raw:
                            exit_code = int(raw.split()[0])
                            break
                except Exception:
                    pass

                now = time.monotonic()
                if now >= deadline:
                    timed_out = True
                    await self._try_cancel(job_id)
                    break

                # Queue-based detection
                try:
                    in_queue = await self._job_in_queue(job_id)
                except Exception:
                    in_queue = True
                if not in_queue:
                    # Job left the queue; attempt to resolve exit code via sacct.
                    try:
                        exit_code = await self._try_sacct_exit_code(job_id)
                    except Exception:
                        exit_code = None
                    break

                await asyncio.sleep(max(0.2, poll_interval))
        finally:
            self._active_job_ids.discard(job_id)

        # Collect logs (best-effort, tail-only)
        stdout = _read_file_tail(stdout_path)
        stderr = _read_file_tail(stderr_path)
        if exit_code is None:
            exit_code = -1 if not timed_out else 124
            if timed_out:
                stderr = (stderr + "\n" if stderr else "") + "[autoforge] slurm job timed out"
            else:
                stderr = (stderr + "\n" if stderr else "") + "[autoforge] slurm job finished but exit code was unavailable"

        duration = time.monotonic() - start
        result = SandboxResult(
            exit_code=int(exit_code),
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_seconds=float(duration),
            used_fallback=False,
        )
        if self.telemetry is not None:
            try:
                payload2: dict[str, Any] = {
                    "cwd": str(self.working_dir),
                    "command": str(command),
                    "capability": str(capability or ""),
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "duration_seconds": result.duration_seconds,
                    "used_fallback": result.used_fallback,
                    "slurm": True,
                    "slurm_job_id": job_id,
                }
                if bool(getattr(self.telemetry, "capture_command_output", False)):
                    payload2["stdout"] = result.stdout
                    payload2["stderr"] = result.stderr
                self.telemetry.record_command(payload2)
            except Exception:
                pass
        self._record_command_receipt(command=command, capability=capability, result=result, extra={"slurm": True, "slurm_job_id": job_id})
        return result

    async def exec_args(
        self,
        args: list[str],
        timeout: int = 120,
        *,
        capability: str | None = None,
    ) -> SandboxResult:
        """Execute an argument-vector command (posix quoting)."""
        if not args:
            return SandboxResult(exit_code=-1, stdout="", stderr="No command provided")
        command = " ".join(shlex.quote(str(a)) for a in args)
        return await self.exec(command, timeout=timeout, capability=capability)

    async def stop(self) -> None:
        # Best-effort cancel any outstanding jobs from this sandbox instance.
        if not self._active_job_ids:
            return
        job_ids = list(self._active_job_ids)
        self._active_job_ids.clear()
        for jid in job_ids:
            await self._try_cancel(jid)


def _docker_available() -> bool:
    """Check if Docker is available and the daemon is running."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def create_sandbox(
    config: ForgeConfig,
    working_dir: Path,
    *,
    telemetry: TelemetrySink | None = None,
) -> SandboxBase:
    """Factory: create the appropriate sandbox based on configuration."""
    backend = str(getattr(config, "execution_backend", "auto") or "auto").strip().lower()
    if backend not in ("auto", "docker", "subprocess", "slurm"):
        backend = "auto"

    selected = backend
    if backend == "auto":
        if bool(getattr(config, "docker_enabled", False)):
            if _docker_available():
                selected = "docker"
            else:
                if bool(getattr(config, "docker_required", False)):
                    raise RuntimeError("Docker requested but not available (docker_required=true)")
                selected = "slurm" if _slurm_available() else "subprocess"
        else:
            selected = "slurm" if _slurm_available() else "subprocess"

    env_overrides = build_env_overrides(config, backend=selected)

    if selected == "docker":
        if not _docker_available():
            raise RuntimeError("Docker backend requested but Docker is not available")
        logger.info("Using Docker sandbox")
        sb: SandboxBase = DockerSandbox(
            image=config.sandbox_image,
            working_dir=working_dir,
            memory_limit=str(getattr(config, "docker_memory_limit", "2g")),
            cpu_limit=str(getattr(config, "docker_cpu_limit", "2")),
            network_mode=str(getattr(config, "docker_network_mode", "none")),
            pids_limit=int(getattr(config, "docker_pids_limit", 512) or 0),
            env=env_overrides,
        )
    elif selected == "slurm":
        if not _slurm_available():
            raise RuntimeError("Slurm backend requested but sbatch was not found on PATH")
        logger.info("Using Slurm sandbox")
        sb = SlurmSandbox(working_dir=working_dir, config=config)
    else:
        logger.info("Using subprocess sandbox")
        sb = SubprocessSandbox(working_dir=working_dir, env=env_overrides, config=config)
    sb.telemetry = telemetry
    harness_root = (
        resolve_development_harness_root(config=config, project_dir=working_dir)
        / "execution_harness"
    )
    sb.configure_execution_harness(
        root=harness_root,
        backend=selected,
        config=config,
        working_dir=working_dir,
        env_overrides=env_overrides,
    )
    return sb
