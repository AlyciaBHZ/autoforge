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
import signal
import subprocess
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoforge.engine.config import ForgeConfig

logger = logging.getLogger(__name__)


def _shell_quote(s: str) -> str:
    """Quote a string for shell use, cross-platform."""
    if sys.platform == "win32":
        return f'"{s}"'
    return shlex.quote(s)


@dataclass
class SandboxResult:
    """Result of a command execution in the sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class SandboxBase(ABC):
    """Abstract sandbox for command execution."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize the sandbox."""

    @abstractmethod
    async def exec(self, command: str, timeout: int = 120) -> SandboxResult:
        """Execute a command in the sandbox."""

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
        r"\brm\s+-rf\s+/",              # rm -rf /
        r"\brm\s+-fr\s+/",              # rm -fr /
        r"\bmkfs\b",                     # mkfs (format disk)
        r"\bdd\s+if=",                   # dd if= (raw disk write)
        r"\bchmod\s+777\s+/",           # chmod 777 / (recursive perm open)
        r"\bcurl\b.*\|\s*\bsh\b",       # curl | sh (remote code exec)
        r"\bcurl\b.*\|\s*\bbash\b",     # curl | bash
        r"\bwget\b.*\|\s*\bsh\b",       # wget | sh
        r"\bwget\b.*\|\s*\bbash\b",     # wget | bash
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
    ]

    def __init__(self, working_dir: Path, env: dict[str, str] | None = None) -> None:
        self.working_dir = working_dir
        self.env = env

    @classmethod
    def _sanitize_command(cls, command: str) -> None:
        """Check command against blocked patterns and raise ValueError if matched."""
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                raise ValueError(
                    f"Blocked dangerous command pattern: {pattern!r}"
                )

    async def start(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SubprocessSandbox ready at {self.working_dir}")

    async def exec(self, command: str, timeout: int = 120) -> SandboxResult:
        """Execute a command as a subprocess."""
        logger.debug(f"[sandbox] exec: {command[:100]}")

        # Validate command against blocked patterns
        try:
            self._sanitize_command(command)
        except ValueError as e:
            logger.warning(f"[sandbox] Command blocked: {e}")
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command blocked by security filter: {e}",
            )

        # On Windows, wrap Unix-style commands to run via bash/sh if available
        shell_command = command
        if sys.platform == "win32" and not command.startswith("cmd"):
            sh = shutil.which("bash") or shutil.which("sh")
            if sh:
                shell_command = f'"{sh}" -c {_shell_quote(command)}'

        proc: asyncio.subprocess.Process | None = None
        try:
            # On POSIX, start a new process group so we can kill the entire
            # tree on timeout (not just the shell parent).
            kwargs: dict = {}
            if sys.platform != "win32":
                kwargs["preexec_fn"] = os.setsid

            proc = await asyncio.create_subprocess_shell(
                shell_command,
                cwd=self.working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env,
                **kwargs,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return SandboxResult(
                exit_code=proc.returncode if proc.returncode is not None else -1,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )
        except asyncio.TimeoutError:
            if proc is not None:
                try:
                    if sys.platform != "win32" and proc.pid is not None:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    await proc.communicate()
                except Exception:
                    pass
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                timed_out=True,
            )
        except Exception as e:
            return SandboxResult(
                exit_code=-1, stdout="", stderr=str(e)
            )

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
    ) -> None:
        self.image = image
        self.working_dir = working_dir
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self._container_id: str | None = None
        self._container_name: str = f"autoforge-sandbox-{uuid.uuid4().hex[:8]}"

    async def start(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        container_name = self._container_name

        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-v", f"{self.working_dir.resolve()}:/workspace",
            "-w", "/workspace",
            "--network", "none",
            "--memory", self.memory_limit,
            "--cpus", self.cpu_limit,
            self.image,
            "tail", "-f", "/dev/null",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode == 0:
            self._container_id = stdout.decode().strip()
            logger.info(f"DockerSandbox started: {self._container_id[:12]}")
        else:
            raise RuntimeError(f"Failed to start Docker sandbox: {stderr.decode()}")

    async def exec(self, command: str, timeout: int = 120) -> SandboxResult:
        if not self._container_id:
            raise RuntimeError("Docker sandbox not started")

        cmd = ["docker", "exec", self._container_id, "bash", "-c", command]
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return SandboxResult(
                exit_code=proc.returncode if proc.returncode is not None else -1,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )
        except asyncio.TimeoutError:
            if proc is not None:
                try:
                    proc.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    await proc.communicate()
                except Exception:
                    pass
            try:
                kill_cmd = [
                    "docker", "exec", self._container_id,
                    "bash", "-c", "kill -9 -1 2>/dev/null || true",
                ]
                kill_proc = await asyncio.create_subprocess_exec(
                    *kill_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(kill_proc.communicate(), timeout=5)
            except Exception:
                pass
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                timed_out=True,
            )

    async def stop(self) -> None:
        if self._container_id:
            cmd = ["docker", "rm", "-f", self._container_id]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            logger.info(f"DockerSandbox stopped: {self._container_id[:12]}")
            self._container_id = None


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


def create_sandbox(config: ForgeConfig, working_dir: Path) -> SandboxBase:
    """Factory: create the appropriate sandbox based on configuration."""
    if config.docker_enabled and _docker_available():
        logger.info("Using Docker sandbox")
        return DockerSandbox(image=config.sandbox_image, working_dir=working_dir)
    else:
        if config.docker_enabled:
            logger.warning("Docker requested but not available, falling back to subprocess")
        logger.info("Using subprocess sandbox")
        return SubprocessSandbox(working_dir=working_dir)
