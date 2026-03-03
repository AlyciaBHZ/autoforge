"""Sandbox — safe command execution environment.

Supports two modes:
- SubprocessSandbox: Direct subprocess execution (default, no Docker required)
- DockerSandbox: Isolated Docker container execution (optional)
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from autoforge.engine.config import ForgeConfig

logger = logging.getLogger(__name__)


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

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()


class SubprocessSandbox(SandboxBase):
    """Sandbox that runs commands as local subprocesses.

    Used when Docker is not available. Commands run in a specified
    working directory with timeout enforcement.
    """

    def __init__(self, working_dir: Path, env: dict[str, str] | None = None) -> None:
        self.working_dir = working_dir
        self.env = env

    async def start(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SubprocessSandbox ready at {self.working_dir}")

    async def exec(self, command: str, timeout: int = 120) -> SandboxResult:
        """Execute a command as a subprocess."""
        logger.debug(f"[sandbox] exec: {command[:100]}")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self.working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return SandboxResult(
                exit_code=proc.returncode or 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )
        except asyncio.TimeoutError:
            proc.kill()
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

    async def start(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        container_name = f"autoforge-sandbox-{id(self) % 100000}"

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
                exit_code=proc.returncode or 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )
        except asyncio.TimeoutError:
            proc.kill()
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
    """Check if Docker is available and running."""
    return shutil.which("docker") is not None


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
