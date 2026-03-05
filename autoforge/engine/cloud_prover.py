"""Cloud Prover — Remote Lean 4 Compilation & Proof Verification Service.

Offloads heavy Lean 4 compilation and proof search to cloud environments.
Supports multiple backends:
  1. GitHub Codespaces (Lean 4 dev container)
  2. Docker-based cloud build (any cloud provider with Docker)
  3. LeanDojo API (if available)
  4. Self-hosted Lean server (SSH-based)

This module handles the orchestration of remote proof verification,
including job submission, status polling, result retrieval, and caching.

Design:
  - CloudProverConfig controls which backends are enabled
  - Each backend implements the ProverBackendProtocol
  - Jobs are tracked with unique IDs and persisted for resume
  - Results are cached to avoid redundant compilation

Integration:
  - PaperFormalizer submits Lean code for cloud compilation
  - DiscoveryOrchestrator can verify discovered theorems at scale
  - Orchestrator wires this in when cloud_prover_enabled is True
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════


class CloudBackend(str, Enum):
    """Available cloud prover backends."""
    DOCKER_LOCAL = "docker_local"
    DOCKER_REMOTE = "docker_remote"
    SSH_SERVER = "ssh_server"
    GITHUB_CODESPACE = "github_codespace"
    LEANDOJO_API = "leandojo_api"


class JobStatus(str, Enum):
    """Status of a cloud proof job."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class CloudProverConfig:
    """Configuration for cloud prover backends."""
    # General
    enabled: bool = False
    preferred_backend: CloudBackend = CloudBackend.DOCKER_LOCAL
    max_concurrent_jobs: int = 4
    job_timeout_seconds: int = 600
    cache_dir: Path | None = None

    # Docker settings
    docker_image: str = "leanprover/lean4:v4.15.0"
    docker_mathlib_image: str = "autoforge-lean-mathlib:latest"
    docker_remote_host: str = ""
    docker_remote_user: str = ""

    # SSH server
    ssh_host: str = ""
    ssh_user: str = ""
    ssh_key_path: str = ""
    ssh_lean_path: str = "/usr/local/bin/lean"

    # GitHub Codespace
    codespace_name: str = ""
    github_token: str = ""

    # LeanDojo
    leandojo_api_url: str = ""
    leandojo_api_key: str = ""


@dataclass
class ProofJob:
    """A single proof verification job."""
    job_id: str
    lean_code: str
    label: str = ""
    backend: CloudBackend = CloudBackend.DOCKER_LOCAL
    status: JobStatus = JobStatus.QUEUED
    result: str = ""
    errors: list[str] = field(default_factory=list)
    submitted_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    duration_seconds: float = 0.0
    has_sorry: bool = False
    compiled_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "label": self.label,
            "backend": self.backend.value,
            "status": self.status.value,
            "result": self.result[:2000],
            "errors": self.errors,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "has_sorry": self.has_sorry,
            "compiled_ok": self.compiled_ok,
        }


# ══════════════════════════════════════════════════════════════
# Backend Protocol
# ══════════════════════════════════════════════════════════════


class ProverBackendProtocol(Protocol):
    """Protocol for cloud prover backends."""

    async def is_available(self) -> bool:
        """Check if the backend is available and configured."""
        ...

    async def submit(self, job: ProofJob) -> ProofJob:
        """Submit a proof job for compilation/verification."""
        ...

    async def poll(self, job: ProofJob) -> ProofJob:
        """Poll job status."""
        ...

    async def cancel(self, job: ProofJob) -> None:
        """Cancel a running job."""
        ...


# ══════════════════════════════════════════════════════════════
# Docker Local Backend
# ══════════════════════════════════════════════════════════════


class DockerLocalBackend:
    """Run Lean 4 compilation in a local Docker container.

    This is the simplest and most reliable backend. Requires Docker
    to be installed and the Lean 4 image to be available.
    """

    def __init__(self, config: CloudProverConfig) -> None:
        self.config = config
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    async def is_available(self) -> bool:
        """Check if Docker is installed and the image exists."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def submit(self, job: ProofJob) -> ProofJob:
        """Run Lean 4 compilation in a Docker container."""
        job.status = JobStatus.RUNNING
        job.submitted_at = time.time()

        try:
            # Write Lean code to temp file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.lean', delete=False, prefix='autoforge_'
            ) as f:
                f.write(job.lean_code)
                lean_file = f.name

            lean_dir = os.path.dirname(lean_file)
            lean_name = os.path.basename(lean_file)

            # Run in Docker container
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{lean_dir}:/workspace",
                "-w", "/workspace",
                "--memory", "4g",
                "--cpus", "2",
                self.config.docker_image,
                "lean", lean_name,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.job_timeout_seconds,
                )

                job.completed_at = time.time()
                job.duration_seconds = job.completed_at - job.submitted_at

                stdout_text = stdout.decode("utf-8", errors="replace")
                stderr_text = stderr.decode("utf-8", errors="replace")

                if process.returncode == 0:
                    job.status = JobStatus.COMPLETED
                    job.compiled_ok = True
                    job.has_sorry = "sorry" in job.lean_code.lower()
                    job.result = stdout_text or "Compilation successful"
                else:
                    job.status = JobStatus.FAILED
                    job.errors.append(stderr_text[:2000])
                    job.result = stderr_text

            except asyncio.TimeoutError:
                process.kill()
                job.status = JobStatus.TIMEOUT
                job.errors.append(f"Timeout after {self.config.job_timeout_seconds}s")

            # Clean up
            try:
                os.unlink(lean_file)
            except OSError:
                pass

        except Exception as e:
            job.status = JobStatus.FAILED
            job.errors.append(str(e))
            logger.warning(f"[DockerLocal] Job {job.job_id} failed: {e}")

        return job

    async def poll(self, job: ProofJob) -> ProofJob:
        """No-op for local Docker (submit is synchronous)."""
        return job

    async def cancel(self, job: ProofJob) -> None:
        """Cancel a running Docker job."""
        proc = self._processes.get(job.job_id)
        if proc:
            proc.kill()
            job.status = JobStatus.CANCELLED


# ══════════════════════════════════════════════════════════════
# SSH Server Backend
# ══════════════════════════════════════════════════════════════


class SSHServerBackend:
    """Run Lean 4 compilation on a remote SSH server.

    Useful for powerful cloud VMs with Lean 4 + Mathlib pre-installed.
    """

    def __init__(self, config: CloudProverConfig) -> None:
        self.config = config

    async def is_available(self) -> bool:
        """Check if SSH server is reachable."""
        if not self.config.ssh_host:
            return False
        try:
            import shutil
            if not shutil.which("ssh"):
                return False

            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5",
                 "-o", "StrictHostKeyChecking=no",
                 f"{self.config.ssh_user}@{self.config.ssh_host}",
                 "echo ok"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0 and "ok" in result.stdout
        except Exception:
            return False

    async def submit(self, job: ProofJob) -> ProofJob:
        """Submit Lean code to remote server via SSH."""
        job.status = JobStatus.RUNNING
        job.submitted_at = time.time()

        try:
            # Write to temp file locally
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.lean', delete=False
            ) as f:
                f.write(job.lean_code)
                local_file = f.name

            remote_file = f"/tmp/autoforge_{job.job_id}.lean"
            host = f"{self.config.ssh_user}@{self.config.ssh_host}"

            # SCP file to remote
            scp_cmd = ["scp", "-o", "StrictHostKeyChecking=no"]
            if self.config.ssh_key_path:
                scp_cmd.extend(["-i", self.config.ssh_key_path])
            scp_cmd.extend([local_file, f"{host}:{remote_file}"])

            scp_proc = await asyncio.create_subprocess_exec(
                *scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await scp_proc.communicate()

            # Run lean on remote
            ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
            if self.config.ssh_key_path:
                ssh_cmd.extend(["-i", self.config.ssh_key_path])
            import shlex
            ssh_cmd.extend([
                host,
                f"{shlex.quote(self.config.ssh_lean_path)} {shlex.quote(remote_file)} && rm {shlex.quote(remote_file)}",
            ])

            lean_proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    lean_proc.communicate(),
                    timeout=self.config.job_timeout_seconds,
                )

                job.completed_at = time.time()
                job.duration_seconds = job.completed_at - job.submitted_at

                if lean_proc.returncode == 0:
                    job.status = JobStatus.COMPLETED
                    job.compiled_ok = True
                    job.has_sorry = "sorry" in job.lean_code.lower()
                    job.result = stdout.decode("utf-8", errors="replace")
                else:
                    job.status = JobStatus.FAILED
                    job.errors.append(stderr.decode("utf-8", errors="replace")[:2000])

            except asyncio.TimeoutError:
                lean_proc.kill()
                job.status = JobStatus.TIMEOUT

            os.unlink(local_file)

        except Exception as e:
            job.status = JobStatus.FAILED
            job.errors.append(str(e))

        return job

    async def poll(self, job: ProofJob) -> ProofJob:
        return job

    async def cancel(self, job: ProofJob) -> None:
        pass


# ══════════════════════════════════════════════════════════════
# Result Cache
# ══════════════════════════════════════════════════════════════


class ProofCache:
    """Cache proof compilation results to avoid redundant work."""

    _MAX_MEMORY_ENTRIES: int = 2048  # Bounded in-memory cache

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir
        self._memory_cache: dict[str, ProofJob] = {}
        self._access_order: list[str] = []  # LRU tracking

    def _hash_code(self, lean_code: str) -> str:
        return hashlib.sha256(lean_code.encode("utf-8")).hexdigest()[:16]

    def get(self, lean_code: str) -> ProofJob | None:
        """Look up a cached result."""
        key = self._hash_code(lean_code)

        # Memory cache first
        if key in self._memory_cache:
            logger.debug(f"[ProofCache] Memory hit: {key}")
            # Move to end of access order (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._memory_cache[key]

        # Disk cache
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    job = ProofJob(
                        job_id=data["job_id"],
                        lean_code=lean_code,
                        label=data.get("label", ""),
                        status=JobStatus(data["status"]),
                        result=data.get("result", ""),
                        errors=data.get("errors", []),
                        compiled_ok=data.get("compiled_ok", False),
                        has_sorry=data.get("has_sorry", False),
                    )
                    self._memory_cache[key] = job
                    logger.debug(f"[ProofCache] Disk hit: {key}")
                    return job
                except Exception as e:
                    logger.debug(f"[ProofCache] Failed to load {key}: {e}")

        return None

    def put(self, job: ProofJob) -> None:
        """Cache a completed job result (LRU-bounded)."""
        key = self._hash_code(job.lean_code)
        # Evict oldest entries when at capacity
        while len(self._memory_cache) >= self._MAX_MEMORY_ENTRIES:
            if self._access_order:
                evict_key = self._access_order.pop(0)
                self._memory_cache.pop(evict_key, None)
            else:
                # Fallback: clear an arbitrary entry
                self._memory_cache.pop(next(iter(self._memory_cache)), None)
                break
        self._memory_cache[key] = job
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_dir / f"{key}.json"
            try:
                cache_file.write_text(
                    json.dumps(job.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.debug(f"[ProofCache] Failed to save {key}: {e}")


# ══════════════════════════════════════════════════════════════
# Cloud Prover Orchestrator
# ══════════════════════════════════════════════════════════════


class CloudProver:
    """Orchestrate cloud-based Lean 4 proof verification.

    Manages multiple backends, job scheduling, caching, and result aggregation.
    """

    def __init__(self, config: CloudProverConfig | None = None) -> None:
        self.config = config or CloudProverConfig()
        self._cache = ProofCache(self.config.cache_dir)
        self._backends: dict[CloudBackend, Any] = {}
        self._jobs: list[ProofJob] = []
        self._init_backends()

    def _init_backends(self) -> None:
        """Initialize enabled backends."""
        self._backends[CloudBackend.DOCKER_LOCAL] = DockerLocalBackend(self.config)
        if self.config.ssh_host:
            self._backends[CloudBackend.SSH_SERVER] = SSHServerBackend(self.config)

    async def detect_available(self) -> list[CloudBackend]:
        """Detect which backends are available."""
        available = []
        for backend_type, backend in self._backends.items():
            try:
                if await backend.is_available():
                    available.append(backend_type)
                    logger.info(f"[CloudProver] Backend available: {backend_type.value}")
            except Exception as e:
                logger.debug(f"[CloudProver] Backend {backend_type.value} check failed: {e}")
        return available

    async def verify_lean(
        self,
        lean_code: str,
        label: str = "",
        *,
        backend: CloudBackend | None = None,
    ) -> ProofJob:
        """Verify a single Lean 4 file.

        Args:
            lean_code: Complete Lean 4 source code
            label: Human-readable label for the job
            backend: Force a specific backend (or use preferred)

        Returns:
            ProofJob with results
        """
        # Check cache first
        cached = self._cache.get(lean_code)
        if cached and cached.status == JobStatus.COMPLETED:
            logger.info(f"[CloudProver] Cache hit for {label}")
            return cached

        # Create job
        job_id = hashlib.sha256(
            f"{lean_code}{time.time()}".encode()
        ).hexdigest()[:12]

        job = ProofJob(
            job_id=job_id,
            lean_code=lean_code,
            label=label,
            backend=backend or self.config.preferred_backend,
        )

        # Select backend
        backend_impl = self._backends.get(job.backend)
        if backend_impl is None:
            job.status = JobStatus.FAILED
            job.errors.append(f"Backend {job.backend.value} not initialized")
            return job

        # Submit
        logger.info(f"[CloudProver] Submitting {label} to {job.backend.value}")
        job = await backend_impl.submit(job)
        self._jobs.append(job)

        # Cache result
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            self._cache.put(job)

        return job

    async def verify_batch(
        self,
        items: list[tuple[str, str]],  # (lean_code, label) pairs
        *,
        backend: CloudBackend | None = None,
    ) -> list[ProofJob]:
        """Verify multiple Lean files concurrently.

        Respects max_concurrent_jobs limit.
        """
        results: list[ProofJob] = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_jobs)

        async def _verify_one(code: str, label: str) -> ProofJob:
            async with semaphore:
                return await self.verify_lean(code, label, backend=backend)

        tasks = [_verify_one(code, label) for code, label in items]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about cloud prover usage."""
        stats: dict[str, int] = {
            "total_jobs": len(self._jobs),
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "compiled_ok": 0,
            "has_sorry": 0,
        }
        total_duration = 0.0

        for job in self._jobs:
            if job.status == JobStatus.COMPLETED:
                stats["completed"] += 1
            elif job.status == JobStatus.FAILED:
                stats["failed"] += 1
            elif job.status == JobStatus.TIMEOUT:
                stats["timeout"] += 1

            if job.compiled_ok:
                stats["compiled_ok"] += 1
            if job.has_sorry:
                stats["has_sorry"] += 1
            total_duration += job.duration_seconds

        stats_out: dict[str, Any] = dict(stats)
        stats_out["total_duration_seconds"] = total_duration
        stats_out["avg_duration_seconds"] = (
            total_duration / max(stats["total_jobs"], 1)
        )
        return stats_out

    def save_state(self, path: Path) -> None:
        """Save cloud prover state for resume."""
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "jobs": [j.to_dict() for j in self._jobs],
            "statistics": self.get_statistics(),
        }
        (path / "cloud_prover_state.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_state(self, path: Path) -> None:
        """Load cloud prover state."""
        state_file = path / "cloud_prover_state.json"
        if not state_file.exists():
            return
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            # Reconstruct jobs
            for jdata in data.get("jobs", []):
                job = ProofJob(
                    job_id=jdata["job_id"],
                    lean_code="",  # Don't store full code in state
                    label=jdata.get("label", ""),
                    backend=CloudBackend(jdata.get("backend", "docker_local")),
                    status=JobStatus(jdata.get("status", "completed")),
                    result=jdata.get("result", ""),
                    compiled_ok=jdata.get("compiled_ok", False),
                    has_sorry=jdata.get("has_sorry", False),
                )
                self._jobs.append(job)
            logger.info(f"[CloudProver] Loaded {len(self._jobs)} jobs from state")
        except Exception as e:
            logger.warning(f"[CloudProver] Failed to load state: {e}")
