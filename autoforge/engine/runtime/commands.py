"""Unified command execution with Windows-safe fallback.

This is the single source of truth for subprocess execution semantics:
  - async first (asyncio subprocess)
  - safe fallback to sync subprocess.run when pipes are disallowed (Windows sandbox)
  - timeout enforcement + best-effort process cleanup
  - bounded stdout/stderr capture to prevent context blowups
"""

from __future__ import annotations

import asyncio
import locale
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_STDOUT_CHARS = 5000
DEFAULT_MAX_STDERR_CHARS = 2000


_PREFERRED_ENCODING = locale.getpreferredencoding(False) or "utf-8"


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = max(0, limit - 64)
    return text[:head] + "\n... (truncated) ...\n" + text[-48:]


def _decode(data: bytes | None) -> str:
    if not data:
        return ""
    try:
        # Prefer UTF-8 (most modern tools), then fall back to locale encoding.
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode(_PREFERRED_ENCODING, errors="replace")
        except Exception:
            return data.decode(errors="replace")
    except Exception:
        return data.decode(errors="replace")


def _merge_env(env: Mapping[str, str] | None) -> Mapping[str, str] | None:
    """Merge an env overlay onto the current process environment.

    Python's subprocess APIs treat env= as a full replacement, which is almost
    never what we want in an orchestration engine. Call-sites should provide
    only overrides; this function ensures PATH and other essentials remain.
    """
    if env is None:
        return None
    merged = dict(os.environ)
    for k, v in env.items():
        merged[str(k)] = str(v)
    return merged


@dataclass(frozen=True)
class CommandSpec:
    cwd: Path
    timeout_s: float = DEFAULT_TIMEOUT_S
    env: Mapping[str, str] | None = None
    stdin: bytes | None = None

    # Exactly one of these must be set.
    args: Sequence[str] | None = None
    command: str | None = None
    shell: bool = False

    # Output bounding
    max_stdout_chars: int = DEFAULT_MAX_STDOUT_CHARS
    max_stderr_chars: int = DEFAULT_MAX_STDERR_CHARS

    label: str = ""

    def display(self) -> str:
        if self.args is not None:
            return " ".join(str(x) for x in self.args)
        return str(self.command or "")


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    used_fallback: bool
    exception: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def _process_kwargs() -> dict:
    kwargs: dict = {}
    if sys.platform != "win32":
        kwargs["preexec_fn"] = os.setsid
    return kwargs


async def run_command(spec: CommandSpec) -> CommandResult:
    """Run a command from a CommandSpec."""
    if (spec.args is None) == (spec.command is None):
        raise ValueError("CommandSpec must set exactly one of args= or command=")

    start = time.monotonic()
    used_fallback = False
    timed_out = False
    merged_env = _merge_env(spec.env)

    async def _run_async() -> CommandResult:
        proc: asyncio.subprocess.Process | None = None
        try:
            if spec.args is not None:
                proc = await asyncio.create_subprocess_exec(
                    *list(spec.args),
                    cwd=str(spec.cwd),
                    env=dict(merged_env) if merged_env else None,
                    stdin=asyncio.subprocess.PIPE if spec.stdin is not None else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **_process_kwargs(),
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    str(spec.command or ""),
                    cwd=str(spec.cwd),
                    env=dict(merged_env) if merged_env else None,
                    stdin=asyncio.subprocess.PIPE if spec.stdin is not None else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **_process_kwargs(),
                )

            if spec.stdin is not None:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=spec.stdin), timeout=spec.timeout_s
                )
            else:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=spec.timeout_s)
            out = _decode(stdout_b)
            err = _decode(stderr_b)
            return CommandResult(
                exit_code=int(proc.returncode or 0),
                stdout=_truncate(out, spec.max_stdout_chars),
                stderr=_truncate(err, spec.max_stderr_chars),
                timed_out=False,
                duration_s=time.monotonic() - start,
                used_fallback=False,
            )
        except asyncio.CancelledError:
            # Ensure spawned process doesn't leak if the orchestration cancels the task.
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise
        except asyncio.TimeoutError:
            nonlocal timed_out
            timed_out = True
            if proc is not None:
                try:
                    if sys.platform != "win32" and proc.pid is not None:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except (ProcessLookupError, OSError):
                            proc.kill()
                    else:
                        proc.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    await proc.communicate()
                except Exception:
                    pass
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=_truncate(f"Command timed out after {spec.timeout_s:.0f}s", spec.max_stderr_chars),
                timed_out=True,
                duration_s=time.monotonic() - start,
                used_fallback=False,
            )

    def _run_sync() -> CommandResult:
        nonlocal used_fallback, timed_out
        used_fallback = True
        try:
            common: dict[str, object] = {
                "cwd": str(spec.cwd),
                "env": dict(merged_env) if merged_env else None,
                "capture_output": True,
                "timeout": spec.timeout_s,
            }
            if spec.stdin is not None:
                common["input"] = spec.stdin
            if spec.args is not None:
                completed = subprocess.run(
                    list(spec.args),
                    shell=False,
                    **common,
                )
            else:
                completed = subprocess.run(
                    str(spec.command or ""),
                    shell=True,
                    **common,
                )
            return CommandResult(
                exit_code=int(completed.returncode),
                stdout=_truncate(_decode(completed.stdout), spec.max_stdout_chars),
                stderr=_truncate(_decode(completed.stderr), spec.max_stderr_chars),
                timed_out=False,
                duration_s=time.monotonic() - start,
                used_fallback=True,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=_truncate(f"Command timed out after {spec.timeout_s:.0f}s", spec.max_stderr_chars),
                timed_out=True,
                duration_s=time.monotonic() - start,
                used_fallback=True,
            )
        except FileNotFoundError as e:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=_truncate(str(e), spec.max_stderr_chars),
                timed_out=False,
                duration_s=time.monotonic() - start,
                used_fallback=True,
                exception=str(e),
            )
        except Exception as e:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=_truncate(str(e), spec.max_stderr_chars),
                timed_out=False,
                duration_s=time.monotonic() - start,
                used_fallback=True,
                exception=str(e),
            )

    try:
        return await _run_async()
    except (OSError, PermissionError, NotImplementedError) as e:
        # Some sandboxed Windows environments disallow asyncio subprocess pipes.
        res = await asyncio.to_thread(_run_sync)
        if not res.exception:
            res.exception = f"asyncio_subprocess_failed: {type(e).__name__}: {e}"
        return res


async def run_args(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_stdout_chars: int = DEFAULT_MAX_STDOUT_CHARS,
    max_stderr_chars: int = DEFAULT_MAX_STDERR_CHARS,
    label: str = "",
    stdin: bytes | None = None,
) -> CommandResult:
    return await run_command(
        CommandSpec(
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            args=list(args),
            command=None,
            shell=False,
            max_stdout_chars=max_stdout_chars,
            max_stderr_chars=max_stderr_chars,
            label=label,
        )
    )


async def run_shell(
    command: str,
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_stdout_chars: int = DEFAULT_MAX_STDOUT_CHARS,
    max_stderr_chars: int = DEFAULT_MAX_STDERR_CHARS,
    label: str = "",
    stdin: bytes | None = None,
) -> CommandResult:
    return await run_command(
        CommandSpec(
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            args=None,
            command=str(command),
            shell=True,
            max_stdout_chars=max_stdout_chars,
            max_stderr_chars=max_stderr_chars,
            label=label,
        )
    )


async def spawn_exec(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    stdin_pipe: bool = True,
    stdout_pipe: bool = True,
    stderr_pipe: bool = True,
    label: str = "",
) -> asyncio.subprocess.Process:
    """Spawn a subprocess for interactive protocols (REPLs).

    This does *not* provide the sync fallback that run_command() has; interactive
    processes require asyncio pipes. Callers should handle exceptions and offer
    graceful degradation if the environment is too restricted.
    """
    if not args:
        raise ValueError("spawn_exec requires a non-empty args list")
    merged_env = _merge_env(env)
    _ = label  # reserved for future telemetry integration
    return await asyncio.create_subprocess_exec(
        *list(args),
        cwd=str(cwd),
        env=dict(merged_env) if merged_env else None,
        stdin=asyncio.subprocess.PIPE if stdin_pipe else None,
        stdout=asyncio.subprocess.PIPE if stdout_pipe else None,
        stderr=asyncio.subprocess.PIPE if stderr_pipe else None,
        **_process_kwargs(),
    )
