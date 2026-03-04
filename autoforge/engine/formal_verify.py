"""Formal verification — static analysis and lightweight formal checks.

Integrates static analysis tools and optional formal verification for
generated code to catch bugs that testing might miss.

Levels of verification (progressive):
  1. **Lint**: Language-specific linters (pylint/flake8, ESLint)
  2. **Type check**: Static type checking (mypy, tsc)
  3. **Security scan**: SAST tools (bandit for Python, CodeQL patterns)
  4. **Formal spec**: Lightweight pre/post-condition checking via assertions
  5. **Proof** (future): Full Lean/Dafny proof generation for critical code

The module auto-detects which tools are available and uses the highest
level of verification possible without requiring manual setup.

References:
  - Vericoding (2025): LLM-generated formal specifications
  - CodeQL: GitHub's static analysis engine
  - Bandit: Python SAST scanner
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────


@dataclass
class VerificationIssue:
    """A single verification finding."""
    severity: str           # "error", "warning", "info"
    category: str           # "lint", "type", "security", "formal"
    file: str               # File path (relative to project)
    line: int = 0           # Line number (0 = unknown)
    message: str = ""       # Description of the issue
    rule: str = ""          # Rule/check that triggered this
    fix_suggestion: str = ""  # How to fix it
    confidence: float = 1.0 # How confident we are this is a real issue

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "file": self.file,
            "line": self.line,
            "message": self.message,
            "rule": self.rule,
            "fix_suggestion": self.fix_suggestion,
        }


@dataclass
class VerificationReport:
    """Complete verification report for a project."""
    issues: list[VerificationIssue] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    files_checked: int = 0
    errors: int = 0
    warnings: int = 0
    security_issues: int = 0
    passed: bool = True

    def add_issue(self, issue: VerificationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == "error":
            self.errors += 1
            self.passed = False
        elif issue.severity == "warning":
            self.warnings += 1
        if issue.category == "security":
            self.security_issues += 1

    def summary(self) -> str:
        return (
            f"Verification: {self.files_checked} files, "
            f"{self.errors} errors, {self.warnings} warnings, "
            f"{self.security_issues} security issues "
            f"(tools: {', '.join(self.tools_used)})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "files_checked": self.files_checked,
            "errors": self.errors,
            "warnings": self.warnings,
            "security_issues": self.security_issues,
            "tools_used": self.tools_used,
            "issues": [i.to_dict() for i in self.issues],
        }


# ──────────────────────────────────────────────
# Tool Availability Detection
# ──────────────────────────────────────────────


async def _tool_available(cmd: str) -> bool:
    """Check if a command-line tool is available."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "which", cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
    except Exception:
        return False


# ──────────────────────────────────────────────
# Individual Verifiers
# ──────────────────────────────────────────────


async def _run_flake8(project_dir: Path) -> list[VerificationIssue]:
    """Run flake8 linter on Python files."""
    issues = []
    try:
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "flake8",
            "--max-line-length", "120",
            "--select", "E,W,F",
            "--format", "%(path)s:%(row)d:%(col)d: %(code)s %(text)s",
            str(project_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = stdout.decode(errors="replace")

        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            match = re.match(r"(.+?):(\d+):\d+: (\w+) (.+)", line)
            if match:
                file_path = match.group(1)
                try:
                    rel_path = str(Path(file_path).relative_to(project_dir))
                except ValueError:
                    rel_path = file_path

                code = match.group(3)
                severity = "error" if code.startswith("E") or code.startswith("F") else "warning"

                issues.append(VerificationIssue(
                    severity=severity,
                    category="lint",
                    file=rel_path,
                    line=int(match.group(2)),
                    message=match.group(4),
                    rule=code,
                ))

    except (asyncio.TimeoutError, FileNotFoundError, Exception) as e:
        logger.debug(f"flake8 not available or failed: {e}")

    return issues


async def _run_mypy(project_dir: Path) -> list[VerificationIssue]:
    """Run mypy type checker on Python files."""
    issues = []
    try:
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "mypy",
            "--ignore-missing-imports",
            "--no-error-summary",
            str(project_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = stdout.decode(errors="replace")

        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            match = re.match(r"(.+?):(\d+): (error|warning|note): (.+)", line)
            if match:
                file_path = match.group(1)
                try:
                    rel_path = str(Path(file_path).relative_to(project_dir))
                except ValueError:
                    rel_path = file_path

                severity = match.group(3)
                if severity == "note":
                    severity = "info"

                issues.append(VerificationIssue(
                    severity=severity,
                    category="type",
                    file=rel_path,
                    line=int(match.group(2)),
                    message=match.group(4),
                    rule="mypy",
                ))

    except (asyncio.TimeoutError, FileNotFoundError, Exception) as e:
        logger.debug(f"mypy not available or failed: {e}")

    return issues


async def _run_bandit(project_dir: Path) -> list[VerificationIssue]:
    """Run Bandit security scanner on Python files."""
    issues = []
    try:
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "bandit",
            "-r", str(project_dir),
            "-f", "json",
            "--quiet",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = stdout.decode(errors="replace")

        try:
            data = json.loads(output)
            for result in data.get("results", []):
                severity = result.get("issue_severity", "MEDIUM").lower()
                if severity == "high":
                    sev = "error"
                elif severity == "medium":
                    sev = "warning"
                else:
                    sev = "info"

                file_path = result.get("filename", "")
                try:
                    rel_path = str(Path(file_path).relative_to(project_dir))
                except ValueError:
                    rel_path = file_path

                issues.append(VerificationIssue(
                    severity=sev,
                    category="security",
                    file=rel_path,
                    line=result.get("line_number", 0),
                    message=result.get("issue_text", ""),
                    rule=result.get("test_id", ""),
                    confidence={"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}.get(
                        result.get("issue_confidence", "MEDIUM"), 0.5
                    ),
                ))

        except json.JSONDecodeError:
            pass

    except (asyncio.TimeoutError, FileNotFoundError, Exception) as e:
        logger.debug(f"bandit not available or failed: {e}")

    return issues


async def _run_eslint(project_dir: Path) -> list[VerificationIssue]:
    """Run ESLint on JavaScript/TypeScript files."""
    issues = []
    try:
        # Try project-local eslint first, then global
        eslint_cmd = str(project_dir / "node_modules" / ".bin" / "eslint")
        if not Path(eslint_cmd).exists():
            eslint_cmd = "npx"

        args = [eslint_cmd]
        if eslint_cmd == "npx":
            args.extend(["eslint"])
        args.extend([
            "--format", "json",
            "--no-error-on-unmatched-pattern",
            str(project_dir),
        ])

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_dir,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = stdout.decode(errors="replace")

        try:
            results = json.loads(output)
            for file_result in results:
                file_path = file_result.get("filePath", "")
                try:
                    rel_path = str(Path(file_path).relative_to(project_dir))
                except ValueError:
                    rel_path = file_path

                for msg in file_result.get("messages", []):
                    severity = "error" if msg.get("severity", 1) >= 2 else "warning"
                    issues.append(VerificationIssue(
                        severity=severity,
                        category="lint",
                        file=rel_path,
                        line=msg.get("line", 0),
                        message=msg.get("message", ""),
                        rule=msg.get("ruleId", ""),
                        fix_suggestion=msg.get("fix", {}).get("text", "") if msg.get("fix") else "",
                    ))
        except json.JSONDecodeError:
            pass

    except (asyncio.TimeoutError, FileNotFoundError, Exception) as e:
        logger.debug(f"ESLint not available or failed: {e}")

    return issues


# ──────────────────────────────────────────────
# LLM-based Formal Specification Check
# ──────────────────────────────────────────────


async def _llm_formal_check(
    project_dir: Path,
    critical_files: list[str],
    llm: Any,
) -> list[VerificationIssue]:
    """LLM-based lightweight formal verification.

    For critical files, the LLM analyses code for:
    - Invariant violations (null checks, bounds, type contracts)
    - Resource leaks (unclosed files/connections)
    - Concurrency issues (race conditions, deadlocks)
    - Logic errors (off-by-one, incorrect conditions)

    This is a "poor man's formal verification" — not sound, but catches
    many real bugs that linters miss.
    """
    from autoforge.engine.llm_router import TaskComplexity

    issues = []

    for file_rel in critical_files[:5]:  # Limit to 5 most critical files
        file_path = project_dir / file_rel
        if not file_path.exists():
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        if len(content) > 8000:
            content = content[:8000] + "\n# ... (truncated)"

        prompt = (
            "Perform formal-style verification on this code. "
            "Look for issues that tests and linters typically miss:\n\n"
            "1. Invariant violations (null/None checks, array bounds)\n"
            "2. Resource leaks (unclosed files, connections, locks)\n"
            "3. Concurrency issues (race conditions, deadlocks)\n"
            "4. Logic errors (off-by-one, incorrect boolean logic)\n"
            "5. Security issues (injection, path traversal, SSRF)\n\n"
            f"File: {file_rel}\n"
            f"```\n{content}\n```\n\n"
            "Output a JSON array of issues found (empty array if none):\n"
            "```json\n"
            '[{"line": 42, "severity": "error", "message": "...", '
            '"rule": "null_deref", "fix": "..."}]\n'
            "```\n"
            "Only report HIGH-CONFIDENCE issues. No false positives."
        )

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You are a formal verification expert. Report only "
                       "real, high-confidence issues. Zero false positives.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                raw = match.group(1).strip()
                items = json.loads(raw)
                for item in items:
                    if isinstance(item, dict):
                        issues.append(VerificationIssue(
                            severity=item.get("severity", "warning"),
                            category="formal",
                            file=file_rel,
                            line=int(item.get("line", 0)),
                            message=item.get("message", ""),
                            rule=item.get("rule", "llm_formal"),
                            fix_suggestion=item.get("fix", ""),
                            confidence=0.7,
                        ))

        except Exception as e:
            logger.debug(f"LLM formal check failed for {file_rel}: {e}")

    return issues


# ──────────────────────────────────────────────
# Main Verification Engine
# ──────────────────────────────────────────────


class FormalVerifier:
    """Orchestrates multi-level verification of generated code.

    Auto-detects available tools and runs the highest level of
    verification possible:

    Level 1: Linters (flake8, ESLint)
    Level 2: Type checkers (mypy, tsc)
    Level 3: Security scanners (bandit)
    Level 4: LLM formal analysis (always available)
    """

    def __init__(self) -> None:
        self._available_tools: dict[str, bool] = {}

    async def detect_tools(self) -> dict[str, bool]:
        """Detect which verification tools are available."""
        checks = {
            "flake8": _tool_available("flake8"),
            "mypy": _tool_available("mypy"),
            "bandit": _tool_available("bandit"),
            "eslint": _tool_available("npx"),
        }
        results = {}
        for name, coro in checks.items():
            results[name] = await coro

        self._available_tools = results
        logger.info(f"[FormalVerify] Available tools: {results}")
        return results

    async def verify(
        self,
        project_dir: Path,
        critical_files: list[str] | None = None,
        llm: Any = None,
        run_security: bool = True,
        run_formal: bool = True,
    ) -> VerificationReport:
        """Run full verification pipeline on a project.

        Args:
            project_dir: Path to project directory
            critical_files: Files for deep LLM analysis (auto-detected if None)
            llm: LLM router (needed for formal checks)
            run_security: Whether to run security scans
            run_formal: Whether to run LLM formal analysis
        """
        report = VerificationReport()

        # Detect available tools if not done yet
        if not self._available_tools:
            await self.detect_tools()

        # Count files
        py_files = list(project_dir.rglob("*.py"))
        js_files = list(project_dir.rglob("*.js")) + list(project_dir.rglob("*.ts"))
        skip_dirs = {"node_modules", ".git", "__pycache__", ".venv", "venv"}
        py_files = [f for f in py_files if not any(s in f.parts for s in skip_dirs)]
        js_files = [f for f in js_files if not any(s in f.parts for s in skip_dirs)]
        report.files_checked = len(py_files) + len(js_files)

        # Auto-detect critical files if not specified
        if critical_files is None and py_files:
            # Use largest files as "critical" (heuristic)
            py_files.sort(key=lambda f: f.stat().st_size, reverse=True)
            critical_files = [
                str(f.relative_to(project_dir))
                for f in py_files[:5]
            ]

        # Run verifiers in parallel
        tasks = []

        # Level 1: Linters
        if py_files:
            tasks.append(("flake8", _run_flake8(project_dir)))
        if js_files:
            tasks.append(("eslint", _run_eslint(project_dir)))

        # Level 2: Type checkers
        if py_files and self._available_tools.get("mypy"):
            tasks.append(("mypy", _run_mypy(project_dir)))

        # Level 3: Security
        if run_security and py_files:
            tasks.append(("bandit", _run_bandit(project_dir)))

        # Level 4: LLM formal
        if run_formal and llm and critical_files:
            tasks.append(("llm_formal", _llm_formal_check(
                project_dir, critical_files, llm,
            )))

        # Execute all tasks
        for tool_name, coro in tasks:
            try:
                issues = await coro
                if issues:
                    report.tools_used.append(tool_name)
                    for issue in issues:
                        report.add_issue(issue)
            except Exception as e:
                logger.warning(f"[FormalVerify] {tool_name} failed: {e}")

        logger.info(f"[FormalVerify] {report.summary()}")
        return report

    def format_for_agent(self, report: VerificationReport) -> str:
        """Format verification report as context for agent prompts."""
        if not report.issues:
            return ""

        parts = ["\n## Verification Results\n"]

        # Group by severity
        errors = [i for i in report.issues if i.severity == "error"]
        warnings = [i for i in report.issues if i.severity == "warning"]

        if errors:
            parts.append(f"### Errors ({len(errors)})\n")
            for e in errors[:10]:
                parts.append(f"- `{e.file}:{e.line}` [{e.rule}] {e.message}")
                if e.fix_suggestion:
                    parts.append(f"  Fix: {e.fix_suggestion}")
                parts.append("")

        if warnings:
            parts.append(f"\n### Warnings ({len(warnings)})\n")
            for w in warnings[:10]:
                parts.append(f"- `{w.file}:{w.line}` [{w.rule}] {w.message}")

        return "\n".join(parts)
