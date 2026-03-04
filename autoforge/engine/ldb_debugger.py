"""LDB — Block-Level Debugging via Runtime Execution Tracing.

Inspired by LDB (ACL 2024, Zhong et al.):
  "Debug like a Human: A Large Language Model Debugger via
   Verifying Runtime Execution Step-by-step"

Key result: +9.8% on HumanEval, 98.2% accuracy with GPT-4o.

Approach: when tests fail, instead of giving the LLM the entire error
and hoping for the best, we:
  1. Decompose the failing code into basic blocks (control-flow segments)
  2. Trace variable values at each block boundary (or simulate them)
  3. Have the LLM verify each block's correctness independently
  4. Pinpoint the exact block where values diverge from expectations
  5. Generate a targeted fix for JUST that block

This is far more effective than "here's the error, fix it" because
it localises the fault to a specific code region.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CodeBlock:
    """A basic block of code with metadata."""

    index: int
    lines: list[str]
    start_line: int
    end_line: int
    block_type: str  # "function_def" | "conditional" | "loop" | "assignment" | "return" | "other"
    variables_in: list[str] = field(default_factory=list)  # Variables read
    variables_out: list[str] = field(default_factory=list)  # Variables written

    @property
    def code(self) -> str:
        return "\n".join(self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "code": self.code,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "block_type": self.block_type,
            "variables_in": self.variables_in,
            "variables_out": self.variables_out,
        }


@dataclass
class BlockVerification:
    """LLM verification of a single block."""

    block_index: int
    is_correct: bool
    expected_state: str  # What variables should be after this block
    actual_state: str  # What they actually are (from trace or simulation)
    explanation: str
    confidence: float = 0.0
    fix_suggestion: str = ""


@dataclass
class DebugReport:
    """Full debugging report for a failing function."""

    file_path: str
    function_name: str
    error_message: str
    blocks: list[CodeBlock]
    verifications: list[BlockVerification]
    faulty_block: int | None = None
    root_cause: str = ""
    suggested_fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "function_name": self.function_name,
            "error_message": self.error_message,
            "blocks": [b.to_dict() for b in self.blocks],
            "faulty_block": self.faulty_block,
            "root_cause": self.root_cause,
            "suggested_fix": self.suggested_fix,
        }


class LDBDebugger:
    """Block-level debugger using LLM runtime simulation.

    Decomposes code into blocks, traces execution mentally (or via sandbox),
    and pinpoints the exact location of bugs.
    """

    # Python control flow keywords that start new blocks
    BLOCK_STARTERS = {"if", "elif", "else", "for", "while", "try", "except",
                      "finally", "with", "def", "class", "return", "raise", "yield"}

    def __init__(self) -> None:
        pass

    # ── Core API ─────────────────────────────────

    async def debug_failure(
        self,
        file_path: Path,
        function_name: str,
        error_message: str,
        test_input: str,
        expected_output: str,
        llm: Any,
        sandbox: Any | None = None,
    ) -> DebugReport:
        """Full LDB debugging pipeline.

        1. Read and decompose the function into basic blocks
        2. Simulate (or trace) execution through blocks
        3. Have LLM verify each block
        4. Identify the faulty block
        5. Generate targeted fix
        """
        # Read the source code
        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return DebugReport(
                file_path=str(file_path),
                function_name=function_name,
                error_message=f"Could not read file: {e}",
                blocks=[], verifications=[],
            )

        # Extract the function
        func_source = self._extract_function(source, function_name)
        if not func_source:
            return DebugReport(
                file_path=str(file_path),
                function_name=function_name,
                error_message=f"Function '{function_name}' not found in {file_path}",
                blocks=[], verifications=[],
            )

        # Step 1: Decompose into blocks
        blocks = self._decompose_into_blocks(func_source)

        # Step 2: Try real tracing via sandbox, fall back to LLM simulation
        trace_data = None
        if sandbox:
            trace_data = await self._trace_execution(
                sandbox, file_path, function_name, test_input,
            )

        # Step 3: LLM verifies each block
        verifications = await self._verify_blocks(
            blocks, function_name, test_input, expected_output,
            error_message, trace_data, llm,
        )

        # Step 4: Identify faulty block
        faulty_idx = self._identify_faulty_block(verifications)

        # Step 5: Generate targeted fix
        root_cause = ""
        suggested_fix = ""
        if faulty_idx is not None:
            root_cause, suggested_fix = await self._generate_fix(
                blocks, verifications, faulty_idx, function_name,
                error_message, llm,
            )

        return DebugReport(
            file_path=str(file_path),
            function_name=function_name,
            error_message=error_message,
            blocks=blocks,
            verifications=verifications,
            faulty_block=faulty_idx,
            root_cause=root_cause,
            suggested_fix=suggested_fix,
        )

    async def debug_test_failure(
        self,
        project_dir: Path,
        failure_info: dict[str, Any],
        llm: Any,
        sandbox: Any | None = None,
    ) -> DebugReport | None:
        """Debug a test failure from test results data.

        Parses standard test failure info and delegates to debug_failure.
        """
        error_msg = failure_info.get("error", failure_info.get("message", ""))
        test_name = failure_info.get("test", failure_info.get("name", ""))

        # Try to locate the failing file and function from the error
        file_path, func_name = self._parse_failure_location(
            error_msg, project_dir,
        )

        if not file_path or not func_name:
            return None

        return await self.debug_failure(
            file_path=file_path,
            function_name=func_name,
            error_message=error_msg[:1000],
            test_input=test_name,
            expected_output="(from test assertion)",
            llm=llm,
            sandbox=sandbox,
        )

    def format_for_agent(self, report: DebugReport) -> str:
        """Format a debug report as context for an agent prompt."""
        if not report.faulty_block and not report.root_cause:
            return ""

        parts = [f"## LDB Debug Analysis: {report.function_name}"]

        if report.faulty_block is not None and report.blocks:
            block = report.blocks[report.faulty_block]
            parts.append(f"\n**Faulty block** (lines {block.start_line}-{block.end_line}):")
            parts.append(f"```\n{block.code}\n```")

        if report.root_cause:
            parts.append(f"\n**Root cause**: {report.root_cause}")

        if report.suggested_fix:
            parts.append(f"\n**Suggested fix**:\n```\n{report.suggested_fix}\n```")

        return "\n".join(parts)

    # ── Block decomposition ──────────────────────

    def _decompose_into_blocks(self, func_source: str) -> list[CodeBlock]:
        """Decompose a function into basic blocks based on control flow."""
        lines = func_source.split("\n")
        blocks: list[CodeBlock] = []
        current_block_lines: list[str] = []
        current_start = 0
        block_idx = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                if current_block_lines:
                    current_block_lines.append(line)
                continue

            # Check if this line starts a new block
            first_word = stripped.split("(")[0].split(":")[0].split(" ")[0]
            is_block_start = first_word in self.BLOCK_STARTERS

            if is_block_start and current_block_lines:
                # Save current block
                block_type = self._classify_block(current_block_lines)
                blocks.append(CodeBlock(
                    index=block_idx,
                    lines=current_block_lines,
                    start_line=current_start,
                    end_line=i - 1,
                    block_type=block_type,
                    variables_in=self._extract_vars_read(current_block_lines),
                    variables_out=self._extract_vars_written(current_block_lines),
                ))
                block_idx += 1
                current_block_lines = [line]
                current_start = i
            else:
                if not current_block_lines:
                    current_start = i
                current_block_lines.append(line)

        # Don't forget the last block
        if current_block_lines:
            block_type = self._classify_block(current_block_lines)
            blocks.append(CodeBlock(
                index=block_idx,
                lines=current_block_lines,
                start_line=current_start,
                end_line=len(lines) - 1,
                block_type=block_type,
                variables_in=self._extract_vars_read(current_block_lines),
                variables_out=self._extract_vars_written(current_block_lines),
            ))

        return blocks

    @staticmethod
    def _classify_block(lines: list[str]) -> str:
        """Classify a block by its primary control flow type."""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def "):
                return "function_def"
            if stripped.startswith(("if ", "elif ", "else:")):
                return "conditional"
            if stripped.startswith(("for ", "while ")):
                return "loop"
            if stripped.startswith("return "):
                return "return"
            if "=" in stripped and not stripped.startswith(("if", "while", "for", "==")):
                return "assignment"
        return "other"

    @staticmethod
    def _extract_vars_read(lines: list[str]) -> list[str]:
        """Extract variable names that are read (rough heuristic)."""
        code = "\n".join(lines)
        # Match identifiers that aren't being assigned to
        identifiers = set(re.findall(r'\b([a-zA-Z_]\w*)\b', code))
        # Remove Python keywords and builtins
        keywords = {"def", "class", "if", "else", "elif", "for", "while", "return",
                    "import", "from", "try", "except", "finally", "with", "as",
                    "True", "False", "None", "and", "or", "not", "in", "is",
                    "print", "len", "range", "str", "int", "float", "list", "dict",
                    "set", "tuple", "type", "self", "cls", "super"}
        return sorted(identifiers - keywords)[:10]

    @staticmethod
    def _extract_vars_written(lines: list[str]) -> list[str]:
        """Extract variable names that are written to."""
        written: list[str] = []
        for line in lines:
            stripped = line.strip()
            # Simple assignment detection
            match = re.match(r'(\w+)\s*[+\-*\/]?=\s', stripped)
            if match and match.group(1) not in ("if", "while", "for", "return"):
                written.append(match.group(1))
        return written[:10]

    # ── Function extraction ──────────────────────

    @staticmethod
    def _extract_function(source: str, func_name: str) -> str | None:
        """Extract a function's source code from a file."""
        lines = source.split("\n")
        func_start = None
        func_indent = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(f"def {func_name}(") or stripped.startswith(f"def {func_name} ("):
                func_start = i
                func_indent = len(line) - len(line.lstrip())
                break
            # Also match async def
            if stripped.startswith(f"async def {func_name}("):
                func_start = i
                func_indent = len(line) - len(line.lstrip())
                break

        if func_start is None:
            return None

        # Collect all lines that are part of this function
        func_lines = [lines[func_start]]
        for i in range(func_start + 1, len(lines)):
            line = lines[i]
            if line.strip() == "":
                func_lines.append(line)
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= func_indent and line.strip():
                # Back to same or lower indent = end of function
                break
            func_lines.append(line)

        return "\n".join(func_lines)

    # ── Execution tracing ────────────────────────

    async def _trace_execution(
        self,
        sandbox: Any,
        file_path: Path,
        function_name: str,
        test_input: str,
    ) -> dict[str, Any] | None:
        """Try to trace execution via sandbox (optional, best-effort)."""
        # Generate a tracing script
        trace_script = f"""\
import sys, json, traceback
sys.path.insert(0, '{file_path.parent}')

# Simple variable tracer
_trace_log = []
_original_settrace = sys.settrace

def _tracer(frame, event, arg):
    if event == 'line' and frame.f_code.co_name == '{function_name}':
        _trace_log.append({{
            'line': frame.f_lineno,
            'locals': {{k: repr(v)[:100] for k, v in frame.f_locals.items()
                       if not k.startswith('_')}},
        }})
    return _tracer

sys.settrace(_tracer)
try:
    from {file_path.stem} import {function_name}
    result = {function_name}({test_input})
    print(json.dumps({{"trace": _trace_log[:50], "result": repr(result)}}))
except Exception as e:
    print(json.dumps({{"trace": _trace_log[:50], "error": str(e)}}))
finally:
    sys.settrace(None)
"""
        try:
            result = await sandbox.run(f"python3 -c {repr(trace_script)}", timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except Exception as e:
            logger.debug(f"[LDB] Tracing failed: {e}")
        return None

    # ── Block verification ───────────────────────

    async def _verify_blocks(
        self,
        blocks: list[CodeBlock],
        function_name: str,
        test_input: str,
        expected_output: str,
        error_message: str,
        trace_data: dict[str, Any] | None,
        llm: Any,
    ) -> list[BlockVerification]:
        """Have LLM verify each block's correctness."""
        if not blocks:
            return []

        blocks_text = "\n\n".join(
            f"### Block {b.index} (lines {b.start_line}-{b.end_line}, type={b.block_type}):\n"
            f"```python\n{b.code}\n```\n"
            f"Variables read: {', '.join(b.variables_in[:5]) or 'none'}\n"
            f"Variables written: {', '.join(b.variables_out[:5]) or 'none'}"
            for b in blocks
        )

        trace_info = ""
        if trace_data:
            trace_entries = trace_data.get("trace", [])[:20]
            trace_info = "\n## Runtime Trace (actual variable values):\n"
            for entry in trace_entries:
                trace_info += f"Line {entry.get('line', '?')}: {json.dumps(entry.get('locals', {}))}\n"

        prompt = f"""\
You are a debugger. Analyze this function block-by-block to find the bug.

## Function: {function_name}
## Test input: {test_input}
## Expected output: {expected_output}
## Actual error: {error_message[:500]}
{trace_info}

## Code Blocks:
{blocks_text}

For each block, determine if it is CORRECT or INCORRECT given the inputs.
Focus on: variable values, logic errors, off-by-one errors, missing edge cases.

Reply with a JSON array:
[
  {{"block": 0, "correct": true/false, "expected_state": "...", "actual_state": "...", "explanation": "...", "confidence": 0.X, "fix": "...if incorrect..."}},
  ...
]"""

        try:
            response = await llm.query(
                system="You are a precise code debugger. Verify each block step-by-step.",
                messages=[{"role": "user", "content": prompt}],
                complexity="complex",
            )

            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            # Parse JSON array from response
            if "[" in text:
                json_str = text[text.index("["):text.rindex("]") + 1]
                data = json.loads(json_str)

                return [
                    BlockVerification(
                        block_index=item.get("block", i),
                        is_correct=item.get("correct", True),
                        expected_state=item.get("expected_state", ""),
                        actual_state=item.get("actual_state", ""),
                        explanation=item.get("explanation", ""),
                        confidence=float(item.get("confidence", 0.5)),
                        fix_suggestion=item.get("fix", ""),
                    )
                    for i, item in enumerate(data)
                ]

        except Exception as e:
            logger.warning(f"[LDB] Block verification failed: {e}")

        return []

    @staticmethod
    def _identify_faulty_block(verifications: list[BlockVerification]) -> int | None:
        """Find the first incorrect block with highest confidence."""
        incorrect = [v for v in verifications if not v.is_correct]
        if not incorrect:
            return None

        # Return the first incorrect block (execution order matters)
        incorrect.sort(key=lambda v: (-v.confidence, v.block_index))
        return incorrect[0].block_index

    async def _generate_fix(
        self,
        blocks: list[CodeBlock],
        verifications: list[BlockVerification],
        faulty_idx: int,
        function_name: str,
        error_message: str,
        llm: Any,
    ) -> tuple[str, str]:
        """Generate a targeted fix for the faulty block."""
        faulty_block = blocks[faulty_idx]
        verification = next(
            (v for v in verifications if v.block_index == faulty_idx), None,
        )

        context_blocks = ""
        for b in blocks:
            marker = " ← BUG HERE" if b.index == faulty_idx else ""
            context_blocks += f"# Block {b.index}{marker}\n{b.code}\n\n"

        diagnosis = verification.explanation if verification else error_message

        prompt = f"""\
Fix the bug in block {faulty_idx} of function `{function_name}`.

## Diagnosis
{diagnosis}

## Full function with blocks:
```python
{context_blocks}
```

## Error: {error_message[:300]}

Reply with JSON:
{{"root_cause": "one sentence explanation", "fixed_block": "the corrected code for block {faulty_idx} only"}}"""

        try:
            response = await llm.query(
                system="You are a code repair expert. Fix only the faulty block.",
                messages=[{"role": "user", "content": prompt}],
                complexity="standard",
            )

            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            if "{" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]
                data = json.loads(json_str)
                return (
                    data.get("root_cause", ""),
                    data.get("fixed_block", ""),
                )
        except Exception as e:
            logger.warning(f"[LDB] Fix generation failed: {e}")

        return ("", "")

    # ── Failure location parsing ─────────────────

    @staticmethod
    def _parse_failure_location(
        error_msg: str, project_dir: Path,
    ) -> tuple[Path | None, str | None]:
        """Parse a Python traceback to find the failing file and function."""
        # Look for patterns like: File "path/to/file.py", line N, in func_name
        pattern = r'File "([^"]+)", line \d+, in (\w+)'
        matches = re.findall(pattern, error_msg)

        if not matches:
            return None, None

        # Use the last match (deepest in the call stack, likely the real fault)
        file_str, func_name = matches[-1]
        file_path = Path(file_str)

        # Try to resolve relative to project dir
        if not file_path.is_absolute():
            file_path = project_dir / file_path

        if file_path.exists():
            return file_path, func_name

        # Try searching in project dir
        for candidate in project_dir.rglob(file_path.name):
            return candidate, func_name

        return None, None
