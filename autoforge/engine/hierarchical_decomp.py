"""Hierarchical Task Decomposition — Parsel-style compositional code generation.

Inspired by:
  - Parsel (NeurIPS 2023, Zelikman et al.): 75% higher pass rate than direct
    generation on competition-level problems via decomposition → implementation
    → compositional testing
  - CodePlan (ACM 2024): repository-level coding via dependency-aware planning
  - MapCoder (ACL 2024): multi-agent code gen with recall → plan → code → debug

Key insight: instead of asking an LLM to generate an entire module at once,
decompose it into a dependency graph of sub-functions, implement bottom-up
with per-function tests, then compose into the final module.

Pipeline:
  1. DECOMPOSE: LLM breaks a task into natural-language function specs
  2. DEPENDENCY SORT: topological sort by function dependencies
  3. IMPLEMENT: bottom-up implementation (leaf functions first)
  4. COMPOSE: assemble functions into the final module
  5. VALIDATE: run the composed module through tests

This is wired into _build_single_task for tasks estimated as COMPLEX/EXTREME
by the AdaptiveComputeRouter.
"""

from __future__ import annotations

import heapq
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from autoforge.engine.llm_router import TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class FunctionSpec:
    """Natural language spec for a single function."""

    name: str
    description: str
    inputs: list[str]  # Parameter descriptions
    output: str  # Return value description
    dependencies: list[str]  # Names of other functions this depends on
    test_cases: list[dict[str, str]] = field(default_factory=list)  # input→output examples
    implementation: str = ""  # Filled in during IMPLEMENT phase
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "output": self.output,
            "dependencies": self.dependencies,
            "test_cases": self.test_cases,
            "implementation": self.implementation,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FunctionSpec:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DecompositionPlan:
    """Complete decomposition of a task into function specs."""

    task_description: str
    module_name: str
    functions: list[FunctionSpec]
    execution_order: list[str]  # Topologically sorted function names
    shared_imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_description": self.task_description,
            "module_name": self.module_name,
            "functions": [f.to_dict() for f in self.functions],
            "execution_order": self.execution_order,
            "shared_imports": self.shared_imports,
        }


class HierarchicalDecomposer:
    """Decomposes complex tasks into function-level dependency graphs.

    Implements the Parsel / CodePlan approach:
    decompose → sort → implement → compose → validate.
    """

    # Minimum complexity for decomposition to be worth it
    MIN_FUNCTIONS = 3
    MAX_FUNCTIONS = 15

    def __init__(self) -> None:
        pass

    # ── Core API ─────────────────────────────────

    async def decompose(
        self,
        task_description: str,
        module_name: str,
        spec: dict[str, Any],
        llm: Any,
    ) -> DecompositionPlan | None:
        """Decompose a task into a function-level dependency graph.

        Returns None if the task is too simple to benefit from decomposition.
        """
        prompt = f"""\
You are a software architect. Decompose this coding task into individual functions.

## Task
{task_description[:1000]}

## Module name: {module_name}

## Project context
{json.dumps(spec.get("tech_stack", {}), indent=2)[:500]}

## Instructions
Break this task into 3-12 individual functions. For each function, specify:
- name: snake_case function name
- description: what the function does (1-2 sentences)
- inputs: list of parameter descriptions
- output: what it returns
- dependencies: list of OTHER function names from this decomposition that
  this function calls/depends on
- test_cases: 1-2 input→output examples (optional but helpful)

Also list any shared imports needed.

Reply with JSON:
{{
  "imports": ["import x", "from y import z"],
  "functions": [
    {{
      "name": "func_name",
      "description": "what it does",
      "inputs": ["param1: type - description"],
      "output": "return type and meaning",
      "dependencies": ["other_func"],
      "test_cases": [{{"input": "...", "expected": "..."}}]
    }}
  ]
}}"""

        try:
            response = await llm.call(
                system="You are a precise software architect. Decompose tasks into testable functions.",
                messages=[{"role": "user", "content": prompt}],
                complexity=TaskComplexity.HIGH,
            )

            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            if "{" not in text:
                return None

            json_str = text[text.index("{"):text.rindex("}") + 1]
            data = json.loads(json_str)

            functions = [
                FunctionSpec(
                    name=f["name"],
                    description=f.get("description", ""),
                    inputs=f.get("inputs", []),
                    output=f.get("output", ""),
                    dependencies=f.get("dependencies", []),
                    test_cases=f.get("test_cases", []),
                )
                for f in data.get("functions", [])
            ]

            if len(functions) < self.MIN_FUNCTIONS:
                logger.info(
                    f"[Decomp] Only {len(functions)} functions — "
                    f"too simple for decomposition"
                )
                return None

            # Truncate if too many
            functions = functions[:self.MAX_FUNCTIONS]

            # Topological sort
            execution_order = self._topological_sort(functions)

            plan = DecompositionPlan(
                task_description=task_description[:500],
                module_name=module_name,
                functions=functions,
                execution_order=execution_order,
                shared_imports=data.get("imports", []),
            )

            logger.info(
                f"[Decomp] Decomposed into {len(functions)} functions, "
                f"order: {execution_order}"
            )
            return plan

        except Exception as e:
            logger.warning(f"[Decomp] Decomposition failed: {e}")
            return None

    async def implement_bottom_up(
        self,
        plan: DecompositionPlan,
        llm: Any,
        existing_code: str = "",
    ) -> str:
        """Implement functions bottom-up (leaves first).

        Each function is implemented with its dependencies already available
        as context, following the topological order.
        """
        implemented: dict[str, str] = {}
        func_map = {f.name: f for f in plan.functions}

        for func_name in plan.execution_order:
            func = func_map.get(func_name)
            if not func:
                continue

            # Build context from already-implemented dependencies
            dep_context = ""
            for dep_name in func.dependencies:
                if dep_name in implemented:
                    dep_context += f"\n# Already implemented:\n{implemented[dep_name]}\n"

            # Build test context
            test_context = ""
            if func.test_cases:
                test_context = "\n## Test cases:\n"
                for tc in func.test_cases[:3]:
                    test_context += f"- Input: {tc.get('input', '?')} → Expected: {tc.get('expected', '?')}\n"

            prompt = f"""\
Implement this Python function. Use the already-implemented dependencies below.

## Function spec
Name: {func.name}
Description: {func.description}
Inputs: {', '.join(func.inputs)}
Output: {func.output}
{test_context}

## Available dependencies
{dep_context or "(no dependencies — this is a leaf function)"}

## Module imports available
{chr(10).join(plan.shared_imports) or "(standard library only)"}

Reply with ONLY the function implementation (def statement + body).
No explanations, no markdown, no tests — just the function code."""

            try:
                response = await llm.call(
                    system="You are an expert Python developer. Write clean, correct code.",
                    messages=[{"role": "user", "content": prompt}],
                    complexity=TaskComplexity.STANDARD,
                )

                code = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        code += block.text

                # Clean up: extract just the function definition
                code = self._extract_function_code(code, func.name)
                func.implementation = code
                implemented[func.name] = code

                logger.debug(f"[Decomp] Implemented {func.name} ({len(code)} chars)")

            except Exception as e:
                logger.warning(f"[Decomp] Failed to implement {func.name}: {e}")
                # Create a stub
                params = ", ".join(
                    p.split(":")[0].strip() if ":" in p else p
                    for p in func.inputs
                )
                stub = f"def {func.name}({params}):\n    raise NotImplementedError('{func.name}')\n"
                func.implementation = stub
                implemented[func.name] = stub

        return self._compose_module(plan, implemented)

    def compose_module(self, plan: DecompositionPlan) -> str:
        """Compose all implemented functions into a single module."""
        implemented = {f.name: f.implementation for f in plan.functions if f.implementation}
        return self._compose_module(plan, implemented)

    def build_context_for_agent(self, plan: DecompositionPlan) -> str:
        """Build a decomposition plan as context for a Builder agent.

        Instead of implementing bottom-up ourselves, we can give the plan
        to the builder agent as structured guidance.
        """
        parts = [
            f"## Hierarchical Decomposition Plan for {plan.module_name}",
            f"\nThis module should be built from {len(plan.functions)} composable functions.",
            f"Implementation order (bottom-up): {' → '.join(plan.execution_order)}",
            "\n### Function Specifications:\n",
        ]

        func_map = {f.name: f for f in plan.functions}
        for name in plan.execution_order:
            f = func_map.get(name)
            if not f:
                continue
            parts.append(f"**{f.name}**")
            parts.append(f"  - Description: {f.description}")
            parts.append(f"  - Inputs: {', '.join(f.inputs)}")
            parts.append(f"  - Output: {f.output}")
            if f.dependencies:
                parts.append(f"  - Depends on: {', '.join(f.dependencies)}")
            if f.test_cases:
                for tc in f.test_cases[:2]:
                    parts.append(f"  - Test: {tc.get('input', '')} → {tc.get('expected', '')}")
            parts.append("")

        parts.append(
            "\nIMPORTANT: Implement functions in the order listed above "
            "(dependencies first). Each function should be independently testable."
        )

        return "\n".join(parts)

    # ── Topological sort ─────────────────────────

    @staticmethod
    def _topological_sort(functions: list[FunctionSpec]) -> list[str]:
        """Topological sort of functions by dependencies (Kahn's algorithm).

        Functions with no dependencies come first (leaf nodes).
        """
        func_names = {f.name for f in functions}
        # Build adjacency: dep → dependent
        in_degree: dict[str, int] = {f.name: 0 for f in functions}
        dependents: dict[str, list[str]] = {f.name: [] for f in functions}

        for f in functions:
            for dep in f.dependencies:
                if dep in func_names:
                    in_degree[f.name] += 1
                    dependents[dep].append(f.name)

        # Start with leaf nodes (no dependencies), using a min-heap for
        # efficient sorted extraction (O(n log n) instead of O(n^2))
        heap = [name for name, deg in in_degree.items() if deg == 0]
        heapq.heapify(heap)
        order: list[str] = []

        while heap:
            # Pop alphabetically smallest for determinism
            current = heapq.heappop(heap)
            order.append(current)

            for dependent in dependents.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    heapq.heappush(heap, dependent)

        # Handle cycles: add remaining nodes
        remaining = [f.name for f in functions if f.name not in order]
        order.extend(remaining)

        return order

    # ── Module composition ───────────────────────

    def _compose_module(
        self,
        plan: DecompositionPlan,
        implementations: dict[str, str],
    ) -> str:
        """Compose individual function implementations into a module."""
        parts: list[str] = []

        # Module docstring
        parts.append(f'"""{plan.module_name} — auto-generated via hierarchical decomposition.')
        parts.append(f"\n{plan.task_description[:200]}")
        parts.append('"""\n')

        # Imports
        if plan.shared_imports:
            parts.extend(plan.shared_imports)
            parts.append("")

        # Functions in topological order
        for func_name in plan.execution_order:
            impl = implementations.get(func_name, "")
            if impl:
                parts.append("")
                parts.append(impl)
                parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _extract_function_code(raw: str, func_name: str) -> str:
        """Extract function code from LLM response, cleaning markdown etc."""
        # Remove markdown code fences
        code = raw.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        # Find the def statement
        lines = code.split("\n")
        start = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(f"def {func_name}(") or stripped.startswith(f"async def {func_name}("):
                start = i
                break

        if start is not None:
            # Extract from def to end of function
            func_lines = [lines[start]]
            base_indent = len(lines[start]) - len(lines[start].lstrip())

            for i in range(start + 1, len(lines)):
                line = lines[i]
                if line.strip() == "":
                    func_lines.append(line)
                    continue
                indent = len(line) - len(line.lstrip())
                if indent <= base_indent and line.strip():
                    break
                func_lines.append(line)

            return "\n".join(func_lines).rstrip() + "\n"

        # Fallback: return cleaned code as-is
        return code.strip() + "\n"
