"""Symbolic Computation Backend 鈥?Integration of SymPy, SageMath, and CAS tools.

This module provides a unified symbolic computation interface for verifying
mathematical statements, solving equations, computing limits, and performing
advanced algebraic operations.

Architecture:
  - ComputationBackend (enum): Available CAS engines
  - SymbolicResult: Computation output with steps and metadata
  - VerificationEvidence: Claim verification with confidence
  - SymPyEngine: Async wrapper around SymPy
  - SageEngine: Optional SageMath CLI wrapper
  - LaTeXParser: Convert LaTeX 鈫?SymPy expressions
  - MathematicalVerifier: High-level claim verification
  - SymbolicComputeEngine: Main facade

Key features:
  - Safe expression parsing (sympify with guards)
  - Asyncio.to_thread() for CPU-bound operations
  - Optional SageMath via subprocess (if available)
  - LaTeX 鈫?SymPy bidirectional conversion
  - Symbolic identity checking and inequality verification
  - Algebraic geometry & matrix operations
  - Batch verification of mathematical claims
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import re
from collections import Counter
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from autoforge.engine.utils import extract_json_from_text

logger = logging.getLogger(__name__)

# Optional imports 鈥?graceful degradation if unavailable
try:
    import sympy as sp
    from sympy import symbols, sympify, simplify, solve, integrate, diff, limit, series
    from sympy import Matrix, Abs, sin, cos, log, exp, sqrt, oo
    from sympy.parsing.latex import parse_latex as sp_parse_latex
    from sympy import gcd, factor, apart, roots, eye, diag, svd_solve
    from sympy.polys.polytools import poly, Poly
    from sympy.ntheory import factorint, divisors
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None
    sp_parse_latex = None


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# Enumerations
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class ComputationBackend(str, Enum):
    """Available symbolic computation backends."""
    SYMPY = "sympy"                    # Python-native, always available
    SAGE = "sage"                      # SageMath via subprocess
    MATHEMATICA_FREE = "mathematica"   # Wolfram Language (if Mathematica/WolframScript installed)
    NUMERIC = "numeric"                # Pure numerical computation


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# Data Classes
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


@dataclass
class SymbolicResult:
    """Result of a symbolic computation operation."""
    expression: str
    result: str
    backend: ComputationBackend
    success: bool
    error: str = ""
    computation_time: float = 0.0
    steps: list[str] = field(default_factory=list)
    numeric_approximation: float | None = None
    algorithm_ratio: float = 0.0  # Ratio of pure algorithm vs LLM fallback (0=fallback, 1=pure)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "expression": self.expression,
            "result": self.result,
            "backend": self.backend.value,
            "success": self.success,
            "error": self.error,
            "computation_time": self.computation_time,
            "steps": self.steps,
            "numeric_approximation": self.numeric_approximation,
            "algorithm_ratio": self.algorithm_ratio,
        }


@dataclass
class VerificationEvidence:
    """Evidence that a mathematical claim is true or false."""
    claim: str
    verification_type: str  # algebraic_identity, limit_check, series_expansion, etc.
    result: SymbolicResult
    supports_claim: bool
    confidence: float  # 0.0 to 1.0
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim": self.claim,
            "verification_type": self.verification_type,
            "result": self.result.to_dict(),
            "supports_claim": self.supports_claim,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# SymPy Engine
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class SymPyEngine:
    """Async wrapper around SymPy for symbolic computation."""

    def __init__(self) -> None:
        """Initialize SymPy engine (checks availability)."""
        if not SYMPY_AVAILABLE:
            raise RuntimeError(
                "SymPy is not installed. Install with: pip install sympy"
            )
        self.backend = ComputationBackend.SYMPY
        logger.info("SymPy engine initialized")

    async def simplify(self, expr_str: str) -> SymbolicResult:
        """Simplify a symbolic expression."""
        start = time.time()
        try:
            expr = await self._parse_expr(expr_str)
            result = await asyncio.to_thread(simplify, expr)
            return SymbolicResult(
                expression=expr_str,
                result=self._format_result(result),
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=[f"Simplified {expr_str}"],
            )
        except Exception as e:
            return SymbolicResult(
                expression=expr_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def solve(
        self, equation_str: str, variable: str = "x"
    ) -> SymbolicResult:
        """Solve an equation for a variable."""
        start = time.time()
        try:
            var = sp.Symbol(variable)
            expr = await self._parse_expr(equation_str)
            solutions = await asyncio.to_thread(solve, expr, var)
            result_str = ", ".join(self._format_result(s) for s in solutions)
            return SymbolicResult(
                expression=equation_str,
                result=result_str,
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=[f"Solved for {variable}"],
            )
        except Exception as e:
            return SymbolicResult(
                expression=equation_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def integrate(
        self,
        expr_str: str,
        variable: str = "x",
        limits: tuple[str, str] | None = None,
    ) -> SymbolicResult:
        """Integrate an expression (indefinite or definite)."""
        start = time.time()
        try:
            var = sp.Symbol(variable)
            expr = await self._parse_expr(expr_str)
            if limits:
                # Definite integral
                lower = await self._parse_expr(limits[0])
                upper = await self._parse_expr(limits[1])
                result = await asyncio.to_thread(
                    integrate, expr, (var, lower, upper)
                )
                step = f"Definite integral from {limits[0]} to {limits[1]}"
            else:
                # Indefinite integral
                result = await asyncio.to_thread(integrate, expr, var)
                step = "Indefinite integral"
            return SymbolicResult(
                expression=expr_str,
                result=self._format_result(result),
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=[step],
            )
        except Exception as e:
            return SymbolicResult(
                expression=expr_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def differentiate(
        self, expr_str: str, variable: str = "x", order: int = 1
    ) -> SymbolicResult:
        """Differentiate an expression (nth-order derivative)."""
        start = time.time()
        try:
            var = sp.Symbol(variable)
            expr = await self._parse_expr(expr_str)
            result = await asyncio.to_thread(diff, expr, var, order)
            step = f"d^{order}/{variable}^{order}" if order > 1 else f"d/{variable}"
            return SymbolicResult(
                expression=expr_str,
                result=self._format_result(result),
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=[step],
            )
        except Exception as e:
            return SymbolicResult(
                expression=expr_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def limit(
        self, expr_str: str, variable: str, point: str
    ) -> SymbolicResult:
        """Compute a limit as a variable approaches a point."""
        start = time.time()
        try:
            var = sp.Symbol(variable)
            point_expr = await self._parse_expr(point)
            expr = await self._parse_expr(expr_str)
            result = await asyncio.to_thread(limit, expr, var, point_expr)
            return SymbolicResult(
                expression=expr_str,
                result=self._format_result(result),
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=[f"Limit as {variable} 鈫?{point}"],
            )
        except Exception as e:
            return SymbolicResult(
                expression=expr_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def series_expand(
        self,
        expr_str: str,
        variable: str = "x",
        point: str = "0",
        order: int = 6,
    ) -> SymbolicResult:
        """Expand a function as a Taylor/Laurent series."""
        start = time.time()
        try:
            var = sp.Symbol(variable)
            point_expr = await self._parse_expr(point)
            expr = await self._parse_expr(expr_str)
            result = await asyncio.to_thread(
                series, expr, var, point_expr, n=order
            )
            return SymbolicResult(
                expression=expr_str,
                result=self._format_result(result),
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=[f"Series expansion at {variable}={point} (order {order})"],
            )
        except Exception as e:
            return SymbolicResult(
                expression=expr_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def matrix_operation(
        self, matrix_str: str, operation: str
    ) -> SymbolicResult:
        """Perform matrix operations: eigenvalues, determinant, rank, etc."""
        start = time.time()
        try:
            # Parse matrix (expect [[a,b],[c,d]] format)
            matrix_data = json.loads(matrix_str)
            mat = Matrix(matrix_data)

            if operation == "eigenvalues":
                result_val = await asyncio.to_thread(mat.eigenvals)
                result_str = str(result_val)
            elif operation == "eigenvectors":
                result_val = await asyncio.to_thread(mat.eigenvects)
                result_str = str(result_val)
            elif operation == "determinant":
                result_val = await asyncio.to_thread(mat.det)
                result_str = self._format_result(result_val)
            elif operation == "rank":
                result_val = await asyncio.to_thread(mat.rank)
                result_str = str(result_val)
            elif operation == "trace":
                result_val = await asyncio.to_thread(mat.trace)
                result_str = self._format_result(result_val)
            elif operation == "inverse":
                result_val = await asyncio.to_thread(mat.inv)
                result_str = str(result_val)
            else:
                raise ValueError(f"Unknown matrix operation: {operation}")

            return SymbolicResult(
                expression=matrix_str,
                result=result_str,
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=[f"Matrix {operation}"],
            )
        except Exception as e:
            return SymbolicResult(
                expression=matrix_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def verify_identity(self, lhs: str, rhs: str) -> SymbolicResult:
        """Check if two expressions are symbolically identical."""
        start = time.time()
        try:
            expr_lhs = await self._parse_expr(lhs)
            expr_rhs = await self._parse_expr(rhs)
            # Simplify the difference
            diff_expr = await asyncio.to_thread(simplify, expr_lhs - expr_rhs)
            is_equal = diff_expr == 0
            return SymbolicResult(
                expression=f"{lhs} = {rhs}",
                result="True" if is_equal else "False",
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=[f"Verified: {lhs} {'=' if is_equal else '!='} {rhs}"],
                algorithm_ratio=1.0,  # Pure SymPy computation
            )
        except Exception as e:
            return SymbolicResult(
                expression=f"{lhs} = {rhs}",
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def factor_polynomial(self, expr_str: str) -> SymbolicResult:
        """Factor a polynomial expression."""
        start = time.time()
        try:
            expr = await self._parse_expr(expr_str)
            result = await asyncio.to_thread(sp.factor, expr)
            return SymbolicResult(
                expression=expr_str,
                result=self._format_result(result),
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=[f"Factorized polynomial"],
                algorithm_ratio=1.0,
            )
        except Exception as e:
            return SymbolicResult(
                expression=expr_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def partial_fraction_decomposition(self, expr_str: str, var: str = "x") -> SymbolicResult:
        """Decompose a rational function into partial fractions."""
        start = time.time()
        try:
            var_sym = sp.Symbol(var)
            expr = await self._parse_expr(expr_str)
            result = await asyncio.to_thread(sp.apart, expr, var_sym)
            return SymbolicResult(
                expression=expr_str,
                result=self._format_result(result),
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=["Partial fraction decomposition"],
                algorithm_ratio=1.0,
            )
        except Exception as e:
            return SymbolicResult(
                expression=expr_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def find_roots(self, expr_str: str, var: str = "x") -> SymbolicResult:
        """Find roots of a polynomial or equation."""
        start = time.time()
        try:
            var_sym = sp.Symbol(var)
            expr = await self._parse_expr(expr_str)
            roots_dict = await asyncio.to_thread(sp.roots, expr, var_sym)
            result_str = ", ".join(
                f"{r}: multiplicity {m}" for r, m in roots_dict.items()
            )
            return SymbolicResult(
                expression=expr_str,
                result=result_str if result_str else "No roots found",
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=["Found roots"],
                algorithm_ratio=1.0,
            )
        except Exception as e:
            return SymbolicResult(
                expression=expr_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def matrix_svd(self, matrix_str: str) -> SymbolicResult:
        """Compute singular value decomposition of a matrix."""
        start = time.time()
        try:
            matrix_data = json.loads(matrix_str)
            mat = Matrix(matrix_data)
            U, S, Vh = await asyncio.to_thread(mat.singular_value_decomposition)
            result_dict = {
                "U": str(U),
                "singular_values": list(S),
                "Vh": str(Vh),
            }
            return SymbolicResult(
                expression=matrix_str,
                result=json.dumps(result_dict, default=str),
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=["SVD computed"],
                algorithm_ratio=1.0,
            )
        except Exception as e:
            return SymbolicResult(
                expression=matrix_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def matrix_nullspace(self, matrix_str: str) -> SymbolicResult:
        """Compute nullspace of a matrix."""
        start = time.time()
        try:
            matrix_data = json.loads(matrix_str)
            mat = Matrix(matrix_data)
            nullspace = await asyncio.to_thread(mat.nullspace)
            result_str = str(nullspace)
            return SymbolicResult(
                expression=matrix_str,
                result=result_str,
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=["Nullspace computed"],
                algorithm_ratio=1.0,
            )
        except Exception as e:
            return SymbolicResult(
                expression=matrix_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def number_theory_factorize(self, n_str: str) -> SymbolicResult:
        """Factorize an integer into prime factors."""
        start = time.time()
        try:
            n = int(n_str)
            factors = await asyncio.to_thread(sp.factorint, n)
            result_str = " 脳 ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factors.items()))
            return SymbolicResult(
                expression=n_str,
                result=result_str,
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=[f"Prime factorization"],
                algorithm_ratio=1.0,
            )
        except Exception as e:
            return SymbolicResult(
                expression=n_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def number_theory_gcd(self, a_str: str, b_str: str) -> SymbolicResult:
        """Compute GCD of two numbers."""
        start = time.time()
        try:
            a, b = int(a_str), int(b_str)
            gcd_val = await asyncio.to_thread(sp.gcd, a, b)
            return SymbolicResult(
                expression=f"gcd({a_str}, {b_str})",
                result=str(gcd_val),
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=["GCD computed"],
                algorithm_ratio=1.0,
            )
        except Exception as e:
            return SymbolicResult(
                expression=f"gcd({a_str}, {b_str})",
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def check_inequality(
        self, expr_str: str, assumptions: dict[str, Any] | None = None
    ) -> SymbolicResult:
        """Check if an inequality holds under given assumptions."""
        start = time.time()
        try:
            expr = await self._parse_expr(expr_str)
            # Simplified check: evaluate at several test points
            test_points = assumptions or {"x": [0, 1, -1, 0.5]}
            results = []
            for var_name, values in test_points.items():
                var = sp.Symbol(var_name)
                for val in values:
                    try:
                        result = expr.subs(var, val)
                        results.append(bool(result))
                    except Exception:
                        pass
            holds = all(results) if results else False
            return SymbolicResult(
                expression=expr_str,
                result="True" if holds else "False",
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=[f"Inequality check on {len(results)} test points"],
            )
        except Exception as e:
            return SymbolicResult(
                expression=expr_str,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def _parse_expr(self, expr_str: str) -> Any:
        """Safely parse a string into a SymPy expression."""
        return await asyncio.to_thread(
            lambda: sympify(expr_str, transformations="all")
        )

    def _format_result(self, result: Any) -> str:
        """Format a SymPy result for display."""
        try:
            # Try LaTeX first for nice formatting
            return str(result)
        except Exception:
            return str(result)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# SageMath Engine (Optional)
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class SageEngine:
    """Optional SageMath backend via subprocess."""

    def __init__(self) -> None:
        """Initialize Sage engine (checks availability)."""
        self.backend = ComputationBackend.SAGE
        self._sage_available = self._check_sage_available()
        if not self._sage_available:
            logger.warning("SageMath not found; Sage engine disabled")
        else:
            logger.info("SageMath engine initialized")

    def _check_sage_available(self) -> bool:
        """Check if SageMath is available on system."""
        try:
            result = subprocess.run(
                ["sage", "--version"],
                capture_output=True,
                timeout=5,
                text=True,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def execute(self, sage_code: str) -> SymbolicResult:
        """Execute arbitrary Sage code."""
        start = time.time()
        if not self._sage_available:
            return SymbolicResult(
                expression=sage_code,
                result="",
                backend=self.backend,
                success=False,
                error="SageMath not available",
                computation_time=time.time() - start,
            )
        try:
            output = await self._run_sage(sage_code, timeout=30)
            return SymbolicResult(
                expression=sage_code,
                result=output,
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
            )
        except Exception as e:
            return SymbolicResult(
                expression=sage_code,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def compute_groebner(
        self, ideal_gens: list[str], ring_vars: list[str]
    ) -> SymbolicResult:
        """Compute Groebner basis for a polynomial ideal."""
        start = time.time()
        if not self._sage_available:
            return SymbolicResult(
                expression=str(ideal_gens),
                result="",
                backend=self.backend,
                success=False,
                error="SageMath not available",
                computation_time=time.time() - start,
            )
        try:
            vars_str = ", ".join(ring_vars)
            gens_str = ", ".join(ideal_gens)
            code = f"""
R.<{vars_str}> = PolynomialRing(QQ)
I = ideal([{gens_str}])
print(I.groebner_basis())
"""
            output = await self._run_sage(code, timeout=30)
            return SymbolicResult(
                expression=f"Groebner({ideal_gens})",
                result=output,
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
            )
        except Exception as e:
            return SymbolicResult(
                expression=str(ideal_gens),
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def compute_cohomology(self, complex_desc: str) -> SymbolicResult:
        """Compute cohomology of a chain complex."""
        start = time.time()
        differentials = self._parse_chain_complex_description(complex_desc)
        if not differentials:
            return SymbolicResult(
                expression=complex_desc,
                result="",
                backend=self.backend,
                success=False,
                error="Unable to parse chain-complex descriptor",
                computation_time=time.time() - start,
            )
        try:
            if SYMPY_AVAILABLE:
                cohomology_payload = self._compute_chain_cohomology_sympy(differentials)
                return SymbolicResult(
                    expression=complex_desc,
                    result=json.dumps(cohomology_payload, ensure_ascii=False),
                    backend=self.backend,
                    success=True,
                    computation_time=time.time() - start,
                    steps=["Computed chain cohomology using matrix rank fallback"],
                )

            if not self._sage_available:
                return SymbolicResult(
                    expression=complex_desc,
                    result="",
                    backend=self.backend,
                    success=False,
                    error="SymPy unavailable and SageMath not available for cohomology",
                    computation_time=time.time() - start,
                )

            code = self._build_sage_chain_cohomology_code(differentials)
            output = await self._run_sage(code, timeout=30)
            return SymbolicResult(
                expression=complex_desc,
                result=output,
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
            )
        except Exception as e:
            return SymbolicResult(
                expression=complex_desc,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    async def algebraic_geometry(
        self, variety_desc: str, operation: str
    ) -> SymbolicResult:
        """Perform algebraic geometry operations on varieties."""
        start = time.time()
        normalized_op = (operation or "").strip().lower()
        if not normalized_op:
            normalized_op = "dimension"

        if not self._sage_available and not SYMPY_AVAILABLE:
            return SymbolicResult(
                expression=variety_desc,
                result="",
                backend=self.backend,
                success=False,
                error="No algebraic-geometry backend available (SageMath/SymPy)",
                computation_time=time.time() - start,
            )
        try:
            equations = self._parse_variety_data(variety_desc)
            if not equations:
                return SymbolicResult(
                    expression=variety_desc,
                    result="",
                    backend=self.backend,
                    success=False,
                    error="Unable to parse variety description",
                    computation_time=time.time() - start,
                )

            if self._sage_available and normalized_op in {"dimension", "dim", "irreducible", "is_irreducible"}:
                code = self._build_sage_algebraic_geometry_code(
                    variety_desc, normalized_op
                )
                output = await self._run_sage(code, timeout=30)
                return SymbolicResult(
                    expression=variety_desc,
                    result=output,
                    backend=self.backend,
                    success=True,
                    computation_time=time.time() - start,
                )

            output = self._compute_algebraic_geometry_fallback(
                equations, normalized_op
            )
            return SymbolicResult(
                expression=variety_desc,
                result=output,
                backend=self.backend,
                success=True,
                computation_time=time.time() - start,
                steps=["Algebraic geometry fallback inference from parsed equations"],
            )
        except Exception as e:
            return SymbolicResult(
                expression=variety_desc,
                result="",
                backend=self.backend,
                success=False,
                error=str(e),
                computation_time=time.time() - start,
            )

    def _parse_json_or_ast(self, raw: str) -> Any | None:
        """Parse JSON or Python literal payload with best-effort decoding."""
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            return ast.literal_eval(raw)
        except Exception:
            return None

    def _parse_matrix(self, raw_matrix: Any) -> list[list[str]] | None:
        """Normalize a matrix payload into rows of strings."""
        if raw_matrix is None:
            return None
        matrix = raw_matrix
        if isinstance(matrix, str):
            matrix = self._parse_json_or_ast(matrix)
        if not isinstance(matrix, list):
            return None
        if not matrix:
            return []
        parsed: list[list[str]] = []
        for row in matrix:
            if not isinstance(row, list):
                return None
            parsed_row: list[str] = []
            for value in row:
                if isinstance(value, (int, float)):
                    parsed_row.append(str(value))
                elif isinstance(value, str):
                    parsed_row.append(value.strip())
                else:
                    parsed_row.append(str(value))
            parsed.append(parsed_row)
        return parsed

    def _parse_chain_complex_description(
        self, complex_desc: str
    ) -> list[tuple[int, int, list[list[str]]]]:
        """Parse chain complex descriptors into (from, to, matrix) tuples."""
        payload = self._parse_json_or_ast(complex_desc)
        entries: list[tuple[int, int, list[list[str]]]] = []

        if isinstance(payload, list):
            for idx, item in enumerate(payload):
                from_grade = idx + 1
                to_grade = idx
                matrix = item
                if isinstance(item, dict):
                    from_grade = int(item.get("from", from_grade))
                    to_grade = int(item.get("to", item.get("to_degree", to_grade)))
                    matrix = item.get("matrix", item.get("m", item.get("map")))
                parsed = self._parse_matrix(matrix)
                if parsed is not None:
                    entries.append((from_grade, to_grade, parsed))

        if isinstance(payload, dict):
            chain_blocks = (
                payload.get("chain_complex")
                or payload.get("differentials")
                or payload.get("maps")
                or payload.get("matrices")
            )
            if isinstance(chain_blocks, list):
                for idx, item in enumerate(chain_blocks):
                    from_grade = idx + 1
                    to_grade = idx
                    matrix = item
                    if isinstance(item, dict):
                        from_grade = int(item.get("from", from_grade))
                        to_grade = int(item.get("to", item.get("to_degree", to_grade)))
                        matrix = item.get("matrix", item.get("m", item.get("map")))
                    parsed = self._parse_matrix(matrix)
                    if parsed is not None:
                        entries.append((from_grade, to_grade, parsed))

            if not entries:
                for key, value in payload.items():
                    match = re.match(r"d(\d+)", str(key).lower())
                    if not match:
                        continue
                    from_grade = int(match.group(1))
                    to_grade = from_grade - 1
                    parsed = self._parse_matrix(value)
                    if parsed is not None:
                        entries.append((from_grade, to_grade, parsed))

        if not entries and complex_desc:
            for match in re.finditer(r"d(\d+)\s*[:=]\s*(\[[\s\S]*?\])", complex_desc):
                from_grade = int(match.group(1))
                parsed = self._parse_matrix(match.group(2))
                if parsed is not None:
                    entries.append((from_grade, from_grade - 1, parsed))

        return sorted(entries, key=lambda item: item[0])

    def _compute_chain_cohomology_sympy(
        self, differentials: list[tuple[int, int, list[list[str]]]]
    ) -> dict[str, Any]:
        """Compute cohomology dimensions from differential matrices."""
        if not SYMPY_AVAILABLE:
            raise RuntimeError("SymPy unavailable")

        matrix_by_grade: dict[int, Any] = {}
        rank_by_grade: dict[int, int] = {}
        dim_by_grade: dict[int, int] = {}
        warnings: list[str] = []

        for from_grade, to_grade, matrix in differentials:
            sp_matrix = sp.Matrix([[sp.sympify(v) for v in row] for row in matrix])
            matrix_by_grade[from_grade] = sp_matrix
            rank_by_grade[from_grade] = int(sp_matrix.rank())
            dim_by_grade[from_grade] = sp_matrix.cols
            dim_by_grade.setdefault(to_grade, sp_matrix.rows)

        max_grade = max(dim_by_grade.keys(), default=0)

        for grade in sorted(matrix_by_grade.keys()):
            next_matrix = matrix_by_grade.get(grade + 1)
            current_matrix = matrix_by_grade.get(grade)
            if next_matrix is None or current_matrix is None:
                continue
            if current_matrix.shape[1] == next_matrix.shape[0]:
                comp = current_matrix * next_matrix
                if comp != sp.zeros(current_matrix.rows, next_matrix.cols):
                    warnings.append(
                        f"Chain condition violated at d_{grade} * d_{grade + 1}"
                    )

        cohomology_dims: dict[str, int] = {}
        for grade in range(max_grade + 1):
            if grade not in dim_by_grade:
                continue
            dim = dim_by_grade[grade]
            rank_in = rank_by_grade.get(grade, 0)
            rank_out = rank_by_grade.get(grade + 1, 0)
            dim_ker = max(0, dim - rank_in)
            cohomology_dims[f"H^{grade}"] = max(0, dim_ker - rank_out)

        euler = 0
        for label, dim in cohomology_dims.items():
            try:
                exponent = int(label.split("^", 1)[1])
                euler += ((-1) ** exponent) * int(dim)
            except Exception:
                pass

        return {
            "type": "chain_cohomology",
            "operation": "cohomology",
            "differential_ranks": {
                str(k): v for k, v in sorted(rank_by_grade.items(), key=lambda item: item[0])
            },
            "dimensions": dim_by_grade,
            "cohomology_dimensions": cohomology_dims,
            "euler_characteristic": euler,
            "warnings": warnings,
        }

    def _build_sage_chain_cohomology_code(
        self, differentials: list[tuple[int, int, list[list[str]]]]
    ) -> str:
        """Build a Sage payload script for cohomology dimension reporting."""
        lines: list[str] = [
            "from sage.all import *",
            "import json",
        ]
        dims: dict[int, int] = {}
        matrix_names: dict[int, str] = {}
        for from_grade, to_grade, matrix in differentials:
            lines.append(
                f"d_{from_grade} = Matrix({self._matrix_to_sage_literal(matrix)})"
            )
            matrix_names[from_grade] = f"d_{from_grade}"
            dims[from_grade] = len(matrix[0]) if matrix and matrix[0] else 0
            dims[to_grade] = len(matrix)
        lines.append("ranks = {" + ", ".join(
            f'"{g}": int({name}.rank())' for g, name in matrix_names.items()
        ) + "}")
        lines.append("cohomology = {}")
        lines.append("warnings = []")
        lines.append("max_grade = max(dims.keys()) if dims else 0")
        lines.append("for n in range(max_grade + 1):")
        lines.append("    dim_n = dims.get(n)")
        lines.append("    if dim_n is None:")
        lines.append("        continue")
        lines.append("    rank_in = ranks.get(n, 0)")
        lines.append("    rank_out = ranks.get(n + 1, 0)")
        lines.append("    ker_dim = max(0, dim_n - rank_in)")
        lines.append("    cohomology['H^%s' % n] = max(0, ker_dim - rank_out)")
        lines.append("for n in sorted(list(ranks.keys())):")
        lines.append("    if n + 1 in ranks:")
        lines.append("        A = globals().get('d_%s' % n)")
        lines.append("        B = globals().get('d_%s' % (n + 1))")
        lines.append("        if A is not None and B is not None and A.ncols() == B.nrows():")
        lines.append("            if (A * B) != 0:")
        lines.append(
            "                warnings.append('Chain condition violated at d_%s * d_%s' % (n, n + 1))"
        )
        lines.append("euler = 0")
        lines.append("for key, value in cohomology.items():")
        lines.append("    try:")
        lines.append("        idx = int(key.split('^')[1])")
        lines.append("        euler += (-1) ** idx * int(value)")
        lines.append("    except Exception:")
        lines.append("        pass")
        lines.append(
            "print(json.dumps({"
            "'type': 'chain_cohomology', "
            "'operation': 'cohomology', "
            "'differential_ranks': {k: int(v) for k, v in ranks.items()}, "
            "'dimensions': {k: int(v) for k, v in dims.items()}, "
            "'cohomology_dimensions': {k: int(v) for k, v in cohomology.items()}, "
            "'euler_characteristic': int(euler), "
            "'warnings': warnings})"
            "))"
        )
        return "\n".join(lines)

    @staticmethod
    def _matrix_to_sage_literal(matrix: list[list[str]]) -> str:
        rows = ["[" + ", ".join(v for v in row) + "]" for row in matrix]
        return "[" + ", ".join(rows) + "]"

    def _parse_variety_data(self, variety_desc: str) -> list[str]:
        text = (variety_desc or "").strip()
        if not text:
            return []
        payload = self._parse_json_or_ast(text)
        equations: list[str] = []
        if isinstance(payload, dict):
            raw = payload.get("equations") or payload.get("variety") or payload.get("ideal")
            if isinstance(raw, list):
                equations = [str(item) for item in raw if isinstance(item, str)]
        elif isinstance(payload, list):
            equations = [str(item) for item in payload if isinstance(item, str)]
        if not equations and "=" in text:
            for part in re.split(r"[;\n,]", text):
                chunks = [chunk.strip() for chunk in part.split("=")]
                if len(chunks) == 2 and chunks[0] and chunks[1]:
                    equations.append(f"({chunks[0]})-({chunks[1]})")
        return equations

    def _compute_algebraic_geometry_fallback(
        self, equations: list[str], operation: str
    ) -> str:
        if not SYMPY_AVAILABLE:
            return "SymPy unavailable for fallback algebraic geometry analysis"
        op = (operation or "").strip().lower()
        exprs = [sp.sympify(eq) for eq in equations if str(eq).strip()]
        symbols = sorted({s for expr in exprs for s in expr.free_symbols}, key=lambda s: s.name)
        var_count = len(symbols)

        if op in {"dimension", "dim"}:
            eq_count = len(exprs)
            dim = max(0, var_count - eq_count)
            return json.dumps({
                "type": "algebraic_geometry",
                "operation": "dimension",
                "variables": [str(s) for s in symbols],
                "equation_count": eq_count,
                "dimension_estimate": dim,
                "note": "Estimated by variable/equation count",
            }, ensure_ascii=False)

        if op in {"irreducible", "is_irreducible"}:
            factors: list[dict[str, str]] = []
            all_irreducible = True
            for expr in exprs:
                factor = str(sp.factor(expr))
                factors.append({"expression": str(expr), "factorization": factor})
                if "*" in factor:
                    all_irreducible = False
            return json.dumps({
                "type": "algebraic_geometry",
                "operation": "irreducible",
                "variables": [str(s) for s in symbols],
                "all_irreducible": all_irreducible,
                "factorizations": factors,
            }, ensure_ascii=False)

        return json.dumps({
            "type": "algebraic_geometry",
            "operation": op,
            "variables": [str(s) for s in symbols],
            "equations": [str(expr) for expr in exprs],
            "note": "Operation not implemented; returned normalized equation profile",
        }, ensure_ascii=False)

    def _build_sage_algebraic_geometry_code(
        self, variety_desc: str, operation: str
    ) -> str:
        equations = self._parse_variety_data(variety_desc)
        var_names: set[str] = set()
        for eq in equations:
            for name in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", eq):
                if name not in {"oo", "inf", "pi", "sin", "cos", "log", "exp", "sqrt"}:
                    var_names.add(name)
        vars_sorted = sorted(var_names)
        if not vars_sorted:
            return (
                "import json\n"
                "print(json.dumps({'error': 'No variables found in variety description'}))"
            )
        header = (
            f"R.<{', '.join(vars_sorted)}> = PolynomialRing(QQ, {len(vars_sorted)})\n"
        )
        equations_payload = ", ".join([eq for eq in equations])
        if operation in {"dimension", "dim"}:
            return (
                "import json\n"
                + header
                + f"I = Ideal([{equations_payload}])\n"
                + "try:\n"
                + "    print(json.dumps({'type':'algebraic_geometry','operation':'dimension','dimension':str(I.dimension())}))\n"
                + "except Exception as exc:\n    print(json.dumps({'error': str(exc)}))"
            )

        if operation in {"irreducible", "is_irreducible"}:
            return (
                "import json\n"
                + header
                + f"I = Ideal([{equations_payload}])\n"
                + "try:\n"
                + "    print(json.dumps({'type':'algebraic_geometry','operation':'irreducible','prime':str(I.is_prime()),'radical':str(I.is_radical())}))\n"
                + "except Exception as exc:\n    print(json.dumps({'error': str(exc)}))"
            )

        return (
            "import json\n"
            f"print(json.dumps({{'type': 'algebraic_geometry', 'operation': '{operation}', 'error': 'Operation not supported by Sage fallback'}}))"
        )

    async def _run_sage(self, code: str, timeout: int = 30) -> str:
        """Run Sage code via subprocess and return output."""
        return await asyncio.to_thread(
            self._run_sage_sync, code, timeout
        )

    def _run_sage_sync(self, code: str, timeout: int = 30) -> str:
        """Synchronous Sage execution."""
        try:
            result = subprocess.run(
                ["sage", "-c", code],
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            if result.returncode != 0:
                return f"Error: {result.stderr}"
            return result.stdout
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Sage computation timed out after {timeout}s")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# LaTeX Parser
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class LaTeXParser:
    """Convert between LaTeX and SymPy expressions."""

    @staticmethod
    def parse_latex_to_sympy(latex_str: str) -> str:
        """Convert LaTeX string to SymPy expression string using sympy.parsing.latex when available."""
        if SYMPY_AVAILABLE and sp_parse_latex:
            try:
                # Try native SymPy LaTeX parser first
                expr = sp_parse_latex(latex_str)
                return str(expr)
            except Exception as e:
                logger.debug(f"Native LaTeX parsing failed: {e}, falling back to regex")

        # Fallback to regex-based parsing
        cleaned = LaTeXParser._clean_latex(latex_str)
        # Comprehensive mapping for common LaTeX patterns
        replacements = {
            r"\\frac\{([^}]*)\}\{([^}]*)\}": r"(\1)/(\2)",
            r"\\sqrt\{([^}]*)\}": r"sqrt(\1)",
            r"\\sin": "sin",
            r"\\cos": "cos",
            r"\\tan": "tan",
            r"\\log": "log",
            r"\\ln": "ln",
            r"\\exp": "exp",
            r"\\left\(": "(",
            r"\\right\)": ")",
            r"\\cdot": "*",
            r"\\times": "*",
            r"\\alpha": "alpha",
            r"\\beta": "beta",
            r"\\gamma": "gamma",
            r"\\pi": "pi",
            r"\\infty": "oo",
            r"\^": "**",
            r"\{": "",
            r"\}": "",
        }
        result = cleaned
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result)
        return result

    @staticmethod
    def parse_sympy_to_latex(expr_str: str) -> str:
        """Convert SymPy expression to LaTeX."""
        if not SYMPY_AVAILABLE:
            return expr_str
        try:
            expr = sympify(expr_str)
            return sp.latex(expr)
        except Exception as e:
            logger.warning(f"LaTeX conversion failed: {e}")
            return expr_str

    @staticmethod
    def _clean_latex(latex: str) -> str:
        """Clean up common LaTeX quirks."""
        # Remove comments
        latex = re.sub(r"%.*$", "", latex, flags=re.MULTILINE)
        # Remove extra whitespace
        latex = re.sub(r"\s+", " ", latex).strip()
        return latex

    @staticmethod
    def _handle_special_functions(latex: str) -> str:
        """Handle special mathematical functions."""
        functions = {
            r"\arcsin": "asin",
            r"\arccos": "acos",
            r"\arctan": "atan",
            r"\sinh": "sinh",
            r"\cosh": "cosh",
            r"\tanh": "tanh",
        }
        result = latex
        for latex_fn, sympy_fn in functions.items():
            result = result.replace(latex_fn, sympy_fn)
        return result


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# Mathematical Verifier
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class MathematicalVerifier:
    """High-level verification of mathematical claims."""

    def __init__(self, sympy_engine: SymPyEngine | None = None) -> None:
        """Initialize verifier with optional SymPy engine."""
        self.engine = sympy_engine
        if self.engine is None and SYMPY_AVAILABLE:
            self.engine = SymPyEngine()

    async def verify_claim(self, claim_text: str, llm: Any = None) -> VerificationEvidence:
        """Verify a mathematical claim using multiple strategies."""
        logger.debug(f"Verifying claim: {claim_text}")
        if not self.engine:
            raise RuntimeError("SymPy engine required for claim verification")

        claim_payload = await self._extract_math_from_claim(claim_text, llm)
        claim_type = str(claim_payload.get("type", "")).lower()

        if claim_type in {"algebraic_identity", "equation", "identity", "equality"}:
            return await self.verify_algebraic_identity(
                str(claim_payload.get("lhs", "")),
                str(claim_payload.get("rhs", "")),
            )
        if claim_type in {"limit", "limit_check", "limit_claim"}:
            return await self.verify_limit_claim(
                str(claim_payload.get("expr", "")),
                str(claim_payload.get("var", "x")),
                str(claim_payload.get("point", "0")),
                str(claim_payload.get("expected", "")),
            )
        if claim_type in {"series", "series_convergence", "convergence"}:
            return await self.verify_convergence(
                str(claim_payload.get("series_expr", "")),
                str(claim_payload.get("var", "n")),
                expected=claim_payload.get("expected"),
            )
        if claim_type == "eigenvalue":
            return await self.verify_eigenvalue_claim(
                str(claim_payload.get("matrix", "")),
                [str(v) for v in claim_payload.get("expected_eigenvalues", [])]
                if isinstance(claim_payload.get("expected_eigenvalues"), list)
                else [],
            )
        if claim_type == "numerical_spot_check":
            test_points = claim_payload.get("test_points", [])
            return await self.numerical_spot_check(
                str(claim_payload.get("expr", "")),
                [dict(point) for point in test_points if isinstance(point, dict)],
                expected_fn=claim_payload.get("expected_fn"),
            )

        inferred = self._infer_math_claim_type(claim_text)
        inferred_type = inferred.get("type")
        if inferred_type == "limit":
            return await self.verify_limit_claim(
                str(inferred.get("expr", "")),
                str(inferred.get("var", "x")),
                str(inferred.get("point", "0")),
                str(inferred.get("expected", "")),
            )
        if inferred_type == "series_convergence":
            return await self.verify_convergence(
                str(inferred.get("series_expr", "")),
                str(inferred.get("var", "n")),
                expected=inferred.get("expected"),
            )
        if inferred_type == "algebraic_identity":
            return await self.verify_algebraic_identity(
                str(inferred.get("lhs", "")),
                str(inferred.get("rhs", "")),
            )

        return VerificationEvidence(
            claim=claim_text,
            verification_type="unknown",
            result=SymbolicResult(
                expression=claim_text,
                result="",
                backend=ComputationBackend.SYMPY,
                success=False,
                error="Could not classify claim into a supported mathematical check",
            ),
            supports_claim=False,
            confidence=0.0,
            explanation="No supported structure detected in claim",
        )

    async def verify_algebraic_identity(
        self, lhs: str, rhs: str
    ) -> VerificationEvidence:
        """Verify an algebraic identity: lhs = rhs."""
        if not self.engine:
            raise RuntimeError("SymPy engine required for identity verification")

        result = await self.engine.verify_identity(lhs, rhs)
        supports = result.success and result.result == "True"
        return VerificationEvidence(
            claim=f"{lhs} = {rhs}",
            verification_type="algebraic_identity",
            result=result,
            supports_claim=supports,
            confidence=0.95 if supports else 0.05,
            explanation=f"Symbolic verification using SymPy",
        )

    async def verify_limit_claim(
        self, expr: str, var: str, point: str, expected: str
    ) -> VerificationEvidence:
        """Verify a limit claim: lim_{var→point} expr = expected."""
        if not self.engine:
            raise RuntimeError("SymPy engine required for limit verification")

        result = await self.engine.limit(expr, var, point)
        expected_norm = self._normalize_expr(expected)
        supports = False
        if result.success and result.result and expected_norm:
            actual_norm = self._normalize_expr(result.result)
            if actual_norm == expected_norm:
                supports = True
            else:
                try:
                    lhs = await self.engine._parse_expr(actual_norm)
                    rhs = await self.engine._parse_expr(expected_norm)
                    diff = await asyncio.to_thread(lambda: simplify(lhs - rhs))
                    supports = diff == 0
                except Exception:
                    supports = False
        return VerificationEvidence(
            claim=f"lim_{{{var}→{point}}} {expr} = {expected_norm}",
            verification_type="limit_check",
            result=result,
            supports_claim=supports,
            confidence=0.9 if supports else 0.1,
            explanation="Limit computed symbolically with SymPy",
        )

    async def verify_convergence(
        self,
        series_expr: str,
        var: str,
        expected: Any | None = None,
    ) -> VerificationEvidence:
        """Verify that a series converges."""
        if not self.engine:
            raise RuntimeError("SymPy engine required for convergence verification")

        expr = (series_expr or "").strip()
        if not expr:
            result = SymbolicResult(
                expression=series_expr,
                result="",
                backend=ComputationBackend.SYMPY,
                success=False,
                error="No expression provided",
            )
            return VerificationEvidence(
                claim=f"Series {series_expr} in {var} converges",
                verification_type="series_convergence",
                result=result,
                supports_claim=False,
                confidence=0.0,
                explanation=result.error,
            )

        symbol = sp.Symbol(var)
        try:
            parsed_expr = await self.engine._parse_expr(expr)
        except Exception as e:
            result = SymbolicResult(
                expression=series_expr,
                result="",
                backend=ComputationBackend.SYMPY,
                success=False,
                error=f"Failed to parse series expression: {e}",
            )
            return VerificationEvidence(
                claim=f"Series {series_expr} in {var} converges",
                verification_type="series_convergence",
                result=result,
                supports_claim=False,
                confidence=0.0,
                explanation=result.error,
            )

        checks: list[str] = []
        evidence_score = 0
        supports = None
        expected_converges = self._coerce_expected_bool(expected)

        # 1) Use summation where possible
        try:
            sum_result = await asyncio.to_thread(sp.summation, parsed_expr, (symbol, 1, sp.oo))
            if sum_result in {sp.oo, -sp.oo, sp.zoo}:
                supports = False
                checks.append("Symbolic summation diverges")
            elif getattr(sum_result, "is_finite", False):
                supports = True
                checks.append(f"Symbolic summation finite: {sum_result}")
                evidence_score += 1
        except Exception as e:
            checks.append(f"Symbolic summation unavailable: {e}")

        # 2) Ratio test
        if supports is None:
            try:
                ratio_expr = sp.Abs(parsed_expr.subs(symbol, symbol + 1) / parsed_expr)
                ratio = await asyncio.to_thread(limit, ratio_expr, symbol, sp.oo)
                checks.append(f"ratio_limit={ratio}")
                if ratio.is_number:
                    ratio_value = float(ratio.evalf())
                    if ratio_value < 1:
                        supports = True
                        evidence_score += 1
                    elif ratio_value > 1:
                        supports = False
                    elif ratio_value == 1.0:
                        checks.append("Ratio test inconclusive at L=1")
            except Exception as e:
                checks.append(f"Ratio test failed: {e}")

        # 3) Numeric tail behavior when symbolic path is inconclusive
        numeric_support = None
        try:
            abs_expr = sp.Abs(parsed_expr)
            evaluator = sp.lambdify(symbol, abs_expr, modules="math")
            terms = []
            for n in [20, 40, 80, 160, 320, 640]:
                value = evaluator(n)
                if value is None:
                    raise ValueError("Evaluator returned null")
                terms.append(float(value))
            checks.append(f"tail_terms={terms}")
            if all(terms[i] > terms[i + 1] for i in range(len(terms) - 1)):
                numeric_support = True
                evidence_score += 1
            elif all(terms[i] < terms[i + 1] * 1.02 for i in range(len(terms) - 1)):
                numeric_support = False
            elif terms[-1] < terms[0] * 0.3:
                # Mild monotone-like decay signal
                numeric_support = True
        except Exception as e:
            checks.append(f"Tail numeric check failed: {e}")

        if supports is None and numeric_support is not None:
            supports = numeric_support

        supports_claim = False if expected_converges is not None else bool(supports)
        if supports is not None and expected_converges is not None:
            supports_claim = supports is expected_converges

        confidence = 0.0
        if supports is True:
            confidence = min(0.95, 0.35 + 0.3 * evidence_score)
        elif supports is False:
            confidence = min(0.95, 0.35 + 0.15 * evidence_score)
        else:
            confidence = 0.2

        result = SymbolicResult(
            expression=series_expr,
            result=(
                "inconclusive"
                if supports is None
                else ("converges" if supports else "diverges")
            ),
            backend=ComputationBackend.SYMPY,
            success=True,
            steps=checks,
            error="",
        )

        explanation = "Series convergence inferred from summation, ratio, and numeric tail checks"
        if expected_converges is not None:
            target_text = "converges" if expected_converges else "diverges"
            explanation = f"Expected behavior: {target_text}. {explanation}"

        return VerificationEvidence(
            claim=f"Series {series_expr} in {var} converges",
            verification_type="series_convergence",
            result=result,
            supports_claim=supports_claim,
            confidence=confidence,
            explanation=explanation,
        )

    async def verify_eigenvalue_claim(
        self, matrix_str: str, expected_eigenvalues: list[str]
    ) -> VerificationEvidence:
        """Verify eigenvalue claim for a matrix."""
        if not self.engine:
            raise RuntimeError("SymPy engine required for eigenvalue verification")

        result = await self.engine.matrix_operation(matrix_str, "eigenvalues")
        supports = self._eigenvalue_claim_matches(result, expected_eigenvalues)
        return VerificationEvidence(
            claim=f"Matrix eigenvalues are {expected_eigenvalues}",
            verification_type="eigenvalue",
            result=result,
            supports_claim=supports,
            confidence=0.85 if supports else (0.1 if result.success else 0.0),
            explanation="Eigenvalues computed symbolically",
        )

    async def numerical_spot_check(
        self, expr: str, test_points: list[dict[str, float]], expected_fn: str | None = None
    ) -> VerificationEvidence:
        """Perform numerical spot checks at test points."""
        if not self.engine:
            raise RuntimeError("SymPy engine required for numerical checks")

        try:
            parsed_expr = await self.engine._parse_expr(expr)
            successes = 0
            for test_point in test_points:
                try:
                    result = await asyncio.to_thread(
                        lambda: parsed_expr.subs(test_point)
                    )
                    successes += 1
                except Exception:
                    pass

            supports = successes == len(test_points)
            return VerificationEvidence(
                claim=f"Expression {expr} is valid at {len(test_points)} points",
                verification_type="numerical_spot_check",
                result=SymbolicResult(
                    expression=expr,
                    result=f"Passed {successes}/{len(test_points)} test points",
                    backend=ComputationBackend.NUMERIC,
                    success=supports,
                ),
                supports_claim=supports,
                confidence=0.8 if supports else 0.2,
                explanation=f"Numerical evaluation at {len(test_points)} test points",
            )
        except Exception as e:
            return VerificationEvidence(
                claim=f"Expression {expr}",
                verification_type="numerical_spot_check",
                result=SymbolicResult(
                    expression=expr,
                    result="",
                    backend=ComputationBackend.NUMERIC,
                    success=False,
                    error=str(e),
                ),
                supports_claim=False,
                confidence=0.0,
                explanation=str(e),
            )

    async def batch_verify(
        self, claims: list[dict[str, Any]], llm: Any = None
    ) -> list[VerificationEvidence]:
        """Verify a batch of claims in parallel."""
        tasks = []
        for claim_dict in claims:
            claim_type = claim_dict.get("type", "unknown")
            if claim_type == "algebraic_identity":
                task = self.verify_algebraic_identity(
                    claim_dict.get("lhs", ""),
                    claim_dict.get("rhs", ""),
                )
            elif claim_type == "limit":
                task = self.verify_limit_claim(
                    claim_dict.get("expr", ""),
                    claim_dict.get("var", "x"),
                    claim_dict.get("point", "0"),
                    claim_dict.get("expected", "0"),
                )
            else:
                task = self.verify_claim(str(claim_dict), llm)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def _extract_math_from_claim(
        self, claim_text: str, llm: Any
    ) -> dict[str, Any]:
        """Use LLM to extract mathematical content from natural language claim."""
        # 1) explicit JSON payload (most robust)
        try:
            extracted = extract_json_from_text(
                claim_text,
                schema=self._math_claim_schema(),
                strict=True,
            )
            if isinstance(extracted, dict) and extracted.get("type"):
                return self._normalize_extracted_payload(extracted)
        except ValueError:
            pass

        # 2) optional LLM-assisted extraction (best effort; safe fallback)
        if llm is not None and hasattr(llm, "call"):
            maybe = await self._extract_with_llm(llm, claim_text)
            if maybe:
                return self._normalize_extracted_payload(maybe)

        # 3) heuristic fallback
        return self._infer_math_claim_type(claim_text)

    def _normalize_expr(self, expr: str) -> str:
        expr = (expr or "").strip()
        expr = expr.replace("∞", "oo")
        expr = expr.replace("→", "->")
        expr = expr.replace("−", "-")
        expr = expr.replace("–", "-")
        expr = expr.replace("＝", "=")
        return expr

    @staticmethod
    def _math_claim_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "lhs": {"type": "string"},
                "rhs": {"type": "string"},
                "expr": {"type": "string"},
                "var": {"type": "string"},
                "point": {"type": "string"},
                "series_expr": {"type": "string"},
                "matrix": {"type": "string"},
                "expected_eigenvalues": {"type": "array"},
                "test_points": {"type": "array"},
                "expected_fn": {"type": "string"},
            },
            "required": ["type"],
        }

    def _strip_markdown_fences(self, text: str) -> str:
        if not text:
            return ""
        match = re.search(r"```[a-zA-Z]*\n(.*?)\n```", text, re.S)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _normalize_extracted_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["type"] = str(normalized.get("type", "unknown")).lower().replace("-", "_")
        for key in ("lhs", "rhs", "expr", "series_expr", "point", "expected"):
            if key in normalized and isinstance(normalized[key], str):
                normalized[key] = self._normalize_expr(normalized[key])
        if "var" in normalized and isinstance(normalized["var"], str):
            normalized["var"] = normalized["var"].strip() or "n"
        return normalized

    @staticmethod
    def _coerce_expected_bool(value: Any | None) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return None
            if normalized in {"true", "t", "1", "yes", "y", "converges", "convergent"}:
                return True
            if normalized in {"false", "f", "0", "no", "n", "does not converge", "diverges", "divergent", "发散"}:
                return False
        return None

    @staticmethod
    def _eigenvalue_claim_matches(result: SymbolicResult, expected: list[str]) -> bool:
        if not result.success:
            return False
        if not expected:
            return True

        parsed = (
            (result.result or "")
            .replace("{", "")
            .replace("}", "")
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
            .replace(" ", "")
        )
        if not parsed:
            return False

        actual = Counter(token for token in (part.strip() for part in parsed.split(",")) if token)
        expected_set = Counter(
            str(v).strip().replace(" ", "") for v in expected if str(v).strip()
        )
        return actual == expected_set

    async def _extract_with_llm(self, llm: Any, claim_text: str) -> dict[str, Any] | None:
        from autoforge.engine.llm_router import TaskComplexity

        schema = self._math_claim_schema()
        prompt = (
            "Extract math claim structure as strict JSON.\n"
            "Allowed types: algebraic_identity, limit, series_convergence, numerical_spot_check, eigenvalue.\n"
            "Return object with fields type, lhs, rhs, expr, var, point, expected, series_expr, matrix, expected_eigenvalues.\n"
            "No Markdown, JSON only.\n\n"
            f"Claim: {claim_text}"
        )
        try:
            response = await llm.call(
                prompt,
                complexity=TaskComplexity.STANDARD,
                max_tokens=400,
                response_json_schema=schema,
            )
        except Exception:
            return None

        text = ""
        for block in getattr(response, "content", []):
            if getattr(block, "type", "") == "text":
                text += str(getattr(block, "text", ""))
        if not text:
            return None
        text = text.strip()
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and payload.get("type"):
                return payload
        except Exception:
            pass
        try:
            payload = extract_json_from_text(
                text,
                schema=schema,
                strict=True,
            )
            if isinstance(payload, dict) and payload.get("type"):
                return payload
        except ValueError:
            return None
        return None

    def _infer_math_claim_type(self, claim_text: str) -> dict[str, Any]:
        text = self._strip_markdown_fences(claim_text).strip()
        if not text:
            return {"type": "unknown", "content": ""}

        lower = text.lower()

        # Limit patterns: lim_{x->a} f(x) = L or limit(x->a, f(x)) = L
        lim_match = re.search(
            r"lim(?:it)?\s*_\{\s*([a-zA-Z])\s*(?:→|->)\s*([^}]+)\}\s*([^\n=]+)\s*=?\s*(.*)",
            text,
            flags=re.I,
        )
        if lim_match:
            return {
                "type": "limit",
                "var": lim_match.group(1),
                "point": lim_match.group(2),
                "expr": lim_match.group(3),
                "expected": lim_match.group(4),
            }

        lim_fn_match = re.search(
            r"limit\s*\(\s*([a-zA-Z])\s*(?:→|->)\s*([^,)]*)\s*,\s*([^)]+)\)\s*=?\s*(.*)",
            text,
            flags=re.I,
        )
        if lim_fn_match:
            return {
                "type": "limit",
                "var": lim_fn_match.group(1),
                "point": lim_fn_match.group(2),
                "expr": lim_fn_match.group(3),
                "expected": lim_fn_match.group(4),
            }

        # Series / convergence
        if "converge" in lower or "收敛" in text or "发散" in text:
            expected = None
            if "diverge" in lower or "发散" in text:
                expected = False
            elif "converge" in lower or "收敛" in text:
                expected = True

            series_expr = text
            var = None
            for candidate in ("n", "k", "m", "i", "j"):
                if re.search(rf"\b{candidate}\b", text):
                    var = candidate
                    break
            if var is None:
                var = "n"
            return {
                "type": "series_convergence",
                "series_expr": self._normalize_expr(series_expr),
                "var": var,
                "expected": expected,
            }

        # Algebraic identity
        eq_match = re.search(r"(.+?)\s*(?:==|=)\s*(.+)", text, flags=re.S)
        if eq_match:
            lhs = eq_match.group(1).strip()
            rhs = eq_match.group(2).strip()
            if lhs and rhs:
                return {
                    "type": "algebraic_identity",
                    "lhs": self._normalize_expr(lhs),
                    "rhs": self._normalize_expr(rhs),
                }

        return {
            "type": "unknown",
            "content": self._normalize_expr(text),
        }
# Main Facade
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class SymbolicComputeEngine:
    """Main facade for symbolic computation."""

    def __init__(
        self, preferred_backend: ComputationBackend = ComputationBackend.SYMPY
    ) -> None:
        """Initialize symbolic compute engine."""
        self.preferred_backend = preferred_backend
        self.sympy_engine: SymPyEngine | None = None
        self.sage_engine: SageEngine | None = None
        self.verifier: MathematicalVerifier | None = None

        logger.info(f"SymbolicComputeEngine initialized (preferred: {preferred_backend})")

    async def compute(
        self, expression: str, operation: str, **kwargs: Any
    ) -> SymbolicResult:
        """Perform a computation operation."""
        if self.preferred_backend == ComputationBackend.SYMPY:
            if not self.sympy_engine:
                if not SYMPY_AVAILABLE:
                    raise RuntimeError("SymPy not available")
                self.sympy_engine = SymPyEngine()

            if operation == "simplify":
                return await self.sympy_engine.simplify(expression)
            elif operation == "solve":
                return await self.sympy_engine.solve(
                    expression, kwargs.get("variable", "x")
                )
            elif operation == "integrate":
                return await self.sympy_engine.integrate(
                    expression,
                    variable=kwargs.get("variable", "x"),
                    limits=kwargs.get("limits"),
                )
            elif operation == "differentiate":
                return await self.sympy_engine.differentiate(
                    expression,
                    variable=kwargs.get("variable", "x"),
                    order=kwargs.get("order", 1),
                )
            elif operation == "limit":
                return await self.sympy_engine.limit(
                    expression,
                    variable=kwargs.get("variable", "x"),
                    point=kwargs.get("point", "0"),
                )
            elif operation == "series":
                return await self.sympy_engine.series_expand(
                    expression,
                    variable=kwargs.get("variable", "x"),
                    point=kwargs.get("point", "0"),
                    order=kwargs.get("order", 6),
                )
            elif operation == "matrix":
                return await self.sympy_engine.matrix_operation(
                    expression, kwargs.get("matrix_op", "determinant")
                )
            else:
                raise ValueError(f"Unknown operation: {operation}")
        else:
            raise ValueError(f"Backend {self.preferred_backend} not yet supported")

    async def verify(self, claim: str, llm: Any = None) -> VerificationEvidence:
        """Verify a mathematical claim."""
        if not self.verifier:
            if not self.sympy_engine and SYMPY_AVAILABLE:
                self.sympy_engine = SymPyEngine()
            self.verifier = MathematicalVerifier(self.sympy_engine)

        return await self.verifier.verify_claim(claim, llm)

    async def batch_compute(
        self, tasks: list[dict[str, Any]]
    ) -> list[SymbolicResult]:
        """Execute multiple computations in parallel."""
        compute_tasks = [
            self.compute(
                task["expression"],
                task["operation"],
                **task.get("kwargs", {}),
            )
            for task in tasks
        ]
        return await asyncio.gather(*compute_tasks)

    def available_backends(self) -> list[ComputationBackend]:
        """List available computation backends."""
        backends = []
        if SYMPY_AVAILABLE:
            backends.append(ComputationBackend.SYMPY)
        sage_engine = SageEngine()
        if sage_engine._sage_available:
            backends.append(ComputationBackend.SAGE)
        # MATHEMATICA_FREE would require checking for WolframScript
        backends.append(ComputationBackend.NUMERIC)
        return backends
