"""
Real Lean 4 Lake project integration for AutoForge framework.

Provides production-quality Lean 4 theorem proving with Mathlib dependencies,
replacing bare `lean file.lean` approach with proper Lake project management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoforge.engine.llm_router import TaskComplexity

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration & Data Classes
# ============================================================================


@dataclass
class LakeProjectConfig:
    """Configuration for Lake project creation and Lean compilation."""

    name: str = "AutoForgeProof"
    lean_version: str = "v4.15.0"
    mathlib_version: str = "master"
    extra_dependencies: list[dict] = field(default_factory=list)
    auto_implicit: bool = False
    max_heartbeats: int = 400000
    timeout_seconds: int = 600

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "name": self.name,
            "lean_version": self.lean_version,
            "mathlib_version": self.mathlib_version,
            "extra_dependencies": self.extra_dependencies,
            "auto_implicit": self.auto_implicit,
            "max_heartbeats": self.max_heartbeats,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class LakeVerificationResult:
    """Result of Lean theorem verification."""

    success: bool
    has_sorry: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    compilation_time: float = 0.0
    lean_version: str = ""
    mathlib_used: bool = False


# ============================================================================
# MathlibImportResolver
# ============================================================================


class MathlibImportResolver:
    """Maps mathematical concepts to Mathlib4 import paths."""

    IMPORT_MAP: dict[str, str] = {
        "measure": "import Mathlib.MeasureTheory.Measure.MeasureSpace",
        "topology": "import Mathlib.Topology.Basic",
        "group": "import Mathlib.Algebra.Group.Basic",
        "ring": "import Mathlib.Algebra.Ring.Basic",
        "field": "import Mathlib.Algebra.Field.Basic",
        "nat": "import Mathlib.Data.Nat.Basic",
        "real": "import Mathlib.Analysis.SpecialLimits.Basic",
        "probability": "import Mathlib.Probability.ProbabilityMassFunction.Basic",
        "combinatorics": "import Mathlib.Combinatorics.SimpleGraph.Basic",
        "linear_algebra": "import Mathlib.LinearAlgebra.Basic",
        "matrix": "import Mathlib.LinearAlgebra.Matrix.Basic",
        "category": "import Mathlib.CategoryTheory.Category.Basic",
        "functor": "import Mathlib.CategoryTheory.Functor.Basic",
        "algebra": "import Mathlib.Algebra.Algebra.Basic",
        "module": "import Mathlib.LinearAlgebra.Module.Basic",
        "finite": "import Mathlib.Data.Fintype.Basic",
        "set": "import Mathlib.Data.Set.Basic",
        "list": "import Mathlib.Data.List.Basic",
        "array": "import Mathlib.Data.Array.Basic",
        "hash": "import Mathlib.Data.HashMap.Basic",
        "order": "import Mathlib.Order.Basic",
        "lattice": "import Mathlib.Order.Lattice",
        "logic": "import Mathlib.Logic.Basic",
        "function": "import Mathlib.Logic.Function.Basic",
        "equiv": "import Mathlib.Logic.Equiv.Basic",
        "calculus": "import Mathlib.Analysis.Calculus.Deriv.Basic",
        "integral": "import Mathlib.Analysis.Integral.Lebesgue",
        "series": "import Mathlib.Analysis.SpecialFunctions.Pow.Real",
        "metric": "import Mathlib.Topology.MetricSpace.Basic",
        "normed": "import Mathlib.Analysis.Normed.Group.Basic",
        "polynomial": "import Mathlib.Data.Polynomial.Basic",
        "number_theory": "import Mathlib.NumberTheory.Basic",
    }

    @staticmethod
    def resolve_imports(lean_code: str) -> list[str]:
        """
        Analyze Lean code to determine needed Mathlib imports.

        Args:
            lean_code: Lean code to analyze

        Returns:
            List of import statements needed
        """
        imports = set()
        code_lower = lean_code.lower()

        # Check for explicit namespaces and definitions
        for concept, import_stmt in MathlibImportResolver.IMPORT_MAP.items():
            if concept in code_lower:
                imports.add(import_stmt)

        # Check for specific Lean identifiers
        lean_identifiers = {
            "Fintype": "import Mathlib.Data.Fintype.Basic",
            "Equiv": "import Mathlib.Logic.Equiv.Basic",
            "Functor": "import Mathlib.CategoryTheory.Functor.Basic",
            "Subgroup": "import Mathlib.Algebra.Group.Subgroup.Basic",
            "Ideal": "import Mathlib.RingTheory.Ideal.Basic",
            "Module": "import Mathlib.LinearAlgebra.Module.Basic",
            "Morphism": "import Mathlib.CategoryTheory.Functor.Basic",
        }

        for ident, import_stmt in lean_identifiers.items():
            if re.search(rf"\b{ident}\b", lean_code):
                imports.add(import_stmt)

        # Always include basic imports
        imports.add("import Mathlib.Tactic")

        return sorted(list(imports))

    @staticmethod
    async def resolve_with_llm(lean_code: str, llm: Any) -> list[str]:
        """
        Use LLM to suggest imports for complex cases.

        Args:
            lean_code: Lean code to analyze
            llm: LLM interface with async call method

        Returns:
            List of import statements suggested by LLM
        """
        prompt = f"""Analyze this Lean 4 code and suggest all necessary Mathlib4 imports.
Return ONLY import statements, one per line, in the format: import Mathlib.Path.To.Module

Code:
```lean
{lean_code}
```

Imports:"""

        try:
            result = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            imports_text = result.content if hasattr(result, "content") else str(result)

            imports = []
            for line in imports_text.strip().split("\n"):
                line = line.strip()
                if line.startswith("import Mathlib"):
                    imports.append(line)

            return imports if imports else MathlibImportResolver.resolve_imports(lean_code)
        except Exception as e:
            logger.warning(f"LLM import resolution failed: {e}, falling back to pattern matching")
            return MathlibImportResolver.resolve_imports(lean_code)


# ============================================================================
# ElanManager
# ============================================================================


class ElanManager:
    """Static utility for managing Elan and Lean toolchain."""

    @staticmethod
    async def check_installation() -> dict[str, Any]:
        """
        Check Lean installation status.

        Returns:
            Dictionary with version info and cache sizes
        """
        result: dict[str, Any] = {
            "lean_version": None,
            "lake_version": None,
            "elan_version": None,
            "mathlib_cache_size": 0,
        }

        # Check lean
        try:
            proc = await asyncio.create_subprocess_exec(
                "lean",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=5,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            result["lean_version"] = stdout.decode().strip()
        except Exception as e:
            logger.debug(f"Lean not found: {e}")

        # Check lake
        try:
            proc = await asyncio.create_subprocess_exec(
                "lake",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=5,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            result["lake_version"] = stdout.decode().strip()
        except Exception as e:
            logger.debug(f"Lake not found: {e}")

        # Check elan
        try:
            proc = await asyncio.create_subprocess_exec(
                "elan",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=5,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            result["elan_version"] = stdout.decode().strip()
        except Exception as e:
            logger.debug(f"Elan not found: {e}")

        # Estimate Mathlib cache
        cache_dir = Path.home() / ".elan" / "downloads"
        if cache_dir.exists():
            try:
                total = sum(
                    f.stat().st_size
                    for f in cache_dir.rglob("*")
                    if f.is_file()
                )
                result["mathlib_cache_size"] = total
            except Exception as e:
                logger.debug(f"Could not calculate cache size: {e}")

        return result

    @staticmethod
    async def install_elan() -> bool:
        """
        Install Elan toolchain manager.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Installing Elan...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "curl",
                "--proto",
                "=https",
                "--tlsv1.2",
                "-sSf",
                "https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=30,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                logger.error(f"Failed to download elan-init.sh: {stderr.decode()}")
                return False

            # Execute installer
            proc = await asyncio.create_subprocess_exec(
                "sh",
                "-s",
                "--",
                "-y",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=120,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdout), timeout=120
            )

            success = proc.returncode == 0
            if success:
                logger.info("Elan installed successfully")
            else:
                logger.error(f"Elan installation failed: {stderr.decode()}")
            return success
        except Exception as e:
            logger.error(f"Error installing Elan: {e}")
            return False

    @staticmethod
    async def install_toolchain(version: str) -> bool:
        """
        Install specific Lean toolchain version.

        Args:
            version: Lean version to install (e.g., 'v4.15.0')

        Returns:
            True if successful
        """
        logger.info(f"Installing Lean toolchain {version}...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "elan",
                "toolchain",
                "install",
                version,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=300,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

            success = proc.returncode == 0
            if success:
                logger.info(f"Toolchain {version} installed")
            else:
                logger.error(f"Failed to install {version}: {stderr.decode()}")
            return success
        except Exception as e:
            logger.error(f"Error installing toolchain: {e}")
            return False

    @staticmethod
    async def cache_mathlib() -> bool:
        """
        Pre-cache Mathlib for faster builds.

        Returns:
            True if successful
        """
        logger.info("Caching Mathlib...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "lake",
                "exe",
                "cache",
                "get",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=600,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)

            success = proc.returncode == 0
            if success:
                logger.info("Mathlib cache updated")
            else:
                logger.warning(f"Cache update had issues: {stderr.decode()}")
            return success
        except Exception as e:
            logger.error(f"Error caching Mathlib: {e}")
            return False


# ============================================================================
# LakeProject
# ============================================================================


class LakeProject:
    """Manages a real Lean 4 Lake project with Mathlib support."""

    def __init__(self, project_dir: Path, config: LakeProjectConfig | None = None):
        """
        Initialize Lake project manager.

        Args:
            project_dir: Directory for the Lake project
            config: Project configuration (defaults to standard config)
        """
        self.project_dir = Path(project_dir)
        self.config = config or LakeProjectConfig()
        self.source_dir = self.project_dir / self.config.name
        self.root_file = self.source_dir / f"{self.config.name}.lean"
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if project has been initialized."""
        return self._initialized and (self.project_dir / "lakefile.lean").exists()

    @property
    def mathlib_available(self) -> bool:
        """Check if Mathlib is available in project."""
        return (self.project_dir / ".lake" / "build").exists()

    async def initialize(self) -> bool:
        """
        Create full Lake project structure with Mathlib dependency.

        Returns:
            True if successful
        """
        logger.info(f"Initializing Lake project at {self.project_dir}")

        try:
            # Create directories
            self.project_dir.mkdir(parents=True, exist_ok=True)
            self.source_dir.mkdir(parents=True, exist_ok=True)

            # Create lean-toolchain
            toolchain_path = self.project_dir / "lean-toolchain"
            toolchain_path.write_text(self.config.lean_version)
            logger.debug(f"Created lean-toolchain with {self.config.lean_version}")

            # Create lakefile.lean
            lakefile_path = self.project_dir / "lakefile.lean"
            lakefile_path.write_text(self._generate_lakefile())
            logger.debug("Created lakefile.lean")

            # Create root import file
            root_path = self.source_dir / f"{self.config.name}.lean"
            root_path.write_text(self._generate_root_import([]))
            logger.debug(f"Created root file {root_path}")

            # Run lake update
            logger.info("Running 'lake update' to fetch Mathlib (this may take a while)...")
            proc = await asyncio.create_subprocess_exec(
                "lake",
                "update",
                cwd=str(self.project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=600,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
                output = stdout.decode() + stderr.decode()

                if proc.returncode != 0:
                    logger.error(f"lake update failed: {output}")
                    return False

                logger.info("Mathlib dependencies fetched successfully")
                self._initialized = True
                return True
            except asyncio.TimeoutError:
                logger.error("lake update timed out")
                proc.kill()
                await proc.wait()
                return False
        except Exception as e:
            logger.error(f"Error initializing Lake project: {e}")
            return False

    async def build(self) -> tuple[bool, str]:
        """
        Build the Lake project.

        Returns:
            Tuple of (success, output_message)
        """
        if not self.is_initialized:
            return False, "Project not initialized"

        logger.info(f"Building Lake project {self.project_dir}")

        try:
            proc = await asyncio.create_subprocess_exec(
                "lake",
                "build",
                cwd=str(self.project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=self.config.timeout_seconds,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.timeout_seconds,
                )
                output = stdout.decode() + stderr.decode()

                if proc.returncode == 0:
                    logger.info("Build successful")
                    return True, output
                else:
                    logger.warning(f"Build failed: {output}")
                    return False, output
            except asyncio.TimeoutError:
                logger.error("Build timed out")
                proc.kill()
                await proc.wait()
                return False, "Build timed out"
        except Exception as e:
            logger.error(f"Error building project: {e}")
            return False, str(e)

    async def check_file(self, lean_file: Path) -> tuple[bool, list[str]]:
        """
        Check a single Lean file using lake env lean.

        Args:
            lean_file: Path to .lean file to check

        Returns:
            Tuple of (success, errors_list)
        """
        if not self.is_initialized:
            return False, ["Project not initialized"]

        logger.debug(f"Checking file {lean_file}")

        try:
            proc = await asyncio.create_subprocess_exec(
                "lake",
                "env",
                "lean",
                str(lean_file),
                cwd=str(self.project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=self.config.timeout_seconds,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.timeout_seconds,
                )
                output = stdout.decode() + stderr.decode()

                errors = []
                for line in output.split("\n"):
                    if "error" in line.lower() or "failed" in line.lower():
                        errors.append(line.strip())

                success = proc.returncode == 0
                return success, errors
            except asyncio.TimeoutError:
                logger.error("File check timed out")
                proc.kill()
                await proc.wait()
                return False, ["Verification timed out"]
        except Exception as e:
            logger.error(f"Error checking file: {e}")
            return False, [str(e)]

    async def add_file(self, name: str, content: str) -> Path:
        """
        Add a .lean file to the project.

        Args:
            name: File name (without .lean extension)
            content: File content

        Returns:
            Path to created file
        """
        file_path = self.source_dir / f"{name}.lean"
        file_path.write_text(content)
        logger.debug(f"Added file {file_path}")

        # Update root import
        files = [
            f.stem
            for f in self.source_dir.glob("*.lean")
            if f.stem != self.config.name
        ]
        root_content = self._generate_root_import(files)
        self.root_file.write_text(root_content)

        return file_path

    async def verify_theorem(
        self,
        statement: str,
        proof: str,
        imports: list[str] | None = None,
    ) -> LakeVerificationResult:
        """
        Verify a theorem by creating and compiling a temporary file.

        Args:
            statement: Theorem statement
            proof: Proof code
            imports: List of import statements

        Returns:
            Verification result
        """
        if imports is None:
            imports = []

        logger.debug(f"Verifying theorem: {statement[:50]}...")
        start_time = time.time()

        try:
            # Create temp file with unique name
            import uuid
            temp_name = f"verify_{uuid.uuid4().hex[:12]}"
            temp_file = await self.add_file(temp_name, "")

            # Build content
            import_lines = "\n".join(imports) if imports else ""
            full_content = f"""{import_lines}

{statement}
{proof}
"""

            temp_file.write_text(full_content)

            # Check file
            success, errors = await self.check_file(temp_file)

            # Analyze output
            has_sorry = "sorry" in proof.lower()
            compilation_time = time.time() - start_time

            # Clean up
            try:
                temp_file.unlink()
            except Exception:
                pass

            # Get Lean version
            elan_info = await ElanManager.check_installation()
            lean_version = elan_info.get("lean_version", "unknown")

            return LakeVerificationResult(
                success=success,
                has_sorry=has_sorry,
                errors=errors,
                warnings=[],
                compilation_time=compilation_time,
                lean_version=lean_version,
                mathlib_used=bool(imports),
            )
        except Exception as e:
            logger.error(f"Error verifying theorem: {e}")
            return LakeVerificationResult(
                success=False,
                has_sorry=False,
                errors=[str(e)],
                warnings=[],
                compilation_time=time.time() - start_time,
                lean_version="unknown",
                mathlib_used=False,
            )

    def _generate_lakefile(self) -> str:
        """Generate lakefile.lean with Mathlib dependency."""
        dependencies = [
            '    { name = "mathlib", url = "https://github.com/leanprover-community/mathlib4", '
            f'branch = "{self.config.mathlib_version}" }},'
        ]

        for dep in self.config.extra_dependencies:
            name = dep.get("name", "unknown")
            url = dep.get("url", "")
            branch = dep.get("branch", "main")
            dependencies.append(
                f'    {{ name = "{name}", url = "{url}", branch = "{branch}" }},'
            )

        deps_section = "\n".join(dependencies)

        auto_implicit_str = "true" if self.config.auto_implicit else "false"

        return f"""import Lake
open Lake DSL

package {self.config.name} {{
  version : String := "0.1.0"
  leanVersion : String := "{self.config.lean_version}"
  maxHeartbeats : UInt64 := {self.config.max_heartbeats}
  -- Lean language options
  moreLeanArgs := #["--autoImplicit={auto_implicit_str}"]
}}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4"
  branch "{self.config.mathlib_version}"
{f'{chr(10)}'.join([dep for dep in self.config.extra_dependencies])}

@[default_target]
lean_lib {self.config.name} {{
  roots := #[`{self.config.name}]
}}
"""

    def _generate_toolchain(self) -> str:
        """Generate lean-toolchain file."""
        return self.config.lean_version

    def _generate_root_import(self, files: list[str]) -> str:
        """Generate root import file that includes all project files."""
        imports = [f"import {self.config.name}.{f}" for f in sorted(files)]
        return "\n".join(imports) if imports else "-- Project root"


# ============================================================================
# LakeProjectPool
# ============================================================================


class LakeProjectPool:
    """Manages a pool of pre-initialized Lake projects for concurrent verification."""

    def __init__(self, pool_dir: Path, pool_size: int = 3):
        """
        Initialize project pool.

        Args:
            pool_dir: Directory for pool projects
            pool_size: Number of projects to maintain
        """
        self.pool_dir = Path(pool_dir)
        self.pool_size = pool_size
        self.projects: list[LakeProject] = []
        self.available: asyncio.Queue[LakeProject] = asyncio.Queue()
        self._lock = asyncio.Lock()

    async def initialize_pool(self, config: LakeProjectConfig | None = None) -> None:
        """
        Initialize all projects in the pool.

        Args:
            config: Project configuration for all pool projects
        """
        if config is None:
            config = LakeProjectConfig()

        logger.info(f"Initializing pool of {self.pool_size} Lake projects...")

        async with self._lock:
            self.pool_dir.mkdir(parents=True, exist_ok=True)

            # Create and initialize projects in parallel
            tasks = []
            for i in range(self.pool_size):
                project_dir = self.pool_dir / f"project_{i}"
                project = LakeProject(project_dir, config)
                self.projects.append(project)

                tasks.append(project.initialize())

            # Wait for all initializations
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log results
            success_count = sum(
                1 for r in results if isinstance(r, bool) and r
            )
            logger.info(
                f"Pool initialized: {success_count}/{self.pool_size} projects ready"
            )

            # Queue available projects
            for project in self.projects:
                if project.is_initialized:
                    await self.available.put(project)

    async def acquire(self) -> LakeProject:
        """
        Acquire an available project from the pool.

        Returns:
            An initialized LakeProject

        Raises:
            RuntimeError if pool not initialized
        """
        if self.available.empty() and not self.projects:
            raise RuntimeError("Pool not initialized")

        try:
            project = self.available.get_nowait()
            logger.debug("Acquired project from pool")
            return project
        except asyncio.QueueEmpty:
            logger.warning("Pool exhausted, waiting for project...")
            project = await asyncio.wait_for(self.available.get(), timeout=300)
            return project

    async def release(self, project: LakeProject) -> None:
        """
        Return a project to the pool and clean up temp files.

        Args:
            project: Project to release
        """
        # Clean up temp files
        for temp_file in project.source_dir.glob("verify_*.lean"):
            try:
                temp_file.unlink()
            except Exception as e:
                logger.debug(f"Could not delete temp file: {e}")

        # Return to pool
        await self.available.put(project)
        logger.debug("Released project back to pool")

    async def verify_theorem(
        self,
        statement: str,
        proof: str,
        imports: list[str] | None = None,
    ) -> LakeVerificationResult:
        """
        Verify a theorem using a pooled project.

        Args:
            statement: Theorem statement
            proof: Proof code
            imports: List of import statements

        Returns:
            Verification result
        """
        project = await self.acquire()
        try:
            result = await project.verify_theorem(statement, proof, imports)
            return result
        finally:
            await self.release(project)


__all__ = [
    "LakeProjectConfig",
    "LakeVerificationResult",
    "LakeProject",
    "LakeProjectPool",
    "MathlibImportResolver",
    "ElanManager",
]
