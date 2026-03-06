"""Pantograph REPL client for incremental Lean 4 proof interaction.

Pantograph (TACAS 2025) provides a machine-to-machine REPL interface for
Lean 4 that enables incremental tactic application without full recompilation.
This is significantly faster than the command-line Lake build approach.

Key capabilities:
  - Incremental tactic application (no full recompile per step)
  - Goal state inspection after each tactic
  - Backtracking to previous proof states
  - Environment loading (Mathlib, custom projects)
  - Proof state serialization for MCTS integration

References:
  - Pantograph: A Machine-to-Machine Interaction Interface for Lean 4 (TACAS 2025)
  - LeanDojo: Theorem Proving with Retrieval-Augmented Language Models (NeurIPS 2023)
  - Lean Copilot: LLMs as Copilots for Theorem Proving in Lean (NeurIPS 2024)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from autoforge.engine.runtime.commands import spawn_exec

logger = logging.getLogger(__name__)


@dataclass
class PantographConfig:
    """Configuration for Pantograph REPL client."""

    pantograph_path: str = "pantograph"
    """Path to the pantograph executable."""

    project_path: Optional[Path] = None
    """Path to the Lean project root. If None, use Mathlib."""

    timeout_seconds: int = 30
    """Timeout for individual REPL commands."""

    max_backtrack_depth: int = 50
    """Maximum proof tree depth to track for backtracking."""

    env_imports: list[str] = field(default_factory=lambda: ["Mathlib"])
    """List of imports to load on startup."""


@dataclass
class GoalState:
    """Represents the state of a proof goal."""

    goal_id: str
    """Unique identifier for this goal state."""

    hypotheses: list[str]
    """List of hypotheses in the context."""

    target: str
    """The target proposition to prove."""

    is_solved: bool = False
    """Whether this goal has been completely solved."""

    parent_id: Optional[str] = None
    """ID of the parent goal state (before tactic application)."""

    tactic_applied: Optional[str] = None
    """The tactic that led to this state from parent."""

    depth: int = 0
    """Depth in proof tree (0 = root)."""

    created_at: float = field(default_factory=time.time)
    """Timestamp when this state was created."""


@dataclass
class TacticResult:
    """Result of applying a tactic."""

    success: bool
    """Whether the tactic application succeeded."""

    new_goals: list[GoalState] = field(default_factory=list)
    """Goal states created after tactic application."""

    error_message: str = ""
    """Error message if tactic failed."""

    time_ms: float = 0.0
    """Time taken to apply the tactic."""

    proof_term: Optional[str] = None
    """Proof term if goal was solved."""


class ProofTree:
    """Tracks the tree of goal states with backtracking support."""

    def __init__(self, max_depth: int = 50):
        """Initialize the proof tree.

        Args:
            max_depth: Maximum depth to track before pruning.
        """
        self.max_depth = max_depth
        self.states: dict[str, GoalState] = {}
        self.root_id: Optional[str] = None
        self.children: dict[str, list[str]] = {}
        self.current_goals: set[str] = set()

    def add_state(self, state: GoalState) -> None:
        """Add a new state to the tree.

        Args:
            state: The goal state to add.
        """
        self.states[state.goal_id] = state

        if state.parent_id is None:
            self.root_id = state.goal_id
        else:
            if state.parent_id not in self.children:
                self.children[state.parent_id] = []
            self.children[state.parent_id].append(state.goal_id)

        if not state.is_solved:
            self.current_goals.add(state.goal_id)

    def get_state(self, goal_id: str) -> Optional[GoalState]:
        """Get a state by ID.

        Args:
            goal_id: The goal state ID.

        Returns:
            The goal state or None if not found.
        """
        return self.states.get(goal_id)

    def get_children(self, goal_id: str) -> list[GoalState]:
        """Get all child states of a given state.

        Args:
            goal_id: The parent goal ID.

        Returns:
            List of child goal states.
        """
        child_ids = self.children.get(goal_id, [])
        return [self.states[cid] for cid in child_ids if cid in self.states]

    def get_path_to_root(self, goal_id: str) -> list[GoalState]:
        """Get the path from a state to the root.

        Args:
            goal_id: The goal state ID.

        Returns:
            List of states from the given state to root (inclusive).
        """
        path = []
        current_id = goal_id
        while current_id is not None:
            state = self.get_state(current_id)
            if state is None:
                break
            path.append(state)
            current_id = state.parent_id
        return path

    def backtrack_to(self, goal_id: str) -> bool:
        """Backtrack to a previous state.

        Args:
            goal_id: The goal state ID to backtrack to.

        Returns:
            True if backtrack was successful, False otherwise.
        """
        if goal_id not in self.states:
            return False

        # Remove all states that are descendants of goal_id
        to_remove = set()

        def mark_descendants(gid: str) -> None:
            """Mark all descendants for removal."""
            for child_id in self.children.get(gid, []):
                to_remove.add(child_id)
                mark_descendants(child_id)

        mark_descendants(goal_id)

        for remove_id in to_remove:
            self.states.pop(remove_id, None)
            self.current_goals.discard(remove_id)

        return True

    def get_all_open_goals(self) -> list[GoalState]:
        """Get all currently open (unsolved) goals.

        Returns:
            List of unsolved goal states.
        """
        return [
            self.states[gid]
            for gid in self.current_goals
            if gid in self.states and not self.states[gid].is_solved
        ]

    def mark_solved(self, goal_id: str) -> None:
        """Mark a goal as solved.

        Args:
            goal_id: The goal state ID.
        """
        if goal_id in self.states:
            self.states[goal_id].is_solved = True
            self.current_goals.discard(goal_id)

    def is_complete(self) -> bool:
        """Check if the entire proof tree is complete.

        Returns:
            True if all goals are solved, False otherwise.
        """
        return len(self.current_goals) == 0


class PantographREPL:
    """Main REPL interface for Pantograph."""

    def __init__(self, config: PantographConfig):
        """Initialize Pantograph REPL client.

        Args:
            config: Pantograph configuration.
        """
        self.config = config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.proof_tree = ProofTree(config.max_backtrack_depth)
        self._running = False
        self._command_counter = 0
        self._response_lock = asyncio.Lock()
        self._proof_context_id: Optional[str] = None

    async def start(self) -> None:
        """Start the Pantograph subprocess.

        Raises:
            RuntimeError: If subprocess fails to start.
        """
        try:
            cwd = (self.config.project_path or Path(".")).resolve()
            self.process = await spawn_exec(
                [self.config.pantograph_path],
                cwd=cwd,
                stdin_pipe=True,
                stdout_pipe=True,
                stderr_pipe=True,
                label="pantograph.start",
            )
            self._running = True
            logger.info("Pantograph REPL started")

            # Load imports
            if self.config.env_imports:
                await self.load_environment(self.config.env_imports)

        except FileNotFoundError as e:
            raise RuntimeError(
                f"Pantograph executable not found: {self.config.pantograph_path}"
            ) from e

    async def stop(self) -> None:
        """Gracefully terminate the Pantograph subprocess."""
        if self.process is None:
            return

        self._running = False

        # Send exit command
        try:
            await self._send_command({"command": "exit"})
        except Exception:
            pass

        # Terminate subprocess
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

        logger.info("Pantograph REPL stopped")

    async def _send_command(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON command to Pantograph and read the response.

        Args:
            cmd: Command dictionary to send.

        Returns:
            Response dictionary from Pantograph.

        Raises:
            RuntimeError: If subprocess is not running or communication fails.
        """
        if self.process is None or not self._running:
            raise RuntimeError("Pantograph REPL is not running")

        async with self._response_lock:
            self._command_counter += 1
            cmd_id = self._command_counter

            # Add ID to command for tracking
            cmd["id"] = cmd_id

            try:
                # Send command
                cmd_json = json.dumps(cmd) + "\n"
                self.process.stdin.write(cmd_json.encode())
                await self.process.stdin.drain()

                # Read response with timeout
                response_json = await asyncio.wait_for(
                    self.process.stdout.readline(), timeout=self.config.timeout_seconds
                )

                if not response_json:
                    raise RuntimeError("Pantograph REPL closed unexpectedly")

                response = json.loads(response_json.decode())
                logger.debug(f"Command {cmd_id}: {response}")

                return response

            except asyncio.TimeoutError:
                logger.error(f"Pantograph command {cmd_id} timed out")
                raise RuntimeError("Pantograph REPL command timeout")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Pantograph response: {e}")
                raise RuntimeError("Invalid JSON response from Pantograph")

    async def load_environment(self, imports: list[str]) -> None:
        """Load imports into the Pantograph environment.

        Args:
            imports: List of module names to import.

        Raises:
            RuntimeError: If loading fails.
        """
        for module in imports:
            cmd = {"command": "load_environment", "module": module}
            response = await self._send_command(cmd)

            if not response.get("success", False):
                raise RuntimeError(
                    f"Failed to load environment {module}: "
                    f"{response.get('error', 'Unknown error')}"
                )

        logger.info(f"Loaded environments: {imports}")

    async def create_proof_context(self, theorem_statement: str) -> GoalState:
        """Start a new proof context.

        Args:
            theorem_statement: The theorem to prove (e.g., "theorem test : 1 = 1").

        Returns:
            The initial goal state.

        Raises:
            RuntimeError: If context creation fails.
        """
        cmd = {"command": "create_proof_context", "theorem": theorem_statement}
        response = await self._send_command(cmd)

        if not response.get("success", False):
            raise RuntimeError(
                f"Failed to create proof context: {response.get('error', 'Unknown error')}"
            )

        self._proof_context_id = response.get("context_id")

        # Parse initial goal
        goal_data = response.get("goal", {})
        initial_goal = GoalState(
            goal_id=goal_data.get("id", "goal_0"),
            hypotheses=goal_data.get("hypotheses", []),
            target=goal_data.get("target", ""),
            is_solved=False,
            parent_id=None,
            depth=0,
        )

        self.proof_tree.add_state(initial_goal)
        logger.info(f"Created proof context: {theorem_statement}")

        return initial_goal

    async def apply_tactic(self, goal_id: str, tactic: str) -> TacticResult:
        """Apply a single tactic to a goal.

        Args:
            goal_id: The goal state ID.
            tactic: The tactic to apply.

        Returns:
            Result of tactic application.
        """
        if not self._proof_context_id:
            return TacticResult(
                success=False, error_message="No active proof context"
            )

        start_time = time.time()

        cmd = {
            "command": "apply_tactic",
            "context_id": self._proof_context_id,
            "goal_id": goal_id,
            "tactic": tactic,
        }

        try:
            response = await self._send_command(cmd)
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return TacticResult(
                success=False, error_message=str(e), time_ms=elapsed_ms
            )

        elapsed_ms = (time.time() - start_time) * 1000

        if not response.get("success", False):
            return TacticResult(
                success=False,
                error_message=response.get("error", "Unknown error"),
                time_ms=elapsed_ms,
            )

        # Parse new goals
        new_goals = []
        for goal_data in response.get("goals", []):
            new_goal = GoalState(
                goal_id=goal_data.get("id"),
                hypotheses=goal_data.get("hypotheses", []),
                target=goal_data.get("target", ""),
                is_solved=goal_data.get("is_solved", False),
                parent_id=goal_id,
                tactic_applied=tactic,
                depth=self.proof_tree.get_state(goal_id).depth + 1
                if self.proof_tree.get_state(goal_id)
                else 1,
            )
            new_goals.append(new_goal)
            self.proof_tree.add_state(new_goal)

        # Update parent goal
        parent_state = self.proof_tree.get_state(goal_id)
        if parent_state:
            if len(new_goals) == 0:
                # Goal was solved
                self.proof_tree.mark_solved(goal_id)

        proof_term = response.get("proof_term")

        return TacticResult(
            success=True,
            new_goals=new_goals,
            time_ms=elapsed_ms,
            proof_term=proof_term,
        )

    async def apply_tactic_sequence(
        self, goal_id: str, tactics: list[str]
    ) -> list[TacticResult]:
        """Apply multiple tactics in sequence.

        Args:
            goal_id: The initial goal state ID.
            tactics: List of tactics to apply in sequence.

        Returns:
            List of TacticResult for each tactic.
        """
        results = []
        current_goal_id = goal_id

        for tactic in tactics:
            result = await self.apply_tactic(current_goal_id, tactic)
            results.append(result)

            if not result.success:
                logger.warning(f"Tactic failed: {tactic}")
                break

            # Move to first new goal
            if result.new_goals:
                current_goal_id = result.new_goals[0].goal_id
            else:
                # Goal was solved
                break

        return results

    async def get_current_goals(self) -> list[GoalState]:
        """Get all currently open goals.

        Returns:
            List of open goal states.
        """
        return self.proof_tree.get_all_open_goals()

    async def backtrack(self, goal_id: str) -> bool:
        """Backtrack to a previous proof state.

        Args:
            goal_id: The goal state ID to backtrack to.

        Returns:
            True if backtrack succeeded, False otherwise.
        """
        if not self._proof_context_id:
            return False

        cmd = {
            "command": "backtrack",
            "context_id": self._proof_context_id,
            "goal_id": goal_id,
        }

        try:
            response = await self._send_command(cmd)
            success = response.get("success", False)

            if success:
                self.proof_tree.backtrack_to(goal_id)
                logger.info(f"Backtracked to goal {goal_id}")

            return success
        except Exception as e:
            logger.error(f"Backtrack failed: {e}")
            return False

    async def check_proof_complete(self) -> bool:
        """Check if the entire proof is complete.

        Returns:
            True if all goals are solved, False otherwise.
        """
        return self.proof_tree.is_complete()

    async def get_proof_term(self) -> Optional[str]:
        """Extract the proof term if the proof is complete.

        Returns:
            The proof term as a string, or None if proof is incomplete.
        """
        if not self._proof_context_id:
            return None

        cmd = {"command": "get_proof_term", "context_id": self._proof_context_id}

        try:
            response = await self._send_command(cmd)

            if response.get("success", False):
                return response.get("proof_term")
        except Exception as e:
            logger.error(f"Failed to get proof term: {e}")

        return None

    async def __aenter__(self) -> PantographREPL:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse a Pantograph JSON response.

        Args:
            raw: Raw JSON response string.

        Returns:
            Parsed response dictionary.

        Raises:
            ValueError: If JSON is invalid.
        """
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")


class PantographProofSearch:
    """Integration of Pantograph REPL with proof search (BFS/DFS)."""

    def __init__(
        self,
        repl: PantographREPL,
        policy_network: Optional[Callable[[GoalState], list[str]]] = None,
        max_depth: int = 50,
    ):
        """Initialize proof search.

        Args:
            repl: Pantograph REPL instance.
            policy_network: Optional function to generate tactic suggestions from a goal.
            max_depth: Maximum proof tree depth.
        """
        self.repl = repl
        self.policy_network = policy_network
        self.max_depth = max_depth
        self.stats = {
            "goals_explored": 0,
            "tactics_tried": 0,
            "successful_tactics": 0,
            "search_time_ms": 0,
        }

    async def search(
        self, theorem: str, timeout_seconds: int = 60, use_dfs: bool = False
    ) -> Optional[str]:
        """Perform BFS/DFS proof search.

        Args:
            theorem: The theorem statement to prove.
            timeout_seconds: Timeout for the entire search.
            use_dfs: Use DFS instead of BFS.

        Returns:
            The proof term if found, None otherwise.
        """
        start_time = time.time()

        try:
            # Create proof context
            initial_goal = await self.repl.create_proof_context(theorem)
            self.stats["goals_explored"] = 1

            # Search
            if use_dfs:
                result = await self._dfs_search(
                    initial_goal.goal_id, timeout_seconds
                )
            else:
                result = await self._bfs_search(
                    initial_goal.goal_id, timeout_seconds
                )

            elapsed = (time.time() - start_time) * 1000
            self.stats["search_time_ms"] = elapsed

            if result:
                proof_term = await self.repl.get_proof_term()
                return proof_term

            return None

        except Exception as e:
            logger.error(f"Proof search failed: {e}")
            return None

    async def _bfs_search(
        self, initial_goal_id: str, timeout_seconds: int
    ) -> bool:
        """Breadth-first search for proof.

        Args:
            initial_goal_id: The initial goal ID.
            timeout_seconds: Timeout for search.

        Returns:
            True if proof found, False otherwise.
        """
        queue: asyncio.Queue[str] = asyncio.Queue()
        await queue.put(initial_goal_id)
        visited = {initial_goal_id}

        deadline = time.time() + timeout_seconds

        while not queue.empty():
            if time.time() > deadline:
                logger.warning("BFS proof search timed out")
                return False

            goal_id = await queue.get()
            self.stats["goals_explored"] += 1

            goal_state = self.repl.proof_tree.get_state(goal_id)
            if goal_state is None:
                continue

            # Check if solved
            if goal_state.is_solved or await self.repl.check_proof_complete():
                return True

            # Generate tactics
            tactics = self._get_tactics(goal_state)

            for tactic in tactics:
                if time.time() > deadline:
                    return False

                self.stats["tactics_tried"] += 1

                result = await self.repl.apply_tactic(goal_id, tactic)

                if result.success:
                    self.stats["successful_tactics"] += 1

                    for new_goal in result.new_goals:
                        if new_goal.goal_id not in visited:
                            visited.add(new_goal.goal_id)
                            await queue.put(new_goal.goal_id)

                    if await self.repl.check_proof_complete():
                        return True

        return False

    async def _dfs_search(
        self, goal_id: str, timeout_seconds: int
    ) -> bool:
        """Depth-first search for proof.

        Args:
            goal_id: Current goal ID.
            timeout_seconds: Timeout for search.

        Returns:
            True if proof found, False otherwise.
        """
        deadline = time.time() + timeout_seconds

        async def dfs_helper(gid: str, depth: int) -> bool:
            if time.time() > deadline:
                return False

            if depth > self.max_depth:
                return False

            self.stats["goals_explored"] += 1

            goal_state = self.repl.proof_tree.get_state(gid)
            if goal_state is None:
                return False

            if goal_state.is_solved or await self.repl.check_proof_complete():
                return True

            tactics = self._get_tactics(goal_state)

            for tactic in tactics:
                if time.time() > deadline:
                    return False

                self.stats["tactics_tried"] += 1

                result = await self.repl.apply_tactic(gid, tactic)

                if result.success:
                    self.stats["successful_tactics"] += 1

                    for new_goal in result.new_goals:
                        if await dfs_helper(new_goal.goal_id, depth + 1):
                            return True

                    if await self.repl.check_proof_complete():
                        return True

            return False

        return await dfs_helper(goal_id, 0)

    async def interactive_prove(
        self,
        theorem: str,
        tactic_generator: Callable[[GoalState], str],
        max_steps: int = 100,
    ) -> Optional[str]:
        """Interactive proof with external tactic generator (e.g., LLM).

        Args:
            theorem: The theorem statement to prove.
            tactic_generator: Function that takes a goal and returns a tactic string.
            max_steps: Maximum number of tactic steps.

        Returns:
            The proof term if found, None otherwise.
        """
        try:
            # Create proof context
            initial_goal = await self.repl.create_proof_context(theorem)

            current_goal_id = initial_goal.goal_id

            for step in range(max_steps):
                # Check if complete
                if await self.repl.check_proof_complete():
                    return await self.repl.get_proof_term()

                # Get current goal
                goal_state = self.repl.proof_tree.get_state(current_goal_id)
                if goal_state is None:
                    break

                # Generate tactic
                tactic = tactic_generator(goal_state)

                # Apply tactic
                result = await self.repl.apply_tactic(current_goal_id, tactic)

                if not result.success:
                    logger.warning(f"Tactic failed at step {step}: {tactic}")
                    break

                # Move to next goal
                if result.new_goals:
                    current_goal_id = result.new_goals[0].goal_id
                else:
                    # Goal solved
                    break

            if await self.repl.check_proof_complete():
                return await self.repl.get_proof_term()

            return None

        except Exception as e:
            logger.error(f"Interactive proof failed: {e}")
            return None

    def get_search_stats(self) -> dict[str, Any]:
        """Get search statistics.

        Returns:
            Dictionary of search statistics.
        """
        return self.stats.copy()

    def _get_tactics(self, goal: GoalState) -> list[str]:
        """Generate list of tactics to try for a goal.

        Args:
            goal: The goal state.

        Returns:
            List of tactic strings.
        """
        if self.policy_network:
            return self.policy_network(goal)

        # Default tactics
        return [
            "rfl",
            "simp",
            "omega",
            "ring",
            "norm_num",
            "exact?",
            "sorry",
        ]
