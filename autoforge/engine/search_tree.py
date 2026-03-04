"""Search tree — branching, evaluation, and backtracking for agent decisions.

Implements a tree search mechanism inspired by SWE-Search, Tree-of-Thoughts,
and RethinkMCTS. Instead of a single linear agent loop, key decision points
generate multiple candidate approaches, evaluate them, and select the best.

This module provides:
  - BranchNode: A node in the search tree representing one approach
  - SearchTree: Manages branching, evaluation, and selection with MCTS
  - MCTSNode / MCTSSearchTree: Full MCTS with UCB1, backpropagation, and
    execution-feedback-guided thought refinement (RethinkMCTS)
  - BranchEvaluator: Lightweight LLM-based scoring of candidate approaches

The tree integrates with git worktrees for code-level branching and uses
Haiku-class models for cost-effective evaluation.

Reference papers:
  - SWE-Search (ICLR 2025): MCTS for software agents
  - Tree of Thoughts (NeurIPS 2023): Deliberate problem solving
  - CodeTree (NAACL 2025): Agent-guided tree search for code generation
  - RethinkMCTS (2024): Refining erroneous thoughts via execution feedback
  - RPM-MCTS (2025): Retrieval as process reward for code generation
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of a search tree node."""
    PENDING = "pending"        # Not yet explored
    EXPLORING = "exploring"    # Currently being executed
    EVALUATED = "evaluated"    # Scored but not selected
    SELECTED = "selected"      # Chosen as the winning branch
    PRUNED = "pruned"          # Discarded (low score or duplicate)
    FAILED = "failed"          # Execution failed


@dataclass
class BranchNode:
    """A node in the search tree representing one approach/strategy.

    Each node captures:
      - A description of the approach
      - Its evaluation score (0.0 - 1.0)
      - Parent node (for backtracking)
      - Children (for branching)
      - Metadata (files written, tokens used, etc.)
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    strategy: str = ""             # The actual approach/plan text
    status: NodeStatus = NodeStatus.PENDING
    score: float = 0.0             # 0.0 to 1.0 evaluation score
    confidence: float = 0.0        # Evaluator's confidence in the score
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    # Execution metadata
    turn_started: int = 0
    turn_ended: int = 0
    files_written: list[str] = field(default_factory=list)
    tokens_used: int = 0
    # Evaluation feedback
    evaluation_reason: str = ""
    # Git worktree branch (if using git isolation)
    git_branch: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "score": self.score,
            "confidence": self.confidence,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "depth": self.depth,
            "files_written": self.files_written,
            "evaluation_reason": self.evaluation_reason,
        }


@dataclass
class SearchTree:
    """Manages a tree of approach candidates with branching and backtracking.

    Usage in the agent loop:
        1. At a decision point, call `branch()` to generate candidates
        2. Call `evaluate()` to score each candidate
        3. Call `select_best()` to pick the winner
        4. If the winner fails later, call `backtrack()` to try the next best

    The tree keeps history for analysis and meta-learning.
    """
    nodes: dict[str, BranchNode] = field(default_factory=dict)
    root_id: str | None = None
    current_id: str | None = None
    max_depth: int = 3             # Max branching depth
    max_children: int = 3          # Max candidates per branch point
    # History for meta-learning
    history: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def create_root(self, description: str, strategy: str = "") -> BranchNode:
        """Create the root node (initial approach)."""
        node = BranchNode(
            description=description,
            strategy=strategy,
            status=NodeStatus.EXPLORING,
            depth=0,
        )
        self.nodes[node.id] = node
        self.root_id = node.id
        self.current_id = node.id
        return node

    def branch(
        self,
        parent_id: str,
        candidates: list[dict[str, str]],
    ) -> list[BranchNode]:
        """Create child nodes for multiple candidate approaches.

        Args:
            parent_id: The node to branch from.
            candidates: List of {"description": ..., "strategy": ...} dicts.

        Returns:
            List of new BranchNode instances.
        """
        parent = self.nodes.get(parent_id)
        if parent is None:
            raise ValueError(f"Parent node not found: {parent_id}")

        if parent.depth >= self.max_depth:
            logger.warning(f"Max depth ({self.max_depth}) reached, skipping branch")
            return []

        children = []
        for i, cand in enumerate(candidates[:self.max_children]):
            node = BranchNode(
                description=cand.get("description", f"Candidate {i+1}"),
                strategy=cand.get("strategy", ""),
                status=NodeStatus.PENDING,
                parent_id=parent_id,
                depth=parent.depth + 1,
            )
            self.nodes[node.id] = node
            parent.children_ids.append(node.id)
            children.append(node)

        self._log_event("branch", {
            "parent_id": parent_id,
            "num_candidates": len(children),
            "depth": parent.depth + 1,
        })
        return children

    def evaluate_node(
        self,
        node_id: str,
        score: float,
        confidence: float = 1.0,
        reason: str = "",
    ) -> None:
        """Record evaluation results for a node."""
        node = self.nodes.get(node_id)
        if node is None:
            return
        node.score = max(0.0, min(1.0, score))
        node.confidence = max(0.0, min(1.0, confidence))
        node.evaluation_reason = reason
        node.status = NodeStatus.EVALUATED
        self._log_event("evaluate", {
            "node_id": node_id,
            "score": node.score,
            "confidence": node.confidence,
            "reason": reason[:200],
        })

    def select_best(self, parent_id: str) -> BranchNode | None:
        """Select the highest-scoring child of a parent node.

        Uses score * confidence as the selection criterion.
        """
        parent = self.nodes.get(parent_id)
        if parent is None or not parent.children_ids:
            return None

        candidates = [
            self.nodes[cid] for cid in parent.children_ids
            if cid in self.nodes and self.nodes[cid].status == NodeStatus.EVALUATED
        ]

        if not candidates:
            return None

        # Sort by weighted score (score * confidence)
        candidates.sort(key=lambda n: n.score * n.confidence, reverse=True)
        best = candidates[0]

        # Check for diversity: if top 2 are too similar in score, prefer higher confidence
        if len(candidates) >= 2:
            gap = best.score - candidates[1].score
            if gap < 0.1 and candidates[1].confidence > best.confidence:
                best = candidates[1]

        best.status = NodeStatus.SELECTED
        self.current_id = best.id

        # Prune the rest
        for c in candidates:
            if c.id != best.id:
                c.status = NodeStatus.PRUNED

        self._log_event("select", {
            "selected_id": best.id,
            "score": best.score,
            "pruned": len(candidates) - 1,
        })
        return best

    def backtrack(self) -> BranchNode | None:
        """Backtrack to the next-best unexplored branch.

        Called when the current approach fails. Finds the highest-scoring
        pruned or pending node and switches to it.
        """
        current = self.nodes.get(self.current_id or "")
        if current is None:
            return None

        # Mark current as failed
        current.status = NodeStatus.FAILED

        # Find the best alternative: walk up to parent, find pruned siblings
        parent = self.nodes.get(current.parent_id or "")
        if parent is None:
            return None

        # Among siblings, find pruned nodes sorted by score
        siblings = [
            self.nodes[cid] for cid in parent.children_ids
            if cid in self.nodes and self.nodes[cid].status == NodeStatus.PRUNED
        ]
        siblings.sort(key=lambda n: n.score * n.confidence, reverse=True)

        if siblings:
            alt = siblings[0]
            alt.status = NodeStatus.SELECTED
            self.current_id = alt.id
            self._log_event("backtrack", {
                "from_id": current.id,
                "to_id": alt.id,
                "alt_score": alt.score,
            })
            return alt

        # No siblings — try backtracking further up
        if parent.parent_id:
            self.current_id = parent.parent_id
            return self.backtrack()

        self._log_event("backtrack_exhausted", {"from_id": current.id})
        return None

    def get_current_node(self) -> BranchNode | None:
        """Get the currently active node."""
        return self.nodes.get(self.current_id or "")

    def get_path_to_root(self, node_id: str | None = None) -> list[BranchNode]:
        """Get the path from a node back to the root."""
        nid = node_id or self.current_id
        path = []
        while nid and nid in self.nodes:
            path.append(self.nodes[nid])
            nid = self.nodes[nid].parent_id
        path.reverse()
        return path

    def summary(self) -> dict[str, Any]:
        """Return a summary of the search tree state."""
        status_counts: dict[str, int] = {}
        for node in self.nodes.values():
            status_counts[node.status.value] = status_counts.get(node.status.value, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "status_counts": status_counts,
            "max_depth_reached": max((n.depth for n in self.nodes.values()), default=0),
            "current_node": self.current_id,
            "history_length": len(self.history),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire tree for persistence."""
        return {
            "root_id": self.root_id,
            "current_id": self.current_id,
            "max_depth": self.max_depth,
            "max_children": self.max_children,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "history": self.history,
        }

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Record an event in history for meta-learning analysis."""
        self.history.append({
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
        })
        logger.debug(f"[SearchTree] {event_type}: {data}")


async def generate_candidates(
    llm: Any,
    task_description: str,
    context: str,
    num_candidates: int = 3,
    system_prompt: str = "",
) -> list[dict[str, str]]:
    """Use the LLM to generate multiple candidate approaches for a task.

    This is the "expansion" step of the search tree — analogous to MCTS expansion.
    Uses higher temperature for diversity.

    Args:
        llm: LLMRouter instance.
        task_description: What needs to be done.
        context: Project context (spec, architecture, existing files).
        num_candidates: How many approaches to generate.
        system_prompt: Optional system prompt override.

    Returns:
        List of {"description": ..., "strategy": ...} dicts.
    """
    from autoforge.engine.llm_router import TaskComplexity

    prompt = (
        f"You are generating {num_candidates} DIFFERENT approaches to solve a task.\n\n"
        f"## Task\n{task_description}\n\n"
        f"## Context\n{context}\n\n"
        f"## Instructions\n"
        f"Generate exactly {num_candidates} distinct approaches. Each approach should:\n"
        f"1. Use a genuinely different strategy (different libraries, patterns, or architectures)\n"
        f"2. Be concrete enough to implement\n"
        f"3. Have clear trade-offs\n\n"
        f"Output a JSON array:\n"
        f"```json\n"
        f"[\n"
        f'  {{"description": "Brief name", "strategy": "Detailed plan..."}},\n'
        f"  ...\n"
        f"]\n"
        f"```"
    )

    if not system_prompt:
        system_prompt = (
            "You are a software architect. Generate diverse, high-quality "
            "technical approaches. Prioritize practical, proven solutions over novelty."
        )

    try:
        response = await llm.call(
            complexity=TaskComplexity.STANDARD,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        # Parse JSON from response
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            candidates = json.loads(match.group(1).strip())
        else:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                candidates = json.loads(text[start:end + 1])
            else:
                logger.warning("Could not parse candidates from LLM response")
                return []

        return [
            {"description": c.get("description", ""), "strategy": c.get("strategy", "")}
            for c in candidates
            if isinstance(c, dict)
        ][:num_candidates]

    except Exception as e:
        logger.error(f"Failed to generate candidates: {e}")
        return []


async def evaluate_candidate(
    llm: Any,
    candidate: dict[str, str],
    task_description: str,
    context: str,
) -> tuple[float, float, str]:
    """Evaluate a single candidate approach using a lightweight LLM call.

    Returns (score, confidence, reason).

    This is the "simulation/evaluation" step — analogous to MCTS rollout.
    Uses a fast model (Haiku/mini) for cost efficiency.
    """
    from autoforge.engine.llm_router import TaskComplexity

    prompt = (
        f"Evaluate this approach for the given task.\n\n"
        f"## Task\n{task_description}\n\n"
        f"## Context\n{context[:2000]}\n\n"
        f"## Approach\n"
        f"**{candidate.get('description', '')}**\n"
        f"{candidate.get('strategy', '')}\n\n"
        f"## Evaluation Criteria\n"
        f"1. Feasibility (can this actually work?)\n"
        f"2. Simplicity (is this the simplest viable approach?)\n"
        f"3. Reliability (are the dependencies well-maintained?)\n"
        f"4. Compatibility (does this fit the project's tech stack?)\n\n"
        f"Output JSON: {{\"score\": 0.0-1.0, \"confidence\": 0.0-1.0, \"reason\": \"...\"}}"
    )

    try:
        response = await llm.call(
            complexity=TaskComplexity.STANDARD,
            system="You are a technical evaluator. Be concise and critical.",
            messages=[{"role": "user", "content": prompt}],
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        # Parse evaluation
        import re
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return (
                float(data.get("score", 0.5)),
                float(data.get("confidence", 0.5)),
                str(data.get("reason", "")),
            )

    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")

    return 0.5, 0.3, "Evaluation failed — default score"


# ──────────────────────────────────────────────
# RethinkMCTS — Execution-Feedback-Guided Tree Search
# ──────────────────────────────────────────────
#
# RethinkMCTS enhances standard tree search with:
#   1. UCB1 selection for explore-exploit balance
#   2. Value backpropagation from leaf to root
#   3. Thought refinement: when execution fails, the LLM re-examines
#      the reasoning chain and corrects erroneous thoughts
#   4. Process reward integration: step-level rewards guide expansion
#
# This subsumes the basic SearchTree for code generation tasks.
# The basic SearchTree is still useful for architecture exploration
# (where execution feedback isn't available).


import math


@dataclass
class MCTSNode:
    """A node in the MCTS search tree with UCB1 statistics.

    Enhanced over BranchNode with MCTS-specific fields:
      - visit_count: For UCB1 selection
      - value_sum: Accumulated value from simulations
      - prior: Initial value estimate from the LLM evaluation
      - thought_chain: Reasoning steps leading to this node
      - execution_feedback: Actual code execution results
      - is_refined: Whether this node's thought was corrected
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    strategy: str = ""
    thought_chain: list[str] = field(default_factory=list)
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    # MCTS statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.5             # Initial LLM estimate
    # Execution feedback (RethinkMCTS core)
    execution_feedback: str = ""
    execution_success: bool | None = None  # None = not tested
    process_reward: float = 0.0    # From ProcessRewardModel
    # Refinement tracking
    is_refined: bool = False
    original_strategy: str = ""    # Before refinement
    refinement_reason: str = ""
    # Status
    status: NodeStatus = NodeStatus.PENDING

    @property
    def q_value(self) -> float:
        """Average value (exploitation signal)."""
        if self.visit_count == 0:
            return self.prior
        return self.value_sum / self.visit_count

    def ucb1(self, parent_visits: int, exploration_constant: float = 1.41) -> float:
        """UCB1 score for selection.

        Balances exploitation (high q_value) with exploration
        (low visit_count relative to parent).
        """
        if self.visit_count == 0:
            return float("inf")  # Unvisited nodes have infinite UCB1

        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visit_count
        )
        return exploitation + exploration

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "strategy": self.strategy[:300],
            "thought_chain": self.thought_chain[-5:],
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "depth": self.depth,
            "visit_count": self.visit_count,
            "value_sum": self.value_sum,
            "q_value": self.q_value,
            "prior": self.prior,
            "execution_success": self.execution_success,
            "process_reward": self.process_reward,
            "is_refined": self.is_refined,
            "status": self.status.value,
        }


class MCTSSearchTree:
    """Full MCTS search tree with RethinkMCTS enhancements.

    The MCTS loop:
      1. SELECT: Use UCB1 to choose the most promising leaf node
      2. EXPAND: Generate child nodes (candidate approaches)
      3. SIMULATE: Evaluate via LLM + execution feedback
      4. BACKPROPAGATE: Update values from leaf to root
      5. REFINE: If execution failed, correct reasoning (RethinkMCTS)

    This is used during BUILD phase for code generation decisions.
    The basic SearchTree handles architecture exploration (SPEC/BUILD boundary).

    Usage:
        mcts = MCTSSearchTree(llm=router)
        root = mcts.create_root("Implement user auth module", context)
        for _ in range(mcts_iterations):
            leaf = mcts.select()
            children = await mcts.expand(leaf)
            for child in children:
                value = await mcts.simulate(child)
                mcts.backpropagate(child.id, value)
                if value < 0.3:
                    await mcts.refine(child)
        best = mcts.get_best_action()
    """

    def __init__(
        self,
        llm: Any = None,
        max_depth: int = 4,
        max_children: int = 3,
        exploration_constant: float = 1.41,
        max_iterations: int = 9,
    ) -> None:
        self.llm = llm
        self.nodes: dict[str, MCTSNode] = {}
        self.root_id: str | None = None
        self.max_depth = max_depth
        self.max_children = max_children
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        self._iteration_count = 0
        self.history: list[dict[str, Any]] = []

    def create_root(
        self,
        description: str,
        strategy: str = "",
        thought_chain: list[str] | None = None,
    ) -> MCTSNode:
        """Create the root node."""
        node = MCTSNode(
            description=description,
            strategy=strategy,
            thought_chain=thought_chain or [],
            depth=0,
            status=NodeStatus.EXPLORING,
        )
        self.nodes[node.id] = node
        self.root_id = node.id
        return node

    # ──────── Phase 1: SELECT ────────

    def select(self) -> MCTSNode:
        """Select the most promising leaf node using UCB1.

        Traverses from root to leaf, at each level choosing the child
        with the highest UCB1 score.
        """
        node_id = self.root_id
        node = self.nodes[node_id]

        while node.children_ids:
            children = [
                self.nodes[cid] for cid in node.children_ids
                if cid in self.nodes
            ]
            if not children:
                break

            # UCB1 selection
            best_child = max(
                children,
                key=lambda c: c.ucb1(node.visit_count, self.exploration_constant),
            )

            # If best child is unvisited, select it
            if best_child.visit_count == 0:
                return best_child

            node = best_child

        return node

    # ──────── Phase 2: EXPAND ────────

    async def expand(
        self,
        node: MCTSNode,
        task_description: str = "",
        context: str = "",
        num_children: int | None = None,
    ) -> list[MCTSNode]:
        """Generate child nodes for a leaf.

        Uses the LLM to propose diverse sub-approaches, considering
        the thought chain from root to this node.
        """
        if node.depth >= self.max_depth:
            return []

        num = num_children or self.max_children

        # Build thought chain context
        thought_ctx = self._get_thought_chain_context(node)

        candidates = await generate_candidates(
            self.llm,
            task_description or node.description,
            context + f"\n\nReasoning so far:\n{thought_ctx}",
            num_candidates=num,
        )

        children = []
        for cand in candidates:
            child = MCTSNode(
                description=cand.get("description", ""),
                strategy=cand.get("strategy", ""),
                thought_chain=node.thought_chain + [cand.get("description", "")],
                parent_id=node.id,
                depth=node.depth + 1,
                status=NodeStatus.PENDING,
            )
            self.nodes[child.id] = child
            node.children_ids.append(child.id)
            children.append(child)

        self._log_event("expand", {
            "node_id": node.id,
            "num_children": len(children),
            "depth": node.depth + 1,
        })
        return children

    # ──────── Phase 3: SIMULATE ────────

    async def simulate(
        self,
        node: MCTSNode,
        task_description: str = "",
        context: str = "",
        process_reward: float | None = None,
    ) -> float:
        """Simulate/evaluate a node.

        Combines LLM evaluation with optional process reward from CodePRM.
        Returns a value in [0, 1].
        """
        # If we have a process reward signal, use it
        if process_reward is not None:
            node.process_reward = process_reward

        # LLM evaluation
        candidate = {"description": node.description, "strategy": node.strategy}
        score, confidence, reason = await evaluate_candidate(
            self.llm,
            candidate,
            task_description or node.description,
            context,
        )

        node.prior = score
        node.status = NodeStatus.EVALUATED

        # Blend LLM score with process reward if available
        if process_reward is not None:
            # 60% LLM evaluation, 40% execution-grounded process reward
            value = 0.6 * score + 0.4 * process_reward
        else:
            value = score

        self._log_event("simulate", {
            "node_id": node.id,
            "score": score,
            "confidence": confidence,
            "process_reward": process_reward,
            "blended_value": value,
        })
        return value

    # ──────── Phase 4: BACKPROPAGATE ────────

    def backpropagate(self, node_id: str, value: float) -> None:
        """Propagate a simulation value from leaf to root.

        Updates visit counts and value sums along the path,
        enabling UCB1 to learn which branches are most promising.
        """
        current_id = node_id
        while current_id is not None:
            node = self.nodes.get(current_id)
            if node is None:
                break
            node.visit_count += 1
            node.value_sum += value
            current_id = node.parent_id

        self._log_event("backpropagate", {
            "from_node": node_id,
            "value": value,
        })

    # ──────── Phase 5: REFINE (RethinkMCTS) ────────

    async def refine_thought(
        self,
        node: MCTSNode,
        execution_feedback: str,
        task_description: str = "",
        context: str = "",
    ) -> MCTSNode | None:
        """RethinkMCTS core: refine erroneous thoughts using execution feedback.

        When a node's execution fails, instead of discarding it, we:
        1. Present the thought chain + execution error to the LLM
        2. Ask it to identify which reasoning step was wrong
        3. Generate a corrected thought and strategy
        4. Create a new "refined" child node with the correction

        This is what distinguishes RethinkMCTS from vanilla MCTS:
        instead of blind random rollouts, we use execution feedback
        to CORRECT the reasoning process.
        """
        from autoforge.engine.llm_router import TaskComplexity

        node.execution_feedback = execution_feedback
        node.execution_success = False

        # Build the thought chain for analysis
        thought_chain_text = "\n".join(
            f"  Step {i+1}: {thought}"
            for i, thought in enumerate(node.thought_chain)
        )

        prompt = (
            f"A code generation approach has failed. Analyze the reasoning "
            f"chain and identify which step was wrong.\n\n"
            f"## Task\n{task_description or node.description}\n\n"
            f"## Reasoning Chain (thought process)\n{thought_chain_text}\n\n"
            f"## Final Strategy\n{node.strategy[:500]}\n\n"
            f"## Execution Feedback (error)\n"
            f"```\n{execution_feedback[:1000]}\n```\n\n"
        )
        if context:
            prompt += f"## Project Context\n{context[:500]}\n\n"

        prompt += (
            f"## Instructions\n"
            f"1. Identify which reasoning step caused the failure\n"
            f"2. Explain why it was wrong\n"
            f"3. Propose a corrected approach\n\n"
            f"Output JSON:\n"
            f'{{"error_step": 1-N, "error_reason": "why it was wrong", '
            f'"corrected_description": "new approach name", '
            f'"corrected_strategy": "detailed corrected plan"}}'
        )

        try:
            response = await self.llm.call(
                complexity=TaskComplexity.STANDARD,
                system=(
                    "You are a debugging expert. Analyze the thought chain, "
                    "identify the erroneous step, and propose a correction. "
                    "Respond with JSON only."
                ),
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None

            data = json.loads(match.group())

            # Create refined node
            corrected = data.get("corrected_strategy", "")
            if not corrected:
                return None

            # Build corrected thought chain: take steps before the error,
            # then replace from the error point
            error_step = int(data.get("error_step", len(node.thought_chain))) - 1
            corrected_chain = node.thought_chain[:error_step] + [
                data.get("corrected_description", "corrected approach")
            ]

            refined = MCTSNode(
                description=data.get("corrected_description", "refined approach"),
                strategy=corrected,
                thought_chain=corrected_chain,
                parent_id=node.parent_id,  # Sibling of the failed node
                depth=node.depth,
                is_refined=True,
                original_strategy=node.strategy[:200],
                refinement_reason=data.get("error_reason", ""),
                status=NodeStatus.PENDING,
            )

            # Add as sibling (child of the same parent)
            parent = self.nodes.get(node.parent_id or "")
            if parent:
                self.nodes[refined.id] = refined
                parent.children_ids.append(refined.id)

            # Mark original as failed
            node.status = NodeStatus.FAILED

            self._log_event("refine", {
                "original_id": node.id,
                "refined_id": refined.id,
                "error_step": data.get("error_step"),
                "error_reason": data.get("error_reason", "")[:100],
            })

            logger.info(
                f"[RethinkMCTS] Refined {node.id} -> {refined.id}: "
                f"error at step {data.get('error_step')}"
            )
            return refined

        except Exception as e:
            logger.warning(f"[RethinkMCTS] Thought refinement failed: {e}")
            return None

    # ──────── MCTS Main Loop ────────

    async def run_search(
        self,
        task_description: str,
        context: str,
        process_reward_fn: Any | None = None,
    ) -> MCTSNode | None:
        """Run the full MCTS search loop.

        Args:
            task_description: What needs to be done
            context: Project context
            process_reward_fn: Optional async callable(node) -> float
                              from ProcessRewardModel for execution-grounded scoring

        Returns:
            The best action node, or None if search fails.
        """
        if not self.root_id:
            self.create_root(task_description)

        for iteration in range(self.max_iterations):
            self._iteration_count = iteration + 1

            # 1. SELECT
            leaf = self.select()

            # 2. EXPAND (if not at max depth)
            if leaf.depth < self.max_depth and leaf.visit_count > 0:
                children = await self.expand(leaf, task_description, context)
                if children:
                    leaf = children[0]  # Evaluate first child

            # 3. SIMULATE
            pr = None
            if process_reward_fn:
                try:
                    pr = await process_reward_fn(leaf)
                except Exception:
                    pass

            value = await self.simulate(leaf, task_description, context, pr)

            # 4. BACKPROPAGATE
            self.backpropagate(leaf.id, value)

            # 5. REFINE (RethinkMCTS) — if value is very low
            if value < 0.3 and leaf.execution_feedback:
                refined = await self.refine_thought(
                    leaf, leaf.execution_feedback, task_description, context,
                )
                if refined:
                    # Evaluate the refined node
                    ref_value = await self.simulate(
                        refined, task_description, context, pr,
                    )
                    self.backpropagate(refined.id, ref_value)

            logger.debug(
                f"[MCTS] Iteration {iteration + 1}/{self.max_iterations}: "
                f"node={leaf.id} value={value:.2f}"
            )

        return self.get_best_action()

    def get_best_action(self) -> MCTSNode | None:
        """Get the best action from root's children (most visited = most robust).

        MCTS best practice: select by visit count (robust), not by value
        (which can be noisy from few samples).
        """
        root = self.nodes.get(self.root_id or "")
        if root is None or not root.children_ids:
            return root

        children = [
            self.nodes[cid] for cid in root.children_ids
            if cid in self.nodes
        ]
        if not children:
            return root

        # Select by visit count (most explored = most confident)
        best = max(children, key=lambda c: c.visit_count)
        best.status = NodeStatus.SELECTED

        logger.info(
            f"[MCTS] Best action: '{best.description}' "
            f"(visits={best.visit_count}, q={best.q_value:.2f})"
        )
        return best

    # ──────── Utility Methods ────────

    def _get_thought_chain_context(self, node: MCTSNode) -> str:
        """Build a text summary of the thought chain from root to node."""
        path = []
        current = node
        while current:
            path.append(current)
            current = self.nodes.get(current.parent_id or "")
        path.reverse()

        lines = []
        for n in path:
            if n.description:
                prefix = "(refined) " if n.is_refined else ""
                lines.append(f"  -> {prefix}{n.description}")
        return "\n".join(lines)

    def inject_execution_feedback(
        self,
        node_id: str,
        feedback: str,
        success: bool,
    ) -> None:
        """Inject execution feedback from the build process.

        Called by the orchestrator when a builder produces code that
        can be tested. This feedback is used by RethinkMCTS to
        refine erroneous thoughts.
        """
        node = self.nodes.get(node_id)
        if node:
            node.execution_feedback = feedback
            node.execution_success = success
            if success:
                # Boost value for successful execution
                self.backpropagate(node_id, 0.9)
            else:
                # Low value for failed execution, triggering refinement
                self.backpropagate(node_id, 0.2)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the MCTS state."""
        nodes_by_status: dict[str, int] = {}
        for node in self.nodes.values():
            key = node.status.value
            nodes_by_status[key] = nodes_by_status.get(key, 0) + 1

        refined_count = sum(1 for n in self.nodes.values() if n.is_refined)

        return {
            "total_nodes": len(self.nodes),
            "iterations_completed": self._iteration_count,
            "max_iterations": self.max_iterations,
            "status_counts": nodes_by_status,
            "max_depth_reached": max(
                (n.depth for n in self.nodes.values()), default=0
            ),
            "total_visits": sum(n.visit_count for n in self.nodes.values()),
            "refined_thoughts": refined_count,
            "root_q_value": self.nodes[self.root_id].q_value if self.root_id else 0,
        }

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Record an event in history."""
        self.history.append({
            "type": event_type,
            "timestamp": time.time(),
            "iteration": self._iteration_count,
            "data": data,
        })
        logger.debug(f"[MCTS] {event_type}: {data}")
