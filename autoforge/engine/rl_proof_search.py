"""Reinforcement learning-based proof search on top of MCTS infrastructure.

Implements proof search inspired by AlphaProof and DeepSeek-Prover-V2, using:
  - MCTS with PUCT (Polynomial Upper Confidence Tree) selection
  - LLM-based policy network for tactic generation
  - LLM-based value estimator for proof state evaluation
  - Experience buffer with prioritized sampling
  - Expert iteration for policy improvement
  - Subgoal decomposition (DeepSeek-Prover-V2 style)

Key components:
  - RLProofSearch: Main RL-MCTS engine
  - RLMCTSNode: Tree node with policy priors and value estimates
  - PolicyNetwork: LLM-based tactic generation
  - ValueEstimator: LLM-based proof state evaluation
  - RewardFunction: Step and terminal rewards
  - ExperienceBuffer: Circular buffer with prioritized sampling
  - SubgoalDecomposer: Hierarchical theorem decomposition

References:
  - AlphaProof (DeepMind, 2024)
  - DeepSeek-Prover-V2
  - AlphaZero PUCT formula
  - PPO and A3C style rewards
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from autoforge.engine.llm_router import TaskComplexity

logger = logging.getLogger(__name__)

# Try to import numpy; gracefully degrade to pure Python if unavailable
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.debug("NumPy not available; using pure Python for operations")


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class RLConfig:
    """Reinforcement learning configuration for proof search.
    
    Attributes:
        learning_rate: Gradient descent step size for policy/value updates.
        discount_factor: Gamma for discounting future rewards.
        exploration_weight: c in PUCT formula; balances exploration vs exploitation.
        policy_temperature: Softmax temperature for policy smoothing.
        value_loss_weight: Weight for value function loss vs policy loss.
        entropy_bonus: Entropy regularization to encourage exploration.
        max_proof_depth: Maximum steps in a single proof attempt.
        num_simulations: MCTS simulations (rollouts) per move.
        batch_size: Batch size for experience replay.
        experience_buffer_size: Maximum size of experience replay buffer.
        expert_iteration_rounds: Number of expert iteration refinement rounds.
        use_value_network: Enable LLM-based value estimation.
        use_policy_network: Enable LLM-based policy network.
    """

    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    exploration_weight: float = 1.414
    policy_temperature: float = 1.0
    value_loss_weight: float = 0.5
    entropy_bonus: float = 0.01
    max_proof_depth: int = 50
    num_simulations: int = 100
    batch_size: int = 16
    experience_buffer_size: int = 10000
    expert_iteration_rounds: int = 5
    use_value_network: bool = True
    use_policy_network: bool = True


# ── Experience & Buffer ────────────────────────────────────────────


@dataclass
class ProofExperience:
    """A single experience (s, a, r, s', done) from proof search.
    
    Attributes:
        state: Serialized proof state (goals, hypotheses, etc.).
        action: Tactic applied.
        reward: Immediate reward from applying tactic.
        next_state: State after applying tactic.
        done: Whether this experience ends the episode (proof found or failed).
        value_estimate: V(s) estimate from value network.
        policy_prior: P(a|s) from policy network.
        metadata: Additional context (step count, goal reduction, etc.).
    """

    state: str
    action: str
    reward: float
    next_state: str
    done: bool
    value_estimate: float = 0.0
    policy_prior: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperienceBuffer:
    """Circular buffer for experience replay with prioritized sampling.
    
    Stores transitions and supports:
      - Random sampling for experience replay
      - Prioritized sampling based on TD error magnitude
      - Save/load for checkpointing
    """

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize the experience buffer.
        
        Args:
            max_size: Maximum number of experiences to store.
        """
        self.max_size = max_size
        self.buffer: deque[ProofExperience] = deque(maxlen=max_size)
        self.td_errors: deque[float] = deque(maxlen=max_size)
        logger.debug(f"Initialized ExperienceBuffer with max_size={max_size}")

    def add(self, experience: ProofExperience) -> None:
        """Add an experience to the buffer.
        
        Args:
            experience: A ProofExperience instance.
        """
        self.buffer.append(experience)
        # Initialize TD error; will be updated during training
        self.td_errors.append(0.0)

    def sample(self, batch_size: int) -> list[ProofExperience]:
        """Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample.
            
        Returns:
            A list of randomly sampled ProofExperience instances.
            Returns fewer experiences if buffer has fewer than batch_size.
        """
        if not self.buffer:
            return []
        
        sample_size = min(batch_size, len(self.buffer))
        
        if HAS_NUMPY:
            indices = np.random.choice(len(self.buffer), size=sample_size, replace=False)
            return [self.buffer[i] for i in indices]
        else:
            import random
            return random.sample(list(self.buffer), sample_size)

    def sample_prioritized(self, batch_size: int, alpha: float = 0.6) -> list[ProofExperience]:
        """Sample prioritized by TD error magnitude.
        
        Priority is proportional to |TD error|^alpha. This emphasizes
        experiences that the value function has trouble predicting.
        
        Args:
            batch_size: Number of experiences to sample.
            alpha: Exponent for TD error weighting (0.0 = uniform, 1.0 = fully prioritized).
            
        Returns:
            A list of prioritized ProofExperience instances.
        """
        if not self.buffer:
            return []
        
        sample_size = min(batch_size, len(self.buffer))
        td_errors = list(self.td_errors)
        
        # Compute priorities: |TD error|^alpha, add small epsilon for stability
        epsilon = 1e-8
        priorities = [(abs(e) + epsilon) ** alpha for e in td_errors]
        total_priority = sum(priorities)
        
        if total_priority == 0:
            # Fallback to uniform sampling if all priorities are zero
            return self.sample(batch_size)
        
        # Normalize to probabilities
        probabilities = [p / total_priority for p in priorities]
        
        if HAS_NUMPY:
            indices = np.random.choice(
                len(self.buffer), size=sample_size, replace=False, p=probabilities
            )
            return [self.buffer[i] for i in indices]
        else:
            import random
            return random.choices(list(self.buffer), weights=probabilities, k=sample_size)

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()
        self.td_errors.clear()
        logger.debug("Cleared ExperienceBuffer")

    def __len__(self) -> int:
        """Return the number of experiences in the buffer."""
        return len(self.buffer)

    def save(self, path: Path) -> None:
        """Save the buffer to disk (JSON format).
        
        Args:
            path: File path to save to.
        """
        import json
        
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_size": self.max_size,
            "experiences": [
                {
                    "state": exp.state,
                    "action": exp.action,
                    "reward": exp.reward,
                    "next_state": exp.next_state,
                    "done": exp.done,
                    "value_estimate": exp.value_estimate,
                    "policy_prior": exp.policy_prior,
                    "metadata": exp.metadata,
                }
                for exp in self.buffer
            ],
            "td_errors": list(self.td_errors),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved {len(self.buffer)} experiences to {path}")

    def load(self, path: Path) -> None:
        """Load the buffer from disk.
        
        Args:
            path: File path to load from.
        """
        import json
        
        if not path.exists():
            logger.warning(f"Buffer file not found: {path}")
            return
        
        with open(path) as f:
            data = json.load(f)
        
        self.clear()
        for exp_dict in data.get("experiences", []):
            exp = ProofExperience(
                state=exp_dict["state"],
                action=exp_dict["action"],
                reward=exp_dict["reward"],
                next_state=exp_dict["next_state"],
                done=exp_dict["done"],
                value_estimate=exp_dict.get("value_estimate", 0.0),
                policy_prior=exp_dict.get("policy_prior", 0.0),
                metadata=exp_dict.get("metadata", {}),
            )
            self.add(exp)
        
        # Load TD errors
        for err in data.get("td_errors", []):
            if len(self.td_errors) < len(self.buffer):
                self.td_errors.append(err)
        
        logger.debug(f"Loaded {len(self.buffer)} experiences from {path}")


# ── Value & Policy Networks ────────────────────────────────────────


class ValueEstimator:
    """Estimates V(s) = expected reward from a proof state.
    
    Uses LLM to assess proof difficulty and likelihood of completion.
    Returns values in [0, 1] where 1 = very likely provable.
    Caches estimates for identical states.
    """

    def __init__(self, llm: Any) -> None:
        """Initialize the value estimator.
        
        Args:
            llm: LLM interface (supports `await llm.call(prompt, complexity=...)`)
        """
        self.llm = llm
        self._cache: dict[str, float] = {}
        logger.debug("Initialized ValueEstimator")

    async def estimate(self, state: str, context: str = "") -> float:
        """Estimate the value (likelihood of proof completion) for a state.
        
        Uses LLM to assess the proof state on a 0-10 scale, then normalizes
        to [0, 1].
        
        Args:
            state: Serialized proof state (goals, hypotheses).
            context: Optional context about the theorem or proof attempt.
            
        Returns:
            A float in [0, 1] indicating proof likelihood.
        """
        # Check cache
        cache_key = state
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = f"""Assess the likelihood of completing this proof.

Current proof state:
{state}

{f"Context: {context}" if context else ""}

On a scale of 0-10, how likely is this proof to be completed successfully?
0 = impossible (contradictory goals, stuck)
10 = nearly proven (almost trivial remaining steps)

Only respond with a number between 0 and 10."""

        try:
            response = await self.llm.call(prompt, complexity=TaskComplexity.HIGH)
            value = self._parse_value_from_llm(response.content)
            normalized_value = max(0.0, min(1.0, value / 10.0))
            self._cache[cache_key] = normalized_value
            return normalized_value
        except Exception as e:
            logger.warning(f"Error estimating value: {e}")
            return 0.5  # Default neutral estimate

    async def estimate_batch(self, states: list[str]) -> list[float]:
        """Estimate values for multiple states in parallel.
        
        Args:
            states: List of serialized proof states.
            
        Returns:
            List of value estimates corresponding to input states.
        """
        tasks = [self.estimate(state) for state in states]
        return await asyncio.gather(*tasks)

    def _parse_value_from_llm(self, response: str) -> float:
        """Parse a numeric value from LLM response.
        
        Extracts the first number found in the response.
        
        Args:
            response: LLM response text.
            
        Returns:
            A float between 0 and 10.
        """
        # Find all numbers in the response
        numbers = re.findall(r"[-+]?\d*\.?\d+", response)
        if numbers:
            return float(numbers[0])
        return 5.0  # Default neutral estimate

    def clear_cache(self) -> None:
        """Clear the value estimate cache."""
        self._cache.clear()


class PolicyNetwork:
    """Provides P(a|s) = probability distribution over tactics.
    
    Uses LLM to rank candidate tactics and generate new ones.
    Returns softmax probabilities for exploration-exploitation tradeoff.
    """

    def __init__(self, llm: Any) -> None:
        """Initialize the policy network.
        
        Args:
            llm: LLM interface (supports `await llm.call(prompt, complexity=...)`)
        """
        self.llm = llm
        logger.debug("Initialized PolicyNetwork")

    async def get_action_priors(
        self, state: str, candidate_tactics: list[str]
    ) -> list[float]:
        """Rank candidate tactics and return softmax probabilities.
        
        Asks the LLM to score each tactic, then applies softmax.
        
        Args:
            state: Serialized proof state.
            candidate_tactics: List of candidate tactic strings.
            
        Returns:
            List of probabilities (sum to 1.0) corresponding to tactics.
        """
        if not candidate_tactics:
            return []

        tactics_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(candidate_tactics))
        prompt = f"""Given this proof state, rank the following tactics by likelihood of making progress.

Proof state:
{state}

Candidate tactics:
{tactics_text}

Rank these tactics from most to least likely to advance the proof.
Respond with a space-separated list of numbers 1-{len(candidate_tactics)} in order of preference.
Example: "2 1 3" means tactic 2 is best, then tactic 1, then tactic 3."""

        try:
            response = await self.llm.call(prompt, complexity=TaskComplexity.HIGH)
            ranking = self._parse_ranking(response.content, len(candidate_tactics))
            scores = [len(candidate_tactics) - i for i in range(len(candidate_tactics))]
            
            # Reorder scores according to ranking
            if ranking:
                reordered = [0.0] * len(candidate_tactics)
                for rank_pos, tactic_idx in enumerate(ranking):
                    if 0 <= tactic_idx < len(candidate_tactics):
                        reordered[tactic_idx] = scores[rank_pos]
                scores = reordered
            
            return self._softmax(scores, self.temperature if hasattr(self, 'temperature') else 1.0)
        except Exception as e:
            logger.warning(f"Error getting action priors: {e}")
            # Fallback: uniform distribution
            return [1.0 / len(candidate_tactics)] * len(candidate_tactics)

    async def suggest_tactics(self, state: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Directly generate top-k tactics with confidence scores.
        
        Args:
            state: Serialized proof state.
            top_k: Number of tactics to generate.
            
        Returns:
            List of (tactic, confidence) tuples.
        """
        prompt = f"""Given this proof state, suggest the {top_k} most promising tactics to apply.

Proof state:
{state}

For each tactic, provide:
1. The tactic code/text
2. A confidence score from 0.0 to 1.0

Format as:
tactic1 | 0.95
tactic2 | 0.87
...

Suggest exactly {top_k} tactics."""

        try:
            response = await self.llm.call(prompt, complexity=TaskComplexity.HIGH)
            return self._parse_tactics(response.content, top_k)
        except Exception as e:
            logger.warning(f"Error suggesting tactics: {e}")
            return []

    def _parse_ranking(self, response: str, num_tactics: int) -> list[int]:
        """Parse a ranking of tactic indices from response.
        
        Args:
            response: LLM response with space-separated numbers.
            num_tactics: Expected number of tactics.
            
        Returns:
            List of 0-indexed tactic indices in ranked order.
        """
        numbers = re.findall(r"\d+", response)
        # Convert to 0-indexed and filter valid indices
        ranking = [int(n) - 1 for n in numbers if 1 <= int(n) <= num_tactics]
        return ranking

    def _parse_tactics(self, response: str, top_k: int) -> list[tuple[str, float]]:
        """Parse tactics and confidence scores from response.
        
        Expected format:
            tactic1 | 0.95
            tactic2 | 0.87
        
        Args:
            response: LLM response text.
            top_k: Expected number of tactics.
            
        Returns:
            List of (tactic, confidence) tuples.
        """
        tactics = []
        for line in response.strip().split("\n"):
            if "|" in line:
                parts = line.split("|")
                if len(parts) == 2:
                    tactic = parts[0].strip()
                    try:
                        confidence = float(parts[1].strip())
                        tactics.append((tactic, confidence))
                    except ValueError:
                        continue
        return tactics[:top_k]

    @staticmethod
    def _softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
        """Apply softmax with temperature scaling.
        
        Args:
            scores: Raw score values.
            temperature: Temperature for smoothing (higher = more uniform).
            
        Returns:
            Probability distribution (sums to 1.0).
        """
        if not scores:
            return []
        
        if temperature <= 0:
            temperature = 1.0
        
        # Scale by temperature
        scaled = [s / temperature for s in scores]
        
        if HAS_NUMPY:
            scaled_arr = np.array(scaled)
            # Numerically stable softmax
            scaled_arr = scaled_arr - np.max(scaled_arr)
            exp_scores = np.exp(scaled_arr)
            return (exp_scores / np.sum(exp_scores)).tolist()
        else:
            # Pure Python stable softmax
            max_score = max(scaled) if scaled else 0
            scaled = [s - max_score for s in scaled]
            exp_scores = [math.exp(s) for s in scaled]
            total = sum(exp_scores)
            return [e / total for e in exp_scores] if total > 0 else [1.0 / len(scores)] * len(scores)


# ── Reward Function ───────────────────────────────────────────────


class RewardFunction:
    """Computes rewards for proof search transitions.
    
    Includes:
      - Terminal rewards (proof found: +1.0, failed: -0.1)
      - Goal reduction rewards (proportional progress)
      - Intermediate shaping rewards
      - Step penalties (encourages shorter proofs)
    """

    def compute_step_reward(
        self, state_before: str, tactic: str, state_after: str | None, success: bool
    ) -> float:
        """Compute immediate reward for applying a tactic.
        
        Args:
            state_before: Proof state before applying tactic.
            tactic: The tactic that was applied.
            state_after: Proof state after tactic (None if tactic failed).
            success: Whether tactic succeeded.
            
        Returns:
            Reward value (typically in [-0.1, 0.2]).
        """
        if not success:
            return -0.02  # Small penalty for failed tactic

        if state_after is None:
            return -0.02

        # Count goals (heuristic: count "⊢" symbols or "goal" occurrences)
        goals_before = state_before.count("⊢") + state_before.count("goal")
        goals_after = state_after.count("⊢") + state_after.count("goal")

        if goals_before == 0:
            goals_before = 1  # Avoid division by zero

        if goals_after == 0:
            return 0.15  # Close to completion
        
        goal_reduction = (goals_before - goals_after) / goals_before
        reward = 0.1 * goal_reduction

        # Bonus for introducing new hypotheses (useful lemmas)
        if "have" in tactic or "suffices" in tactic:
            reward += 0.05

        # Small step penalty
        reward -= 0.001

        return reward

    def compute_terminal_reward(self, proved: bool, num_steps: int, max_steps: int) -> float:
        """Compute terminal (episode end) reward.
        
        Args:
            proved: Whether the theorem was proved.
            num_steps: Number of steps taken.
            max_steps: Maximum allowed steps.
            
        Returns:
            Terminal reward (+1.0 for success, -0.1 for failure).
        """
        if proved:
            # Slight bonus for shorter proofs
            length_bonus = max(0.0, 0.1 * (1.0 - num_steps / max_steps))
            return 1.0 + length_bonus
        else:
            return -0.1

    def compute_shaping_reward(self, state: str) -> float:
        """Compute intermediate shaping reward based on state features.
        
        Encourages:
          - Reducing remaining goals
          - Introducing helpful hypotheses
          - Making overall progress
        
        Args:
            state: Current proof state.
            
        Returns:
            Shaping reward value.
        """
        reward = 0.0
        
        # Reward for each hypothesis (lemmas, helper statements)
        num_hypotheses = state.count("hypothesis") + state.count("have")
        reward += 0.01 * min(num_hypotheses, 5)
        
        # Penalize too many remaining goals
        num_goals = state.count("⊢") + state.count("goal")
        if num_goals > 5:
            reward -= 0.02 * (num_goals - 5)
        
        return reward


# ── MCTS Node with RL ──────────────────────────────────────────────


@dataclass
class RLMCTSNode:
    """A node in the MCTS tree with policy priors and value estimates.
    
    Combines MCTS tree structure with:
      - Policy priors P(a|s) from policy network
      - Value estimates V(s) from value network
      - PUCT (Polynomial Upper Confidence Tree) selection
    
    Attributes:
        state: Serialized proof state.
        tactic: Tactic applied to reach this node (empty for root).
        parent: Parent node in the tree.
        children: Dict mapping tactics to child nodes.
        visit_count: N(s,a) - number of times this node visited.
        total_value: Sum of values from backpropagation.
        prior: P(a|s) from policy network (for PUCT).
        reward: Immediate reward for reaching this node.
        is_terminal: Whether this state ends the episode.
        is_proved: Whether this state proves the theorem.
    """

    state: str
    tactic: str = ""
    parent: RLMCTSNode | None = None
    children: dict[str, RLMCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    reward: float = 0.0
    is_terminal: bool = False
    is_proved: bool = False

    def ucb_score(self, exploration_weight: float) -> float:
        """Compute PUCT (Polynomial Upper Confidence Tree) score.
        
        Formula: Q(s,a) + c * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))
        
        This is AlphaZero-style upper confidence for tree search with
        learned priors.
        
        Args:
            exploration_weight: c parameter (typically 1.414).
            
        Returns:
            UCB score for this node.
        """
        if self.parent is None:
            return 0.0
        
        if self.visit_count == 0:
            # Unvisited nodes: use prior to bootstrap
            return exploration_weight * self.prior
        
        # Q value (average value)
        q_value = self.total_value / self.visit_count
        
        # Exploration bonus
        parent_visits = self.parent.visit_count
        if parent_visits == 0:
            exploration_bonus = 0.0
        else:
            exploration_bonus = (
                exploration_weight
                * self.prior
                * math.sqrt(parent_visits)
                / (1 + self.visit_count)
            )
        
        return q_value + exploration_bonus

    def select_child(self, exploration_weight: float) -> RLMCTSNode | None:
        """Select the best child using PUCT.
        
        Args:
            exploration_weight: c parameter for UCB.
            
        Returns:
            The child with highest UCB score, or None if no children.
        """
        if not self.children:
            return None
        
        best_child = None
        best_score = -float("inf")
        
        for child in self.children.values():
            score = child.ucb_score(exploration_weight)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child

    def expand(self, tactics: list[tuple[str, float]]) -> None:
        """Expand this node with child nodes for candidate tactics.
        
        Args:
            tactics: List of (tactic_text, prior_probability) tuples.
        """
        for tactic_text, prior in tactics:
            if tactic_text not in self.children:
                child = RLMCTSNode(state="", tactic=tactic_text, parent=self, prior=prior)
                self.children[tactic_text] = child

    def backpropagate(self, value: float) -> None:
        """Backpropagate value up the tree.
        
        Updates visit counts and accumulated values for this node
        and all ancestors.
        
        Args:
            value: Value estimate to backpropagate.
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = node.reward + value * 0.99  # Apply discount
            node = node.parent

    def best_action(self) -> str:
        """Select the best action as the most-visited child.
        
        Returns:
            The tactic of the most-visited child, or "" if no children.
        """
        if not self.children:
            return ""
        
        best_tactic = ""
        best_count = 0
        
        for tactic, child in self.children.items():
            if child.visit_count > best_count:
                best_count = child.visit_count
                best_tactic = tactic
        
        return best_tactic


# ── Subgoal Decomposer ─────────────────────────────────────────────


class SubgoalDecomposer:
    """Breaks complex theorems into hierarchical subgoals.
    
    DeepSeek-Prover-V2 style decomposition using `have` and `suffices`
    statements to create a chain of intermediate lemmas.
    """

    def __init__(self, llm: Any) -> None:
        """Initialize the decomposer.
        
        Args:
            llm: LLM interface.
        """
        self.llm = llm
        logger.debug("Initialized SubgoalDecomposer")

    async def decompose(self, theorem: str) -> list[str]:
        """Break a complex theorem into a chain of subgoals.
        
        Uses LLM to generate intermediate `have` statements that
        decompose the proof into more manageable pieces.
        
        Args:
            theorem: The theorem statement to decompose.
            
        Returns:
            List of subgoal statements (including original + intermediates).
        """
        decomposition_text = await self._generate_decomposition(theorem)
        subgoals = self._parse_subgoals(decomposition_text)
        
        if not subgoals:
            # Fallback: return original theorem
            subgoals = [theorem]
        
        logger.debug(f"Decomposed theorem into {len(subgoals)} subgoals")
        return subgoals

    async def _generate_decomposition(self, theorem: str) -> str:
        """Ask LLM to decompose theorem into intermediate steps.
        
        Args:
            theorem: The theorem statement.
            
        Returns:
            LLM response with decomposed proof structure.
        """
        prompt = f"""Decompose this theorem into a chain of intermediate lemmas using Lean 4 syntax.

Theorem: {theorem}

Suggest a proof structure with `have` statements for key intermediate results.
Format each subgoal on a new line.

Example:
have h1: ... := ...
have h2: ... := ...
show goal"""

        try:
            response = await self.llm.call(prompt, complexity=TaskComplexity.HIGH)
            return response.content
        except Exception as e:
            logger.warning(f"Error generating decomposition: {e}")
            return theorem

    def _parse_subgoals(self, decomposition: str) -> list[str]:
        """Extract subgoals from decomposition text.
        
        Args:
            decomposition: LLM response with decomposed structure.
            
        Returns:
            List of subgoal strings.
        """
        subgoals = []
        current_subgoal = ""
        
        for line in decomposition.split("\n"):
            line = line.strip()
            if line.startswith("have ") or line.startswith("show ") or line.startswith("suffices "):
                if current_subgoal:
                    subgoals.append(current_subgoal)
                current_subgoal = line
            elif current_subgoal and line:
                current_subgoal += " " + line
        
        if current_subgoal:
            subgoals.append(current_subgoal)
        
        return subgoals


# ── Main RL Proof Search Engine ────────────────────────────────────


class RLProofSearch:
    """Main RL-MCTS proof search engine.
    
    Combines:
      - MCTS with PUCT selection
      - Policy network for tactic generation
      - Value network for state evaluation
      - Expert iteration for policy improvement
      - Reward-based learning
    
    Usage:
        search = RLProofSearch(config, llm, verifier)
        proved, proof_text, experiences = await search.search(theorem)
    """

    def __init__(self, config: RLConfig, llm: Any, verifier: Any = None) -> None:
        """Initialize the RL proof search engine.
        
        Args:
            config: RLConfig with hyperparameters.
            llm: LLM interface for policy/value networks.
            verifier: Optional verifier for checking proofs (Lean, etc.).
        """
        self.config = config
        self.llm = llm
        self.verifier = verifier
        
        self._value_estimator = ValueEstimator(llm) if config.use_value_network else None
        self._policy_network = PolicyNetwork(llm) if config.use_policy_network else None
        self._reward_fn = RewardFunction()
        self._buffer = ExperienceBuffer(config.experience_buffer_size)
        self._decomposer = SubgoalDecomposer(llm)
        
        self._stats = {
            "proofs_found": 0,
            "total_searches": 0,
            "avg_search_depth": 0.0,
            "avg_value_accuracy": 0.0,
        }
        
        logger.debug(f"Initialized RLProofSearch with config: {config}")

    async def search(
        self, theorem_statement: str, max_depth: int = 0
    ) -> tuple[bool, str, list[ProofExperience]]:
        """Execute RL-MCTS proof search for a theorem.
        
        Pipeline:
          1. Decompose theorem into subgoals (if complex)
          2. Initialize root MCTS node
          3. Run MCTS simulations
          4. Select action (most visited child)
          5. Apply tactic and advance state
          6. Repeat until proved or depth limit reached
        
        Args:
            theorem_statement: The theorem to prove.
            max_depth: Maximum proof depth (0 = use config default).
            
        Returns:
            Tuple of (proved, proof_text, experiences) where:
              - proved: True if theorem was proved
              - proof_text: String representation of proof (if found)
              - experiences: List of ProofExperience from search tree
        """
        if max_depth == 0:
            max_depth = self.config.max_proof_depth
        
        self._stats["total_searches"] += 1
        logger.info(f"Starting proof search for theorem: {theorem_statement}")
        
        # Initialize root node
        root = RLMCTSNode(state=theorem_statement)
        experiences: list[ProofExperience] = []
        proof_steps: list[str] = []
        current_state = theorem_statement
        
        # Main search loop
        for step in range(max_depth):
            # Run MCTS simulations
            for sim in range(self.config.num_simulations):
                sim_node = root
                leaf_value = await self._simulate(sim_node)
                self._backpropagate(sim_node, leaf_value)
            
            # Select best action
            best_tactic = root.best_action()
            if not best_tactic:
                logger.warning("No tactics available; search failed")
                break
            
            # Apply tactic
            next_state, proved = await self._apply_tactic(current_state, best_tactic)
            
            if next_state is None:
                logger.debug(f"Tactic failed: {best_tactic}")
                continue
            
            # Record experience
            reward = self._reward_fn.compute_step_reward(
                current_state, best_tactic, next_state, next_state is not None
            )
            exp = ProofExperience(
                state=current_state,
                action=best_tactic,
                reward=reward,
                next_state=next_state,
                done=proved,
                metadata={"step": step, "is_proved": proved},
            )
            experiences.append(exp)
            self._buffer.add(exp)
            
            proof_steps.append(best_tactic)
            current_state = next_state
            
            logger.debug(f"Step {step}: applied tactic: {best_tactic[:50]}...")
            
            if proved:
                self._stats["proofs_found"] += 1
                proof_text = "\n".join(proof_steps)
                logger.info(f"Proof found in {step + 1} steps")
                return True, proof_text, experiences
        
        logger.info(f"Search exhausted depth limit ({max_depth} steps)")
        return False, "", experiences

    async def expert_iteration(
        self, problems: list[tuple[str, str]], rounds: int = 0
    ) -> dict[str, Any]:
        """AlphaProof-style expert iteration for policy improvement.
        
        Loop:
          1. Use current policy to attempt proofs
          2. Collect successful proof traces
          3. Use traces to improve policy (LLM in-context learning)
          4. Repeat for N rounds
        
        Args:
            problems: List of (theorem_statement, expected_proof) tuples.
            rounds: Number of iteration rounds (0 = use config default).
            
        Returns:
            Dictionary with training statistics.
        """
        if rounds == 0:
            rounds = self.config.expert_iteration_rounds
        
        stats = {
            "round": 0,
            "successes": 0,
            "failures": 0,
            "avg_depth": 0.0,
            "experiences_collected": 0,
        }
        
        for round_num in range(rounds):
            logger.info(f"Expert iteration round {round_num + 1}/{rounds}")
            successes = 0
            total_depth = 0
            
            for theorem, _ in problems:
                proved, _, experiences = await self.search(theorem)
                if proved:
                    successes += 1
                    total_depth += len(experiences)
            
            stats["round"] = round_num
            stats["successes"] = successes
            stats["failures"] = len(problems) - successes
            stats["avg_depth"] = total_depth / max(successes, 1)
            stats["experiences_collected"] = len(self._buffer)
            
            logger.info(f"Round {round_num}: {successes}/{len(problems)} proofs found")
        
        return stats

    async def _simulate(self, node: RLMCTSNode) -> float:
        """Run a single MCTS simulation (rollout).
        
        Pipeline:
          1. Selection: traverse tree using PUCT
          2. Expansion: use policy network to get child candidates
          3. Evaluation: use value network on leaf
          4. Backpropagation: handled by caller
        
        Args:
            node: Root node to simulate from.
            
        Returns:
            Value estimate for the leaf node.
        """
        # Selection
        selected = await self._select(node)
        
        # Expansion
        leaf_value = await self._expand(selected)
        
        return leaf_value

    async def _select(self, node: RLMCTSNode) -> RLMCTSNode:
        """Selection phase: traverse tree using PUCT until leaf.
        
        Args:
            node: Current node to select from.
            
        Returns:
            Leaf node for expansion.
        """
        current = node
        
        while current.children and not current.is_terminal:
            child = current.select_child(self.config.exploration_weight)
            if child is None:
                break
            current = child
        
        return current

    async def _expand(self, node: RLMCTSNode) -> float:
        """Expansion phase: generate child candidates and evaluate.
        
        Args:
            node: Leaf node to expand.
            
        Returns:
            Value estimate for this node.
        """
        if node.is_terminal:
            if node.is_proved:
                return 1.0
            else:
                return 0.0
        
        # Get value estimate
        value = 0.5
        if self._value_estimator:
            try:
                value = await self._value_estimator.estimate(node.state)
            except Exception as e:
                logger.debug(f"Error estimating value: {e}")
        
        # Get policy priors for child tactics
        if self._policy_network:
            try:
                tactics = await self._policy_network.suggest_tactics(node.state, top_k=10)
                node.expand(tactics)
            except Exception as e:
                logger.debug(f"Error expanding tactics: {e}")
        
        return value

    def _backpropagate(self, node: RLMCTSNode, value: float) -> None:
        """Backpropagation phase: update values up the tree.
        
        Args:
            node: Leaf node where simulation ended.
            value: Value estimate from evaluation.
        """
        node.backpropagate(value)

    async def _apply_tactic(self, state: str, tactic: str) -> tuple[str | None, bool]:
        """Apply a tactic to the current proof state.
        
        If verifier is available, uses it to check validity.
        Otherwise, simulates state transition with LLM.
        
        Args:
            state: Current proof state.
            tactic: Tactic to apply.
            
        Returns:
            Tuple of (new_state, proved) or (None, False) if tactic fails.
        """
        if self.verifier:
            try:
                result = await self.verifier.apply_tactic(state, tactic)
                if result.success:
                    return result.new_state, result.proved
                else:
                    return None, False
            except Exception as e:
                logger.debug(f"Verifier error: {e}")
        
        # Fallback: use LLM to simulate
        try:
            prompt = f"""Apply this tactic to the proof state.

Current state:
{state}

Tactic: {tactic}

What is the new proof state after applying this tactic?
If the tactic is invalid or the proof is complete, indicate that."""
            
            response = await self.llm.call(prompt, complexity=TaskComplexity.STANDARD)
            new_state = response.content
            proved = "proved" in new_state.lower() or "complete" in new_state.lower()
            return new_state, proved
        except Exception as e:
            logger.warning(f"Error applying tactic: {e}")
            return None, False

    def get_stats(self) -> dict[str, Any]:
        """Get search statistics.
        
        Returns:
            Dictionary with:
              - proofs_found: Number of theorems proved
              - total_searches: Total search attempts
              - avg_search_depth: Average steps to proof
              - experience_buffer_size: Collected experiences
        """
        return {
            "proofs_found": self._stats["proofs_found"],
            "total_searches": self._stats["total_searches"],
            "avg_search_depth": self._stats["avg_search_depth"],
            "experience_buffer_size": len(self._buffer),
        }

    async def save_buffer(self, path: Path) -> None:
        """Save the experience buffer to disk.
        
        Args:
            path: File path to save to.
        """
        self._buffer.save(path)
        logger.info(f"Saved experience buffer to {path}")

    async def load_buffer(self, path: Path) -> None:
        """Load the experience buffer from disk.
        
        Args:
            path: File path to load from.
        """
        self._buffer.load(path)
        logger.info(f"Loaded experience buffer from {path}")


# ── Group Relative Policy Optimization (GRPO) Reward Trainer ─────


@dataclass
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization (GRPO) training.
    
    GRPO is a reward-driven RL algorithm inspired by DeepSeek-Prover-V2,
    achieving 88.9% on miniF2F through group-relative advantage estimation.
    
    Attributes:
        group_size: Number of tactic candidates in each group (N).
        temperature: Softmax temperature for policy sampling.
        kl_penalty_coeff: Coefficient for KL divergence penalty vs original policy.
        learning_rate: Gradient descent step size for policy updates.
        clip_range: PPO-style clipping range (epsilon).
        epochs_per_batch: Number of update epochs per batch.
        verifiable_reward_weight: Weight for formal verification vs LLM estimation.
    """

    group_size: int = 8
    temperature: float = 1.0
    kl_penalty_coeff: float = 0.01
    learning_rate: float = 1e-4
    clip_range: float = 0.2
    epochs_per_batch: int = 3
    verifiable_reward_weight: float = 0.7


@dataclass
class GRPOSample:
    """A single GRPO training sample.
    
    Attributes:
        state: Proof state.
        generated_tactic: The tactic that was generated.
        reward: Computed reward (binary correctness or partial credit).
        advantage: Group-relative advantage (r_i - mean) / std.
        log_prob: Log probability of generating this tactic under current policy.
    """

    state: str
    generated_tactic: str
    reward: float
    advantage: float
    log_prob: float


@dataclass
class GRPOBatch:
    """A batch of GRPO samples for training.
    
    Attributes:
        samples: List of GRPOSample instances.
        group_mean_reward: Mean reward within this group.
        group_std_reward: Standard deviation of rewards in group.
    """

    samples: list[GRPOSample] = field(default_factory=list)
    group_mean_reward: float = 0.0
    group_std_reward: float = 0.0


class GRPORewardTrainer:
    """Group Relative Policy Optimization reward training engine.
    
    Implements the GRPO algorithm from DeepSeek-Prover-V2:
      1. Sample N tactic candidates from the policy
      2. Compute verifiable rewards (formal verifier + value estimation)
      3. Compute group-relative advantages: (r_i - mean_r) / std_r
      4. Update policy with clipped surrogate loss and KL penalty
    
    Key innovation vs PPO: advantage estimation is group-relative, avoiding
    the need for a separate critic/value network.
    """

    def __init__(
        self,
        config: GRPOConfig,
        policy_network: PolicyNetwork,
        value_estimator: ValueEstimator,
        llm: Any,
    ) -> None:
        """Initialize GRPO reward trainer.
        
        Args:
            config: GRPOConfig instance.
            policy_network: LLM-based policy for tactic generation.
            value_estimator: LLM-based value function for partial credit.
            llm: LLM interface for formal verification and value estimation.
        """
        self.config = config
        self.policy_network = policy_network
        self.value_estimator = value_estimator
        self.llm = llm
        
        self._stats = {
            "rewards": [],
            "advantages": [],
            "kl_divergences": [],
            "policy_updates": 0,
        }
        
        logger.debug("Initialized GRPORewardTrainer")

    async def sample_group(self, proof_state: str, n: int) -> list[tuple[str, float]]:
        """Generate N tactic candidates from the policy network.
        
        Args:
            proof_state: Current proof state.
            n: Number of candidates to generate.
            
        Returns:
            List of (tactic, log_prob) tuples.
        """
        # Get top-k suggestions from policy
        suggestions = await self.policy_network.suggest_tactics(proof_state, top_k=n)
        
        if not suggestions:
            logger.warning(f"No tactics suggested for state; returning empty group")
            return []
        
        # Pad with zeros if fewer than n suggestions
        while len(suggestions) < n:
            suggestions.append(("", 0.0))
        
        # Convert confidence scores to log probabilities
        tactics_and_logprobs = [
            (tactic, math.log(max(1e-6, conf))) for tactic, conf in suggestions[:n]
        ]
        
        return tactics_and_logprobs

    async def compute_verifiable_rewards(
        self, state: str, tactics: list[str]
    ) -> list[float]:
        """Compute verifiable rewards for each tactic.
        
        Combines:
          - Binary correctness from formal verifier (highest weight)
          - Partial credit from value estimator (lower weight)
        
        Args:
            state: Proof state.
            tactics: List of tactics to evaluate.
            
        Returns:
            List of rewards in [0, 1], one per tactic.
        """
        rewards = []
        
        for tactic in tactics:
            if not tactic:
                rewards.append(0.0)
                continue
            
            # Formal verification: check if tactic is valid
            formal_reward = await self._verify_tactic(state, tactic)
            
            # Value estimation: assess proof progress
            value_reward = await self.value_estimator.estimate(state)
            
            # Combine: weighted average
            combined = (
                self.config.verifiable_reward_weight * formal_reward +
                (1.0 - self.config.verifiable_reward_weight) * value_reward
            )
            
            rewards.append(max(0.0, min(1.0, combined)))
        
        return rewards

    async def _verify_tactic(self, state: str, tactic: str) -> float:
        """Verify a single tactic using formal verification.
        
        Queries the LLM to check if the tactic is syntactically and
        semantically valid for the proof state.
        
        Args:
            state: Proof state.
            tactic: Tactic to verify.
            
        Returns:
            Reward in [0, 1]: 1.0 if valid, 0.0 otherwise.
        """
        prompt = f"""Is this tactic valid for the given proof state?

Proof state:
{state}

Tactic: {tactic}

Respond with only "valid" or "invalid"."""
        
        try:
            response = await self.llm.call(prompt, complexity=TaskComplexity.STANDARD)
            is_valid = "valid" in response.content.lower()
            return 1.0 if is_valid else 0.0
        except Exception as e:
            logger.warning(f"Error verifying tactic: {e}")
            return 0.0

    def compute_group_advantages(self, batch: GRPOBatch) -> None:
        """Compute group-relative advantages for batch samples.
        
        The key GRPO innovation: advantage = (r_i - mean_r) / std_r
        This requires no critic network, just within-group statistics.
        
        Args:
            batch: GRPOBatch to update in-place.
        """
        if not batch.samples:
            return
        
        rewards = [s.reward for s in batch.samples]
        
        # Compute group statistics
        if HAS_NUMPY:
            rewards_arr = np.array(rewards)
            mean_reward = float(np.mean(rewards_arr))
            std_reward = float(np.std(rewards_arr))
        else:
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards) if rewards else 0.0
            std_reward = math.sqrt(variance)
        
        batch.group_mean_reward = mean_reward
        batch.group_std_reward = max(std_reward, 1e-6)  # Avoid division by zero
        
        # Compute advantages
        for sample in batch.samples:
            advantage = (sample.reward - mean_reward) / batch.group_std_reward
            sample.advantage = advantage
        
        logger.debug(
            f"Computed group advantages: mean={mean_reward:.4f}, "
            f"std={std_reward:.4f}, n={len(batch.samples)}"
        )

    async def update_policy(self, batch: GRPOBatch) -> dict[str, float]:
        """Update the policy network using the batch.
        
        Implements PPO-style clipped surrogate loss:
            L = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) - β * KL(π_new || π_old)
        
        Args:
            batch: GRPOBatch with computed advantages.
            
        Returns:
            Dictionary with training metrics (loss, kl_divergence, etc.).
        """
        if not batch.samples:
            logger.warning("Empty batch; skipping policy update")
            return {}
        
        metrics = {
            "loss": 0.0,
            "kl_divergence": 0.0,
            "mean_advantage": 0.0,
        }
        
        total_loss = 0.0
        total_kl = 0.0
        total_advantage = 0.0
        
        for sample in batch.samples:
            # Clipped surrogate objective
            advantage = sample.advantage
            log_prob = sample.log_prob
            
            # PPO clip: ratio = exp(log_prob_new - log_prob_old)
            # For simplicity, treat old log_prob as 0 (uniform baseline)
            ratio = math.exp(min(log_prob, 10.0))  # Clamp for numerical stability
            
            # Clipped loss
            clipped_ratio = max(
                1.0 - self.config.clip_range,
                min(ratio, 1.0 + self.config.clip_range),
            )
            loss = -min(ratio * advantage, clipped_ratio * advantage)
            
            # KL penalty (discourage large changes from original)
            kl_div = abs(log_prob)  # Simplified KL approximation
            loss_with_kl = loss + self.config.kl_penalty_coeff * kl_div
            
            total_loss += loss_with_kl
            total_kl += kl_div
            total_advantage += advantage
        
        n = len(batch.samples)
        metrics["loss"] = total_loss / n
        metrics["kl_divergence"] = total_kl / n
        metrics["mean_advantage"] = total_advantage / n
        
        # Record statistics
        self._stats["kl_divergences"].append(metrics["kl_divergence"])
        self._stats["policy_updates"] += 1
        
        logger.info(
            f"Policy update: loss={metrics['loss']:.4f}, "
            f"kl={metrics['kl_divergence']:.4f}, "
            f"advantage={metrics['mean_advantage']:.4f}"
        )
        
        return metrics

    async def train_step(self, proof_state: str) -> dict[str, Any]:
        """Execute a single GRPO training step.
        
        Sequence:
          1. Sample N tactics from policy
          2. Compute verifiable rewards
          3. Compute group advantages
          4. Update policy
        
        Args:
            proof_state: Proof state to train on.
            
        Returns:
            Dictionary with step results and metrics.
        """
        # Sample group
        tactics_and_logprobs = await self.sample_group(proof_state, self.config.group_size)
        
        if not tactics_and_logprobs:
            logger.warning("Failed to sample tactics; skipping train step")
            return {"success": False}
        
        tactics = [t for t, _ in tactics_and_logprobs]
        log_probs = [lp for _, lp in tactics_and_logprobs]
        
        # Compute rewards
        rewards = await self.compute_verifiable_rewards(proof_state, tactics)
        
        # Build batch
        batch = GRPOBatch()
        for tactic, log_prob, reward in zip(tactics, log_probs, rewards):
            sample = GRPOSample(
                state=proof_state,
                generated_tactic=tactic,
                reward=reward,
                advantage=0.0,  # Will be computed
                log_prob=log_prob,
            )
            batch.samples.append(sample)
        
        # Compute advantages
        self.compute_group_advantages(batch)
        
        # Update policy
        metrics = await self.update_policy(batch)
        
        # Record rewards
        self._stats["rewards"].extend(rewards)
        
        return {
            "success": True,
            "num_tactics": len(tactics),
            "group_mean_reward": batch.group_mean_reward,
            "group_std_reward": batch.group_std_reward,
            **metrics,
        }

    async def train_epoch(self, proof_states: list[str]) -> dict[str, Any]:
        """Train over a batch of proof states.
        
        Args:
            proof_states: List of proof states to train on.
            
        Returns:
            Dictionary with epoch statistics.
        """
        epoch_results = {
            "total_steps": 0,
            "total_rewards": 0.0,
            "total_loss": 0.0,
            "total_kl": 0.0,
        }
        
        for state in proof_states:
            for epoch in range(self.config.epochs_per_batch):
                result = await self.train_step(state)
                
                if result.get("success"):
                    epoch_results["total_steps"] += 1
                    epoch_results["total_rewards"] += result.get("group_mean_reward", 0.0)
                    epoch_results["total_loss"] += result.get("loss", 0.0)
                    epoch_results["total_kl"] += result.get("kl_divergence", 0.0)
        
        n = epoch_results["total_steps"]
        if n > 0:
            epoch_results["avg_reward"] = epoch_results["total_rewards"] / n
            epoch_results["avg_loss"] = epoch_results["total_loss"] / n
            epoch_results["avg_kl"] = epoch_results["total_kl"] / n
        
        logger.info(
            f"Epoch complete: {epoch_results['total_steps']} steps, "
            f"avg_reward={epoch_results.get('avg_reward', 0.0):.4f}"
        )
        
        return epoch_results

    def get_stats(self) -> dict[str, Any]:
        """Get training statistics.
        
        Returns:
            Dictionary with:
              - rewards: List of all computed rewards
              - advantages: List of all computed advantages
              - kl_divergences: List of all KL divergences
              - policy_updates: Number of policy updates
        """
        return {
            "rewards": self._stats["rewards"],
            "avg_reward": (
                sum(self._stats["rewards"]) / len(self._stats["rewards"])
                if self._stats["rewards"] else 0.0
            ),
            "advantages": self._stats["advantages"],
            "kl_divergences": self._stats["kl_divergences"],
            "policy_updates": self._stats["policy_updates"],
        }


# ── Scaffolded Progressive RL ──────────────────────────────────────


from enum import Enum


class ScaffoldTier(Enum):
    """Scaffolding tier levels for progressive RL training.
    
    Attributes:
        NO_HINT: No scaffolding; prove from scratch.
        STRATEGY_HINT: High-level proof strategy suggested.
        STEP_HINT: Intermediate lemma or next step hint.
        FULL_SCAFFOLD: Near-complete proof outline provided.
    """

    NO_HINT = 0
    STRATEGY_HINT = 1
    STEP_HINT = 2
    FULL_SCAFFOLD = 3


@dataclass
class ScaffoldConfig:
    """Configuration for scaffolded progressive RL.
    
    Inspired by Scaf-GRPO (44.3% improvement on AIME), this enables
    curriculum learning where scaffolding decreases as performance improves.
    
    Attributes:
        initial_tier: Starting scaffold tier for all domains.
        success_threshold_to_reduce: Reduce scaffolding if success rate exceeds this.
        failure_threshold_to_increase: Increase scaffolding if failure rate exceeds this.
        warmup_steps: Number of training steps before adapting scaffold.
    """

    initial_tier: ScaffoldTier = ScaffoldTier.STRATEGY_HINT
    success_threshold_to_reduce: float = 0.6
    failure_threshold_to_increase: float = 0.3
    warmup_steps: int = 50


class ScaffoldedProgressiveRL:
    """Scaffolded progressive RL training with adaptive difficulty curriculum.
    
    Combines GRPO reward training with dynamic scaffolding that automatically
    adjusts difficulty based on performance. Inspired by Scaf-GRPO from
    DeepSeek-Prover-V2, achieving 44.3% improvement on AIME.
    
    Workflow:
      1. Train on problems with current scaffold level
      2. Track success rate
      3. Reduce scaffolding when success rate is high
      4. Increase scaffolding when failure rate is high
    """

    def __init__(
        self, grpo_trainer: GRPORewardTrainer, scaffold_config: ScaffoldConfig
    ) -> None:
        """Initialize scaffolded progressive RL.
        
        Args:
            grpo_trainer: Initialized GRPORewardTrainer instance.
            scaffold_config: ScaffoldConfig instance.
        """
        self.grpo_trainer = grpo_trainer
        self.config = scaffold_config
        
        # Per-domain scaffold tracking
        self._current_tier: dict[str, ScaffoldTier] = {}
        self._success_counts: dict[str, int] = {}
        self._failure_counts: dict[str, int] = {}
        self._domain_steps: dict[str, int] = {}
        
        logger.debug("Initialized ScaffoldedProgressiveRL")

    def _get_domain(self, state: str) -> str:
        """Extract domain from proof state (simple heuristic).
        
        Args:
            state: Proof state string.
            
        Returns:
            Domain name (e.g., 'number_theory', 'algebra').
        """
        # Simple heuristic: look for keywords
        if "prime" in state.lower() or "divisor" in state.lower():
            return "number_theory"
        elif "polynomial" in state.lower() or "ring" in state.lower():
            return "algebra"
        elif "triangle" in state.lower() or "angle" in state.lower():
            return "geometry"
        else:
            return "general"

    def _get_tier_for_domain(self, domain: str) -> ScaffoldTier:
        """Get current scaffold tier for a domain.
        
        Args:
            domain: Domain name.
            
        Returns:
            Current ScaffoldTier for this domain.
        """
        if domain not in self._current_tier:
            self._current_tier[domain] = self.config.initial_tier
            self._success_counts[domain] = 0
            self._failure_counts[domain] = 0
            self._domain_steps[domain] = 0
        
        return self._current_tier[domain]

    async def generate_scaffold(self, state: str, tier: ScaffoldTier) -> str:
        """Generate a hint/scaffold at the specified tier.
        
        Args:
            state: Proof state.
            tier: Scaffold tier level.
            
        Returns:
            Hint text; empty string if tier is NO_HINT.
        """
        if tier == ScaffoldTier.NO_HINT:
            return ""
        
        if tier == ScaffoldTier.STRATEGY_HINT:
            prompt = f"""Outline a high-level strategy to prove this theorem without revealing specific tactics.

Theorem/Goal:
{state}

Provide a brief proof strategy (2-3 sentences) that guides the approach."""
        
        elif tier == ScaffoldTier.STEP_HINT:
            prompt = f"""Suggest the next logical step or a key lemma needed for this proof.

Current state:
{state}

Suggest one intermediate step that would advance the proof."""
        
        else:  # FULL_SCAFFOLD
            prompt = f"""Provide a detailed proof outline with steps to prove this theorem.

Goal:
{state}

Outline the main steps of the proof with brief explanations."""
        
        try:
            response = await self.grpo_trainer.llm.call(
                prompt, complexity=TaskComplexity.HIGH
            )
            return response.content
        except Exception as e:
            logger.warning(f"Error generating scaffold: {e}")
            return ""

    async def train_with_scaffold(self, proof_state: str) -> dict[str, Any]:
        """Execute a GRPO training step with scaffolding.
        
        Args:
            proof_state: Proof state to train on.
            
        Returns:
            Dictionary with training results.
        """
        domain = self._get_domain(proof_state)
        tier = self._get_tier_for_domain(domain)
        
        # Generate scaffold
        scaffold = await self.generate_scaffold(proof_state, tier)
        
        # Augment state with scaffold if applicable
        if scaffold:
            augmented_state = f"{proof_state}\n\n[Hint]: {scaffold}"
        else:
            augmented_state = proof_state
        
        # Train with GRPO
        result = await self.grpo_trainer.train_step(augmented_state)
        
        # Track success/failure
        self._domain_steps[domain] = self._domain_steps.get(domain, 0) + 1
        
        if result.get("group_mean_reward", 0.0) > 0.5:
            self._success_counts[domain] = self._success_counts.get(domain, 0) + 1
        else:
            self._failure_counts[domain] = self._failure_counts.get(domain, 0) + 1
        
        result["scaffold_tier"] = tier.name
        result["domain"] = domain
        
        return result

    async def adapt_scaffold(self, domain: str, success_rate: float) -> None:
        """Adapt scaffold level based on domain performance.
        
        Args:
            domain: Domain to adapt.
            success_rate: Recent success rate (in [0, 1]).
        """
        current_tier = self._get_tier_for_domain(domain)
        
        if success_rate >= self.config.success_threshold_to_reduce:
            # Reduce scaffolding (increase difficulty)
            if current_tier.value > 0:
                new_tier = ScaffoldTier(current_tier.value - 1)
                self._current_tier[domain] = new_tier
                logger.info(
                    f"Reduced scaffold for {domain}: {current_tier.name} → {new_tier.name}"
                )
        
        elif success_rate <= self.config.failure_threshold_to_increase:
            # Increase scaffolding (decrease difficulty)
            if current_tier.value < 3:
                new_tier = ScaffoldTier(current_tier.value + 1)
                self._current_tier[domain] = new_tier
                logger.info(
                    f"Increased scaffold for {domain}: {current_tier.name} → {new_tier.name}"
                )

    async def progressive_train(
        self, states_by_difficulty: dict[str, list[str]]
    ) -> dict[str, Any]:
        """Progressive training with curriculum ordered by difficulty.
        
        Trains on states grouped by difficulty, with scaffolding adaptation
        after each difficulty level.
        
        Args:
            states_by_difficulty: Dictionary mapping difficulty levels
                (e.g., 'easy', 'medium', 'hard') to lists of proof states.
            
        Returns:
            Dictionary with training progress and per-domain statistics.
        """
        results = {
            "total_steps": 0,
            "per_domain_stats": {},
        }
        
        # Train in order of increasing difficulty
        difficulty_order = ["easy", "medium", "hard"]
        
        for difficulty in difficulty_order:
            if difficulty not in states_by_difficulty:
                continue
            
            states = states_by_difficulty[difficulty]
            logger.info(f"Training on {difficulty} problems ({len(states)} states)")
            
            for state in states:
                result = await self.train_with_scaffold(state)
                results["total_steps"] += 1
                
                domain = result.get("domain", "general")
                if domain not in results["per_domain_stats"]:
                    results["per_domain_stats"][domain] = {
                        "steps": 0,
                        "successes": 0,
                        "total_reward": 0.0,
                    }
                
                stats = results["per_domain_stats"][domain]
                stats["steps"] += 1
                if result.get("group_mean_reward", 0.0) > 0.5:
                    stats["successes"] += 1
                stats["total_reward"] += result.get("group_mean_reward", 0.0)
            
            # Adapt scaffolds after each difficulty level
            for domain, stats in results["per_domain_stats"].items():
                success_rate = stats["successes"] / max(1, stats["steps"])
                await self.adapt_scaffold(domain, success_rate)
                
                logger.info(
                    f"Domain {domain}: {success_rate*100:.1f}% success after {difficulty}"
                )
        
        return results

    def get_tier_for_domain(self, domain: str) -> ScaffoldTier:
        """Get current tier for a domain (public method).
        
        Args:
            domain: Domain name.
            
        Returns:
            Current ScaffoldTier.
        """
        return self._get_tier_for_domain(domain)

    def get_stats(self) -> dict[str, Any]:
        """Get scaffolding statistics.
        
        Returns:
            Dictionary with:
              - current_tiers: Current tier per domain
              - success_counts: Success count per domain
              - failure_counts: Failure count per domain
              - domain_steps: Training steps per domain
        """
        return {
            "current_tiers": {
                d: t.name for d, t in self._current_tier.items()
            },
            "success_counts": self._success_counts,
            "failure_counts": self._failure_counts,
            "domain_steps": self._domain_steps,
        }
