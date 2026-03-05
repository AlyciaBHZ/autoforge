"""Lean 4 Proof Search — MCTS, tactic generation, decomposition, and self-play.

Contains:
  - TacticGenerator: Multi-source tactic generation (ReProver + COPRA style)
  - MCTSProofSearch: Monte Carlo Tree Search over Lean tactic space
  - RecursiveProofDecomposer: Hilbert + DeepSeek-Prover-V2 style decomposition
  - ConjectureEngine: STP self-play conjecture generation and proving
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from autoforge.engine.provers.lean_core import (
    Conjecture,
    ProofAttempt,
    ProofSearchNode,
    ProofState,
    ProofStatus,
    ProofStep,
    TacticCandidate,
    TacticSource,
    LeanEnvironment,
    PantographREPL,
    LeanVerificationResult,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Adaptive Beam Search (Width Adaptation)
# ══════════════════════════════════════════════════════════════


class AdaptiveBeamSearch:
    """Beam search for proof search with adaptive width.

    Dynamically adjusts beam width based on search diversity:
      - Start with initial_width (typically 5-8)
      - If candidates significantly exceed beam capacity, expand width
      - If candidates are sparse, contract width to avoid waste
      - Cap at max_width to prevent memory explosion

    This reduces redundant exploration while maintaining breadth when diversity is high.
    """

    def __init__(self, initial_width: int = 5, max_width: int = 20) -> None:
        """Initialize adaptive beam search.

        Args:
            initial_width: Starting beam width (candidates to keep per level)
            max_width: Maximum width to prevent memory overflow
        """
        self.width = initial_width
        self.max_width = max_width
        self._success_rates: list[float] = []
        self._algorithm_ratio = 0.5

    async def search(
        self,
        initial_state: ProofState,
        expand_fn: Any,
        evaluate_fn: Any,
        max_depth: int = 50,
    ) -> list[ProofState]:
        """Run beam search with adaptive width.

        Algorithm:
          1. Initialize beam with root state
          2. For each depth level:
             - Expand all states in beam (generate children)
             - Score all children via evaluate_fn
             - Keep top-width candidates
             - Adapt width based on candidate diversity
          3. Return any completed proofs (all goals closed)

        Args:
            initial_state: Starting proof state
            expand_fn: Function(state) → list[child_states]
            evaluate_fn: Function(state) → float (quality score)
            max_depth: Maximum search depth

        Returns:
            List of completed proof states (empty if no proofs found)
        """
        beam = [(0.0, initial_state)]
        completed_proofs = []

        for depth in range(max_depth):
            candidates = []

            # Expand all states in current beam
            for score, state in beam:
                try:
                    children = await expand_fn(state) if asyncio.iscoroutinefunction(expand_fn) else expand_fn(state)
                    if not isinstance(children, list):
                        children = [children] if children else []

                    for child in children:
                        child_score = await evaluate_fn(child) if asyncio.iscoroutinefunction(evaluate_fn) else evaluate_fn(child)
                        candidates.append((child_score, child))
                except Exception as e:
                    logger.debug(f"[AdaptiveBeamSearch] Expand/evaluate failed: {e}")
                    continue

            if not candidates:
                break

            # Check for solutions in candidates
            for score, state in candidates:
                if state.goals == []:  # All goals closed
                    completed_proofs.append(state)

            # Select top-width candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[: self.width]

            # Adapt width based on diversity
            num_candidates = len(candidates)
            if num_candidates > self.width * 2:
                self.width = min(self.width + 1, self.max_width)
                logger.debug(f"[AdaptiveBeamSearch] Expanding width to {self.width}")
            elif num_candidates < self.width // 2:
                self.width = max(3, self.width - 1)
                logger.debug(f"[AdaptiveBeamSearch] Contracting width to {self.width}")

        # Track success rate for algorithm ratio
        found_solution = 1.0 if completed_proofs else 0.0
        self._success_rates.append(found_solution)
        if len(self._success_rates) > 100:
            self._success_rates = self._success_rates[-100:]

        if len(self._success_rates) > 0:
            self._algorithm_ratio = sum(self._success_rates) / len(self._success_rates)

        return completed_proofs

    def get_stats(self) -> dict[str, Any]:
        """Get search statistics."""
        return {
            "current_width": self.width,
            "max_width": self.max_width,
            "algorithm_ratio": self._algorithm_ratio,
        }


# ══════════════════════════════════════════════════════════════
# Tactic Database with BM25 Retrieval
# ══════════════════════════════════════════════════════════════


class TacticDatabase:
    """Index proven tactics by goal pattern for efficient retrieval.

    Maintains a memory-based database of successful (goal_pattern, tactic) pairs,
    indexed by BM25-style relevance scores for fast retrieval during proof search.

    This enables transfer learning: tactics that worked on similar goals before
    are suggested first for new goals.
    """

    def __init__(self) -> None:
        """Initialize tactic database."""
        self._tactics: list[dict[str, Any]] = []  # Each: {goal, tactic, success}
        self._idf: dict[str, float] = {}  # Inverse document frequency cache
        self._algorithm_ratio = 0.5

    def index(self, goal_pattern: str, tactic: str, success: bool) -> None:
        """Add a tactic application to the database.

        Args:
            goal_pattern: Goal string or pattern
            tactic: The tactic that was tried
            success: Whether the tactic succeeded
        """
        self._tactics.append({
            "goal": goal_pattern,
            "tactic": tactic,
            "success": success,
        })
        # Update IDF for new terms
        self._update_idf()

    def retrieve(self, goal: str, top_k: int = 5) -> list[str]:
        """BM25-style retrieval of relevant tactics.

        Scoring:
          1. Tokenize both goal and pattern
          2. Compute overlap (intersection of tokens)
          3. Weight by IDF + success rate of tactic
          4. Return top-k tactics

        Args:
            goal: Current proof goal
            top_k: Number of tactics to return

        Returns:
            List of tactic strings ranked by relevance
        """
        if not self._tactics:
            return []

        goal_tokens = set(goal.lower().split())
        scored = []

        for entry in self._tactics:
            if not entry["success"]:
                continue

            pattern_tokens = set(entry["goal"].lower().split())
            overlap = goal_tokens & pattern_tokens

            if not overlap:
                continue

            # BM25-style relevance
            jaccard = len(overlap) / (len(goal_tokens) + len(pattern_tokens) - len(overlap))
            score = jaccard

            # Bonus for IDF (rare terms are more informative)
            for token in overlap:
                if token in self._idf:
                    score += 0.1 * self._idf[token]

            scored.append((score, entry["tactic"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        tactics = [t for _, t in scored[:top_k]]

        # Track algorithm ratio based on retrieval success
        if tactics:
            self._algorithm_ratio = min(1.0, len(tactics) / top_k)

        return tactics

    def _update_idf(self) -> None:
        """Update IDF scores for terms in database."""
        if not self._tactics:
            return

        term_doc_count: dict[str, int] = {}
        total_docs = len(self._tactics)

        for entry in self._tactics:
            tokens = set(entry["goal"].lower().split())
            for token in tokens:
                term_doc_count[token] = term_doc_count.get(token, 0) + 1

        # IDF = log(N / df)
        for term, count in term_doc_count.items():
            self._idf[term] = math.log(total_docs / max(1, count))

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        successful = sum(1 for t in self._tactics if t["success"])
        return {
            "total_entries": len(self._tactics),
            "successful_tactics": successful,
            "unique_goals": len(set(t["goal"] for t in self._tactics)),
            "algorithm_ratio": self._algorithm_ratio,
        }

    def save(self, path: Path) -> None:
        """Save database to JSON file."""
        data = {
            "tactics": self._tactics,
            "idf": self._idf,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"[TacticDB] Saved {len(self._tactics)} entries to {path}")

    def load(self, path: Path) -> None:
        """Load database from JSON file."""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._tactics = data.get("tactics", [])
            self._idf = data.get("idf", {})
            logger.info(f"[TacticDB] Loaded {len(self._tactics)} entries from {path}")
        except Exception as e:
            logger.warning(f"[TacticDB] Failed to load: {e}")


# ══════════════════════════════════════════════════════════════
# Tactic Generator (ReProver + COPRA style)
# ══════════════════════════════════════════════════════════════


class TacticGenerator:
    """Generate Lean 4 tactics using LLM with retrieval augmentation.

    Combines:
      - Direct tactic generation (LLM predicts next tactic)
      - Retrieval-augmented generation (find relevant lemmas first)
      - Informal reasoning → tactic (Lean-STaR style interleaved thinking)
      - Failure dictionary (COPRA): track what didn't work to avoid repeats
    """

    # Common Lean 4 automation tactics to try first
    AUTO_TACTICS = [
        "simp", "omega", "norm_num", "ring", "linarith",
        "aesop", "decide", "exact?", "apply?",
        "tauto", "trivial", "contradiction",
    ]

    def __init__(self) -> None:
        self._failure_dict: dict[str, set[str]] = {}  # goal_hash → failed tactics
        self._success_cache: dict[str, str] = {}       # goal_hash → successful tactic
        self._lemma_index: list[dict[str, str]] = []   # Retrieved lemma library
        self._tactic_db = TacticDatabase()  # BM25-indexed tactic database

    async def generate_candidates(
        self,
        state: ProofState,
        theorem_context: str,
        llm: Any,
        *,
        num_candidates: int = 8,
        informal_hint: str = "",
    ) -> list[TacticCandidate]:
        """Generate ranked tactic candidates for a proof state.

        Strategy (Hilbert-inspired multi-source):
          1. Check cache for known solutions
          2. Try automation tactics (simp, omega, aesop, etc.)
          3. Tactic database retrieval (BM25 goal pattern matching)
          4. Retrieval-augmented: find relevant lemmas
          5. LLM informal reasoning → tactic (Lean-STaR)
          6. LLM direct tactic generation
          7. Filter out known failures
        """
        goal_hash = self._hash_goal(state)
        candidates: list[TacticCandidate] = []

        # 1. Cache hit
        if goal_hash in self._success_cache:
            candidates.append(TacticCandidate(
                tactic=self._success_cache[goal_hash],
                source=TacticSource.RETRIEVAL,
                confidence=0.95,
            ))

        # 2. Automation tactics (high confidence for simple goals)
        for tactic in self.AUTO_TACTICS:
            candidates.append(TacticCandidate(
                tactic=tactic,
                source=TacticSource.AESOP if tactic == "aesop"
                       else TacticSource.SIMP if tactic == "simp"
                       else TacticSource.OMEGA if tactic == "omega"
                       else TacticSource.LLM_DIRECT,
                confidence=0.3,
            ))

        # 3. Tactic database retrieval (BM25 on goal patterns)
        goal_text = " ".join(state.goals) if state.goals else ""
        db_tactics = self._tactic_db.retrieve(goal_text, top_k=3)
        for tactic in db_tactics:
            candidates.append(TacticCandidate(
                tactic=tactic,
                source=TacticSource.RETRIEVAL,
                confidence=0.5,
            ))

        # 4. Retrieval-augmented candidates (lemmas from library)
        if self._lemma_index:
            relevant = self._retrieve_lemmas(state, top_k=5)
            if relevant:
                candidates.extend(await self._retrieval_augmented_tactics(
                    state, relevant, llm,
                ))

        # 4. LLM informal reasoning → tactic (Lean-STaR style)
        if informal_hint:
            candidates.extend(await self._informal_to_tactic(
                state, informal_hint, theorem_context, llm,
            ))

        # 5. LLM direct tactic generation
        candidates.extend(await self._llm_direct_tactics(
            state, theorem_context, llm, num_candidates=num_candidates,
        ))

        # 6. Filter known failures
        failed = self._failure_dict.get(goal_hash, set())
        candidates = [c for c in candidates if c.tactic not in failed]

        # Deduplicate and sort by confidence
        seen: set[str] = set()
        unique = []
        for c in candidates:
            if c.tactic not in seen:
                seen.add(c.tactic)
                unique.append(c)
        unique.sort(key=lambda x: x.confidence, reverse=True)

        return unique[:num_candidates * 2]  # Return extra for search diversity

    async def _llm_direct_tactics(
        self,
        state: ProofState,
        theorem_context: str,
        llm: Any,
        num_candidates: int = 8,
    ) -> list[TacticCandidate]:
        """Generate tactics directly via LLM (ReProver style)."""
        from autoforge.engine.llm_router import TaskComplexity

        goals_text = "\n".join(f"⊢ {g}" for g in state.goals) if state.goals else state.context
        hyps_text = "\n".join(state.hypotheses) if state.hypotheses else "(none)"

        prompt = f"""You are an expert Lean 4 theorem prover. Given the current proof state,
suggest {num_candidates} distinct tactics to try next. Return ONLY a JSON array of tactic strings.

## Theorem Context
{theorem_context[:2000]}

## Current Proof State
### Hypotheses
{hyps_text}

### Goals
{goals_text}

### Depth: {state.depth}

## Instructions
- Suggest diverse tactics: automation (simp, omega, ring), structural (intro, apply, exact),
  case analysis (cases, induction, rcases), rewriting (rw, conv), and custom lemma applications
- Order by estimated likelihood of success
- Each tactic must be valid Lean 4 syntax
- If a goal looks like it needs a specific lemma, use `exact` or `apply` with the lemma name
- For equality goals, try `rfl`, `ext`, `funext`, or `congr`

Return JSON array: ["tactic1", "tactic2", ...]"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You are a Lean 4 tactic generator. Return ONLY valid JSON.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            # Parse JSON array
            tactics = self._parse_tactic_list(text)
            return [
                TacticCandidate(
                    tactic=t,
                    source=TacticSource.LLM_DIRECT,
                    confidence=max(0.1, 0.7 - i * 0.08),
                )
                for i, t in enumerate(tactics[:num_candidates])
            ]
        except Exception as e:
            logger.debug(f"[TacticGen] LLM direct failed: {e}")
            return []

    async def _informal_to_tactic(
        self,
        state: ProofState,
        informal_hint: str,
        theorem_context: str,
        llm: Any,
    ) -> list[TacticCandidate]:
        """Convert informal reasoning into Lean 4 tactics (Lean-STaR style).

        Key insight from Lean-STaR (ICLR 2025): interleaving natural language
        thinking with formal tactic steps significantly improves proving.
        """
        from autoforge.engine.llm_router import TaskComplexity

        goals_text = "\n".join(f"⊢ {g}" for g in state.goals) if state.goals else state.context

        prompt = f"""You are translating informal mathematical reasoning into Lean 4 tactics.

## Informal Proof Hint
{informal_hint}

## Current Goal
{goals_text}

## Task
Based on the informal reasoning, determine what the next formal proof step should be.
Think step by step about how the informal argument maps to Lean 4 tactics.

Return JSON:
{{
  "thinking": "your step-by-step reasoning about how to formalize this",
  "tactics": ["tactic1", "tactic2"]
}}"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You bridge informal math reasoning and Lean 4 formal proofs.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            data = self._extract_json(text)
            if data and "tactics" in data:
                thinking = data.get("thinking", "")
                return [
                    TacticCandidate(
                        tactic=t,
                        source=TacticSource.LLM_INFORMAL,
                        confidence=0.65,
                        informal_reasoning=thinking,
                    )
                    for t in data["tactics"][:4]
                ]
        except Exception as e:
            logger.debug(f"[TacticGen] Informal→tactic failed: {e}")
        return []

    async def _retrieval_augmented_tactics(
        self,
        state: ProofState,
        relevant_lemmas: list[dict[str, str]],
        llm: Any,
    ) -> list[TacticCandidate]:
        """Generate tactics using retrieved relevant lemmas (ReProver style)."""
        from autoforge.engine.llm_router import TaskComplexity

        lemma_text = "\n".join(
            f"- `{l['name']}`: {l.get('type', '')}" for l in relevant_lemmas[:10]
        )
        goals_text = "\n".join(f"⊢ {g}" for g in state.goals) if state.goals else state.context

        prompt = f"""Given these relevant lemmas from Mathlib/our library:
{lemma_text}

And the current goal:
{goals_text}

Suggest Lean 4 tactics that use these lemmas. Return JSON array: ["tactic1", ...]"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.STANDARD,
                system="You are a Lean 4 premise selector and tactic generator.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            tactics = self._parse_tactic_list(text)
            return [
                TacticCandidate(
                    tactic=t,
                    source=TacticSource.RETRIEVAL,
                    confidence=0.6,
                    retrieval_context=[l["name"] for l in relevant_lemmas],
                )
                for t in tactics[:4]
            ]
        except Exception as e:
            logger.debug(f"[TacticGen] Retrieval-augmented failed: {e}")
            return []

    def record_success(self, state: ProofState, tactic: str) -> None:
        """Cache successful tactic and add to database for transfer learning."""
        goal_hash = self._hash_goal(state)
        self._success_cache[goal_hash] = tactic
        self._failure_dict.pop(goal_hash, None)
        # Record in tactic database for future retrieval
        goal_text = " ".join(state.goals) if state.goals else "unknown"
        self._tactic_db.index(goal_text, tactic, success=True)

    def record_failure(self, state: ProofState, tactic: str) -> None:
        """Track failed tactic to avoid retrying (COPRA failure dictionary)."""
        goal_hash = self._hash_goal(state)
        if goal_hash not in self._failure_dict:
            self._failure_dict[goal_hash] = set()
        self._failure_dict[goal_hash].add(tactic)
        # Record failed attempt in database for negative evidence
        goal_text = " ".join(state.goals) if state.goals else "unknown"
        self._tactic_db.index(goal_text, tactic, success=False)

    def add_lemmas(self, lemmas: list[dict[str, str]]) -> None:
        """Add lemmas to the retrieval index."""
        self._lemma_index.extend(lemmas)

    def _retrieve_lemmas(self, state: ProofState, top_k: int = 5) -> list[dict[str, str]]:
        """Simple BM25-style retrieval of relevant lemmas."""
        if not self._lemma_index:
            return []

        query_words = set()
        for g in state.goals:
            query_words.update(re.findall(r'\w+', g.lower()))
        for h in state.hypotheses:
            query_words.update(re.findall(r'\w+', h.lower()))

        scored = []
        for lemma in self._lemma_index:
            text = f"{lemma.get('name', '')} {lemma.get('type', '')}".lower()
            lemma_words = set(re.findall(r'\w+', text))
            overlap = len(query_words & lemma_words)
            if overlap > 0:
                scored.append((overlap, lemma))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:top_k]]

    @staticmethod
    def _hash_goal(state: ProofState) -> str:
        """Hash proof state for caching."""
        content = "|".join(sorted(state.goals + state.hypotheses))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _parse_tactic_list(text: str) -> list[str]:
        """Extract list of tactic strings from LLM output."""
        # Try JSON array
        try:
            if "[" in text:
                json_str = text[text.index("["):text.rindex("]") + 1]
                items = json.loads(json_str)
                return [str(t).strip() for t in items if isinstance(t, str) and t.strip()]
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: line-by-line
        tactics = []
        for line in text.splitlines():
            line = line.strip().strip("-\u2022").strip()
            if line.startswith("`") and line.endswith("`"):
                line = line.strip("`")
            if line and not line.startswith("#") and not line.startswith("//"):
                tactics.append(line)
        return tactics[:12]

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Robustly extract JSON from LLM output."""
        if "{" not in text:
            return None
        try:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"\{[^{}]*\}", text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None


# ══════════════════════════════════════════════════════════════
# Tactic Executor Abstractions
# ══════════════════════════════════════════════════════════════


class TacticExecutor(Protocol):
    """Execution backend for applying tactics during search."""

    async def start(self, theorem_context: str) -> bool:
        ...

    async def apply(self, state: ProofState, tactic: str) -> ProofState | None:
        ...

    async def undo(self) -> ProofState:
        ...

    async def close(self) -> None:
        ...

    @property
    def backend_name(self) -> str:
        ...

    @property
    def is_formal(self) -> bool:
        ...


class HeuristicExecutor:
    """Heuristic fallback executor (non-formal)."""

    @property
    def backend_name(self) -> str:
        return "heuristic"

    @property
    def is_formal(self) -> bool:
        return False

    async def start(self, theorem_context: str) -> bool:
        return True

    async def apply(self, state: ProofState, tactic: str) -> ProofState | None:
        new_goals = list(state.goals)
        new_hyps = list(state.hypotheses)

        if tactic in ("simp", "norm_num", "omega", "ring", "decide", "trivial", "rfl"):
            if new_goals:
                new_goals = new_goals[1:]
        elif tactic.startswith("intro"):
            if new_goals:
                new_hyps.append(f"(introduced by {tactic})")
        elif tactic.startswith("cases") or tactic.startswith("induction"):
            if new_goals:
                new_goals = [f"case 1 of {new_goals[0]}", f"case 2 of {new_goals[0]}"] + new_goals[1:]

        return ProofState(
            goals=new_goals,
            hypotheses=new_hyps,
            context=f"After: {tactic}\n{state.context}",
            depth=state.depth + 1,
            parent_tactic=tactic,
        )

    async def undo(self) -> ProofState:
        return ProofState(goals=["[undo_not_supported]"], hypotheses=[])

    async def close(self) -> None:
        return None


class PantographExecutor:
    """Pantograph-backed executor with heuristic fallback."""

    def __init__(self, lean_env: LeanEnvironment, fallback: TacticExecutor | None = None) -> None:
        self._repl = PantographREPL(lean_env)
        self._fallback = fallback or HeuristicExecutor()
        self._formal = False

    @property
    def backend_name(self) -> str:
        return "pantograph" if self._formal else f"{self._fallback.backend_name}_fallback"

    @property
    def is_formal(self) -> bool:
        return self._formal

    async def start(self, theorem_context: str) -> bool:
        self._formal = await self._repl.start_session()
        if not self._formal:
            await self._fallback.start(theorem_context)
        return True

    async def apply(self, state: ProofState, tactic: str) -> ProofState | None:
        if self._formal:
            return await self._repl.send_tactic(tactic)
        return await self._fallback.apply(state, tactic)

    async def undo(self) -> ProofState:
        if self._formal:
            return await self._repl.undo()
        return await self._fallback.undo()

    async def close(self) -> None:
        if self._formal:
            await self._repl.close()
        await self._fallback.close()


# ══════════════════════════════════════════════════════════════
# MCTS Proof Search (DeepSeek-Prover-V1.5 style)
# ══════════════════════════════════════════════════════════════


class MCTSProofSearch:
    """Monte Carlo Tree Search over Lean tactic space.

    Inspired by DeepSeek-Prover-V1.5 (ICLR 2025):
      - Tree nodes are proof states
      - Actions are tactics
      - Terminal states: all goals closed (success) or dead end (failure)
      - Value: estimated proof completion probability

    Hybrid with iterative deepening:
      - Start with shallow search (depth 3-5)
      - Deepen promising branches
      - Prune branches with low value
    """

    UCB_C = 1.4            # Exploration constant
    MAX_DEPTH = 30          # Maximum proof depth
    MAX_CHILDREN = 8        # Max tactic candidates per node
    MIN_VISITS_EXPAND = 2   # Minimum visits before expanding

    def __init__(self, tactic_gen: TacticGenerator, executor: TacticExecutor | None = None) -> None:
        self._tactic_gen = tactic_gen
        self._executor: TacticExecutor = executor or HeuristicExecutor()
        self._stats = {"nodes_explored": 0, "proofs_found": 0, "backtracks": 0}
        self._last_search_formal = False
        self._last_backend_name = self._executor.backend_name

    async def search(
        self,
        root_state: ProofState,
        theorem_context: str,
        llm: Any,
        *,
        max_iterations: int = 200,
        informal_hint: str = "",
    ) -> list[ProofStep] | None:
        """Run MCTS to find a proof.

        Returns list of proof steps if successful, None otherwise.
        """
        root = ProofSearchNode(state=root_state)
        best_proof: list[ProofStep] | None = None

        await self._executor.start(theorem_context)
        self._last_search_formal = self._executor.is_formal
        self._last_backend_name = self._executor.backend_name

        try:
            for iteration in range(max_iterations):
                node = self._select(root)
                if node.is_terminal:
                    continue

                if node.depth >= self.MAX_DEPTH:
                    node.is_terminal = True
                    self._backpropagate(node, 0.0)
                    self._stats["backtracks"] += 1
                    continue

                if not node.children:
                    candidates = await self._tactic_gen.generate_candidates(
                        node.state, theorem_context, llm,
                        informal_hint=informal_hint if node.depth < 3 else "",
                    )
                    for candidate in candidates[:self.MAX_CHILDREN]:
                        next_state = await self._executor.apply(node.state, candidate.tactic)
                        if next_state is None:
                            continue
                        child = ProofSearchNode(
                            state=next_state,
                            tactic=candidate.tactic,
                            parent=node,
                            depth=node.depth + 1,
                            is_terminal=self._is_proof_complete(next_state),
                        )
                        node.children.append(child)

                    if not node.children:
                        node.is_terminal = True
                        self._backpropagate(node, 0.0)
                        self._stats["backtracks"] += 1
                        continue

                rollout_children = sorted(
                    node.children,
                    key=lambda c: (c.visits == 0, c.value / c.visits if c.visits else c.value),
                    reverse=True,
                )[: min(3, len(node.children))]

                for leaf in rollout_children:
                    value = await self._evaluate_state(leaf.state, llm)
                    if self._is_proof_complete(leaf.state):
                        leaf.is_terminal = True
                        value = 1.0
                        proof_steps = self._extract_proof_path(leaf)
                        if best_proof is None or len(proof_steps) < len(best_proof):
                            best_proof = proof_steps
                            self._stats["proofs_found"] += 1
                            logger.info(
                                f"[MCTS] Proof found at iteration {iteration}, depth {leaf.depth}, "
                                f"backend={self._last_backend_name}"
                            )
                    self._backpropagate(leaf, value)

                self._stats["nodes_explored"] += 1
        finally:
            await self._executor.close()

        return best_proof

    def _select(self, node: ProofSearchNode) -> ProofSearchNode:
        """Select leaf node using UCB1."""
        while node.children and not node.is_terminal:
            # UCB1 selection
            best_score = -float("inf")
            best_child = node.children[0]

            for child in node.children:
                if child.visits == 0:
                    return child  # Prioritize unexplored

                exploit = child.value / child.visits
                explore = self.UCB_C * math.sqrt(math.log(node.visits) / child.visits)
                score = exploit + explore

                if score > best_score:
                    best_score = score
                    best_child = child

            node = best_child
        return node

    async def _evaluate_state(self, state: ProofState, llm: Any) -> float:
        """Evaluate proof state — estimated probability of completion.

        Inspired by LeanProgress (arXiv 2502.17925):
        predict remaining proof distance from current state.
        """
        if not state.goals:
            return 1.0  # No goals = proof complete

        # Heuristic evaluation
        score = 0.5
        num_goals = len(state.goals)
        score -= min(0.3, num_goals * 0.05)  # Penalty for many open goals
        score -= min(0.2, state.depth * 0.01)  # Slight depth penalty

        # Bonus for simple-looking goals
        for goal in state.goals:
            if any(kw in goal.lower() for kw in ["true", "rfl", "= 0", "trivial"]):
                score += 0.1

        return max(0.0, min(1.0, score))

    def _is_proof_complete(self, state: ProofState) -> bool:
        """Check if all proof goals are closed."""
        return len(state.goals) == 0

    def _backpropagate(self, node: ProofSearchNode, value: float) -> None:
        """Backpropagate value up the tree."""
        current: ProofSearchNode | None = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent

    def _extract_proof_path(self, node: ProofSearchNode) -> list[ProofStep]:
        """Extract the sequence of proof steps from root to this node."""
        steps = []
        current: ProofSearchNode | None = node
        while current is not None and current.parent is not None:
            steps.append(ProofStep(
                state_before=current.parent.state,
                tactic_applied=current.tactic,
                state_after=current.state,
                success=True,
            ))
            current = current.parent
        steps.reverse()
        return steps

    @property
    def last_search_formal(self) -> bool:
        """Whether the latest search used a formal execution backend."""
        return self._last_search_formal

    @property
    def last_backend_name(self) -> str:
        """Name of backend used in latest search."""
        return self._last_backend_name

    def get_stats(self) -> dict[str, int | str | bool]:
        """Return search statistics."""
        stats: dict[str, int | str | bool] = dict(self._stats)
        stats["formal_backend"] = self._last_search_formal
        stats["backend_name"] = self._last_backend_name
        return stats


# ══════════════════════════════════════════════════════════════
# Recursive Decomposer (Hilbert + DeepSeek-Prover-V2 style)
# ══════════════════════════════════════════════════════════════


class RecursiveProofDecomposer:
    """Recursively decompose complex theorems into provable subgoals.

    Combines two key techniques:

    1. **Hilbert** (Apple, NeurIPS 2025): Four-component architecture
       - Informal reasoner: generates natural language proof sketch
       - Specialized prover: translates sketch into Lean tactics
       - Verifier: checks formal correctness
       - Retriever: finds relevant lemmas
       - On failure: decomposes into subgoals and recurses

    2. **DeepSeek-Prover-V2**: Recursive subgoal decomposition
       - LLM generates informal reasoning structure (chain-of-thought)
       - Extracts formal subgoals from the reasoning
       - Proves each subgoal bottom-up
       - Combines into full proof

    The key insight is that most theorems that stump direct proving
    become tractable when decomposed into 2-5 intermediate lemmas.
    """

    MAX_RECURSION_DEPTH = 5
    MAX_SUBGOALS = 8

    def __init__(
        self,
        tactic_gen: TacticGenerator,
        mcts: MCTSProofSearch,
        lean_env: LeanEnvironment | None = None,
    ) -> None:
        self._tactic_gen = tactic_gen
        self._mcts = mcts
        self._lean_env = lean_env or LeanEnvironment()
        self._decomposition_cache: dict[str, list[str]] = {}

    async def prove(
        self,
        statement: str,
        informal_statement: str,
        llm: Any,
        *,
        depth: int = 0,
    ) -> ProofAttempt:
        """Recursively attempt to prove a theorem.

        Pipeline (Hilbert-style):
          1. Generate informal proof sketch
          2. Translate to formal Lean 4 proof
          3. Verify with Lean
          4. If verification fails → decompose into subgoals
          5. Recursively prove each subgoal
          6. Combine subgoal proofs into complete proof
        """
        attempt = ProofAttempt(
            theorem_id=hashlib.sha256(statement.encode()).hexdigest()[:12],
            statement=statement,
            informal_statement=informal_statement,
            status=ProofStatus.PROVING,
        )

        # Step 1: Generate informal proof sketch
        informal_proof = await self._generate_informal_proof(
            statement, informal_statement, llm,
        )
        attempt.informal_proof = informal_proof

        # Step 2: Direct proof attempt (Draft-Sketch-Prove)
        lean_proof = await self._informal_to_formal(
            statement, informal_proof, llm,
        )

        if lean_proof:
            attempt.lean_proof = lean_proof
            await self._verify_attempt(
                attempt,
                theorem_context=statement,
                require_formal_backend=False,
                proof_origin="direct",
                execution_backend="direct",
                backend_is_formal=True,
            )
            attempt.attempts = 1
            if attempt.status == ProofStatus.PROVED:
                logger.info(f"[Decomposer] Direct proof formally verified for {attempt.theorem_id}")
                return attempt

        # Step 3: MCTS proof search
        root_state = ProofState(
            goals=[statement],
            hypotheses=[],
            context=statement,
        )
        mcts_proof = await self._mcts.search(
            root_state, statement, llm,
            max_iterations=100,
            informal_hint=informal_proof,
        )

        if mcts_proof:
            tactics = [step.tactic_applied for step in mcts_proof]
            attempt.lean_proof = self._assemble_tactic_proof(statement, tactics)
            attempt.steps = mcts_proof
            await self._verify_attempt(
                attempt,
                theorem_context=statement,
                require_formal_backend=True,
                backend_is_formal=self._mcts.last_search_formal,
                proof_origin="mcts",
                execution_backend=self._mcts.last_backend_name,
            )
            if attempt.status == ProofStatus.PROVED:
                logger.info(f"[Decomposer] MCTS proof formally verified for {attempt.theorem_id}")
                return attempt

        # Step 4: Recursive decomposition (if not too deep)
        if depth < self.MAX_RECURSION_DEPTH:
            logger.info(f"[Decomposer] Decomposing {attempt.theorem_id} (depth {depth})")
            subgoals = await self._decompose(statement, informal_proof, llm)

            if subgoals and len(subgoals) <= self.MAX_SUBGOALS:
                attempt.status = ProofStatus.DECOMPOSED
                attempt.subgoals = [sg["statement"] for sg in subgoals]

                subproofs = []
                all_proved = True

                for sg in subgoals:
                    sub_attempt = await self.prove(
                        sg["statement"],
                        sg.get("informal", ""),
                        llm,
                        depth=depth + 1,
                    )
                    subproofs.append(sub_attempt)
                    if sub_attempt.status != ProofStatus.PROVED:
                        all_proved = False

                if all_proved:
                    # Combine subgoal proofs
                    combined = await self._combine_proofs(
                        statement, subgoals, subproofs, llm,
                    )
                    if combined:
                        attempt.lean_proof = combined
                        await self._verify_attempt(
                            attempt,
                            theorem_context=statement,
                            require_formal_backend=False,
                            proof_origin="recursive",
                            execution_backend="recursive_composition",
                            backend_is_formal=True,
                        )
                        if attempt.status == ProofStatus.PROVED:
                            logger.info(f"[Decomposer] Recursive proof complete for "
                                        f"{attempt.theorem_id}")
                            return attempt

        # All attempts failed
        attempt.status = ProofStatus.FAILED
        attempt.attempts = depth + 1
        return attempt

    async def _verify_attempt(
        self,
        attempt: ProofAttempt,
        *,
        theorem_context: str,
        require_formal_backend: bool = True,
        backend_is_formal: bool = True,
        proof_origin: str = "",
        execution_backend: str = "",
    ) -> None:
        """Verify generated Lean proof and set strict proof status."""
        attempt.status = ProofStatus.FORMALIZED
        attempt.proof_origin = proof_origin
        attempt.execution_backend = execution_backend
        attempt.used_formal_backend = backend_is_formal

        if not attempt.lean_proof.strip():
            attempt.status = ProofStatus.FAILED
            attempt.error = "Empty Lean proof"
            return

        if require_formal_backend and not backend_is_formal:
            attempt.status = ProofStatus.FAILED
            attempt.verification_backend = "unavailable"
            attempt.error = (
                "Proof generated with non-formal backend; formal verification is required "
                "before marking as PROVED"
            )
            return

        lean_available = await self._lean_env.check_lean_installation()
        if not lean_available:
            attempt.status = ProofStatus.FAILED
            attempt.verification_backend = "unavailable"
            attempt.error = "Lean toolchain unavailable; cannot grant formal PROVED status"
            return

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            prefix=f"autoforge_proof_{attempt.theorem_id}_",
            dir=self._lean_env._workspace,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(f"{theorem_context} := by\n{attempt.lean_proof}\n")
            proof_file = Path(tmp.name)

        try:
            verification = await self._lean_env.verify_file(proof_file)
        finally:
            proof_file.unlink(missing_ok=True)

        attempt.verification_time = verification.execution_time
        attempt.verification_backend = verification.backend

        if not verification.is_formal:
            attempt.status = ProofStatus.FAILED
            attempt.error = "Non-formal verification result cannot produce PROVED"
            return

        if verification.success and verification.sorry_count == 0:
            attempt.status = ProofStatus.PROVED
            attempt.error = ""
        elif verification.success and verification.sorry_count > 0:
            attempt.status = ProofStatus.VERIFIED_WITH_SORRY
            attempt.error = f"Proof contains {verification.sorry_count} sorry placeholders"
        elif not verification.errors:
            attempt.status = ProofStatus.COMPILES
            attempt.error = "Compiled but not fully verified"
        else:
            attempt.status = ProofStatus.FAILED
            attempt.error = "; ".join(verification.errors[:3])

    async def _generate_informal_proof(
        self,
        statement: str,
        informal_statement: str,
        llm: Any,
    ) -> str:
        """Generate informal (natural language) proof sketch.

        This is the "Draft" step of Draft-Sketch-Prove.
        Also the informal reasoner component of Hilbert.
        """
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""You are a mathematician. Provide an informal proof sketch for this theorem.

## Theorem (Lean 4)
```lean
{statement}
```

{f"## Natural Language: {informal_statement}" if informal_statement else ""}

## Instructions
Write a clear, step-by-step informal proof. Be specific about:
- What technique to use (induction, contradiction, direct proof, etc.)
- Key intermediate steps
- Which known results or lemmas to apply
- Where case analysis is needed

Keep it concise but precise. This will be translated into Lean 4 tactics."""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You are an expert mathematician providing proof sketches.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text
            return text.strip()
        except Exception as e:
            logger.debug(f"[Decomposer] Informal proof generation failed: {e}")
            return ""

    async def _informal_to_formal(
        self,
        statement: str,
        informal_proof: str,
        llm: Any,
    ) -> str:
        """Translate informal proof into Lean 4 proof (Draft-Sketch-Prove "Sketch" step)."""
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""Translate this informal proof into a complete Lean 4 proof.

## Theorem
```lean
{statement}
```

## Informal Proof
{informal_proof[:3000]}

## Instructions
Write a complete Lean 4 proof using `by` tactic mode.
- Use standard Lean 4 tactics: intro, apply, exact, rw, simp, omega, ring, etc.
- Add `import Mathlib` if Mathlib lemmas are needed
- The proof must compile without `sorry`
- Be precise with names and types

Return ONLY the Lean 4 code block:
```lean
-- your proof here
```"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You are a Lean 4 proof engineer. Write compilable proofs.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            # Extract Lean code block
            match = re.search(r"```lean\s*\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""
        except Exception as e:
            logger.debug(f"[Decomposer] Formal translation failed: {e}")
            return ""

    async def _decompose(
        self,
        statement: str,
        informal_proof: str,
        llm: Any,
    ) -> list[dict[str, str]]:
        """Decompose theorem into subgoals (DeepSeek-Prover-V2 style)."""
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""Decompose this theorem into simpler intermediate lemmas/subgoals.

## Theorem
```lean
{statement}
```

## Informal Proof Sketch
{informal_proof[:2000]}

## Instructions
Break this into 2-5 intermediate lemmas that, when combined, prove the theorem.
Each subgoal should be:
- Simpler than the original
- Expressible as a valid Lean 4 statement
- Self-contained (minimal dependencies between subgoals)

Return JSON array:
[
  {{"statement": "lemma sub1 : ...", "informal": "why this is true"}},
  {{"statement": "lemma sub2 : ...", "informal": "why this is true"}}
]"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You decompose theorems into provable subgoals.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            if "[" in text:
                json_str = text[text.index("["):text.rindex("]") + 1]
                subgoals = json.loads(json_str)
                return [sg for sg in subgoals if isinstance(sg, dict) and "statement" in sg]
        except Exception as e:
            logger.debug(f"[Decomposer] Decomposition failed: {e}")
        return []

    async def _combine_proofs(
        self,
        statement: str,
        subgoals: list[dict[str, str]],
        subproofs: list[ProofAttempt],
        llm: Any,
    ) -> str:
        """Combine subgoal proofs into a complete proof."""
        from autoforge.engine.llm_router import TaskComplexity

        subgoal_text = ""
        for sg, sp in zip(subgoals, subproofs):
            subgoal_text += f"\n-- Subgoal: {sg['statement']}\n{sp.lean_proof}\n"

        prompt = f"""Combine these proved subgoals into a complete proof of the main theorem.

## Main Theorem
```lean
{statement}
```

## Proved Subgoals
{subgoal_text}

## Instructions
Write a single Lean 4 proof that uses `have` or `let` to introduce the subgoals,
then combines them to prove the main theorem. Return ONLY Lean code:
```lean
-- combined proof
```"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You combine subgoal proofs into complete Lean 4 proofs.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            match = re.search(r"```lean\s*\n?(.*?)\n?```", text, re.DOTALL)
            return match.group(1).strip() if match else ""
        except Exception as e:
            logger.debug(f"[Decomposer] Proof combination failed: {e}")
            return ""

    @staticmethod
    def _assemble_tactic_proof(statement: str, tactics: list[str]) -> str:
        """Assemble a list of tactics into a by-mode proof."""
        tactic_block = "\n  ".join(tactics)
        # Extract the theorem header (up to ':=')
        header = statement.split(":=")[0].strip() if ":=" in statement else statement
        return f"{header} := by\n  {tactic_block}"


# ══════════════════════════════════════════════════════════════
# Self-Play Conjecture Engine (STP — ICML 2025)
# ══════════════════════════════════════════════════════════════


class ConjectureEngine:
    """Generate and prove novel conjectures via self-play.

    Inspired by STP (Self-play Theorem Prover, ICML 2025):
      - Conjecturer: generates novel theorems by mutating known results
      - Prover: attempts to prove the conjectures
      - Difficulty escalation: progressively harder conjectures
      - Mutual training: conjecturer learns from barely-provable problems

    This allows the system to autonomously expand its proof library
    without external supervision — a form of emergent mathematical discovery.

    Also supports the user's "zero-axiom" philosophy: start from basic logic,
    generate conjectures, prove them, use them to generate harder conjectures.
    """

    MUTATION_STRATEGIES = [
        "generalize",        # Weaken preconditions
        "specialize",        # Strengthen conclusions
        "analogize",         # Transfer to related domain
        "compose",           # Combine two known results
        "contrapositive",    # State the contrapositive
        "strengthen",        # Add additional conclusion
        "weaken",            # Relax conditions
        "boundary",          # Test edge cases
    ]

    def __init__(self) -> None:
        self._conjectures: list[Conjecture] = []
        self._proved_library: list[Conjecture] = []
        self._generation_round = 0
        self._difficulty_target = 0.4  # Target: barely provable

    async def generate_conjectures(
        self,
        known_theorems: list[str],
        llm: Any,
        *,
        num_conjectures: int = 5,
        domain: str = "general",
    ) -> list[Conjecture]:
        """Generate novel conjectures from known theorems (STP conjecturer).

        Strategy:
          1. Select a known theorem as seed
          2. Apply mutation strategy (generalize, specialize, compose, etc.)
          3. Generate Lean 4 statement for the conjecture
          4. Estimate difficulty
          5. Filter: keep conjectures near target difficulty
        """
        from autoforge.engine.llm_router import TaskComplexity
        self._generation_round += 1

        # Select seed theorems (bias toward recently proved ones)
        seeds = known_theorems[-10:] if known_theorems else [
            "theorem nat_add_comm (a b : Nat) : a + b = b + a",
            "theorem nat_zero_add (n : Nat) : 0 + n = n",
        ]

        prompt = f"""You are a mathematical conjecturer. Generate {num_conjectures} novel
conjectures that are plausibly true and expressible in Lean 4.

## Known Theorems (seeds for mutation)
{chr(10).join(f"- {t}" for t in seeds[:8])}

## Mutation Strategies
Apply these strategies to generate new conjectures:
{chr(10).join(f"- {s}" for s in self.MUTATION_STRATEGIES)}

## Domain Focus: {domain}

## Difficulty Target
Generate conjectures of moderate difficulty — not trivial (provable by simp alone)
but not impossibly hard. Target: can be proved in 5-20 tactic steps.

## Instructions
For each conjecture, provide:
1. A valid Lean 4 theorem statement
2. Brief informal description
3. Which seed theorem it's derived from
4. Estimated difficulty (0.0 = trivial, 1.0 = research-level)

Return JSON array:
[
  {{
    "lean_statement": "theorem conj1 ...",
    "informal": "description",
    "source": "which seed theorem",
    "difficulty": 0.4
  }}
]"""

        try:
            response = await llm.call(
                complexity=TaskComplexity.COMPLEX,
                system="You generate novel mathematical conjectures for Lean 4.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            conjectures = []
            if "[" in text:
                json_str = text[text.index("["):text.rindex("]") + 1]
                items = json.loads(json_str)
                for item in items:
                    if isinstance(item, dict) and "lean_statement" in item:
                        conj = Conjecture(
                            id=hashlib.sha256(
                                item["lean_statement"].encode()
                            ).hexdigest()[:12],
                            lean_statement=item["lean_statement"],
                            informal_statement=item.get("informal", ""),
                            source_theorem=item.get("source", ""),
                            difficulty_estimate=float(item.get("difficulty", 0.5)),
                            generation_round=self._generation_round,
                        )
                        conjectures.append(conj)

            # Filter by difficulty target (keep barely-provable ones)
            conjectures = [
                c for c in conjectures
                if abs(c.difficulty_estimate - self._difficulty_target) < 0.3
            ]

            self._conjectures.extend(conjectures)
            logger.info(f"[STP] Generated {len(conjectures)} conjectures "
                        f"(round {self._generation_round})")
            return conjectures

        except Exception as e:
            logger.warning(f"[STP] Conjecture generation failed: {e}")
            return []

    def record_proof(self, conjecture: Conjecture, proof: str) -> None:
        """Record that a conjecture was proved (updates difficulty target)."""
        conjecture.proved = True
        conjecture.proof = proof
        self._proved_library.append(conjecture)

        # Adjust difficulty target (STP: escalate over time)
        proved_count = len(self._proved_library)
        total = len(self._conjectures)
        if total > 0:
            success_rate = proved_count / total
            if success_rate > 0.7:
                self._difficulty_target = min(0.9, self._difficulty_target + 0.05)
            elif success_rate < 0.3:
                self._difficulty_target = max(0.2, self._difficulty_target - 0.05)

    def get_proved_theorems(self) -> list[str]:
        """Get all proved conjecture statements for use as seeds."""
        return [c.lean_statement for c in self._proved_library]

    def get_stats(self) -> dict[str, Any]:
        """Return self-play statistics."""
        return {
            "total_conjectures": len(self._conjectures),
            "proved": len(self._proved_library),
            "success_rate": (
                len(self._proved_library) / max(1, len(self._conjectures))
            ),
            "difficulty_target": self._difficulty_target,
            "generation_round": self._generation_round,
        }

    def save_state(self, path: Path) -> None:
        """Persist conjecture state."""
        data = {
            "generation_round": self._generation_round,
            "difficulty_target": self._difficulty_target,
            "proved_library": [
                {
                    "id": c.id,
                    "lean_statement": c.lean_statement,
                    "informal": c.informal_statement,
                    "proof": c.proof,
                    "difficulty": c.difficulty_estimate,
                }
                for c in self._proved_library
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"[STP] Saved state: {len(self._proved_library)} proved theorems")

    def load_state(self, path: Path) -> None:
        """Load conjecture state."""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._generation_round = data.get("generation_round", 0)
            self._difficulty_target = data.get("difficulty_target", 0.4)
            for item in data.get("proved_library", []):
                self._proved_library.append(Conjecture(
                    id=item["id"],
                    lean_statement=item["lean_statement"],
                    informal_statement=item.get("informal", ""),
                    proof=item.get("proof", ""),
                    difficulty_estimate=item.get("difficulty", 0.5),
                    proved=True,
                ))
            logger.info(f"[STP] Loaded state: {len(self._proved_library)} proved theorems")
        except Exception as e:
            logger.warning(f"[STP] Failed to load state: {e}")
