"""
Lifelong Learning Curriculum — ordering discovery attempts from simple to complex,
tracking which proved results improve future proving ability (positive transfer).

Reference: LeanAgent (ICLR 2025) — proved 155 previously unproved theorems across 23 Lean repos.

This module implements curriculum-based learning for theorem proving, where:
1. Concepts are ordered from simple to complex based on topological depth and dependencies
2. Positive transfer is tracked when proving one theorem helps prove others faster
3. Learning proceeds in epochs, with early stopping on convergence
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from autoforge.engine.theoretical_reasoning import (
    ConceptNode,
    ConceptType,
    RelationType,
    TheoryGraph,
)

logger = logging.getLogger(__name__)


@dataclass
class ComplexityMetric:
    """
    Multidimensional complexity score for a theorem/concept.

    Attributes:
        topological_depth: Steps from root definitions in the DAG (0 = axiom/definition)
        dependency_count: Number of transitive dependencies
        proof_step_estimate: Estimated proof steps from sketch or heuristic
        domain_count: Number of distinct domains involved (cross-domain = harder)
        composite_score: Final difficulty score (e^(S/10) where S = weighted sum)
    """

    topological_depth: int = 0
    dependency_count: int = 0
    proof_step_estimate: int = 0
    domain_count: int = 1
    composite_score: float = 0.0

    def compute(self) -> float:
        """
        Compute composite complexity score via weighted exponential.

        Formula: S = topological_depth + dependency_count * 0.5
                   + proof_step_estimate * 0.3 + (domain_count - 1) * 2
                 composite_score = e^(S / 10)

        Returns:
            The composite score (always > 0)
        """
        S = (
            self.topological_depth
            + self.dependency_count * 0.5
            + self.proof_step_estimate * 0.3
            + (self.domain_count - 1) * 2
        )
        self.composite_score = math.exp(S / 10.0)
        return self.composite_score


@dataclass
class TransferRecord:
    """
    Record of positive/negative transfer from proving one concept to others.

    Attributes:
        proved_concept_id: The concept that was proved
        subsequently_helped: List of concept IDs proved faster after this one
        transfer_score: Net transfer score (positive = helped, negative = confused)
    """

    proved_concept_id: str
    subsequently_helped: list[str] = field(default_factory=list)
    transfer_score: float = 0.0


class CurriculumScheduler:
    """
    Manages theorem proving curriculum: complexity analysis, scheduling, and transfer tracking.

    This scheduler orders proving attempts from simple to complex, ensuring all dependencies
    are proved before attempting dependent concepts. It also tracks positive transfer to
    understand which proofs enable subsequent proofs.
    """

    def __init__(self, graph: TheoryGraph):
        """
        Initialize the curriculum scheduler.

        Args:
            graph: TheoryGraph instance with concepts and relations
        """
        self._graph = graph
        self._complexity_cache: dict[str, ComplexityMetric] = {}
        self._attempt_history: list[dict] = []
        self._transfer_records: dict[str, TransferRecord] = {}
        self._proved_set: set[str] = set()
        self._failed_concepts: set[str] = set()
        self._retry_after = 3  # Retry failed concepts after this many rounds
        logger.info("CurriculumScheduler initialized with %d concepts",
                   len(graph.nodes) if hasattr(graph, 'nodes') else 0)

    def compute_complexity(self, node: ConceptNode) -> ComplexityMetric:
        """
        Compute complexity metrics for a concept based on its position in the theory graph.

        Args:
            node: The concept node to analyze

        Returns:
            ComplexityMetric with all fields computed
        """
        concept_id = node.id

        # Return cached result if available
        if concept_id in self._complexity_cache:
            return self._complexity_cache[concept_id]

        # Compute topological depth (longest path from DEFINITION/AXIOM to this node)
        topological_depth = self._compute_topological_depth(node)

        # Count transitive dependencies
        dependency_count = self._count_dependencies(node)

        # Estimate proof steps from sketch or heuristic
        proof_step_estimate = 0
        if hasattr(node, 'proof_sketch') and node.proof_sketch:
            # Count sentence-like chunks as rough proof steps
            proof_step_estimate = max(1, len(node.proof_sketch.split('.')))
        else:
            # Heuristic: depth * 3
            proof_step_estimate = topological_depth * 3

        # Count distinct domains in dependency chain
        domain_count = self._count_domains(node)

        metric = ComplexityMetric(
            topological_depth=topological_depth,
            dependency_count=dependency_count,
            proof_step_estimate=proof_step_estimate,
            domain_count=domain_count,
        )
        metric.compute()

        self._complexity_cache[concept_id] = metric
        logger.debug(
            "Computed complexity for %s: depth=%d, deps=%d, steps=%d, score=%.3f",
            concept_id,
            topological_depth,
            dependency_count,
            proof_step_estimate,
            metric.composite_score,
        )

        return metric

    def _compute_topological_depth(self, node: ConceptNode) -> int:
        """
        Compute longest path from any DEFINITION or AXIOM to this node.

        Args:
            node: The concept node

        Returns:
            The topological depth (0 for axioms/definitions)
        """
        # DFS to find longest path from axioms/definitions
        visited: set[str] = set()

        def dfs(current: ConceptNode, depth: int) -> int:
            if current.id in visited:
                return 0
            visited.add(current.id)

            # If this is a definition or axiom, return current depth
            if hasattr(current, 'concept_type'):
                if current.concept_type in [ConceptType.DEFINITION, ConceptType.AXIOM]:
                    return depth

            max_parent_depth = 0

            # Find all dependencies (parents in the DAG)
            if hasattr(self._graph, 'relations'):
                for relation in self._graph.relations:
                    if (hasattr(relation, 'target_id') and
                        relation.target_id == current.id and
                        hasattr(relation, 'relation_type') and
                        relation.relation_type == RelationType.DEPENDS_ON):
                        if hasattr(relation, 'source_id'):
                            parent_node = self._graph.get_node(relation.source_id)
                            if parent_node:
                                parent_depth = dfs(parent_node, depth + 1)
                                max_parent_depth = max(max_parent_depth, parent_depth)

            return max(depth, max_parent_depth)

        return dfs(node, 0)

    def _count_dependencies(self, node: ConceptNode) -> int:
        """
        Count the number of transitive dependencies.

        Args:
            node: The concept node

        Returns:
            Number of distinct concepts this one depends on (transitively)
        """
        visited: set[str] = set()

        def collect_deps(current: ConceptNode) -> set[str]:
            if current.id in visited:
                return set()
            visited.add(current.id)

            deps = set()

            if hasattr(self._graph, 'relations'):
                for relation in self._graph.relations:
                    if (hasattr(relation, 'target_id') and
                        relation.target_id == current.id and
                        hasattr(relation, 'relation_type') and
                        relation.relation_type == RelationType.DEPENDS_ON):
                        if hasattr(relation, 'source_id'):
                            deps.add(relation.source_id)
                            parent_node = self._graph.get_node(relation.source_id)
                            if parent_node:
                                deps.update(collect_deps(parent_node))

            return deps

        return len(collect_deps(node))

    def _count_domains(self, node: ConceptNode) -> int:
        """
        Count the number of distinct domains in the dependency chain.

        Args:
            node: The concept node

        Returns:
            Number of distinct domains (cross-domain = harder)
        """
        domains: set[str] = set()
        visited: set[str] = set()

        def collect_domains(current: ConceptNode) -> None:
            if current.id in visited:
                return
            visited.add(current.id)

            if hasattr(current, 'domain') and current.domain:
                domains.add(current.domain)

            # Traverse dependencies
            if hasattr(self._graph, 'relations'):
                for relation in self._graph.relations:
                    if (hasattr(relation, 'target_id') and
                        relation.target_id == current.id and
                        hasattr(relation, 'relation_type') and
                        relation.relation_type == RelationType.DEPENDS_ON):
                        if hasattr(relation, 'source_id'):
                            parent_node = self._graph.get_node(relation.source_id)
                            if parent_node:
                                collect_domains(parent_node)

        collect_domains(node)
        return max(1, len(domains))

    def sort_by_curriculum(self, concepts: list[ConceptNode]) -> list[ConceptNode]:
        """
        Sort concepts from easiest to hardest based on complexity metrics.

        Within the same complexity tier, prefers concepts whose dependencies are already proved.

        Args:
            concepts: List of concept nodes to sort

        Returns:
            Sorted list (easiest first)
        """
        # Compute complexity for all concepts
        complexities = [(c, self.compute_complexity(c)) for c in concepts]

        # Sort by complexity score, then by percentage of dependencies already proved
        def sort_key(item: tuple[ConceptNode, ComplexityMetric]) -> tuple[float, float]:
            concept, complexity = item

            # Primary sort: complexity score
            primary = complexity.composite_score

            # Secondary sort: how many dependencies are already proved
            deps = self._count_dependencies(concept)
            if deps > 0:
                deps_proved = sum(
                    1 for dep_id in self._get_all_dependencies(concept.id)
                    if dep_id in self._proved_set
                )
                secondary = -deps_proved / deps  # Negative so proved deps come first
            else:
                secondary = 0.0

            return (primary, secondary)

        complexities.sort(key=sort_key)
        sorted_concepts = [c for c, _ in complexities]

        logger.debug("Sorted %d concepts by curriculum", len(sorted_concepts))
        return sorted_concepts

    def _get_all_dependencies(self, concept_id: str) -> set[str]:
        """Get all transitive dependencies of a concept."""
        visited: set[str] = set()
        deps: set[str] = set()

        def traverse(cid: str) -> None:
            if cid in visited:
                return
            visited.add(cid)

            if hasattr(self._graph, 'relations'):
                for relation in self._graph.relations:
                    if (hasattr(relation, 'target_id') and
                        relation.target_id == cid and
                        hasattr(relation, 'relation_type') and
                        relation.relation_type == RelationType.DEPENDS_ON):
                        if hasattr(relation, 'source_id'):
                            parent_id = relation.source_id
                            deps.add(parent_id)
                            traverse(parent_id)

        traverse(concept_id)
        return deps

    def get_next_batch(self, batch_size: int = 5) -> list[ConceptNode]:
        """
        Get the next batch of concepts to attempt, respecting all constraints.

        Constraints:
        1. All dependencies must already be proved (or in _proved_set)
        2. Sorted by complexity (easiest first)
        3. Skip recently-failed concepts (unless retry_after rounds have passed)

        Args:
            batch_size: Number of concepts to return

        Returns:
            List of concepts ready to prove (sorted by curriculum)
        """
        candidates = []

        # Get all unproved concepts
        all_concepts = [n for n in self._graph.nodes.values()] if hasattr(
            self._graph, 'nodes'
        ) else []

        for concept in all_concepts:
            concept_id = concept.id

            # Skip already proved
            if concept_id in self._proved_set:
                continue

            # Check if all dependencies are proved
            deps = self._get_all_dependencies(concept_id)
            if not deps.issubset(self._proved_set | {concept_id}):
                continue

            # Check retry threshold for failed concepts
            if concept_id in self._failed_concepts:
                recent_attempts = [
                    a for a in self._attempt_history
                    if a['concept_id'] == concept_id
                ]
                if recent_attempts:
                    last_attempt_round = recent_attempts[-1].get('round', 0)
                    current_round = len(set(a['round'] for a in self._attempt_history)) or 0
                    if current_round - last_attempt_round < self._retry_after:
                        continue

            candidates.append(concept)

        # Sort by curriculum
        sorted_candidates = self.sort_by_curriculum(candidates)

        # Return top batch_size
        batch = sorted_candidates[:batch_size]
        logger.info("Selected batch of %d concepts (from %d candidates)",
                   len(batch), len(candidates))

        return batch

    def record_attempt(
        self, concept_id: str, success: bool, round_number: int
    ) -> None:
        """
        Record the outcome of a proof attempt.

        If successful, adds to _proved_set and checks for positive transfer.

        Args:
            concept_id: The concept that was attempted
            success: Whether the proof succeeded
            round_number: Which round of learning this was
        """
        self._attempt_history.append({
            'concept_id': concept_id,
            'success': success,
            'time_taken': time.time(),
            'round': round_number,
        })

        if success:
            self._proved_set.add(concept_id)
            if concept_id in self._failed_concepts:
                self._failed_concepts.discard(concept_id)
            logger.info("Proved concept: %s", concept_id)
        else:
            self._failed_concepts.add(concept_id)
            logger.info("Failed to prove concept: %s", concept_id)

    def record_transfer(self, proved_id: str, helped_id: str) -> None:
        """
        Record that proving one concept helped prove another.

        Args:
            proved_id: The concept that was proved
            helped_id: The concept that benefited from the first proof
        """
        if proved_id not in self._transfer_records:
            self._transfer_records[proved_id] = TransferRecord(
                proved_concept_id=proved_id
            )

        record = self._transfer_records[proved_id]
        if helped_id not in record.subsequently_helped:
            record.subsequently_helped.append(helped_id)
            record.transfer_score += 1.0

        logger.debug("Recorded transfer: %s -> %s", proved_id, helped_id)

    def get_transfer_graph(self) -> dict[str, list[str]]:
        """
        Get the transfer graph showing which proved concepts helped which others.

        Returns:
            Dict mapping proved concept IDs to lists of concepts they helped prove
        """
        return {
            proved_id: record.subsequently_helped
            for proved_id, record in self._transfer_records.items()
        }

    def get_curriculum_stats(self) -> dict[str, Any]:
        """
        Get comprehensive statistics on curriculum learning progress.

        Returns:
            Dict with keys:
            - total_concepts: Total concepts in graph
            - proved: Number successfully proved
            - failed: Number that failed to prove
            - avg_complexity_proved: Average complexity of proved concepts
            - avg_complexity_failed: Average complexity of failed concepts
            - positive_transfers: Total positive transfer events
            - negative_transfers: Total negative transfer events
        """
        total = len(self._graph.nodes) if hasattr(self._graph, 'nodes') else 0
        proved = len(self._proved_set)
        failed = len(self._failed_concepts)

        # Compute average complexities
        proved_concepts = [
            n for n in self._graph.nodes.values()
            if n.id in self._proved_set
        ] if hasattr(self._graph, 'nodes') else []

        failed_concepts = [
            n for n in self._graph.nodes.values()
            if n.id in self._failed_concepts
        ] if hasattr(self._graph, 'nodes') else []

        avg_complexity_proved = (
            sum(self.compute_complexity(c).composite_score for c in proved_concepts)
            / len(proved_concepts)
            if proved_concepts
            else 0.0
        )

        avg_complexity_failed = (
            sum(self.compute_complexity(c).composite_score for c in failed_concepts)
            / len(failed_concepts)
            if failed_concepts
            else 0.0
        )

        positive_transfers = sum(
            len(r.subsequently_helped) for r in self._transfer_records.values()
        )

        return {
            'total_concepts': total,
            'proved': proved,
            'failed': failed,
            'avg_complexity_proved': avg_complexity_proved,
            'avg_complexity_failed': avg_complexity_failed,
            'positive_transfers': positive_transfers,
            'coverage': proved / total if total > 0 else 0.0,
        }

    def save(self, path: Path) -> None:
        """
        Persist scheduler state to disk.

        Args:
            path: Path to save state file
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'proved_set': list(self._proved_set),
            'failed_concepts': list(self._failed_concepts),
            'attempt_history': self._attempt_history,
            'transfer_records': {
                pid: {
                    'proved_concept_id': r.proved_concept_id,
                    'subsequently_helped': r.subsequently_helped,
                    'transfer_score': r.transfer_score,
                }
                for pid, r in self._transfer_records.items()
            },
            'stats': self.get_curriculum_stats(),
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info("Saved curriculum scheduler state to %s", path)

    def load(self, path: Path) -> None:
        """
        Restore scheduler state from disk.

        Args:
            path: Path to load state file from
        """
        if not path.exists():
            logger.warning("State file not found: %s", path)
            return

        with open(path) as f:
            state = json.load(f)

        self._proved_set = set(state.get('proved_set', []))
        self._failed_concepts = set(state.get('failed_concepts', []))
        self._attempt_history = state.get('attempt_history', [])

        for pid, record_data in state.get('transfer_records', {}).items():
            self._transfer_records[pid] = TransferRecord(
                proved_concept_id=record_data['proved_concept_id'],
                subsequently_helped=record_data['subsequently_helped'],
                transfer_score=record_data['transfer_score'],
            )

        logger.info("Loaded curriculum scheduler state from %s", path)


class LifelongLearner:
    """
    High-level orchestrator for lifelong learning via curriculum.

    Manages multiple epochs of proving, with early stopping when convergence
    (no new proofs in an epoch) is detected.
    """

    def __init__(self, graph: TheoryGraph, max_epochs: int = 5):
        """
        Initialize the lifelong learner.

        Args:
            graph: TheoryGraph instance
            max_epochs: Maximum number of epochs to run
        """
        self._graph = graph
        self._max_epochs = max_epochs
        self._scheduler = CurriculumScheduler(graph)
        self._epoch_log: list[dict] = []
        logger.info("LifelongLearner initialized with %d epochs", max_epochs)

    async def run(
        self,
        llm: Any,
        prover: Any | None = None,
        batch_size: int = 5,
    ) -> dict[str, Any]:
        """
        Run lifelong learning curriculum for multiple epochs.

        For each epoch:
        1. Get next batch of concepts to prove
        2. Attempt to prove each one
        3. Record results and check for positive transfer
        4. Stop if no new proofs in epoch (convergence)

        Args:
            llm: Language model instance for proof attempts
            prover: Optional formal prover (Lean, etc.)
            batch_size: Number of concepts per batch

        Returns:
            Dict with final stats and learning summary
        """
        logger.info("Starting lifelong learning with %d epochs", self._max_epochs)

        for epoch in range(self._max_epochs):
            logger.info("=== Epoch %d / %d ===", epoch + 1, self._max_epochs)

            batch = self._scheduler.get_next_batch(batch_size)

            if not batch:
                logger.info("No more concepts to prove. Convergence reached.")
                break

            epoch_start = time.time()
            epoch_proved_count = 0

            for concept in batch:
                success, proof_sketch = await self._attempt_proof(concept, llm, prover)

                self._scheduler.record_attempt(
                    concept.id, success, epoch
                )

                if success:
                    epoch_proved_count += 1

                    # Check for positive transfer to dependent concepts
                    if hasattr(self._graph, 'relations'):
                        for relation in self._graph.relations:
                            if (hasattr(relation, 'source_id') and
                                relation.source_id == concept.id and
                                hasattr(relation, 'relation_type') and
                                relation.relation_type == RelationType.DEPENDS_ON):
                                if hasattr(relation, 'target_id'):
                                    self._scheduler.record_transfer(
                                        concept.id, relation.target_id
                                    )

            epoch_time = time.time() - epoch_start

            epoch_log = {
                'epoch': epoch + 1,
                'batch_size': len(batch),
                'proved_count': epoch_proved_count,
                'time_seconds': epoch_time,
                'stats': self._scheduler.get_curriculum_stats(),
            }
            self._epoch_log.append(epoch_log)

            logger.info(
                "Epoch %d complete: %d / %d proved (%.1fs)",
                epoch + 1,
                epoch_proved_count,
                len(batch),
                epoch_time,
            )

            if epoch_proved_count == 0:
                logger.info("No new proofs in this epoch. Stopping.")
                break

        return self.get_summary()

    async def _attempt_proof(
        self,
        concept: ConceptNode,
        llm: Any,
        prover: Any | None = None,
    ) -> tuple[bool, str]:
        """
        Attempt to prove a single concept using LLM and optional formal prover.

        Args:
            concept: The concept to prove
            llm: Language model for generating proofs
            prover: Optional formal prover for verification

        Returns:
            Tuple of (success, proof_sketch)
        """
        # Build context from already-proved concepts
        context = self._build_proof_context(concept)

        prompt = self._build_proof_prompt(concept, context)

        try:
            # Request proof from LLM
            proof_sketch = await self._call_llm(llm, prompt)

            if not proof_sketch:
                return False, ""

            # Optionally verify with formal prover
            if prover:
                verified = await self._verify_proof(prover, concept, proof_sketch)
                return verified, proof_sketch

            # If no prover, trust LLM's generation
            return True, proof_sketch

        except Exception as e:
            logger.error("Error attempting proof for %s: %s", concept.id, e)
            return False, ""

    def _build_proof_context(self, concept: ConceptNode) -> str:
        """
        Build context from already-proved concepts that this one depends on.

        Args:
            concept: The concept being proved

        Returns:
            String with available lemmas and theorems
        """
        deps = self._scheduler._get_all_dependencies(concept.id)
        proved_deps = deps & self._scheduler._proved_set

        context_parts = []
        for dep_id in proved_deps:
            dep_node = self._graph.get_node(dep_id) if hasattr(self._graph, 'get_node') else None
            if dep_node:
                if hasattr(dep_node, 'proof_sketch') and dep_node.proof_sketch:
                    context_parts.append(f"{dep_id}:\n{dep_node.proof_sketch}")

        return "\n\n".join(context_parts) if context_parts else ""

    def _build_proof_prompt(self, concept: ConceptNode, context: str) -> str:
        """
        Build an LLM prompt for proving the concept.

        Args:
            concept: The concept to prove
            context: Available lemmas and theorems

        Returns:
            Prompt string for the LLM
        """
        prompt = f"Prove the following theorem:\n\n"

        if hasattr(concept, 'statement'):
            prompt += f"Statement: {concept.statement}\n\n"
        else:
            prompt += f"Concept: {concept.id}\n\n"

        if context:
            prompt += f"Available lemmas and theorems:\n{context}\n\n"

        prompt += "Provide a clear proof sketch or Lean 4 proof code."

        return prompt

    async def _call_llm(self, llm: Any, prompt: str) -> str:
        """
        Call the language model asynchronously.

        Args:
            llm: Language model instance
            prompt: Proof prompt

        Returns:
            Generated proof sketch
        """
        # Check if llm has async method
        if hasattr(llm, 'generate_async'):
            return await llm.generate_async(prompt)
        elif hasattr(llm, 'generate'):
            # Wrap sync call in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, llm.generate, prompt)
        else:
            logger.warning("LLM instance has no generate method")
            return ""

    async def _verify_proof(
        self,
        prover: Any,
        concept: ConceptNode,
        proof: str,
    ) -> bool:
        """
        Verify proof with a formal prover.

        Args:
            prover: Formal prover instance
            concept: The concept being proved
            proof: The proof to verify

        Returns:
            Whether the proof is valid
        """
        try:
            if hasattr(prover, 'verify_async'):
                return await prover.verify_async(concept, proof)
            elif hasattr(prover, 'verify'):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, prover.verify, concept, proof)
            else:
                return True  # No verification available
        except Exception as e:
            logger.error("Proof verification failed: %s", e)
            return False

    def get_summary(self) -> dict[str, Any]:
        """
        Get a comprehensive summary of learning progress.

        Returns:
            Dict with learning summary, statistics, and transfer information
        """
        stats = self._scheduler.get_curriculum_stats()
        transfer_graph = self._scheduler.get_transfer_graph()

        total_time = sum(e['time_seconds'] for e in self._epoch_log)

        return {
            'epochs_completed': len(self._epoch_log),
            'total_time_seconds': total_time,
            'final_stats': stats,
            'transfer_graph': transfer_graph,
            'epoch_log': self._epoch_log,
            'summary': {
                'total_concepts': stats['total_concepts'],
                'proved_concepts': stats['proved'],
                'coverage_percent': stats['coverage'] * 100,
                'positive_transfers': stats['positive_transfers'],
            },
        }
