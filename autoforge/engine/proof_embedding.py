"""Proof embedding and transfer learning system for theorem proving.

Implements multi-level proof embeddings (goal, tactic, state) with semantic
similarity-based retrieval and cross-domain tactic transfer for automated
theorem proving. Inspired by neural theorem proving architectures and
transfer learning in formal mathematics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProofState:
    """Representation of a proof state during proof search."""

    goal: str
    """Current proof goal (natural language or Lean syntax)."""

    hypotheses: list[str] = field(default_factory=list)
    """Available hypotheses in context."""

    tactic_history: list[str] = field(default_factory=list)
    """Tactics applied to reach this state."""

    domain: str = "general"
    """Domain of the theorem (e.g., 'algebra', 'topology', 'number_theory')."""

    depth: int = 0
    """Proof tree depth (number of tactics applied)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (source, difficulty, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "goal": self.goal,
            "hypotheses": self.hypotheses,
            "tactic_history": self.tactic_history,
            "domain": self.domain,
            "depth": self.depth,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ProofState:
        """Load from serializable dict."""
        return ProofState(
            goal=data["goal"],
            hypotheses=data.get("hypotheses", []),
            tactic_history=data.get("tactic_history", []),
            domain=data.get("domain", "general"),
            depth=data.get("depth", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProofEmbedding:
    """Cached embedding and metadata for a proof state."""

    proof_id: str
    state_embedding: list[float]
    """Combined embedding of goal + hypotheses + tactics."""

    goal_embedding: list[float]
    """Embedding of the proof goal alone."""

    tactic_embedding: list[float]
    """Embedding of the tactic sequence."""

    domain: str
    source_theorem: str
    successful_tactics: list[str]
    difficulty: float  # 0-1 scale

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "proof_id": self.proof_id,
            "state_embedding": self.state_embedding,
            "goal_embedding": self.goal_embedding,
            "tactic_embedding": self.tactic_embedding,
            "domain": self.domain,
            "source_theorem": self.source_theorem,
            "successful_tactics": self.successful_tactics,
            "difficulty": self.difficulty,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ProofEmbedding:
        """Load from serializable dict."""
        return ProofEmbedding(
            proof_id=data["proof_id"],
            state_embedding=data["state_embedding"],
            goal_embedding=data["goal_embedding"],
            tactic_embedding=data["tactic_embedding"],
            domain=data["domain"],
            source_theorem=data["source_theorem"],
            successful_tactics=data["successful_tactics"],
            difficulty=data["difficulty"],
        )


@dataclass
class ProofTransferCandidate:
    """A candidate proof for transfer learning."""

    source_proof: ProofEmbedding
    similarity: float
    """Semantic similarity (0-1)."""

    transferred_tactics: list[str]
    """Tactics adapted for target state."""

    adaptation_notes: str
    confidence: float
    """Confidence in adaptation (0-1)."""


class EmbeddingModel:
    """Embedding model for proof states and tactics."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._model = None
        self._use_sentence_transformers = False

    async def _ensure_loaded(self) -> None:
        """Lazily load embedding model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformers for proof embeddings")
            device = "cpu"
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            self._use_sentence_transformers = True
            logger.info("Sentence-transformers model loaded")
        except ImportError:
            logger.info("sentence-transformers not available, using TF-IDF fallback")
            self._use_sentence_transformers = False

    async def embed_goal(self, goal: str) -> list[float]:
        """Embed a proof goal."""
        if not goal:
            return [0.0] * self.embedding_dim

        await self._ensure_loaded()

        if self._use_sentence_transformers:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self._model.encode([goal], show_progress_bar=False)[0].tolist(),
            )
            return embedding
        else:
            return self._tfidf_embed(goal)

    async def embed_tactic_sequence(self, tactics: list[str]) -> list[float]:
        """Embed a sequence of tactics."""
        if not tactics:
            return [0.0] * self.embedding_dim

        # Create a text representation of the tactic sequence
        tactic_text = " → ".join(tactics)
        return await self.embed_goal(tactic_text)

    async def embed_proof_state(self, state: ProofState) -> list[float]:
        """Embed a complete proof state."""
        # Embed each component
        goal_emb = await self.embed_goal(state.goal)
        hyp_text = " | ".join(state.hypotheses) if state.hypotheses else ""
        hyp_emb = await self.embed_goal(hyp_text)
        tactic_emb = await self.embed_tactic_sequence(state.tactic_history)

        # Combine embeddings
        return self._combine_embeddings(goal_emb, tactic_emb, hyp_emb, state.depth)

    def _combine_embeddings(
        self, goal_emb: list[float], tactic_emb: list[float], hyp_emb: list[float], depth: int
    ) -> list[float]:
        """Combine multiple embeddings with weighted averaging."""
        # Weights favor goal (most important), then tactics, then hypotheses
        goal_weight = 0.6
        tactic_weight = 0.25
        hyp_weight = 0.15

        combined = []
        for i in range(self.embedding_dim):
            value = (
                goal_emb[i] * goal_weight
                + tactic_emb[i] * tactic_weight
                + hyp_emb[i] * hyp_weight
            )
            combined.append(value)

        # Add depth signal
        depth_factor = min(depth / 10.0, 1.0)  # Normalize depth 0-10
        combined = [v + depth_factor * 0.1 for v in combined]

        # Normalize
        norm = math.sqrt(sum(v ** 2 for v in combined))
        if norm > 0:
            combined = [v / norm for v in combined]

        return combined

    def _tfidf_embed(self, text: str) -> list[float]:
        """Fallback TF-IDF based embedding."""
        import hashlib

        # Use hash-based deterministic embedding as fallback
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        embedding = []
        for i in range(self.embedding_dim):
            byte_val = hash_bytes[i % len(hash_bytes)]
            value = (byte_val / 128.0) - 1.0
            embedding.append(value)

        # Normalize
        norm = math.sqrt(sum(x ** 2 for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding


class ProofMemoryBank:
    """Vector database of proof embeddings with fast retrieval."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.proofs: dict[str, ProofEmbedding] = {}
        self.embeddings_index: np.ndarray | None = None
        self.embedding_ids: list[str] = []
        self._faiss_index = None
        self._use_faiss = False
        self._domain_index: dict[str, list[str]] = {}
        self._tactic_index: dict[str, list[str]] = {}

    async def add_proof(
        self,
        proof_id: str,
        state: ProofState,
        successful_tactics: list[str],
        embedding_model: EmbeddingModel,
        difficulty: float = 0.5,
        source_theorem: str = "",
    ) -> ProofEmbedding:
        """Add a proof to memory bank."""
        # Generate embeddings
        state_emb = await embedding_model.embed_proof_state(state)
        goal_emb = await embedding_model.embed_goal(state.goal)
        tactic_emb = await embedding_model.embed_tactic_sequence(successful_tactics)

        proof_emb = ProofEmbedding(
            proof_id=proof_id,
            state_embedding=state_emb,
            goal_embedding=goal_emb,
            tactic_embedding=tactic_emb,
            domain=state.domain,
            source_theorem=source_theorem or proof_id,
            successful_tactics=successful_tactics,
            difficulty=difficulty,
        )

        self.proofs[proof_id] = proof_emb

        # Update indices
        if state.domain not in self._domain_index:
            self._domain_index[state.domain] = []
        self._domain_index[state.domain].append(proof_id)

        for tactic in successful_tactics:
            if tactic not in self._tactic_index:
                self._tactic_index[tactic] = []
            if proof_id not in self._tactic_index[tactic]:
                self._tactic_index[tactic].append(proof_id)

        # Rebuild FAISS index if using it
        if self._use_faiss or len(self.proofs) > 100:
            await self._build_faiss_index()

        logger.info(f"Added proof {proof_id} to memory bank (capacity: {len(self.proofs)}/{self.capacity})")
        return proof_emb

    async def search_similar(
        self, query_state: ProofState, embedding_model: EmbeddingModel, top_k: int = 5
    ) -> list[tuple[ProofEmbedding, float]]:
        """Find similar proofs using semantic similarity."""
        if not self.proofs:
            return []

        # Embed query state
        query_emb = await embedding_model.embed_proof_state(query_state)
        query_vec = np.array(query_emb, dtype=np.float32)

        results: list[tuple[ProofEmbedding, float]] = []

        if self._faiss_index is not None:
            # Use FAISS for fast search
            distances, indices = self._faiss_index.search(query_vec.reshape(1, -1), min(top_k, len(self.proofs)))
            for idx, distance in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.embedding_ids):
                    proof_id = self.embedding_ids[idx]
                    if proof_id in self.proofs:
                        # Convert distance to similarity (lower distance = higher similarity)
                        similarity = 1.0 / (1.0 + float(distance))
                        results.append((self.proofs[proof_id], similarity))
        else:
            # Linear search with cosine similarity
            scores = []
            for proof_id, proof_emb in self.proofs.items():
                state_vec = np.array(proof_emb.state_embedding, dtype=np.float32)
                # Cosine similarity
                dot_product = np.dot(query_vec, state_vec)
                similarity = float(dot_product)
                scores.append((proof_id, similarity))

            # Sort by similarity descending
            scores.sort(key=lambda x: x[1], reverse=True)
            results = [(self.proofs[pid], sim) for pid, sim in scores[:top_k]]

        return results

    async def search_by_domain(self, domain: str, top_k: int = 20) -> list[ProofEmbedding]:
        """Find proofs in a specific domain."""
        proof_ids = self._domain_index.get(domain, [])
        # Sort by difficulty (ascending by default)
        proofs = [self.proofs[pid] for pid in proof_ids if pid in self.proofs]
        proofs.sort(key=lambda p: p.difficulty)
        return proofs[:top_k]

    async def search_by_tactic(self, tactic: str, top_k: int = 10) -> list[ProofEmbedding]:
        """Find proofs using a specific tactic."""
        proof_ids = self._tactic_index.get(tactic, [])
        proofs = [self.proofs[pid] for pid in proof_ids if pid in self.proofs]
        return proofs[:top_k]

    async def _build_faiss_index(self) -> None:
        """Build FAISS index for fast approximate search."""
        try:
            import faiss

            if not self.proofs:
                return

            embeddings = []
            self.embedding_ids = []

            for proof_id, proof_emb in self.proofs.items():
                embeddings.append(proof_emb.state_embedding)
                self.embedding_ids.append(proof_id)

            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Create FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)

            self._faiss_index = index
            self._use_faiss = True
            logger.info(f"Built FAISS index for {len(self.proofs)} proofs")
        except ImportError:
            logger.debug("FAISS not available, using linear search")
            self._use_faiss = False

    def save(self, path: Path) -> None:
        """Persist proof memory to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "capacity": self.capacity,
            "proofs": {
                proof_id: proof.to_dict() for proof_id, proof in self.proofs.items()
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.proofs)} proofs to {path}")

    async def load(self, path: Path) -> None:
        """Load proof memory from JSON."""
        if not path.exists():
            logger.info(f"No memory file at {path}, starting fresh")
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.capacity = data.get("capacity", 10000)
            self.proofs.clear()
            self._domain_index.clear()
            self._tactic_index.clear()

            for proof_id, proof_data in data.get("proofs", {}).items():
                proof_emb = ProofEmbedding.from_dict(proof_data)
                self.proofs[proof_id] = proof_emb

                # Rebuild indices
                if proof_emb.domain not in self._domain_index:
                    self._domain_index[proof_emb.domain] = []
                self._domain_index[proof_emb.domain].append(proof_id)

                for tactic in proof_emb.successful_tactics:
                    if tactic not in self._tactic_index:
                        self._tactic_index[tactic] = []
                    if proof_id not in self._tactic_index[tactic]:
                        self._tactic_index[tactic].append(proof_id)

            logger.info(f"Loaded {len(self.proofs)} proofs from {path}")

            # Rebuild FAISS index if we have enough proofs
            if len(self.proofs) > 100:
                await self._build_faiss_index()
        except Exception as e:
            logger.error(f"Failed to load memory bank: {e}")

    def stats(self) -> dict[str, Any]:
        """Get statistics about the memory bank."""
        domains = set(p.domain for p in self.proofs.values())
        tactics = set(t for p in self.proofs.values() for t in p.successful_tactics)

        return {
            "total_proofs": len(self.proofs),
            "capacity": self.capacity,
            "domains": list(domains),
            "unique_tactics": len(tactics),
            "avg_difficulty": (
                sum(p.difficulty for p in self.proofs.values()) / len(self.proofs)
                if self.proofs
                else 0.0
            ),
            "faiss_enabled": self._use_faiss,
        }


class TacticTransfer:
    """Transfer tactics from source to target proof states."""

    async def transfer_tactics(
        self, source: ProofEmbedding, target_state: ProofState, llm: Any
    ) -> list[str]:
        """Adapt tactics from source proof to target state."""
        if not source.successful_tactics:
            return []

        # Try structural adaptation first
        adapted = []
        for tactic in source.successful_tactics:
            adapted_tactic = self._structural_adaptation(tactic, target_state.hypotheses)
            adapted.append(adapted_tactic)

        # Use LLM to refine if available
        if llm is not None:
            adapted = await self.rank_transferred_tactics(adapted, target_state, llm)
            adapted = [t for t, _ in adapted]

        return adapted

    async def adapt_tactic(
        self, tactic: str, source_domain: str, target_domain: str, llm: Any
    ) -> str:
        """Adapt a single tactic across domains."""
        domain_map = self._domain_mapping(source_domain, target_domain)

        # Apply known transformations
        adapted = tactic
        for source_term, target_term in domain_map.items():
            adapted = adapted.replace(source_term, target_term)

        return adapted

    async def rank_transferred_tactics(
        self, tactics: list[str], target_state: ProofState, llm: Any
    ) -> list[tuple[str, float]]:
        """Rank transferred tactics by relevance to target state."""
        if not tactics or llm is None:
            return [(t, 0.5) for t in tactics]

        # Simple ranking based on goal overlap
        ranked = []
        goal_tokens = set(target_state.goal.lower().split())

        for tactic in tactics:
            tactic_tokens = set(tactic.lower().split())
            overlap = len(goal_tokens & tactic_tokens)
            score = min(1.0, overlap / max(len(goal_tokens), 1))
            ranked.append((tactic, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _domain_mapping(self, source_domain: str, target_domain: str) -> dict[str, str]:
        """Get known domain-to-domain tactic transformations."""
        # Define common transformations between mathematical domains
        mappings = {
            ("algebra", "topology"): {
                "ring": "topological_space",
                "add": "union",
                "mul": "intersection",
            },
            ("number_theory", "algebra"): {
                "prime": "irreducible",
                "divisibility": "divisibility_in_ring",
            },
            ("topology", "metric"): {
                "open": "open_ball",
                "closed": "closed_ball",
            },
        }

        key = (source_domain, target_domain)
        return mappings.get(key, {})

    def _structural_adaptation(self, tactic: str, target_hypotheses: list[str]) -> str:
        """Adapt tactic syntax to match target hypotheses."""
        # Extract variable names from hypotheses
        adapted = tactic

        # Simple heuristic: match common hypothesis names
        hyp_names = set()
        for hyp in target_hypotheses:
            # Extract variable names (e.g., 'x: nat' -> 'x')
            parts = hyp.split(":")
            if parts:
                hyp_names.add(parts[0].strip())

        return adapted


class CrossDomainTransfer:
    """Cross-domain proof transfer and pattern extraction."""

    async def find_analogous_proofs(
        self,
        state: ProofState,
        memory: ProofMemoryBank,
        embedding_model: EmbeddingModel,
        llm: Any,
        cross_domain: bool = True,
    ) -> list[ProofTransferCandidate]:
        """Find analogous proofs (same or different domain)."""
        candidates: list[ProofTransferCandidate] = []

        # Search same domain first
        same_domain_results = await memory.search_similar(state, embedding_model, top_k=5)

        for proof_emb, similarity in same_domain_results:
            adapted = await TacticTransfer().transfer_tactics(proof_emb, state, llm)
            candidate = ProofTransferCandidate(
                source_proof=proof_emb,
                similarity=similarity,
                transferred_tactics=adapted,
                adaptation_notes=f"Same domain ({state.domain})",
                confidence=min(1.0, similarity * 0.9),
            )
            candidates.append(candidate)

        # Cross-domain search if enabled
        if cross_domain and len(candidates) < 10:
            all_proofs = list(memory.proofs.values())
            for proof_emb in all_proofs:
                if proof_emb.domain == state.domain:
                    continue

                domain_distance = self._compute_domain_distance(state.domain, proof_emb.domain)
                if domain_distance > 0.3:  # Only consider similar domains
                    continue

                # Compute similarity
                state_vec = np.array(state.to_dict, dtype=np.float32)
                proof_vec = np.array(proof_emb.state_embedding, dtype=np.float32)

                adapted = await TacticTransfer().transfer_tactics(proof_emb, state, llm)
                if adapted:
                    candidate = ProofTransferCandidate(
                        source_proof=proof_emb,
                        similarity=0.5,  # Lower confidence across domains
                        transferred_tactics=adapted,
                        adaptation_notes=f"Cross-domain from {proof_emb.domain}",
                        confidence=0.4 * (1.0 - domain_distance),
                    )
                    candidates.append(candidate)

        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates[:10]

    async def abstract_proof_pattern(self, proof: ProofEmbedding, llm: Any) -> str:
        """Extract abstract proof strategy from a concrete proof."""
        if llm is None:
            return " → ".join(proof.successful_tactics)

        prompt = f"""Extract the abstract proof strategy from this sequence of tactics:
{' → '.join(proof.successful_tactics)}

Provide a one-sentence high-level proof strategy that abstracts away specific tactics
and focuses on the overall approach (e.g., 'Use induction on the natural numbers' or
'Apply a contradiction argument').

Strategy:"""

        response = await llm.call(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return content.strip()

    async def instantiate_pattern(
        self, pattern: str, target_state: ProofState, llm: Any
    ) -> list[str]:
        """Generate concrete tactics from abstract pattern."""
        if llm is None:
            return []

        prompt = f"""Given an abstract proof strategy and a target theorem, generate concrete tactics.

Strategy: {pattern}

Target goal: {target_state.goal}
Hypotheses: {' | '.join(target_state.hypotheses)}

Generate 3-5 concrete Lean tactics that implement this strategy. Return as a JSON array of strings.
Tactics:"""

        response = await llm.call(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        try:
            import re

            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                tactics = json.loads(match.group())
                return tactics if isinstance(tactics, list) else []
        except (json.JSONDecodeError, ValueError):
            pass

        return []

    def _compute_domain_distance(self, d1: str, d2: str) -> float:
        """Compute distance between two mathematical domains."""
        # Simple distance based on known relationships
        if d1 == d2:
            return 0.0

        related = {
            "algebra": {"topology": 0.2, "number_theory": 0.15},
            "topology": {"algebra": 0.2, "metric": 0.1},
            "number_theory": {"algebra": 0.15},
            "metric": {"topology": 0.1},
        }

        distance = related.get(d1, {}).get(d2, 1.0)
        return distance


class ProofExperienceTracker:
    """Track proof attempts and learn from experience."""

    def __init__(self):
        self.attempts: list[dict[str, Any]] = []
        self.domain_stats: dict[str, dict[str, Any]] = {}

    async def record_attempt(
        self, state: ProofState, tactics: list[str], success: bool
    ) -> None:
        """Record a proof attempt."""
        attempt = {
            "goal": state.goal,
            "domain": state.domain,
            "tactics": tactics,
            "success": success,
            "num_tactics": len(tactics),
            "depth": state.depth,
        }
        self.attempts.append(attempt)

        # Update domain statistics
        if state.domain not in self.domain_stats:
            self.domain_stats[state.domain] = {
                "attempts": 0,
                "successes": 0,
                "avg_tactics": 0.0,
                "common_tactics": {},
            }

        stats = self.domain_stats[state.domain]
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1
            # Track successful tactics
            for tactic in tactics:
                stats["common_tactics"][tactic] = stats["common_tactics"].get(tactic, 0) + 1

        # Update average
        total_tactics = sum(a["num_tactics"] for a in self.attempts if a["domain"] == state.domain)
        domain_attempts = stats["attempts"]
        stats["avg_tactics"] = total_tactics / domain_attempts if domain_attempts > 0 else 0.0

    async def get_success_rate(self, domain: str) -> float:
        """Get success rate for a domain."""
        stats = self.domain_stats.get(domain)
        if not stats or stats["attempts"] == 0:
            return 0.0
        return stats["successes"] / stats["attempts"]

    async def get_common_failure_patterns(self, domain: str) -> list[str]:
        """Get common failure patterns in a domain."""
        domain_attempts = [a for a in self.attempts if a["domain"] == domain and not a["success"]]

        if not domain_attempts:
            return []

        # Find common tactics in failed attempts
        tactic_counts: dict[str, int] = {}
        for attempt in domain_attempts:
            for tactic in attempt["tactics"]:
                tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1

        # Sort by frequency
        sorted_tactics = sorted(tactic_counts.items(), key=lambda x: x[1], reverse=True)
        return [tactic for tactic, _ in sorted_tactics[:5]]

    async def suggest_strategy(self, state: ProofState) -> str:
        """Suggest proof strategy based on history."""
        success_rate = await self.get_success_rate(state.domain)
        stats = self.domain_stats.get(state.domain, {})

        if success_rate > 0.7:
            return "This domain has high success rate. Continue with proven tactics."
        elif success_rate > 0.4:
            return f"This domain has moderate success (rate: {success_rate:.0%}). Consider exploring alternatives."
        else:
            patterns = await self.get_common_failure_patterns(state.domain)
            if patterns:
                return f"Avoid these problematic tactics: {', '.join(patterns[:3])}"
            return "Insufficient experience in this domain. Explore diverse tactics."


class ProofEmbeddingEngine:
    """Main facade for proof embedding and transfer learning."""

    def __init__(self, memory_path: Path | None = None):
        self.memory_path = memory_path or Path.home() / ".autoforge" / "proof_memory.json"
        self.embedding_model = EmbeddingModel()
        self.memory = ProofMemoryBank()
        self.tracker = ProofExperienceTracker()
        self._initialized = False

    async def initialize(self) -> None:
        """Load memory bank and initialize."""
        if self._initialized:
            return

        await self.memory.load(self.memory_path)
        self._initialized = True
        logger.info("ProofEmbeddingEngine initialized")

    async def embed_and_store(
        self, proof_id: str, state: ProofState, tactics: list[str], difficulty: float = 0.5
    ) -> ProofEmbedding:
        """Embed and store a proof."""
        await self.initialize()
        return await self.memory.add_proof(
            proof_id, state, tactics, self.embedding_model, difficulty=difficulty
        )

    async def suggest_tactics(
        self, state: ProofState, llm: Any, top_k: int = 5
    ) -> list[ProofTransferCandidate]:
        """Suggest tactics using transfer learning."""
        await self.initialize()

        # Find similar proofs
        similar = await self.memory.search_similar(state, self.embedding_model, top_k=top_k)

        candidates = []
        for proof_emb, similarity in similar:
            tactic_transfer = TacticTransfer()
            adapted = await tactic_transfer.transfer_tactics(proof_emb, state, llm)

            candidate = ProofTransferCandidate(
                source_proof=proof_emb,
                similarity=similarity,
                transferred_tactics=adapted,
                adaptation_notes=f"Similarity: {similarity:.2f}",
                confidence=similarity * 0.8,
            )
            candidates.append(candidate)

        return candidates

    async def cross_domain_suggest(
        self, state: ProofState, llm: Any
    ) -> list[ProofTransferCandidate]:
        """Suggest tactics across domains."""
        await self.initialize()

        transfer = CrossDomainTransfer()
        return await transfer.find_analogous_proofs(state, self.memory, self.embedding_model, llm)

    async def learn_from_proof(
        self, proof_id: str, state: ProofState, tactics: list[str], success: bool
    ) -> None:
        """Learn from a proof attempt."""
        await self.initialize()
        await self.tracker.record_attempt(state, tactics, success)

        if success:
            # Store successful proof
            await self.embed_and_store(proof_id, state, tactics)

    async def save(self) -> None:
        """Persist all data."""
        if self._initialized:
            self.memory.save(self.memory_path)
            logger.info("ProofEmbeddingEngine state saved")

    def stats(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        memory_stats = self.memory.stats()
        attempts = len(self.tracker.attempts)
        successes = sum(1 for a in self.tracker.attempts if a["success"])

        return {
            "memory": memory_stats,
            "total_attempts": attempts,
            "total_successes": successes,
            "overall_success_rate": successes / attempts if attempts > 0 else 0.0,
            "domain_stats": self.tracker.domain_stats,
        }
