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
    """Embedding model for proof states and tactics with fallback strategies.

    Supports three embedding backends (in order of preference):
      1. sentence-transformers (all-MiniLM-L6-v2): Real semantic embeddings
      2. TF-IDF with BM25 scoring: Lightweight fallback
      3. Hash-based deterministic embedding: Minimal dependencies

    Algorithm tracks success rate of retrieval vs. direct LLM generation.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._model = None
        self._use_sentence_transformers = False
        self._tfidf_vectorizer = None
        self._use_tfidf = False
        self._algorithm_ratio = 0.5  # Ratio of success for algorithm-based retrieval
        self._retrieval_successes = 0
        self._retrieval_attempts = 0

    async def _ensure_loaded(self) -> None:
        """Lazily load embedding model on first use with fallback chain."""
        if self._model is not None or self._tfidf_vectorizer is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformers for proof embeddings")
            device = "cpu"
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            self._use_sentence_transformers = True
            logger.info("Sentence-transformers model loaded")
        except ImportError:
            logger.debug("sentence-transformers not available, trying TF-IDF fallback")
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.embedding_dim,
                    lowercase=True,
                    stop_words="english",
                )
                self._use_tfidf = True
                logger.info("TF-IDF embedding fallback initialized")
            except ImportError:
                logger.info("Neither sentence-transformers nor sklearn available; using hash-based fallback")
                self._use_sentence_transformers = False

    async def embed_goal(self, goal: str) -> list[float]:
        """Embed a proof goal using available embedding backend."""
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
        elif self._use_tfidf:
            return self._tfidf_embed(goal)
        else:
            return self._hash_embed(goal)

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
        """TF-IDF based embedding using sklearn."""
        if not self._tfidf_vectorizer:
            return self._hash_embed(text)

        try:
            vector = self._tfidf_vectorizer.fit_transform([text])
            dense = vector.toarray()[0]

            # Resize to embedding_dim
            if len(dense) < self.embedding_dim:
                dense = list(dense) + [0.0] * (self.embedding_dim - len(dense))
            else:
                dense = dense[: self.embedding_dim]

            # Normalize
            norm = math.sqrt(sum(x ** 2 for x in dense))
            if norm > 0:
                dense = [x / norm for x in dense]

            return dense
        except Exception as e:
            logger.debug(f"TF-IDF embedding failed: {e}, falling back to hash")
            return self._hash_embed(text)

    def _hash_embed(self, text: str) -> list[float]:
        """Hash-based deterministic embedding (minimal dependencies)."""
        import hashlib

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

    def record_retrieval_success(self, success: bool) -> None:
        """Track retrieval success for algorithm ratio calculation."""
        self._retrieval_attempts += 1
        if success:
            self._retrieval_successes += 1
        self._algorithm_ratio = (
            self._retrieval_successes / max(1, self._retrieval_attempts)
        )

    def get_stats(self) -> dict[str, Any]:
        """Get embedding model statistics."""
        backend = "sentence-transformers" if self._use_sentence_transformers else \
                  "tfidf" if self._use_tfidf else "hash"
        return {
            "backend": backend,
            "embedding_dim": self.embedding_dim,
            "algorithm_ratio": self._algorithm_ratio,
            "retrieval_successes": self._retrieval_successes,
            "retrieval_attempts": self._retrieval_attempts,
        }


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
        self._retrieval_successes = 0
        self._retrieval_attempts = 0
        self._algorithm_ratio = 0.5

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
        """Find similar proofs using semantic similarity.

        Uses FAISS for fast approximate search if available (O(log n)),
        otherwise falls back to linear search with cosine similarity (O(n)).

        Algorithm tracks success rate of retrieval-based suggestions.
        """
        if not self.proofs:
            return []

        # Embed query state
        query_emb = await embedding_model.embed_proof_state(query_state)
        query_vec = np.array(query_emb, dtype=np.float32)

        results: list[tuple[ProofEmbedding, float]] = []

        if self._faiss_index is not None:
            # Use FAISS for fast search (O(log n) amortized)
            distances, indices = self._faiss_index.search(query_vec.reshape(1, -1), min(top_k, len(self.proofs)))
            for idx, distance in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.embedding_ids):
                    proof_id = self.embedding_ids[idx]
                    if proof_id in self.proofs:
                        # Convert distance to similarity (lower distance = higher similarity)
                        similarity = 1.0 / (1.0 + float(distance))
                        results.append((self.proofs[proof_id], similarity))
        else:
            # Linear search with cosine similarity (O(n))
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

        # Track retrieval success (high similarity suggests useful match)
        if results:
            best_similarity = results[0][1]
            self.record_retrieval_success(best_similarity > 0.5)

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
            "algorithm_ratio": self._algorithm_ratio,
            "retrieval_successes": self._retrieval_successes,
            "retrieval_attempts": self._retrieval_attempts,
        }

    def record_retrieval_success(self, success: bool) -> None:
        """Track success of retrieval-based tactic suggestion."""
        self._retrieval_attempts += 1
        if success:
            self._retrieval_successes += 1
        self._algorithm_ratio = (
            self._retrieval_successes / max(1, self._retrieval_attempts)
        )


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


@dataclass
class TacticPair:
    """Represents a preference pair for DPO training: chosen vs rejected tactic."""

    proof_state: ProofState
    """The proof state where the choice was made."""

    chosen_tactic: str
    """The preferred tactic (successful or more efficient)."""

    rejected_tactic: str
    """The rejected tactic (failed or less efficient)."""

    chosen_success: bool
    """Whether the chosen tactic succeeded."""

    rejected_success: bool
    """Whether the rejected tactic succeeded."""

    state_embedding: list[float]
    """Embedding of the proof state for similarity computation."""

    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    """When this pair was collected."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (domain, difficulty, depth, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "proof_state": self.proof_state.to_dict(),
            "chosen_tactic": self.chosen_tactic,
            "rejected_tactic": self.rejected_tactic,
            "chosen_success": self.chosen_success,
            "rejected_success": self.rejected_success,
            "state_embedding": self.state_embedding,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""

    beta: float = 0.1
    """KL constraint weight (lower = stricter KL penalty vs preference fitting)."""

    reference_model_weight: float = 0.5
    """Weight for reference model logits (0=ignore ref, 1=fully use ref)."""

    max_pairs_per_batch: int = 32
    """Maximum preference pairs per DPO batch."""

    preference_margin: float = 0.1
    """Minimum preference strength required to distinguish tactics."""

    min_pairs_for_training: int = 10
    """Minimum collected pairs before enabling DPO ranking."""


class DPOTacticOptimizer:
    """Direct Preference Optimization for tactic selection in proof search.

    Inspired by BFS-Prover-V2's state-tactic DPO approach. Learns to rank
    tactics given proof states from preference pairs collected during proof
    search (successful vs failed paths). Key advantage: no separate reward
    model needed, preferences optimized directly via DPO loss.
    """

    def __init__(
        self,
        config: DPOConfig | None = None,
        embedding_model: EmbeddingModel | None = None,
        memory_bank: ProofMemoryBank | None = None,
        llm: Any = None,
    ):
        """Initialize DPO optimizer.

        Args:
            config: DPO configuration (defaults to DPOConfig()).
            embedding_model: Embedding model for state vectors.
            memory_bank: Proof memory bank for retrieval context.
            llm: LLM for tactic scoring (optional).
        """
        self.config = config or DPOConfig()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.memory_bank = memory_bank or ProofMemoryBank()
        self.llm = llm

        self._preference_pairs: list[TacticPair] = []
        self._tactic_scores: dict[str, float] = {}  # Running tactic quality scores
        self._state_preference_matrix: dict[str, dict[str, float]] = {}  # State -> tactic -> score
        self._pair_counter = 0

        logger.info(f"DPOTacticOptimizer initialized with beta={self.config.beta}")

    async def collect_pair(
        self, state: ProofState, successful_tactic: str, failed_tactic: str
    ) -> None:
        """Record a preference pair from proof search experience.

        Args:
            state: The proof state.
            successful_tactic: Tactic that succeeded (or is preferred).
            failed_tactic: Tactic that failed (or is rejected).
        """
        state_embedding = await self.embedding_model.embed_state(state)

        pair = TacticPair(
            proof_state=state,
            chosen_tactic=successful_tactic,
            rejected_tactic=failed_tactic,
            chosen_success=True,
            rejected_success=False,
            state_embedding=state_embedding,
            metadata={
                "domain": state.domain,
                "depth": state.depth,
                "num_hypotheses": len(state.hypotheses),
            },
        )

        self._preference_pairs.append(pair)
        self._pair_counter += 1

        logger.debug(
            f"Collected pair {self._pair_counter}: chosen='{successful_tactic}', "
            f"rejected='{failed_tactic}' in domain '{state.domain}'"
        )

    async def collect_pairs_from_search(
        self,
        proof_states: list[ProofState],
        tactic_sequences: list[list[str]],
        outcomes: list[bool],
    ) -> int:
        """Batch collection of preference pairs from a completed proof search.

        For each outcome, pairs are created comparing successful vs failed paths.

        Args:
            proof_states: States visited during search.
            tactic_sequences: Tactic sequences for each state.
            outcomes: Success/failure for each sequence.

        Returns:
            Number of new pairs collected.
        """
        if len(proof_states) != len(tactic_sequences) or len(tactic_sequences) != len(outcomes):
            logger.warning("Mismatched array lengths in collect_pairs_from_search")
            return 0

        initial_count = len(self._preference_pairs)

        # Partition by outcome
        successful_idx = [i for i, outcome in enumerate(outcomes) if outcome]
        failed_idx = [i for i, outcome in enumerate(outcomes) if not outcome]

        # Create pairs between successful and failed paths at same depth
        for succ_i in successful_idx:
            for fail_i in failed_idx:
                succ_state = proof_states[succ_i]
                fail_state = proof_states[fail_i]

                # Pair states at similar depths for fair comparison
                if abs(succ_state.depth - fail_state.depth) <= 1:
                    succ_tactics = tactic_sequences[succ_i]
                    fail_tactics = tactic_sequences[fail_i]

                    # Use last tactic applied as the decision point
                    if succ_tactics and fail_tactics:
                        await self.collect_pair(
                            succ_state,
                            successful_tactic=succ_tactics[-1],
                            failed_tactic=fail_tactics[-1],
                        )

        new_pairs = len(self._preference_pairs) - initial_count
        logger.info(f"Collected {new_pairs} pairs from search (total: {len(self._preference_pairs)})")
        return new_pairs

    def _compute_implicit_reward(self, tactic: str, state: ProofState) -> float:
        """Compute heuristic reward for a tactic without formal verification.

        Uses domain knowledge, tactic frequency, and state properties.

        Args:
            tactic: The tactic to evaluate.
            state: The proof state context.

        Returns:
            Reward score (higher = better).
        """
        reward = 0.0

        # Domain-specific base scores
        domain_affinity = {
            "algebra": {"ring": 1.0, "field_simp": 0.9, "norm_num": 0.8},
            "topology": {"continuity": 1.0, "open": 0.9, "compact": 0.85},
            "number_theory": {"norm_num": 1.0, "omega": 0.9, "prime": 0.8},
        }

        if state.domain in domain_affinity:
            reward = domain_affinity[state.domain].get(tactic, 0.5)

        # Historical success rate
        if tactic in self._tactic_scores:
            reward += self._tactic_scores[tactic] * 0.3

        # Penalize repetition in recent history
        if tactic in state.tactic_history[-3:]:
            reward *= 0.7

        # Depth consideration
        if state.depth > 10:
            reward *= (1 - state.depth * 0.05)  # Discount for deep searches

        return max(0.0, reward)

    async def compute_dpo_loss(self, batch: list[TacticPair]) -> float:
        """Compute DPO loss for a batch of preference pairs.

        DPO loss: E[log σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))]

        Where:
        - π: current policy (tactic predictor)
        - π_ref: reference policy (prior)
        - y_w: chosen (winning) tactic
        - y_l: rejected (losing) tactic
        - β: KL penalty weight

        Args:
            batch: List of TacticPair instances.

        Returns:
            Scalar DPO loss value.
        """
        if not batch:
            return 0.0

        total_loss = 0.0

        for pair in batch:
            # Get implicit rewards (serves as reference policy)
            ref_reward_w = self._compute_implicit_reward(pair.chosen_tactic, pair.proof_state)
            ref_reward_l = self._compute_implicit_reward(
                pair.rejected_tactic, pair.proof_state
            )

            # Log probabilities under reference policy
            log_pi_ref_w = math.log(max(ref_reward_w, 1e-6))
            log_pi_ref_l = math.log(max(ref_reward_l, 1e-6))

            # Log probabilities under current policy (learned from embeddings)
            state_key = f"{pair.proof_state.goal}:{pair.proof_state.depth}"
            policy_score_w = self._state_preference_matrix.get(state_key, {}).get(
                pair.chosen_tactic, 0.5
            )
            policy_score_l = self._state_preference_matrix.get(state_key, {}).get(
                pair.rejected_tactic, 0.5
            )

            log_pi_w = math.log(max(policy_score_w, 1e-6))
            log_pi_l = math.log(max(policy_score_l, 1e-6))

            # DPO loss computation
            policy_logits = self.config.beta * (log_pi_w - log_pi_l)
            ref_logits = log_pi_ref_w - log_pi_ref_l

            # Sigmoid of difference
            dpo_score = 1.0 / (1.0 + math.exp(-(policy_logits - ref_logits)))

            # Loss is negative log of sigmoid (cross-entropy)
            pair_loss = -math.log(max(dpo_score, 1e-6))
            total_loss += pair_loss

        return total_loss / len(batch) if batch else 0.0

    async def rank_tactics(self, state: ProofState, candidate_tactics: list[str]) -> list[tuple[str, float]]:
        """Rank tactic candidates using learned DPO preferences.

        Args:
            state: Current proof state.
            candidate_tactics: List of tactic strings to rank.

        Returns:
            List of (tactic, score) tuples sorted by score (highest first).
        """
        if not candidate_tactics:
            return []

        # Check if we have enough pairs for reliable ranking
        if len(self._preference_pairs) < self.config.min_pairs_for_training:
            logger.debug(
                f"Insufficient pairs ({len(self._preference_pairs)} < "
                f"{self.config.min_pairs_for_training}) for DPO ranking"
            )
            # Fall back to implicit rewards
            return [
                (tactic, self._compute_implicit_reward(tactic, state))
                for tactic in candidate_tactics
            ]

        ranked = []
        state_key = f"{state.goal}:{state.depth}"

        for tactic in candidate_tactics:
            # Combine DPO-learned score with similarity-weighted preferences
            dpo_score = self._state_preference_matrix.get(state_key, {}).get(tactic, 0.5)
            implicit_reward = self._compute_implicit_reward(tactic, state)

            # Weight blend: DPO (if available) + implicit reward
            final_score = (
                0.7 * dpo_score + 0.3 * implicit_reward
                if len(self._preference_pairs) >= self.config.min_pairs_for_training
                else implicit_reward
            )

            ranked.append((tactic, final_score))

        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Ranked {len(ranked)} tactics for state depth={state.depth}")
        return ranked

    async def suggest_from_preferences(
        self, state: ProofState, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Suggest tactics based on DPO-learned preferences.

        Args:
            state: Current proof state.
            top_k: Number of suggestions to return.

        Returns:
            List of (tactic, confidence) tuples.
        """
        if len(self._preference_pairs) < self.config.min_pairs_for_training:
            logger.debug(
                f"Insufficient experience ({len(self._preference_pairs)} pairs) "
                f"for preference-based suggestions"
            )
            return []

        # Collect all tactics seen in preferences for this state context
        candidate_tactics = set()
        state_key = f"{state.goal}:{state.depth}"

        for pair in self._preference_pairs:
            # Include tactics from similar proof states
            similarity = self._similarity_weighted_preference(state, pair)
            if similarity > 0.3:
                candidate_tactics.add(pair.chosen_tactic)
                candidate_tactics.add(pair.rejected_tactic)

        if not candidate_tactics:
            return []

        # Rank candidates
        ranked = await self.rank_tactics(state, list(candidate_tactics))

        return ranked[:top_k]

    def _similarity_weighted_preference(self, query_state: ProofState, pair: TacticPair) -> float:
        """Compute similarity weight between query state and preference pair state.

        Uses embedding cosine similarity and structural properties.

        Args:
            query_state: The state we're querying.
            pair: The preference pair to weight.

        Returns:
            Similarity score in [0, 1].
        """
        # Embedding-based similarity
        query_embedding = np.array(
            asyncio.get_event_loop().run_until_complete(
                self.embedding_model.embed_state(query_state)
            )
        )
        pair_embedding = np.array(pair.state_embedding)

        # Cosine similarity
        norm_q = np.linalg.norm(query_embedding)
        norm_p = np.linalg.norm(pair_embedding)

        if norm_q < 1e-6 or norm_p < 1e-6:
            embedding_sim = 0.0
        else:
            embedding_sim = float(np.dot(query_embedding, pair_embedding) / (norm_q * norm_p))

        # Structural similarity
        domain_match = 1.0 if query_state.domain == pair.proof_state.domain else 0.5
        depth_proximity = max(0.0, 1.0 - abs(query_state.depth - pair.proof_state.depth) * 0.1)
        hypothesis_overlap = (
            len(set(query_state.hypotheses) & set(pair.proof_state.hypotheses))
            / max(len(query_state.hypotheses), len(pair.proof_state.hypotheses), 1)
        )

        # Weighted combination
        total_similarity = (
            0.5 * embedding_sim
            + 0.2 * domain_match
            + 0.15 * depth_proximity
            + 0.15 * hypothesis_overlap
        )

        return max(0.0, min(1.0, total_similarity))

    def save(self, path: Path) -> None:
        """Persist DPO optimizer state to disk.

        Args:
            path: Path to save JSON file.
        """
        data = {
            "config": {
                "beta": self.config.beta,
                "reference_model_weight": self.config.reference_model_weight,
                "max_pairs_per_batch": self.config.max_pairs_per_batch,
                "preference_margin": self.config.preference_margin,
                "min_pairs_for_training": self.config.min_pairs_for_training,
            },
            "preference_pairs": [pair.to_dict() for pair in self._preference_pairs],
            "tactic_scores": self._tactic_scores,
            "pair_counter": self._pair_counter,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"DPOTacticOptimizer saved to {path} ({len(self._preference_pairs)} pairs)")

    async def load(self, path: Path) -> None:
        """Load DPO optimizer state from disk.

        Args:
            path: Path to load JSON file from.
        """
        if not path.exists():
            logger.warning(f"DPOTacticOptimizer path does not exist: {path}")
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Restore config
            cfg_data = data.get("config", {})
            self.config = DPOConfig(
                beta=cfg_data.get("beta", 0.1),
                reference_model_weight=cfg_data.get("reference_model_weight", 0.5),
                max_pairs_per_batch=cfg_data.get("max_pairs_per_batch", 32),
                preference_margin=cfg_data.get("preference_margin", 0.1),
                min_pairs_for_training=cfg_data.get("min_pairs_for_training", 10),
            )

            # Restore pairs (note: limited reconstruction without full state objects)
            self._pair_counter = data.get("pair_counter", 0)
            self._tactic_scores = data.get("tactic_scores", {})

            logger.info(f"DPOTacticOptimizer loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load DPOTacticOptimizer: {e}")

    def stats(self) -> dict[str, Any]:
        """Get statistics about collected pairs and preference quality.

        Returns:
            Dictionary with comprehensive statistics.
        """
        if not self._preference_pairs:
            return {
                "total_pairs": 0,
                "total_pairs_collected": self._pair_counter,
                "avg_state_depth": 0.0,
                "domain_distribution": {},
                "tactic_frequencies": {},
                "success_rate": 0.0,
                "unique_tactics": 0,
            }

        depths = [pair.proof_state.depth for pair in self._preference_pairs]
        domains = [pair.proof_state.domain for pair in self._preference_pairs]
        successes = sum(1 for pair in self._preference_pairs if pair.chosen_success)

        # Tactic frequencies
        tactic_freq: dict[str, int] = {}
        for pair in self._preference_pairs:
            tactic_freq[pair.chosen_tactic] = tactic_freq.get(pair.chosen_tactic, 0) + 1
            tactic_freq[pair.rejected_tactic] = tactic_freq.get(pair.rejected_tactic, 0) + 1

        # Domain distribution
        domain_dist: dict[str, int] = {}
        for domain in domains:
            domain_dist[domain] = domain_dist.get(domain, 0) + 1

        return {
            "total_pairs": len(self._preference_pairs),
            "total_pairs_collected": self._pair_counter,
            "avg_state_depth": float(np.mean(depths)) if depths else 0.0,
            "max_state_depth": int(max(depths)) if depths else 0,
            "domain_distribution": domain_dist,
            "tactic_frequencies": dict(sorted(tactic_freq.items(), key=lambda x: x[1], reverse=True)),
            "success_rate": successes / len(self._preference_pairs) if self._preference_pairs else 0.0,
            "unique_tactics": len(tactic_freq),
            "avg_pairs_per_domain": (
                len(self._preference_pairs) / len(set(domains)) if domains else 0
            ),
        }
