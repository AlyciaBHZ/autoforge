"""
Dense embedding-based premise selection for theorem proving.

Inspired by ReProver (NeurIPS 2023) and LeanDojo (ICLR 2024).
Uses various embedding backends (sentence-transformers, TF-IDF, LLM) for
efficient semantic retrieval of relevant lemmas and theorems.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingBackend(str, Enum):
    """Available embedding backends."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    LOCAL_TFIDF = "local_tfidf"
    LLM_EMBED = "llm_embed"


@dataclass
class PremiseEntry:
    """A single premise/lemma with metadata and cached embedding."""
    id: str
    name: str
    statement: str
    module_path: str
    docstring: str = ""
    embedding: list[float] = field(default_factory=list)
    usage_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "id": self.id,
            "name": self.name,
            "statement": self.statement,
            "module_path": self.module_path,
            "docstring": self.docstring,
            "embedding": self.embedding,
            "usage_count": self.usage_count,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PremiseEntry:
        """Load from serializable dict."""
        return PremiseEntry(
            id=data["id"],
            name=data["name"],
            statement=data["statement"],
            module_path=data["module_path"],
            docstring=data.get("docstring", ""),
            embedding=data.get("embedding", []),
            usage_count=data.get("usage_count", 0),
        )


@dataclass
class RetrievalConfig:
    """Configuration for dense retrieval."""
    backend: EmbeddingBackend = EmbeddingBackend.LOCAL_TFIDF
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    top_k: int = 20
    similarity_threshold: float = 0.3
    cache_dir: Path = field(default_factory=lambda: Path(".autoforge/embeddings"))
    batch_size: int = 64
    use_gpu: bool = False


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts asynchronously."""
        pass

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._model = None
        self._dimension = None

    async def _ensure_loaded(self) -> None:
        """Lazily load the model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            device = "cuda" if self.use_gpu else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
            # Infer dimension from a test embedding
            test_embedding = self._model.encode(["test"], show_progress_bar=False)
            self._dimension = test_embedding.shape[1]
            logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Falling back to TF-IDF provider."
            )
            raise RuntimeError("sentence-transformers required for this backend")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts using sentence-transformers."""
        await self._ensure_loaded()
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, show_progress_bar=False).tolist()
        )
        return embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        if self._dimension is None:
            raise RuntimeError("Model not loaded yet. Call _ensure_loaded() first.")
        return self._dimension


class TFIDFProvider(EmbeddingProvider):
    """Pure Python TF-IDF based embedding provider with fallback."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._vectorizer = None
        self._tfidf_matrix = None
        self._svd = None
        self._corpus = []

    def _build_tfidf_matrix(self, texts: list[str]) -> np.ndarray:
        """Build TF-IDF matrix using pure Python or sklearn."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD

            logger.info("Using sklearn for TF-IDF computation")
            self._vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 2),
                token_pattern=r"\b\w{2,}\b",
            )
            tfidf = self._vectorizer.fit_transform(texts)
            tfidf_array = tfidf.toarray()

            if tfidf_array.shape[1] > self.embedding_dim:
                logger.info(f"Reducing dimensions from {tfidf_array.shape[1]} to {self.embedding_dim}")
                self._svd = TruncatedSVD(n_components=self.embedding_dim)
                return self._svd.fit_transform(tfidf_array).astype(np.float32)

            # Pad with zeros if dimension is smaller
            if tfidf_array.shape[1] < self.embedding_dim:
                padded = np.zeros((tfidf_array.shape[0], self.embedding_dim), dtype=np.float32)
                padded[:, :tfidf_array.shape[1]] = tfidf_array
                return padded

            return tfidf_array.astype(np.float32)
        except ImportError:
            logger.warning("sklearn not available. Using pure Python TF-IDF.")
            return self._pure_python_tfidf(texts)

    def _pure_python_tfidf(self, texts: list[str]) -> np.ndarray:
        """Pure Python TF-IDF implementation without sklearn."""
        # Simple tokenization
        corpus_tokens = []
        vocab = {}
        for text in texts:
            tokens = re.findall(r"\b\w{2,}\b", text.lower())
            corpus_tokens.append(tokens)
            for token in tokens:
                vocab[token] = vocab.get(token, 0) + 1

        # Remove low-frequency tokens
        min_df = max(1, len(texts) // 100)
        vocab = {k: v for k, v in vocab.items() if v >= min_df and v <= len(texts) * 0.95}
        token_to_id = {token: i for i, token in enumerate(sorted(vocab.keys()))}

        # Build TF-IDF matrix
        n_docs = len(texts)
        vocab_size = len(token_to_id)
        embeddings = np.zeros((n_docs, min(vocab_size, self.embedding_dim)), dtype=np.float32)

        for doc_id, tokens in enumerate(corpus_tokens):
            # Term frequency
            tf = {}
            for token in tokens:
                if token in token_to_id:
                    tf[token] = tf.get(token, 0) + 1

            # TF-IDF calculation
            for token, count in tf.items():
                token_id = token_to_id[token]
                if token_id < self.embedding_dim:
                    # IDF: log(total_docs / docs_with_token)
                    docs_with_token = sum(1 for t in corpus_tokens if token in t)
                    idf = math.log(n_docs / (1 + docs_with_token))
                    embeddings[doc_id, token_id] = (count / len(tokens)) * idf

        # Normalize rows
        for i in range(embeddings.shape[0]):
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm

        # Pad if necessary
        if embeddings.shape[1] < self.embedding_dim:
            padded = np.zeros((n_docs, self.embedding_dim), dtype=np.float32)
            padded[:, :embeddings.shape[1]] = embeddings
            embeddings = padded

        return embeddings

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts using TF-IDF."""
        if not texts:
            return []

        # For batches, we need to update the corpus and recompute
        loop = asyncio.get_event_loop()

        async def _compute():
            # Combine with existing corpus for consistency
            all_texts = self._corpus + texts
            matrix = self._build_tfidf_matrix(all_texts)
            # Return only the new embeddings
            return matrix[-len(texts):].tolist()

        return await loop.run_in_executor(None, lambda: asyncio.run(_compute()))

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def initialize_corpus(self, texts: list[str]) -> None:
        """Initialize with full corpus for consistent embeddings."""
        loop = asyncio.get_event_loop()
        self._corpus = texts.copy()

        async def _build():
            self._tfidf_matrix = self._build_tfidf_matrix(texts)

        await loop.run_in_executor(None, lambda: asyncio.run(_build()))

    async def get_corpus_embeddings(self) -> list[list[float]]:
        """Get embeddings for the entire corpus."""
        if self._tfidf_matrix is None:
            return []
        return self._tfidf_matrix.tolist()

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim


class LLMEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using LLM-generated embeddings."""

    def __init__(self, llm: Any, embedding_dim: int = 384):
        self.llm = llm
        self.embedding_dim = embedding_dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using LLM."""
        from autoforge.engine.llm_router import TaskComplexity

        embeddings = []
        for text in texts:
            embedding = await self.embed_single(text)
            embeddings.append(embedding)
        return embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text using LLM."""
        from autoforge.engine.llm_router import TaskComplexity

        prompt = f"""Generate an embedding for the following text as a JSON array of {self.embedding_dim} numbers between -1 and 1.
Return ONLY the JSON array, no other text.

Text: {text[:500]}

Embedding:"""

        response = await self.llm.call(prompt, complexity=TaskComplexity.MEDIUM)
        content = response.content if hasattr(response, "content") else str(response)

        try:
            # Try to extract JSON array from response
            match = re.search(r"\[-?[\d.,\s]+\]", content)
            if match:
                embedding = json.loads(match.group())
                if len(embedding) == self.embedding_dim:
                    return embedding
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: create deterministic hash-based embedding
        logger.warning("Failed to parse LLM embedding, using hash-based fallback")
        return self._hash_based_embedding(text)

    def _hash_based_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding based on text hash."""
        import hashlib

        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        embedding = []
        for i in range(self.embedding_dim):
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Convert to float between -1 and 1
            value = (byte_val / 128.0) - 1.0
            embedding.append(value)

        # Normalize
        norm = math.sqrt(sum(x ** 2 for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim


class PremiseIndex:
    """Index for fast semantic premise retrieval."""

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.premises: list[PremiseEntry] = []
        self.embeddings: np.ndarray | None = None
        self._faiss_index = None
        self._use_faiss = False
        logger.info(f"Initializing PremiseIndex with backend: {config.backend}")

    async def build_index(self, premises: list[PremiseEntry], provider: EmbeddingProvider) -> None:
        """Build index from premises."""
        if not premises:
            logger.warning("No premises provided for indexing")
            return

        logger.info(f"Building index for {len(premises)} premises")
        self.premises = premises

        # Extract statements for embedding
        statements = [f"{p.name}: {p.statement}" for p in premises]

        # Embed all at once for TF-IDF provider initialization
        if isinstance(provider, TFIDFProvider):
            await provider.initialize_corpus(statements)
            embeddings_list = await provider.get_corpus_embeddings()
        else:
            # Embed in batches
            embeddings_list = []
            for i in range(0, len(statements), self.config.batch_size):
                batch = statements[i : i + self.config.batch_size]
                batch_embeddings = await provider.embed_batch(batch)
                embeddings_list.extend(batch_embeddings)
                logger.debug(f"Embedded batch {i // self.config.batch_size + 1}")

        # Store embeddings and update premises
        self.embeddings = np.array(embeddings_list, dtype=np.float32)

        # Normalize embeddings for cosine similarity
        for i in range(len(self.embeddings)):
            self.embeddings[i] = self._normalize(self.embeddings[i])

        # Store embeddings in premises
        for i, premise in enumerate(self.premises):
            premise.embedding = self.embeddings[i].tolist()

        # Try to initialize FAISS for large indices
        if len(premises) > 100000:
            self._try_init_faiss()

        logger.info(f"Index built successfully. Shape: {self.embeddings.shape}")

    def _try_init_faiss(self) -> None:
        """Try to initialize FAISS index for fast similarity search."""
        try:
            import faiss

            logger.info("Initializing FAISS index for fast search")
            self._faiss_index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self._faiss_index.add(self.embeddings)
            self._use_faiss = True
        except ImportError:
            logger.info("FAISS not available, using numpy for search")
            self._use_faiss = False

    async def search(
        self, query: str, top_k: int = 0, provider: EmbeddingProvider | None = None
    ) -> list[tuple[PremiseEntry, float]]:
        """Search for relevant premises."""
        if top_k <= 0:
            top_k = self.config.top_k

        if not self.premises or self.embeddings is None:
            logger.warning("Index is empty")
            return []

        if provider is None:
            logger.warning("No provider given, cannot search without embedding")
            return []

        # Embed query
        query_embedding = await provider.embed_single(query)
        query_embedding = self._normalize(np.array(query_embedding, dtype=np.float32))

        if self._use_faiss:
            return self._search_faiss(query_embedding, top_k)
        else:
            return self._search_numpy(query_embedding, top_k)

    def _search_numpy(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[PremiseEntry, float]]:
        """Search using numpy (cosine similarity)."""
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(-similarities)[:top_k]

        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity >= self.config.similarity_threshold:
                results.append((self.premises[idx], similarity))

        return results

    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[PremiseEntry, float]]:
        """Search using FAISS."""
        # FAISS uses L2 distance, convert to similarity
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self._faiss_index.search(query_embedding, top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            # Convert L2 distance to cosine similarity approximation
            similarity = 1.0 / (1.0 + float(distance))
            if similarity >= self.config.similarity_threshold:
                results.append((self.premises[int(idx)], similarity))

        return results

    async def search_batch(
        self, queries: list[str], top_k: int = 0, provider: EmbeddingProvider | None = None
    ) -> list[list[tuple[PremiseEntry, float]]]:
        """Search for multiple queries."""
        results = []
        for query in queries:
            result = await self.search(query, top_k, provider)
            results.append(result)
        return results

    async def add_premise(self, premise: PremiseEntry, provider: EmbeddingProvider) -> None:
        """Add a single premise to the index."""
        # Embed the new premise
        statement = f"{premise.name}: {premise.statement}"
        embedding = await provider.embed_single(statement)
        embedding = self._normalize(np.array(embedding, dtype=np.float32))

        # Add to index
        self.premises.append(premise)
        premise.embedding = embedding.tolist()

        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])

        # Update FAISS if in use
        if self._use_faiss and self._faiss_index is not None:
            self._faiss_index.add(embedding.reshape(1, -1))

        logger.debug(f"Added premise: {premise.name}")

    def save(self, path: Path) -> None:
        """Save index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "premises": [p.to_dict() for p in self.premises],
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
            "config": {
                "backend": self.config.backend.value,
                "embedding_dim": self.config.embedding_dim,
                "top_k": self.config.top_k,
                "similarity_threshold": self.config.similarity_threshold,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Index saved to {path}")

    async def load(self, path: Path) -> None:
        """Load index from disk."""
        if not path.exists():
            logger.warning(f"Index file not found: {path}")
            return

        with open(path, "r") as f:
            data = json.load(f)

        self.premises = [PremiseEntry.from_dict(p) for p in data["premises"]]

        if data["embeddings"]:
            self.embeddings = np.array(data["embeddings"], dtype=np.float32)

        logger.info(f"Loaded index from {path} with {len(self.premises)} premises")

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    @staticmethod
    def _normalize(vec: np.ndarray | list[float]) -> np.ndarray:
        """Normalize vector to unit length."""
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class MathlibPremiseLoader:
    """Load premises from Mathlib declarations."""

    @staticmethod
    async def load_from_declarations(mathlib_path: Path) -> list[PremiseEntry]:
        """Parse Lean declaration files to extract premises."""
        premises = []
        olean_dir = mathlib_path / "build" / "lib"

        if not olean_dir.exists():
            logger.warning(f"Mathlib build directory not found: {olean_dir}")
            return premises

        # In practice, this would parse .olean files or .lean source files
        # For now, return empty list as a placeholder
        logger.info(f"Scanning Mathlib at {mathlib_path}")

        return premises

    @staticmethod
    async def load_from_json(json_path: Path) -> list[PremiseEntry]:
        """Load pre-extracted premises from LeanDojo format JSON."""
        if not json_path.exists():
            logger.warning(f"JSON file not found: {json_path}")
            return []

        with open(json_path, "r") as f:
            data = json.load(f)

        premises = []
        for item in data:
            if isinstance(item, dict):
                premise = PremiseEntry(
                    id=item.get("id", item.get("name", "")),
                    name=item.get("name", ""),
                    statement=item.get("statement", item.get("type", "")),
                    module_path=item.get("module", item.get("file", "")),
                    docstring=item.get("docstring", ""),
                )
                premises.append(premise)

        logger.info(f"Loaded {len(premises)} premises from {json_path}")
        return premises

    @staticmethod
    async def load_from_lean_repl(lean_project: Any) -> list[PremiseEntry]:
        """Use Lean REPL to enumerate available declarations."""
        # This would require a running Lean REPL instance
        logger.warning("load_from_lean_repl not yet implemented")
        return []

    @staticmethod
    def _parse_lean_declaration(text: str) -> PremiseEntry | None:
        """Parse a single Lean declaration string."""
        # Extract name and statement from Lean code
        # Example: "theorem Nat.add_comm (n m : Nat) : n + m = m + n"

        match = re.match(r"(theorem|lemma|def)\s+(\S+)\s*(.*):\s*(.*)", text)
        if not match:
            return None

        kind, name, params, statement = match.groups()

        return PremiseEntry(
            id=name,
            name=name,
            statement=statement.strip(),
            module_path="",
            docstring=f"{kind} {params}",
        )


class DenseRetriever:
    """Main interface for dense semantic premise retrieval."""

    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()
        self._provider: EmbeddingProvider | None = None
        self._index: PremiseIndex | None = None
        self._stats = {
            "index_size": 0,
            "total_queries": 0,
            "total_hits": 0,
            "avg_latency_ms": 0.0,
        }

    async def initialize(
        self,
        premises: list[PremiseEntry] | None = None,
        mathlib_path: Path | None = None,
        llm: Any | None = None,
    ) -> None:
        """Initialize retriever with premises."""
        logger.info(f"Initializing DenseRetriever with backend: {self.config.backend}")

        # Initialize provider
        try:
            if self.config.backend == EmbeddingBackend.SENTENCE_TRANSFORMERS:
                self._provider = SentenceTransformerProvider(
                    model_name=self.config.model_name,
                    use_gpu=self.config.use_gpu,
                )
                await self._provider._ensure_loaded()
            elif self.config.backend == EmbeddingBackend.LOCAL_TFIDF:
                self._provider = TFIDFProvider(embedding_dim=self.config.embedding_dim)
            elif self.config.backend == EmbeddingBackend.LLM_EMBED:
                if llm is None:
                    logger.warning("LLM backend selected but no LLM provided, falling back to TF-IDF")
                    self._provider = TFIDFProvider(embedding_dim=self.config.embedding_dim)
                else:
                    self._provider = LLMEmbeddingProvider(llm, embedding_dim=self.config.embedding_dim)
            else:
                raise ValueError(f"Unknown backend: {self.config.backend}")
        except RuntimeError:
            logger.warning(f"Provider initialization failed, falling back to TF-IDF")
            self._provider = TFIDFProvider(embedding_dim=self.config.embedding_dim)

        # Initialize index
        self._index = PremiseIndex(self.config)

        # Load premises
        if premises is not None:
            await self._index.build_index(premises, self._provider)
            self._stats["index_size"] = len(premises)
        elif mathlib_path is not None:
            loader = MathlibPremiseLoader()
            premises = await loader.load_from_json(mathlib_path / "premises.json")
            if premises:
                await self._index.build_index(premises, self._provider)
                self._stats["index_size"] = len(premises)

        logger.info(f"Retriever initialized. Index size: {self._stats['index_size']}")

    async def retrieve_premises(
        self, goal_state: str, top_k: int = 0
    ) -> list[tuple[PremiseEntry, float]]:
        """Retrieve relevant premises for a proof goal."""
        if self._index is None or self._provider is None:
            logger.warning("Retriever not initialized")
            return []

        import time
        start = time.time()

        results = await self._index.search(goal_state, top_k, self._provider)

        elapsed_ms = (time.time() - start) * 1000
        self._stats["total_queries"] += 1
        self._stats["total_hits"] += len(results)
        self._stats["avg_latency_ms"] = (
            self._stats["avg_latency_ms"] * (self._stats["total_queries"] - 1) + elapsed_ms
        ) / self._stats["total_queries"]

        logger.debug(f"Retrieved {len(results)} premises in {elapsed_ms:.1f}ms")

        return results

    async def retrieve_for_tactic(
        self, goal: str, partial_tactic: str = ""
    ) -> list[str]:
        """Retrieve premise names suitable for tactic application."""
        premises = await self.retrieve_premises(goal + " " + partial_tactic)
        # Filter based on tactic type
        names = []
        for premise, score in premises:
            if score >= self.config.similarity_threshold:
                names.append(premise.name)
        return names

    async def rerank_with_llm(
        self, candidates: list[PremiseEntry], goal: str, llm: Any
    ) -> list[PremiseEntry]:
        """Use LLM to rerank top candidates."""
        from autoforge.engine.llm_router import TaskComplexity

        if not candidates:
            return candidates

        logger.info(f"Reranking {len(candidates)} candidates with LLM")

        # Build a concise list for the LLM
        candidate_text = "\n".join(
            [f"{i+1}. {p.name}: {p.statement[:100]}" for i, p in enumerate(candidates[:10])]
        )

        prompt = f"""Given the proof goal below, rank the following lemmas/theorems by relevance.
Return a comma-separated list of lemma numbers (1-indexed) in descending order of relevance.

Goal:
{goal}

Candidates:
{candidate_text}

Ranking (numbers only, comma-separated):"""

        response = await llm.call(prompt, complexity=TaskComplexity.MEDIUM)
        content = response.content if hasattr(response, "content") else str(response)

        try:
            # Parse the ranking
            ranking = [int(x.strip()) - 1 for x in content.split(",") if x.strip().isdigit()]
            reranked = [candidates[i] for i in ranking if i < len(candidates)]

            # Add any missing candidates
            ranked_set = set(ranking)
            for i, candidate in enumerate(candidates):
                if i not in ranked_set:
                    reranked.append(candidate)

            return reranked
        except (ValueError, IndexError):
            logger.warning("Failed to parse LLM ranking, returning original order")
            return candidates

    def get_stats(self) -> dict[str, Any]:
        """Get retrieval statistics."""
        return self._stats.copy()
