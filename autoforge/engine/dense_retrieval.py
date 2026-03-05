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
        self._use_sklearn = False
        self._token_to_id: dict[str, int] = {}
        self._idf_values: np.ndarray | None = None

    def _fit_tfidf(self, texts: list[str]) -> np.ndarray:
        """Fit TF-IDF model on a corpus and return corpus embeddings."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

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
            tfidf_matrix = self._vectorizer.fit_transform(texts)
            self._use_sklearn = True

            feature_dim = tfidf_matrix.shape[1]
            if feature_dim > self.embedding_dim:
                logger.info(
                    "Reducing dimensions from %d to %d",
                    feature_dim,
                    self.embedding_dim,
                )
                self._svd = TruncatedSVD(n_components=self.embedding_dim)
                reduced = self._svd.fit_transform(tfidf_matrix)
                return reduced.astype(np.float32)

            dense = tfidf_matrix.toarray().astype(np.float32)
            return self._pad_to_embedding_dim(dense)
        except ImportError:
            logger.warning("sklearn not available. Using pure Python TF-IDF.")
            self._use_sklearn = False
            return self._fit_pure_python_tfidf(texts)

    def _fit_pure_python_tfidf(self, texts: list[str]) -> np.ndarray:
        """Fit a pure-Python TF-IDF model and return corpus embeddings."""
        tokenized = [self._tokenize(text) for text in texts]
        n_docs = len(tokenized)
        if n_docs == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Document frequency per token.
        df: dict[str, int] = {}
        for doc_tokens in tokenized:
            for token in set(doc_tokens):
                df[token] = df.get(token, 0) + 1

        min_df = max(1, n_docs // 100)
        max_df = int(n_docs * 0.95)
        filtered = {
            token: freq
            for token, freq in df.items()
            if min_df <= freq <= max_df
        }
        sorted_tokens = sorted(
            filtered.keys(),
            key=lambda token: (-filtered[token], token),
        )[: self.embedding_dim]

        self._token_to_id = {token: idx for idx, token in enumerate(sorted_tokens)}
        self._idf_values = np.array(
            [
                math.log((1 + n_docs) / (1 + filtered[token])) + 1.0
                for token in sorted_tokens
            ],
            dtype=np.float32,
        )

        matrix = self._transform_pure_python_tfidf(texts)
        return matrix

    def _transform_tfidf(self, texts: list[str]) -> np.ndarray:
        """Transform texts into the already-fitted embedding space."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if self._use_sklearn and self._vectorizer is not None:
            transformed = self._vectorizer.transform(texts)
            if self._svd is not None:
                return self._svd.transform(transformed).astype(np.float32)
            dense = transformed.toarray().astype(np.float32)
            return self._pad_to_embedding_dim(dense)

        return self._transform_pure_python_tfidf(texts)

    def _transform_pure_python_tfidf(self, texts: list[str]) -> np.ndarray:
        """Transform texts with pure-Python TF-IDF in a fixed vocabulary."""
        if not self._token_to_id:
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        matrix = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for doc_idx, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue

            tf: dict[int, int] = {}
            for token in tokens:
                token_id = self._token_to_id.get(token)
                if token_id is None:
                    continue
                tf[token_id] = tf.get(token_id, 0) + 1

            if not tf:
                continue

            token_count = max(len(tokens), 1)
            for token_id, count in tf.items():
                idf = float(self._idf_values[token_id]) if self._idf_values is not None else 1.0
                matrix[doc_idx, token_id] = (count / token_count) * idf

        # Normalize rows.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        return matrix.astype(np.float32)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w{2,}\b", text.lower())

    def _pad_to_embedding_dim(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[1] >= self.embedding_dim:
            return matrix[:, : self.embedding_dim].astype(np.float32)
        padded = np.zeros((matrix.shape[0], self.embedding_dim), dtype=np.float32)
        padded[:, : matrix.shape[1]] = matrix
        return padded

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts using TF-IDF."""
        if not texts:
            return []
        if self._tfidf_matrix is None:
            matrix = await asyncio.to_thread(self._fit_tfidf, texts)
            self._corpus = texts.copy()
            self._tfidf_matrix = matrix
            return matrix.tolist()

        matrix = await asyncio.to_thread(self._transform_tfidf, texts)
        return matrix.tolist()

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def initialize_corpus(self, texts: list[str]) -> None:
        """Initialize with full corpus for consistent embeddings."""
        self._corpus = texts.copy()
        self._tfidf_matrix = await asyncio.to_thread(self._fit_tfidf, texts)

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

        logger.info(f"Scanning Mathlib at {mathlib_path}")
        lean_files = list((mathlib_path / "Mathlib").rglob("*.lean"))
        if not lean_files:
            # Fallback: scan the whole project path for .lean declarations.
            lean_files = list(mathlib_path.rglob("*.lean"))

        # Keep offline scan bounded.
        max_files = 200
        for file_path in lean_files[:max_files]:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            module_path = str(file_path.relative_to(mathlib_path)).replace("\\", "/")
            for line in text.splitlines():
                parsed = MathlibPremiseLoader._parse_lean_declaration(line.strip())
                if parsed is None:
                    continue
                parsed.module_path = module_path
                parsed.id = f"{module_path}:{parsed.name}"
                premises.append(parsed)

        logger.info("Extracted %d declaration premises from %d files", len(premises), min(len(lean_files), max_files))
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
        if lean_project is None:
            return []

        declarations: list[Any] = []
        try:
            if hasattr(lean_project, "list_declarations_async"):
                declarations = await lean_project.list_declarations_async()
            elif hasattr(lean_project, "list_declarations"):
                declarations = await asyncio.to_thread(lean_project.list_declarations)
            else:
                logger.warning("Lean project does not expose declaration listing APIs")
                return []
        except Exception as exc:
            logger.warning("Failed to enumerate declarations via Lean REPL: %s", exc)
            return []

        premises: list[PremiseEntry] = []
        for item in declarations:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                statement = str(item.get("type", item.get("statement", ""))).strip()
                module_path = str(item.get("module", item.get("file", "")))
                premises.append(
                    PremiseEntry(
                        id=name,
                        name=name,
                        statement=statement,
                        module_path=module_path,
                        docstring=str(item.get("doc", item.get("docstring", ""))),
                    )
                )
            elif isinstance(item, str):
                parsed = MathlibPremiseLoader._parse_lean_declaration(item.strip())
                if parsed is not None:
                    premises.append(parsed)

        logger.info("Loaded %d declarations from Lean REPL", len(premises))
        return premises

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
