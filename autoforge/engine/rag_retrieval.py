"""Library-level RAG — Retrieval-Augmented Generation for code reuse.

Builds a cross-project code knowledge base from completed projects in
workspace/, enabling context-enhanced code generation via BM25 + vector
similarity hybrid retrieval.

Key features:
  - Extracts high-quality code snippets from successful past projects
  - BM25 keyword matching for exact API/function name lookup
  - TF-IDF vector similarity for semantic matching (no external embeddings needed)
  - Hybrid scoring: BM25 score × 0.5 + vector similarity × 0.5
  - Deduplication and quality filtering
  - Persistent index stored in ~/.autoforge/rag_index/

Reference:
  - RACG comprehensive survey (arXiv 2510.04905, 2025)
  - BM25 + vector fusion as optimal retrieval strategy
"""

from __future__ import annotations

import json
import logging
import math
import re
import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────


@dataclass
class CodeSnippet:
    """A reusable code snippet extracted from a past project."""
    id: str
    source_project: str       # Which project this came from
    file_path: str            # Original file path within project
    language: str             # python, javascript, typescript, etc.
    snippet_type: str         # function, class, module, config, test
    name: str                 # Function/class/module name
    content: str              # The actual code
    docstring: str = ""       # Extracted docstring/comment
    dependencies: list[str] = field(default_factory=list)  # imports needed
    quality_score: float = 0.0  # From review if available
    token_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_project": self.source_project,
            "file_path": self.file_path,
            "language": self.language,
            "snippet_type": self.snippet_type,
            "name": self.name,
            "content": self.content,
            "docstring": self.docstring,
            "dependencies": self.dependencies,
            "quality_score": self.quality_score,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeSnippet:
        return cls(
            id=data.get("id", ""),
            source_project=data.get("source_project", ""),
            file_path=data.get("file_path", ""),
            language=data.get("language", ""),
            snippet_type=data.get("snippet_type", ""),
            name=data.get("name", ""),
            content=data.get("content", ""),
            docstring=data.get("docstring", ""),
            dependencies=data.get("dependencies", []),
            quality_score=data.get("quality_score", 0.0),
            token_count=data.get("token_count", 0),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class RetrievalResult:
    """A ranked retrieval result."""
    snippet: CodeSnippet
    bm25_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0

    def format_context(self) -> str:
        """Format as context string for LLM injection."""
        header = f"# {self.snippet.name} ({self.snippet.language})"
        if self.snippet.docstring:
            header += f"\n# {self.snippet.docstring[:200]}"
        return f"{header}\n{self.snippet.content}"


# ──────────────────────────────────────────────
# BM25 Implementation (no external dependencies)
# ──────────────────────────────────────────────


class BM25Index:
    """Okapi BM25 ranking for code snippets.

    Lightweight implementation — no external libraries needed.
    Tokenises code using a code-aware tokenizer that preserves
    identifiers, keywords, and common patterns.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._df: Counter = Counter()  # Document frequency per term
        self._avgdl: float = 0.0
        self._N: int = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Code-aware tokenization: split on non-alphanumeric, preserve identifiers."""
        # Split camelCase and snake_case
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = text.replace("_", " ").replace("-", " ")
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", text.lower())
        return [t for t in tokens if len(t) > 1]  # Skip single chars

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document (snippet) to the index."""
        tokens = self._tokenize(text)
        self._docs.append(tokens)
        self._doc_ids.append(doc_id)

        # Update document frequency
        unique_terms = set(tokens)
        for term in unique_terms:
            self._df[term] += 1

        # Update stats
        self._N = len(self._docs)
        total_len = sum(len(d) for d in self._docs)
        self._avgdl = total_len / self._N if self._N > 0 else 0

    def query(self, query_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Query the index and return top-k (doc_id, score) pairs."""
        query_tokens = self._tokenize(query_text)
        scores: list[tuple[str, float]] = []

        for i, (doc_tokens, doc_id) in enumerate(zip(self._docs, self._doc_ids)):
            score = self._score_document(query_tokens, doc_tokens)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _score_document(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        """Compute BM25 score for a single document."""
        doc_len = len(doc_tokens)
        tf_map = Counter(doc_tokens)
        score = 0.0

        for term in query_tokens:
            if term not in tf_map:
                continue
            tf = tf_map[term]
            df = self._df.get(term, 0)
            # IDF with floor to avoid negative values
            idf = max(0.0, math.log((self._N - df + 0.5) / (df + 0.5) + 1.0))
            # BM25 TF normalization
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1))
            score += idf * numerator / denominator

        return score


# ──────────────────────────────────────────────
# TF-IDF Vector Similarity (lightweight alternative to embeddings)
# ──────────────────────────────────────────────


class TFIDFIndex:
    """TF-IDF cosine similarity index for semantic matching.

    Provides a lightweight vector similarity without needing external
    embedding models. Works surprisingly well for code retrieval.
    """

    def __init__(self) -> None:
        self._docs: dict[str, Counter] = {}
        self._df: Counter = Counter()
        self._N: int = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = text.replace("_", " ").replace("-", " ")
        return [t.lower() for t in re.findall(r"[a-zA-Z][a-zA-Z0-9]{1,}", text)]

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the TF-IDF index."""
        tokens = self._tokenize(text)
        self._docs[doc_id] = Counter(tokens)
        for term in set(tokens):
            self._df[term] += 1
        self._N = len(self._docs)

    def query(self, query_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Find most similar documents by TF-IDF cosine similarity."""
        query_tokens = self._tokenize(query_text)
        query_tf = Counter(query_tokens)
        query_vec = self._to_tfidf(query_tf)

        if not query_vec:
            return []

        scores = []
        for doc_id, doc_tf in self._docs.items():
            doc_vec = self._to_tfidf(doc_tf)
            sim = self._cosine_similarity(query_vec, doc_vec)
            if sim > 0:
                scores.append((doc_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _to_tfidf(self, tf_counter: Counter) -> dict[str, float]:
        """Convert term frequencies to TF-IDF weights."""
        vec = {}
        for term, tf in tf_counter.items():
            df = self._df.get(term, 0)
            if df > 0 and self._N > 0:
                idf = math.log(self._N / df)
                vec[term] = (1 + math.log(tf)) * idf
        return vec

    @staticmethod
    def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        common_keys = set(a.keys()) & set(b.keys())
        if not common_keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ──────────────────────────────────────────────
# Code Extractor
# ──────────────────────────────────────────────


class CodeExtractor:
    """Extract reusable code snippets from project files."""

    # Supported file extensions
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
    }

    # Skip these directories
    SKIP_DIRS = {
        "node_modules", ".git", "__pycache__", ".venv", "venv",
        "dist", "build", ".next", ".autoforge",
    }

    # Minimum snippet size (characters)
    MIN_SNIPPET_SIZE = 50
    # Maximum snippet size (characters)
    MAX_SNIPPET_SIZE = 5000

    def extract_from_project(
        self,
        project_dir: Path,
        project_name: str,
        quality_score: float = 0.0,
    ) -> list[CodeSnippet]:
        """Extract code snippets from a project directory."""
        snippets = []

        for ext, language in self.LANGUAGE_MAP.items():
            for file_path in project_dir.rglob(f"*{ext}"):
                # Skip excluded directories
                if any(skip in file_path.parts for skip in self.SKIP_DIRS):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8")
                    if len(content) < self.MIN_SNIPPET_SIZE:
                        continue

                    rel_path = str(file_path.relative_to(project_dir))
                    file_snippets = self._extract_from_file(
                        content, language, rel_path, project_name, quality_score,
                    )
                    snippets.extend(file_snippets)

                except (OSError, UnicodeDecodeError):
                    continue

        logger.info(f"[RAG] Extracted {len(snippets)} snippets from {project_name}")
        return snippets

    def _extract_from_file(
        self,
        content: str,
        language: str,
        file_path: str,
        project_name: str,
        quality_score: float,
    ) -> list[CodeSnippet]:
        """Extract function/class snippets from a single file."""
        snippets = []

        if language == "python":
            snippets.extend(
                self._extract_python(content, file_path, project_name, quality_score)
            )
        else:
            # For other languages, extract the whole file if it's reasonable size
            if self.MIN_SNIPPET_SIZE <= len(content) <= self.MAX_SNIPPET_SIZE:
                snippet_id = hashlib.sha256(content.encode()).hexdigest()[:12]
                name = Path(file_path).stem
                snippets.append(CodeSnippet(
                    id=snippet_id,
                    source_project=project_name,
                    file_path=file_path,
                    language=language,
                    snippet_type="module",
                    name=name,
                    content=content,
                    quality_score=quality_score,
                    token_count=len(content.split()),
                ))

        return snippets

    def _extract_python(
        self,
        content: str,
        file_path: str,
        project_name: str,
        quality_score: float,
    ) -> list[CodeSnippet]:
        """Extract Python functions and classes using regex (no AST for resilience)."""
        snippets = []

        # Extract functions — match def line + all following indented/blank lines
        func_pattern = re.compile(
            r'^((?:async\s+)?def\s+(\w+)\s*\([^)]*\)[^\n]*\n(?:(?:[ \t]+[^\n]*|[ \t]*)\n)*)',
            re.MULTILINE,
        )
        for match in func_pattern.finditer(content):
            func_body = match.group(1).strip()
            func_name = match.group(2)
            if len(func_body) < self.MIN_SNIPPET_SIZE:
                continue
            if len(func_body) > self.MAX_SNIPPET_SIZE:
                func_body = func_body[:self.MAX_SNIPPET_SIZE]

            # Extract docstring
            docstring = ""
            doc_match = re.search(r'"""(.*?)"""', func_body, re.DOTALL)
            if doc_match:
                docstring = doc_match.group(1).strip()[:200]

            snippet_id = hashlib.sha256(
                f"{file_path}:{func_name}".encode()
            ).hexdigest()[:12]

            snippets.append(CodeSnippet(
                id=snippet_id,
                source_project=project_name,
                file_path=file_path,
                language="python",
                snippet_type="function",
                name=func_name,
                content=func_body,
                docstring=docstring,
                quality_score=quality_score,
                token_count=len(func_body.split()),
            ))

        # Extract classes — match class line + all following indented/blank lines
        class_pattern = re.compile(
            r'^(class\s+(\w+)\s*(?:\([^)]*\))?:[^\n]*\n(?:(?:[ \t]+[^\n]*|[ \t]*)\n)*)',
            re.MULTILINE,
        )
        for match in class_pattern.finditer(content):
            class_body = match.group(1).strip()
            class_name = match.group(2)
            if len(class_body) < self.MIN_SNIPPET_SIZE:
                continue
            if len(class_body) > self.MAX_SNIPPET_SIZE:
                class_body = class_body[:self.MAX_SNIPPET_SIZE]

            snippet_id = hashlib.sha256(
                f"{file_path}:{class_name}".encode()
            ).hexdigest()[:12]

            snippets.append(CodeSnippet(
                id=snippet_id,
                source_project=project_name,
                file_path=file_path,
                language="python",
                snippet_type="class",
                name=class_name,
                content=class_body,
                quality_score=quality_score,
                token_count=len(class_body.split()),
            ))

        return snippets


# ──────────────────────────────────────────────
# RAG Retrieval Engine
# ──────────────────────────────────────────────


class RAGRetrievalEngine:
    """Cross-project code knowledge base with hybrid retrieval.

    Manages the full lifecycle:
    1. **Index**: Extract snippets from completed projects → add to index
    2. **Retrieve**: Given a task description, find relevant past code
    3. **Inject**: Format retrieved code as context for agent prompts

    Uses BM25 + TF-IDF hybrid scoring (no external embedding model needed).
    """

    # Hybrid scoring weights
    BM25_WEIGHT = 0.5
    VECTOR_WEIGHT = 0.5

    # Maximum context tokens to inject
    MAX_CONTEXT_TOKENS = 2000

    # Index persistence
    _INDEX_DIR = "rag_index"

    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            base_dir = Path.home() / ".autoforge"
        self.base_dir = base_dir / self._INDEX_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._snippets: dict[str, CodeSnippet] = {}
        self._bm25 = BM25Index()
        self._tfidf = TFIDFIndex()
        self._extractor = CodeExtractor()
        self._indexed_projects: set[str] = set()

        self._load_index()

    # ──────── Indexing ────────

    def index_project(
        self,
        project_dir: Path,
        project_name: str,
        quality_score: float = 0.0,
    ) -> int:
        """Extract and index code snippets from a project.

        Returns number of snippets indexed.
        """
        if project_name in self._indexed_projects:
            logger.debug(f"[RAG] Project {project_name} already indexed")
            return 0

        snippets = self._extractor.extract_from_project(
            project_dir, project_name, quality_score,
        )

        added = 0
        for snippet in snippets:
            if snippet.id not in self._snippets:
                self._snippets[snippet.id] = snippet
                # Index both code content and name/docstring for BM25
                search_text = f"{snippet.name} {snippet.docstring} {snippet.content}"
                self._bm25.add_document(snippet.id, search_text)
                self._tfidf.add_document(snippet.id, search_text)
                added += 1

        self._indexed_projects.add(project_name)
        self._save_index()

        logger.info(f"[RAG] Indexed {added} new snippets from {project_name}")
        return added

    # ──────── Retrieval ────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        language_filter: str = "",
    ) -> list[RetrievalResult]:
        """Hybrid retrieval: BM25 keyword + TF-IDF vector similarity.

        Args:
            query: Natural language or code description
            top_k: Number of results to return
            language_filter: Optional language filter (e.g. "python")

        Returns:
            Ranked list of RetrievalResult objects.
        """
        if not self._snippets:
            return []

        # Get scores from both indexes
        bm25_results = dict(self._bm25.query(query, top_k=top_k * 2))
        tfidf_results = dict(self._tfidf.query(query, top_k=top_k * 2))

        # Normalise scores to [0, 1]
        bm25_max = max(bm25_results.values()) if bm25_results else 1.0
        tfidf_max = max(tfidf_results.values()) if tfidf_results else 1.0

        # Merge candidates
        all_ids = set(bm25_results.keys()) | set(tfidf_results.keys())
        results = []

        for doc_id in all_ids:
            snippet = self._snippets.get(doc_id)
            if not snippet:
                continue
            if language_filter and snippet.language != language_filter:
                continue

            bm25_norm = bm25_results.get(doc_id, 0) / max(bm25_max, 1e-6)
            tfidf_norm = tfidf_results.get(doc_id, 0) / max(tfidf_max, 1e-6)
            hybrid = bm25_norm * self.BM25_WEIGHT + tfidf_norm * self.VECTOR_WEIGHT

            # Bonus for high-quality snippets
            if snippet.quality_score > 7:
                hybrid *= 1.1

            results.append(RetrievalResult(
                snippet=snippet,
                bm25_score=bm25_norm,
                vector_score=tfidf_norm,
                hybrid_score=hybrid,
            ))

        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return results[:top_k]

    # ──────── Context Injection ────────

    def build_context(
        self,
        query: str,
        top_k: int = 3,
        language_filter: str = "",
    ) -> str:
        """Retrieve relevant code and format as context for agent prompts.

        Returns a formatted string ready to inject into agent context.
        """
        results = self.retrieve(query, top_k=top_k, language_filter=language_filter)
        if not results:
            return ""

        parts = ["\n## Relevant Code from Past Projects\n"]
        total_tokens = 0

        for r in results:
            entry = r.format_context()
            entry_tokens = len(entry.split())
            if total_tokens + entry_tokens > self.MAX_CONTEXT_TOKENS:
                break
            parts.append(f"\n### {r.snippet.source_project}/{r.snippet.file_path}")
            parts.append(f"```{r.snippet.language}\n{r.snippet.content}\n```\n")
            total_tokens += entry_tokens

        if len(parts) == 1:
            return ""

        return "\n".join(parts)

    # ──────── Statistics ────────

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        languages = Counter(s.language for s in self._snippets.values())
        types = Counter(s.snippet_type for s in self._snippets.values())

        return {
            "total_snippets": len(self._snippets),
            "indexed_projects": len(self._indexed_projects),
            "languages": dict(languages),
            "snippet_types": dict(types),
            "projects": sorted(self._indexed_projects),
        }

    # ──────── Persistence ────────

    def _save_index(self) -> None:
        """Save the snippet database to disk (JSON)."""
        try:
            data = {
                "snippets": [s.to_dict() for s in self._snippets.values()],
                "indexed_projects": sorted(self._indexed_projects),
            }
            path = self.base_dir / "snippets.json"
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[RAG] Could not save index: {e}")

    def _load_index(self) -> None:
        """Load the snippet database from disk and rebuild in-memory indexes."""
        path = self.base_dir / "snippets.json"
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for s_data in data.get("snippets", []):
                snippet = CodeSnippet.from_dict(s_data)
                self._snippets[snippet.id] = snippet
                search_text = f"{snippet.name} {snippet.docstring} {snippet.content}"
                self._bm25.add_document(snippet.id, search_text)
                self._tfidf.add_document(snippet.id, search_text)

            self._indexed_projects = set(data.get("indexed_projects", []))
            logger.info(
                f"[RAG] Loaded index: {len(self._snippets)} snippets "
                f"from {len(self._indexed_projects)} projects"
            )
        except Exception as e:
            logger.warning(f"[RAG] Could not load index: {e}")
