"""
Literature-Grounded Discovery Engine

Evidence-based novelty verification using real academic databases (Semantic Scholar, arXiv).
Implements AI Scientist v2 (2025) literature grounding for discovery validation.

This module searches academic literature to check if a conjecture is truly novel by:
1. Extracting key terms from candidate statements
2. Searching Semantic Scholar and arXiv APIs
3. Computing relevance scores and deduplicating results
4. Comparing candidates against found papers with LLM assistance
5. Building citation graphs and conducting semantic analysis
6. Analyzing research gaps and trends
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urlencode
from typing import Any

from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PaperReference:
    """
    Represents a paper found in academic literature.

    Attributes:
        paper_id: Semantic Scholar ID or arXiv ID
        title: Paper title
        authors: List of author names
        year: Publication year
        abstract: Paper abstract text
        url: Direct URL to paper
        relevance_score: Computed relevance to query (0.0-1.0)
        overlap_reason: Human-readable explanation of relevance
        citation_count: Number of citations to this paper
        reference_count: Number of references this paper makes
        influential_citation_count: Number of influential citations
        embedding: Optional semantic embedding vector
    """

    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int = 0
    abstract: str = ""
    url: str = ""
    relevance_score: float = 0.0
    overlap_reason: str = ""
    citation_count: int = 0
    reference_count: int = 0
    influential_citation_count: int = 0
    embedding: list[float] | None = None

    def __eq__(self, other: object) -> bool:
        """Two papers are equal if their titles are very similar."""
        if not isinstance(other, PaperReference):
            return NotImplemented
        return self._title_similarity(self.title, other.title) > 0.85

    def __hash__(self) -> int:
        """Hash based on normalized title."""
        return hash(self.title.lower().strip())

    @staticmethod
    def _title_similarity(title1: str, title2: str) -> float:
        """Compute Jaccard similarity between two titles."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0


@dataclass
class LiteratureSearchConfig:
    """
    Configuration for literature search engine.

    Attributes:
        semantic_scholar_enabled: Whether to query Semantic Scholar API
        arxiv_enabled: Whether to query arXiv API
        max_results_per_query: Maximum papers to fetch per query
        min_relevance_threshold: Minimum relevance score to include results
        cache_ttl_hours: Time-to-live for search cache in hours
    """

    semantic_scholar_enabled: bool = True
    arxiv_enabled: bool = True
    max_results_per_query: int = 10
    min_relevance_threshold: float = 0.3
    cache_ttl_hours: int = 24
    allow_remote_code_models: bool = False


class CitationGraph:
    """
    Build and traverse citation graphs from Semantic Scholar.

    Features:
    - Bidirectional traversal (references and citations)
    - Depth-limited searches for efficiency
    - Paper caching to avoid redundant API calls
    - Finding common ancestors (foundational papers)
    - Research frontier detection
    """

    def __init__(self) -> None:
        """Initialize the citation graph."""
        self._graph: dict[str, list[str]] = {}  # Adjacency list
        self._papers: dict[str, PaperReference] = {}  # Paper cache

    async def build_from_paper(
        self, paper_id: str, depth: int = 2
    ) -> dict[str, Any]:
        """
        Build citation graph by traversing references and citations.

        Args:
            paper_id: Semantic Scholar paper ID
            depth: Maximum traversal depth (default 2)

        Returns:
            Dictionary with graph structure and metadata
        """
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(paper_id, 0)]
        edges: list[tuple[str, str]] = []

        while queue:
            current_id, current_depth = queue.pop(0)

            if current_id in visited or current_depth >= depth:
                continue

            visited.add(current_id)

            # Fetch references and citations
            try:
                references = await self.get_references(current_id)
                citations = await self.get_citations(current_id)

                for ref in references:
                    if ref.paper_id not in visited:
                        queue.append((ref.paper_id, current_depth + 1))
                        edges.append((current_id, ref.paper_id))
                        self._graph.setdefault(current_id, []).append(ref.paper_id)

                for cite in citations:
                    if cite.paper_id not in visited:
                        queue.append((cite.paper_id, current_depth + 1))
                        edges.append((cite.paper_id, current_id))
                        self._graph.setdefault(cite.paper_id, []).append(current_id)

            except Exception as e:
                logger.warning(f"Failed to build graph from {current_id}: {e}")
                continue

        return {
            "root": paper_id,
            "nodes": list(visited),
            "edges": edges,
            "papers": {
                pid: {
                    "title": paper.title,
                    "year": paper.year,
                    "citations": paper.citation_count,
                }
                for pid, paper in self._papers.items()
                if pid in visited
            },
        }

    async def get_references(self, paper_id: str) -> list[PaperReference]:
        """
        Fetch references cited by a paper.

        Args:
            paper_id: Semantic Scholar paper ID

        Returns:
            List of PaperReference objects
        """
        if not paper_id:
            return []

        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
            f"?fields=paperId,title,authors,year,abstract,url,externalIds,"
            f"citationCount,referenceCount,influentialCitationCount,embedding"
            f"&limit=100"
        )

        try:
            response_text = await self._http_get(url)
            if not response_text:
                return []

            data = json.loads(response_text)
            papers: list[PaperReference] = []

            for item in data.get("data", []):
                if "citedPaper" not in item:
                    continue

                cited = item["citedPaper"]
                paper = self._parse_semantic_scholar_paper(cited)
                papers.append(paper)
                self._papers[paper.paper_id] = paper

            logger.debug(f"Got {len(papers)} references for {paper_id}")
            return papers

        except Exception as e:
            logger.warning(f"Failed to get references for {paper_id}: {e}")
            return []

    async def get_citations(self, paper_id: str) -> list[PaperReference]:
        """
        Fetch papers that cite a given paper.

        Args:
            paper_id: Semantic Scholar paper ID

        Returns:
            List of PaperReference objects
        """
        if not paper_id:
            return []

        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
            f"?fields=paperId,title,authors,year,abstract,url,externalIds,"
            f"citationCount,referenceCount,influentialCitationCount,embedding"
            f"&limit=100"
        )

        try:
            response_text = await self._http_get(url)
            if not response_text:
                return []

            data = json.loads(response_text)
            papers: list[PaperReference] = []

            for item in data.get("data", []):
                if "citingPaper" not in item:
                    continue

                citing = item["citingPaper"]
                paper = self._parse_semantic_scholar_paper(citing)
                papers.append(paper)
                self._papers[paper.paper_id] = paper

            logger.debug(f"Got {len(papers)} citations for {paper_id}")
            return papers

        except Exception as e:
            logger.warning(f"Failed to get citations for {paper_id}: {e}")
            return []

    async def find_common_ancestors(
        self, paper_ids: list[str]
    ) -> list[PaperReference]:
        """
        Find foundational papers cited by multiple input papers.

        Args:
            paper_ids: List of Semantic Scholar paper IDs

        Returns:
            List of common referenced papers, sorted by citation count
        """
        if not paper_ids:
            return []

        # Get references for each paper
        all_reference_sets: list[set[str]] = []
        for pid in paper_ids:
            refs = await self.get_references(pid)
            all_reference_sets.append({r.paper_id for r in refs})

        # Find intersection
        if not all_reference_sets:
            return []

        common = all_reference_sets[0]
        for ref_set in all_reference_sets[1:]:
            common &= ref_set

        # Convert back to papers and sort by citation count
        common_papers = [
            self._papers[pid] for pid in common if pid in self._papers
        ]
        common_papers.sort(key=lambda p: p.citation_count, reverse=True)

        return common_papers

    async def find_research_frontier(
        self, topic: str, year_min: int = 2023
    ) -> list[PaperReference]:
        """
        Find most-cited recent papers on a topic.

        Args:
            topic: Research topic
            year_min: Minimum publication year

        Returns:
            List of frontier papers, sorted by citation count and recency
        """
        # Search for topic papers
        params = urlencode(
            {
                "query": topic,
                "limit": 100,
                "fields": (
                    "paperId,title,authors,year,abstract,url,externalIds,"
                    "citationCount,referenceCount,influentialCitationCount,embedding"
                ),
            }
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"

        try:
            response_text = await self._http_get(url)
            if not response_text:
                return []

            data = json.loads(response_text)
            papers: list[PaperReference] = []

            for item in data.get("data", []):
                paper = self._parse_semantic_scholar_paper(item)
                if paper.year >= year_min:
                    papers.append(paper)
                    self._papers[paper.paper_id] = paper

            # Sort by citation count (most cited first)
            papers.sort(key=lambda p: p.citation_count, reverse=True)

            logger.debug(f"Found {len(papers)} frontier papers for {topic}")
            return papers

        except Exception as e:
            logger.warning(f"Failed to find frontier papers for {topic}: {e}")
            return []

    def _parse_semantic_scholar_paper(self, item: dict[str, Any]) -> PaperReference:
        """Parse a Semantic Scholar API response item into PaperReference."""
        paper = PaperReference(
            paper_id=item.get("paperId", ""),
            title=item.get("title", ""),
            authors=[a.get("name", "") for a in item.get("authors", [])],
            year=item.get("year", 0),
            abstract=item.get("abstract", ""),
            url=item.get("url", ""),
            citation_count=item.get("citationCount", 0),
            reference_count=item.get("referenceCount", 0),
            influential_citation_count=item.get("influentialCitationCount", 0),
            embedding=item.get("embedding"),
        )

        # Extract arXiv URL if available
        if "externalIds" in item and "ArXiv" in item["externalIds"]:
            arxiv_id = item["externalIds"]["ArXiv"]
            paper.url = f"https://arxiv.org/abs/{arxiv_id}"

        return paper

    async def _http_get(self, url: str) -> str:
        """Fetch URL content with fallback strategies."""
        try:
            from autoforge.engine.tools.web import fetch_url

            try:
                response = await asyncio.to_thread(fetch_url, url)
                return response
            except Exception as e:
                logger.debug(f"fetch_url failed: {e}")
        except ImportError:
            logger.debug("autoforge.engine.tools.web not available")

        try:
            process = await asyncio.create_subprocess_exec(
                "curl",
                "-s",
                "-L",
                url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10.0)
            return stdout.decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug(f"curl fallback failed: {e}")

        try:
            import urllib.request

            def _urlopen() -> str:
                with urllib.request.urlopen(url, timeout=10) as response:
                    return response.read().decode("utf-8", errors="replace")

            return await asyncio.to_thread(_urlopen)
        except Exception as e:
            logger.error(f"All HTTP strategies failed for {url}: {e}")
            return ""


class SemanticSearchEngine:
    """
    Semantic similarity search for papers using embeddings.

    Features:
    - SPECTER2-style embedding similarity
    - Graceful fallback chain: sentence-transformers -> sklearn TF-IDF -> Jaccard
    - Cosine similarity computation
    """

    def __init__(self, allow_remote_code_models: bool = False) -> None:
        """Initialize semantic search engine."""
        self._embedder: Any = None
        self._tfidf_vectorizer: Any = None
        self._use_embeddings = False
        self._allow_remote_code_models = allow_remote_code_models
        self._init_embedder()

    def _init_embedder(self) -> None:
        """Initialize embedder with fallback chain."""
        # Try sentence-transformers with SPECTER2 model
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(
                "allenai/specter2_base",
                trust_remote_code=self._allow_remote_code_models,
            )
            self._use_embeddings = True
            logger.debug("Initialized SPECTER2 embeddings")
            return
        except Exception as e:
            logger.debug(f"SPECTER2 initialization failed: {e}")

        # Fallback to sklearn TF-IDF
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf_vectorizer = TfidfVectorizer(
                lowercase=True, stop_words="english", max_features=5000
            )
            logger.debug("Initialized TF-IDF fallback")
            return
        except Exception as e:
            logger.debug(f"TF-IDF initialization failed: {e}")

        logger.debug("Using Jaccard similarity fallback")

    async def search_by_embedding(
        self,
        query: str,
        corpus: list[PaperReference],
        top_k: int = 10,
    ) -> list[tuple[PaperReference, float]]:
        """
        Search papers by semantic similarity to query.

        Args:
            query: Query string
            corpus: List of papers to search in
            top_k: Number of results to return

        Returns:
            List of (PaperReference, similarity_score) tuples
        """
        if not corpus:
            return []

        try:
            if self._use_embeddings and self._embedder is not None:
                # Use SPECTER2 embeddings
                query_embedding = await asyncio.to_thread(
                    self._embed_texts, [query]
                )
                query_embedding = query_embedding[0]

                corpus_texts = [
                    f"{p.title} {p.abstract}" for p in corpus
                ]
                corpus_embeddings = await asyncio.to_thread(
                    self._embed_texts, corpus_texts
                )

                scores = [
                    self._cosine_similarity(query_embedding, emb)
                    for emb in corpus_embeddings
                ]

                # Pair papers with scores and sort
                results = list(zip(corpus, scores))
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]

            elif self._tfidf_vectorizer is not None:
                # Use TF-IDF
                corpus_texts = [
                    f"{p.title} {p.abstract}" for p in corpus
                ]
                all_texts = [query] + corpus_texts

                await asyncio.to_thread(
                    self._tfidf_vectorizer.fit_transform, all_texts
                )
                tfidf_matrix = self._tfidf_vectorizer.transform(all_texts)

                # Compute similarities
                from sklearn.metrics.pairwise import cosine_similarity

                similarities = cosine_similarity(
                    tfidf_matrix[0:1], tfidf_matrix[1:]
                )[0]

                results = list(zip(corpus, similarities))
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]

            else:
                # Jaccard fallback
                return await self._jaccard_search(query, corpus, top_k)

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return await self._jaccard_search(query, corpus, top_k)

    async def find_similar_papers(
        self,
        paper: PaperReference,
        corpus: list[PaperReference],
        top_k: int = 5,
    ) -> list[tuple[PaperReference, float]]:
        """
        Find papers similar to a given paper.

        Args:
            paper: Reference paper
            corpus: List of papers to search in
            top_k: Number of results to return

        Returns:
            List of (PaperReference, similarity_score) tuples
        """
        query = f"{paper.title} {paper.abstract}"
        results = await self.search_by_embedding(query, corpus, top_k)

        # Filter out the reference paper itself
        return [(p, s) for p, s in results if p.paper_id != paper.paper_id]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the initialized embedder."""
        if self._embedder is None:
            return []

        embeddings = self._embedder.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0

        import math

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def _jaccard_search(
        self,
        query: str,
        corpus: list[PaperReference],
        top_k: int = 10,
    ) -> list[tuple[PaperReference, float]]:
        """Fallback Jaccard similarity search."""
        query_words = set(query.lower().split())

        scores = []
        for paper in corpus:
            paper_words = set(
                (paper.title + " " + paper.abstract).lower().split()
            )
            if query_words and paper_words:
                jaccard = len(query_words & paper_words) / len(
                    query_words | paper_words
                )
            else:
                jaccard = 0.0
            scores.append((paper, jaccard))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class FullTextAnalyzer:
    """
    Analyze full paper text for methods, datasets, and baselines.

    Features:
    - Full text download and parsing
    - Method extraction
    - Dataset identification
    - Baseline comparison
    - Result extraction
    """

    async def analyze_paper(self, pdf_url: str) -> dict[str, Any]:
        """
        Download and analyze a paper's full text.

        Args:
            pdf_url: URL to paper PDF

        Returns:
            Dictionary with analysis results
        """
        try:
            text = await self._download_pdf(pdf_url)
            if not text:
                return {"error": "Failed to download PDF"}

            methods = await self.extract_methods(text, None)
            datasets = await self.extract_datasets(text)
            baselines = await self.extract_baselines(text)

            return {
                "url": pdf_url,
                "text_length": len(text),
                "methods": methods,
                "datasets": datasets,
                "baselines": baselines,
            }
        except Exception as e:
            logger.error(f"Paper analysis failed for {pdf_url}: {e}")
            return {"error": str(e)}

    async def extract_methods(
        self, text: str, llm: Any | None
    ) -> list[str]:
        """
        Extract method names and descriptions from paper text.

        Args:
            text: Full paper text
            llm: Optional LLM for advanced extraction

        Returns:
            List of method descriptions
        """
        # Simple heuristic: look for method sections
        methods = []

        # Common method section patterns
        method_patterns = [
            r"(?:## |### )?Methods?\s*\n(.*?)(?:\n## |\n### |$)",
            r"(?:## |### )?Approach\s*\n(.*?)(?:\n## |\n### |$)",
            r"(?:## |### )?Proposed\s*\n(.*?)(?:\n## |\n### |$)",
        ]

        for pattern in method_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                sentences = match.split(".")
                for sent in sentences[:3]:  # First 3 sentences
                    if len(sent.strip()) > 20:
                        methods.append(sent.strip())

        # Use LLM for more sophisticated extraction if available
        if llm and methods:
            try:
                from autoforge.engine.llm_router import TaskComplexity

                prompt = f"Extract the main methodological innovations from this text:\n{text[:2000]}"
                response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
                if response and response.content:
                    methods.extend(response.content.split("\n")[:3])
            except Exception as e:
                logger.debug(f"LLM method extraction failed: {e}")

        return list(set(methods))[:10]  # Deduplicate and limit

    async def extract_datasets(self, text: str) -> list[str]:
        """
        Extract dataset names from paper text.

        Args:
            text: Full paper text

        Returns:
            List of dataset names
        """
        datasets = []

        # Common dataset keywords
        dataset_keywords = [
            "ImageNet",
            "COCO",
            "CIFAR",
            "MNIST",
            "Wikipedia",
            "CommonCrawl",
            "SQuAD",
            "GLUE",
            "SuperGLUE",
            "Wikitext",
            "BookCorpus",
        ]

        for keyword in dataset_keywords:
            if keyword.lower() in text.lower():
                datasets.append(keyword)

        # Look for dataset patterns like "Dataset X" or "on the X dataset"
        dataset_pattern = r"(?:on\s+the\s+)?(\w+)\s+(?:dataset|corpus|benchmark)"
        matches = re.findall(dataset_pattern, text, re.IGNORECASE)
        datasets.extend(matches)

        return list(set(datasets))

    async def extract_baselines(self, text: str) -> list[str]:
        """
        Extract baseline method names from paper text.

        Args:
            text: Full paper text

        Returns:
            List of baseline method names
        """
        baselines = []

        # Common baseline patterns
        baseline_pattern = (
            r"(?:compared to|baseline|previous|prior|SOTA)\s+(?:\w+\s+)*(\w+)"
        )
        matches = re.findall(baseline_pattern, text, re.IGNORECASE)
        baselines.extend(matches)

        # Known baseline methods
        known_baselines = [
            "BERT",
            "GPT",
            "ResNet",
            "VGG",
            "Transformer",
            "CNN",
            "RNN",
            "LSTM",
            "GRU",
            "Attention",
        ]

        for baseline in known_baselines:
            if baseline in text:
                baselines.append(baseline)

        return list(set(baselines))[:20]

    async def extract_key_results(
        self, text: str, llm: Any | None = None
    ) -> list[dict[str, str]]:
        """
        Extract key results and metrics from paper.

        Args:
            text: Full paper text
            llm: Optional LLM for result understanding

        Returns:
            List of result dictionaries
        """
        results = []

        # Look for metric patterns: "accuracy: 95.2%", "F1: 0.89"
        metric_pattern = r"(\w+)\s*(?:score|@|:)?\s*([\d.]+)(?:\s*%)?(?:\s*±\s*([\d.]+))?"
        matches = re.findall(metric_pattern, text)

        for metric_name, value, std in matches:
            if metric_name.lower() in [
                "accuracy",
                "f1",
                "precision",
                "recall",
                "bleu",
                "rouge",
                "loss",
            ]:
                results.append(
                    {
                        "metric": metric_name,
                        "value": value,
                        "std": std or "N/A",
                    }
                )

        return results[:10]

    async def compare_methods(
        self, paper1_text: str, paper2_text: str, llm: Any | None = None
    ) -> str:
        """
        Compare methods between two papers.

        Args:
            paper1_text: First paper text
            paper2_text: Second paper text
            llm: Optional LLM for comparison analysis

        Returns:
            Comparison summary
        """
        methods1 = await self.extract_methods(paper1_text, llm)
        methods2 = await self.extract_methods(paper2_text, llm)

        common = set(methods1) & set(methods2)
        unique1 = set(methods1) - set(methods2)
        unique2 = set(methods2) - set(methods1)

        summary = f"""
Method Comparison:
- Common: {', '.join(common) or 'None'}
- Unique to Paper 1: {', '.join(unique1) or 'None'}
- Unique to Paper 2: {', '.join(unique2) or 'None'}
"""
        return summary.strip()

    async def _download_pdf(self, pdf_url: str) -> str:
        """Download PDF and extract text with available parser backends."""
        try:
            import urllib.request
            import tempfile

            def _fetch() -> bytes:
                with urllib.request.urlopen(pdf_url, timeout=10) as response:
                    return response.read()

            content = await asyncio.to_thread(_fetch)
            logger.debug(f"Downloaded {len(content)} bytes from {pdf_url}")

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            try:
                # Preferred backend: pypdf
                try:
                    from pypdf import PdfReader

                    reader = await asyncio.to_thread(PdfReader, str(tmp_path))
                    pages: list[str] = []
                    for page in reader.pages[:12]:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            pages.append(page_text)
                    text = "\n".join(pages).strip()
                    if text:
                        return text[:120_000]
                except Exception:
                    pass

                # Fallback backend: PyPDF2
                try:
                    import PyPDF2

                    with tmp_path.open("rb") as fh:
                        reader = PyPDF2.PdfReader(fh)
                        pages = []
                        for page in reader.pages[:12]:
                            page_text = page.extract_text() or ""
                            if page_text.strip():
                                pages.append(page_text)
                    text = "\n".join(pages).strip()
                    if text:
                        return text[:120_000]
                except Exception:
                    pass
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

            return ""
        except Exception as e:
            logger.warning(f"PDF download failed for {pdf_url}: {e}")
            return ""


class ResearchGapAnalyzer:
    """
    Identify research gaps and under-explored areas.

    Features:
    - Gap analysis from paper corpus
    - Experiment suggestion generation
    """

    async def analyze(
        self, papers: list[PaperReference], llm: Any | None = None
    ) -> list[str]:
        """
        Identify research gaps from a set of papers.

        Args:
            papers: List of relevant papers
            llm: LLM instance for analysis

        Returns:
            List of identified research gaps
        """
        if not papers or not llm:
            return []

        try:
            from autoforge.engine.llm_router import TaskComplexity

            # Prepare paper summaries
            paper_summaries = "\n".join(
                [
                    f"- {p.title} ({p.year}): {p.abstract[:150]}"
                    for p in papers[:10]
                ]
            )

            prompt = f"""
Analyze these papers and identify significant research gaps - areas not well covered:

{paper_summaries}

List 5 key research gaps or under-explored directions. Format: one gap per line.
"""

            response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            if response and response.content:
                gaps = [
                    line.strip()
                    for line in response.content.split("\n")
                    if line.strip() and len(line.strip()) > 10
                ]
                return gaps[:5]

            return []
        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            return []

    async def suggest_experiments(
        self, gaps: list[str], llm: Any | None = None
    ) -> list[dict[str, str]]:
        """
        Suggest experiments to address research gaps.

        Args:
            gaps: List of research gaps
            llm: LLM instance for suggestion generation

        Returns:
            List of experiment suggestions
        """
        if not gaps or not llm:
            return []

        try:
            from autoforge.engine.llm_router import TaskComplexity

            gaps_text = "\n".join([f"- {gap}" for gap in gaps])

            prompt = f"""
Given these research gaps:

{gaps_text}

Suggest 3 concrete experiments that could address these gaps.
For each, provide: title, hypothesis, and methodology.
Format as JSON array.
"""

            response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            if response and response.content:
                try:
                    experiments = json.loads(response.content)
                    return experiments[:3]
                except json.JSONDecodeError:
                    logger.debug("Failed to parse experiment suggestions as JSON")

            return []
        except Exception as e:
            logger.warning(f"Experiment suggestion failed: {e}")
            return []


class LiteratureSearchEngine:
    """
    Search academic literature via Semantic Scholar and arXiv APIs.

    Features:
    - Async HTTP requests with multiple fallback transports
    - TTL-based caching to reduce API calls
    - Deduplication by title similarity
    - Relevance scoring with recency boost
    - Citation graph traversal
    - Semantic similarity search
    - Deep multi-phase search
    - Topic survey generation
    """

    def __init__(self, config: LiteratureSearchConfig | None = None) -> None:
        """
        Initialize the literature search engine.

        Args:
            config: Search configuration. Defaults to LiteratureSearchConfig().
        """
        self.config = config or LiteratureSearchConfig()
        self._cache: dict[str, tuple[float, list[PaperReference]]] = {}
        self._citation_graph: CitationGraph | None = None
        self._semantic_search: SemanticSearchEngine | None = None

    @property
    def citation_graph(self) -> CitationGraph:
        """Lazy-initialize citation graph."""
        if self._citation_graph is None:
            self._citation_graph = CitationGraph()
        return self._citation_graph

    @property
    def semantic_search(self) -> SemanticSearchEngine:
        """Lazy-initialize semantic search engine."""
        if self._semantic_search is None:
            self._semantic_search = SemanticSearchEngine(
                allow_remote_code_models=self.config.allow_remote_code_models
            )
        return self._semantic_search

    async def search(
        self,
        query: str,
        *,
        domains: list[str] | None = None,
    ) -> list[PaperReference]:
        """
        Search academic literature for papers matching the query.

        Args:
            query: Search query string
            domains: Optional list of domains to filter by (e.g., ["math", "cs"])

        Returns:
            List of PaperReference objects sorted by relevance (highest first)
        """
        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []

        # Check cache first
        cached = self._get_cached(query)
        if cached is not None:
            logger.debug(f"Cache hit for query: {query}")
            return cached

        all_results: list[PaperReference] = []

        # Search enabled backends in parallel
        tasks = []
        if self.config.semantic_scholar_enabled:
            tasks.append(
                self._search_semantic_scholar(query, limit=self.config.max_results_per_query)
            )
        if self.config.arxiv_enabled:
            tasks.append(
                self._search_arxiv(query, limit=self.config.max_results_per_query)
            )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Search backend error: {result}")
                else:
                    all_results.extend(result)

        # Deduplicate by title similarity
        unique_results = list(dict.fromkeys(all_results))

        # Filter by relevance threshold
        filtered = [
            p
            for p in unique_results
            if p.relevance_score >= self.config.min_relevance_threshold
        ]

        # Sort by relevance (descending)
        filtered.sort(key=lambda p: p.relevance_score, reverse=True)

        # Cache results
        self._set_cache(query, filtered)

        return filtered

    async def deep_search(
        self, query: str, depth: int = 2
    ) -> list[PaperReference]:
        """
        Conduct deep search combining keywords, citation graph, and semantic reranking.

        Args:
            query: Search query
            depth: Citation graph traversal depth

        Returns:
            Comprehensive list of relevant papers
        """
        # Phase 1: Initial keyword search
        keyword_results = await self.search(query)
        if not keyword_results:
            return []

        all_papers: dict[str, PaperReference] = {
            p.paper_id: p for p in keyword_results
        }

        # Phase 2: Citation graph expansion
        top_paper = keyword_results[0]
        try:
            graph = await self.citation_graph.build_from_paper(
                top_paper.paper_id, depth=depth
            )
            graph_papers = await self.citation_graph.get_references(
                top_paper.paper_id
            )
            for paper in graph_papers:
                if paper.paper_id not in all_papers:
                    all_papers[paper.paper_id] = paper
        except Exception as e:
            logger.debug(f"Citation graph expansion failed: {e}")

        # Phase 3: Semantic reranking
        all_papers_list = list(all_papers.values())
        if all_papers_list:
            reranked = await self.semantic_search.search_by_embedding(
                query, all_papers_list, top_k=min(20, len(all_papers_list))
            )
            return [p for p, _ in reranked]

        return all_papers_list

    async def survey_topic(
        self, topic: str, llm: Any | None = None
    ) -> dict[str, Any]:
        """
        Generate a structured literature survey for a topic.

        Args:
            topic: Research topic
            llm: Optional LLM for analysis

        Returns:
            Dictionary with survey structure
        """
        try:
            # Find frontier papers
            frontier = await self.citation_graph.find_research_frontier(
                topic, year_min=2023
            )

            # Find common ancestors (foundational papers)
            if frontier and len(frontier) > 1:
                frontier_ids = [p.paper_id for p in frontier[:5]]
                ancestors = await self.citation_graph.find_common_ancestors(
                    frontier_ids
                )
            else:
                ancestors = []

            # Analyze gaps
            gap_analyzer = ResearchGapAnalyzer()
            gaps = await gap_analyzer.analyze(frontier[:10], llm)

            # Build timeline
            timeline = self._build_timeline(frontier + ancestors)

            return {
                "topic": topic,
                "frontier_papers": frontier[:10],
                "foundational_papers": ancestors[:5],
                "research_gaps": gaps,
                "timeline": timeline,
                "papers_count": len(frontier),
            }

        except Exception as e:
            logger.error(f"Survey generation failed for {topic}: {e}")
            return {"error": str(e)}

    async def find_research_gaps(
        self, topic: str, llm: Any | None = None
    ) -> list[str]:
        """
        Identify research gaps in a topic area.

        Args:
            topic: Research topic
            llm: LLM instance for analysis

        Returns:
            List of identified research gaps
        """
        try:
            papers = await self.deep_search(topic)
            analyzer = ResearchGapAnalyzer()
            return await analyzer.analyze(papers[:20], llm)
        except Exception as e:
            logger.error(f"Gap finding failed for {topic}: {e}")
            return []

    def _build_timeline(
        self, papers: list[PaperReference]
    ) -> dict[int, int]:
        """Build publication timeline from papers."""
        timeline: dict[int, int] = {}
        for paper in papers:
            if paper.year > 0:
                timeline[paper.year] = timeline.get(paper.year, 0) + 1
        return dict(sorted(timeline.items()))

    async def _search_semantic_scholar(
        self, query: str, limit: int = 10
    ) -> list[PaperReference]:
        """
        Search Semantic Scholar Academic Graph API.

        Endpoint: https://api.semanticscholar.org/graph/v1/paper/search

        Args:
            query: Search query
            limit: Maximum results to fetch

        Returns:
            List of PaperReference objects
        """
        params = urlencode(
            {
                "query": query,
                "limit": limit,
                "fields": (
                    "title,authors,year,abstract,url,externalIds,"
                    "citationCount,referenceCount,influentialCitationCount,embedding"
                ),
            }
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"

        try:
            response_text = await self._http_get(url)
            if not response_text:
                return []

            data = json.loads(response_text)
            papers: list[PaperReference] = []

            for item in data.get("data", []):
                paper = PaperReference(
                    paper_id=item.get("paperId", ""),
                    title=item.get("title", ""),
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    year=item.get("year", 0),
                    abstract=item.get("abstract", ""),
                    url=item.get("url", ""),
                    citation_count=item.get("citationCount", 0),
                    reference_count=item.get("referenceCount", 0),
                    influential_citation_count=item.get("influentialCitationCount", 0),
                    embedding=item.get("embedding"),
                )

                # Extract arXiv URL if available
                if "externalIds" in item and "ArXiv" in item["externalIds"]:
                    arxiv_id = item["externalIds"]["ArXiv"]
                    paper.url = f"https://arxiv.org/abs/{arxiv_id}"

                paper.relevance_score = self._compute_relevance(paper, query)
                papers.append(paper)

            logger.debug(
                f"Semantic Scholar: found {len(papers)} papers for query '{query}'"
            )
            return papers

        except Exception as e:
            logger.error(f"Semantic Scholar search failed for '{query}': {e}")
            return []

    async def _search_arxiv(self, query: str, limit: int = 10) -> list[PaperReference]:
        """
        Search arXiv API.

        Endpoint: http://export.arxiv.org/api/query

        Args:
            query: Search query
            limit: Maximum results to fetch

        Returns:
            List of PaperReference objects
        """
        encoded_query = quote_plus(f"all:{query}")
        url = (
            "https://export.arxiv.org/api/query"
            f"?search_query={encoded_query}&max_results={limit}"
        )

        try:
            response_text = await self._http_get(url)
            if not response_text:
                return []

            papers: list[PaperReference] = []
            root = ET.fromstring(response_text)

            # arXiv uses Atom XML namespace
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", ns):
                paper_id = entry.findtext("atom:id", "", ns).split("/abs/")[-1]
                title = entry.findtext("atom:title", "", ns).strip()
                authors = [
                    author.findtext("atom:name", "", ns)
                    for author in entry.findall("atom:author", ns)
                ]
                abstract = entry.findtext("atom:summary", "", ns).strip()
                url = f"https://arxiv.org/abs/{paper_id}"

                # Parse publication date (arXiv updated date, best proxy for year)
                updated = entry.findtext("atom:updated", "", ns)
                year = 0
                if updated:
                    try:
                        year = int(updated.split("-")[0])
                    except (ValueError, IndexError):
                        pass

                paper = PaperReference(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    url=url,
                )
                paper.relevance_score = self._compute_relevance(paper, query)
                papers.append(paper)

            logger.debug(f"arXiv: found {len(papers)} papers for query '{query}'")
            return papers

        except Exception as e:
            logger.error(f"arXiv search failed for '{query}': {e}")
            return []

    async def _http_get(self, url: str) -> str:
        """
        Fetch URL content with multiple fallback strategies.

        Tries in order:
        1. autoforge.engine.tools.web.fetch_url (if available)
        2. asyncio subprocess with curl
        3. urllib.request as final fallback

        Args:
            url: URL to fetch

        Returns:
            Response text (empty string on failure)
        """
        # Strategy 1: Try project's web tools
        try:
            from autoforge.engine.tools.web import fetch_url

            try:
                response = await asyncio.to_thread(fetch_url, url)
                return response
            except Exception as e:
                logger.debug(f"fetch_url failed: {e}")
        except ImportError:
            logger.debug("autoforge.engine.tools.web not available")

        # Strategy 2: Try asyncio subprocess with curl
        try:
            process = await asyncio.create_subprocess_exec(
                "curl",
                "-s",
                "-L",
                url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10.0)
            return stdout.decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug(f"curl fallback failed: {e}")

        # Strategy 3: urllib.request fallback
        try:
            import urllib.request

            def _urlopen() -> str:
                with urllib.request.urlopen(url, timeout=10) as response:
                    return response.read().decode("utf-8", errors="replace")

            return await asyncio.to_thread(_urlopen)
        except Exception as e:
            logger.error(f"All HTTP strategies failed for {url}: {e}")
            return ""

    def _compute_relevance(self, paper: PaperReference, query: str) -> float:
        """
        Compute relevance score for a paper given a query.

        Uses TF-IDF-like scoring on title and abstract with recency boost.

        Args:
            paper: Paper to score
            query: Original search query

        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Normalize text
        query_terms = set(query.lower().split())
        title_words = set(paper.title.lower().split())
        abstract_words = set(paper.abstract.lower().split())

        # Compute overlap
        title_overlap = len(query_terms & title_words) / len(query_terms) if query_terms else 0
        abstract_overlap = (
            len(query_terms & abstract_words) / len(query_terms) if query_terms else 0
        )

        # Weighted combination (title more important than abstract)
        base_score = 0.7 * title_overlap + 0.3 * abstract_overlap

        # Boost recent papers (within 2 years)
        current_year = datetime.now().year
        if paper.year > 0 and current_year - paper.year <= 2:
            base_score *= 1.1

        return min(base_score, 1.0)

    def _is_cached(self, query: str) -> bool:
        """Check if query results are in cache and not expired."""
        if query not in self._cache:
            return False
        timestamp, _ = self._cache[query]
        age_hours = (time.time() - timestamp) / 3600
        return age_hours < self.config.cache_ttl_hours

    def _get_cached(self, query: str) -> list[PaperReference] | None:
        """Get cached results if still valid."""
        if not self._is_cached(query):
            return None
        _, results = self._cache[query]
        return results

    def _set_cache(self, query: str, results: list[PaperReference]) -> None:
        """Cache search results with current timestamp."""
        self._cache[query] = (time.time(), results)


class LiteratureGroundedNoveltyFilter:
    """
    Novelty filter that grounds decisions in academic literature.

    Combines pattern matching against known statements with literature search
    to verify that a conjecture isn't already published in academic papers.
    """

    def __init__(
        self,
        known_statements: list[str],
        threshold: float = 0.7,
        search_engine: LiteratureSearchEngine | None = None,
    ) -> None:
        """
        Initialize the literature-grounded novelty filter.

        Args:
            known_statements: List of known mathematical statements to check against
            threshold: Novelty score threshold (0.0-1.0)
            search_engine: Optional pre-configured search engine
        """
        self.known_statements = known_statements
        self.threshold = threshold
        self._search_engine = search_engine or LiteratureSearchEngine()
        self._literature_hits: list[PaperReference] = []

    async def check_novelty(
        self,
        candidate: Any,
        llm: Any,
    ) -> tuple[bool, float, str, list[PaperReference]]:
        """
        Check if a candidate conjecture is novel given the literature.

        Args:
            candidate: ConceptNode with formal_statement attribute
            llm: LLM instance with async __call__ method for comparison

        Returns:
            Tuple of (is_novel, novelty_score, reason, related_papers)
        """
        try:
            # Extract key terms from formal statement
            search_query = self._extract_search_terms(candidate.formal_statement)
            if not search_query:
                logger.warning(f"Could not extract search terms from: {candidate.formal_statement}")
                return (True, 0.5, "No searchable terms in statement", [])

            # Search literature
            related_papers = await self._search_engine.search(search_query)
            self._literature_hits.extend(related_papers)

            # If no papers found, candidate is likely novel
            if not related_papers:
                return (
                    True,
                    0.95,
                    "No related papers found in literature",
                    [],
                )

            # Filter highly relevant papers
            highly_relevant = [p for p in related_papers if p.relevance_score > 0.7]
            if not highly_relevant:
                return (
                    True,
                    0.8,
                    f"Found {len(related_papers)} papers but none highly relevant",
                    [],
                )

            # Use LLM to compare candidate against top papers
            is_novel, score, reason = await self._llm_compare_with_literature(
                candidate, highly_relevant, llm
            )

            return (is_novel, score, reason, highly_relevant)

        except Exception as e:
            logger.error(f"Novelty check failed: {e}")
            return (True, 0.5, f"Novelty check error: {e}", [])

    def _extract_search_terms(self, statement: str) -> str:
        """
        Extract mathematical keywords from a formal statement for search.

        Removes LaTeX commands and keeps meaningful terms.

        Args:
            statement: Formal mathematical statement

        Returns:
            Search query string
        """
        # Remove LaTeX commands like \alpha, \beta, etc.
        cleaned = re.sub(r"\\[a-zA-Z]+", "", statement)

        # Remove parentheses and brackets
        cleaned = re.sub(r"[{}\[\]()]", " ", cleaned)

        # Split into terms and filter
        terms = cleaned.split()
        keywords = [
            t for t in terms if len(t) > 2 and not re.match(r"^[=<>+-]$", t)
        ]

        # Return as space-separated query, limit to 5 most relevant terms
        return " ".join(keywords[:5])

    async def _llm_compare_with_literature(
        self,
        candidate: Any,
        papers: list[PaperReference],
        llm: Any,
    ) -> tuple[bool, float, str]:
        """
        Use LLM to compare candidate statement with found papers.

        Args:
            candidate: ConceptNode with formal_statement
            papers: List of relevant PaperReference objects
            llm: LLM instance for comparison

        Returns:
            Tuple of (is_novel, score, reason)
        """
        try:
            from autoforge.engine.llm_router import TaskComplexity

            # Build comparison prompt
            paper_summaries = "\n".join(
                [
                    f"- {p.title} ({p.year}) by {', '.join(p.authors[:2])}: {p.abstract[:200]}"
                    for p in papers[:3]
                ]
            )

            prompt = f"""
Given this candidate conjecture:
{candidate.formal_statement}

And these related papers from the literature:
{paper_summaries}

Is the candidate conjecture truly novel, or has it already been published?
Respond with:
1. NOVEL or PUBLISHED
2. Confidence (0.0-1.0)
3. Brief reason

Format: STATUS|CONFIDENCE|REASON
"""

            response = await llm.call(prompt, complexity=TaskComplexity.HIGH)
            parts = response.content.strip().split("|")

            if len(parts) >= 3:
                status = parts[0].strip().upper()
                try:
                    confidence = float(parts[1].strip())
                except (ValueError, IndexError):
                    confidence = 0.5
                reason = parts[2].strip()

                is_novel = status == "NOVEL"
                return (is_novel, confidence, reason)
            else:
                return (True, 0.5, "LLM response format unclear")

        except Exception as e:
            logger.error(f"LLM comparison failed: {e}")
            return (True, 0.5, f"LLM comparison error: {e}")

    def get_related_papers(self) -> list[PaperReference]:
        """
        Get all papers encountered during novelty checks.

        Useful for citation generation and background synthesis.

        Returns:
            List of accumulated PaperReference objects
        """
        # Deduplicate and return
        return list(dict.fromkeys(self._literature_hits))
