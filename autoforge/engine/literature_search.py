"""
Literature-Grounded Discovery Engine

Evidence-based novelty verification using real academic databases (Semantic Scholar, arXiv).
Implements AI Scientist v2 (2025) literature grounding for discovery validation.

This module searches academic literature to check if a conjecture is truly novel by:
1. Extracting key terms from candidate statements
2. Searching Semantic Scholar and arXiv APIs
3. Computing relevance scores and deduplicating results
4. Comparing candidates against found papers with LLM assistance
"""

from __future__ import annotations

import asyncio
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
    """

    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int = 0
    abstract: str = ""
    url: str = ""
    relevance_score: float = 0.0
    overlap_reason: str = ""

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


class LiteratureSearchEngine:
    """
    Search academic literature via Semantic Scholar and arXiv APIs.

    Features:
    - Async HTTP requests with multiple fallback transports
    - TTL-based caching to reduce API calls
    - Deduplication by title similarity
    - Relevance scoring with recency boost
    """

    def __init__(self, config: LiteratureSearchConfig | None = None) -> None:
        """
        Initialize the literature search engine.

        Args:
            config: Search configuration. Defaults to LiteratureSearchConfig().
        """
        self.config = config or LiteratureSearchConfig()
        self._cache: dict[str, tuple[float, list[PaperReference]]] = {}

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
                "fields": "title,authors,year,abstract,url,externalIds",
            }
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"

        try:
            response_text = await self._http_get(url)
            if not response_text:
                return []

            import json

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

            response = await llm(prompt)
            parts = response.strip().split("|")

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
