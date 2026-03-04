"""Web tools — fetch_url and search_web for agents.

Provides two tool handlers:
  - handle_fetch_url: Fetches a URL and returns text content (HTML stripped)
  - handle_search_web: Searches the web using configurable backends

Dependencies: httpx, html2text, duckduckgo-search (optional: google/bing API keys)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Maximum response size to avoid blowing context windows
_MAX_FETCH_CHARS = 12000
_MAX_FETCH_BYTES = 500_000  # 500 KB download limit
_FETCH_TIMEOUT = 15  # seconds
_DEFAULT_NUM_RESULTS = 5


async def handle_fetch_url(input_data: dict[str, Any]) -> str:
    """Fetch a URL and return its text content (HTML stripped).

    Input: {"url": "https://example.com/docs"}
    Output: Plain text content (truncated to ~12K chars)
    """
    url = input_data.get("url", "").strip()
    if not url:
        return json.dumps({"error": "Missing required parameter: url"})

    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        import httpx
    except ImportError:
        return json.dumps({
            "error": "httpx not installed. Run: pip install httpx"
        })

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=_FETCH_TIMEOUT,
            headers={"User-Agent": "AutoForge/2.0 (AI Development Tool)"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Check content size
            content_bytes = response.content
            if len(content_bytes) > _MAX_FETCH_BYTES:
                content_bytes = content_bytes[:_MAX_FETCH_BYTES]

            content_type = response.headers.get("content-type", "")
            raw_text = content_bytes.decode("utf-8", errors="replace")

            # Convert HTML to readable text
            if "html" in content_type.lower():
                text = _html_to_text(raw_text)
            else:
                text = raw_text

            # Truncate to reasonable size
            if len(text) > _MAX_FETCH_CHARS:
                text = text[:_MAX_FETCH_CHARS] + "\n\n... (truncated)"

            return json.dumps({
                "url": str(response.url),
                "status": response.status_code,
                "content": text,
            })

    except httpx.HTTPStatusError as e:
        return json.dumps({
            "error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            "url": url,
        })
    except httpx.TimeoutException:
        return json.dumps({"error": f"Request timed out after {_FETCH_TIMEOUT}s", "url": url})
    except Exception as e:
        logger.debug(f"fetch_url error for {url}: {e}")
        return json.dumps({"error": str(e), "url": url})


def _html_to_text(html: str) -> str:
    """Convert HTML to readable plain text.

    Uses html2text if available, falls back to basic tag stripping.
    """
    try:
        import html2text

        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        converter.ignore_emphasis = False
        converter.body_width = 0  # Don't wrap lines
        return converter.handle(html)
    except ImportError:
        # Fallback: basic tag stripping
        return _strip_html_tags(html)


def _strip_html_tags(html: str) -> str:
    """Basic HTML tag stripping fallback (no dependencies)."""
    import re

    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Replace common block elements with newlines
    text = re.sub(r"<(?:br|p|div|h[1-6]|li|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


async def handle_search_web(
    input_data: dict[str, Any],
    backend: str = "duckduckgo",
    api_key: str = "",
) -> str:
    """Search the web and return top results.

    Input: {"query": "React 19 new features", "num_results": 5}
    Output: JSON array of [{title, url, snippet}, ...]

    Backends:
      - "duckduckgo" (default, no API key needed)
      - "google" (requires Google Custom Search API key)
    """
    query = input_data.get("query", "").strip()
    if not query:
        return json.dumps({"error": "Missing required parameter: query"})

    num_results = min(input_data.get("num_results", _DEFAULT_NUM_RESULTS), 10)

    try:
        if backend == "google" and api_key:
            results = await _search_google(query, num_results, api_key)
        else:
            # Default: DuckDuckGo (no API key needed)
            results = await _search_duckduckgo(query, num_results)

        return json.dumps({"query": query, "results": results})

    except Exception as e:
        logger.debug(f"search_web error for '{query}': {e}")
        return json.dumps({"error": str(e), "query": query})


async def _search_duckduckgo(query: str, num_results: int) -> list[dict[str, str]]:
    """Search using DuckDuckGo (no API key required)."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [{"error": "duckduckgo-search not installed. Run: pip install duckduckgo-search"}]

    import asyncio

    def _sync_search() -> list[dict[str, str]]:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results

    # DDGS is sync — run in thread pool
    return await asyncio.to_thread(_sync_search)


async def _search_google(query: str, num_results: int, api_key: str) -> list[dict[str, str]]:
    """Search using Google Custom Search API."""
    try:
        import httpx
    except ImportError:
        return [{"error": "httpx not installed. Run: pip install httpx"}]

    # Google Custom Search requires both API key and Search Engine ID
    # The api_key should be in format "API_KEY:SEARCH_ENGINE_ID"
    parts = api_key.split(":", 1)
    if len(parts) != 2:
        return [{"error": "Google search requires api_key in format 'API_KEY:SEARCH_ENGINE_ID'"}]

    key, cx = parts

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": key, "cx": cx, "q": query, "num": num_results},
        )
        response.raise_for_status()
        data = response.json()

    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    return results


# ── Tool definitions (for agent registration) ──────────────────────

FETCH_URL_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": (
                "The URL to fetch. Returns the page content as readable text "
                "(HTML tags stripped). Use this to read documentation, API specs, "
                "or web pages."
            ),
        },
    },
    "required": ["url"],
}

SEARCH_WEB_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Search query to find relevant web pages. "
                "Use this to research frameworks, libraries, APIs, "
                "or find documentation."
            ),
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results to return (default: 5, max: 10)",
        },
    },
    "required": ["query"],
}
