"""Shared utilities for the AutoForge engine.

Contains common helpers used across multiple modules to avoid duplication.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output text.

    Handles:
    1. JSON inside ```json ... ``` fenced code blocks
    2. Raw JSON objects in the text (with proper brace matching)
    3. Nested JSON objects (unlike the naive \\{.*?\\} regex)

    Args:
        text: Raw LLM output that may contain JSON.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    if not text:
        raise ValueError("Empty text — no JSON to extract")

    # Strategy 1: Look for fenced code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass  # Fall through to other strategies

    # Strategy 2: Find balanced braces (handles nested objects)
    start = text.find("{")
    if start != -1:
        # Use json.JSONDecoder to find the end of the JSON object
        decoder = json.JSONDecoder()
        try:
            result, _end = decoder.raw_decode(text, start)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: try from the last '{' if first didn't work
        # (sometimes LLM outputs preamble text with stray braces)
        last_start = text.rfind("{")
        if last_start != start:
            try:
                result, _end = decoder.raw_decode(text, last_start)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

    raise ValueError(
        f"No valid JSON object found in text (length={len(text)})"
    )


def extract_json_list_from_text(text: str) -> list[Any]:
    """Extract a JSON array from LLM output text.

    Similar to extract_json_from_text but expects a list instead of a dict.

    Args:
        text: Raw LLM output that may contain a JSON array.

    Returns:
        Parsed JSON list.

    Raises:
        ValueError: If no valid JSON array can be extracted.
    """
    if not text:
        raise ValueError("Empty text — no JSON to extract")

    # Strategy 1: Look for fenced code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find balanced brackets
    start = text.find("[")
    if start != -1:
        decoder = json.JSONDecoder()
        try:
            result, _end = decoder.raw_decode(text, start)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"No valid JSON array found in text (length={len(text)})"
    )


# Maximum file size for agent write operations (2 MB)
MAX_AGENT_FILE_SIZE = 2 * 1024 * 1024

# Truncation limits for command output
MAX_STDOUT_CHARS = 5000
MAX_STDERR_CHARS = 2000
