"""Shared utilities for the AutoForge engine.

Contains common helpers used across multiple modules to avoid duplication.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)


_TOKEN_ENCODER: Any | None = None


def _load_tiktoken_encoder() -> Any | None:
    """Load a tokenizer on demand.

    tiktoken is optional. If unavailable, fall back to a conservative
    character-based estimate so the system keeps running.
    """
    global _TOKEN_ENCODER
    if _TOKEN_ENCODER is not None:
        return _TOKEN_ENCODER

    try:
        import tiktoken  # type: ignore

        _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # pragma: no cover - optional dependency path
        logger.debug("[utils] tiktoken unavailable, using fallback token estimate: %s", e)
        _TOKEN_ENCODER = None
    return _TOKEN_ENCODER


def _parse_relaxed_json(text: str) -> Any | None:
    """Parse JSON with relaxed fallback.

    Primary path uses json.loads; fallback uses ast.literal_eval for common
    single-quote / tuple-like drift introduced by LLM output.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError):
            return None


def _repair_json(text: str) -> str:
    """Apply conservative repair rules for common JSON syntax glitches."""
    repaired = re.sub(r",(\s*[}\]])", r"\1", text)
    repaired = re.sub(r"//.*(?=[\n\r])", "", repaired)
    repaired = re.sub(r"/\*.*?\*/", "", repaired, flags=re.DOTALL)
    return repaired.strip()


def _extract_balanced_json_prefix(text: str) -> str:
    """Try to recover a top-level JSON object/array from a potentially truncated suffix."""
    stripped = text.lstrip()
    if not stripped:
        return ""

    if stripped[0] not in ("{", "["):
        return ""

    expected_stack: list[str] = ["]" if stripped[0] == "[" else "}"]
    in_string = False
    quote_char = ""
    escape = False

    for idx, ch in enumerate(stripped[1:], start=1):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
            continue

        if ch in {"\"", "'"}:
            in_string = True
            quote_char = ch
            continue

        if ch in "{[":
            expected_stack.append("}" if ch == "{" else "]")
            continue

        if ch in "}]":
            if not expected_stack or expected_stack[-1] != ch:
                return ""
            expected_stack.pop()
            if not expected_stack:
                return stripped[: idx + 1]
            continue

    # Unclosed JSON: close remaining containers.
    return stripped + "".join(reversed(expected_stack))


def _try_parse_with_truncation_correction(text: str) -> list[str]:
    """Return a sequence of repaired snippets for truncated/broken JSON candidates."""
    trimmed = text.strip()
    if not trimmed or trimmed[0] not in {"{", "["}:
        return []

    first_pass = _extract_balanced_json_prefix(trimmed)
    if not first_pass:
        return []

    repaired = _repair_json(first_pass)
    candidates = [first_pass]
    if repaired and repaired != first_pass:
        candidates.append(repaired)
    return candidates


def _iter_candidate_json_snippets(
    text: str,
    *,
    allow_objects: bool = True,
    allow_arrays: bool = True,
) -> list[str]:
    """Collect likely JSON snippets from text in descending reliability order."""
    candidates: list[str] = []
    seen = set()

    # 1) fenced json blocks
    for match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE):
        snippet = match.group(1).strip()
        if snippet and snippet not in seen:
            candidates.append(snippet)
            seen.add(snippet)

    # 2) any fenced block that looks like raw JSON (all language markers)
    if len(candidates) < 10:
    for match in re.finditer(r"```[^\n]*\n(.*?)```", text, re.DOTALL):
        snippet = match.group(1).strip()
        if snippet in seen:
            continue
        if snippet.startswith("{") or snippet.startswith("["):
            candidates.append(snippet)
            seen.add(snippet)
            if len(candidates) >= 10:
                break

    # 3) raw scanner from brace / bracket starts (JSONDecoder boundary-based)
    decoder = json.JSONDecoder()
    candidates_from_raw = []
    for idx, ch in enumerate(text):
        if (ch == "{" and not allow_objects) or (ch == "[" and not allow_arrays):
            continue
        if ch not in ("{", "["):
            continue
        try:
            _, end = decoder.raw_decode(text, idx)
        except (json.JSONDecodeError, TypeError):
            continue
        except Exception:
            continue
        raw = text[idx:end].strip()
        if raw and raw not in seen:
            candidates_from_raw.append((idx, raw))
            seen.add(raw)
            if len(candidates_from_raw) >= 6:
                break

    for _, raw in sorted(candidates_from_raw, key=lambda item: item[0]):
        candidates.append(raw)

    # 4) Last-resort scan from each opening bracket/object boundary.
    for start in [m.start() for m in re.finditer(r"[{[]", text)]:
        if start in seen:
            continue
        raw_snippet = text[start:]
        if not raw_snippet:
            continue
        if raw_snippet[0] == "{" and not allow_objects:
            continue
        if raw_snippet[0] == "[" and not allow_arrays:
            continue
        for repaired in _try_parse_with_truncation_correction(raw_snippet):
            if not repaired:
                continue
            normalized = repaired
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(normalized)
            if len(candidates) >= 20:
                break
        if len(candidates) >= 20:
            break

    return candidates


def _iter_parsed_candidates(
    text: str,
    *,
    allow_objects: bool,
    allow_arrays: bool,
) -> list[Any]:
    parsed_candidates: list[Any] = []
    for raw in _iter_candidate_json_snippets(text, allow_objects=allow_objects, allow_arrays=allow_arrays):
        parsed = _parse_relaxed_json(raw)
        if parsed is not None:
            parsed_candidates.append(parsed)
            continue

        repaired = _repair_json(raw)
        parsed = _parse_relaxed_json(repaired)
        if parsed is not None:
            parsed_candidates.append(parsed)
            continue

    return parsed_candidates


def _required_fields_present(payload: Any, required_fields: Iterable[str] | None) -> bool:
    if not required_fields:
        return True
    if not isinstance(payload, dict):
        return False
    return all(field in payload for field in required_fields)


def _matches_schema(payload: Any, schema: dict[str, Any] | None) -> bool:
    if schema is None:
        return True
    if not isinstance(payload, dict):
        return False

    required = schema.get("required", [])
    if required and not _required_fields_present(payload, required):
        return False
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return True
    for key, rule in properties.items():
        expected = rule.get("type") if isinstance(rule, dict) else None
        if key not in payload:
            continue
        if expected is None:
            continue
        if not _value_matches_schema(payload[key], rule):
            return False

    return True


def _value_matches_schema(value: Any, schema: dict[str, Any] | None) -> bool:
    """Validate one value against a tiny subset of JSON Schema."""
    if schema is None:
        return True
    if not isinstance(schema, dict):
        return True

    schema_type = schema.get("type")
    if not schema_type:
        return True

    if schema_type == "object":
        if not isinstance(value, dict):
            return False
        nested_props = schema.get("properties", {})
        if not isinstance(nested_props, dict):
            return True
        for nested_key, nested_rule in nested_props.items():
            if nested_key not in value:
                continue
            if not _value_matches_schema(value[nested_key], nested_rule):
                return False
        required = schema.get("required", [])
        if required and not all(key in value for key in required):
            return False
        return True

    if schema_type == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        if item_schema is None:
            return True
        return all(_value_matches_schema(item, item_schema) for item in value)

    expected = schema_type
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float)
    if expected == "integer":
        return isinstance(value, int)
    if expected == "null":
        return value is None
    return True


def extract_json_from_text(
    text: str,
    required_fields: Iterable[str] | None = None,
    schema: dict[str, Any] | None = None,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Extract a JSON object from LLM output text."""
    if not text:
        raise ValueError("Empty text no JSON to extract")

    for payload in _iter_parsed_candidates(text, allow_objects=True, allow_arrays=False):
        if not isinstance(payload, dict):
            continue
        if (
            _required_fields_present(payload, required_fields)
            and _matches_schema(payload, schema)
        ):
            return payload

    if strict:
        if required_fields:
            raise ValueError(
                "No valid JSON object found in strict mode with required fields "
                f"{sorted(required_fields)}"
            )
        raise ValueError("No valid JSON object found in strict mode")

    raise ValueError(f"No valid JSON object found in text (length={len(text)})")


def extract_json_list_from_text(
    text: str,
    item_schema: dict[str, Any] | None = None,
    *,
    strict: bool = False,
) -> list[Any]:
    """Extract a JSON array from LLM output text."""
    if not text:
        raise ValueError("Empty text no JSON to extract")

    for payload in _iter_parsed_candidates(text, allow_objects=False, allow_arrays=True):
        if isinstance(payload, list):
            if item_schema is None:
                return payload
            for item in payload:
                if not _matches_schema(item, item_schema):
                    break
            else:
                return payload
            # skip invalid candidates
            continue

    if strict:
        raise ValueError("No valid JSON array found in strict mode")

    raise ValueError(f"No valid JSON array found in text (length={len(text)})")


def count_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses tiktoken if available; falls back to 4 chars/token when unavailable.
    """
    if not text:
        return 0
    encoder = _load_tiktoken_encoder()
    if encoder is None:
        return max(1, len(text) // 4)
    try:
        return len(encoder.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _truncate_by_lines(text: str, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text

    chunks = re.split(r"\n{2,}", text)
    if not chunks:
        return text[: max_tokens * 4]

    out: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks):
        candidate = chunk if i == 0 else f"\n\n{chunk}"
        c_tokens = count_tokens(candidate)
        if used + c_tokens <= max_tokens:
            out.append(candidate)
            used += c_tokens
            continue

        remain = max_tokens - used
        if remain <= 0:
            break
        plain = chunk[: remain * 4]
        if plain:
            out.append(plain if i == 0 else f"\n\n{plain}")
            used += count_tokens(plain)
        break

    return "".join(out)


def _truncate_code_block_safely(code_block: str, max_tokens: int) -> str:
    """Truncate a fenced code block while keeping parser-level boundaries when possible."""
    if not code_block or max_tokens <= 0:
        return ""

    m = re.match(r"```(?P<lang>[^\n]*)\n(?P<body>[\s\S]*?)\n?```$", code_block.strip())
    if not m:
        return _truncate_by_lines(code_block, max_tokens)

    lang = (m.group("lang") or "").strip().lower()
    body = m.group("body")
    header = f"```{m.group('lang')}\n"
    footer = "\n```"

    if count_tokens(body) + count_tokens(header) + count_tokens(footer) <= max_tokens:
        return code_block.strip()

    budget = max(1, max_tokens - count_tokens(header) - count_tokens(footer))
    if budget <= 0:
        return header + footer

    if lang not in {"python", "py", "python3"}:
        return header + _truncate_by_lines(body, budget) + footer

    lines = body.splitlines(keepends=True)
    if not lines:
        return header + footer

    try:
        parsed = ast.parse(body)
    except SyntaxError:
        return header + _truncate_by_lines(body, budget) + footer

    kept: list[str] = []
    used = 0
    for node in parsed.body:
        start = max(0, (node.lineno or 1) - 1)
        end = max(start, (getattr(node, "end_lineno", node.lineno) or node.lineno) - 1)
        snippet = "".join(lines[start:end + 1])
        snippet_tokens = count_tokens(snippet)
        if used + snippet_tokens > budget:
            break
        kept.append(snippet)
        used += snippet_tokens

    if kept:
        return header + "".join(kept).rstrip("\n") + footer

    return header + _truncate_by_lines(body, budget) + footer


def truncate_text_to_token_budget(
    text: str,
    max_tokens: int,
    *,
    preserve_code_blocks: bool = True,
) -> str:
    """Truncate text by token budget with boundary-aware fallback.

    Code-fence boundaries are preserved first so we avoid cutting middle of a
    code block. Remaining budget is consumed by paragraph/line boundaries.
    """
    if not text or max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text

    if not preserve_code_blocks or "```" not in text:
        return _truncate_by_lines(text, max_tokens)

    code_block = re.compile(r"```[^\n]*\n.*?\n```", re.DOTALL)
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in code_block.finditer(text):
        if m.start() > pos:
            parts.append(("text", text[pos:m.start()]))
        parts.append(("code", m.group(0)))
        pos = m.end()
    if pos < len(text):
        parts.append(("text", text[pos:]))

    out: list[str] = []
    used = 0
    for kind, part in parts:
        remaining = max_tokens - used
        if remaining <= 0:
            break

        p_tokens = count_tokens(part)
        if p_tokens <= remaining:
            out.append(part)
            used += p_tokens
            continue

        if kind == "code":
            snippet = _truncate_code_block_safely(part, remaining)
            if snippet:
                out.append(snippet)
                used = count_tokens("".join(out))
            continue

        truncated = _truncate_by_lines(part, remaining)
        if truncated:
            out.append(truncated)
            used += count_tokens(truncated)

    return "".join(out)


# Maximum file size for agent write operations (2 MB)
MAX_AGENT_FILE_SIZE = 2 * 1024 * 1024

# Truncation limits for command output
MAX_STDOUT_CHARS = 5000
MAX_STDERR_CHARS = 2000
