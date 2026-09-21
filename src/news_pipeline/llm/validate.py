"""Validate LLM batch summarization responses (all-or-nothing)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", flags=re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    summaries: list[str]
    error: str | None = None


def strip_think(text: str) -> str:
    """Drop chain-of-thought wrappers before JSON parse."""
    text = _THINK_BLOCK_RE.sub("", text or "")
    open_m = list(_THINK_OPEN_RE.finditer(text))
    if open_m:
        text = text[open_m[-1].end() :]
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = strip_think(text or "")
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("no JSON object in model response")
        text = m.group(0)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object")
    return parsed


def validate_summaries_payload(
    raw_text: str | None,
    *,
    expected_count: int,
) -> ValidationResult:
    """Accept only exact-cardinality string summaries; never partial."""
    if raw_text is None or not str(raw_text).strip():
        return ValidationResult(ok=False, summaries=[], error="empty response")

    try:
        payload = extract_json_object(str(raw_text))
    except Exception as exc:  # noqa: BLE001 — malformed → reject batch
        return ValidationResult(ok=False, summaries=[], error=f"malformed JSON: {exc}")

    summaries = payload.get("summaries")
    if not isinstance(summaries, list):
        return ValidationResult(
            ok=False,
            summaries=[],
            error="summaries missing or not a list",
        )

    string_summaries: list[str] = []
    for entry in summaries:
        if not isinstance(entry, str):
            return ValidationResult(
                ok=False,
                summaries=[],
                error="summaries entries must be strings",
            )
        cleaned = entry.strip()
        if not cleaned:
            return ValidationResult(
                ok=False,
                summaries=[],
                error="summaries entries must be non-empty strings",
            )
        string_summaries.append(cleaned)

    if len(string_summaries) != expected_count:
        return ValidationResult(
            ok=False,
            summaries=[],
            error=(
                f"length mismatch: got {len(string_summaries)}, "
                f"expected {expected_count}"
            ),
        )

    return ValidationResult(ok=True, summaries=string_summaries, error=None)
