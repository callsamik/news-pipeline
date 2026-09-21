"""Prompts for positional batch summarization (no identity fields)."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = (
    "You are a JSON-only news summarizer.\n"
    "Your entire reply MUST be one JSON object. No markdown fences, no prose, "
    "no questions, no apologies, no <think> tags in the final answer.\n"
    "INPUT: feed_items is an ordered array. Each item has title and text only.\n"
    "Do NOT invent identifiers. Do NOT emit news_id, index, or position fields.\n"
    "OUTPUT shape (exactly):\n"
    '{"summaries":["<summary for item 0>","<summary for item 1>", "..."]}\n'
    "Return exactly one string in summaries for every supplied feed item, "
    "same order. Each summary is a faithful 2-4 sentence paragraph using only "
    "that item's title and text. No confidence."
)


def build_feed_payload(items: list[dict[str, str]]) -> dict[str, Any]:
    """Build LLM input: ordered title+text only (no news_id / index / ticker)."""
    return {
        "feed_items": [
            {"title": entry["title"], "text": entry["text"]} for entry in items
        ]
    }


def build_user_prompt(payload: dict[str, Any]) -> str:
    n = len(payload.get("feed_items") or [])
    return (
        f"Summarize all {n} feed items now.\n"
        "Reply with ONLY this JSON object (no other text):\n"
        '{"summaries":["...","...",...]}\n'
        f"The summaries array MUST have length {n}. "
        "strings only; same order as feed_items; no news_id; no index; "
        "do not ask clarifying questions.\n\n"
        "Example for 2 items:\n"
        '{"summaries":["First article summary...","Second article summary..."]}\n\n'
        + json.dumps(payload, default=str)
    )
