"""Compact extractive news summarization (no LLM)."""

from __future__ import annotations

import html
import re
from dataclasses import replace

from .models import NewsItem

DEFAULT_SUMMARY_MAX_CHARS = 400

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def strip_markup(text: str) -> str:
    """HTML-unescape and drop tags / collapse whitespace."""
    if not text:
        return ""
    cleaned = _TAG_RE.sub(" ", html.unescape(text))
    return _WS_RE.sub(" ", cleaned).strip()


def _sentences(text: str) -> list[str]:
    parts = [p.strip(" \t\r\n-•*") for p in _SENTENCE_RE.split(text) if p.strip()]
    return [p for p in parts if p]


def summarize_text(
    title: str,
    body: str,
    *,
    max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
) -> str:
    """Build a short extractive summary from title + body."""
    cap = max(64, int(max_chars))
    title_clean = strip_markup(title)
    body_clean = strip_markup(body)

    if not body_clean:
        return title_clean[:cap]

    title_tokens = {t for t in re.findall(r"[A-Za-z0-9]{3,}", title_clean.upper())}
    scored: list[tuple[int, int, str]] = []
    for idx, sentence in enumerate(_sentences(body_clean)):
        tokens = {t for t in re.findall(r"[A-Za-z0-9]{3,}", sentence.upper())}
        overlap = len(tokens & title_tokens)
        scored.append((-overlap, idx, sentence))
    scored.sort()

    pieces: list[str] = []
    used = 0
    for _, _, sentence in scored:
        if used >= cap:
            break
        if sentence.upper() == title_clean.upper():
            continue
        room = cap - used - (2 if pieces else 0)
        if room < 24:
            break
        chunk = sentence if len(sentence) <= room else sentence[: room - 1].rstrip() + "…"
        pieces.append(chunk)
        used += len(chunk) + (2 if len(pieces) > 1 else 0)

    summary = " ".join(pieces).strip()
    if not summary:
        summary = body_clean[:cap]
    if len(summary) > cap:
        summary = summary[: cap - 1].rstrip() + "…"
    return summary


def summarize_item(
    item: NewsItem,
    *,
    max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
) -> NewsItem:
    """Return a copy with ``summary`` filled; ``raw_text`` is preserved."""
    source_body = item.raw_text or item.summary or ""
    summary = summarize_text(item.title, source_body, max_chars=max_chars)
    if not summary:
        summary = strip_markup(item.title)[: max(64, int(max_chars))]
    return replace(item, summary=summary)
