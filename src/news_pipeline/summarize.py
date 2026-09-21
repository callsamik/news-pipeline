"""TextRank deterministic news summarization (sole non-LLM summarizer).

This module is the **only** deterministic summarization implementation in
``news-pipeline``. Degenerate inputs (empty body, one sentence, no graph edges)
are handled inside TextRank / this function — they are not a second algorithm
(Lead-3, title-overlap extractor, etc.).

Frozen knobs (reproducibility contract — do not make caller-configurable):
  MAX_SENTENCES, SIM_THRESHOLD, REDUNDANCY_THRESHOLD, PAGERANK_DAMPING,
  PAGERANK_ITERS, MAX_GRAPH_SENTENCES, ENGLISH_STOPWORDS.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import replace

from .models import NewsItem

DEFAULT_SUMMARY_MAX_CHARS = 400

# --- Frozen TextRank configuration (OV-NP-DETERMINISTIC-FB-1) ---
MAX_SENTENCES = 3
SIM_THRESHOLD = 0.1
REDUNDANCY_THRESHOLD = 0.7
PAGERANK_DAMPING = 0.85
PAGERANK_ITERS = 40
MAX_GRAPH_SENTENCES = 80

ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "as",
        "by",
        "with",
        "from",
        "into",
        "about",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "i",
        "me",
        "my",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "only",
        "own",
        "same",
        "other",
        "such",
        "up",
        "down",
        "out",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "what",
        "which",
        "who",
        "whom",
        "while",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "through",
        "against",
        "until",
        "because",
        "although",
        "though",
        "whether",
        "either",
        "neither",
        "among",
        "via",
        "per",
        "vs",
        "etc",
    }
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def strip_markup(text: str) -> str:
    """HTML-unescape and drop tags / collapse whitespace."""
    if not text:
        return ""
    cleaned = _TAG_RE.sub(" ", html.unescape(text))
    return _WS_RE.sub(" ", cleaned).strip()


def _sentences(text: str) -> list[str]:
    parts = [p.strip(" \t\r\n-•*") for p in _SENTENCE_RE.split(text) if p.strip()]
    return [p for p in parts if p]


def _tokens(sentence: str) -> frozenset[str]:
    return frozenset(
        t
        for t in _TOKEN_RE.findall(sentence.lower())
        if t not in ENGLISH_STOPWORDS and len(t) > 1
    )


def _cosine(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / math.sqrt(len(a) * len(b))


def _pagerank(weights: list[list[float]]) -> list[float]:
    """Fixed-iteration PageRank on an undirected weighted adjacency matrix."""
    n = len(weights)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # Column-stochastic transition: out-weight normalized rows.
    out_sum = [sum(weights[i]) for i in range(n)]
    scores = [1.0 / n] * n
    d = PAGERANK_DAMPING
    teleport = (1.0 - d) / n

    for _ in range(PAGERANK_ITERS):
        nxt = [teleport] * n
        for i in range(n):
            if out_sum[i] <= 0.0:
                # Dangling: distribute mass uniformly.
                share = d * scores[i] / n
                for j in range(n):
                    nxt[j] += share
                continue
            for j in range(n):
                w = weights[i][j]
                if w > 0.0:
                    nxt[j] += d * scores[i] * (w / out_sum[i])
        scores = nxt
    return scores


def _textrank_select(sentences: list[str], *, max_sentences: int) -> list[int]:
    """Return indices of selected sentences in **article order**."""
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [0]

    token_sets = [_tokens(s) for s in sentences]
    weights = [[0.0] * n for _ in range(n)]
    has_edge = False
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine(token_sets[i], token_sets[j])
            if sim >= SIM_THRESHOLD:
                weights[i][j] = sim
                weights[j][i] = sim
                has_edge = True

    if has_edge:
        scores = _pagerank(weights)
    else:
        # Degenerate TextRank graph (no edges): length then index — not Lead-3.
        scores = [float(len(s)) for s in sentences]

    # Rank: score desc, tie → lower index.
    ranked = sorted(range(n), key=lambda i: (-scores[i], i))

    selected: list[int] = []
    selected_tokens: list[frozenset[str]] = []
    for idx in ranked:
        if len(selected) >= max_sentences:
            break
        cand = token_sets[idx]
        redundant = False
        for prev in selected_tokens:
            if _cosine(cand, prev) >= REDUNDANCY_THRESHOLD:
                redundant = True
                break
        if redundant:
            continue
        selected.append(idx)
        selected_tokens.append(cand)

    selected.sort()  # restore article order
    return selected


def _apply_char_bound(pieces: list[str], *, cap: int) -> str:
    """Drop trailing sentences, then truncate final piece with ellipsis if needed."""
    if not pieces:
        return ""
    kept = list(pieces)
    while len(kept) > 1 and len(" ".join(kept)) > cap:
        kept.pop()
    summary = " ".join(kept).strip()
    if len(summary) > cap:
        summary = summary[: cap - 1].rstrip() + "…"
    return summary


def textrank_summarize(
    text: str,
    *,
    max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
    max_sentences: int = MAX_SENTENCES,
) -> str:
    """Extractive TextRank summary of ``text`` (body only).

    Empty / whitespace → ``\"\"``. Does not consult title (callers may apply
    title fallback separately for empty bodies).
    """
    cleaned = strip_markup(text)
    if not cleaned:
        return ""

    sentences = _sentences(cleaned)
    if not sentences:
        return cleaned[: max(64, int(max_chars))]

    # Pathological bound: first MAX_GRAPH_SENTENCES only.
    if len(sentences) > MAX_GRAPH_SENTENCES:
        sentences = sentences[:MAX_GRAPH_SENTENCES]

    cap = max(64, int(max_chars))
    idxs = _textrank_select(sentences, max_sentences=max(1, int(max_sentences)))
    pieces = [sentences[i] for i in idxs]
    return _apply_char_bound(pieces, cap=cap)


def summarize_text(
    title: str,
    body: str,
    *,
    max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
) -> str:
    """Build a TextRank extractive summary from body; title only if body empty.

    Title-on-empty-body is degenerate handling of this sole summarizer —
    not a Lead-1/Lead-3 algorithm.
    """
    cap = max(64, int(max_chars))
    title_clean = strip_markup(title)
    body_clean = strip_markup(body)

    if not body_clean:
        return title_clean[:cap]

    summary = textrank_summarize(body_clean, max_chars=cap)
    if not summary:
        return title_clean[:cap]
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
