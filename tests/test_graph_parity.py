"""Golden parity: plain run_pipeline matches optional LangGraph adapter (and TextRank composition)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from news_pipeline.fetch import fetch_source
from news_pipeline.freshness import filter_items_by_window
from news_pipeline.identity import compute_item_id, dedupe_by_id
from news_pipeline.models import NewsSource
from news_pipeline.pipeline import run_pipeline
from news_pipeline.summarize import summarize_item

pytest.importorskip("langgraph")

from news_pipeline.graph import invoke_pipeline  # noqa: E402

NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)

DUP_TITLE = "Breaking News"
DUP_URL = "https://example.test/dup"
FIRST_BODY = "First body content kept by dedupe."
SECOND_BODY = "Second body dropped by dedupe."

FIXTURE_RSS = f"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Test Feed</title>
  <item>
    <title>{DUP_TITLE}</title>
    <link>{DUP_URL}</link>
    <description>{FIRST_BODY}</description>
    <pubDate>Fri, 15 Aug 2026 08:00:00 GMT</pubDate>
  </item>
  <item>
    <title>{DUP_TITLE}</title>
    <link>{DUP_URL}</link>
    <description>{SECOND_BODY}</description>
    <pubDate>Fri, 15 Aug 2026 09:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Old Story</title>
    <link>https://example.test/old</link>
    <description>Ancient history filtered by freshness window.</description>
    <pubDate>Mon, 01 Jun 2026 08:00:00 GMT</pubDate>
  </item>
</channel></rss>
"""


def _compose_textrank(sources: list[NewsSource], *, window_hours: int, max_summary_chars: int):
    """Direct composition using the sole deterministic summarizer (TextRank)."""
    source_results = []
    collected = []
    for source in sources:
        result, items = fetch_source(
            source,
            fetch_text=lambda _url: FIXTURE_RSS,
            clock=lambda: NOW,
        )
        source_results.append(result)
        collected.extend(items)
    deduped = dedupe_by_id(collected)
    fresh = filter_items_by_window(deduped, window_hours=window_hours, now=NOW)
    summarized = [summarize_item(item, max_chars=max_summary_chars) for item in fresh]
    return source_results, summarized


def test_plain_pipeline_matches_textrank_composition():
    source = NewsSource(
        source_id="test_feed",
        url="https://example.test/test_feed/rss",
        kind="rss",
    )
    expected_id = compute_item_id(source.source_id, DUP_URL, DUP_TITLE)
    direct_results, direct_items = _compose_textrank(
        [source],
        window_hours=72,
        max_summary_chars=200,
    )

    result = run_pipeline(
        [source],
        window_hours=72,
        max_summary_chars=200,
        fetch_text=lambda _url: FIXTURE_RSS,
        clock=lambda: NOW,
    )

    assert result.started_at == NOW
    assert result.completed_at == NOW
    assert result.stats.summarization_mode == "textrank"
    assert result.stats.llm_batches_ok == 0
    assert result.stats.llm_batches_fallback == 0

    assert len(result.source_results) == len(direct_results)
    for got, expected in zip(result.source_results, direct_results, strict=True):
        assert got.source_id == expected.source_id
        assert got.ok == expected.ok
        assert got.item_count == expected.item_count
        assert got.error == expected.error

    assert len(result.items) == len(direct_items) == 1
    for got, expected in zip(result.items, direct_items, strict=True):
        assert got == expected
        assert got.id == expected_id
        assert got.title == DUP_TITLE
        assert got.raw_text == FIRST_BODY
        assert got.url == DUP_URL
        assert got.published_at == expected.published_at
        assert got.summary == expected.summary


def test_graph_adapter_matches_plain_pipeline():
    source = NewsSource(
        source_id="test_feed",
        url="https://example.test/test_feed/rss",
        kind="rss",
    )
    kwargs = dict(
        window_hours=72,
        max_summary_chars=200,
        fetch_text=lambda _url: FIXTURE_RSS,
        clock=lambda: NOW,
    )
    plain = run_pipeline([source], **kwargs)
    graph = invoke_pipeline(
        [source],
        window_hours=kwargs["window_hours"],
        max_summary_chars=kwargs["max_summary_chars"],
        fetch_text=kwargs["fetch_text"],
        clock=kwargs["clock"],
    )

    assert plain.stats == graph.stats
    assert plain.items == graph.items
    assert len(plain.source_results) == len(graph.source_results)
    for a, b in zip(plain.source_results, graph.source_results, strict=True):
        assert a.source_id == b.source_id
        assert a.ok == b.ok
        assert a.item_count == b.item_count
        assert a.error == b.error
