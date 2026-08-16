"""Pipeline orchestrator: fetch → dedupe → freshness → summarize (Task 7)."""

from __future__ import annotations

from datetime import datetime, timezone

from news_pipeline.identity import compute_item_id
from news_pipeline.models import NewsSource
from news_pipeline.pipeline import run_pipeline
from news_pipeline.store import SqliteNewsStore

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
    <pubDate>Mon, 01 Jul 2026 08:00:00 GMT</pubDate>
  </item>
</channel></rss>
"""


def _feed_source(*, source_id: str = "test_feed") -> NewsSource:
    return NewsSource(
        source_id=source_id,
        url=f"https://example.test/{source_id}/rss",
        kind="rss",
    )


def test_run_pipeline_dedupes_filters_and_summarizes():
    source = _feed_source()
    expected_id = compute_item_id(source.source_id, DUP_URL, DUP_TITLE)

    result = run_pipeline(
        [source],
        window_hours=72,
        max_summary_chars=200,
        fetch_text=lambda _url: FIXTURE_RSS,
        clock=lambda: NOW,
    )

    assert result.started_at == NOW
    assert result.completed_at == NOW
    assert len(result.source_results) == 1
    assert result.source_results[0].ok is True
    assert result.source_results[0].item_count == 3

    assert len(result.items) == 1
    item = result.items[0]
    assert item.id == expected_id
    assert item.title == DUP_TITLE
    assert item.raw_text == FIRST_BODY
    assert item.summary
    assert len(item.summary) <= 200


def test_run_pipeline_optional_store_persists_summarized_items(tmp_path):
    source = _feed_source()
    store = SqliteNewsStore(tmp_path / "news.sqlite")

    result = run_pipeline(
        [source],
        window_hours=72,
        store=store,
        fetch_text=lambda _url: FIXTURE_RSS,
        clock=lambda: NOW,
    )

    assert len(result.items) == 1
    loaded = store.get_item(result.items[0].id)
    assert loaded is not None
    assert loaded.summary == result.items[0].summary
    assert loaded.raw_text == ""


def test_run_pipeline_no_store_when_none():
    source = _feed_source()

    result = run_pipeline(
        [source],
        fetch_text=lambda _url: FIXTURE_RSS,
        clock=lambda: NOW,
    )

    assert len(result.items) == 1


def test_run_pipeline_multiple_sources_collects_per_source_results():
    feed_a = _feed_source(source_id="feed_a")
    feed_b = _feed_source(source_id="feed_b")

    def fetch_text(url: str) -> str:
        if "feed_a" in url:
            return FIXTURE_RSS
        return """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Other Feed Item</title>
    <link>https://example.test/other</link>
    <description>Fresh content from second source.</description>
    <pubDate>Sat, 16 Aug 2026 06:00:00 GMT</pubDate>
  </item>
</channel></rss>"""

    result = run_pipeline(
        [feed_a, feed_b],
        window_hours=72,
        fetch_text=fetch_text,
        clock=lambda: NOW,
    )

    assert len(result.source_results) == 2
    assert {r.source_id for r in result.source_results} == {"feed_a", "feed_b"}
    assert all(r.ok for r in result.source_results)
    assert len(result.items) == 2
