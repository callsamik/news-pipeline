#!/usr/bin/env python3
"""Standalone news-pipeline example — no AIN dependency."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from news_pipeline import (
    NewsSource,
    SqliteNewsStore,
    run_pipeline,
)

FIXTURE_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Markets</title>
  <item>
    <title>TCS wins large capacity expansion deal</title>
    <link>https://example.test/tcs-1</link>
    <description>TCS announced growth and profit beat expectations.</description>
    <pubDate>Sat, 16 Aug 2026 06:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Market rally continues</title>
    <link>https://example.test/mkt-1</link>
    <description>Indices higher on strong global cues.</description>
    <pubDate>Sat, 16 Aug 2026 07:00:00 GMT</pubDate>
  </item>
</channel></rss>
"""

NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)


def main() -> None:
    sources = [
        NewsSource(
            source_id="demo_markets",
            url="https://example.test/markets/rss",
            kind="rss",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "news.sqlite"
        store = SqliteNewsStore(store_path)

        result = run_pipeline(
            sources,
            window_hours=72,
            max_summary_chars=200,
            store=store,
            fetch_text=lambda _url: FIXTURE_RSS,
            clock=lambda: NOW,
        )

        print(f"Pipeline run: {result.started_at.isoformat()} → {result.completed_at.isoformat()}")
        for src in result.source_results:
            status = "ok" if src.ok else "fail"
            print(f"  source {src.source_id}: {status} ({src.item_count} parsed)")

        print(f"\nFresh summarized items ({len(result.items)}):")
        for item in result.items:
            print(f"  {item.id}  {item.title}")
            print(f"    summary: {item.summary[:80]}{'…' if len(item.summary) > 80 else ''}")

        persisted = store.list_by_source("demo_markets")
        print(f"\nPersisted in SQLite ({store_path}): {len(persisted)} item(s)")


if __name__ == "__main__":
    main()
