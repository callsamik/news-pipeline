"""Orchestrator composing fetch, dedupe, freshness, summarize, and optional store."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Callable

import httpx

from news_pipeline.fetch import ClockFn, FetchTextFn, fetch_source
from news_pipeline.freshness import filter_items_by_window
from news_pipeline.identity import dedupe_by_id
from news_pipeline.models import NewsItem, NewsSource, PipelineRunResult, SourceFetchResult
from news_pipeline.store import NewsStore
from news_pipeline.summarize import summarize_item


def run_pipeline(
    sources: Sequence[NewsSource],
    *,
    window_hours: int = 72,
    max_summary_chars: int = 400,
    store: NewsStore | None = None,
    fetch_text: FetchTextFn | None = None,
    client: httpx.Client | None = None,
    clock: ClockFn | None = None,
) -> PipelineRunResult:
    """Run the full news trunk for caller-selected transport sources."""
    now_fn: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))
    started_at = now_fn()

    source_results: list[SourceFetchResult] = []
    collected: list[NewsItem] = []

    for source in sources:
        result, items = fetch_source(
            source,
            client=client,
            fetch_text=fetch_text,
            clock=now_fn,
        )
        source_results.append(result)
        collected.extend(items)

    deduped = dedupe_by_id(collected)
    fresh = filter_items_by_window(deduped, window_hours=window_hours, now=started_at)
    summarized = [
        summarize_item(item, max_chars=max_summary_chars) for item in fresh
    ]

    if store is not None:
        store.upsert_items(summarized)

    completed_at = now_fn()
    return PipelineRunResult(
        started_at=started_at,
        completed_at=completed_at,
        source_results=source_results,
        items=summarized,
    )
