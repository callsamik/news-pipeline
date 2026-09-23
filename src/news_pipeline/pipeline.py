"""Plain-Python orchestrator — required collector path (no LangGraph).

LangGraph, when installed via ``news-pipeline[graph]``, is an optional adapter
in ``news_pipeline.graph`` with parity to this composition.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Callable

import httpx

from news_pipeline.fetch import ClockFn, FetchTextFn, fetch_source
from news_pipeline.freshness import filter_items_by_window
from news_pipeline.identity import dedupe_by_id
from news_pipeline.llm.client_protocol import LLMClient
from news_pipeline.llm.summarize_batch import (
    summarize_items_llm,
    summarize_items_textrank,
)
from news_pipeline.models import (
    NewsItem,
    NewsSource,
    PipelineRunResult,
    SourceFetchResult,
    SummarizationConfig,
)
from news_pipeline.store import NewsStore


def _llm_enabled(
    llm_client: LLMClient | None,
    summarization: SummarizationConfig | None,
) -> bool:
    if llm_client is None:
        return False
    if summarization is None:
        return True
    return bool(summarization.enabled)


def run_pipeline(
    sources: Sequence[NewsSource],
    *,
    window_hours: int = 72,
    max_summary_chars: int = 400,
    store: NewsStore | None = None,
    fetch_text: FetchTextFn | None = None,
    client: httpx.Client | None = None,
    clock: ClockFn | None = None,
    llm_client: LLMClient | None = None,
    summarization: SummarizationConfig | None = None,
) -> PipelineRunResult:
    """Run fetch → dedupe → freshness → summarize → optional persist (plain Python).

    Without ``llm_client``, summarization is TextRank-only. With ``llm_client``,
    optional batch LLM summarization uses all-or-nothing validation and TextRank
    fallback per batch.

    Does **not** require LangGraph. For the optional StateGraph adapter, install
    ``news-pipeline[graph]`` and use ``news_pipeline.graph.invoke_pipeline``.
    """
    now_fn: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))
    started_at = now_fn()
    cfg = summarization or SummarizationConfig(enabled=False)
    use_llm = _llm_enabled(llm_client, summarization)

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
    fresh = filter_items_by_window(
        deduped,
        window_hours=window_hours,
        now=started_at,
    )

    if use_llm:
        assert llm_client is not None
        batch = summarize_items_llm(
            fresh,
            client=llm_client,
            batch_size=cfg.batch_size,
            timeout_seconds=cfg.timeout_seconds,
            max_summary_chars=max_summary_chars,
        )
    else:
        batch = summarize_items_textrank(
            fresh,
            max_summary_chars=max_summary_chars,
        )

    items = batch.items
    if store is not None:
        store.upsert_items(items)
    completed_at = now_fn()

    return PipelineRunResult(
        started_at=started_at,
        completed_at=completed_at,
        source_results=source_results,
        items=items,
        stats=batch.stats,
    )
