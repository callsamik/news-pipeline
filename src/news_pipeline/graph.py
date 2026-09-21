"""LangGraph orchestration for the complete news-pipeline workflow.

LangGraph owns flow (when/where). Domain modules own meaning/mechanics.
PipelineState must never contain AIN investment concepts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Literal, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

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
    SummarizationStats,
)
from news_pipeline.store import NewsStore


class PipelineState(TypedDict, total=False):
    """Workflow state for one pipeline run.

    Allowed: sources, results, items, summarization stats, configuration.
    Forbidden: ticker, asset, catalog, impact, mention, opportunity, recommendation.
    """

    sources: list[NewsSource]
    window_hours: int
    max_summary_chars: int
    started_at: datetime
    completed_at: datetime
    source_results: list[SourceFetchResult]
    collected_items: list[NewsItem]
    deduped_items: list[NewsItem]
    fresh_items: list[NewsItem]
    items: list[NewsItem]
    summarization_mode: Literal["textrank", "llm"]
    batches_total: int
    llm_batches_ok: int
    llm_batches_fallback: int
    llm_latency_ms: float


def _llm_enabled(
    llm_client: LLMClient | None,
    summarization: SummarizationConfig | None,
) -> bool:
    if llm_client is None:
        return False
    if summarization is None:
        return True
    return bool(summarization.enabled)


def build_pipeline_graph(
    *,
    fetch_text: FetchTextFn | None = None,
    http_client: httpx.Client | None = None,
    clock: ClockFn | None = None,
    store: NewsStore | None = None,
    llm_client: LLMClient | None = None,
    summarization: SummarizationConfig | None = None,
) -> Any:
    """Compile the news-pipeline StateGraph with thin nodes over domain modules."""

    now_fn: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))
    cfg = summarization or SummarizationConfig(enabled=False)
    use_llm = _llm_enabled(llm_client, summarization)

    def fetch_sources(state: PipelineState) -> dict[str, Any]:
        sources = list(state["sources"])
        source_results: list[SourceFetchResult] = []
        collected: list[NewsItem] = []
        for source in sources:
            result, items = fetch_source(
                source,
                client=http_client,
                fetch_text=fetch_text,
                clock=now_fn,
            )
            source_results.append(result)
            collected.extend(items)
        return {
            "source_results": source_results,
            "collected_items": collected,
        }

    def deduplicate(state: PipelineState) -> dict[str, Any]:
        return {"deduped_items": dedupe_by_id(state.get("collected_items") or [])}

    def apply_freshness(state: PipelineState) -> dict[str, Any]:
        fresh = filter_items_by_window(
            state.get("deduped_items") or [],
            window_hours=int(state.get("window_hours") or 72),
            now=state["started_at"],
        )
        return {"fresh_items": fresh}

    def textrank_summarize(state: PipelineState) -> dict[str, Any]:
        result = summarize_items_textrank(
            state.get("fresh_items") or [],
            max_summary_chars=int(state.get("max_summary_chars") or 400),
        )
        return {
            "items": result.items,
            "summarization_mode": result.stats.summarization_mode,
            "batches_total": result.stats.batches_total,
            "llm_batches_ok": result.stats.llm_batches_ok,
            "llm_batches_fallback": result.stats.llm_batches_fallback,
            "llm_latency_ms": result.stats.llm_latency_ms,
        }

    def llm_summarize(state: PipelineState) -> dict[str, Any]:
        assert llm_client is not None
        result = summarize_items_llm(
            state.get("fresh_items") or [],
            client=llm_client,
            batch_size=cfg.batch_size,
            timeout_seconds=cfg.timeout_seconds,
            max_summary_chars=int(state.get("max_summary_chars") or 400),
        )
        return {
            "items": result.items,
            "summarization_mode": result.stats.summarization_mode,
            "batches_total": result.stats.batches_total,
            "llm_batches_ok": result.stats.llm_batches_ok,
            "llm_batches_fallback": result.stats.llm_batches_fallback,
            "llm_latency_ms": result.stats.llm_latency_ms,
        }

    def persist(state: PipelineState) -> dict[str, Any]:
        # Graph decides whether; store.py owns serialization/mechanics.
        if store is not None:
            store.upsert_items(state.get("items") or [])
        return {"completed_at": now_fn()}

    def route_summarization(_state: PipelineState) -> str:
        return "llm_summarize" if use_llm else "textrank_summarize"

    graph = StateGraph(PipelineState)
    graph.add_node("fetch_sources", fetch_sources)
    graph.add_node("deduplicate", deduplicate)
    graph.add_node("apply_freshness", apply_freshness)
    graph.add_node("textrank_summarize", textrank_summarize)
    graph.add_node("llm_summarize", llm_summarize)
    graph.add_node("persist", persist)

    graph.add_edge(START, "fetch_sources")
    graph.add_edge("fetch_sources", "deduplicate")
    graph.add_edge("deduplicate", "apply_freshness")
    graph.add_conditional_edges(
        "apply_freshness",
        route_summarization,
        {
            "textrank_summarize": "textrank_summarize",
            "llm_summarize": "llm_summarize",
        },
    )
    graph.add_edge("textrank_summarize", "persist")
    graph.add_edge("llm_summarize", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


def invoke_pipeline(
    sources: list[NewsSource],
    *,
    window_hours: int = 72,
    max_summary_chars: int = 400,
    store: NewsStore | None = None,
    fetch_text: FetchTextFn | None = None,
    http_client: httpx.Client | None = None,
    clock: ClockFn | None = None,
    llm_client: LLMClient | None = None,
    summarization: SummarizationConfig | None = None,
) -> PipelineRunResult:
    """Execute the compiled pipeline graph and build PipelineRunResult."""
    now_fn: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))
    started_at = now_fn()

    app = build_pipeline_graph(
        fetch_text=fetch_text,
        http_client=http_client,
        clock=now_fn,
        store=store,
        llm_client=llm_client,
        summarization=summarization,
    )
    final: PipelineState = app.invoke(
        {
            "sources": list(sources),
            "window_hours": window_hours,
            "max_summary_chars": max_summary_chars,
            "started_at": started_at,
            "source_results": [],
            "collected_items": [],
            "deduped_items": [],
            "fresh_items": [],
            "items": [],
            "summarization_mode": "textrank",
            "batches_total": 0,
            "llm_batches_ok": 0,
            "llm_batches_fallback": 0,
            "llm_latency_ms": 0.0,
        }
    )

    mode = final.get("summarization_mode") or "textrank"
    stats = SummarizationStats(
        summarization_mode=mode,  # type: ignore[arg-type]
        batches_total=int(final.get("batches_total") or 0),
        llm_batches_ok=int(final.get("llm_batches_ok") or 0),
        llm_batches_fallback=int(final.get("llm_batches_fallback") or 0),
        llm_latency_ms=float(final.get("llm_latency_ms") or 0.0),
    )
    return PipelineRunResult(
        started_at=started_at,
        completed_at=final.get("completed_at") or now_fn(),
        source_results=list(final.get("source_results") or []),
        items=list(final.get("items") or []),
        stats=stats,
    )
