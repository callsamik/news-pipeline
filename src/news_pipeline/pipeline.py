"""Orchestrator entrypoint — delegates to LangGraph pipeline workflow."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from news_pipeline.fetch import ClockFn, FetchTextFn
from news_pipeline.graph import invoke_pipeline
from news_pipeline.llm.client_protocol import LLMClient
from news_pipeline.models import NewsSource, PipelineRunResult, SummarizationConfig
from news_pipeline.store import NewsStore


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
    """Run the full news trunk via LangGraph (fetch → … → summarize → persist).

    Without ``llm_client``, summarization is TextRank-only. With ``llm_client``,
    optional batch LLM summarization uses all-or-nothing validation and TextRank
    fallback per batch.
    """
    return invoke_pipeline(
        list(sources),
        window_hours=window_hours,
        max_summary_chars=max_summary_chars,
        store=store,
        fetch_text=fetch_text,
        http_client=client,
        clock=clock,
        llm_client=llm_client,
        summarization=summarization,
    )
