"""Batch LLM summarization with all-or-nothing positional attach + TextRank fallback."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Sequence

from news_pipeline.llm.client_protocol import LLMClient
from news_pipeline.llm.prompts import SYSTEM_PROMPT, build_feed_payload, build_user_prompt
from news_pipeline.llm.validate import validate_summaries_payload
from news_pipeline.models import NewsItem, SummarizationStats
from news_pipeline.summarize import summarize_item


@dataclass(frozen=True)
class BatchSummarizeResult:
    items: list[NewsItem]
    stats: SummarizationStats


def _item_text(item: NewsItem) -> str:
    return item.raw_text or item.summary or ""


def _textrank_batch(
    items: Sequence[NewsItem],
    *,
    max_summary_chars: int,
) -> list[NewsItem]:
    return [summarize_item(item, max_chars=max_summary_chars) for item in items]


def _attach_summaries(
    items: Sequence[NewsItem],
    summaries: Sequence[str],
) -> list[NewsItem]:
    """Positional attach only — caller must have already validated cardinality."""
    return [replace(item, summary=summary) for item, summary in zip(items, summaries, strict=True)]


def _summarize_one_batch(
    client: LLMClient,
    batch: Sequence[NewsItem],
    *,
    timeout_seconds: float,
    max_summary_chars: int,
) -> tuple[list[NewsItem], bool, float]:
    """Return (items, llm_ok, latency_ms). On any failure, TextRank for whole batch."""
    feed = [
        {"title": item.title, "text": _item_text(item)}
        for item in batch
    ]
    payload = build_feed_payload(feed)
    user = build_user_prompt(payload)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    started = time.perf_counter()
    try:
        completion = client.complete(
            messages=messages,
            response_format="json",
            timeout_s=timeout_seconds,
        )
        raw = completion.text if completion is not None else None
        validated = validate_summaries_payload(raw, expected_count=len(batch))
        if not validated.ok:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return _textrank_batch(batch, max_summary_chars=max_summary_chars), False, elapsed_ms
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return _attach_summaries(batch, validated.summaries), True, elapsed_ms
    except Exception:  # noqa: BLE001 — timeout/connection/node failure → fallback
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return _textrank_batch(batch, max_summary_chars=max_summary_chars), False, elapsed_ms


def summarize_items_llm(
    items: Sequence[NewsItem],
    *,
    client: LLMClient,
    batch_size: int = 10,
    timeout_seconds: float = 240.0,
    max_summary_chars: int = 400,
) -> BatchSummarizeResult:
    """Summarize items in batches via LLM; reject whole batch on any validation failure."""
    if not items:
        return BatchSummarizeResult(
            items=[],
            stats=SummarizationStats(
                summarization_mode="llm",
                batches_total=0,
                llm_batches_ok=0,
                llm_batches_fallback=0,
                llm_latency_ms=0.0,
            ),
        )

    size = max(1, int(batch_size))
    out: list[NewsItem] = []
    ok_n = 0
    fallback_n = 0
    latency_total = 0.0

    for start in range(0, len(items), size):
        batch = list(items[start : start + size])
        summarized, llm_ok, latency_ms = _summarize_one_batch(
            client,
            batch,
            timeout_seconds=timeout_seconds,
            max_summary_chars=max_summary_chars,
        )
        out.extend(summarized)
        latency_total += latency_ms
        if llm_ok:
            ok_n += 1
        else:
            fallback_n += 1

    return BatchSummarizeResult(
        items=out,
        stats=SummarizationStats(
            summarization_mode="llm",
            batches_total=ok_n + fallback_n,
            llm_batches_ok=ok_n,
            llm_batches_fallback=fallback_n,
            llm_latency_ms=latency_total,
        ),
    )


def summarize_items_textrank(
    items: Sequence[NewsItem],
    *,
    max_summary_chars: int = 400,
) -> BatchSummarizeResult:
    """Sole deterministic path: TextRank for every item."""
    summarized = _textrank_batch(items, max_summary_chars=max_summary_chars)
    return BatchSummarizeResult(
        items=summarized,
        stats=SummarizationStats(
            summarization_mode="textrank",
            batches_total=1 if summarized else 0,
            llm_batches_ok=0,
            llm_batches_fallback=0,
            llm_latency_ms=0.0,
        ),
    )


# Back-compat alias; prefer summarize_items_textrank.
summarize_items_extractive = summarize_items_textrank
