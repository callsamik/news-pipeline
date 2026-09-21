"""LLM summarization path — Protocol mock; no live provider."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from news_pipeline.identity import compute_item_id
from news_pipeline.llm.client_protocol import LLMClient
from news_pipeline.llm.summarize_batch import summarize_items_llm
from news_pipeline.models import NewsItem, NewsSource, SummarizationConfig
from news_pipeline.pipeline import run_pipeline
from news_pipeline.summarize import summarize_item

NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)

BODY_A = (
    "Alpha Corp reported quarterly revenue growth above analyst estimates. "
    "The company raised guidance for the next fiscal year."
)
BODY_B = (
    "Beta Industries announced a strategic partnership with a logistics firm. "
    "Executives said the deal expands distribution capacity."
)


@dataclass
class FakeCompletion:
    text: str


class FakeLLM:
    def __init__(self, responses: list[str | Exception]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("unexpected complete() call")
        next_resp = self._responses.pop(0)
        if isinstance(next_resp, Exception):
            raise next_resp
        return FakeCompletion(text=next_resp)


def _item(news_id: str, title: str, body: str) -> NewsItem:
    return NewsItem(
        id=news_id,
        source_id="s",
        url=f"https://example.test/{news_id}",
        title=title,
        published_at=NOW,
        ingested_at=NOW,
        raw_text=body,
        summary="",
    )


def test_summarize_items_llm_positional_attach_on_success():
    items = [
        _item("news_a", "Alpha beat estimates", BODY_A),
        _item("news_b", "Beta partnership", BODY_B),
    ]
    summaries = [
        "Alpha Corp beat estimates and raised guidance.",
        "Beta Industries expanded distribution via a new partnership.",
    ]
    client = FakeLLM([json.dumps({"summaries": summaries})])

    result = summarize_items_llm(items, client=client, batch_size=10, timeout_seconds=30)

    assert result.stats.summarization_mode == "llm"
    assert result.stats.batches_total == 1
    assert result.stats.llm_batches_ok == 1
    assert result.stats.llm_batches_fallback == 0
    assert result.items[0].summary == summaries[0]
    assert result.items[1].summary == summaries[1]
    assert result.items[0].id == "news_a"
    assert result.items[1].id == "news_b"
    assert result.items[0].title == items[0].title
    assert result.items[0].url == items[0].url
    assert client.calls[0]["response_format"] == "json"
    assert client.calls[0]["timeout_s"] == 30
    user = client.calls[0]["messages"][1]["content"]
    # Payload must not include identity fields (instruction text may mention "news_id").
    assert '"news_id"' not in user
    assert "news_a" not in user
    assert "feed_items" in user


def test_summarize_items_llm_length_mismatch_falls_back_entire_batch():
    items = [
        _item("news_a", "Alpha beat estimates", BODY_A),
        _item("news_b", "Beta partnership", BODY_B),
    ]
    client = FakeLLM([json.dumps({"summaries": ["only one valid-looking summary text"]})])

    result = summarize_items_llm(items, client=client, batch_size=10)

    assert result.stats.batches_total == 1
    assert result.stats.llm_batches_ok == 0
    assert result.stats.llm_batches_fallback == 1
    expected = [summarize_item(i, max_chars=400) for i in items]
    assert [x.summary for x in result.items] == [x.summary for x in expected]
    assert result.items[0].id == "news_a"
    assert result.items[1].id == "news_b"


def test_summarize_items_llm_timeout_falls_back():
    items = [_item("news_a", "Alpha beat estimates", BODY_A)]
    client = FakeLLM([TimeoutError("timed out")])

    result = summarize_items_llm(items, client=client, timeout_seconds=1)

    assert result.stats.llm_batches_fallback == 1
    assert result.items[0].summary == summarize_item(items[0]).summary


def test_summarize_items_llm_malformed_falls_back():
    items = [_item("news_a", "Alpha beat estimates", BODY_A)]
    client = FakeLLM(["<<<not-json>>>"])

    result = summarize_items_llm(items, client=client)

    assert result.stats.llm_batches_fallback == 1
    assert result.items[0].id == "news_a"


def test_summarize_items_llm_never_partial_across_batch():
    """10→7 must reject all 10 (no partial positional attach)."""
    items = [
        _item(f"news_{i}", f"Title {i} event", f"Body text for article number {i}. " * 5)
        for i in range(10)
    ]
    seven = [f"Summary for item {i} with enough content." for i in range(7)]
    client = FakeLLM([json.dumps({"summaries": seven})])

    result = summarize_items_llm(items, client=client, batch_size=10)

    assert result.stats.llm_batches_ok == 0
    assert result.stats.llm_batches_fallback == 1
    for original, out in zip(items, result.items, strict=True):
        assert out.id == original.id
        assert out.summary == summarize_item(original).summary


FIXTURE_TWO = f"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Alpha beat estimates</title>
    <link>https://example.test/alpha</link>
    <description>{BODY_A}</description>
    <pubDate>Sat, 16 Aug 2026 08:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Beta partnership</title>
    <link>https://example.test/beta</link>
    <description>{BODY_B}</description>
    <pubDate>Sat, 16 Aug 2026 09:00:00 GMT</pubDate>
  </item>
</channel></rss>
"""


def test_run_pipeline_with_llm_client_success():
    source = NewsSource(
        source_id="markets",
        url="https://example.test/markets/rss",
        kind="rss",
    )
    id_a = compute_item_id("markets", "https://example.test/alpha", "Alpha beat estimates")
    id_b = compute_item_id("markets", "https://example.test/beta", "Beta partnership")
    summaries = [
        "Alpha Corp beat estimates and raised guidance for the year.",
        "Beta Industries expanded capacity through a logistics partnership.",
    ]
    client = FakeLLM([json.dumps({"summaries": summaries})])

    result = run_pipeline(
        [source],
        window_hours=72,
        fetch_text=lambda _url: FIXTURE_TWO,
        clock=lambda: NOW,
        llm_client=client,
        summarization=SummarizationConfig(enabled=True, batch_size=10, timeout_seconds=60),
    )

    assert result.stats.summarization_mode == "llm"
    assert result.stats.batches_total == 1
    assert result.stats.llm_batches_ok == 1
    assert len(result.items) == 2
    assert result.items[0].id == id_a
    assert result.items[1].id == id_b
    assert result.items[0].summary == summaries[0]
    assert result.items[1].summary == summaries[1]


def test_run_pipeline_llm_disabled_uses_textrank_even_with_client():
    source = NewsSource(
        source_id="markets",
        url="https://example.test/markets/rss",
        kind="rss",
    )
    client = FakeLLM([])  # must not be called

    result = run_pipeline(
        [source],
        fetch_text=lambda _url: FIXTURE_TWO,
        clock=lambda: NOW,
        llm_client=client,
        summarization=SummarizationConfig(enabled=False),
    )

    assert result.stats.summarization_mode == "textrank"
    assert client.calls == []
    assert all(item.summary for item in result.items)


def test_fake_llm_is_protocol_compatible():
    assert isinstance(FakeLLM(["{}"]), LLMClient)
