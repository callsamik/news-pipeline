"""HTTP fetch and fetch+parse composition (Task 5)."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import respx

from news_pipeline.fetch import fetch_raw, fetch_source
from news_pipeline.models import NewsSource

SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Markets</title>
  <item>
    <title>TCS wins large capacity expansion deal</title>
    <link>https://example.test/tcs-1</link>
    <description>TCS announced growth and profit beat.</description>
    <pubDate>Thu, 24 Jul 2026 08:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Market rally continues</title>
    <link>https://example.test/mkt-1</link>
    <description>Indices higher.</description>
  </item>
</channel></rss>
"""

FIXED_INGESTED = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)


def _rss_source(*, kind: str = "rss") -> NewsSource:
    return NewsSource(
        source_id="et_markets",
        url="https://example.test/rss",
        kind=kind,
    )


def test_unknown_kind_fetch_ok_parse_fails():
    source = NewsSource(source_id="x", url="https://example.com/f", kind="not-a-kind")
    result, items = fetch_source(source, fetch_text=lambda url: "<rss></rss>")
    assert items == []
    assert result.ok is False
    assert "kind" in (result.error or "").lower() or "unsupported" in (
        result.error or ""
    ).lower()


def test_fetch_raw_unknown_kind_does_not_check_parser():
    source = NewsSource(source_id="x", url="https://example.com/f", kind="not-a-kind")
    raw = fetch_raw(source, fetch_text=lambda url: "<rss></rss>")
    assert raw.ok is True
    assert raw.body == "<rss></rss>"
    assert raw.error is None


def test_fetch_source_transport_failure_via_fetch_text():
    source = _rss_source()

    def boom(_url: str) -> str:
        raise ConnectionError("network down")

    result, items = fetch_source(source, fetch_text=boom)
    assert items == []
    assert result.ok is False
    assert "network down" in (result.error or "")


def test_fetch_source_rss_success_with_injected_body():
    source = _rss_source()
    result, items = fetch_source(
        source,
        fetch_text=lambda _url: SAMPLE_RSS,
        clock=lambda: FIXED_INGESTED,
    )
    assert result.ok is True
    assert result.item_count == 2
    assert len(items) == 2
    assert items[0].title == "TCS wins large capacity expansion deal"


def test_fetch_raw_missing_credential(monkeypatch):
    source = NewsSource(
        source_id="paid_api",
        url="https://example.com/api",
        kind="json",
        credential_env="NEWS_API_KEY",
    )
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    raw = fetch_raw(source)
    assert raw.ok is False
    assert raw.body is None
    assert "NEWS_API_KEY" in (raw.error or "")


@respx.mock
def test_fetch_raw_http_success():
    source = _rss_source()
    respx.get("https://example.test/rss").mock(return_value=httpx.Response(200, text=SAMPLE_RSS))
    raw = fetch_raw(source)
    assert raw.ok is True
    assert "TCS wins" in (raw.body or "")


@respx.mock
def test_fetch_raw_retries_retryable_status():
    source = NewsSource(
        source_id="bse",
        url="https://example.test/feed",
        kind="rss",
        retries=1,
    )
    route = respx.get("https://example.test/feed").mock(
        side_effect=[
            httpx.Response(403, text="forbidden"),
            httpx.Response(200, text=SAMPLE_RSS),
        ]
    )
    raw = fetch_raw(source)
    assert raw.ok is True
    assert route.call_count == 2


@respx.mock
def test_fetch_raw_non_retryable_status_fails():
    source = _rss_source()
    respx.get("https://example.test/rss").mock(return_value=httpx.Response(404, text="missing"))
    raw = fetch_raw(source)
    assert raw.ok is False
    assert raw.body is None
    assert raw.error is not None
