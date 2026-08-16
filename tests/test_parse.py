"""Parser registry and builtin kind parsers (Task 4)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from news_pipeline.errors import UnsupportedParserKind
from news_pipeline.identity import compute_item_id
from news_pipeline.models import NewsItem, NewsSource
from news_pipeline.parse import normalize_source_kind, parse_source_text

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

JSON_PAYLOAD = json.dumps(
    {
        "data": {
            "articles": [
                {
                    "headline": "RBI holds repo rate steady",
                    "link": "https://example.test/rbi-hold",
                    "ts": "2026-07-24T08:00:00Z",
                    "body": {"text": "Policy unchanged."},
                },
                {"headline": "", "link": ""},
            ]
        }
    }
)

HTML_PAGE = """
<html><body>
  <a href="/markets/nifty-climbs-on-policy-optimism">Nifty climbs on policy optimism</a>
  <a href="/markets/short">short</a>
  <a href="https://other.test/ads/banner-advert-placement-unit">Advert placement unit</a>
</body></html>
"""

FIXED_INGESTED = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)


def _rss_source(*, kind: str = "rss") -> NewsSource:
    return NewsSource(
        source_id="et_markets",
        url="https://example.test/rss",
        kind=kind,
    )


def test_normalize_source_kind_aliases():
    assert normalize_source_kind("ATOM") == "rss"
    assert normalize_source_kind("feed") == "rss"
    assert normalize_source_kind(" json_api ") == "json"
    assert normalize_source_kind("scrape") == "html"
    assert normalize_source_kind("xml") == "xml_titles"
    assert normalize_source_kind("") == "rss"


def test_parse_rss_items_with_golden_ids():
    source = _rss_source()
    items = parse_source_text(SAMPLE_RSS, source, ingested_at=FIXED_INGESTED)
    assert len(items) == 2
    first = items[0]
    assert first.title == "TCS wins large capacity expansion deal"
    assert first.url == "https://example.test/tcs-1"
    assert first.source_id == "et_markets"
    assert first.raw_text == "TCS announced growth and profit beat."
    assert first.published_at is not None
    assert first.published_at.year == 2026
    expected_id = compute_item_id(
        "et_markets",
        "https://example.test/tcs-1",
        "TCS wins large capacity expansion deal",
    )
    assert first.id == expected_id == "news_e585d75c78d193be"
    assert first.ingested_at == FIXED_INGESTED


def test_atom_alias_parses_as_rss():
    source = _rss_source(kind="atom")
    items = parse_source_text(SAMPLE_RSS, source, ingested_at=FIXED_INGESTED)
    assert len(items) == 2
    assert items[0].title.startswith("TCS")


def test_unknown_kind_raises_unsupported_parser_kind():
    source = NewsSource(source_id="x", url="https://example.test/", kind="telegram")
    with pytest.raises(UnsupportedParserKind) as exc:
        parse_source_text("<rss/>", source)
    assert "telegram" in str(exc.value).lower() or "x" in str(exc.value)


def test_parse_json_api_minimal():
    source = NewsSource(
        source_id="wire_feed",
        url="https://example.test/api",
        kind="json",
        options={
            "item_path": "data.articles",
            "fields": {
                "title": "headline",
                "url": "link",
                "published_at": "ts",
                "summary": "body.text",
            },
        },
    )
    items = parse_source_text(JSON_PAYLOAD, source, ingested_at=FIXED_INGESTED)
    assert len(items) == 1
    item = items[0]
    assert item.title == "RBI holds repo rate steady"
    assert item.url == "https://example.test/rbi-hold"
    assert item.raw_text == "Policy unchanged."
    assert item.published_at is not None
    assert item.published_at.year == 2026
    assert item.id == compute_item_id(
        "wire_feed",
        "https://example.test/rbi-hold",
        "RBI holds repo rate steady",
    )


def test_parse_html_links_minimal():
    source = NewsSource(
        source_id="wire_feed",
        url="https://example.test/markets",
        kind="html",
        options={"min_title_chars": 10, "link_contains": "/markets/"},
    )
    items = parse_source_text(HTML_PAGE, source, ingested_at=FIXED_INGESTED)
    assert [i.title for i in items] == ["Nifty climbs on policy optimism"]
    assert items[0].url == "https://example.test/markets/nifty-climbs-on-policy-optimism"
    assert items[0].id == compute_item_id(
        "wire_feed",
        "https://example.test/markets/nifty-climbs-on-policy-optimism",
        "Nifty climbs on policy optimism",
    )


def test_parse_xml_titles_with_tag_override():
    source = NewsSource(
        source_id="gov_feed",
        url="https://example.test/gov.xml",
        kind="xml_titles",
        options={"title_tags": ["newssub"]},
    )
    xml = (
        "<root><row><NEWSSUB>Board meeting intimation</NEWSSUB>"
        "<TITLE>ignored</TITLE></row></root>"
    )
    items = parse_source_text(xml, source, ingested_at=FIXED_INGESTED)
    assert [i.title for i in items] == ["Board meeting intimation"]


def test_news_items_lack_ain_only_fields():
    source = _rss_source()
    items = parse_source_text(SAMPLE_RSS, source, ingested_at=FIXED_INGESTED)
    assert items
    for item in items:
        assert isinstance(item, NewsItem)
        assert not hasattr(item, "credibility")
        field_names = {f.name for f in item.__dataclass_fields__.values()}
        assert "credibility" not in field_names
        assert "tier" not in field_names
