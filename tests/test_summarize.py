from datetime import datetime, timezone

from news_pipeline.models import NewsItem
from news_pipeline.summarize import summarize_item, summarize_text

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)

BODY = (
    "Reliance Industries reported strong quarterly earnings. "
    "The conglomerate beat analyst estimates on revenue growth. "
    "Reliance Jio added millions of subscribers in the period. "
    "Markets reacted positively to the results announcement."
)


def test_summarize_text_caps_length():
    summary = summarize_text("Reliance earnings beat", BODY, max_chars=120)

    assert len(summary) <= 120
    assert summary


def test_summarize_text_is_deterministic():
    title = "Reliance earnings beat"
    a = summarize_text(title, BODY, max_chars=200)
    b = summarize_text(title, BODY, max_chars=200)

    assert a == b


def test_summarize_text_strips_html_markup():
    body = "<p>Hello <b>world</b>. Second sentence here.</p>"
    summary = summarize_text("Hello world", body, max_chars=200)

    assert "<" not in summary
    assert "Hello world" in summary or "Second sentence" in summary


def test_summarize_item_sets_summary_without_clearing_raw_text():
    item = NewsItem(
        id="news_x",
        source_id="s",
        url="https://example.com/x",
        title="Reliance earnings beat",
        published_at=None,
        ingested_at=NOW,
        raw_text=BODY,
        summary="",
    )

    out = summarize_item(item, max_chars=200)

    assert out.summary
    assert len(out.summary) <= 200
    assert out.raw_text == BODY
    assert out is not item


def test_summarize_item_does_not_mutate_input():
    item = NewsItem(
        id="news_x",
        source_id="s",
        url="https://example.com/x",
        title="Reliance earnings beat",
        published_at=None,
        ingested_at=NOW,
        raw_text=BODY,
        summary="",
    )

    summarize_item(item, max_chars=200)

    assert item.summary == ""
    assert item.raw_text == BODY


def test_summarize_item_falls_back_to_existing_summary_when_raw_text_empty():
    item = NewsItem(
        id="news_x",
        source_id="s",
        url="https://example.com/x",
        title="Reliance earnings beat",
        published_at=None,
        ingested_at=NOW,
        raw_text="",
        summary=BODY,
    )

    out = summarize_item(item, max_chars=200)

    assert out.summary
    assert out.raw_text == ""
