"""TextRank sole deterministic summarizer tests (OV-NP-DETERMINISTIC-FB-1)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from news_pipeline.models import NewsItem
from news_pipeline.summarize import (
    MAX_GRAPH_SENTENCES,
    summarize_item,
    summarize_text,
    textrank_summarize,
)

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
GOLDEN = Path(__file__).resolve().parent / "data" / "textrank_golden_v1.json"

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
    assert summarize_text(title, BODY, max_chars=200) == summarize_text(
        title, BODY, max_chars=200
    )


def test_summarize_text_strips_html_markup():
    body = "<p>Hello <b>world</b>. Second sentence here about markets rising sharply.</p>"
    summary = summarize_text("Hello world", body, max_chars=200)
    assert "<" not in summary
    assert summary


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


def test_empty_body_uses_title_degenerate_handling():
    assert summarize_text("Only Title", "", max_chars=400) == "Only Title"
    assert summarize_text("Only Title", "   ", max_chars=400) == "Only Title"


def test_textrank_empty_text():
    assert textrank_summarize("") == ""
    assert textrank_summarize("   ") == ""


def test_textrank_one_sentence():
    assert textrank_summarize("Only one sentence here.") == "Only one sentence here."


def test_textrank_repeated_sentences_are_filtered():
    body = (
        "Cooling costs can be recovered within four years. "
        "A study shows cooling costs can be recovered within four years. "
        "Factories in Bangladesh may recover cooling investment within four years. "
        "Separately, exporters discussed energy efficiency upgrades."
    )
    summary = summarize_text("Factory cooling", body, max_chars=400)
    # Near-duplicate first two sentences should not both appear.
    assert summary.count("Cooling costs can be recovered within four years.") == 1


def test_textrank_max_chars_drops_trailing_then_truncates():
    body = (
        "Alpha sentence one is reasonably long for testing bounds. "
        "Beta sentence two continues with more financial detail about banks. "
        "Gamma sentence three adds further commentary on market structure today."
    )
    summary = summarize_text("Bounds", body, max_chars=80)
    assert len(summary) <= 80


def test_pathological_sentence_ceiling_is_deterministic():
    sentences = [f"Sentence number {i} discusses markets and banks." for i in range(120)]
    body = " ".join(sentences)
    a = textrank_summarize(body, max_chars=400)
    b = textrank_summarize(body, max_chars=400)
    assert a == b
    assert a
    # Implementation only graphs the first MAX_GRAPH_SENTENCES.
    assert MAX_GRAPH_SENTENCES == 80


def test_golden_corpus_byte_identical():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["corpus_id"] == "np_textrank_golden_v1"
    for item in data["items"]:
        got = summarize_text(item["title"], item["body"], max_chars=data["max_chars"])
        assert got == item["expected_summary"], item["id"]


def test_golden_corpus_triple_run_identical():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for item in data["items"]:
        runs = [
            summarize_text(item["title"], item["body"], max_chars=400) for _ in range(3)
        ]
        assert runs[0] == runs[1] == runs[2]
