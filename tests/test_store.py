"""Optional NewsStore protocol + SQLite reference (Task 6)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from news_pipeline.models import NewsItem
from news_pipeline.store import SqliteNewsStore

FIXED_NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)


def _sample_item(
    *,
    item_id: str = "news_db73226f9bbc729d",
    source_id: str = "src1",
    url: str = "https://example.com/a",
    title: str = "Hello World",
    raw_text: str = "Full RSS body that should not be persisted verbatim.",
    summary: str = "Hello World summary.",
) -> NewsItem:
    return NewsItem(
        id=item_id,
        source_id=source_id,
        url=url,
        title=title,
        published_at=FIXED_NOW,
        ingested_at=FIXED_NOW,
        raw_text=raw_text,
        summary=summary,
        language="en",
    )


def test_upsert_get_round_trip_preserves_id(tmp_path):
    db_path = tmp_path / "nested" / "news.sqlite"
    store = SqliteNewsStore(db_path)
    item = _sample_item()
    store.upsert_items([item])

    loaded = store.get_item(item.id)
    assert loaded is not None
    assert loaded.id == item.id
    assert loaded.source_id == item.source_id
    assert loaded.url == item.url
    assert loaded.title == item.title
    assert loaded.summary == item.summary
    assert loaded.language == item.language
    assert loaded.published_at == item.published_at
    assert loaded.ingested_at == item.ingested_at


def test_stored_payload_omits_raw_text(tmp_path):
    db_path = tmp_path / "news.sqlite"
    store = SqliteNewsStore(db_path)
    item = _sample_item(raw_text="large body " * 100)
    store.upsert_items([item])

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT payload FROM news_items WHERE id=?", (item.id,)
    ).fetchone()
    conn.close()
    assert row is not None
    payload = json.loads(row[0])
    assert payload["raw_text"] == ""


def test_upsert_does_not_mutate_in_memory_item(tmp_path):
    store = SqliteNewsStore(tmp_path / "news.sqlite")
    item = _sample_item(raw_text="keep me unchanged")
    original_raw = item.raw_text

    store.upsert_items([item])

    assert item.raw_text == original_raw


def test_get_item_reconstructs_immutable_news_item(tmp_path):
    store = SqliteNewsStore(tmp_path / "news.sqlite")
    item = _sample_item()
    store.upsert_items([item])

    loaded = store.get_item(item.id)
    assert loaded is not None
    assert loaded.raw_text == ""
    with pytest.raises(Exception):
        loaded.raw_text = "mutate"  # type: ignore[misc]


def test_list_by_source(tmp_path):
    store = SqliteNewsStore(tmp_path / "news.sqlite")
    a = _sample_item(item_id="news_a", source_id="feed_a")
    b = _sample_item(item_id="news_b", source_id="feed_b", url="https://example.com/b", title="B")
    c = _sample_item(item_id="news_c", source_id="feed_a", url="https://example.com/c", title="C")
    store.upsert_items([a, b, c])

    feed_a = store.list_by_source("feed_a")
    assert {i.id for i in feed_a} == {"news_a", "news_c"}


def test_get_item_missing_returns_none(tmp_path):
    store = SqliteNewsStore(tmp_path / "news.sqlite")
    assert store.get_item("news_missing") is None


def test_upsert_same_id_updates_identity(tmp_path):
    store = SqliteNewsStore(tmp_path / "news.sqlite")
    item_id = "news_db73226f9bbc729d"
    first = _sample_item(item_id=item_id, summary="first")
    second = _sample_item(item_id=item_id, summary="second")
    store.upsert_items([first])
    store.upsert_items([second])

    loaded = store.get_item(item_id)
    assert loaded is not None
    assert loaded.id == item_id
    assert loaded.summary == "second"


def test_schema_items_table_only(tmp_path):
    db_path = tmp_path / "news.sqlite"
    SqliteNewsStore(db_path)

    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert tables == {"news_items"}
