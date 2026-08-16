"""Optional item persistence — items-only NewsStore protocol."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence

from .models import NewsItem


class NewsStore(Protocol):
    """Minimal items-only persistence contract."""

    def upsert_items(self, items: Sequence[NewsItem]) -> None: ...

    def get_item(self, item_id: str) -> NewsItem | None: ...

    def list_by_source(self, source_id: str) -> list[NewsItem]: ...


def _dt_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _dt_parse(raw: str | None) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(raw)


def _serialize_item_for_storage(item: NewsItem) -> dict:
    """Derived representation: summary-centric, raw_text omitted."""
    return {
        "id": item.id,
        "source_id": item.source_id,
        "url": item.url,
        "title": item.title,
        "published_at": _dt_iso(item.published_at),
        "ingested_at": item.ingested_at.isoformat(),
        "summary": item.summary or "",
        "raw_text": "",
        "language": item.language,
    }


def _deserialize_item(data: dict) -> NewsItem:
    return NewsItem(
        id=str(data["id"]),
        source_id=str(data["source_id"]),
        url=str(data["url"]),
        title=str(data["title"]),
        published_at=_dt_parse(data.get("published_at")),
        ingested_at=datetime.fromisoformat(str(data["ingested_at"])),
        raw_text="",
        summary=str(data.get("summary") or ""),
        language=str(data.get("language") or "en"),
    )


class SqliteNewsStore:
    """SQLite-backed NewsStore; caller supplies path, items table only."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_items (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                published_at TEXT,
                ingested_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def upsert_items(self, items: Sequence[NewsItem]) -> None:
        for item in items:
            payload = json.dumps(_serialize_item_for_storage(item))
            self._conn.execute(
                """
                INSERT INTO news_items(id, source_id, url, title, published_at, ingested_at, payload)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    source_id=excluded.source_id,
                    url=excluded.url,
                    title=excluded.title,
                    published_at=excluded.published_at,
                    ingested_at=excluded.ingested_at,
                    payload=excluded.payload
                """,
                (
                    item.id,
                    item.source_id,
                    item.url,
                    item.title,
                    _dt_iso(item.published_at),
                    item.ingested_at.isoformat(),
                    payload,
                ),
            )
        self._conn.commit()

    def get_item(self, item_id: str) -> NewsItem | None:
        row = self._conn.execute(
            "SELECT payload FROM news_items WHERE id=?", (item_id,)
        ).fetchone()
        if row is None:
            return None
        return _deserialize_item(json.loads(row[0]))

    def list_by_source(self, source_id: str) -> list[NewsItem]:
        rows = self._conn.execute(
            "SELECT payload FROM news_items WHERE source_id=? ORDER BY ingested_at DESC",
            (source_id,),
        ).fetchall()
        return [_deserialize_item(json.loads(row[0])) for row in rows]

    def close(self) -> None:
        self._conn.close()
