"""Transport-only source and generic news item models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class NewsSource:
    source_id: str
    url: str
    kind: str = "rss"
    headers: dict[str, str] | None = None
    timeout_seconds: float = 15.0
    retries: int = 2
    credential_env: str = ""
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class NewsItem:
    id: str
    source_id: str
    url: str
    title: str
    published_at: datetime | None
    ingested_at: datetime
    raw_text: str = ""
    summary: str = ""
    language: str = "en"
