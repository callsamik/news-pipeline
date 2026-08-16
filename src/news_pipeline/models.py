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

    def option(self, name: str, default: Any = None) -> Any:
        """Return a parser/transport option from ``options``, or ``default``."""
        if not self.options:
            return default
        return self.options.get(name, default)


@dataclass(frozen=True)
class RawFetch:
    source_id: str
    ok: bool
    body: str | None
    error: str | None


@dataclass(frozen=True)
class SourceFetchResult:
    source_id: str
    ok: bool
    item_count: int = 0
    error: str | None = None


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
