"""Item freshness window filtering (no sweep SLA semantics)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

from .models import NewsItem


def ensure_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def filter_items_by_window(
    items: Sequence[NewsItem],
    *,
    window_hours: int,
    now: datetime | None = None,
) -> list[NewsItem]:
    """Keep items published (or ingested) within ``window_hours``."""
    clock = ensure_aware(now) or datetime.now(timezone.utc)
    cutoff = clock - timedelta(hours=max(1, int(window_hours)))
    kept: list[NewsItem] = []
    for item in items:
        stamp = ensure_aware(item.published_at) or ensure_aware(item.ingested_at)
        if stamp is None:
            kept.append(item)
            continue
        if stamp >= cutoff:
            kept.append(item)
    return kept
