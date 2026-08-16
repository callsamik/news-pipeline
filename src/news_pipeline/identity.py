"""Identity v1: deterministic item IDs and batch deduplication."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from news_pipeline.models import NewsItem


def compute_item_id(source_id: str, url: str, title: str, *, source_url: str = "") -> str:
    url_eff = (url or "").strip() or (source_url or "").strip()
    title_eff = (title or "").strip() or url_eff
    material = f"{source_id}|{url_eff}|{title_eff}".encode("utf-8")
    return "news_" + hashlib.sha1(material).hexdigest()[:16]


def dedupe_by_id(items: Sequence[NewsItem]) -> list[NewsItem]:
    seen: set[str] = set()
    out: list[NewsItem] = []
    for item in items:
        if item.id in seen:
            continue
        seen.add(item.id)
        out.append(item)
    return out
