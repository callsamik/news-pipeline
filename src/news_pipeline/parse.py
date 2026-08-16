"""Source-kind registry and builtin parsers for news ingest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

import feedparser
from xml.etree import ElementTree as ET

from news_pipeline.errors import UnsupportedParserKind
from news_pipeline.identity import compute_item_id
from news_pipeline.models import NewsItem, NewsSource

SourceParser = Callable[..., list[NewsItem]]

VALID_SOURCE_KINDS = frozenset({"rss", "xml_titles", "json", "html"})

SOURCE_KIND_ALIASES = {
    "atom": "rss",
    "feed": "rss",
    "xml": "xml_titles",
    "titles": "xml_titles",
    "json_api": "json",
    "rest": "json",
    "scrape": "html",
}

DEFAULT_TITLE_TAGS = ("title", "newssub", "headline", "subject")
DEFAULT_JSON_FIELDS = {
    "title": "title",
    "url": "url",
    "published_at": "published_at",
    "summary": "summary",
}


def normalize_source_kind(kind: str) -> str:
    """Lowercase + alias map (``atom`` → ``rss``)."""
    key = (kind or "").strip().lower()
    if not key:
        return "rss"
    return SOURCE_KIND_ALIASES.get(key, key)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(raw: Any) -> datetime | None:
    """Best-effort RFC-822 / ISO-8601 parse; ``None`` when unusable."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return parsedate_to_datetime(text).astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text.replace("Z", "+0000"), fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _make_item(
    source: NewsSource,
    *,
    title: str,
    url: str,
    published_at: datetime | None,
    ingested_at: datetime,
    raw_text: str = "",
) -> NewsItem:
    link = (url or source.url).strip()
    label = (title or link).strip()
    return NewsItem(
        id=compute_item_id(source.source_id, link, label),
        source_id=source.source_id,
        url=link,
        title=label,
        published_at=published_at,
        ingested_at=ingested_at,
        raw_text=raw_text,
    )


def _title_tags(source: NewsSource) -> tuple[str, ...]:
    raw = source.option("title_tags") or DEFAULT_TITLE_TAGS
    if isinstance(raw, str):
        raw = [raw]
    return tuple(str(t).strip().lower() for t in raw if str(t).strip())


def parse_xml_titles(
    text: str,
    source: NewsSource,
    *,
    ingested_at: datetime | None = None,
) -> list[NewsItem]:
    """Pull headline-ish text nodes out of non-RSS XML."""
    stamp = ingested_at or _utc_now()
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    wanted = _title_tags(source)
    items: list[NewsItem] = []
    for el in root.iter():
        tag = el.tag.lower().split("}")[-1]
        if tag in wanted and (el.text or "").strip():
            items.append(
                _make_item(
                    source,
                    title=el.text.strip(),
                    url=source.url,
                    published_at=None,
                    ingested_at=stamp,
                )
            )
    return items


def parse_rss(
    text: str,
    source: NewsSource,
    *,
    ingested_at: datetime | None = None,
) -> list[NewsItem]:
    """Parse RSS/Atom via feedparser, falling back to generic XML titles."""
    stamp = ingested_at or _utc_now()
    parsed = feedparser.parse(text)
    items: list[NewsItem] = []
    for entry in getattr(parsed, "entries", None) or ():
        title = str(entry.get("title") or "").strip()
        link = str(entry.get("link") or entry.get("id") or "").strip()
        summary = str(entry.get("summary") or entry.get("description") or "")
        published = parse_datetime(entry.get("published") or entry.get("updated"))
        if not title and not link:
            continue
        items.append(
            _make_item(
                source,
                title=title,
                url=link,
                published_at=published,
                ingested_at=stamp,
                raw_text=summary,
            )
        )
    if items:
        return items
    return parse_xml_titles(text, source, ingested_at=stamp)


def _dig(data: Any, path: str) -> Any:
    """Walk a dotted path through nested dicts/lists (``data.items.0.title``)."""
    if not path:
        return data
    cursor = data
    for part in str(path).split("."):
        if cursor is None:
            return None
        if isinstance(cursor, Mapping):
            cursor = cursor.get(part)
        elif isinstance(cursor, Sequence) and not isinstance(cursor, (str, bytes)):
            try:
                cursor = cursor[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cursor


def parse_json_api(
    text: str,
    source: NewsSource,
    *,
    ingested_at: datetime | None = None,
) -> list[NewsItem]:
    """Parse a JSON feed using ``options.item_path`` + ``options.fields``."""
    stamp = ingested_at or _utc_now()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    records = _dig(data, str(source.option("item_path") or ""))
    if isinstance(records, Mapping):
        records = [records]
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []

    fields = {**DEFAULT_JSON_FIELDS, **(source.option("fields") or {})}
    items: list[NewsItem] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        title = _dig(record, fields["title"])
        link = _dig(record, fields["url"])
        summary = _dig(record, fields["summary"])
        published = parse_datetime(_dig(record, fields["published_at"]))
        if not title and not link:
            continue
        items.append(
            _make_item(
                source,
                title=str(title or ""),
                url=str(link or ""),
                published_at=published,
                ingested_at=stamp,
                raw_text=str(summary or ""),
            )
        )
    return items


class _AnchorCollector(HTMLParser):
    """Collect ``(href, text)`` pairs from anchors."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[tuple[str, str]] = []
        self._href: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        self._flush()
        self._href = dict(attrs).get("href") or ""
        self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a":
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._buffer.append(data)

    def _flush(self) -> None:
        if self._href is None:
            return
        text = " ".join("".join(self._buffer).split())
        self.anchors.append((self._href, text))
        self._href = None
        self._buffer = []

    def close(self) -> None:
        super().close()
        self._flush()


def parse_html_links(
    text: str,
    source: NewsSource,
    *,
    ingested_at: datetime | None = None,
) -> list[NewsItem]:
    """Scrape headline anchors from an HTML index page (stdlib parser only)."""
    stamp = ingested_at or _utc_now()
    collector = _AnchorCollector()
    try:
        collector.feed(text)
        collector.close()
    except Exception:
        return []

    min_chars = int(source.option("min_title_chars") or 25)
    needle = str(source.option("link_contains") or "").lower()
    base = urlparse(source.url)
    items: list[NewsItem] = []
    for href, label in collector.anchors:
        if len(label) < min_chars:
            continue
        if needle and needle not in href.lower():
            continue
        link = href.strip()
        if link.startswith("//"):
            link = f"{base.scheme}:{link}"
        elif link.startswith("/"):
            link = f"{base.scheme}://{base.netloc}{link}"
        elif not link.lower().startswith(("http://", "https://")):
            continue
        items.append(
            _make_item(
                source,
                title=label,
                url=link,
                published_at=None,
                ingested_at=stamp,
            )
        )
    return items


PARSERS: dict[str, SourceParser] = {
    "rss": parse_rss,
    "xml_titles": parse_xml_titles,
    "json": parse_json_api,
    "html": parse_html_links,
}


def parse_source_text(
    text: str,
    source: NewsSource,
    *,
    ingested_at: datetime | None = None,
) -> list[NewsItem]:
    """Dispatch to the parser registered for ``source.kind``."""
    kind = normalize_source_kind(source.kind)
    parser = PARSERS.get(kind)
    if parser is None:
        raise UnsupportedParserKind(
            f"source {source.source_id!r} has unknown kind {source.kind!r}; "
            f"valid kinds: {sorted(VALID_SOURCE_KINDS)}"
        )
    return parser(text, source, ingested_at=ingested_at)
