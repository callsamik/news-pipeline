"""HTTP fetch and fetch+parse composition."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Callable
from urllib.parse import urlencode, urlparse, urlunparse

import httpx

from news_pipeline.errors import MissingCredential, UnsupportedParserKind
from news_pipeline.models import NewsItem, NewsSource, RawFetch, SourceFetchResult
from news_pipeline.parse import parse_source_text

RETRYABLE_STATUSES = frozenset({403, 406, 408, 425, 429, 500, 502, 503, 504})
RETRY_BACKOFF_SECONDS = 1.0

FetchTextFn = Callable[[str], str]
ClockFn = Callable[[], datetime]


def resolve_request(source: NewsSource) -> tuple[str, dict[str, str]]:
    """Final URL + headers for a source, applying config query/key options.

    Secrets never live in source config: ``credential_env`` names an environment
    variable, injected as a query param or header per ``options``.
    """
    headers = {str(k): str(v) for k, v in (source.headers or {}).items()}
    params: dict[str, str] = {
        str(k): str(v) for k, v in (source.option("query") or {}).items()
    }

    if source.credential_env:
        key = (os.environ.get(source.credential_env) or "").strip()
        if not key:
            raise MissingCredential(
                f"source {source.source_id!r} needs {source.credential_env} in the "
                "environment — credentials never live in source config"
            )
        param = str(source.option("api_key_param") or "")
        header = str(source.option("api_key_header") or "")
        if param:
            params[param] = key
        elif header:
            headers[header] = key
        else:
            headers["Authorization"] = f"Bearer {key}"

    url = source.url
    if params:
        parts = urlparse(url)
        query = f"{parts.query}&{urlencode(params)}" if parts.query else urlencode(params)
        url = urlunparse(parts._replace(query=query))
    return url, headers


def _fetch_text_http(
    source: NewsSource,
    url: str,
    headers: dict[str, str],
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """HTTP GET one URL with configured headers, timeout, and retries."""
    attempts = max(1, int(source.retries) + 1)
    last_error: Exception | None = None

    for attempt in range(attempts):
        owns = client is None
        http = client or httpx.Client(
            timeout=float(source.timeout_seconds or 15.0),
            follow_redirects=True,
        )
        try:
            resp = http.get(url, headers=headers) if headers else http.get(url)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in RETRYABLE_STATUSES:
                raise
            last_error = exc
        except httpx.TransportError as exc:
            last_error = exc
        finally:
            if owns:
                http.close()
        if attempt < attempts - 1:
            sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    assert last_error is not None
    raise last_error


def fetch_raw(
    source: NewsSource,
    *,
    client: httpx.Client | None = None,
    fetch_text: FetchTextFn | None = None,
) -> RawFetch:
    """Fetch raw response body for a source. No parser or kind checks."""
    try:
        url, headers = resolve_request(source)
    except MissingCredential as exc:
        return RawFetch(
            source_id=source.source_id,
            ok=False,
            body=None,
            error=str(exc),
        )

    try:
        if fetch_text is not None:
            body = fetch_text(url)
        else:
            body = _fetch_text_http(source, url, headers, client=client)
        return RawFetch(
            source_id=source.source_id,
            ok=True,
            body=body,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 — transport failures become RawFetch
        return RawFetch(
            source_id=source.source_id,
            ok=False,
            body=None,
            error=str(exc),
        )


def fetch_source(
    source: NewsSource,
    *,
    client: httpx.Client | None = None,
    fetch_text: FetchTextFn | None = None,
    clock: ClockFn | None = None,
) -> tuple[SourceFetchResult, list[NewsItem]]:
    """Fetch then parse one source. Never raises — failures become SourceFetchResult."""
    raw = fetch_raw(source, client=client, fetch_text=fetch_text)
    if not raw.ok:
        return (
            SourceFetchResult(
                source_id=source.source_id,
                ok=False,
                error=raw.error,
            ),
            [],
        )

    ingested_at = clock() if clock is not None else None
    try:
        items = parse_source_text(raw.body or "", source, ingested_at=ingested_at)
        return (
            SourceFetchResult(
                source_id=source.source_id,
                ok=True,
                item_count=len(items),
            ),
            items,
        )
    except UnsupportedParserKind as exc:
        return (
            SourceFetchResult(
                source_id=source.source_id,
                ok=False,
                error=str(exc),
            ),
            [],
        )
    except Exception as exc:  # noqa: BLE001 — one bad source must not crash callers
        return (
            SourceFetchResult(
                source_id=source.source_id,
                ok=False,
                error=str(exc),
            ),
            [],
        )
