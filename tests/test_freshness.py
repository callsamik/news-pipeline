from datetime import datetime, timedelta, timezone

from news_pipeline.freshness import filter_items_by_window
from news_pipeline.models import NewsItem

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


def _item(
    *,
    item_id: str = "news_x",
    published_at: datetime | None = None,
    ingested_at: datetime | None = None,
) -> NewsItem:
    return NewsItem(
        id=item_id,
        source_id="s",
        url="https://example.com/x",
        title="Title",
        published_at=published_at,
        ingested_at=ingested_at or NOW,
    )


def test_unknown_age_kept_when_both_stamps_missing():
    unknown = _item(item_id="unknown", published_at=None, ingested_at=None)
    fresh = _item(item_id="fresh", published_at=NOW - timedelta(hours=1))
    stale = _item(item_id="stale", published_at=NOW - timedelta(hours=48))

    kept = filter_items_by_window([unknown, fresh, stale], window_hours=24, now=NOW)

    assert [i.id for i in kept] == ["unknown", "fresh"]


def test_window_excludes_old_published_at():
    stale = _item(item_id="stale", published_at=NOW - timedelta(hours=30))
    fresh = _item(item_id="fresh", published_at=NOW - timedelta(hours=2))

    kept = filter_items_by_window([stale, fresh], window_hours=24, now=NOW)

    assert [i.id for i in kept] == ["fresh"]


def test_published_at_precedence_over_ingested_at():
    # Recent publish, old ingest → kept (published_at wins).
    kept_case = _item(
        item_id="kept",
        published_at=NOW - timedelta(hours=2),
        ingested_at=NOW - timedelta(days=7),
    )
    # Old publish, recent ingest → excluded (published_at wins).
    dropped_case = _item(
        item_id="dropped",
        published_at=NOW - timedelta(hours=30),
        ingested_at=NOW - timedelta(hours=1),
    )

    kept = filter_items_by_window([kept_case, dropped_case], window_hours=24, now=NOW)

    assert [i.id for i in kept] == ["kept"]


def test_falls_back_to_ingested_at_when_published_at_missing():
    via_ingest = _item(
        item_id="via_ingest",
        published_at=None,
        ingested_at=NOW - timedelta(hours=3),
    )
    stale_ingest = _item(
        item_id="stale_ingest",
        published_at=None,
        ingested_at=NOW - timedelta(hours=30),
    )

    kept = filter_items_by_window([via_ingest, stale_ingest], window_hours=24, now=NOW)

    assert [i.id for i in kept] == ["via_ingest"]


def test_window_hours_minimum_is_one():
    # With window_hours=0, cutoff is now - 1 hour (not now).
    borderline = _item(item_id="borderline", published_at=NOW - timedelta(minutes=50))
    too_old = _item(item_id="too_old", published_at=NOW - timedelta(hours=2))

    kept = filter_items_by_window([borderline, too_old], window_hours=0, now=NOW)

    assert [i.id for i in kept] == ["borderline"]


def test_naive_datetimes_treated_as_utc():
    naive_recent = _item(
        item_id="naive",
        published_at=datetime(2026, 8, 16, 11, 30),  # noqa: DTZ001 — intentional naive
    )

    kept = filter_items_by_window([naive_recent], window_hours=24, now=NOW)

    assert [i.id for i in kept] == ["naive"]
