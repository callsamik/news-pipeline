from datetime import datetime, timezone

from news_pipeline.identity import compute_item_id, dedupe_by_id
from news_pipeline.models import NewsItem


def test_identity_golden_vectors():
    assert compute_item_id("src1", "https://example.com/a", "Hello World") == "news_db73226f9bbc729d"
    assert compute_item_id("bse", "https://www.bseindia.com/x", "") == "news_d8791a2b13347bcd"
    assert compute_item_id("s", "  https://x.com/y  ", "  T  ") == "news_f806446bfed958fe"


def test_dedupe_first_wins():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    a = NewsItem(id="news_x", source_id="s", url="u", title="t1", published_at=None, ingested_at=now, raw_text="a")
    b = NewsItem(id="news_x", source_id="s", url="u", title="t2", published_at=None, ingested_at=now, raw_text="b")
    out = dedupe_by_id([a, b])
    assert len(out) == 1 and out[0].raw_text == "a"
