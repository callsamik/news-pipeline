"""news-pipeline — generic news fetch/parse/dedupe/freshness/summarize pipeline."""

from __future__ import annotations

from news_pipeline.errors import MissingCredential, NewsPipelineError, UnsupportedParserKind
from news_pipeline.fetch import fetch_raw, fetch_source
from news_pipeline.freshness import filter_items_by_window
from news_pipeline.identity import compute_item_id, dedupe_by_id
from news_pipeline.models import (
    NewsItem,
    NewsSource,
    PipelineRunResult,
    RawFetch,
    SourceFetchResult,
)
from news_pipeline.parse import normalize_source_kind, parse_source_text
from news_pipeline.pipeline import run_pipeline
from news_pipeline.store import NewsStore, SqliteNewsStore
from news_pipeline.summarize import summarize_item, summarize_text

__version__ = "0.1.0a1"

__all__ = [
    "__version__",
    "MissingCredential",
    "NewsItem",
    "NewsPipelineError",
    "NewsSource",
    "NewsStore",
    "PipelineRunResult",
    "RawFetch",
    "SourceFetchResult",
    "SqliteNewsStore",
    "UnsupportedParserKind",
    "compute_item_id",
    "dedupe_by_id",
    "fetch_raw",
    "fetch_source",
    "filter_items_by_window",
    "normalize_source_kind",
    "parse_source_text",
    "run_pipeline",
    "summarize_item",
    "summarize_text",
]
