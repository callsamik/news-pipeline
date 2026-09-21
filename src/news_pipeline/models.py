"""Transport-only source and generic news item models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal


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


SummarizationMode = Literal["textrank", "llm"]


@dataclass(frozen=True)
class SummarizationConfig:
    """Pipeline-owned summarization knobs (not provider/model selection).

    ``timeout_seconds`` applies to **one LLM batch invocation only**.
    There are no retries in v1.
    """

    enabled: bool = True
    batch_size: int = 10
    timeout_seconds: float = 240.0


@dataclass(frozen=True)
class SummarizationStats:
    summarization_mode: SummarizationMode
    batches_total: int = 0
    llm_batches_ok: int = 0
    llm_batches_fallback: int = 0
    llm_latency_ms: float = 0.0


@dataclass(frozen=True)
class PipelineRunResult:
    started_at: datetime
    completed_at: datetime
    source_results: list[SourceFetchResult]
    items: list[NewsItem]
    stats: SummarizationStats = SummarizationStats(summarization_mode="textrank")
