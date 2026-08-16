# news-pipeline

Generic news fetch → parse → identity dedupe → freshness window → extractive summarize → optional persist.

**Import:** `news_pipeline`  
**Status:** Public alpha (`0.1.0a1`)

## Owns / Consumes / Produces / Extends

| | |
| :--- | :--- |
| **Owns** | Retrieve → parse → normalize → identity/hash dedupe → freshness window → generic extractive summarize → optional persist of **content items** |
| **Consumes** | Caller-selected transport-only `NewsSource` list; HTTP (or injected `fetch_text`); optional `NewsStore`; injected `clock` |
| **Produces** | Generic `NewsItem`, `SourceFetchResult`, `PipelineRunResult` — never investment features |
| **Extends** | Parser kinds; fetch injection; store backend — no AIN types |

The library does **not** own tier/credibility/tax policy, poll/sweep SLA, mentions, or impact analysis. Callers apply source policy before calling `run_pipeline`.

## Identity v1

Deterministic item IDs (AIN cutover parity):

```text
url_eff   = strip(url) or strip(source.url)
title_eff = strip(title) or url_eff
id        = "news_" + SHA1(utf8(source_id + "|" + url_eff + "|" + title_eff)).hexdigest()[:16]
```

Golden vectors are tested in `tests/test_identity.py`. Batch dedupe within a run: **first occurrence wins**.

## Optional persistence

Pass a `NewsStore` (e.g. `SqliteNewsStore(path)`) to `run_pipeline` to upsert summarized items. Omit `store` (default `None`) for in-memory-only runs. The library never discovers `DATA_DIR` or AIN storage paths — the caller supplies the path.

Stored payloads omit large `raw_text` (summary-centric); in-memory `NewsItem` values are never mutated.

## Standalone example (no AIN)

```bash
pip install -e ".[dev]"
python examples/minimal_pipeline.py
```

The example uses injected fixture RSS (offline), runs `run_pipeline`, prints item ids/titles/summaries, and optionally persists to a temp SQLite file. It imports only `news_pipeline`.

## Requirements

- Python `>=3.11,<4`
- `httpx>=0.27,<1`
- `feedparser>=6.0,<7`

## Install

```bash
pip install -e ".[dev]"
```

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
pytest -q
```

## Public API

```python
from news_pipeline import (
    NewsSource,
    NewsItem,
    run_pipeline,
    compute_item_id,
    dedupe_by_id,
    filter_items_by_window,
    summarize_item,
    fetch_source,
    SqliteNewsStore,
)
```

## License

MIT (see `LICENSE`).
