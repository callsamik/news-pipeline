# news-pipeline

Generic news fetch → parse → identity dedupe → freshness window → summarize → optional persist.

**Default orchestration is plain Python.** LangGraph is an **optional** adapter (`news-pipeline[graph]`) with parity to the plain path. Domain modules remain authoritative for what each step means.

**Import:** `news_pipeline`  
**Status:** Public alpha (`0.1.0a1`) — **library freeze candidate** after FB-1

## Architecture

| Layer | Owns |
| :--- | :--- |
| **`run_pipeline` (plain Python)** | Required collector flow — when/where steps run |
| **`news_pipeline.graph` (optional)** | LangGraph StateGraph adapter — install `[graph]`; not a public extension point |
| **Domain modules** | Meaning/mechanics — fetch, identity, **TextRank** summarize, store |
| **`LLMClient` Protocol** → **multiprovider-llm** | Model access (optional) |
| **AIN / consumer** | Linking, impact, mentions, gates, digests — never imported here |

```text
Consumer / AIN
     │ configures multiprovider-llm
     │ calls run_pipeline(llm_client=...)   ← plain Python (required)
     │ optional: news_pipeline.graph        ← only with [graph]
     ▼
fetch → dedupe → freshness → summarize? → persist?
                              /        \
                       TextRank      LLM batch
                       (sole det.)       ↓
                                      validate
                                       /    \
                                    attach  TextRank fallback
```

**One sentence:** `news-pipeline` uses an LLM for primary summarization when configured; otherwise, or whenever the LLM batch fails validation, it uses its single deterministic TextRank summarizer as the safety fallback. LangGraph is optional orchestration for AIN/subgraph use; AIN remains entirely outside the library.

**Hard firewall:** pipeline state and `NewsItem` never carry ticker/asset/catalog/impact/mention semantics.

## Owns / Consumes / Produces / Extends

| | |
| :--- | :--- |
| **Owns** | Retrieve → parse → normalize → identity/hash dedupe → freshness → TextRank summarize → optional LLM batch summarize (positional, all-or-nothing) → optional persist |
| **Consumes** | Caller-selected `NewsSource` list; HTTP (or injected `fetch_text`); optional `NewsStore`; injected `clock`; optional Protocol-compatible `llm_client` |
| **Produces** | Generic `NewsItem`, `SourceFetchResult`, `PipelineRunResult` (+ summarization stats) — never investment features |
| **Extends** | Parser kinds; fetch injection; store backend; LLM client injection — no AIN types |

## Summarization

| Role | Implementation |
| :--- | :--- |
| **Primary** (when `llm_client` configured) | LLM batch via `multiprovider-llm` |
| **Sole deterministic** (no LLM, or LLM batch rejected) | **TextRank** (stdlib; no second algorithm) |

Without `llm_client`, behavior is TextRank-only.

```python
from news_pipeline import SummarizationConfig, run_pipeline

result = run_pipeline(
    sources,
    llm_client=multiprovider_client,  # implements LLMClient Protocol
    summarization=SummarizationConfig(
        enabled=True,
        batch_size=10,
        timeout_seconds=240,  # one LLM batch invocation only; no retries in v1
    ),
)
```

- LLM sees ordered `{title, text}` only — never `news_id`, tickers, or catalog.
- Accept only `{"summaries":[...]}` with exact length and non-empty strings.
- On any failure: reject the **entire** batch → TextRank for every item in that batch.
- Never fuzzy-match or recover identity from LLM output.
- Stats: `summarization_mode` is `"llm"` or `"textrank"`; fallback counts use `llm_batches_fallback`.

**TextRank degenerate handling (same implementation, not a second summarizer):** empty body → `title[:cap]`; no graph edges → rank by sentence length then index.

Production client (separate package; implements `LLMClient`):

```bash
pip install "git+https://github.com/callsamik/multiprovider-llm.git"
```

Provider/model selection (e.g. Ollama + DeepSeek-R1:14B) is **caller configuration** via multiprovider-llm — not a library default.

## Identity v1

Deterministic item IDs (AIN cutover parity):

```text
url_eff   = strip(url) or strip(source.url)
title_eff = strip(title) or url_eff
id        = "news_" + SHA1(utf8(source_id + "|" + url_eff + "|" + title_eff)).hexdigest()[:16]
```

Golden vectors are tested in `tests/test_identity.py`. Batch dedupe within a run: **first occurrence wins**.

## Optional persistence

Pass a `NewsStore` (e.g. `SqliteNewsStore(path)`) to `run_pipeline` to upsert summarized items. The orchestrator decides *whether* to persist; `store.py` owns *how*.

## Standalone example (no AIN)

```bash
pip install -e ".[dev]"
python examples/minimal_pipeline.py
```

## Requirements

- Python `>=3.11,<4`
- `httpx>=0.27,<1`
- `feedparser>=6.0,<7`
- Optional: `langgraph>=1.0,<2` via `pip install "news-pipeline[graph]"`

## Install

```bash
pip install -e ".[dev]"          # includes langgraph for parity tests
pip install -e ".[graph]"        # optional StateGraph adapter only
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
    SummarizationConfig,
    SummarizationStats,
    run_pipeline,
    compute_item_id,
    dedupe_by_id,
    filter_items_by_window,
    summarize_item,
    summarize_text,
    textrank_summarize,
    fetch_source,
    SqliteNewsStore,
)
# Optional (requires news-pipeline[graph]):
# from news_pipeline.graph import invoke_pipeline, build_pipeline_graph
```

## Evidence gates

```bash
# TextRank golden + benchmark (no LLM):
.venv/bin/python examples/ov_np_deterministic_fb_1.py

# Real multiprovider-llm seam (Ollama; prefer when AIN worker idle):
.venv/bin/python examples/ov_np_llm_api_1.py --dry-run
.venv/bin/python examples/ov_np_llm_api_1.py
```

### Evidence freeze

| Gate | Status |
| :--- | :--- |
| `OV-NP-LLM-SUMMARIZE-3` | **ACCEPTED** — DeepSeek-R1:14B satisfies the frozen batch/no-ID contract |
| `OV-NP-LLM-API-1` | **PASS** — real multiprovider-llm → `run_pipeline` (plain; graph optional) |
| `OV-NP-DETERMINISTIC-FB-1` | **PASS** — TextRank sole deterministic / LLM fallback (see artifact) |

```text
LLM                         PRIMARY
TextRank                    SOLE DETERMINISTIC SUMMARIZER / FALLBACK
plain run_pipeline          REQUIRED ORCHESTRATION
LangGraph                   OPTIONAL ADAPTER ([graph] extra)
multiprovider-llm           MODEL/PROVIDER ACCESS ONLY
AIN                         INVESTMENT INTELLIGENCE ONLY (unchanged; cutover separate)
────────────────────────────────────
news-pipeline               FROZEN (library boundary)
```

Artifacts: `examples/artifacts/np_llm_api_1_latest.json`, `examples/artifacts/np_deterministic_fb_1_latest.json`.

## License

MIT (see `LICENSE`).
