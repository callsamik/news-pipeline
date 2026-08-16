# news-pipeline

Generic news fetch/parse/dedupe/freshness/summarize pipeline.

**Import:** `news_pipeline`  
**Status:** Public alpha (`0.1.0a1`). Scaffold only — no models, identity, or parsers yet.

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

## License

MIT (see `LICENSE`).
