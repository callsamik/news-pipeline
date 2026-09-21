#!/usr/bin/env python3
"""OV-NP-LLM-API-1 — Real multiprovider-llm integration validation gate.

Integration-validation only. Not a new summarization-quality experiment.
Does **not** modify AIN.

Seam under test:

  DeepSeek-R1:14B / Ollama
          ↓
  multiprovider-llm.Client  (+ Ollama native adapter for num_ctx)
          ↓
  LLMClient Protocol
          ↓
  news_pipeline.run_pipeline(llm_client=...)
          ↓
  LangGraph
          ↓
  validated NewsItems + PipelineRunResult.stats

Frozen SUMMARIZE-3 inputs (do not change to make the gate pass):
  corpus: examples/data/np_llm_summarize_1_corpus_v1.json
  model:  deepseek-r1:14b
  batch:  10
  num_ctx: 16384
  timeout: 240s / one batch
  no retries

Usage:

  .venv/bin/python examples/ov_np_llm_api_1.py --dry-run
  .venv/bin/python examples/ov_np_llm_api_1.py

Prefer running when the AIN production worker is idle so Ollama is not contended.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import httpx

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "data" / "np_llm_summarize_1_corpus_v1.json"
DEFAULT_OUT = HERE / "artifacts"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "deepseek-r1:14b"
DEFAULT_NUM_CTX = 16384
DEFAULT_BATCH = 10
DEFAULT_TIMEOUT = 240.0
EXPERIMENT = "OV-NP-LLM-API-1"

# Source fetch order preserves frozen corpus item order (moneycontrol → worldbank → reuters).
SOURCE_ORDER = ("moneycontrol_top", "worldbank_news", "reuters_top")


def _load_corpus(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = list(raw.get("items") or [])
    if len(items) != 10:
        raise SystemExit(f"expected 10 corpus items, got {len(items)} from {path}")
    return raw


def _group_items(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {sid: [] for sid in SOURCE_ORDER}
    for item in items:
        sid = str(item["source_id"])
        if sid not in grouped:
            raise SystemExit(f"unexpected source_id in corpus: {sid}")
        grouped[sid].append(item)
    return grouped


def _rss_for_items(items: list[dict[str, Any]]) -> str:
    """Build RSS whose parse order matches the given corpus slice."""
    parts = [
        '<?xml version="1.0"?>',
        '<rss version="2.0"><channel>',
        "<title>OV-NP-LLM-API-1 frozen corpus</title>",
    ]
    for item in items:
        published = item.get("published_at") or ""
        try:
            dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
            pub = format_datetime(dt.astimezone(timezone.utc))
        except ValueError:
            pub = "Mon, 21 Sep 2026 12:00:00 GMT"
        parts.extend(
            [
                "<item>",
                f"<title>{escape(str(item['title']))}</title>",
                f"<link>{escape(str(item['url']))}</link>",
                f"<description>{escape(str(item.get('text') or ''))}</description>",
                f"<pubDate>{pub}</pubDate>",
                "</item>",
            ]
        )
    parts.append("</channel></rss>")
    return "\n".join(parts)


def _make_fetch_text(grouped: dict[str, list[dict[str, Any]]]):
    payloads = {
        sid: _rss_for_items(grouped[sid]) for sid in SOURCE_ORDER if grouped[sid]
    }

    def fetch_text(url: str) -> str:
        for sid, body in payloads.items():
            if sid in url:
                return body
        raise KeyError(f"no fixture RSS for url={url}")

    return fetch_text


def _clock_for_corpus(items: list[dict[str, Any]]):
    """Fixed clock after newest published_at so freshness keeps all 10."""
    stamps: list[datetime] = []
    for item in items:
        raw = item.get("published_at")
        if not raw:
            continue
        try:
            stamps.append(
                datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(
                    timezone.utc
                )
            )
        except ValueError:
            continue
    latest = max(stamps) if stamps else datetime(2026, 9, 21, 18, 0, tzinfo=timezone.utc)
    # One day after latest published item.
    now = latest.replace(tzinfo=timezone.utc)
    from datetime import timedelta

    fixed = now + timedelta(days=1)

    def clock() -> datetime:
        return fixed

    return clock


def _build_ollama_client(*, base_url: str, model: str, num_ctx: int):
    """Real multiprovider-llm Client with Ollama native adapter (num_ctx support).

    Adapter lives in this example only — news-pipeline has no provider logic.
    Native /api/chat is required: OpenAI-compat /v1 often ignores num_ctx on Ollama.
    """
    from multiprovider_llm import Client, config_from_dict
    from multiprovider_llm.types import ProviderRequest, ProviderResponse, Usage

    class OllamaNativeAdapter:
        name = "ollama"

        def __init__(self, *, root: str, num_ctx: int) -> None:
            self._root = root.rstrip("/")
            if self._root.endswith("/v1"):
                self._root = self._root[: -len("/v1")]
            self._num_ctx = int(num_ctx)

        def complete(self, req: ProviderRequest) -> ProviderResponse:
            body: dict[str, Any] = {
                "model": req.model,
                "messages": [
                    {"role": m.role, "content": m.content} for m in req.messages
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": self._num_ctx,
                },
            }
            # Library requests response_format=json; Ollama honors format=json.
            if req.response_format == "json":
                body["format"] = "json"
            with httpx.Client(timeout=req.timeout_s) as client:
                response = client.post(f"{self._root}/api/chat", json=body)
            response.raise_for_status()
            data = response.json()
            message = data.get("message") or {}
            content = str(message.get("content") or "")
            # Some builds put reasoning separately; prefer content.
            if not content.strip():
                content = str(
                    message.get("thinking")
                    or message.get("reasoning")
                    or data.get("response")
                    or ""
                )
            prompt_eval = data.get("prompt_eval_count")
            eval_count = data.get("eval_count")
            total = None
            if isinstance(prompt_eval, int) and isinstance(eval_count, int):
                total = prompt_eval + eval_count
            return ProviderResponse(
                text=content,
                usage=Usage(
                    prompt_tokens=prompt_eval if isinstance(prompt_eval, int) else None,
                    completion_tokens=eval_count if isinstance(eval_count, int) else None,
                    total_tokens=total,
                ),
                status_code=response.status_code,
                headers=dict(response.headers),
                raw=data if req.include_raw else None,
            )

        async def acomplete(self, req: ProviderRequest) -> ProviderResponse:
            return self.complete(req)

    config = config_from_dict(
        {
            "providers": {
                "ollama": {
                    "enabled": True,
                    "freshness_ok": False,
                    "models": {
                        "simple": model,
                        "standard": model,
                        "complex": model,
                    },
                    "default_model": model,
                    "base_url": base_url.rstrip("/") + "/v1",
                    "api_key_env": "OLLAMA_API_KEY",
                    "rate_limits": {"max_inflight": 1},
                }
            },
            "provider_order": ["ollama"],
            "tier_routing": {
                "simple": ["ollama"],
                "standard": ["ollama"],
                "complex": ["ollama"],
            },
        }
    )
    return Client(
        config,
        adapters={
            "ollama": OllamaNativeAdapter(root=base_url, num_ctx=num_ctx),
        },
    )


def _assert_no_ain_imports() -> list[str]:
    import news_pipeline
    import pathlib

    root = pathlib.Path(news_pipeline.__file__).resolve().parent
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import ain" in text or "from ain" in text:
            offenders.append(str(path))
    return offenders


def _provider_logic_offenders() -> list[str]:
    """news-pipeline must not hard-code provider/model branches or clients."""
    import news_pipeline
    import pathlib
    import re

    root = pathlib.Path(news_pipeline.__file__).resolve().parent
    # Look for executable provider coupling, not incidental docstring words.
    needles = re.compile(
        r"(?:"
        r"from\s+ollama\b|import\s+ollama\b|"
        r"openai\.|anthropic\.|google\.generativeai|"
        r"if\s+.*\b(provider|model)\s*==|"
        r"base_url\s*=\s*[\"']http://127\.0\.0\.1:11434|"
        r"/api/chat|"
        r"OpenAI\(|Anthropic\("
        r")",
        re.IGNORECASE,
    )
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if needles.search(text):
            offenders.append(str(path.relative_to(root.parent)))
    return offenders


def _evaluate(
    *,
    corpus_items: list[dict[str, Any]],
    result: Any,
) -> dict[str, Any]:
    expected_ids = [str(it["news_id"]) for it in corpus_items]
    got_ids = [item.id for item in result.items]
    identity_unchanged = got_ids == expected_ids

    stats = result.stats
    contract_ok = (
        len(result.items) == 10
        and stats.summarization_mode == "llm"
        and stats.batches_total == 1
        and stats.llm_batches_ok == 1
        and stats.llm_batches_fallback == 0
        and stats.llm_latency_ms > 0
        and identity_unchanged
        and all(bool(item.summary.strip()) for item in result.items)
    )

    # Summaries must differ from pure extractive fallback for a successful LLM path
    # is not required — gate is API contract. But fallback flag must be 0.
    return {
        "input_count": 10,
        "output_count": len(result.items),
        "identity_unchanged": identity_unchanged,
        "expected_ids": expected_ids,
        "got_ids": got_ids,
        "all_summaries_nonempty": all(bool(i.summary.strip()) for i in result.items),
        "stats": {
            "summarization_mode": stats.summarization_mode,
            "batches_total": stats.batches_total,
            "llm_batches_ok": stats.llm_batches_ok,
            "llm_batches_fallback": stats.llm_batches_fallback,
            "llm_latency_ms": round(float(stats.llm_latency_ms), 3),
        },
        "contract_ok": contract_ok,
        "validation": "PASS" if contract_ok else "FAIL",
        "items": [
            {
                "news_id": item.id,
                "source_id": item.source_id,
                "title": item.title[:120],
                "summary_chars": len(item.summary),
                "summary": item.summary,
            }
            for item in result.items
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus = _load_corpus(args.corpus)
    items = list(corpus["items"])
    grouped = _group_items(items)
    ain_offenders = _assert_no_ain_imports()
    provider_offenders = _provider_logic_offenders()

    from news_pipeline import NewsSource, SummarizationConfig, run_pipeline
    from news_pipeline.identity import compute_item_id

    # Preflight: corpus IDs still match identity formula.
    id_mismatches = [
        {
            "news_id": it["news_id"],
            "computed": compute_item_id(it["source_id"], it["url"], it["title"]),
        }
        for it in items
        if compute_item_id(it["source_id"], it["url"], it["title"]) != it["news_id"]
    ]

    sources = [
        NewsSource(
            source_id=sid,
            url=f"https://example.test/{sid}/rss",
            kind="rss",
        )
        for sid in SOURCE_ORDER
        if grouped[sid]
    ]

    if args.dry_run:
        artifact = {
            "experiment": EXPERIMENT,
            "dry_run": True,
            "corpus_id": corpus.get("corpus_id"),
            "corpus_path": str(args.corpus),
            "model": args.model,
            "batch_size": args.batch_size,
            "num_ctx": args.num_ctx,
            "timeout_seconds": args.timeout,
            "input_count": 10,
            "ain_import_offenders": ain_offenders,
            "provider_logic_offenders": provider_offenders,
            "identity_formula_mismatches": id_mismatches,
            "would_call": {
                "seam": "multiprovider_llm.Client → run_pipeline(llm_client=...)",
                "graph": "LangGraph news-pipeline",
                "base_url": args.base_url,
                "sources": [s.source_id for s in sources],
            },
            "validation": (
                "DRY_RUN_READY"
                if not ain_offenders and not provider_offenders and not id_mismatches
                else "DRY_RUN_BLOCKED"
            ),
        }
        print(json.dumps(artifact, indent=2))
        if artifact["validation"] != "DRY_RUN_READY":
            raise SystemExit(2)
        return

    if ain_offenders or provider_offenders or id_mismatches:
        print(
            json.dumps(
                {
                    "experiment": EXPERIMENT,
                    "validation": "FAIL",
                    "ain_import_offenders": ain_offenders,
                    "provider_logic_offenders": provider_offenders,
                    "identity_formula_mismatches": id_mismatches,
                },
                indent=2,
            )
        )
        raise SystemExit(2)

    try:
        llm_client = _build_ollama_client(
            base_url=args.base_url,
            model=args.model,
            num_ctx=args.num_ctx,
        )
    except ImportError as exc:
        raise SystemExit(
            "multiprovider-llm is required for live OV-NP-LLM-API-1.\n"
            '  pip install "git+https://github.com/callsamik/multiprovider-llm.git"\n'
            f"ImportError: {exc}"
        ) from exc

    fetch_text = _make_fetch_text(grouped)
    clock = _clock_for_corpus(items)

    result = run_pipeline(
        sources,
        window_hours=168,
        max_summary_chars=400,
        fetch_text=fetch_text,
        clock=clock,
        llm_client=llm_client,
        summarization=SummarizationConfig(
            enabled=True,
            batch_size=args.batch_size,
            timeout_seconds=args.timeout,
        ),
    )

    evaluation = _evaluate(corpus_items=items, result=result)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = {
        "experiment": EXPERIMENT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_id": corpus.get("corpus_id"),
        "corpus_path": str(args.corpus),
        "model": args.model,
        "num_ctx": args.num_ctx,
        "batch_size": args.batch_size,
        "timeout_seconds": args.timeout,
        "base_url": args.base_url,
        "input_count": evaluation["input_count"],
        "output_count": evaluation["output_count"],
        "validation": evaluation["validation"],
        "identity_unchanged": evaluation["identity_unchanged"],
        "contract_ok": evaluation["contract_ok"],
        "stats": evaluation["stats"],
        "ain_import_offenders": ain_offenders,
        "provider_logic_offenders": provider_offenders,
        "pipeline": {
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat(),
            "source_results": [
                {
                    "source_id": sr.source_id,
                    "ok": sr.ok,
                    "item_count": sr.item_count,
                    "error": sr.error,
                }
                for sr in result.source_results
            ],
        },
        "items": evaluation["items"],
        "decision_rule": (
            "PASS iff 10→10 + llm stats ok + identity unchanged "
            "(API/contract gate; not a model-quality re-experiment)"
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamped = args.out_dir / f"np_llm_api_1_{stamp}.json"
    latest = args.out_dir / "np_llm_api_1_latest.json"
    payload = json.dumps(artifact, indent=2)
    stamped.write_text(payload + "\n", encoding="utf-8")
    latest.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    print(f"\nWrote {stamped}", file=sys.stderr)
    print(f"Wrote {latest}", file=sys.stderr)

    if evaluation["validation"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
