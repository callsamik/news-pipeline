#!/usr/bin/env python3
"""Offline LLM ID-protocol probe — outside AIN, using Ollama + httpx only.

Does **not** import ``ain`` or modify ``news_pipeline`` core (still extractive).
Uses the same fixed residual corpus as AIN OV-FEED-SUMMARIZE-PROBE-1.

Protocols:
  A  baseline_id   — legacy prompt; input field ``id``; model returns news_id
  B  schema_id     — explicit news_id I/O schema + worked example
  C  batch_no_id   — 10 items; ordered summaries by index (Python owns identity)
  D  single_no_id  — 1 item/call; summary only

Usage (from news-pipeline repo root):

  .venv/bin/python examples/probe_llm_id_protocols.py --dry-run
  .venv/bin/python examples/probe_llm_id_protocols.py
  .venv/bin/python examples/probe_llm_id_protocols.py --protocols A,B,C

Prefer running when the AIN production worker is idle so Ollama is not contended.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "data" / "summarize_probe_corpus_v1.json"
DEFAULT_OUT = HERE / "artifacts"
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3:14b"

BASELINE_SYSTEM = (
    "You summarize already-fetched news feed items. "
    "Respond with a single JSON object only — no markdown fences, no <think> tags "
    "in the final JSON, no prose outside JSON. "
    "Use exactly one top-level key: analyses (array). "
    "Each analysis: news_id (string), summary (concise faithful paragraph, "
    "2-4 sentences). "
    "Return exactly one analysis for every supplied feed item — never return an "
    "empty analyses array when items were provided. "
    "For each item use only its title and text; do not add information not present "
    "in that item. Do not invent prices or market moves. "
    "Do NOT include confidence."
)

SCHEMA_SYSTEM = (
    "You summarize already-fetched news feed items.\n"
    "\n"
    "INPUT structure (what you receive in feed_items):\n"
    "{\n"
    '  "feed_items": [\n'
    "    {\n"
    '      "news_id": "<opaque string — copy exactly>",\n'
    '      "source_id": "<string>",\n'
    '      "title": "<string>",\n'
    '      "text": "<string>"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "OUTPUT structure (respond with this JSON object only):\n"
    "{\n"
    '  "analyses": [\n'
    "    {\n"
    '      "news_id": "<exact copy of the supplied feed_items[].news_id>",\n'
    '      "summary": "<faithful 2-4 sentence paragraph>"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "Rules:\n"
    "- Single JSON object only — no markdown fences, no prose outside JSON.\n"
    "- Exactly one analysis per feed item; never an empty analyses array.\n"
    "- Copy news_id exactly — do not invent, renumber, slugify, replace with "
    "titles/URLs/project codes, or otherwise transform it.\n"
    "- Use only title and text. No confidence."
)

NO_ID_SYSTEM = (
    "You summarize already-fetched news feed items.\n"
    "Respond with a single JSON object only — no markdown fences, no prose outside JSON.\n"
    "INPUT: feed_items is an ordered array. Each item has index (0-based integer), "
    "title, and text. Do NOT invent identifiers.\n"
    "OUTPUT:\n"
    "{\n"
    '  "summaries": [\n'
    '    {"index": 0, "summary": "<faithful 2-4 sentence paragraph>"}\n'
    "  ]\n"
    "}\n"
    "Return exactly one summary for every supplied index. Copy index exactly. "
    "Use only title and text. No news_id field."
)


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = list(raw.get("items") or [])
    if not items:
        raise SystemExit(f"empty corpus: {path}")
    return items


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("no JSON object in model response")
        text = m.group(0)
    return json.loads(text)


def _chat(
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    timeout: float,
) -> tuple[dict[str, Any], int, float]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
    elapsed = time.perf_counter() - t0
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    tokens = int(usage.get("total_tokens") or 0)
    return _extract_json(content), tokens, elapsed


def _user_baseline(payload: dict[str, Any]) -> str:
    return (
        "Summarize each of the following feed items.\n"
        "Return {\"analyses\":[{\"news_id\":\"...\",\"summary\":\"...\"}, ...]}.\n"
        "One analysis per item. JSON only.\n\n"
        + json.dumps(payload, default=str)
    )


def _user_schema(payload: dict[str, Any]) -> str:
    return (
        "Summarize each feed item.\n"
        "INPUT: feed_items[].news_id is the opaque identity — copy it exactly.\n"
        "OUTPUT: {\"analyses\":[{\"news_id\":\"<exact copy>\",\"summary\":\"...\"}, ...]}\n"
        "\n"
        "Example — if input has news_id \"news_8e0175a5d8447e6f\", a correct output "
        "uses that same news_id string. Wrong: \"1\", titles, URLs, project codes.\n\n"
        + json.dumps(payload, default=str)
    )


def _user_no_id(payload: dict[str, Any]) -> str:
    return (
        "Summarize each feed item by index.\n"
        "OUTPUT: {\"summaries\":[{\"index\":0,\"summary\":\"...\"}, ...]}\n"
        "Copy index exactly. Do not emit news_id.\n\n"
        + json.dumps(payload, default=str)
    )


def _score_id(payload: dict[str, Any], *, allowed: set[str]) -> dict[str, Any]:
    analyses = payload.get("analyses")
    if not isinstance(analyses, list):
        return {
            "schema_ok": False,
            "accepted": 0,
            "exact_id_rate": 0.0,
            "emitted": [],
            "error": "analyses missing",
        }
    emitted: list[str] = []
    matched: set[str] = set()
    for entry in analyses:
        if not isinstance(entry, dict):
            continue
        nid = str(entry.get("news_id") or "").strip()
        summary = str(entry.get("summary") or "").strip()
        emitted.append(nid)
        if nid in allowed and summary:
            matched.add(nid)
    accepted = len(matched)
    return {
        "schema_ok": True,
        "accepted": accepted,
        "exact_id_rate": accepted / len(allowed) if allowed else 0.0,
        "raw_analyses_len": len(analyses),
        "emitted": emitted,
        "error": None,
    }


def _score_no_id(payload: dict[str, Any], *, n_items: int) -> dict[str, Any]:
    summaries = payload.get("summaries")
    if not isinstance(summaries, list):
        return {
            "schema_ok": False,
            "accepted": 0,
            "completion_rate": 0.0,
            "error": "summaries missing",
        }
    by_index: dict[int, str] = {}
    for entry in summaries:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        summary = str(entry.get("summary") or "").strip()
        if summary:
            by_index[idx] = summary
    accepted = sum(1 for i in range(n_items) if i in by_index)
    return {
        "schema_ok": True,
        "accepted": accepted,
        "completion_rate": accepted / n_items if n_items else 0.0,
        "indexes": sorted(by_index),
        "error": None,
    }


def run_protocol(
    *,
    code: str,
    items: list[dict[str, Any]],
    base_url: str,
    model: str,
    timeout: float,
    single: bool,
) -> dict[str, Any]:
    batches = [[it] for it in items] if single else [items]
    batch_results = []
    total_accepted = 0
    total_attempted = 0
    total_tokens = 0
    total_elapsed = 0.0

    for batch in batches:
        total_attempted += len(batch)
        if code == "A":
            payload = {
                "feed_items": [
                    {
                        "id": it["news_id"],
                        "source_id": it["source_id"],
                        "title": it["title"],
                        "text": it.get("text") or it["title"],
                    }
                    for it in batch
                ]
            }
            system, user = BASELINE_SYSTEM, _user_baseline(payload)
        elif code == "B":
            payload = {
                "feed_items": [
                    {
                        "news_id": it["news_id"],
                        "source_id": it["source_id"],
                        "title": it["title"],
                        "text": it.get("text") or it["title"],
                    }
                    for it in batch
                ]
            }
            system, user = SCHEMA_SYSTEM, _user_schema(payload)
        else:
            payload = {
                "feed_items": [
                    {
                        "index": i,
                        "source_id": it["source_id"],
                        "title": it["title"],
                        "text": it.get("text") or it["title"],
                    }
                    for i, it in enumerate(batch)
                ]
            }
            system, user = NO_ID_SYSTEM, _user_no_id(payload)

        try:
            parsed, tokens, elapsed = _chat(
                base_url=base_url,
                model=model,
                system=system,
                user=user,
                timeout=timeout,
            )
            err = None
        except Exception as exc:  # noqa: BLE001
            parsed, tokens, elapsed, err = {}, 0, 0.0, str(exc)[:300]

        total_tokens += tokens
        total_elapsed += elapsed
        if code in {"A", "B"}:
            score = _score_id(
                parsed, allowed={it["news_id"] for it in batch}
            )
        else:
            score = _score_no_id(parsed, n_items=len(batch))
        total_accepted += int(score.get("accepted") or 0)
        batch_results.append(
            {
                "n_items": len(batch),
                "elapsed_s": round(elapsed, 3),
                "tokens": tokens,
                "error": err,
                "score": score,
                "payload_keys": sorted(parsed.keys()) if parsed else [],
            }
        )

    return {
        "protocol": code,
        "single": single,
        "n_items": len(items),
        "n_calls": len(batches),
        "accepted": total_accepted,
        "attempted": total_attempted,
        "acceptance_rate": (
            total_accepted / total_attempted if total_attempted else 0.0
        ),
        "total_elapsed_s": round(total_elapsed, 3),
        "total_tokens": total_tokens,
        "batches": batch_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--protocols", default="A,B,C,D")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items = _load_corpus(args.corpus)
    wanted = {p.strip().upper() for p in args.protocols.split(",") if p.strip()}
    plan = [
        ("A", False, "baseline_id"),
        ("B", False, "schema_id"),
        ("C", False, "batch_no_id"),
        ("D", True, "single_no_id"),
    ]

    results: dict[str, Any] = {
        "probe": "news-pipeline/examples/probe_llm_id_protocols",
        "ain_experiment_ref": "OV-FEED-SUMMARIZE-PROBE-1",
        "imports_ain": False,
        "uses_news_pipeline_core_llm": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "corpus": str(args.corpus),
        "n_items": len(items),
        "model": args.model,
        "base_url": args.base_url,
        "dry_run": args.dry_run,
        "protocols": {},
    }

    for code, single, label in plan:
        if code not in wanted:
            continue
        print(f"=== protocol {code} ({label}) ===", flush=True)
        if args.dry_run:
            results["protocols"][code] = {
                "label": label,
                "single": single,
                "n_items": len(items),
                "n_calls": len(items) if single else 1,
            }
            continue
        results["protocols"][code] = run_protocol(
            code=code,
            items=items,
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            single=single,
        )
        pr = results["protocols"][code]
        print(
            f"  accepted={pr['accepted']}/{pr['attempted']} "
            f"rate={pr['acceptance_rate']:.3f} "
            f"elapsed={pr['total_elapsed_s']}s tokens={pr['total_tokens']}",
            flush=True,
        )

    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"probe_results_{stamp}.json"
    latest = args.out_dir / "probe_results_latest.json"
    blob = json.dumps(results, indent=2, default=str)
    out_path.write_text(blob, encoding="utf-8")
    latest.write_text(blob, encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
