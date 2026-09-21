#!/usr/bin/env python3
"""OV-NP-LLM-SUMMARIZE-1 — batch/no-ID probe on full-body residual corpus.

Evidence only. Does **not** change news_pipeline core or AIN.

Protocol (arm C only):
  INPUT: ordered feed_items with title + text only (no news_id).
  OUTPUT: {"summaries": ["...", "...", "..."]}  # strings by position
  Identity: attach news_id by array index in this probe (library simulation).
  Fail-soft: timeout / malformed / empty / len mismatch → extractive fallback.

Usage (from news-pipeline repo root):

  .venv/bin/python examples/probe_np_llm_summarize_1.py --dry-run
  .venv/bin/python examples/probe_np_llm_summarize_1.py
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

from news_pipeline.summarize import summarize_text

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "data" / "np_llm_summarize_1_corpus_v1.json"
DEFAULT_OUT = HERE / "artifacts"
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3:14b"

SYSTEM = (
    "You are a JSON-only news summarizer.\n"
    "Your entire reply MUST be one JSON object. No markdown fences, no prose, "
    "no questions, no apologies, no <think> tags in the final answer.\n"
    "INPUT: feed_items is an ordered array. Each item has title and text only.\n"
    "Do NOT invent identifiers. Do NOT emit news_id, index, or position fields.\n"
    "OUTPUT shape (exactly):\n"
    '{"summaries":["<summary for item 0>","<summary for item 1>", "..."]}\n'
    "Return exactly one string in summaries for every supplied feed item, "
    "same order. Each summary is a faithful 2-4 sentence paragraph using only "
    "that item's title and text. No confidence."
)


def _load_corpus(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = list(raw.get("items") or [])
    if not items:
        raise SystemExit(f"empty corpus: {path}")
    return raw


def _strip_think(text: str) -> str:
    """Drop Qwen/DeepSeek-style chain-of-thought wrappers before JSON parse."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE)
    # Unclosed think block: keep trailing content after the last opener.
    open_m = list(re.finditer(r"<think>", text, flags=re.IGNORECASE))
    if open_m:
        text = text[open_m[-1].end() :]
    return text.strip()


def _extract_json(text: str) -> dict[str, Any]:
    text = _strip_think(text or "")
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
    response_format_json: bool = True,
    num_ctx: int | None = None,
) -> tuple[dict[str, Any] | None, int, float, str | None, str]:
    """Return (parsed, tokens, elapsed, parse_error, raw_content_prefix)."""
    # OpenAI-compat /v1 often ignores options.num_ctx on this Ollama build; native
    # /api/chat honors it — required when frozen prompt > default 4096 ctx.
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    use_native = num_ctx is not None and int(num_ctx) > 0
    if use_native:
        url = root + "/api/chat"
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": int(num_ctx)},
        }
        if response_format_json:
            body["format"] = "json"
    else:
        url = root + "/v1/chat/completions"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "stream": False,
        }
        if response_format_json:
            body["response_format"] = {"type": "json_object"}

    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=body)
        if resp.status_code >= 400:
            detail = (resp.text or "")[:500]
            raise httpx.HTTPStatusError(
                f"{resp.status_code} Bad Request: {detail}",
                request=resp.request,
                response=resp,
            )
        data = resp.json()
    elapsed = time.perf_counter() - t0

    if use_native:
        message = data.get("message") or {}
        content = str(message.get("content") or "")
        thinking = str(message.get("thinking") or "")
        if not content.strip() and thinking.strip():
            content = thinking
        eval_count = int(data.get("eval_count") or 0)
        prompt_eval = int(data.get("prompt_eval_count") or 0)
        tokens = eval_count + prompt_eval
    else:
        message = data["choices"][0]["message"]
        content = str(message.get("content") or "")
        reasoning = str(
            message.get("reasoning") or message.get("reasoning_content") or ""
        )
        if not content.strip() and reasoning.strip():
            content = reasoning
        usage = data.get("usage") or {}
        tokens = int(usage.get("total_tokens") or 0)

    try:
        return _extract_json(content), tokens, elapsed, None, content[:8000]
    except Exception as exc:  # noqa: BLE001
        return None, tokens, elapsed, str(exc)[:240], content[:8000]


def _user_prompt(payload: dict[str, Any]) -> str:
    n = len(payload.get("feed_items") or [])
    return (
        f"Summarize all {n} feed items now.\n"
        "Reply with ONLY this JSON object (no other text):\n"
        '{"summaries":["...","...",...]}\n'
        f"The summaries array MUST have length {n}. "
        "strings only; same order as feed_items; no news_id; no index; "
        "do not ask clarifying questions.\n\n"
        "Example for 2 items:\n"
        '{"summaries":["First article summary...","Second article summary..."]}\n\n'
        + json.dumps(payload, default=str)
    )


def _token_overlap(a: str, b: str) -> float:
    ta = {t for t in re.findall(r"[A-Za-z0-9]{3,}", (a or "").upper())}
    tb = {t for t in re.findall(r"[A-Za-z0-9]{3,}", (b or "").upper())}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


def _validity(summary: str, *, title: str, body: str) -> dict[str, Any]:
    s = (summary or "").strip()
    reasons: list[str] = []
    if not s:
        reasons.append("empty")
    if len(s) < 40:
        reasons.append("too_short")
    if len(s) > 1200:
        reasons.append("too_long")
    if s and s == (title or "").strip():
        reasons.append("title_echo")
    overlap = _token_overlap(s, f"{title}\n{body}")
    if s and overlap < 0.08:
        reasons.append("low_overlap")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "chars": len(s),
        "title_token_overlap": round(overlap, 4),
    }


def _score_llm_response(
    payload: dict[str, Any] | None,
    *,
    items: list[dict[str, Any]],
    error: str | None,
) -> dict[str, Any]:
    n = len(items)
    if error:
        return {
            "schema_ok": False,
            "length_match": False,
            "completion_rate": 0.0,
            "valid_rate": 0.0,
            "accepted": 0,
            "error": error,
            "summaries": [],
            "attached": [],
        }
    assert payload is not None
    summaries = payload.get("summaries")
    if not isinstance(summaries, list):
        return {
            "schema_ok": False,
            "length_match": False,
            "completion_rate": 0.0,
            "valid_rate": 0.0,
            "accepted": 0,
            "error": "summaries missing or not a list",
            "summaries": [],
            "attached": [],
        }
    # Only plain strings count toward length match (reject index-objects).
    string_summaries: list[str] = []
    for entry in summaries:
        if isinstance(entry, str):
            string_summaries.append(entry.strip())
        else:
            return {
                "schema_ok": False,
                "length_match": False,
                "completion_rate": 0.0,
                "valid_rate": 0.0,
                "accepted": 0,
                "error": "summaries entries must be strings (no index/news_id objects)",
                "raw_summaries_len": len(summaries),
                "summaries": [],
                "attached": [],
            }

    length_match = len(string_summaries) == n
    attached: list[dict[str, Any]] = []
    valid_n = 0
    non_empty = 0
    # Always retain emitted strings for forensics (even on length mismatch).
    forensic_map: list[dict[str, Any]] = []
    for j, summary in enumerate(string_summaries):
        overlaps = []
        for i, item in enumerate(items):
            ov = _token_overlap(summary, f"{item['title']}\n{item.get('text') or ''}")
            overlaps.append((ov, i, item["news_id"], item["source_id"], item["title"][:80]))
        overlaps.sort(reverse=True)
        best = overlaps[0] if overlaps else (0.0, -1, "", "", "")
        forensic_map.append(
            {
                "emit_index": j,
                "summary_chars": len(summary),
                "summary": summary,
                "best_input_index": best[1],
                "best_overlap": round(best[0], 4),
                "best_news_id": best[2],
                "best_source_id": best[3],
                "best_title": best[4],
                "top3": [
                    {
                        "input_index": i,
                        "overlap": round(ov, 4),
                        "news_id": nid,
                        "title": title,
                    }
                    for ov, i, nid, _sid, title in overlaps[:3]
                ],
            }
        )
    covered = {m["best_input_index"] for m in forensic_map if m["best_overlap"] >= 0.08}
    missing_inputs = [
        {
            "input_index": i,
            "news_id": items[i]["news_id"],
            "source_id": items[i]["source_id"],
            "title": items[i]["title"][:100],
            "body_chars": len(items[i].get("text") or ""),
        }
        for i in range(n)
        if i not in covered
    ]
    if length_match:
        for i, item in enumerate(items):
            summary = string_summaries[i]
            if summary:
                non_empty += 1
            val = _validity(summary, title=item["title"], body=item.get("text") or "")
            if val["ok"]:
                valid_n += 1
            attached.append(
                {
                    "news_id": item["news_id"],
                    "source_id": item["source_id"],
                    "index": i,
                    "summary": summary,
                    "validity": val,
                }
            )
    return {
        "schema_ok": True,
        "length_match": length_match,
        "completion_rate": (non_empty / n) if length_match and n else 0.0,
        "valid_rate": (valid_n / n) if length_match and n else 0.0,
        "accepted": valid_n if length_match else 0,
        "raw_summaries_len": len(string_summaries),
        "error": None if length_match else "length_mismatch",
        "summaries": string_summaries if length_match else [],
        "emitted_summaries": string_summaries,  # forensic: keep even on mismatch
        "forensic_map": forensic_map,
        "likely_missing_inputs": missing_inputs,
        "attached": attached,
    }


def extractive_baseline(items: list[dict[str, Any]], *, max_chars: int = 400) -> dict[str, Any]:
    attached = []
    for i, item in enumerate(items):
        summary = summarize_text(item["title"], item.get("text") or "", max_chars=max_chars)
        val = _validity(summary, title=item["title"], body=item.get("text") or "")
        attached.append(
            {
                "news_id": item["news_id"],
                "source_id": item["source_id"],
                "index": i,
                "summary": summary,
                "validity": val,
            }
        )
    valid_n = sum(1 for a in attached if a["validity"]["ok"])
    return {
        "protocol": "extractive_baseline",
        "n": len(items),
        "completion_rate": 1.0 if items else 0.0,
        "length_match": True,
        "valid_rate": valid_n / len(items) if items else 0.0,
        "accepted": valid_n,
        "attached": attached,
    }


def run_batch_no_id(
    *,
    items: list[dict[str, Any]],
    base_url: str,
    model: str,
    timeout: float,
    response_format_json: bool = True,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    payload = {
        "feed_items": [
            {"title": it["title"], "text": it.get("text") or it["title"]} for it in items
        ]
    }
    in_chars = sum(len(x["title"]) + len(x["text"]) for x in payload["feed_items"])
    raw_prefix = ""
    try:
        parsed, tokens, elapsed, parse_error, raw_prefix = _chat(
            base_url=base_url,
            model=model,
            system=SYSTEM,
            user=_user_prompt(payload),
            timeout=timeout,
            response_format_json=response_format_json,
            num_ctx=num_ctx,
        )
        if parse_error:
            score = _score_llm_response(None, items=items, error=parse_error)
            status = "error"
        else:
            score = _score_llm_response(parsed, items=items, error=None)
            status = "ok" if score["length_match"] and score["error"] is None else "fail"
    except Exception as exc:  # noqa: BLE001 — probe records fail-soft
        parsed = None
        tokens = 0
        elapsed = 0.0
        score = _score_llm_response(None, items=items, error=str(exc)[:240])
        status = "error"

    # Fail-soft: extractive attach when LLM path fails length/schema/timeout
    used_extractive = False
    if not score.get("length_match"):
        used_extractive = True
        fb = extractive_baseline(items)
        score["fallback_attached"] = fb["attached"]
        score["fallback"] = "extractive"

    return {
        "protocol": "C_batch_no_id_string_array",
        "status": status,
        "model": model,
        "response_format_json": response_format_json,
        "num_ctx": num_ctx,
        "n": len(items),
        "in_chars": in_chars,
        "elapsed_s": round(elapsed, 3),
        "tokens": tokens,
        "used_extractive_fallback": used_extractive,
        **score,
        "raw_response_keys": sorted(parsed.keys()) if isinstance(parsed, dict) else [],
        "raw_content_prefix": raw_prefix,
    }


def _positional_gates(
    items: list[dict[str, Any]], llm: dict[str, Any]
) -> dict[str, Any]:
    """Gate 1 = length/order contract; Gate 2 = per-article faithfulness."""
    n = len(items)
    length_match = bool(llm.get("length_match"))
    attached = list(llm.get("attached") or [])
    low_own = 0
    better_other = 0
    per_item: list[dict[str, Any]] = []
    if length_match and len(attached) == n:
        for i, a in enumerate(attached):
            summary = str(a.get("summary") or "")
            own = _token_overlap(
                summary, f"{items[i]['title']}\n{items[i].get('text') or ''}"
            )
            others = [
                (
                    _token_overlap(
                        summary,
                        f"{items[j]['title']}\n{items[j].get('text') or ''}",
                    ),
                    j,
                )
                for j in range(n)
                if j != i
            ]
            others.sort(reverse=True)
            best_o, best_j = others[0] if others else (0.0, -1)
            flags: list[str] = []
            if own < 0.08:
                low_own += 1
                flags.append("low_own")
            if best_o > own + 0.05:
                better_other += 1
                flags.append(f"better_other[{best_j}]")
            per_item.append(
                {
                    "index": i,
                    "own_overlap": round(own, 4),
                    "best_other_index": best_j,
                    "best_other_overlap": round(best_o, 4),
                    "validity_ok": bool((a.get("validity") or {}).get("ok")),
                    "flags": flags,
                    "summary_chars": len(summary),
                }
            )
    gate1 = length_match and int(llm.get("raw_summaries_len") or 0) == n
    # On length match, skip/split/reorder heuristics ≈ better_other / forensic dups.
    # Gate1 also fails if length mismatch (implicit skip/merge).
    gate1_pass = bool(gate1) and better_other == 0
    gate2_pass = (
        gate1
        and low_own == 0
        and all(p.get("validity_ok") for p in per_item)
        and len(per_item) == n
    )
    return {
        "gate1_length_and_order": gate1_pass,
        "gate2_faithfulness": gate2_pass,
        "length_match": length_match,
        "low_own_count": low_own,
        "better_other_count": better_other,
        "per_item": per_item,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument(
        "--no-response-format",
        action="store_true",
        help="Omit OpenAI response_format=json_object (needed for some Ollama models).",
    )
    ap.add_argument(
        "--num-ctx",
        type=int,
        default=0,
        help="Ollama options.num_ctx (0 = runner default). Use when prompt exceeds default 4096.",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Independent repeats (SUMMARIZE-3 repeatability). Default 1.",
    )
    ap.add_argument(
        "--experiment",
        default="",
        help="Override experiment id stamped into artifacts.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus = _load_corpus(args.corpus)
    items = list(corpus["items"])
    baseline = extractive_baseline(items)
    runs_n = max(1, int(args.runs))

    if args.dry_run:
        result = {
            "experiment": args.experiment or "OV-NP-LLM-SUMMARIZE-1",
            "dry_run": True,
            "corpus_id": corpus.get("corpus_id"),
            "n": len(items),
            "runs": runs_n,
            "body_chars": corpus.get("body_chars"),
            "extractive_baseline": {
                k: baseline[k]
                for k in (
                    "n",
                    "completion_rate",
                    "length_match",
                    "valid_rate",
                    "accepted",
                )
            },
            "would_call": {
                "model": args.model,
                "base_url": args.base_url,
                "timeout": args.timeout,
                "protocol": "C_batch_no_id_string_array",
                "num_ctx": args.num_ctx or None,
                "response_format_json": not args.no_response_format,
            },
        }
        print(json.dumps(result, indent=2))
        return

    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model)
    if args.experiment:
        exp_id = args.experiment
    elif runs_n > 1 and "deepseek" in args.model.lower():
        exp_id = "OV-NP-LLM-SUMMARIZE-3"
    elif "deepseek" in args.model.lower():
        exp_id = "OV-NP-LLM-SUMMARIZE-2"
    else:
        exp_id = "OV-NP-LLM-SUMMARIZE-1"

    run_rows: list[dict[str, Any]] = []
    for i in range(runs_n):
        llm = run_batch_no_id(
            items=items,
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            response_format_json=not args.no_response_format,
            num_ctx=(args.num_ctx or None),
        )
        gates = _positional_gates(items, llm)
        row = {
            "run": i + 1,
            "status": llm.get("status"),
            "length_match": llm.get("length_match"),
            "raw_summaries_len": llm.get("raw_summaries_len"),
            "completion_rate": llm.get("completion_rate"),
            "valid_rate": llm.get("valid_rate"),
            "elapsed_s": llm.get("elapsed_s"),
            "tokens": llm.get("tokens"),
            "used_extractive_fallback": llm.get("used_extractive_fallback"),
            "gates": gates,
            "llm_batch_no_id": llm,
        }
        run_rows.append(row)
        print(
            json.dumps(
                {
                    "run": i + 1,
                    "of": runs_n,
                    "length_match": row["length_match"],
                    "gate1": gates["gate1_length_and_order"],
                    "gate2": gates["gate2_faithfulness"],
                    "elapsed_s": row["elapsed_s"],
                    "tokens": row["tokens"],
                },
                indent=2,
            ),
            flush=True,
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gate1_pass = sum(1 for r in run_rows if r["gates"]["gate1_length_and_order"])
    gate2_pass = sum(1 for r in run_rows if r["gates"]["gate2_faithfulness"])
    both_pass = sum(
        1
        for r in run_rows
        if r["gates"]["gate1_length_and_order"] and r["gates"]["gate2_faithfulness"]
    )
    out = {
        "experiment": exp_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_id": corpus.get("corpus_id"),
        "corpus_path": str(args.corpus),
        "n": len(items),
        "runs": runs_n,
        "model": args.model,
        "num_ctx": args.num_ctx or None,
        "response_format_json": not args.no_response_format,
        "body_chars": corpus.get("body_chars"),
        "extractive_baseline": {
            k: baseline[k]
            for k in ("n", "completion_rate", "length_match", "valid_rate", "accepted")
        },
        "aggregate": {
            "gate1_pass": gate1_pass,
            "gate2_pass": gate2_pass,
            "both_gates_pass": both_pass,
            "length_match_pass": sum(1 for r in run_rows if r["length_match"]),
            "elapsed_s": [r["elapsed_s"] for r in run_rows],
            "tokens": [r["tokens"] for r in run_rows],
        },
        "runs_detail": run_rows,
        "promotion": "evidence_only — no Accept claimed by this script",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"np_llm_summarize_{model_slug}_{stamp}.json"
    latest = args.out_dir / f"np_llm_summarize_{model_slug}_latest.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    latest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if runs_n > 1:
        soak = args.out_dir / f"np_llm_summarize_3_{model_slug}_soak_latest.json"
        soak.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if args.model == DEFAULT_MODEL and runs_n == 1:
        (args.out_dir / "np_llm_summarize_1_latest.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "wrote": str(path),
                "experiment": exp_id,
                "runs": runs_n,
                "aggregate": out["aggregate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
