#!/usr/bin/env python3
"""OV-NP-DETERMINISTIC-FB-1 — TextRank sole deterministic summarizer gate.

Validates the frozen TextRank implementation (no LLM, no AIN).

  .venv/bin/python examples/ov_np_deterministic_fb_1.py
  .venv/bin/python examples/ov_np_deterministic_fb_1.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from news_pipeline.summarize import (
    MAX_GRAPH_SENTENCES,
    MAX_SENTENCES,
    PAGERANK_DAMPING,
    PAGERANK_ITERS,
    REDUNDANCY_THRESHOLD,
    SIM_THRESHOLD,
    summarize_text,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLDEN = ROOT / "tests" / "data" / "textrank_golden_v1.json"
DEFAULT_OUT = HERE / "artifacts"
EXPERIMENT = "OV-NP-DETERMINISTIC-FB-1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repeats", type=int, default=25)
    args = ap.parse_args()

    data = json.loads(args.golden.read_text(encoding="utf-8"))
    items = list(data["items"])
    mismatches = []
    for item in items:
        got = summarize_text(item["title"], item["body"], max_chars=int(data["max_chars"]))
        if got != item["expected_summary"]:
            mismatches.append(
                {
                    "id": item["id"],
                    "expected": item["expected_summary"],
                    "got": got,
                }
            )

    # Triple-run determinism
    nondeterministic = []
    for item in items:
        runs = [
            summarize_text(item["title"], item["body"], max_chars=400) for _ in range(3)
        ]
        if not (runs[0] == runs[1] == runs[2]):
            nondeterministic.append(item["id"])

    latencies_ms: list[float] = []
    out_lens: list[int] = []
    in_lens: list[int] = []
    for _ in range(max(1, args.repeats)):
        for item in items:
            body = item["body"]
            in_lens.append(len(body))
            t0 = time.perf_counter()
            summary = summarize_text(item["title"], body, max_chars=400)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            out_lens.append(len(summary))

    latencies_ms.sort()
    p95_idx = max(0, int(round(0.95 * (len(latencies_ms) - 1))))
    benchmark = {
        "article_count": len(items),
        "repeats": args.repeats,
        "samples": len(latencies_ms),
        "mean_latency_ms": round(statistics.fmean(latencies_ms), 4),
        "p95_latency_ms": round(latencies_ms[p95_idx], 4),
        "max_latency_ms": round(max(latencies_ms), 4),
        "mean_input_chars": round(statistics.fmean(in_lens), 2),
        "mean_output_chars": round(statistics.fmean(out_lens), 2),
    }

    contract_ok = not mismatches and not nondeterministic
    artifact = {
        "experiment": EXPERIMENT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "corpus_id": data.get("corpus_id"),
        "golden_path": str(args.golden),
        "config": {
            "max_sentences": MAX_SENTENCES,
            "sim_threshold": SIM_THRESHOLD,
            "redundancy_threshold": REDUNDANCY_THRESHOLD,
            "pagerank_damping": PAGERANK_DAMPING,
            "pagerank_iters": PAGERANK_ITERS,
            "max_graph_sentences": MAX_GRAPH_SENTENCES,
        },
        "golden_mismatches": mismatches,
        "nondeterministic_ids": nondeterministic,
        "benchmark": benchmark,
        "validation": "PASS" if contract_ok else "FAIL",
        "notes": [
            "TextRank is the sole deterministic summarizer / LLM fallback.",
            "No edges → length-then-index ranking is TextRank degenerate handling.",
            "Empty body → title[:cap] is summarize_text degenerate handling.",
        ],
    }

    if args.dry_run:
        print(json.dumps(artifact, indent=2))
        raise SystemExit(0 if contract_ok else 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stamped = args.out_dir / f"np_deterministic_fb_1_{stamp}.json"
    latest = args.out_dir / "np_deterministic_fb_1_latest.json"
    payload = json.dumps(artifact, indent=2) + "\n"
    stamped.write_text(payload, encoding="utf-8")
    latest.write_text(payload, encoding="utf-8")
    print(payload)
    raise SystemExit(0 if contract_ok else 1)


if __name__ == "__main__":
    main()
