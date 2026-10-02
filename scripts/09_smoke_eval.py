"""Phase 8 — Smoke-test the evaluation pipeline on 6 golden items.

Picks a balanced subset (3 factual, 2 multi_hop, 1 unanswerable) from the
golden dataset and runs the full evaluator against the real RAG pipeline.

Cost: ~$0.02 per run (6 generation + ~18 judge calls via OpenRouter).
Prerequisite: make index  (Chroma indexes must exist).

Exit codes:
  0 — all PASS thresholds met
  1 — one or more FAIL thresholds breached

Usage:
    python scripts/09_smoke_eval.py [--config PATH] [--golden PATH]
    make smoke-eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Thresholds ────────────────────────────────────────────────────────────────

THRESHOLDS: dict[str, tuple[str, float]] = {
    # metric_key: (direction, threshold)  direction: "gte" or "lte"
    "refusal_correctness": ("gte", 0.85),
    "hallucination_rate":  ("lte", 0.15),
    "latency_ms_p50":      ("lte", 8000.0),
}

_GOLDEN_PATH = Path("data/eval/golden.jsonl")
_SMOKE_ITEM_COUNT = 6


def _pick_items(golden_path: Path) -> list[dict]:
    """Return a balanced subset: 3 factual, 2 multi_hop, 1 unanswerable."""
    all_items: list[dict] = [
        json.loads(line) for line in golden_path.read_text().splitlines() if line.strip()
    ]
    factual    = [i for i in all_items if i["type"] == "factual"][:3]
    multi_hop  = [i for i in all_items if i["type"] == "multi_hop"][:2]
    unanswer   = [i for i in all_items if i["type"] == "unanswerable"][:1]
    items = factual + multi_hop + unanswer
    if len(items) < _SMOKE_ITEM_COUNT:
        print(f"WARNING: only {len(items)} items available (expected {_SMOKE_ITEM_COUNT})")
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the evaluation pipeline (6 items)")
    parser.add_argument("--config", default=None, help="Path to experiment config YAML")
    parser.add_argument("--golden", default=str(_GOLDEN_PATH), help="Path to golden JSONL")
    parser.add_argument("--chunking-index", type=int, default=0)
    parser.add_argument("--embedding-index", type=int, default=0)
    args = parser.parse_args(argv)

    from docintel.utils.config import load_config
    from docintel.utils.logging import configure_logging

    configure_logging()

    config = load_config(args.config)

    # Build a temp subset golden file
    import tempfile
    golden_path = Path(args.golden)
    items = _pick_items(golden_path)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for item in items:
        tmp.write(json.dumps(item) + "\n")
    tmp.close()
    smoke_golden = Path(tmp.name)

    print(f"\n{'='*60}")
    print(f"Smoke eval — {len(items)} items")
    print(f"  factual   : {sum(1 for i in items if i['type'] == 'factual')}")
    print(f"  multi_hop : {sum(1 for i in items if i['type'] == 'multi_hop')}")
    print(f"  unanswerable: {sum(1 for i in items if i['type'] == 'unanswerable')}")
    print(f"{'='*60}\n")

    from docintel.evaluation.evaluator import Evaluator, _build_pipeline_and_metrics

    pipeline, metrics, judge, strategy = _build_pipeline_and_metrics(
        config, args.chunking_index, args.embedding_index
    )
    evaluator = Evaluator(
        pipeline=pipeline,
        metrics=metrics,
        judge=judge,
        config=config,
        strategy=strategy,
        golden_path=smoke_golden,
    )

    result = evaluator.run()
    smoke_golden.unlink(missing_ok=True)

    agg = result["aggregate_metrics"]

    print("\n── Metrics ─────────────────────────────────────────────────────")
    for k, v in sorted(agg.items()):
        print(f"  {k:<30} {v:.4f}")

    print("\n── Threshold checks ────────────────────────────────────────────")
    all_passed = True
    for key, (direction, threshold) in THRESHOLDS.items():
        value = agg.get(key)
        if value is None:
            status = "SKIP (not in aggregates)"
            print(f"  {key:<30} {status}")
            continue
        if direction == "gte":
            passed = value >= threshold
            marker = f">= {threshold}"
        else:
            passed = value <= threshold
            marker = f"<= {threshold}"
        icon = "PASS" if passed else "FAIL"
        print(f"  {key:<30} {value:.4f}  {marker}  [{icon}]")
        if not passed:
            all_passed = False

    print(f"\n── Summary {'─'*50}")
    print(f"  Questions  : {result['n_questions']}")
    print(f"  Total cost : ${result['total_cost_usd']:.4f}")
    if all_passed:
        print("\n  OVERALL: PASS\n")
        return 0
    else:
        print("\n  OVERALL: FAIL — one or more thresholds breached\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
