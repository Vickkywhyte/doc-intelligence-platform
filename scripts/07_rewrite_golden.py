"""Phase 7 gap-fix — Rewrite fragmented gold answers into complete sentences.

Reads data/eval/golden.jsonl, calls gpt-4o-mini via OpenRouter to rewrite
each factual/multi_hop gold answer, and writes the result to
data/eval/golden.rewritten.jsonl.

NEVER modifies data/eval/golden.jsonl.
A human must review golden.rewritten.jsonl before promoting entries.

Run: make golden-rewrite
     python scripts/07_rewrite_golden.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import structlog
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv()

from docintel.evaluation.models import GoldenItem  # noqa: E402
from docintel.generation.llm_client import OpenRouterClient  # noqa: E402
from docintel.utils.config import load_config  # noqa: E402
from docintel.utils.cost_tracker import cost_tracker  # noqa: E402

log = structlog.get_logger()

GOLDEN_PATH = Path("data/eval/golden.jsonl")
OUT_PATH = Path("data/eval/golden.rewritten.jsonl")
CHUNKS_DIR = Path("data/chunks")
BUDGET_USD = 0.10

_SYSTEM_PROMPT = (
    "You rewrite fragmented gold answers into complete, natural-language sentences. "
    "Keep all facts. Keep the length to at most 2 sentences. "
    "Preserve quotes where they are definitional. "
    "Do not add facts not present in the source chunk."
)


# ── Chunk loader ──────────────────────────────────────────────────────────────


def _build_chunk_index() -> dict[str, str]:
    """Return {chunk_id: text} for every chunk across all strategies."""
    index: dict[str, str] = {}
    for jsonl in CHUNKS_DIR.rglob("*.jsonl"):
        for line in jsonl.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                index[c["chunk_id"]] = c["text"]
    return index


def _source_text(item: GoldenItem, chunk_index: dict[str, str]) -> str:
    """Return the text of the first 'fixed' strategy chunk for this item."""
    fixed_ids = item.gold_chunk_ids.get("fixed", [])
    if not fixed_ids:
        return ""
    cid = fixed_ids[0]
    return chunk_index.get(cid, "")


# ── Rewriter ──────────────────────────────────────────────────────────────────


def _build_user_prompt(question: str, current_answer: str, source_text: str) -> str:
    return (
        f"QUESTION: {question}\n\n"
        f"CURRENT ANSWER (may be fragmented): {current_answer}\n\n"
        f"SOURCE CHUNK:\n{source_text}\n\n"
        "Rewrite the CURRENT ANSWER into 1–2 complete, natural-language sentences "
        "using only facts from the SOURCE CHUNK. Output only the rewritten answer."
    )


def main(dry_run: bool = False) -> None:
    if not GOLDEN_PATH.exists():
        log.error("golden_not_found", path=str(GOLDEN_PATH))
        sys.exit(1)

    items = [
        GoldenItem.model_validate(json.loads(l))
        for l in GOLDEN_PATH.read_text().splitlines()
        if l.strip()
    ]

    answerable = [i for i in items if i.type in ("factual", "multi_hop")]
    unanswerable = [i for i in items if i.type == "unanswerable"]

    log.info(
        "rewrite_start",
        total=len(items),
        to_rewrite=len(answerable),
        skipped_unanswerable=len(unanswerable),
        dry_run=dry_run,
        output=str(OUT_PATH),
    )

    chunk_index = _build_chunk_index()
    log.info("chunk_index_built", n_chunks=len(chunk_index))

    cfg = load_config()
    llm = None if dry_run else OpenRouterClient(cfg.generation, system_prompt=_SYSTEM_PROMPT)

    total_cost = 0.0
    rewrites: list[dict] = []
    preview_rows: list[tuple[str, str, str]] = []  # (id, original, rewritten)

    for item in answerable:
        if total_cost >= BUDGET_USD:
            log.warning(
                "budget_cap_reached",
                budget_usd=BUDGET_USD,
                spent_usd=round(total_cost, 5),
                processed=len(rewrites),
                remaining=len(answerable) - len(rewrites),
            )
            # Write remaining entries unchanged
            for remaining in answerable[len(rewrites):]:
                entry = remaining.model_dump()
                entry["rewrite_status"] = "budget_cap_skipped"
                rewrites.append(entry)
            break

        src = _source_text(item, chunk_index)
        if not src:
            log.warning("no_source_chunk", item_id=item.id)
            entry = item.model_dump()
            entry["rewrite_status"] = "no_source_chunk"
            rewrites.append(entry)
            continue

        user_prompt = _build_user_prompt(item.question, item.gold_answer, src)

        if dry_run:
            rewritten = f"[DRY RUN] {item.gold_answer}"
            entry_cost = 0.0
        else:
            t0 = time.monotonic()
            response = llm.generate(user_prompt, temperature=0.0, max_tokens=256)
            latency_ms = int((time.monotonic() - t0) * 1000)
            rewritten = response.text.strip()
            entry_cost = response.cost_usd
            total_cost += entry_cost
            cost_tracker.track(
                kind="llm",
                model=response.model,
                cost_usd=entry_cost,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
            log.info(
                "entry_rewritten",
                id=item.id,
                type=item.type,
                cost_usd=round(entry_cost, 6),
                total_cost_usd=round(total_cost, 5),
                latency_ms=latency_ms,
            )

        entry = item.model_dump()
        entry["gold_answer"] = rewritten
        entry["rewrite_status"] = "rewritten"
        entry["original_answer"] = item.gold_answer
        rewrites.append(entry)

        if len(preview_rows) < 10:
            preview_rows.append((item.id, item.gold_answer, rewritten))

    # Append unanswerable entries unchanged
    for item in unanswerable:
        entry = item.model_dump()
        entry["rewrite_status"] = "skipped_unanswerable"
        rewrites.append(entry)

    # Sort back to original order by id
    id_order = {item.id: i for i, item in enumerate(items)}
    rewrites.sort(key=lambda e: id_order.get(e["id"], 9999))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fh:
        for entry in rewrites:
            fh.write(json.dumps(entry) + "\n")

    log.info(
        "rewrite_complete",
        output=str(OUT_PATH),
        total_entries=len(rewrites),
        rewritten=sum(1 for r in rewrites if r.get("rewrite_status") == "rewritten"),
        total_cost_usd=round(total_cost, 5),
    )

    # ── Side-by-side preview ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  golden-rewrite preview (first {len(preview_rows)} entries)")
    print(f"  output: {OUT_PATH}  |  total_cost=${total_cost:.5f}")
    print(f"{'='*70}\n")
    for eid, original, rewritten in preview_rows:
        print(f"[{eid}]")
        print(f"  BEFORE: {original[:120]}")
        print(f"  AFTER : {rewritten[:120]}")
        print()

    print(f"Rewrote {sum(1 for r in rewrites if r.get('rewrite_status') == 'rewritten')} entries.")
    print(f"Total cost: ${total_cost:.5f} (budget: ${BUDGET_USD:.2f})")
    print()
    print("NEXT STEPS:")
    print("  1. Review data/eval/golden.rewritten.jsonl")
    print("  2. For each entry, compare 'original_answer' vs 'gold_answer'")
    print("  3. Manually copy approved rewrites into data/eval/golden.jsonl")
    print("  4. Run: make golden-validate")
    print()
    print("golden.jsonl was NOT modified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; write placeholder rewrites to verify pipeline",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
