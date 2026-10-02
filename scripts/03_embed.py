"""scripts/03_embed.py — Embed all chunks for every configured embedder.

Reads data/chunks/{strategy}/{doc_id}.jsonl, embeds each chunk's text,
and caches vectors in data/cache/embeddings/.

Run twice to verify caching:
  First run:  cache_hits=0   (all misses — model is called)
  Second run: cache_hit_rate=1.00 (all hits — no model call)

Usage:
    python scripts/03_embed.py [--config CONFIG]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog

from docintel.embeddings.registry import get_embedder
from docintel.utils.cache import EmbeddingCache
from docintel.utils.config import load_config
from docintel.utils.logging import configure_logging

log = structlog.get_logger()


def _load_chunks(strategy_dir: Path) -> list[tuple[str, str]]:
    """Return list of (chunk_id, text) from all JSONL files in strategy_dir."""
    pairs: list[tuple[str, str]] = []
    for jsonl_path in sorted(strategy_dir.glob("*.jsonl")):
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                pairs.append((obj["chunk_id"], obj["text"]))
    return pairs


def main(config_path: str | None = None) -> None:
    configure_logging(log_dir=Path("logs"))
    cfg = load_config(config_path)

    chunks_root = Path("data/chunks")
    cache = EmbeddingCache()

    strategy_dirs = sorted(d for d in chunks_root.iterdir() if d.is_dir())
    if not strategy_dirs:
        log.warning("embed_no_chunks", chunks_root=str(chunks_root))
        return

    for emb_cfg in cfg.embeddings:
        embedder = get_embedder(emb_cfg, cache=cache)

        total_texts = 0
        total_hits = 0

        for strat_dir in strategy_dirs:
            pairs = _load_chunks(strat_dir)
            if not pairs:
                continue

            texts = [text for _, text in pairs]
            total_texts += len(texts)

            # Count pre-existing cache hits before calling embed
            hits_before = sum(
                1 for t in texts if cache.get(t, embedder.name) is not None
            )
            total_hits += hits_before

            embedder.embed(texts)

        hit_rate = total_hits / total_texts if total_texts else 0.0
        log.info(
            "embed_complete",
            embedder=embedder.name,
            total_texts=total_texts,
            cache_hits=total_hits,
            cache_hit_rate=round(hit_rate, 4),
            cost_usd=0.0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed all chunks.")
    parser.add_argument("--config", default=None, help="Path to experiment YAML")
    args = parser.parse_args()
    main(args.config)
