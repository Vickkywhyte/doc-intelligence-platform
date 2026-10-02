"""Phase 5 — Embed chunks and upsert into Chroma indexes.

Reads:  data/chunks/{strategy}/{doc_id}.jsonl   (all strategies)
Writes: data/indexes/{strategy}_{embedder}/      (6 dirs: 3 × 2)

Run: make index
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import structlog

# Ensure src/ is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.chunking.base import Chunk
from docintel.embeddings.registry import get_embedder
from docintel.utils.config import load_config
from docintel.vectorstore.registry import get_store

log = structlog.get_logger()


def load_chunks(chunks_dir: Path, strategy: str) -> list[Chunk]:
    strategy_dir = chunks_dir / strategy
    if not strategy_dir.exists():
        log.warning("chunks_dir_missing", strategy=strategy, path=str(strategy_dir))
        return []

    chunks: list[Chunk] = []
    for jsonl_path in sorted(strategy_dir.glob("*.jsonl")):
        with jsonl_path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(Chunk.model_validate(json.loads(line)))
    return chunks


def main() -> None:
    cfg = load_config()
    chunks_dir = Path("data/chunks")
    indexes_dir = Path("data/indexes")
    indexes_dir.mkdir(parents=True, exist_ok=True)

    strategies = [c.name for c in cfg.chunking]
    embedder_cfgs = cfg.embeddings

    total_upserted = 0

    for strategy in strategies:
        chunks = load_chunks(chunks_dir, strategy)
        if not chunks:
            log.warning("no_chunks_found", strategy=strategy)
            continue

        log.info("strategy_loaded", strategy=strategy, chunks=len(chunks))
        texts = [c.text for c in chunks]

        for emb_cfg in embedder_cfgs:
            embedder = get_embedder(emb_cfg)
            t0 = time.monotonic()

            log.info(
                "embed_and_index_start",
                strategy=strategy,
                embedder=embedder.name,
                n_chunks=len(chunks),
            )

            vectors: np.ndarray = embedder.embed(texts)

            store = get_store(strategy, embedder.name, indexes_dir)
            store.upsert(chunks, vectors)

            elapsed = round((time.monotonic() - t0) * 1000)
            total_upserted += len(chunks)

            log.info(
                "index_complete",
                strategy=strategy,
                embedder=embedder.name,
                n_chunks=len(chunks),
                store_count=store.count(),
                latency_ms=elapsed,
            )

    log.info("embed_and_index_done", total_upserted=total_upserted)


if __name__ == "__main__":
    main()
