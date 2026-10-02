"""scripts/02_chunk.py — Chunk all ingested documents.

Reads every data/processed/{doc_id}.json, runs each configured chunking
strategy, and writes data/chunks/{strategy}/{doc_id}.jsonl.

Usage:
    python scripts/02_chunk.py [--config CONFIG]

Output per strategy:
    strategy=<name> docs=<N> chunks=<M> avg_tokens=<Y.Y>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog

from docintel.chunking.registry import get_chunker
from docintel.exceptions import ChunkingError
from docintel.ingestion.loader import Document
from docintel.utils.config import load_config
from docintel.utils.logging import configure_logging

log = structlog.get_logger()


def _load_doc(path: Path) -> Document:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Document.model_validate(raw)


def main(config_path: str | None = None) -> None:
    configure_logging(log_dir=Path("logs"))
    cfg = load_config(config_path)

    processed_dir = Path("data/processed")
    doc_paths = sorted(processed_dir.glob("*.json"))

    if not doc_paths:
        log.warning("chunk_no_documents", processed_dir=str(processed_dir))
        return

    docs: list[Document] = []
    for p in doc_paths:
        try:
            docs.append(_load_doc(p))
        except Exception as exc:
            log.error("chunk_load_doc_failed", path=str(p), error=str(exc))

    log.info("chunk_docs_loaded", count=len(docs))

    for chunk_cfg in cfg.chunking:
        strategy = chunk_cfg.name
        out_dir = Path("data/chunks") / strategy
        out_dir.mkdir(parents=True, exist_ok=True)

        chunker = get_chunker(chunk_cfg)
        total_chunks = 0
        total_tokens = 0
        failed = 0

        for doc in docs:
            try:
                chunks = chunker.chunk(doc)
            except Exception as exc:
                log.error(
                    "chunk_doc_failed",
                    strategy=strategy,
                    doc_id=doc.doc_id,
                    error=str(exc),
                )
                failed += 1
                continue

            out_path = out_dir / f"{doc.doc_id}.jsonl"
            with out_path.open("w", encoding="utf-8") as fh:
                for chunk in chunks:
                    fh.write(chunk.model_dump_json() + "\n")

            total_chunks += len(chunks)
            total_tokens += sum(c.token_count for c in chunks)

        avg_tokens = total_tokens / total_chunks if total_chunks else 0.0
        log.info(
            "chunk_strategy_complete",
            strategy=strategy,
            docs=len(docs) - failed,
            chunks=total_chunks,
            avg_tokens=round(avg_tokens, 1),
            failed=failed,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk all ingested documents.")
    parser.add_argument("--config", default=None, help="Path to experiment YAML")
    args = parser.parse_args()
    main(args.config)
