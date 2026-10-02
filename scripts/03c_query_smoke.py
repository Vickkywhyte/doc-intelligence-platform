"""Phase 5 — Smoke-test retrieval with 3 seed questions.

Queries each of the 6 (strategy × embedder) index combinations, prints
top-3 results per combo so a human can verify the results are on-topic.

Run: make query-smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.embeddings.registry import get_embedder
from docintel.retrieval.retriever import DenseRetriever
from docintel.utils.config import load_config
from docintel.vectorstore.registry import get_store

log = structlog.get_logger()

SMOKE_QUERIES = [
    "What are the key terms and payment conditions in the contracts?",
    "What confidentiality and non-disclosure obligations apply?",
    "What services and deliverables are described in the documents?",
]


def main() -> None:
    cfg = load_config()
    indexes_dir = Path("data/indexes")

    strategies = [c.name for c in cfg.chunking]
    embedder_cfgs = cfg.embeddings

    for strategy in strategies:
        for emb_cfg in embedder_cfgs:
            embedder = get_embedder(emb_cfg)
            store = get_store(strategy, embedder.name, indexes_dir)

            if store.count() == 0:
                log.warning(
                    "empty_index",
                    store=store.name,
                    hint="run 'make index' first",
                )
                continue

            retriever = DenseRetriever(store, embedder, cfg.retrieval)

            print(f"\n{'='*70}")
            print(f"  Store: {store.name}  ({store.count()} chunks)")
            print(f"{'='*70}")

            for query in SMOKE_QUERIES:
                results = retriever.retrieve(query, k=3)
                print(f"\n  Q: {query}")
                for r in results:
                    preview = r.chunk.text[:120].replace("\n", " ")
                    print(
                        f"  [{r.rank}] score={r.score:.4f}  "
                        f"doc={r.chunk.doc_id[:8]}  "
                        f"page={r.chunk.page}  "
                        f'"{preview}…"'
                    )


if __name__ == "__main__":
    main()
