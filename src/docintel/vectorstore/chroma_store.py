"""ChromaDB-backed VectorStore with cosine distance (ADR-002).

Directory layout: data/indexes/{strategy}_{embedder_name}/
Collection metadata: {"hnsw:space": "cosine"} — must be set at creation time.

Chroma cosine distance: 0 = identical, 2 = opposite.
Score returned as cosine similarity: score = 1 - distance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from docintel.chunking.base import Chunk
from docintel.vectorstore.base import RetrievedChunk, VectorStore

log = structlog.get_logger()

_UPSERT_BATCH = 500


def _chunk_to_meta(chunk: Chunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "strategy": chunk.strategy,
        "index": chunk.index,
        "page": chunk.page,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "token_count": chunk.token_count,
        "chunk_metadata": json.dumps(chunk.metadata),
    }


def _meta_to_chunk(chunk_id: str, meta: dict[str, Any], document: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=meta["doc_id"],
        strategy=meta["strategy"],
        index=int(meta["index"]),
        page=int(meta["page"]),
        char_start=int(meta["char_start"]),
        char_end=int(meta["char_end"]),
        token_count=int(meta["token_count"]),
        text=document,
        metadata=json.loads(meta.get("chunk_metadata", "{}")),
    )


class ChromaStore(VectorStore):
    """Persistent Chroma collection for one (strategy, embedder) pair."""

    def __init__(
        self,
        strategy: str,
        embedder_name: str,
        base_dir: Path = Path("data/indexes"),
        *,
        client: Any = None,
    ) -> None:
        """
        Args:
            strategy: chunking strategy name, e.g. "fixed".
            embedder_name: embedder registry name, e.g. "bge_small_en_v15".
            base_dir: root directory for all indexes.
            client: optional pre-built Chroma client (for tests).
        """
        self._strategy = strategy
        self._embedder_name = embedder_name
        self._collection_name = f"{strategy}_{embedder_name}"

        if client is None:
            import chromadb

            persist_path = base_dir / self._collection_name
            persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_path))
        else:
            self._client = client

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.debug(
            "chroma_store_ready",
            collection=self._collection_name,
            count=self._collection.count(),
        )

    @property
    def name(self) -> str:
        return self._collection_name

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        metas = [_chunk_to_meta(c) for c in chunks]
        docs = [c.text for c in chunks]
        vecs_list = vectors.tolist()

        for start in range(0, len(ids), _UPSERT_BATCH):
            end = start + _UPSERT_BATCH
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=vecs_list[start:end],
                metadatas=metas[start:end],
                documents=docs[start:end],
            )

        log.info(
            "chroma_upsert",
            collection=self._collection_name,
            n=len(ids),
            total=self._collection.count(),
        )

    def query(
        self,
        vector: np.ndarray,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        n = min(k, self.count())
        if n == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [vector.tolist()],
            "n_results": n,
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        retrieved: list[RetrievedChunk] = []
        for rank, (chunk_id, meta, doc, dist) in enumerate(
            zip(
                result["ids"][0],
                result["metadatas"][0],
                result["documents"][0],
                result["distances"][0],
            ),
            start=1,
        ):
            chunk = _meta_to_chunk(chunk_id, meta, doc)
            score = float(1.0 - dist)
            retrieved.append(
                RetrievedChunk(chunk=chunk, score=score, rank=rank, retriever="topk")
            )

        return retrieved

    def count(self) -> int:
        return self._collection.count()
