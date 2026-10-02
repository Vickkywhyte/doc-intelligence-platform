"""FakeVectorStore — in-memory VectorStore for unit tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from docintel.chunking.base import Chunk
from docintel.vectorstore.base import RetrievedChunk, VectorStore


class FakeVectorStore(VectorStore):
    """In-memory vector store backed by a numpy matrix."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "fake"

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._chunks = list(chunks)
        self._vectors = vectors.astype(np.float32)

    def query(
        self,
        vector: np.ndarray,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not self._chunks or self._vectors is None:
            return []

        sims = (self._vectors @ vector.astype(np.float32)).tolist()
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        ranked = ranked[: min(k, len(ranked))]

        return [
            RetrievedChunk(
                chunk=self._chunks[i],
                score=float(s),
                rank=rank,
                retriever="topk",
            )
            for rank, (i, s) in enumerate(ranked, start=1)
        ]

    def count(self) -> int:
        return len(self._chunks)
