"""VectorStore ABC and RetrievedChunk model (DATA_MODEL §2.3 / §3.3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from docintel.chunking.base import Chunk


class RetrievedChunk(BaseModel):
    """A Chunk returned by a retriever, decorated with its rank and score."""

    model_config = {"extra": "ignore"}

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    retriever: Literal["topk", "mmr", "rerank"]


class VectorStore(ABC):
    """Persist and query (chunk, vector) pairs for one (strategy, embedder) combo."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Store or replace chunks and their corresponding vectors.

        Args:
            chunks: list of Chunk objects (length N).
            vectors: float32 array of shape [N, dim], L2-normalised.
        """

    @abstractmethod
    def query(
        self,
        vector: np.ndarray,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k nearest chunks to `vector`.

        Args:
            vector: float32 query vector of shape [dim], L2-normalised.
            k: number of results to return.
            where: optional Chroma-style metadata filter dict.

        Returns:
            list of RetrievedChunk sorted by descending score (cosine similarity),
            with rank starting at 1.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors currently in the store."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this store, e.g. 'fixed_bge_small_en_v15'."""
