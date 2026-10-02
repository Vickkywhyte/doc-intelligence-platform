"""Reranker ABC and no-op stub (v1.1 will add cross-encoder reranking)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from docintel.vectorstore.base import RetrievedChunk


class Reranker(ABC):
    """Post-retrieval reranker — takes a candidate list and re-scores it."""

    @abstractmethod
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Re-score and re-rank chunks relative to query.

        Args:
            query: the original natural-language question.
            chunks: initial ranked list from the retriever.

        Returns:
            list of RetrievedChunk with retriever="rerank", ranks reassigned.
        """

    @property
    @abstractmethod
    def name(self) -> str: ...


class NoOpReranker(Reranker):
    """Pass-through reranker — order unchanged, retriever field updated to 'rerank'.

    Placeholder until a cross-encoder is integrated in v1.1.
    """

    @property
    def name(self) -> str:
        return "noop"

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk=rc.chunk,
                score=rc.score,
                rank=rc.rank,
                retriever="rerank",
            )
            for rc in chunks
        ]
