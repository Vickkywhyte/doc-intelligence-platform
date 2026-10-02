"""Retriever ABC and DenseRetriever — top-k and MMR modes (DATA_MODEL §3.4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import structlog

from docintel.embeddings.base import Embedder
from docintel.utils.config import RetrievalConfig
from docintel.vectorstore.base import RetrievedChunk, VectorStore

log = structlog.get_logger()


class Retriever(ABC):
    """Base interface for all retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks for `query`.

        Args:
            query: natural-language question.
            k: number of chunks to return.
            where: optional Chroma-style metadata filter.

        Returns:
            list of RetrievedChunk sorted by relevance, rank starting at 1.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this retriever, e.g. 'dense_topk'."""


class DenseRetriever(Retriever):
    """Dense embedding retriever with optional MMR diversification.

    Mode is selected by RetrievalConfig.mmr:
    - False → top-k by cosine similarity
    - True  → Maximal Marginal Relevance (Carbonell & Goldstein 1998)
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        config: RetrievalConfig,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config

    @property
    def name(self) -> str:
        mode = "mmr" if self._config.mmr else "topk"
        return f"dense_{mode}"

    def retrieve(
        self,
        query: str,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        q_vec: np.ndarray = self._embedder.embed([query])[0]

        if self._config.mmr:
            results = self._mmr(q_vec, k, where)
        else:
            results = self._topk(q_vec, k, where)

        log.info(
            "retrieve_complete",
            retriever=self.name,
            store=self._store.name,
            k=len(results),
        )
        return results

    def _topk(
        self,
        q_vec: np.ndarray,
        k: int,
        where: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        return self._store.query(q_vec, k, where)

    def _mmr(
        self,
        q_vec: np.ndarray,
        k: int,
        where: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """Maximal Marginal Relevance selection.

        Fetches a candidate pool of min(4*k, store.count()) chunks, then
        iteratively picks the candidate maximising:
            lambda * sim(doc, query) - (1 - lambda) * max_sim(doc, selected)

        Returns k RetrievedChunks with retriever="mmr".
        """
        pool_size = max(k, min(k * 4, self._store.count()))
        candidates = self._store.query(q_vec, pool_size, where)
        if not candidates:
            return []

        # Embed candidate texts to compute inter-candidate similarity
        cand_vecs = self._embedder.embed([r.chunk.text for r in candidates])
        lam = self._config.mmr_lambda

        selected: list[int] = []
        selected_vecs: list[np.ndarray] = []

        while len(selected) < k and len(selected) < len(candidates):
            best_idx = -1
            best_score = float("-inf")

            for i, (cand, vec) in enumerate(zip(candidates, cand_vecs)):
                if i in selected:
                    continue

                rel = float(cand.score)
                redundancy = (
                    max(float(vec @ sv) for sv in selected_vecs)
                    if selected_vecs
                    else 0.0
                )
                score = lam * rel - (1.0 - lam) * redundancy

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1:
                break

            selected.append(best_idx)
            selected_vecs.append(cand_vecs[best_idx])

        return [
            RetrievedChunk(
                chunk=candidates[i].chunk,
                score=candidates[i].score,
                rank=rank,
                retriever="mmr",
            )
            for rank, i in enumerate(selected, start=1)
        ]
