"""Retrieval precision metrics: P@k, R@k, MRR, NDCG@k — EVAL_SPEC §2."""

from __future__ import annotations

import math

import structlog

from docintel.evaluation.metrics.base import Metric
from docintel.evaluation.models import GoldenItem
from docintel.generation.models import RAGResponse

log = structlog.get_logger()


def _precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """P@k = |top-k(R) ∩ G| / k."""
    return len(set(retrieved[:k]) & gold) / k


def _recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """R@k = |top-k(R) ∩ G| / |G|."""
    return len(set(retrieved[:k]) & gold) / len(gold)


def _mrr(retrieved: list[str], gold: set[str]) -> float:
    """RR = 1/rank_of_first_relevant, 0 if none found."""
    for i, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """NDCG@k with binary relevance (EVAL_SPEC §2.4).

    DCG@k  = Σ_{i=1..k}  rel_i / log2(i + 1)
    IDCG@k = Σ_{i=1..min(|G|,k)} 1 / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k  (0 if IDCG@k == 0)
    """
    top_k = retrieved[:k]
    dcg = sum(
        (1.0 if c in gold else 0.0) / math.log2(i + 1)
        for i, c in enumerate(top_k, start=1)
    )
    n_ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


class RetrievalPrecisionMetric(Metric):
    """Computes P@k, R@k, MRR, and NDCG@k for each k in k_values.

    Returns an empty dict for unanswerable items (gold_chunk_ids is empty
    for the current strategy), which excludes them from aggregates.
    """

    def __init__(self, k_values: list[int] | None = None) -> None:
        self._k_values: list[int] = k_values if k_values is not None else [1, 3, 5]

    @property
    def name(self) -> str:
        return "retrieval_precision"

    def compute(
        self,
        item: GoldenItem,
        response: RAGResponse,
        *,
        strategy: str,
    ) -> dict[str, float]:
        gold = set(item.gold_chunk_ids.get(strategy, []))
        if not gold:
            # Unanswerable or strategy missing — N/A, excluded from aggregates
            return {}

        retrieved = [r.chunk.chunk_id for r in response.retrieved]
        results: dict[str, float] = {}

        for k in self._k_values:
            results[f"p_at_{k}"] = _precision_at_k(retrieved, gold, k)
            results[f"recall_at_{k}"] = _recall_at_k(retrieved, gold, k)
            results[f"ndcg_at_{k}"] = _ndcg_at_k(retrieved, gold, k)

        results["mrr"] = _mrr(retrieved, gold)
        return results
