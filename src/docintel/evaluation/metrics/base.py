"""Metric ABC — DATA_MODEL §3.6."""

from __future__ import annotations

from abc import ABC, abstractmethod

from docintel.evaluation.models import GoldenItem
from docintel.generation.models import RAGResponse


class Metric(ABC):
    """Computes 1+ numeric scores for one (GoldenItem, RAGResponse) pair.

    Implementations are stateless with respect to the dataset; each compute()
    call is independent.  Some implementations hold construction-time
    dependencies (judge client, embedder) that are fixed for a run.
    """

    @abstractmethod
    def compute(
        self,
        item: GoldenItem,
        response: RAGResponse,
        *,
        strategy: str,
    ) -> dict[str, float]:
        """Return {metric_name: value}.  Names are flat, e.g. 'p_at_5'.

        Return an empty dict when the metric does not apply to this item
        (e.g. retrieval metrics on unanswerable items).  The evaluator
        excludes empty-dict entries from aggregates.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'retrieval_precision'."""
