"""Refusal correctness metric — EVAL_SPEC §3.3."""

from __future__ import annotations

from docintel.evaluation.metrics.base import Metric
from docintel.evaluation.models import GoldenItem
from docintel.generation.models import RAGResponse


class RefusalMetric(Metric):
    """Measures whether the system correctly refuses unanswerable questions.

    Per-question: returns {"refusal_correct": 1.0} when the item is
    unanswerable and the response.refused flag is True; 0.0 otherwise.

    Returns an empty dict for factual and multi_hop items so they are
    excluded from the refusal_correct aggregate.

    Aggregate (computed by Evaluator):
        refusal_correctness = mean(refusal_correct) over unanswerable items
    """

    @property
    def name(self) -> str:
        return "refusal"

    def compute(
        self,
        item: GoldenItem,
        response: RAGResponse,
        *,
        strategy: str,
    ) -> dict[str, float]:
        if item.type != "unanswerable":
            return {}
        return {"refusal_correct": 1.0 if response.refused else 0.0}
