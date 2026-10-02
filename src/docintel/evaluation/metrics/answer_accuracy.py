"""Answer accuracy metrics: semantic similarity + LLM judge — EVAL_SPEC §3.1-3.2."""

from __future__ import annotations

import numpy as np
import structlog

from docintel.embeddings.base import Embedder
from docintel.evaluation.judge import JudgeClient
from docintel.evaluation.metrics.base import Metric
from docintel.evaluation.models import GoldenItem
from docintel.generation.models import RAGResponse

log = structlog.get_logger()

_SCORED_TYPES = frozenset({"factual", "multi_hop"})


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalised vectors."""
    return float(np.dot(a, b))


class AnswerAccuracyMetric(Metric):
    """Semantic similarity + judge correctness score.

    Applies only to factual and multi_hop items; returns {} for
    unanswerable items (which are scored by RefusalMetric instead).

    Returned keys:
        sem_sim          — cosine(embed(gold_answer), embed(answer))  ∈ [0, 1]
        answer_accuracy  — mean(judge_score) / 5                      ∈ [0, 1]
    """

    def __init__(self, embedder: Embedder, judge: JudgeClient) -> None:
        self._embedder = embedder
        self._judge = judge

    @property
    def name(self) -> str:
        return "answer_accuracy"

    def compute(
        self,
        item: GoldenItem,
        response: RAGResponse,
        *,
        strategy: str,
    ) -> dict[str, float]:
        if item.type not in _SCORED_TYPES:
            return {}

        # ── Semantic similarity ───────────────────────────────────────────────
        try:
            vecs = self._embedder.embed([item.gold_answer, response.answer])
            sem_sim = _cosine_sim(vecs[0], vecs[1])
            # Clamp to [0, 1] — cosine can be slightly negative for NL pairs
            sem_sim = max(0.0, min(1.0, sem_sim))
        except Exception as exc:
            log.warning("answer_accuracy_sem_sim_failed", error=str(exc))
            sem_sim = 0.0

        # ── LLM correctness judge (1–5 → [0, 1]) ─────────────────────────────
        try:
            verdict = self._judge.judge_correctness(
                question=item.question,
                gold_answer=item.gold_answer,
                answer=response.answer,
            )
            score = int(verdict.get("score", 1))
            score = max(1, min(5, score))
            answer_accuracy = score / 5.0
        except Exception as exc:
            log.warning("answer_accuracy_judge_failed", error=str(exc))
            answer_accuracy = 0.0

        return {
            "sem_sim": sem_sim,
            "answer_accuracy": answer_accuracy,
        }
