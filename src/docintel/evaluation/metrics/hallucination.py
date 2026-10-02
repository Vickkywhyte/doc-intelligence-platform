"""Claim-level hallucination metric — EVAL_SPEC §4, ADR-011.

Algorithm:
  1. Claim decomposition  (LLM, temp=0) — atomic factual claims from the answer.
  2. Entailment check per claim (LLM, temp=0) — yes | partial | no.
  3. Rate = (n_no + partial_weight * n_partial) / n_claims.

Distinct from faithfulness (§5, ADR-012).
"""

from __future__ import annotations

import structlog

from docintel.evaluation.judge import JudgeClient
from docintel.evaluation.metrics.base import Metric
from docintel.evaluation.models import GoldenItem
from docintel.generation.models import RAGResponse
from docintel.utils.config import EvaluationConfig

log = structlog.get_logger()


def _hallucination_rate(
    verdicts: list[str],
    partial_weight: float,
) -> float:
    """Compute the hallucination rate from a list of verdict strings.

    verdicts: list of "yes" | "partial" | "no" strings.
    partial_weight: contribution of a partial verdict (spec default 0.5).

    Returns 0.0 when the list is empty (refusal / zero-claim answer).
    """
    n = len(verdicts)
    if n == 0:
        return 0.0
    n_no = sum(1 for v in verdicts if v == "no")
    n_partial = sum(1 for v in verdicts if v == "partial")
    return (n_no + partial_weight * n_partial) / n


class HallucinationMetric(Metric):
    """Claim-level hallucination rate.

    Side-channel attributes (populated after each compute() call):
        last_claims           — all atomic claims extracted from the answer
        last_verdicts         — per-claim entailment verdicts
        last_hallucinated     — claims whose verdict was "no"
        last_warning          — non-empty string if flagged (e.g. zero_claims)
    """

    def __init__(
        self,
        judge: JudgeClient,
        eval_config: EvaluationConfig,
    ) -> None:
        self._judge = judge
        self._partial_weight = eval_config.partial_claim_weight

        # Side-channel state — set by compute(), read by Evaluator
        self.last_claims: list[str] = []
        self.last_verdicts: list[str] = []
        self.last_hallucinated: list[str] = []
        self.last_warning: str = ""

    @property
    def name(self) -> str:
        return "hallucination"

    def compute(
        self,
        item: GoldenItem,
        response: RAGResponse,
        *,
        strategy: str,
    ) -> dict[str, float]:
        # Reset side-channel
        self.last_claims = []
        self.last_verdicts = []
        self.last_hallucinated = []
        self.last_warning = ""

        # Refusal → 0 claims → rate = 0 (explicitly documented behavior)
        if response.refused:
            return {"hallucination_rate": 0.0}

        # Empty answer → rate = 0, flagged
        if not response.answer or not response.answer.strip():
            self.last_warning = "zero_claims"
            log.warning("hallucination_zero_claims", item_id=item.id)
            return {"hallucination_rate": 0.0}

        # Step 1: Claim decomposition
        claims = self._judge.decompose_claims(response.answer)
        self.last_claims = claims

        if not claims:
            self.last_warning = "zero_claims"
            log.warning("hallucination_zero_claims", item_id=item.id)
            return {"hallucination_rate": 0.0}

        # Step 2: Batch entailment check (single LLM call for all claims)
        chunk_context = [
            (r.chunk.chunk_id, r.chunk.text) for r in response.retrieved
        ]
        verdicts: list[str] = []
        hallucinated: list[str] = []

        results = self._judge.batch_check_entailment(claims, chunk_context)
        for claim, result in zip(claims, results):
            verdict = result.get("verdict", "no")
            verdicts.append(verdict)
            if verdict == "no":
                hallucinated.append(claim)

        self.last_verdicts = verdicts
        self.last_hallucinated = hallucinated

        # Step 3: Rate
        rate = _hallucination_rate(verdicts, self._partial_weight)
        return {"hallucination_rate": rate}
