"""Statement-level faithfulness metric — EVAL_SPEC §5, ADR-012.

Faithfulness is DISTINCT from hallucination (ADR-012):
  - Hallucination: strict — partial counts as 0.5 hallucinated.
  - Faithfulness:  loose  — partial counts as fully supported.

Reuses claim decomposition from hallucination (same JudgeClient → disk cache
serves the identical prompt, so no extra LLM call).
"""

from __future__ import annotations

import structlog

from docintel.evaluation.judge import JudgeClient
from docintel.evaluation.metrics.base import Metric
from docintel.evaluation.models import GoldenItem
from docintel.generation.models import RAGResponse

log = structlog.get_logger()


def _faithfulness_score(verdicts: list[str]) -> float:
    """Fraction of claims supported by retrieved context.

    "Supported" = verdict in {yes, partial} — both count fully.
    Returns 0.0 when verdicts is empty.
    """
    n = len(verdicts)
    if n == 0:
        return 0.0
    supported = sum(1 for v in verdicts if v in ("yes", "partial"))
    return supported / n


class FaithfulnessMetric(Metric):
    """Measures what fraction of the answer is grounded in retrieved context.

    Side-channel attributes (populated after each compute() call):
        last_claims   — atomic claims extracted from the answer
        last_verdicts — per-claim entailment verdicts
        last_warning  — non-empty string if flagged (e.g. zero_claims)
    """

    def __init__(self, judge: JudgeClient) -> None:
        self._judge = judge

        # Side-channel state
        self.last_claims: list[str] = []
        self.last_verdicts: list[str] = []
        self.last_warning: str = ""

    @property
    def name(self) -> str:
        return "faithfulness"

    def compute(
        self,
        item: GoldenItem,
        response: RAGResponse,
        *,
        strategy: str,
    ) -> dict[str, float]:
        self.last_claims = []
        self.last_verdicts = []
        self.last_warning = ""

        # Refusal → no claims → faithfulness = 0 (vacuously faithful? spec doesn't
        # address this; we return 0.0 to avoid inflating the aggregate)
        if response.refused:
            return {"faithfulness": 0.0}

        if not response.answer or not response.answer.strip():
            self.last_warning = "zero_claims"
            return {"faithfulness": 0.0}

        claims = self._judge.decompose_claims(response.answer)
        self.last_claims = claims

        if not claims:
            self.last_warning = "zero_claims"
            return {"faithfulness": 0.0}

        chunk_context = [
            (r.chunk.chunk_id, r.chunk.text) for r in response.retrieved
        ]
        # Single batched LLM call for all claims (entailment_v2)
        results = self._judge.batch_check_entailment(claims, chunk_context)
        verdicts = [r.get("verdict", "no") for r in results]

        self.last_verdicts = verdicts
        return {"faithfulness": _faithfulness_score(verdicts)}
