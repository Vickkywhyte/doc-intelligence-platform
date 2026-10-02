"""Integration test for the full evaluation pipeline.

Uses only fake/mock components — no network calls, no real LLM, no disk indexes.
Prevents regression of Phase 8 bugs:
  - refusal_correctness mis-scoring
  - missing per-question cost attribution
  - broken aggregate metric assembly

Run with:
    pytest tests/test_eval_integration.py -v -m integration
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.chunking.base import Chunk
from docintel.evaluation.evaluator import Evaluator
from docintel.evaluation.judge import JudgeClient
from docintel.evaluation.metrics import (
    AnswerAccuracyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    RefusalMetric,
    RetrievalPrecisionMetric,
)
from docintel.evaluation.models import GoldenItem
from docintel.generation.models import Citation, RAGResponse
from docintel.utils.config import (
    ChunkingConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    GenerationConfig,
    RetrievalConfig,
)
from docintel.vectorstore.base import RetrievedChunk
from tests.mocks.embedder import FakeEmbedder
from tests.mocks.llm import FakeLLMClient

# ── Helpers ───────────────────────────────────────────────────────────────────

_STRATEGY = "fixed"
_DOC_ID = "aabbccdd11223344"
_CHUNK_ID = f"{_DOC_ID}:{_STRATEGY}:00001"


def _make_chunk(chunk_id: str = _CHUNK_ID) -> Chunk:
    parts = chunk_id.split(":")
    doc_id, strategy, idx = parts[0], parts[1], parts[2]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        strategy=strategy,
        index=int(idx),
        text="The notice period is 30 days written notice.",
        page=1,
        char_start=0,
        char_end=50,
        token_count=10,
    )


def _make_retrieved(chunk_id: str = _CHUNK_ID, score: float = 0.9) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=_make_chunk(chunk_id), score=score, rank=1, retriever="topk")]


def _factual_response(question: str, chunk_id: str = _CHUNK_ID) -> RAGResponse:
    return RAGResponse(
        query=question,
        answer=f"The answer is stated in the document [{_DOC_ID}:1:{chunk_id}].",
        refused=False,
        citations=[Citation(doc_id=_DOC_ID, page=1, chunk_id=chunk_id)],
        retrieved=_make_retrieved(chunk_id),
        model="fake",
        latency_ms=100,
        cost_usd=0.001,
    )


def _refusal_response(question: str) -> RAGResponse:
    return RAGResponse(
        query=question,
        answer="I don't know based on the provided documents.",
        refused=True,
        citations=[],
        retrieved=_make_retrieved(score=0.2),
        model="fake",
        latency_ms=50,
        cost_usd=0.0005,
    )


def _make_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        chunking=[ChunkingConfig(name="fixed")],
        embeddings=[EmbeddingConfig(name="bge_small_en_v15")],
        retrieval=RetrievalConfig(k=5),
        generation=GenerationConfig(),
        evaluation=EvaluationConfig(
            metrics=["retrieval", "answer_accuracy", "hallucination", "faithfulness", "refusal"],
            bootstrap_samples=100,  # fast
            seed=0,
        ),
    )


def _make_judge(tmp_path: Path) -> JudgeClient:
    """Judge backed by FakeLLMClient that returns correct JSON for all prompt types."""
    responses = [
        # claim_decomposition → 1 claim
        '{"claims": ["The answer is stated."]}',
        # batch entailment_v2 → 1 claim, verdict yes
        '{"results": [{"claim_index": 0, "verdict": "yes", "chunk_id": null}]}',
        # correctness → score 5
        '{"score": 5, "rationale": "Correct answer."}',
    ] * 10  # enough for all non-refusal items
    llm = FakeLLMClient(responses, default='{"claims": ["placeholder"]}')
    cfg = _make_experiment_config().evaluation
    return JudgeClient(llm, cfg, cache_dir=tmp_path / "judge_cache")


def _make_golden_items() -> list[GoldenItem]:
    """6 items: 3 factual, 2 multi_hop, 1 unanswerable."""
    return [
        GoldenItem(
            id="t001", question="What is the notice period?",
            gold_answer="30 days.",
            gold_chunk_ids={_STRATEGY: [_CHUNK_ID]},
            type="factual",
        ),
        GoldenItem(
            id="t002", question="What fees apply after 30 days?",
            gold_answer="Late fees apply.",
            gold_chunk_ids={_STRATEGY: [_CHUNK_ID]},
            type="factual",
        ),
        GoldenItem(
            id="t003", question="What is the payment schedule?",
            gold_answer="Monthly invoices.",
            gold_chunk_ids={_STRATEGY: [_CHUNK_ID]},
            type="factual",
        ),
        GoldenItem(
            id="t004", question="How does the notice period affect late fees?",
            gold_answer="After 30 days, late fees apply.",
            gold_chunk_ids={_STRATEGY: [_CHUNK_ID]},
            type="multi_hop",
        ),
        GoldenItem(
            id="t005", question="What happens when payment is delayed and notice given?",
            gold_answer="Contract may be terminated.",
            gold_chunk_ids={_STRATEGY: [_CHUNK_ID]},
            type="multi_hop",
        ),
        GoldenItem(
            id="t006", question="What is TechCorp's cryptocurrency reimbursement policy?",
            gold_answer="N/A",  # unanswerable: gold_answer min_length=1 requires a placeholder
            gold_chunk_ids={},
            type="unanswerable",
        ),
    ]


# ── Test ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_full_pipeline_on_golden_subset(tmp_path: Path) -> None:
    """End-to-end: 6 items through the evaluator with fake components."""
    items = _make_golden_items()
    config = _make_experiment_config()

    # Write golden items to a temp file
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(
        "\n".join(json.dumps(item.model_dump()) for item in items)
    )

    # Mock pipeline: answer factual/multi_hop correctly, refuse unanswerable
    mock_pipeline = MagicMock()
    def _fake_query(question: str) -> RAGResponse:
        item = next((i for i in items if i.question == question), None)
        if item and item.type == "unanswerable":
            return _refusal_response(question)
        return _factual_response(question)
    mock_pipeline.query.side_effect = _fake_query

    # Build real metrics with fake judge + embedder
    judge = _make_judge(tmp_path)
    embedder = FakeEmbedder()
    metrics = [
        RetrievalPrecisionMetric(k_values=[1, 3, 5]),
        AnswerAccuracyMetric(embedder, judge),
        HallucinationMetric(judge, config.evaluation),
        FaithfulnessMetric(judge),
        RefusalMetric(),
    ]

    evaluator = Evaluator(
        pipeline=mock_pipeline,
        metrics=metrics,
        judge=judge,
        config=config,
        strategy=_STRATEGY,
        golden_path=golden_path,
    )

    result = evaluator.run()

    # ── Structure assertions ──────────────────────────────────────────────────
    assert result["n_questions"] == 6
    assert "aggregate_metrics" in result
    assert "aggregate_ci" in result
    assert "per_question" in result
    assert len(result["per_question"]) == 6

    # All required aggregate keys present
    agg = result["aggregate_metrics"]
    for key in ("p_at_1", "p_at_5", "mrr", "ndcg_at_5"):
        assert key in agg, f"Missing aggregate key: {key}"
    assert "latency_ms_p50" in agg
    assert "cost_per_1k_q_usd" in agg

    # ── Value range assertions ────────────────────────────────────────────────
    for key, val in agg.items():
        if key in ("latency_ms_p50", "cost_per_1k_q_usd"):
            assert val >= 0, f"{key}={val} should be >= 0"
        else:
            assert 0.0 <= val <= 1.0, f"{key}={val} not in [0, 1]"

    # ── Refusal correctness: the 1 unanswerable was correctly refused ─────────
    assert agg["refusal_correctness"] == 1.0, (
        f"refusal_correctness={agg['refusal_correctness']} — the mock refused the unanswerable item"
    )

    # ── Per-question cost sums to run total ───────────────────────────────────
    per_q_cost_sum = sum(q["cost_usd"] for q in result["per_question"])
    assert abs(per_q_cost_sum - result["total_cost_usd"]) < 1e-6, (
        f"Per-question cost sum {per_q_cost_sum:.6f} ≠ total {result['total_cost_usd']:.6f}"
    )

    # ── Result file written ───────────────────────────────────────────────────
    run_files = list(Path("results/runs").glob("*.json"))
    assert any(result["run_id"] in f.name for f in run_files), (
        "No result file written to results/runs/"
    )
