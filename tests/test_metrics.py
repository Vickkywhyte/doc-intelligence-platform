"""Phase 8 — Unit tests for all evaluation metrics.

Tests assert correctness at 1e-6 tolerance against hand-crafted fixtures
from EVAL_SPEC §2.5, §4.5, §5.4, §8 and tests/fixtures/metrics_cases.json.

No real LLM calls are made — the judge is fed through FakeLLMClient.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.chunking.base import Chunk
from docintel.evaluation.judge import JudgeClient
from docintel.evaluation.metrics.answer_accuracy import AnswerAccuracyMetric, _cosine_sim
from docintel.evaluation.metrics.faithfulness import FaithfulnessMetric, _faithfulness_score
from docintel.evaluation.metrics.hallucination import HallucinationMetric, _hallucination_rate
from docintel.evaluation.metrics.refusal import RefusalMetric
from docintel.evaluation.metrics.retrieval_precision import (
    RetrievalPrecisionMetric,
    _mrr,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
)
from docintel.evaluation.models import GoldenItem
from docintel.evaluation.stats import bootstrap_ci
from docintel.generation.models import Citation, RAGResponse
from docintel.utils.config import EvaluationConfig
from docintel.vectorstore.base import RetrievedChunk
from tests.mocks.embedder import FakeEmbedder
from tests.mocks.llm import FakeLLMClient

_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "metrics_cases.json").read_text()
)
_TOL = 1e-6

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_golden(
    type: str = "factual",
    strategy: str = "fixed",
    gold_chunks: list[str] | None = None,
) -> GoldenItem:
    chunks = gold_chunks if gold_chunks is not None else ["doc1:fixed:00000"]
    chunk_ids = {} if type == "unanswerable" else {strategy: chunks}
    return GoldenItem(
        id="test_g001",
        question="What is the notice period?",
        gold_answer="30 days written notice.",
        gold_chunk_ids=chunk_ids,
        type=type,
    )


def _make_chunk(chunk_id: str) -> Chunk:
    doc_id, strategy, idx = chunk_id.split(":")
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        strategy=strategy,
        index=int(idx),
        text=f"Text for {chunk_id}.",
        page=1,
        char_start=0,
        char_end=50,
        token_count=10,
    )


def _make_retrieved(chunk_ids: list[str]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=_make_chunk(cid),
            score=1.0 - (i * 0.1),
            rank=i + 1,
            retriever="topk",
        )
        for i, cid in enumerate(chunk_ids)
    ]


def _make_response(
    chunk_ids: list[str],
    answer: str = "30 days written notice.",
    refused: bool = False,
) -> RAGResponse:
    return RAGResponse(
        query="What is the notice period?",
        answer=answer,
        refused=refused,
        citations=[],
        retrieved=_make_retrieved(chunk_ids),
        model="fake",
        latency_ms=0,
        cost_usd=0.0,
    )


def _eval_cfg(**kwargs) -> EvaluationConfig:
    defaults = dict(
        judge_model="fake",
        judge_prompt_version="v1",
        metrics=["p_at_k", "answer_accuracy", "hallucination", "faithfulness"],
        k_values=[1, 3, 5],
        bootstrap_samples=100,
        budget_usd=5.0,
        seed=0,
        partial_claim_weight=0.5,
    )
    defaults.update(kwargs)
    return EvaluationConfig(**defaults)


def _make_judge(llm_responses: list[str]) -> JudgeClient:
    """JudgeClient backed by FakeLLMClient with scripted responses."""
    llm = FakeLLMClient(llm_responses)
    cfg = _eval_cfg()
    # Use a temp dir so tests don't pollute the real cache
    import tempfile

    cache_dir = Path(tempfile.mkdtemp())
    return JudgeClient(llm, cfg, cache_dir=cache_dir)


# ── Retrieval math functions (direct) ────────────────────────────────────────


class TestPrecisionAtK:
    def test_perfect(self):
        assert abs(_precision_at_k(["c1", "c2", "c3"], {"c1"}, 3) - 1 / 3) < _TOL

    def test_zero(self):
        assert _precision_at_k(["c1", "c2"], {"c9"}, 2) == 0.0

    def test_full_hit(self):
        assert _precision_at_k(["c1"], {"c1"}, 1) == 1.0


class TestRecallAtK:
    def test_full_recall(self):
        assert _recall_at_k(["c1", "c2"], {"c1"}, 2) == 1.0

    def test_partial_recall(self):
        assert abs(_recall_at_k(["c1", "c2"], {"c1", "c3"}, 2) - 0.5) < _TOL

    def test_zero(self):
        assert _recall_at_k(["c1"], {"c9"}, 1) == 0.0


class TestMRR:
    def test_first_rank(self):
        assert _mrr(["c1", "c2"], {"c1"}) == 1.0

    def test_rank_4(self):
        assert abs(_mrr(["c1", "c2", "c3", "c4"], {"c4"}) - 0.25) < _TOL

    def test_not_found(self):
        assert _mrr(["c1", "c2"], {"c9"}) == 0.0


class TestNDCG:
    def test_perfect(self):
        assert abs(_ndcg_at_k(["c1", "c2", "c3"], {"c1"}, 3) - 1.0) < _TOL

    def test_gold_at_rank_4(self):
        expected = 1.0 / math.log2(5)
        assert abs(_ndcg_at_k(["c1", "c2", "c3", "c4", "c5"], {"c4"}, 5) - expected) < _TOL

    def test_empty_gold_returns_zero(self):
        assert _ndcg_at_k(["c1", "c2"], set(), 2) == 0.0

    def test_multi_gold(self):
        # G={c1,c3}, R=[c1,c2,c3,c4,c5], k=5 → NDCG=0.9197...
        expected = 1.5 / (1.0 + 1.0 / math.log2(3))
        result = _ndcg_at_k(["c1", "c2", "c3", "c4", "c5"], {"c1", "c3"}, 5)
        assert abs(result - expected) < _TOL


# ── RetrievalPrecisionMetric end-to-end with fixtures ────────────────────────


class TestRetrievalPrecisionMetric:
    def _chunks_from_ids(self, ids: list[str]) -> list[str]:
        """Convert bare IDs like 'c1' to valid chunk_id format."""
        return [f"doc:{i}:{idx:05d}" for idx, i in enumerate(ids)]

    @pytest.fixture(autouse=True)
    def setup(self):
        self.metric = RetrievalPrecisionMetric(k_values=[1, 3, 5])

    def _run(self, gold_ids: list[str], retrieved_ids: list[str], strategy: str = "fixed"):
        item = _make_golden(
            strategy=strategy,
            gold_chunks=gold_ids,
        )
        response = _make_response(retrieved_ids)
        return self.metric.compute(item, response, strategy=strategy)

    def test_perfect_retrieval(self):
        gold = ["doc:1:00001"]
        retrieved = ["doc:1:00001", "doc:2:00002", "doc:3:00003", "doc:4:00004", "doc:5:00005"]
        scores = self._run(gold, retrieved)
        assert abs(scores["p_at_5"] - 0.2) < _TOL
        assert abs(scores["mrr"] - 1.0) < _TOL
        assert abs(scores["ndcg_at_5"] - 1.0) < _TOL

    def test_gold_at_rank_4(self):
        gold = ["doc:4:00004"]
        retrieved = [
            "doc:1:00001", "doc:2:00002", "doc:3:00003",
            "doc:4:00004", "doc:5:00005",
        ]
        scores = self._run(gold, retrieved)
        assert abs(scores["mrr"] - 0.25) < _TOL
        assert abs(scores["ndcg_at_5"] - (1.0 / math.log2(5))) < _TOL
        assert scores["ndcg_at_3"] == 0.0

    def test_multi_gold(self):
        gold = ["doc:1:00001", "doc:3:00003"]
        retrieved = [
            "doc:1:00001", "doc:2:00002", "doc:3:00003",
            "doc:4:00004", "doc:5:00005",
        ]
        scores = self._run(gold, retrieved)
        assert abs(scores["p_at_5"] - 0.4) < _TOL
        assert abs(scores["recall_at_5"] - 1.0) < _TOL
        assert abs(scores["mrr"] - 1.0) < _TOL

    def test_no_gold_retrieved(self):
        gold = ["doc:9:00009"]
        retrieved = [
            "doc:1:00001", "doc:2:00002", "doc:3:00003",
            "doc:4:00004", "doc:5:00005",
        ]
        scores = self._run(gold, retrieved)
        assert scores["p_at_5"] == 0.0
        assert scores["mrr"] == 0.0
        assert scores["ndcg_at_5"] == 0.0

    def test_unanswerable_returns_empty(self):
        item = _make_golden(type="unanswerable")
        response = _make_response(["doc:1:00001"])
        scores = self.metric.compute(item, response, strategy="fixed")
        assert scores == {}


# ── JSON fixture parametrize test ────────────────────────────────────────────


@pytest.mark.parametrize("name,case", _FIXTURES["retrieval"].items())
def test_retrieval_fixture(name: str, case: dict):
    """Assert retrieval metrics match hand-computed values from metrics_cases.json."""
    metric = RetrievalPrecisionMetric(k_values=[1, 3, 5])

    if not case["gold"]:
        # unanswerable — just verify empty return
        item = _make_golden(type="unanswerable")
        response = _make_response([f"doc:x:{i:05d}" for i in range(5)])
        assert metric.compute(item, response, strategy="fixed") == {}
        return

    # Map short IDs "c1"..."c5" to valid chunk_id format
    id_map = {f"c{i}": f"doc:x:{i:05d}" for i in range(1, 10)}
    gold = [id_map.get(g, g) for g in case["gold"]]
    retrieved_ids = [id_map.get(r, r) for r in case["retrieved"]]

    item = _make_golden(strategy="fixed", gold_chunks=gold)
    response = _make_response(retrieved_ids)
    scores = metric.compute(item, response, strategy="fixed")

    for key, expected in case["expected"].items():
        assert key in scores, f"[{name}] missing key {key}"
        assert abs(scores[key] - expected) < _TOL, (
            f"[{name}] {key}: expected {expected}, got {scores[key]}"
        )


# ── Hallucination math (direct) ──────────────────────────────────────────────


class TestHallucinationRate:
    @pytest.mark.parametrize("case_name,case", _FIXTURES["hallucination"].items())
    def test_fixtures(self, case_name: str, case: dict):
        rate = _hallucination_rate(case["verdicts"], case["partial_weight"])
        assert abs(rate - case["expected_rate"]) < _TOL, (
            f"[{case_name}] expected {case['expected_rate']}, got {rate}"
        )

    def test_partial_weight_configurable(self):
        # With weight=1.0, partial counts as full hallucination
        assert abs(_hallucination_rate(["partial"], 1.0) - 1.0) < _TOL
        # With weight=0.0, partial counts as clean
        assert abs(_hallucination_rate(["partial"], 0.0) - 0.0) < _TOL

    def test_mixed_verdicts(self):
        # ["yes", "no", "partial"], weight=0.5 → (1 + 0.5) / 3 = 0.5
        result = _hallucination_rate(["yes", "no", "partial"], 0.5)
        assert abs(result - 0.5) < _TOL


# ── Faithfulness math (direct) ────────────────────────────────────────────────


class TestFaithfulnessScore:
    @pytest.mark.parametrize("case_name,case", _FIXTURES["faithfulness"].items())
    def test_fixtures(self, case_name: str, case: dict):
        score = _faithfulness_score(case["verdicts"])
        assert abs(score - case["expected"]) < _TOL, (
            f"[{case_name}] expected {case['expected']}, got {score}"
        )

    def test_partial_counts_as_supported(self):
        assert abs(_faithfulness_score(["partial"]) - 1.0) < _TOL

    def test_all_no_returns_zero(self):
        assert _faithfulness_score(["no", "no"]) == 0.0


# ── HallucinationMetric with mocked judge ────────────────────────────────────


class TestHallucinationMetric:
    def test_clean_answer(self):
        claims_resp = '{"claims": ["Notice is 30 days."]}'
        entail_resp = '{"verdict": "yes", "chunk_id": "doc:1:00000"}'
        judge = _make_judge([claims_resp, entail_resp])
        cfg = _eval_cfg()
        metric = HallucinationMetric(judge, cfg)

        item = _make_golden()
        response = _make_response(["doc:1:00000"])
        scores = metric.compute(item, response, strategy="fixed")
        assert abs(scores["hallucination_rate"] - 0.0) < _TOL
        assert metric.last_hallucinated == []

    def test_fabricated_claim(self):
        claims_resp = '{"claims": ["Notice is 60 days."]}'
        entail_resp = '{"verdict": "no", "chunk_id": null}'
        judge = _make_judge([claims_resp, entail_resp])
        cfg = _eval_cfg()
        metric = HallucinationMetric(judge, cfg)

        item = _make_golden()
        response = _make_response(["doc:1:00000"], answer="Notice is 60 days.")
        scores = metric.compute(item, response, strategy="fixed")
        assert abs(scores["hallucination_rate"] - 1.0) < _TOL
        assert len(metric.last_hallucinated) == 1

    def test_refusal_returns_zero(self):
        judge = _make_judge([])
        cfg = _eval_cfg()
        metric = HallucinationMetric(judge, cfg)

        item = _make_golden()
        response = _make_response([], answer="I don't know based on the provided documents.", refused=True)
        scores = metric.compute(item, response, strategy="fixed")
        assert scores["hallucination_rate"] == 0.0

    def test_empty_answer_returns_zero_and_flags(self):
        judge = _make_judge([])
        cfg = _eval_cfg()
        metric = HallucinationMetric(judge, cfg)

        item = _make_golden()
        response = _make_response([], answer="")
        scores = metric.compute(item, response, strategy="fixed")
        assert scores["hallucination_rate"] == 0.0
        assert metric.last_warning == "zero_claims"

    def test_partial_claim(self):
        claims_resp = '{"claims": ["Notice is 30 days, effective immediately."]}'
        entail_resp = '{"verdict": "partial", "chunk_id": "doc:1:00000"}'
        judge = _make_judge([claims_resp, entail_resp])
        cfg = _eval_cfg(partial_claim_weight=0.5)
        metric = HallucinationMetric(judge, cfg)

        item = _make_golden()
        response = _make_response(["doc:1:00000"], answer="Notice is 30 days, effective immediately.")
        scores = metric.compute(item, response, strategy="fixed")
        assert abs(scores["hallucination_rate"] - 0.5) < _TOL


# ── FaithfulnessMetric with mocked judge ─────────────────────────────────────


class TestFaithfulnessMetric:
    def test_paraphrase_is_fully_faithful(self):
        claims_resp = '{"claims": ["Either side can end the deal with a month notice."]}'
        entail_resp = '{"verdict": "yes", "chunk_id": "doc:1:00000"}'
        judge = _make_judge([claims_resp, entail_resp])
        metric = FaithfulnessMetric(judge)

        item = _make_golden()
        response = _make_response(
            ["doc:1:00000"],
            answer="Either side can end the deal with a month notice.",
        )
        scores = metric.compute(item, response, strategy="fixed")
        assert abs(scores["faithfulness"] - 1.0) < _TOL

    def test_unsupported_extra(self):
        claims_resp = '{"claims": ["Notice is 30 days.", "Payment is due on signing."]}'
        # Two entailment calls: first yes, second no
        judge = _make_judge([
            claims_resp,
            '{"verdict": "yes", "chunk_id": "doc:1:00000"}',
            '{"verdict": "no", "chunk_id": null}',
        ])
        metric = FaithfulnessMetric(judge)

        item = _make_golden()
        response = _make_response(
            ["doc:1:00000"],
            answer="Notice is 30 days. Payment is due on signing.",
        )
        scores = metric.compute(item, response, strategy="fixed")
        assert abs(scores["faithfulness"] - 0.5) < _TOL


# ── RefusalMetric ─────────────────────────────────────────────────────────────


class TestRefusalMetric:
    def setup_method(self):
        self.metric = RefusalMetric()

    def test_correct_refusal(self):
        item = _make_golden(type="unanswerable")
        response = _make_response([], refused=True)
        scores = self.metric.compute(item, response, strategy="fixed")
        assert scores == {"refusal_correct": 1.0}

    def test_incorrect_refusal(self):
        item = _make_golden(type="unanswerable")
        response = _make_response([], refused=False, answer="30 days notice.")
        scores = self.metric.compute(item, response, strategy="fixed")
        assert scores == {"refusal_correct": 0.0}

    def test_factual_returns_empty(self):
        item = _make_golden(type="factual")
        response = _make_response(["doc:1:00000"])
        scores = self.metric.compute(item, response, strategy="fixed")
        assert scores == {}

    def test_multi_hop_returns_empty(self):
        item = _make_golden(type="multi_hop", gold_chunks=["doc:1:00000"])
        response = _make_response(["doc:1:00000"])
        scores = self.metric.compute(item, response, strategy="fixed")
        assert scores == {}


# ── AnswerAccuracyMetric with mocked judge ────────────────────────────────────


class TestAnswerAccuracyMetric:
    def _metric(self, judge_score: int = 5) -> AnswerAccuracyMetric:
        verdict = json.dumps({"score": judge_score, "rationale": "test"})
        judge = _make_judge([verdict])
        embedder = FakeEmbedder()
        return AnswerAccuracyMetric(embedder, judge)

    def test_perfect_answer_gives_one(self):
        metric = self._metric(judge_score=5)
        item = _make_golden()
        response = _make_response(["doc:1:00000"])
        scores = metric.compute(item, response, strategy="fixed")
        assert abs(scores["answer_accuracy"] - 1.0) < _TOL

    def test_wrong_answer_gives_low_score(self):
        metric = self._metric(judge_score=1)
        item = _make_golden()
        response = _make_response(["doc:1:00000"])
        scores = metric.compute(item, response, strategy="fixed")
        assert abs(scores["answer_accuracy"] - 0.2) < _TOL

    def test_unanswerable_returns_empty(self):
        metric = self._metric()
        item = _make_golden(type="unanswerable")
        response = _make_response([], refused=True)
        scores = metric.compute(item, response, strategy="fixed")
        assert scores == {}

    def test_sem_sim_is_in_unit_interval(self):
        metric = self._metric()
        item = _make_golden()
        response = _make_response(["doc:1:00000"])
        scores = metric.compute(item, response, strategy="fixed")
        assert 0.0 <= scores["sem_sim"] <= 1.0


# ── AnswerAccuracy: sem_sim direct test ──────────────────────────────────────


class TestCosineSim:
    def test_identical_vectors_give_one(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(_cosine_sim(v, v) - 1.0) < _TOL

    def test_orthogonal_vectors_give_zero(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(_cosine_sim(v1, v2)) < _TOL


# ── Bootstrap CI ─────────────────────────────────────────────────────────────


class TestBootstrapCI:
    def test_constant_distribution(self):
        scores = [0.5] * 100
        lo, hi = bootstrap_ci(scores, n_samples=500, seed=42)
        assert abs(lo - 0.5) < 1e-9
        assert abs(hi - 0.5) < 1e-9

    def test_empty_scores(self):
        assert bootstrap_ci([]) == (0.0, 0.0)

    def test_ci_is_ordered(self):
        scores = [float(x) / 10 for x in range(10)]
        lo, hi = bootstrap_ci(scores, n_samples=1000, seed=0)
        assert lo <= hi

    def test_ci_contains_mean(self):
        import random
        random.seed(7)
        scores = [random.gauss(0.7, 0.1) for _ in range(50)]
        lo, hi = bootstrap_ci(scores, n_samples=1000, seed=0)
        mean = sum(scores) / len(scores)
        assert lo <= mean <= hi

    def test_deterministic_with_seed(self):
        scores = [0.1 * i for i in range(10)]
        r1 = bootstrap_ci(scores, n_samples=500, seed=99)
        r2 = bootstrap_ci(scores, n_samples=500, seed=99)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        scores = [0.1 * i for i in range(10)]
        r1 = bootstrap_ci(scores, n_samples=500, seed=1)
        r2 = bootstrap_ci(scores, n_samples=500, seed=2)
        # Not guaranteed, but very likely for a non-trivial distribution
        # Just verify both are valid intervals
        assert r1[0] <= r1[1]
        assert r2[0] <= r2[1]

    def test_bootstrap_ci_accepts_numpy_array(self):
        arr = np.array([0.9, 0.8, 0.85, 0.7, 0.95])
        lo, hi = bootstrap_ci(arr)
        assert lo <= hi


# ── Judge cache ───────────────────────────────────────────────────────────────


class TestJudgeCache:
    def test_same_prompt_served_from_cache(self, tmp_path):
        llm = FakeLLMClient(['{"score": 5, "rationale": "perfect"}'])
        cfg = _eval_cfg()
        judge = JudgeClient(llm, cfg, cache_dir=tmp_path)

        result1 = judge.judge_correctness("Q?", "A gold.", "A ans.")
        assert llm.call_count == 1

        # Second call — same inputs → served from cache, no additional LLM call
        result2 = judge.judge_correctness("Q?", "A gold.", "A ans.")
        assert llm.call_count == 1  # still 1
        assert result1 == result2

    def test_different_prompts_call_llm(self, tmp_path):
        llm = FakeLLMClient([
            '{"score": 5, "rationale": "perfect"}',
            '{"score": 3, "rationale": "partial"}',
        ])
        cfg = _eval_cfg()
        judge = JudgeClient(llm, cfg, cache_dir=tmp_path)

        judge.judge_correctness("Q1?", "A1 gold.", "A1 ans.")
        judge.judge_correctness("Q2?", "A2 gold.", "A2 ans.")
        assert llm.call_count == 2

    def test_cache_key_deterministic(self, tmp_path):
        llm = FakeLLMClient(['{"score": 4, "rationale": "good"}'])
        cfg = _eval_cfg()
        j1 = JudgeClient(llm, cfg, cache_dir=tmp_path)
        j2 = JudgeClient(llm, cfg, cache_dir=tmp_path)

        j1.judge_correctness("What?", "Gold ans.", "Gen ans.")
        # j2 uses same cache dir — should find the cached result
        j2.judge_correctness("What?", "Gold ans.", "Gen ans.")
        assert llm.call_count == 1
