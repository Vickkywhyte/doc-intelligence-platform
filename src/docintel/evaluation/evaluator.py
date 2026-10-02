"""Evaluation orchestrator — EVAL_SPEC §6, DATA_MODEL §2.5.

Runs one (chunking, embedding) configuration over the full golden dataset and
produces a complete EvalResult written to results/runs/{run_id}.json.

CLI entry point:
    python -m docintel.evaluation.evaluator [--config path] [--chunking-index N] [--embedding-index N]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import structlog

from docintel.evaluation.judge import JudgeClient
from docintel.evaluation.metrics import (
    AnswerAccuracyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    RefusalMetric,
    RetrievalPrecisionMetric,
)
from docintel.evaluation.models import GoldenItem, QuestionResult
from docintel.evaluation.stats import bootstrap_ci
from docintel.exceptions import BudgetExceededError
from docintel.generation.llm_client import LLMClient
from docintel.generation.models import RAGResponse
from docintel.pipeline.rag_pipeline import RAGPipeline
from docintel.utils.config import EvaluationConfig, ExperimentConfig
from docintel.utils.cost_tracker import cost_tracker

log = structlog.get_logger()

_GOLDEN_PATH = Path("data/eval/golden.jsonl")
_RESULTS_DIR = Path("results/runs")

# Metric keys that contribute to the headline aggregate_ci table
_CI_METRICS = frozenset({
    "p_at_1", "p_at_3", "p_at_5",
    "recall_at_1", "recall_at_3", "recall_at_5",
    "mrr", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5",
    "sem_sim", "answer_accuracy",
    "hallucination_rate", "faithfulness",
    "refusal_correct",
})


class Evaluator:
    """Orchestrates a single-config evaluation run.

    Usage:
        evaluator = Evaluator(pipeline, metrics, judge, config, strategy=strategy)
        result = evaluator.run()
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        metrics: list,
        judge: JudgeClient,
        config: ExperimentConfig,
        *,
        strategy: str,
        golden_path: Path = _GOLDEN_PATH,
    ) -> None:
        self._pipeline = pipeline
        self._metrics = metrics
        self._judge = judge
        self._config = config
        self._strategy = strategy
        self._golden_path = golden_path
        # Locate hallucination/faithfulness metrics for side-channel access
        self._hallucination_m = next(
            (m for m in metrics if isinstance(m, HallucinationMetric)), None
        )
        self._faithfulness_m = next(
            (m for m in metrics if isinstance(m, FaithfulnessMetric)), None
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Run the full evaluation and return the EvalResult as a dict."""
        items = self._load_golden()
        run_id = self._make_run_id()
        config_hash = self._config.hash()

        cost_tracker.reset_run(run_id, self._config.evaluation.budget_usd)
        import structlog.contextvars
        structlog.contextvars.bind_contextvars(run_id=run_id)

        log.info("eval_run_start", run_id=run_id, n_questions=len(items), strategy=self._strategy)
        started_at = datetime.now(timezone.utc)

        per_question: list[QuestionResult] = []
        try:
            for item in items:
                qr = self._eval_one(item)
                per_question.append(qr)
        except BudgetExceededError as exc:
            log.error("eval_run_aborted", reason="budget_exceeded", spent=exc.spent)
            # Persist partial results
            self._save_partial(run_id, per_question)
            raise

        ended_at = datetime.now(timezone.utc)
        total_cost_usd = cost_tracker.run_total

        aggregate_metrics, aggregate_ci = self._aggregate(per_question, total_cost_usd)

        result = self._build_eval_result(
            run_id=run_id,
            config_hash=config_hash,
            started_at=started_at,
            ended_at=ended_at,
            per_question=per_question,
            aggregate_metrics=aggregate_metrics,
            aggregate_ci=aggregate_ci,
            total_cost_usd=total_cost_usd,
        )

        self._save(run_id, result)
        log.info(
            "eval_run_complete",
            run_id=run_id,
            n_questions=result["n_questions"],
            total_cost_usd=round(total_cost_usd, 4),
        )
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_golden(self) -> list[GoldenItem]:
        lines = self._golden_path.read_text().splitlines()
        items = []
        for line in lines:
            line = line.strip()
            if line:
                items.append(GoldenItem.model_validate(json.loads(line)))
        return items

    def _eval_one(self, item: GoldenItem) -> QuestionResult:
        log.debug("eval_question_start", item_id=item.id, type=item.type)
        t0 = time.monotonic()

        response: RAGResponse = self._pipeline.query(item.question)

        scores: dict[str, float] = {}
        for metric in self._metrics:
            try:
                result = metric.compute(item, response, strategy=self._strategy)
                scores.update(result)
            except Exception as exc:
                log.warning(
                    "metric_compute_failed",
                    metric=metric.name,
                    item_id=item.id,
                    error=str(exc),
                )

        # Collect hallucinated claims from side-channel
        hallucinated_claims: list[str] = []
        if self._hallucination_m is not None:
            hallucinated_claims = list(self._hallucination_m.last_hallucinated)

        # Build judge_verdict summary
        judge_verdict: dict[str, Any] = {}
        if self._hallucination_m is not None and self._hallucination_m.last_warning:
            judge_verdict["warning"] = self._hallucination_m.last_warning

        total_latency_ms = round((time.monotonic() - t0) * 1000)

        return QuestionResult(
            item_id=item.id,
            question=item.question,
            answer=response.answer,
            refused=response.refused,
            retrieved_chunk_ids=[r.chunk.chunk_id for r in response.retrieved],
            metrics=scores,
            hallucinated_claims=hallucinated_claims,
            judge_verdict=judge_verdict,
            cost_usd=response.cost_usd,
            latency_ms=total_latency_ms,
        )

    def _aggregate(
        self,
        per_question: list[QuestionResult],
        total_cost_usd: float,
    ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        """Compute aggregate metrics and bootstrap CIs."""
        # Collect per-metric score lists
        per_metric: dict[str, list[float]] = {}
        for qr in per_question:
            for key, val in qr.metrics.items():
                per_metric.setdefault(key, []).append(val)

        n = len(per_question)
        agg: dict[str, float] = {}
        ci: dict[str, tuple[float, float]] = {}
        bs = self._config.evaluation.bootstrap_samples
        seed = self._config.evaluation.seed

        for key, scores in per_metric.items():
            agg[key] = float(np.mean(scores))
            if key in _CI_METRICS and len(scores) > 1:
                ci[key] = bootstrap_ci(scores, n_samples=bs, seed=seed)

        # Rename refusal_correct → refusal_correctness in aggregates
        if "refusal_correct" in agg:
            agg["refusal_correctness"] = agg.pop("refusal_correct")
            if "refusal_correct" in ci:
                ci["refusal_correctness"] = ci.pop("refusal_correct")

        # Operational metrics (no CI)
        latencies = [qr.latency_ms for qr in per_question]
        if latencies:
            agg["latency_ms_p50"] = float(median(latencies))

        if n > 0:
            agg["cost_per_1k_q_usd"] = (total_cost_usd / n) * 1000.0

        return agg, ci

    def _make_run_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        cfg_hash = self._config.hash()[:8]
        return f"{ts}_{cfg_hash}"

    def _build_eval_result(
        self,
        *,
        run_id: str,
        config_hash: str,
        started_at: datetime,
        ended_at: datetime,
        per_question: list[QuestionResult],
        aggregate_metrics: dict[str, float],
        aggregate_ci: dict[str, tuple[float, float]],
        total_cost_usd: float,
    ) -> dict[str, Any]:
        git_sha = _git_sha()
        return {
            "run_id": run_id,
            "config_hash": config_hash,
            "config": self._config.model_dump(mode="json"),
            "git_sha": git_sha,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "package_versions": _package_versions(),
            "judge_model": self._config.evaluation.judge_model,
            "judge_prompt_version": self._config.evaluation.judge_prompt_version,
            "seed": self._config.evaluation.seed,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "n_questions": len(per_question),
            "aggregate_metrics": aggregate_metrics,
            "aggregate_ci": {k: list(v) for k, v in aggregate_ci.items()},
            "per_question": [qr.model_dump() for qr in per_question],
            "total_cost_usd": total_cost_usd,
            "notes": None,
        }

    def _save(self, run_id: str, result: dict[str, Any]) -> None:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = _RESULTS_DIR / f"{run_id}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        log.info("eval_result_saved", path=str(path))
        print(f"\nResult written to: {path}")

    def _save_partial(self, run_id: str, per_question: list[QuestionResult]) -> None:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = _RESULTS_DIR / f"{run_id}.partial.json"
        data = [qr.model_dump() for qr in per_question]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        log.warning("eval_partial_saved", path=str(path), n=len(data))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    pkgs = ["pydantic", "chromadb", "sentence-transformers", "openai", "numpy"]
    result: dict[str, str] = {}
    for pkg in pkgs:
        try:
            result[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


# ── CLI entry-point ───────────────────────────────────────────────────────────


def _build_pipeline_and_metrics(
    config: ExperimentConfig,
    chunking_index: int,
    embedding_index: int,
) -> tuple[RAGPipeline, list, JudgeClient, str]:
    """Instantiate pipeline components from config for a single run.

    Returns (pipeline, metrics_list, judge, strategy_name).
    """
    from docintel.embeddings.registry import get_embedder
    from docintel.generation.llm_client import OpenRouterClient
    from docintel.retrieval.retriever import DenseRetriever
    from docintel.utils.config import GenerationConfig
    from docintel.vectorstore.chroma_store import ChromaStore

    chunking_cfg = config.chunking[chunking_index]
    embedding_cfg = config.embeddings[embedding_index]
    strategy = chunking_cfg.name

    print(f"Strategy : {strategy}")
    print(f"Embedder : {embedding_cfg.name}")
    print(f"Judge    : {config.evaluation.judge_model}")

    embedder = get_embedder(embedding_cfg)
    store = ChromaStore(strategy=strategy, embedder_name=embedding_cfg.name)
    retriever = DenseRetriever(store, embedder, config.retrieval)

    gen_llm = OpenRouterClient(config.generation)
    pipeline = RAGPipeline(retriever, gen_llm, k=config.retrieval.k)

    judge_gen_cfg = GenerationConfig(
        model=config.evaluation.judge_model,
        temperature=0.0,
        max_tokens=512,
    )
    judge_llm: LLMClient = OpenRouterClient(judge_gen_cfg)
    judge = JudgeClient(judge_llm, config.evaluation)

    metrics: list = [
        RetrievalPrecisionMetric(k_values=config.evaluation.k_values),
        AnswerAccuracyMetric(embedder, judge),
        HallucinationMetric(judge, config.evaluation),
        FaithfulnessMetric(judge),
        RefusalMetric(),
    ]

    return pipeline, metrics, judge, strategy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run single-config evaluation over the golden set")
    parser.add_argument("--config", default=None, help="Path to experiment config YAML")
    parser.add_argument("--chunking-index", type=int, default=0, help="Index into config.chunking list")
    parser.add_argument("--embedding-index", type=int, default=0, help="Index into config.embeddings list")
    parser.add_argument("--golden", default=str(_GOLDEN_PATH), help="Path to golden JSONL file")
    args = parser.parse_args(argv)

    from docintel.utils.config import load_config
    from docintel.utils.logging import configure_logging
    configure_logging()

    config = load_config(args.config)

    if args.chunking_index >= len(config.chunking):
        print(f"ERROR: chunking-index {args.chunking_index} out of range (max {len(config.chunking)-1})", file=sys.stderr)
        return 1
    if args.embedding_index >= len(config.embeddings):
        print(f"ERROR: embedding-index {args.embedding_index} out of range (max {len(config.embeddings)-1})", file=sys.stderr)
        return 1

    pipeline, metrics, judge, strategy = _build_pipeline_and_metrics(
        config, args.chunking_index, args.embedding_index
    )

    evaluator = Evaluator(
        pipeline=pipeline,
        metrics=metrics,
        judge=judge,
        config=config,
        strategy=strategy,
        golden_path=Path(args.golden),
    )

    result = evaluator.run()
    print(f"\nAggregate metrics:")
    for k, v in sorted(result["aggregate_metrics"].items()):
        print(f"  {k:<28} {v:.4f}")
    print(f"\nTotal questions : {result['n_questions']}")
    print(f"Total cost      : ${result['total_cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
