"""Matrix runner — iterates all chunking × embedding configurations.

Public API:
    run_matrix(config_path, output_dir, *, evaluator_fn, budget_usd, golden_path)
        -> MatrixResult

See ROADMAP Phase 9, ARCHITECTURE §3.3, ADR-015.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import structlog
from pydantic import BaseModel

from docintel.exceptions import BudgetExceededError
from docintel.utils.config import ExperimentConfig, load_config

log = structlog.get_logger()

# Mapping from full strategy name to 3-char abbreviation used in config_code
_STRATEGY_ABBREV: dict[str, str] = {
    "fixed": "fix",
    "recursive": "rec",
    "semantic": "sem",
}

# Mapping from full embedder name to short abbreviation used in config_code
_EMBEDDER_ABBREV: dict[str, str] = {
    "bge_small_en_v15": "bge",
    "minilm_l6_v2": "mini",
    "openai_text_embedding_3_small": "oai",
}


# ── Pydantic models ────────────────────────────────────────────────────────────


class MatrixConfigEntry(BaseModel):
    """Summary record for one completed (or skipped) config in the matrix."""

    model_config = {"extra": "ignore"}

    config_code: str
    run_id: str
    path: str
    aggregate_metrics: dict[str, float]


class MatrixResult(BaseModel):
    """Top-level record written to results/runs/{matrix_id}.json."""

    model_config = {"extra": "ignore"}

    matrix_id: str
    configs: list[MatrixConfigEntry]
    started_at: str
    ended_at: str
    total_cost_usd: float
    n_configs_completed: int
    n_configs_total: int
    status: Literal["complete", "aborted"]


# ── Public API ─────────────────────────────────────────────────────────────────


def run_matrix(
    config_path: str,
    output_dir: str,
    *,
    evaluator_fn: Callable[[ExperimentConfig, int, int], dict[str, Any]] | None = None,
    budget_usd: float | None = None,
    golden_path: str | None = None,
) -> MatrixResult:
    """Run the full experiment matrix and return a MatrixResult.

    Iterates all combinations of chunking × embeddings defined in the config
    (ADR-015: 3 × 2 = 6 configurations). For each, calls evaluator_fn, which
    returns an EvalResult dict. Aborts cleanly if accumulated spend exceeds
    budget_usd and writes a partial matrix JSON.

    Args:
        config_path:   Path to the experiment YAML config.
        output_dir:    Base output directory (e.g. "results/"). Matrix JSON is
                       written to {output_dir}/runs/{matrix_id}.json.
        evaluator_fn:  Callable(config, chunking_idx, embedding_idx) -> EvalResult dict.
                       Defaults to a wrapper around the Evaluator class.
                       THIS PARAMETER IS MANDATORY for tests — always inject a fake.
        budget_usd:    Hard cap on accumulated cost across all configs. Falls back to
                       config.evaluation.budget_usd when None.
        golden_path:   Path to the golden JSONL file passed to the default evaluator_fn.
                       Ignored when evaluator_fn is explicitly provided.

    Returns:
        MatrixResult with status "complete" or "aborted".

    Raises:
        RuntimeError: If a required index directory is missing before any run starts.
        BudgetExceededError: If accumulated cost exceeds budget_usd.
    """
    config = load_config(config_path)
    effective_budget = budget_usd if budget_usd is not None else config.evaluation.budget_usd

    if evaluator_fn is None:
        evaluator_fn = _make_default_evaluator_fn(golden_path)

    chunking_configs = config.chunking
    embedding_configs = config.embeddings
    n_total = len(chunking_configs) * len(embedding_configs)

    # ── Pre-flight: assert all index directories exist ────────────────────────
    _assert_index_dirs(chunking_configs, embedding_configs)

    # ── Setup ─────────────────────────────────────────────────────────────────
    matrix_id = _make_matrix_id()
    output_path = Path(output_dir) / "runs"
    output_path.mkdir(parents=True, exist_ok=True)

    import structlog.contextvars
    structlog.contextvars.bind_contextvars(matrix_id=matrix_id)

    started_at = datetime.now(timezone.utc)
    log.info("matrix_run_start", matrix_id=matrix_id, n_total=n_total)

    completed: list[MatrixConfigEntry] = []
    accumulated_cost: float = 0.0

    # ── Run each config ───────────────────────────────────────────────────────
    idx = 0
    for ci, chunk_cfg in enumerate(chunking_configs):
        for ei, embed_cfg in enumerate(embedding_configs):
            idx += 1
            strategy = chunk_cfg.name
            embedder = embed_cfg.name
            config_code = _make_config_code(strategy, embedder)

            log.info(
                "matrix_config_start",
                config_code=config_code,
                chunking_idx=ci,
                embedding_idx=ei,
                n=idx,
                n_total=n_total,
            )

            try:
                result = evaluator_fn(config, ci, ei)
            except BudgetExceededError as exc:
                log.error(
                    "matrix_run_aborted",
                    reason="evaluator_budget_exceeded",
                    config_code=config_code,
                    spent=exc.spent,
                )
                return _write_partial(
                    matrix_id, completed, n_total, accumulated_cost,
                    started_at, output_path,
                )

            run_id = result["run_id"]
            agg = result.get("aggregate_metrics", {})
            run_cost = float(result.get("total_cost_usd", 0.0))
            run_path = str(Path(output_dir) / "runs" / f"{run_id}.json")

            accumulated_cost += run_cost

            entry = MatrixConfigEntry(
                config_code=config_code,
                run_id=run_id,
                path=run_path,
                aggregate_metrics=agg,
            )
            completed.append(entry)

            # Live progress to stderr
            acc = float(agg.get("answer_accuracy", float("nan")))
            hall = float(agg.get("hallucination_rate", float("nan")))
            sys.stderr.write(
                f"[{idx}/{n_total}] {strategy}×{embedder}"
                f"  acc={acc:.3f}  halluc={hall:.3f}  ${run_cost:.4f}\n"
            )
            sys.stderr.flush()

            log.info(
                "matrix_config_complete",
                config_code=config_code,
                run_id=run_id,
                run_cost_usd=run_cost,
                accumulated_cost_usd=accumulated_cost,
            )

            # Budget guard: check after adding latest run's cost
            if accumulated_cost > effective_budget:
                log.error(
                    "matrix_run_aborted",
                    reason="accumulated_budget_exceeded",
                    accumulated_cost_usd=accumulated_cost,
                    budget_usd=effective_budget,
                )
                partial = _write_partial(
                    matrix_id, completed, n_total, accumulated_cost,
                    started_at, output_path,
                )
                raise BudgetExceededError(spent=accumulated_cost, budget=effective_budget)

    # ── All configs done ──────────────────────────────────────────────────────
    ended_at = datetime.now(timezone.utc)
    matrix_result = MatrixResult(
        matrix_id=matrix_id,
        configs=completed,
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        total_cost_usd=accumulated_cost,
        n_configs_completed=len(completed),
        n_configs_total=n_total,
        status="complete",
    )

    matrix_file = output_path / f"{matrix_id}.json"
    matrix_file.write_text(
        json.dumps(matrix_result.model_dump(), indent=2, ensure_ascii=False)
    )
    log.info(
        "matrix_run_complete",
        matrix_id=matrix_id,
        n_completed=len(completed),
        total_cost_usd=round(accumulated_cost, 4),
        path=str(matrix_file),
    )

    return matrix_result


# ── Internal helpers ───────────────────────────────────────────────────────────


def _make_matrix_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}_matrix"


def _make_config_code(strategy: str, embedder: str) -> str:
    s = _STRATEGY_ABBREV.get(strategy, strategy[:3])
    e = _EMBEDDER_ABBREV.get(embedder, embedder[:4])
    return f"{s}·{e}"


def _assert_index_dirs(chunking_configs: list, embedding_configs: list) -> None:
    """Raise RuntimeError if any required index directory is missing."""
    missing = []
    for chunk_cfg in chunking_configs:
        for embed_cfg in embedding_configs:
            index_dir = Path("data/indexes") / f"{chunk_cfg.name}_{embed_cfg.name}"
            if not index_dir.exists():
                missing.append(str(index_dir))

    if missing:
        dirs_str = "\n  ".join(missing)
        raise RuntimeError(
            f"Missing index director{'y' if len(missing)==1 else 'ies'} "
            f"(run `make index` to build them):\n  {dirs_str}"
        )


def _write_partial(
    matrix_id: str,
    completed: list[MatrixConfigEntry],
    n_total: int,
    accumulated_cost: float,
    started_at: datetime,
    output_path: Path,
) -> MatrixResult:
    ended_at = datetime.now(timezone.utc)
    partial = MatrixResult(
        matrix_id=matrix_id,
        configs=completed,
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        total_cost_usd=accumulated_cost,
        n_configs_completed=len(completed),
        n_configs_total=n_total,
        status="aborted",
    )
    partial_file = output_path / f"{matrix_id}.json"
    partial_file.write_text(
        json.dumps(partial.model_dump(), indent=2, ensure_ascii=False)
    )
    log.warning(
        "matrix_partial_saved",
        matrix_id=matrix_id,
        n_completed=len(completed),
        path=str(partial_file),
    )
    return partial


def _make_default_evaluator_fn(
    golden_path: str | None,
) -> Callable[[ExperimentConfig, int, int], dict[str, Any]]:
    """Build the default evaluator_fn using the Evaluator class."""

    def _fn(config: ExperimentConfig, chunking_idx: int, embedding_idx: int) -> dict[str, Any]:
        from docintel.evaluation.evaluator import Evaluator, _build_pipeline_and_metrics

        kw: dict[str, Any] = {}
        if golden_path is not None:
            kw["golden_path"] = Path(golden_path)

        pipeline, metrics, judge, strategy = _build_pipeline_and_metrics(
            config, chunking_idx, embedding_idx
        )
        ev = Evaluator(
            pipeline=pipeline,
            metrics=metrics,
            judge=judge,
            config=config,
            strategy=strategy,
            **kw,
        )
        return ev.run()

    return _fn
