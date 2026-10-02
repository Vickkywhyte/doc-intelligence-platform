"""Phase 9 — Run the full experiment matrix over the golden dataset.

Usage:
    python scripts/07_run_eval_matrix.py [options]

Options:
    --config      Path to experiment YAML (default: configs/experiment_matrix.yaml)
    --output-dir  Base output directory   (default: results/)
    --budget      Hard USD cap on total spend (default: from config)
    --golden      Path to golden JSONL     (default: data/eval/golden.jsonl)

Writes:
    results/runs/{matrix_id}.json  — MatrixResult with 6 config entries
    results/runs/{run_id}.json     — one EvalResult per config (written by evaluator)

Exit codes:
    0  — complete
    1  — aborted (budget exceeded or other error)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.exceptions import BudgetExceededError
from docintel.pipeline.experiment import run_matrix
from docintel.utils.logging import configure_logging


def _print_summary_table(matrix_result) -> None:
    """Print a compact summary table to stdout."""
    print("\nMatrix summary")
    print("-" * 72)
    print(f"{'Config':<12}  {'Acc':>6}  {'Halluc':>8}  {'MRR':>6}  {'Cost/1k':>8}")
    print("-" * 72)
    for cfg in matrix_result.configs:
        agg = cfg.aggregate_metrics
        acc = agg.get("answer_accuracy", float("nan"))
        hall = agg.get("hallucination_rate", float("nan"))
        mrr = agg.get("mrr", float("nan"))
        cost = agg.get("cost_per_1k_q_usd", float("nan"))
        print(f"{cfg.config_code:<12}  {acc:>6.3f}  {hall:>8.3f}  {mrr:>6.3f}  {cost:>8.4f}")
    print("-" * 72)
    print(
        f"\nStatus           : {matrix_result.status}"
        f"\nConfigs completed: {matrix_result.n_configs_completed}/{matrix_result.n_configs_total}"
        f"\nTotal cost       : ${matrix_result.total_cost_usd:.4f}"
        f"\nMatrix ID        : {matrix_result.matrix_id}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run experiment matrix (3 chunking × 2 embedding = 6 configs) "
                    "over the golden dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_matrix.yaml",
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Base output directory (matrix JSON written to {output-dir}/runs/)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Hard USD cap on accumulated cost (default: config.evaluation.budget_usd)",
    )
    parser.add_argument(
        "--golden",
        default="data/eval/golden.jsonl",
        help="Path to golden JSONL evaluation set",
    )
    args = parser.parse_args(argv)

    configure_logging()

    try:
        matrix_result = run_matrix(
            config_path=args.config,
            output_dir=args.output_dir,
            budget_usd=args.budget,
            golden_path=args.golden,
        )
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except BudgetExceededError as exc:
        print(
            f"\nBudget exceeded: spent ${exc.spent:.4f} of ${exc.budget:.2f}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        return 1

    _print_summary_table(matrix_result)
    return 0 if matrix_result.status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
