"""Phase 9 — Generate Markdown + CSV + PNG report from a matrix result JSON.

Usage:
    python scripts/08_generate_report.py [options]

Options:
    --matrix     Path to matrix JSON; if omitted, picks newest results/runs/*_matrix.json
    --output-dir Base output directory (default: results/)
    --metric     Metric for heatmap color axis (default: answer_accuracy)

Writes:
    results/reports/{matrix_id}.md
    results/reports/{matrix_id}.csv
    results/figures/{matrix_id}_heatmap.png
    results/figures/{matrix_id}_hallucination_bar.png
    results/figures/{matrix_id}_cost_quality_scatter.png

Exit codes:
    0  — success
    1  — no matrix result found or unexpected error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.evaluation.report import generate_report
from docintel.utils.logging import configure_logging


def _find_newest_matrix(runs_dir: Path) -> Path | None:
    """Return the newest *_matrix.json in runs_dir, or None."""
    candidates = sorted(runs_dir.glob("*_matrix.json"))
    return candidates[-1] if candidates else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate comparison report from an experiment matrix result JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--matrix",
        default=None,
        help="Path to matrix JSON (default: newest results/runs/*_matrix.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Base output directory",
    )
    parser.add_argument(
        "--metric",
        default="answer_accuracy",
        help="Metric for heatmap color axis",
    )
    args = parser.parse_args(argv)

    configure_logging()

    matrix_path: str | None = args.matrix
    if matrix_path is None:
        runs_dir = Path(args.output_dir) / "runs"
        found = _find_newest_matrix(runs_dir)
        if found is None:
            print(
                f"ERROR: No *_matrix.json found in {runs_dir}. "
                f"Run scripts/07_run_eval_matrix.py first.",
                file=sys.stderr,
            )
            return 1
        matrix_path = str(found)
        print(f"Using matrix: {matrix_path}")

    try:
        paths = generate_report(
            matrix_path=matrix_path,
            output_dir=args.output_dir,
            metric=args.metric,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\nReport outputs:")
    print(f"  Markdown : {paths.md}")
    print(f"  CSV      : {paths.csv}")
    print(f"  Heatmap  : {paths.heatmap}")
    print(f"  Bar      : {paths.bar}")
    print(f"  Scatter  : {paths.scatter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
