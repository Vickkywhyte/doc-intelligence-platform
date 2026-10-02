"""Tests for evaluation/stats.py — bootstrap CI and helper functions.

Covers the existing bootstrap_ci and the Phase 9 additions:
  bootstrap_ci_for_metric, ci_half_width.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.evaluation.stats import (
    bootstrap_ci,
    bootstrap_ci_for_metric,
    ci_half_width,
)


# ── ci_half_width ──────────────────────────────────────────────────────────────


class TestCiHalfWidth:
    def test_symmetric_interval(self) -> None:
        assert ci_half_width(0.6, 0.8) == pytest.approx(0.1, abs=1e-9)

    def test_zero_width(self) -> None:
        assert ci_half_width(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)

    def test_wide_interval(self) -> None:
        assert ci_half_width(0.0, 1.0) == pytest.approx(0.5, abs=1e-9)

    def test_negative_bounds(self) -> None:
        assert ci_half_width(-0.4, -0.2) == pytest.approx(0.1, abs=1e-9)

    def test_formula_is_hi_minus_lo_over_2(self) -> None:
        lo, hi = 0.73, 0.91
        assert ci_half_width(lo, hi) == pytest.approx((hi - lo) / 2, abs=1e-12)


# ── bootstrap_ci_for_metric ────────────────────────────────────────────────────


class TestBootstrapCiForMetric:
    def test_returns_tuple_of_floats(self) -> None:
        result = bootstrap_ci_for_metric([0.8, 0.9, 0.7, 0.85])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_lo_le_mean_le_hi(self) -> None:
        scores = [0.8, 0.9, 0.7, 0.85, 0.88, 0.76]
        lo, hi = bootstrap_ci_for_metric(scores)
        mean = float(np.mean(scores))
        assert lo <= mean <= hi

    def test_empty_list_returns_zeros(self) -> None:
        assert bootstrap_ci_for_metric([]) == (0.0, 0.0)

    def test_single_value_collapses_to_value(self) -> None:
        lo, hi = bootstrap_ci_for_metric([0.75])
        assert lo == pytest.approx(0.75, abs=1e-6)
        assert hi == pytest.approx(0.75, abs=1e-6)

    def test_numpy_array_input(self) -> None:
        arr = np.array([0.6, 0.7, 0.8, 0.75])
        lo, hi = bootstrap_ci_for_metric(arr)
        assert lo < hi

    def test_reproducible_with_same_seed(self) -> None:
        data = [0.5, 0.6, 0.7, 0.8, 0.9]
        r1 = bootstrap_ci_for_metric(data, n_samples=500, seed=42)
        r2 = bootstrap_ci_for_metric(data, n_samples=500, seed=42)
        assert r1 == r2

    def test_different_seeds_may_differ(self) -> None:
        # Use n=100 so the bootstrap distribution is fine-grained enough that
        # different seeds are very likely to produce different percentile values.
        rng = np.random.default_rng(1)
        data = list(rng.normal(0.7, 0.15, 100))
        r1 = bootstrap_ci_for_metric(data, n_samples=1000, seed=0)
        r2 = bootstrap_ci_for_metric(data, n_samples=1000, seed=12345)
        assert r1[0] != r2[0] or r1[1] != r2[1]

    def test_consistent_with_bootstrap_ci(self) -> None:
        """bootstrap_ci_for_metric is a positional wrapper around bootstrap_ci."""
        data = [0.7, 0.75, 0.8, 0.85, 0.9]
        r1 = bootstrap_ci_for_metric(data, n_samples=500, seed=7)
        r2 = bootstrap_ci(data, n_samples=500, seed=7)
        assert r1 == pytest.approx(r2, abs=1e-9)

    def test_95pct_interval_width(self) -> None:
        """With many samples the interval width should be non-trivial."""
        rng = np.random.default_rng(0)
        data = list(rng.normal(0.7, 0.1, 50))
        lo, hi = bootstrap_ci_for_metric(data, n_samples=2000, seed=0)
        assert hi - lo > 0.01, "CI width should be non-trivial for n=50"
        assert hi - lo < 0.20, "CI width should not be absurdly wide"

    def test_all_same_values(self) -> None:
        lo, hi = bootstrap_ci_for_metric([0.5, 0.5, 0.5, 0.5])
        assert lo == pytest.approx(0.5, abs=1e-6)
        assert hi == pytest.approx(0.5, abs=1e-6)


# ── existing bootstrap_ci (smoke test; real tests were added in Phase 8) ──────


class TestBootstrapCiExisting:
    def test_returns_lo_le_hi(self) -> None:
        lo, hi = bootstrap_ci([0.8, 0.9, 0.7], n_samples=100, seed=0)
        assert lo <= hi

    def test_empty(self) -> None:
        assert bootstrap_ci([]) == (0.0, 0.0)
