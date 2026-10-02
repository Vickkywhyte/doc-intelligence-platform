"""Statistical helpers for the evaluation pipeline — §7.4 bootstrap CI."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    scores: list[float],
    *,
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean of *scores*.

    Args:
        scores:     Per-question metric values.
        n_samples:  Number of bootstrap resamples (spec: 1000).
        confidence: Width of the interval (spec: 0.95 → [2.5, 97.5]).
        seed:       RNG seed for reproducibility.

    Returns:
        (lo, hi) — the lower and upper bound of the CI.
        Returns (0.0, 0.0) when *scores* is empty.
    """
    if scores is None or len(scores) == 0:
        return (0.0, 0.0)

    arr = np.array(scores, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)

    boot_means = np.array(
        [arr[rng.integers(0, n, size=n)].mean() for _ in range(n_samples)]
    )

    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_means, alpha / 2 * 100))
    hi = float(np.percentile(boot_means, (1.0 - alpha / 2) * 100))
    return (lo, hi)
