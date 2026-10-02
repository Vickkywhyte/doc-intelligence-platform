"""Embedder ABC (DATA_MODEL.md §3.2).

All embedders return float32 L2-normalised vectors of shape [N, dim].
Batching and cache are the responsibility of each concrete subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Text → dense vectors. Handles batching + caching internally."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return shape [len(texts), dim], float32, L2-normalised.

        Empty input returns shape [0, dim].
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry key, e.g. 'bge_small_en_v15'."""
        ...
