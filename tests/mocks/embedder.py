"""Deterministic hash-based fake Embedder for tests.

Produces L2-normalised float32 vectors seeded from the SHA-256 of each text,
so identical input always returns the same vector.  No model download, no
network, no sentence_transformers import.

Usage:
    from tests.mocks.embedder import FakeEmbedder
    from docintel.utils.config import EmbeddingConfig

    cfg = EmbeddingConfig(name="bge_small_en_v15", batch=64)
    emb = FakeEmbedder(cfg)
    vectors = emb.embed(["hello world"])  # shape [1, 384]
"""

from __future__ import annotations

import hashlib

import numpy as np

from docintel.embeddings.base import Embedder
from docintel.utils.config import EmbeddingConfig

_DEFAULT_DIM = 384


class FakeEmbedder(Embedder):
    """Deterministic, dependency-free fake for unit tests.

    Each text maps to a reproducible normalised vector via seeded RNG.
    The constructor accepts an EmbeddingConfig so it can stand in for any
    real embedder in tests that call get_embedder().
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        dim: int = _DEFAULT_DIM,
        name: str = "fake",
    ) -> None:
        self._dim = dim
        self._name = name
        if config is not None:
            self._name = config.name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        result = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed % (2**32))
            vec = rng.standard_normal(self._dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            result[i] = vec / norm if norm > 0 else vec
        return result
