"""MiniLM-L6-v2 embedder (sentence-transformers/all-MiniLM-L6-v2, 384d).

Local model; downloaded on first use to ~/.cache/huggingface (or $HF_HOME).
Same model used by SemanticChunker for boundary detection — share the cached
weights; they come from different SentenceTransformer instances but the OS
file-cache keeps the load fast after the first download.

Zero cost — emits cost_usd=0.00 to CostTracker per ADR-015.
"""

from __future__ import annotations

import time

import numpy as np
import structlog

from docintel.utils.cache import EmbeddingCache, embedding_cache
from docintel.utils.config import EmbeddingConfig
from docintel.utils.cost_tracker import cost_tracker

from .base import Embedder

log = structlog.get_logger()

_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_DIM = 384
_REGISTRY_NAME = "minilm_l6_v2"


class MiniLMEmbedder(Embedder):
    """all-MiniLM-L6-v2 — 384d ubiquitous baseline (ADR-003, ADR-015)."""

    def __init__(
        self,
        config: EmbeddingConfig,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self._batch_size = config.batch
        self._cache = cache or embedding_cache
        self._model = None

    @property
    def name(self) -> str:
        return _REGISTRY_NAME

    @property
    def dim(self) -> int:
        return _DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, _DIM), dtype=np.float32)

        result = np.zeros((len(texts), _DIM), dtype=np.float32)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text, self.name)
            if cached is not None:
                result[i] = cached
                log.debug("embed_cache_hit", model=self.name, index=i)
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            model = self._load_model()
            n_batches = (len(miss_texts) + self._batch_size - 1) // self._batch_size

            for b in range(n_batches):
                batch_texts = miss_texts[b * self._batch_size : (b + 1) * self._batch_size]
                batch_idx = miss_indices[b * self._batch_size : (b + 1) * self._batch_size]

                log.debug(
                    "embed_batch_start",
                    model=self.name,
                    batch=b,
                    size=len(batch_texts),
                )
                t0 = time.monotonic()

                vectors: np.ndarray = model.encode(
                    batch_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).astype(np.float32)

                latency_ms = round((time.monotonic() - t0) * 1000)
                token_count = sum(len(t.split()) for t in batch_texts)

                log.info(
                    "embed_batch_complete",
                    model=self.name,
                    batch=b,
                    size=len(batch_texts),
                    latency_ms=latency_ms,
                )
                cost_tracker.track(
                    kind="embedding",
                    model=self.name,
                    cost_usd=0.0,
                    tokens=token_count,
                )

                for j, (idx, text) in enumerate(zip(batch_idx, batch_texts)):
                    result[idx] = vectors[j]
                    self._cache.put(text, self.name, vectors[j], _DIM)

        return result

    def _load_model(self):  # type: ignore[return]
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("embedder_load_model", model=_MODEL_ID)
            self._model = SentenceTransformer(_MODEL_ID)
        return self._model
