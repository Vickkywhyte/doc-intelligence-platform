"""Content-hash keyed embedding cache.

Layout on disk:
  data/cache/embeddings/{sha256hex}.npy   — float32 vector, shape [dim]
  data/cache/embeddings/{sha256hex}.json  — {model, dim, text_preview}

Cache key: sha256(text + "||" + model_name).hexdigest()
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    pass

log = structlog.get_logger()

_DEFAULT_CACHE_DIR = Path("data/cache/embeddings")


class EmbeddingCache:
    """Persistent per-text embedding cache backed by .npy files."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def key(self, text: str, model_name: str) -> str:
        """Return the 64-char hex SHA-256 cache key for (text, model)."""
        payload = f"{text}||{model_name}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, text: str, model_name: str) -> np.ndarray | None:
        """Return the cached float32 vector [dim] or None on a miss."""
        path = self._dir / f"{self.key(text, model_name)}.npy"
        if path.exists():
            return np.load(str(path))
        return None

    def put(self, text: str, model_name: str, vector: np.ndarray, dim: int) -> None:
        """Persist a 1-D float32 vector and its sidecar metadata."""
        k = self.key(text, model_name)
        npy_path = self._dir / f"{k}.npy"
        json_path = self._dir / f"{k}.json"

        np.save(str(npy_path), vector.astype(np.float32))
        json_path.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "dim": dim,
                    "text_preview": text[:120],
                }
            )
        )

    def stats(self) -> dict[str, int]:
        """Return {total_entries} count based on .npy files present."""
        return {"total_entries": len(list(self._dir.glob("*.npy")))}


# Module-level singleton — shared across all embedders in a process.
embedding_cache = EmbeddingCache()
