"""Embedding strategy registry: name → class mapping + factory.

Usage:
    from docintel.embeddings.registry import get_embedder
    from docintel.utils.config import load_config

    cfg = load_config()
    for emb_cfg in cfg.embeddings:
        embedder = get_embedder(emb_cfg)
        vectors = embedder.embed(texts)
"""

from __future__ import annotations

from docintel.utils.cache import EmbeddingCache
from docintel.utils.config import EmbeddingConfig

from .base import Embedder
from .bge_embedder import BgeEmbedder
from .minilm_embedder import MiniLMEmbedder

REGISTRY: dict[str, type[Embedder]] = {
    "bge_small_en_v15": BgeEmbedder,
    "minilm_l6_v2": MiniLMEmbedder,
}


def get_embedder(
    config: EmbeddingConfig,
    cache: EmbeddingCache | None = None,
) -> Embedder:
    """Instantiate and return the embedder for the given config.

    Raises KeyError if config.name is not in REGISTRY.
    """
    cls = REGISTRY[config.name]
    return cls(config, cache=cache)
