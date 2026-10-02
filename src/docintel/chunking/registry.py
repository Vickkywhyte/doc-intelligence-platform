"""Chunking strategy registry: name → class mapping + factory.

Usage:
    from docintel.chunking.registry import get_chunker
    from docintel.utils.config import load_config

    cfg = load_config()
    for chunk_cfg in cfg.chunking:
        chunker = get_chunker(chunk_cfg)
        chunks = chunker.chunk(doc)
"""

from __future__ import annotations

from docintel.utils.config import ChunkingConfig

from .base import ChunkStrategy
from .fixed import FixedChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker

REGISTRY: dict[str, type[ChunkStrategy]] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}


def get_chunker(config: ChunkingConfig) -> ChunkStrategy:
    """Instantiate and return the chunker for the given config.

    Raises KeyError if config.name is not in REGISTRY (shouldn't happen with
    Pydantic's ChunkingStrategyName Literal validation upstream).
    """
    cls = REGISTRY[config.name]
    return cls(config)
