"""Chunking package: three strategies behind one ABC.

Public surface:
    Chunk           — data model (DATA_MODEL.md §2.2)
    ChunkStrategy   — ABC (DATA_MODEL.md §3.1)
    get_chunker     — factory from ChunkingConfig
    REGISTRY        — {name: class}
"""

from __future__ import annotations

from .base import Chunk, ChunkStrategy
from .registry import REGISTRY, get_chunker

__all__ = ["Chunk", "ChunkStrategy", "REGISTRY", "get_chunker"]
