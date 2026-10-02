"""Embeddings package: two local embedders behind one ABC (ADR-003, ADR-015).

Public surface:
    Embedder        — ABC (DATA_MODEL.md §3.2)
    BgeEmbedder     — BAAI/bge-small-en-v1.5 (384d)
    MiniLMEmbedder  — sentence-transformers/all-MiniLM-L6-v2 (384d)
    get_embedder    — factory from EmbeddingConfig
    REGISTRY        — {name: class}
"""

from __future__ import annotations

from .base import Embedder
from .bge_embedder import BgeEmbedder
from .minilm_embedder import MiniLMEmbedder
from .registry import REGISTRY, get_embedder

__all__ = [
    "Embedder",
    "BgeEmbedder",
    "MiniLMEmbedder",
    "REGISTRY",
    "get_embedder",
]
