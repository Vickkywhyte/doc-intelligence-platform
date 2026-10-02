"""VectorStore registry — one entry per backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docintel.vectorstore.base import VectorStore
from docintel.vectorstore.chroma_store import ChromaStore

REGISTRY: dict[str, type[VectorStore]] = {
    "chroma": ChromaStore,
}


def get_store(
    strategy: str,
    embedder_name: str,
    base_dir: Path = Path("data/indexes"),
    *,
    client: Any = None,
) -> ChromaStore:
    """Return a ChromaStore for the given (strategy, embedder) pair.

    Args:
        strategy: chunking strategy name ("fixed", "recursive", "semantic").
        embedder_name: embedder registry name ("bge_small_en_v15", "minilm_l6_v2").
        base_dir: root directory for all index subdirectories.
        client: optional pre-built Chroma client (for tests using EphemeralClient).
    """
    return ChromaStore(strategy, embedder_name, base_dir, client=client)
