"""FastAPI dependency injection — pipeline, config, and mock overrides.

When USE_HOSTED=false (default for tests), returns a mock pipeline so tests
never require a real OpenRouter key.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import structlog
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.utils.config import ExperimentConfig, load_config

log = structlog.get_logger()

_USE_HOSTED = os.environ.get("USE_HOSTED", "false").lower() in ("1", "true", "yes")

# ── Config ─────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_config() -> ExperimentConfig:
    """Return the canonical experiment config (cached after first load)."""
    config_path = os.environ.get("CONFIG", "configs/experiment_matrix.yaml")
    return load_config(config_path)


# ── Pipeline ───────────────────────────────────────────────────────────────────


def get_pipeline():
    """Return a RAGPipeline instance.

    In hosted mode: builds from the real config + ChromaDB + LLMClient.
    In local/test mode (USE_HOSTED=false): returns a mock pipeline backed by
    FakeLLMClient + FakeVectorStore so no network calls are made.
    """
    if not _USE_HOSTED:
        return _build_fake_pipeline()
    return _build_real_pipeline()


def _build_fake_pipeline():
    """Build a pipeline that never calls the network (for dev/test)."""
    import numpy as np
    from typing import Any

    from tests.mocks.llm import FakeLLMClient
    from tests.mocks.vectorstore import FakeVectorStore

    from docintel.chunking.base import Chunk
    from docintel.pipeline.rag_pipeline import RAGPipeline
    from docintel.retrieval.retriever import Retriever
    from docintel.vectorstore.base import RetrievedChunk

    store = FakeVectorStore()
    chunk = Chunk(
        chunk_id="fake_doc:fixed:00001",
        doc_id="fake_doc",
        page=1,
        strategy="fixed",
        index=0,
        text="This is a placeholder document. No real documents have been indexed yet.",
        char_start=0,
        char_end=70,
        token_count=12,
    )
    vec = np.random.default_rng(0).random(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    store.upsert([chunk], vec.reshape(1, -1))

    # Minimal concrete retriever that bypasses embedding (returns from store directly)
    class _DirectRetriever(Retriever):
        """Returns chunks directly from the FakeVectorStore without embedding the query."""

        def retrieve(self, query: str, k: int, where: dict[str, Any] | None = None) -> list[RetrievedChunk]:
            return [
                RetrievedChunk(chunk=chunk, score=0.5, rank=1, retriever="topk")
            ]

        @property
        def name(self) -> str:
            return "direct_fake"

    retriever = _DirectRetriever()
    llm = FakeLLMClient(
        default="I don't know based on the provided documents.",
        model="fake/local",
    )
    return RAGPipeline(retriever, llm, k=5)


def _build_real_pipeline():
    """Build a pipeline from the real config (requires indexes + API key)."""
    from docintel.embeddings.registry import get_embedder
    from docintel.pipeline.rag_pipeline import RAGPipeline
    from docintel.retrieval.retriever import DenseRetriever
    from docintel.generation.llm_client import OpenRouterClient
    from docintel.vectorstore.chroma_store import ChromaStore

    cfg = get_config()
    # Use first chunking + embedding config as the active pipeline config
    chunk_cfg = cfg.chunking[0]
    emb_cfg = cfg.embeddings[0]
    gen_cfg = cfg.generation

    store = ChromaStore(chunk_cfg.name, emb_cfg.name)
    embedder = get_embedder(emb_cfg)
    retriever = DenseRetriever(store, embedder, cfg.retrieval)
    llm = OpenRouterClient(gen_cfg)
    return RAGPipeline(retriever, llm, k=cfg.retrieval.k)
