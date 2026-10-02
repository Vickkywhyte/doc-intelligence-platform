"""Tests for vectorstore and retrieval packages (Phase 5).

ChromaStore integration tests use chromadb.EphemeralClient() — no disk I/O.
Retriever tests use FakeVectorStore + FakeEmbedder — no model downloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from docintel.chunking.base import Chunk
from docintel.retrieval.reranker import NoOpReranker
from docintel.retrieval.retriever import DenseRetriever
from docintel.utils.config import EmbeddingConfig, RetrievalConfig
from docintel.vectorstore.base import RetrievedChunk, VectorStore
from docintel.vectorstore.chroma_store import ChromaStore
from docintel.vectorstore.registry import REGISTRY, get_store
from tests.mocks.embedder import FakeEmbedder
from tests.mocks.vectorstore import FakeVectorStore

# ── Fixtures ──────────────────────────────────────────────────────────────────

DIM = 384


def _make_chunk(index: int, text: str = "") -> Chunk:
    doc_id = "abcdef1234567890"
    return Chunk(
        chunk_id=f"{doc_id}:fixed:{index:05d}",
        doc_id=doc_id,
        strategy="fixed",
        index=index,
        text=text or f"Chunk number {index} about contracts and services.",
        page=1,
        char_start=index * 100,
        char_end=index * 100 + 50,
        token_count=10,
    )


def _make_chunks(n: int) -> list[Chunk]:
    return [_make_chunk(i) for i in range(n)]


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=DIM)


@pytest.fixture()
def fake_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture()
def populated_store(fake_embedder: FakeEmbedder) -> FakeVectorStore:
    store = FakeVectorStore()
    chunks = _make_chunks(10)
    texts = [c.text for c in chunks]
    vecs = fake_embedder.embed(texts)
    store.upsert(chunks, vecs)
    return store


@pytest.fixture()
def topk_config() -> RetrievalConfig:
    return RetrievalConfig(k=3, mmr=False, mmr_lambda=0.5)


@pytest.fixture()
def mmr_config() -> RetrievalConfig:
    return RetrievalConfig(k=3, mmr=True, mmr_lambda=0.7)


@pytest.fixture()
def ephemeral_chroma(request: pytest.FixtureRequest) -> ChromaStore:
    """Each test gets a fresh Chroma collection via a unique name suffix.

    chromadb.EphemeralClient() may be a process-level singleton in newer
    versions, so collection names must be unique per test to avoid leakage.
    """
    import hashlib
    import chromadb

    suffix = hashlib.md5(request.node.nodeid.encode()).hexdigest()[:8]
    client = chromadb.EphemeralClient()
    return ChromaStore("fixed", f"emb_{suffix}", client=client)


# ── RetrievedChunk model ───────────────────────────────────────────────────────

def test_retrieved_chunk_fields() -> None:
    chunk = _make_chunk(0)
    rc = RetrievedChunk(chunk=chunk, score=0.95, rank=1, retriever="topk")
    assert rc.chunk is chunk
    assert rc.score == pytest.approx(0.95)
    assert rc.rank == 1
    assert rc.retriever == "topk"


def test_retrieved_chunk_extra_ignored() -> None:
    chunk = _make_chunk(0)
    rc = RetrievedChunk(chunk=chunk, score=0.5, rank=1, retriever="mmr", unknown_field="x")
    assert not hasattr(rc, "unknown_field")


def test_retrieved_chunk_rank_ge_1() -> None:
    chunk = _make_chunk(0)
    with pytest.raises(Exception):
        RetrievedChunk(chunk=chunk, score=0.5, rank=0, retriever="topk")


# ── FakeVectorStore ────────────────────────────────────────────────────────────

def test_fake_store_count_empty(fake_store: FakeVectorStore) -> None:
    assert fake_store.count() == 0


def test_fake_store_upsert_count(fake_embedder: FakeEmbedder) -> None:
    store = FakeVectorStore()
    chunks = _make_chunks(5)
    vecs = fake_embedder.embed([c.text for c in chunks])
    store.upsert(chunks, vecs)
    assert store.count() == 5


def test_fake_store_query_returns_k(populated_store: FakeVectorStore, fake_embedder: FakeEmbedder) -> None:
    q = fake_embedder.embed(["contract terms"])[0]
    results = populated_store.query(q, k=3)
    assert len(results) == 3


def test_fake_store_query_scores_descending(populated_store: FakeVectorStore, fake_embedder: FakeEmbedder) -> None:
    q = fake_embedder.embed(["contract terms"])[0]
    results = populated_store.query(q, k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_fake_store_query_ranks_sequential(populated_store: FakeVectorStore, fake_embedder: FakeEmbedder) -> None:
    q = fake_embedder.embed(["query"])[0]
    results = populated_store.query(q, k=4)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_fake_store_query_returns_topk_retriever(populated_store: FakeVectorStore, fake_embedder: FakeEmbedder) -> None:
    q = fake_embedder.embed(["query"])[0]
    results = populated_store.query(q, k=3)
    assert all(r.retriever == "topk" for r in results)


def test_fake_store_query_empty_store(fake_store: FakeVectorStore, fake_embedder: FakeEmbedder) -> None:
    q = fake_embedder.embed(["query"])[0]
    assert fake_store.query(q, k=3) == []


def test_fake_store_name(fake_store: FakeVectorStore) -> None:
    assert fake_store.name == "fake"


# ── ChromaStore integration (ephemeral) ───────────────────────────────────────

def test_chroma_store_name() -> None:
    import chromadb

    client = chromadb.EphemeralClient()
    store = ChromaStore("fixed", "bge_small_en_v15", client=client)
    assert store.name == "fixed_bge_small_en_v15"


def test_chroma_store_empty_count(ephemeral_chroma: ChromaStore) -> None:
    assert ephemeral_chroma.count() == 0


def test_chroma_store_upsert_count(ephemeral_chroma: ChromaStore, fake_embedder: FakeEmbedder) -> None:
    chunks = _make_chunks(5)
    vecs = fake_embedder.embed([c.text for c in chunks])
    ephemeral_chroma.upsert(chunks, vecs)
    assert ephemeral_chroma.count() == 5


def test_chroma_store_query_k(ephemeral_chroma: ChromaStore, fake_embedder: FakeEmbedder) -> None:
    chunks = _make_chunks(8)
    vecs = fake_embedder.embed([c.text for c in chunks])
    ephemeral_chroma.upsert(chunks, vecs)

    q = fake_embedder.embed(["contract services"])[0]
    results = ephemeral_chroma.query(q, k=3)
    assert len(results) == 3


def test_chroma_store_query_scores_in_range(ephemeral_chroma: ChromaStore, fake_embedder: FakeEmbedder) -> None:
    chunks = _make_chunks(5)
    vecs = fake_embedder.embed([c.text for c in chunks])
    ephemeral_chroma.upsert(chunks, vecs)

    q = fake_embedder.embed(["contract"])[0]
    results = ephemeral_chroma.query(q, k=3)
    for r in results:
        assert -1.1 <= r.score <= 1.1


def test_chroma_store_query_scores_descending(ephemeral_chroma: ChromaStore, fake_embedder: FakeEmbedder) -> None:
    chunks = _make_chunks(8)
    vecs = fake_embedder.embed([c.text for c in chunks])
    ephemeral_chroma.upsert(chunks, vecs)

    q = fake_embedder.embed(["contract terms"])[0]
    results = ephemeral_chroma.query(q, k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_chroma_store_chunk_roundtrip(ephemeral_chroma: ChromaStore, fake_embedder: FakeEmbedder) -> None:
    chunk = _make_chunk(0, text="The confidentiality clause protects trade secrets.")
    vecs = fake_embedder.embed([chunk.text])
    ephemeral_chroma.upsert([chunk], vecs)

    q = fake_embedder.embed([chunk.text])[0]
    results = ephemeral_chroma.query(q, k=1)
    assert len(results) == 1
    returned = results[0].chunk
    assert returned.chunk_id == chunk.chunk_id
    assert returned.doc_id == chunk.doc_id
    assert returned.text == chunk.text
    assert returned.page == chunk.page
    assert returned.char_start == chunk.char_start
    assert returned.char_end == chunk.char_end


def test_chroma_store_upsert_idempotent(ephemeral_chroma: ChromaStore, fake_embedder: FakeEmbedder) -> None:
    chunks = _make_chunks(3)
    vecs = fake_embedder.embed([c.text for c in chunks])
    ephemeral_chroma.upsert(chunks, vecs)
    ephemeral_chroma.upsert(chunks, vecs)
    assert ephemeral_chroma.count() == 3  # upsert should not duplicate


def test_chroma_store_query_empty(fake_embedder: FakeEmbedder) -> None:
    import chromadb

    client = chromadb.EphemeralClient()
    store = ChromaStore("empty", "bge_small_en_v15", client=client)
    q = fake_embedder.embed(["query"])[0]
    assert store.query(q, k=3) == []


# ── VectorStore registry ──────────────────────────────────────────────────────

def test_registry_contains_chroma() -> None:
    assert "chroma" in REGISTRY
    assert REGISTRY["chroma"] is ChromaStore


def test_get_store_returns_chroma_store() -> None:
    import chromadb

    client = chromadb.EphemeralClient()
    store = get_store("recursive", "minilm_l6_v2", client=client)
    assert isinstance(store, ChromaStore)
    assert store.name == "recursive_minilm_l6_v2"


# ── DenseRetriever — top-k ─────────────────────────────────────────────────────

def test_topk_returns_k_results(
    populated_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    topk_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(populated_store, fake_embedder, topk_config)
    results = retriever.retrieve("contract payment terms", k=3)
    assert len(results) == 3


def test_topk_ranks_sequential(
    populated_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    topk_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(populated_store, fake_embedder, topk_config)
    results = retriever.retrieve("services", k=4)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_topk_scores_descending(
    populated_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    topk_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(populated_store, fake_embedder, topk_config)
    results = retriever.retrieve("services", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_topk_retriever_field(
    populated_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    topk_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(populated_store, fake_embedder, topk_config)
    results = retriever.retrieve("query", k=3)
    assert all(r.retriever == "topk" for r in results)


def test_topk_empty_store(
    fake_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    topk_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(fake_store, fake_embedder, topk_config)
    assert retriever.retrieve("query", k=3) == []


def test_topk_name(
    fake_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    topk_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(fake_store, fake_embedder, topk_config)
    assert retriever.name == "dense_topk"


# ── DenseRetriever — MMR ──────────────────────────────────────────────────────

def test_mmr_returns_k_results(
    populated_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    mmr_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(populated_store, fake_embedder, mmr_config)
    results = retriever.retrieve("contract terms", k=3)
    assert len(results) == 3


def test_mmr_ranks_sequential(
    populated_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    mmr_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(populated_store, fake_embedder, mmr_config)
    results = retriever.retrieve("query", k=4)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_mmr_retriever_field(
    populated_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    mmr_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(populated_store, fake_embedder, mmr_config)
    results = retriever.retrieve("query", k=3)
    assert all(r.retriever == "mmr" for r in results)


def test_mmr_no_duplicates(
    populated_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    mmr_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(populated_store, fake_embedder, mmr_config)
    results = retriever.retrieve("query", k=5)
    chunk_ids = [r.chunk.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_mmr_name(
    fake_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    mmr_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(fake_store, fake_embedder, mmr_config)
    assert retriever.name == "dense_mmr"


def test_mmr_empty_store(
    fake_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    mmr_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(fake_store, fake_embedder, mmr_config)
    assert retriever.retrieve("query", k=3) == []


# ── NoOpReranker ──────────────────────────────────────────────────────────────

def test_noop_reranker_passthrough(
    populated_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    topk_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(populated_store, fake_embedder, topk_config)
    reranker = NoOpReranker()
    results = retriever.retrieve("contract", k=3)
    reranked = reranker.rerank("contract", results)
    assert len(reranked) == len(results)
    for orig, rer in zip(results, reranked):
        assert rer.chunk.chunk_id == orig.chunk.chunk_id
        assert rer.score == orig.score
        assert rer.rank == orig.rank


def test_noop_reranker_retriever_field(
    populated_store: FakeVectorStore,
    fake_embedder: FakeEmbedder,
    topk_config: RetrievalConfig,
) -> None:
    retriever = DenseRetriever(populated_store, fake_embedder, topk_config)
    reranker = NoOpReranker()
    results = retriever.retrieve("query", k=3)
    reranked = reranker.rerank("query", results)
    assert all(r.retriever == "rerank" for r in reranked)


def test_noop_reranker_name() -> None:
    assert NoOpReranker().name == "noop"


def test_noop_reranker_empty() -> None:
    reranker = NoOpReranker()
    assert reranker.rerank("query", []) == []
