"""End-to-end pipeline integration test — no network, no real LLM.

Uses FakeEmbedder + FakeLLMClient + FakeVectorStore so the test suite
never requires an OpenRouter key.  Marked @pytest.mark.integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.chunking.base import Chunk
from docintel.generation.models import RAGResponse
from docintel.pipeline.rag_pipeline import RAGPipeline
from docintel.utils.config import RetrievalConfig
from docintel.vectorstore.base import RetrievedChunk
from tests.mocks.embedder import FakeEmbedder
from tests.mocks.llm import FakeLLMClient
from tests.mocks.vectorstore import FakeVectorStore


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def retrieval_cfg() -> RetrievalConfig:
    return RetrievalConfig(k=3, mmr=False)


@pytest.fixture()
def fake_store() -> FakeVectorStore:
    """A FakeVectorStore pre-loaded with two synthetic chunks."""
    store = FakeVectorStore()
    chunks = [
        Chunk(
            chunk_id="chunk001",
            doc_id="integ_doc",
            page=1,
            strategy="fixed",
            index=0,
            text="The contract effective date is 1 January 2025.",
            char_start=0,
            char_end=46,
            token_count=9,
        ),
        Chunk(
            chunk_id="chunk002",
            doc_id="integ_doc",
            page=1,
            strategy="fixed",
            index=1,
            text="Payment terms are net-30 days from invoice date.",
            char_start=46,
            char_end=94,
            token_count=9,
        ),
    ]
    embedder = FakeEmbedder()
    vectors = embedder.embed([c.text for c in chunks])
    store.upsert(chunks, vectors)
    return store


@pytest.fixture()
def pipeline(fake_store: FakeVectorStore, retrieval_cfg: RetrievalConfig) -> RAGPipeline:
    from docintel.retrieval.retriever import DenseRetriever

    embedder = FakeEmbedder()
    retriever = DenseRetriever(fake_store, embedder, retrieval_cfg)
    llm = FakeLLMClient(
        responses=["The contract starts on 1 January 2025. [integ_doc:1:chunk001]"],
        model="fake/test",
    )
    return RAGPipeline(retriever, llm, k=retrieval_cfg.k)


# ── Tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_rag_response_shape(pipeline: RAGPipeline) -> None:
    """RAGResponse has all expected fields with correct types."""
    resp = pipeline.query("When does the contract start?")

    assert isinstance(resp, RAGResponse)
    assert isinstance(resp.answer, str) and resp.answer
    assert isinstance(resp.refused, bool)
    assert isinstance(resp.citations, list)
    assert isinstance(resp.retrieved, list)
    assert isinstance(resp.model, str) and resp.model
    assert isinstance(resp.latency_ms, int) and resp.latency_ms >= 0
    assert isinstance(resp.cost_usd, float) and resp.cost_usd >= 0.0
    assert isinstance(resp.warnings, list)


@pytest.mark.integration
def test_retrieves_relevant_chunks(pipeline: RAGPipeline) -> None:
    """Pipeline retrieves at least one chunk from the fake store."""
    resp = pipeline.query("What are the payment terms?")
    assert len(resp.retrieved) >= 1


@pytest.mark.integration
def test_non_refusal_answer(pipeline: RAGPipeline) -> None:
    """Canned answer with a citation is not flagged as a refusal."""
    resp = pipeline.query("When does the contract start?")
    assert resp.refused is False


@pytest.mark.integration
def test_citation_parsed(pipeline: RAGPipeline) -> None:
    """Citation tag in the LLM response is parsed into resp.citations."""
    resp = pipeline.query("When does the contract start?")
    assert len(resp.citations) >= 1


@pytest.mark.integration
def test_refusal_when_llm_says_unknown() -> None:
    """Pipeline flags refused=True when LLM returns the standard refusal phrase."""
    from docintel.retrieval.retriever import DenseRetriever

    store = FakeVectorStore()
    chunks = [
        Chunk(
            chunk_id="refusal_chunk",
            doc_id="doc",
            page=1,
            strategy="fixed",
            index=0,
            text="Some irrelevant text.",
            char_start=0,
            char_end=21,
            token_count=4,
        )
    ]
    embedder = FakeEmbedder()
    vectors = embedder.embed([c.text for c in chunks])
    store.upsert(chunks, vectors)

    cfg = RetrievalConfig(k=3, mmr=False)
    retriever = DenseRetriever(store, embedder, cfg)
    llm = FakeLLMClient(model="fake/test")  # uses default refusal response
    p = RAGPipeline(retriever, llm, k=3)

    resp = p.query("What is the capital of France?")
    assert resp.refused is True
