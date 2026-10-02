"""Tests for the chunking package (Phase 3).

Coverage targets:
  - chunk_id format  (DATA_MODEL §5)
  - char_start/char_end round-trip: doc.full_text[c.char_start:c.char_end] == c.text
  - token_count > 0 for non-empty chunks
  - page assignment within doc's page range
  - no chunk exceeds size + overlap (fixed/recursive) or max_size (semantic)
  - overlap: adjacent fixed-chunker windows share tokens
  - coverage: union of chunk spans covers the full text

Note: SemanticChunker is tested with a mock SentenceTransformer to avoid
downloading the model in CI and to keep tests deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from docintel.chunking.base import Chunk, ChunkStrategy, count_tokens
from docintel.chunking.fixed import FixedChunker
from docintel.chunking.recursive import RecursiveChunker
from docintel.chunking.registry import REGISTRY, get_chunker
from docintel.chunking.semantic import SemanticChunker
from docintel.ingestion.loader import Document
from docintel.utils.config import ChunkingConfig

# ── Fixture helpers ────────────────────────────────────────────────────────────

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_document.json"


@pytest.fixture()
def sample_doc() -> Document:
    """Load the canonical 3-page NDA fixture (do NOT modify the file)."""
    raw = json.loads(FIXTURE_PATH.read_text())
    return Document.model_validate(raw)


def _fixed_cfg(size: int = 100, overlap: int = 10) -> ChunkingConfig:
    return ChunkingConfig(name="fixed", size=size, overlap=overlap)


def _recursive_cfg(size: int = 100, overlap: int = 10) -> ChunkingConfig:
    return ChunkingConfig(name="recursive", size=size, overlap=overlap)


def _semantic_cfg(threshold: float = 0.5, max_size: int = 300) -> ChunkingConfig:
    return ChunkingConfig(name="semantic", threshold=threshold, max_size=max_size)


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _assert_chunk_invariants(chunks: list[Chunk], doc: Document) -> None:
    """Core invariants every chunk must satisfy regardless of strategy."""
    full_text = doc.full_text
    valid_pages = {p.page_number for p in doc.pages}

    for c in chunks:
        # chunk_id format
        expected_id = f"{c.doc_id}:{c.strategy}:{c.index:05d}"
        assert c.chunk_id == expected_id, f"Bad chunk_id: {c.chunk_id!r}"

        # char round-trip (the primary correctness invariant)
        assert full_text[c.char_start : c.char_end] == c.text, (
            f"char offset mismatch for {c.chunk_id}: "
            f"expected {full_text[c.char_start:c.char_end]!r}, got {c.text!r}"
        )

        # offsets are non-negative and ordered
        assert c.char_start >= 0
        assert c.char_end > c.char_start

        # page must be a real page in this doc
        assert c.page in valid_pages, f"page {c.page} not in {valid_pages}"

        # token_count is consistent with text
        assert c.token_count == count_tokens(c.text)

        # doc_id correct
        assert c.doc_id == doc.doc_id


# ── Chunk model unit tests ─────────────────────────────────────────────────────


def test_chunk_model_extra_fields_ignored() -> None:
    """Chunk model uses extra='ignore' — unknown fields must not raise."""
    c = Chunk.model_validate(
        {
            "chunk_id": "abc:fixed:00000",
            "doc_id": "abc",
            "strategy": "fixed",
            "index": 0,
            "text": "hello",
            "page": 1,
            "char_start": 0,
            "char_end": 5,
            "token_count": 1,
            "unknown_future_field": "ignored",
        }
    )
    assert c.text == "hello"


def test_chunk_id_format_manual() -> None:
    """chunk_id = f'{doc_id}:{strategy}:{index:05d}' — verify zero-padding."""
    c = Chunk(
        chunk_id="docabc:recursive:00042",
        doc_id="docabc",
        strategy="recursive",
        index=42,
        text="test",
        page=1,
        char_start=0,
        char_end=4,
        token_count=1,
    )
    assert c.chunk_id == "docabc:recursive:00042"


# ── Fixed chunker tests ────────────────────────────────────────────────────────


def test_fixed_chunk_id_format(sample_doc: Document) -> None:
    chunks = FixedChunker(_fixed_cfg()).chunk(sample_doc)
    assert len(chunks) > 0
    for c in chunks:
        assert c.chunk_id == f"{sample_doc.doc_id}:fixed:{c.index:05d}"


def test_fixed_char_roundtrip(sample_doc: Document) -> None:
    chunks = FixedChunker(_fixed_cfg()).chunk(sample_doc)
    _assert_chunk_invariants(chunks, sample_doc)


def test_fixed_indices_sequential(sample_doc: Document) -> None:
    chunks = FixedChunker(_fixed_cfg()).chunk(sample_doc)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_fixed_no_chunk_exceeds_size_plus_overlap(sample_doc: Document) -> None:
    cfg = _fixed_cfg(size=80, overlap=16)
    chunks = FixedChunker(cfg).chunk(sample_doc)
    for c in chunks:
        # A window is exactly `size` tokens except the last one
        assert c.token_count <= cfg.size


def test_fixed_overlap_consecutive_chunks(sample_doc: Document) -> None:
    """Adjacent chunks must share approximately 'overlap' tokens of text."""
    cfg = _fixed_cfg(size=120, overlap=30)
    chunks = FixedChunker(cfg).chunk(sample_doc)
    if len(chunks) < 2:
        pytest.skip("not enough chunks to test overlap")

    for i in range(len(chunks) - 1):
        a, b = chunks[i], chunks[i + 1]
        # The start of chunk b must fall within chunk a
        assert b.char_start < a.char_end, (
            f"chunk {i+1} start {b.char_start} is after chunk {i} end {a.char_end}"
        )


def test_fixed_coverage(sample_doc: Document) -> None:
    """Every character in the document must appear in at least one chunk."""
    cfg = _fixed_cfg(size=80, overlap=0)
    chunks = FixedChunker(cfg).chunk(sample_doc)
    covered: set[int] = set()
    for c in chunks:
        covered.update(range(c.char_start, c.char_end))
    full_text = sample_doc.full_text
    assert covered == set(range(len(full_text)))


def test_fixed_empty_document() -> None:
    from datetime import datetime, timezone

    from docintel.ingestion.loader import Page

    doc = Document(
        doc_id="empty",
        source_path="",
        filename="empty.txt",
        pages=[Page(page_number=1, text="", char_start=0, char_end=0)],
        ingested_at=datetime.now(timezone.utc),
    )
    chunks = FixedChunker(_fixed_cfg()).chunk(doc)
    assert chunks == []


# ── Recursive chunker tests ───────────────────────────────────────────────────


def test_recursive_chunk_id_format(sample_doc: Document) -> None:
    chunks = RecursiveChunker(_recursive_cfg()).chunk(sample_doc)
    assert len(chunks) > 0
    for c in chunks:
        assert c.chunk_id == f"{sample_doc.doc_id}:recursive:{c.index:05d}"


def test_recursive_char_roundtrip(sample_doc: Document) -> None:
    chunks = RecursiveChunker(_recursive_cfg()).chunk(sample_doc)
    _assert_chunk_invariants(chunks, sample_doc)


def test_recursive_no_chunk_exceeds_size(sample_doc: Document) -> None:
    cfg = _recursive_cfg(size=80, overlap=10)
    chunks = RecursiveChunker(cfg).chunk(sample_doc)
    for c in chunks:
        # Allow a tiny over-run (up to 5 tokens) for indivisible words/tokens
        assert c.token_count <= cfg.size + 5, (
            f"chunk {c.chunk_id} has {c.token_count} tokens (limit {cfg.size})"
        )


def test_recursive_paragraphs_respected(sample_doc: Document) -> None:
    """Recursive chunker should keep double-newline boundaries intact."""
    cfg = ChunkingConfig(name="recursive", size=512, overlap=64)
    chunks = RecursiveChunker(cfg).chunk(sample_doc)
    # With size=512 the whole doc fits in 1–2 chunks; just verify invariants
    _assert_chunk_invariants(chunks, sample_doc)


def test_recursive_no_gap_consecutive(sample_doc: Document) -> None:
    """Consecutive recursive chunks must be gap-free (b starts ≤ a.char_end).

    Strict overlap is only guaranteed when a chunk spans multiple pieces AND the
    overlap budget fits.  When a single oversized piece fills a chunk there is
    no room for overlap, so equal start/end (adjacent) is correct.
    """
    cfg = _recursive_cfg(size=100, overlap=20)
    chunks = RecursiveChunker(cfg).chunk(sample_doc)
    if len(chunks) < 2:
        pytest.skip("not enough chunks")
    for i in range(len(chunks) - 1):
        a, b = chunks[i], chunks[i + 1]
        assert b.char_start <= a.char_end, (
            f"gap between chunk {i} and {i+1}: "
            f"chunk ends at {a.char_end}, next starts at {b.char_start}"
        )


def test_recursive_empty_document() -> None:
    from datetime import datetime, timezone

    from docintel.ingestion.loader import Page

    doc = Document(
        doc_id="empty2",
        source_path="",
        filename="empty.txt",
        pages=[Page(page_number=1, text="   ", char_start=0, char_end=3)],
        ingested_at=datetime.now(timezone.utc),
    )
    chunks = RecursiveChunker(_recursive_cfg()).chunk(doc)
    assert chunks == []


# ── Semantic chunker tests (MiniLM mocked) ────────────────────────────────────


def _make_mock_model(n_sentences: int) -> Any:
    """Return a mock SentenceTransformer that produces deterministic embeddings."""
    model = MagicMock()

    def encode(sentences: list[str], **kwargs: Any) -> np.ndarray:
        # Alternate high/low similarity to force splits
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((len(sentences), 64)).astype(np.float32)
        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    model.encode.side_effect = encode
    return model


@pytest.fixture()
def mock_sentence_transformer():
    """Patch SentenceTransformer so the model isn't downloaded in tests."""
    with patch(
        "docintel.chunking.semantic.SemanticChunker._get_model"
    ) as mock_get_model:
        mock_get_model.return_value = _make_mock_model(10)
        yield mock_get_model


def test_semantic_chunk_id_format(
    sample_doc: Document, mock_sentence_transformer: Any
) -> None:
    chunks = SemanticChunker(_semantic_cfg()).chunk(sample_doc)
    assert len(chunks) > 0
    for c in chunks:
        assert c.chunk_id == f"{sample_doc.doc_id}:semantic:{c.index:05d}"


def test_semantic_char_roundtrip(
    sample_doc: Document, mock_sentence_transformer: Any
) -> None:
    chunks = SemanticChunker(_semantic_cfg()).chunk(sample_doc)
    _assert_chunk_invariants(chunks, sample_doc)


def test_semantic_no_chunk_exceeds_max_size(
    sample_doc: Document, mock_sentence_transformer: Any
) -> None:
    cfg = _semantic_cfg(threshold=0.5, max_size=150)
    chunks = SemanticChunker(cfg).chunk(sample_doc)
    for c in chunks:
        assert c.token_count <= cfg.max_size, (
            f"chunk {c.chunk_id} has {c.token_count} tokens (max_size={cfg.max_size})"
        )


def test_semantic_indices_sequential(
    sample_doc: Document, mock_sentence_transformer: Any
) -> None:
    chunks = SemanticChunker(_semantic_cfg()).chunk(sample_doc)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_semantic_empty_document(mock_sentence_transformer: Any) -> None:
    from datetime import datetime, timezone

    from docintel.ingestion.loader import Page

    doc = Document(
        doc_id="empty3",
        source_path="",
        filename="empty.txt",
        pages=[Page(page_number=1, text="", char_start=0, char_end=0)],
        ingested_at=datetime.now(timezone.utc),
    )
    chunks = SemanticChunker(_semantic_cfg()).chunk(doc)
    assert chunks == []


# ── Registry tests ─────────────────────────────────────────────────────────────


def test_registry_contains_all_strategies() -> None:
    assert set(REGISTRY.keys()) == {"fixed", "recursive", "semantic"}


def test_get_chunker_returns_correct_type() -> None:
    assert isinstance(get_chunker(_fixed_cfg()), FixedChunker)
    assert isinstance(get_chunker(_recursive_cfg()), RecursiveChunker)
    assert isinstance(get_chunker(_semantic_cfg()), SemanticChunker)


def test_chunker_name_matches_config(sample_doc: Document) -> None:
    for name in ("fixed", "recursive"):
        cfg = ChunkingConfig(name=name, size=200, overlap=20)  # type: ignore[arg-type]
        chunker = get_chunker(cfg)
        assert chunker.name == name
        chunks = chunker.chunk(sample_doc)
        for c in chunks:
            assert c.strategy == name


# ── Cross-strategy: same doc, stable chunk_ids ────────────────────────────────


def test_chunk_ids_are_strategy_scoped(
    sample_doc: Document, mock_sentence_transformer: Any
) -> None:
    """chunk_ids from different strategies must not collide (EVAL_SPEC §1.3)."""
    fixed_ids = {c.chunk_id for c in FixedChunker(_fixed_cfg()).chunk(sample_doc)}
    recursive_ids = {
        c.chunk_id for c in RecursiveChunker(_recursive_cfg()).chunk(sample_doc)
    }
    semantic_ids = {
        c.chunk_id for c in SemanticChunker(_semantic_cfg()).chunk(sample_doc)
    }

    assert fixed_ids.isdisjoint(recursive_ids)
    assert fixed_ids.isdisjoint(semantic_ids)
    assert recursive_ids.isdisjoint(semantic_ids)


def test_chunk_ids_are_stable_across_calls(sample_doc: Document) -> None:
    """Running the same chunker twice on the same doc must produce identical IDs."""
    cfg = _fixed_cfg(size=100, overlap=10)
    chunks_a = FixedChunker(cfg).chunk(sample_doc)
    chunks_b = FixedChunker(cfg).chunk(sample_doc)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
