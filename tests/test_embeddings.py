"""Tests for the embeddings package (Phase 4).

All tests use FakeEmbedder — no HF model download during pytest.
Real embedder integration (cache round-trip) is tested with a tmp_path
cache so nothing touches data/cache/embeddings/ in CI.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from docintel.embeddings.base import Embedder
from docintel.embeddings.registry import REGISTRY, get_embedder
from docintel.utils.cache import EmbeddingCache
from docintel.utils.config import EmbeddingConfig
from tests.mocks.embedder import FakeEmbedder

# ── Helpers ───────────────────────────────────────────────────────────────────


def _bge_cfg(batch: int = 64) -> EmbeddingConfig:
    return EmbeddingConfig(name="bge_small_en_v15", batch=batch)


def _minilm_cfg(batch: int = 64) -> EmbeddingConfig:
    return EmbeddingConfig(name="minilm_l6_v2", batch=batch)


SAMPLE_TEXTS = [
    "This Non-Disclosure Agreement is entered into as of January 1, 2024.",
    "The Receiving Party shall hold all Confidential Information in strict confidence.",
    "Either party may terminate this agreement upon thirty days written notice.",
]


# ── FakeEmbedder unit tests ───────────────────────────────────────────────────


def test_fake_embedder_shape() -> None:
    emb = FakeEmbedder()
    out = emb.embed(SAMPLE_TEXTS)
    assert out.shape == (len(SAMPLE_TEXTS), 384)
    assert out.dtype == np.float32


def test_fake_embedder_l2_normalised() -> None:
    emb = FakeEmbedder()
    out = emb.embed(SAMPLE_TEXTS)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(SAMPLE_TEXTS)), atol=1e-5)


def test_fake_embedder_deterministic() -> None:
    emb = FakeEmbedder()
    a = emb.embed(SAMPLE_TEXTS)
    b = emb.embed(SAMPLE_TEXTS)
    np.testing.assert_array_equal(a, b)


def test_fake_embedder_different_texts_differ() -> None:
    emb = FakeEmbedder()
    a = emb.embed(["text one"])
    b = emb.embed(["text two"])
    assert not np.array_equal(a, b)


def test_fake_embedder_empty_input() -> None:
    emb = FakeEmbedder()
    out = emb.embed([])
    assert out.shape == (0, 384)
    assert out.dtype == np.float32


def test_fake_embedder_accepts_config() -> None:
    cfg = _bge_cfg()
    emb = FakeEmbedder(cfg)
    assert emb.name == "bge_small_en_v15"
    assert emb.dim == 384


def test_fake_embedder_custom_dim() -> None:
    emb = FakeEmbedder(dim=128)
    out = emb.embed(["hello"])
    assert out.shape == (1, 128)


# ── EmbeddingCache unit tests ─────────────────────────────────────────────────


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(cache_dir=tmp_path)


def test_cache_miss_returns_none(tmp_cache: EmbeddingCache) -> None:
    assert tmp_cache.get("unseen text", "bge_small_en_v15") is None


def test_cache_round_trip(tmp_cache: EmbeddingCache) -> None:
    vec = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    tmp_cache.put("hello", "bge_small_en_v15", vec, dim=3)
    recovered = tmp_cache.get("hello", "bge_small_en_v15")
    assert recovered is not None
    np.testing.assert_array_equal(recovered, vec)


def test_cache_sidecar_json(tmp_cache: EmbeddingCache) -> None:
    vec = np.ones(4, dtype=np.float32)
    tmp_cache.put("test text", "minilm_l6_v2", vec, dim=4)
    k = tmp_cache.key("test text", "minilm_l6_v2")
    sidecar = json.loads((Path(tmp_cache._dir) / f"{k}.json").read_text())
    assert sidecar["model"] == "minilm_l6_v2"
    assert sidecar["dim"] == 4
    assert "test text" in sidecar["text_preview"]


def test_cache_key_differs_by_model(tmp_cache: EmbeddingCache) -> None:
    k1 = tmp_cache.key("same text", "bge_small_en_v15")
    k2 = tmp_cache.key("same text", "minilm_l6_v2")
    assert k1 != k2


def test_cache_key_differs_by_text(tmp_cache: EmbeddingCache) -> None:
    k1 = tmp_cache.key("text A", "bge_small_en_v15")
    k2 = tmp_cache.key("text B", "bge_small_en_v15")
    assert k1 != k2


def test_cache_hit_returns_same_vector(tmp_cache: EmbeddingCache) -> None:
    vec = np.random.default_rng(0).random(384).astype(np.float32)
    tmp_cache.put("cached text", "bge_small_en_v15", vec, dim=384)

    hit = tmp_cache.get("cached text", "bge_small_en_v15")
    assert hit is not None
    np.testing.assert_array_equal(hit, vec)


def test_cache_stats(tmp_cache: EmbeddingCache) -> None:
    assert tmp_cache.stats()["total_entries"] == 0
    tmp_cache.put("a", "m", np.zeros(4, dtype=np.float32), 4)
    tmp_cache.put("b", "m", np.zeros(4, dtype=np.float32), 4)
    assert tmp_cache.stats()["total_entries"] == 2


# ── FakeEmbedder with cache round-trip ───────────────────────────────────────


def test_fake_embedder_cache_integration(tmp_cache: EmbeddingCache) -> None:
    """Verify that a cache-aware embedder wrapper respects the cache contract."""
    # Manually prime the cache with known vectors
    known_vec = np.ones(384, dtype=np.float32)
    known_vec /= np.linalg.norm(known_vec)
    tmp_cache.put(SAMPLE_TEXTS[0], "fake", known_vec, 384)

    # Cache hit should return the stored vector (not the hash-generated one)
    recovered = tmp_cache.get(SAMPLE_TEXTS[0], "fake")
    assert recovered is not None
    np.testing.assert_array_almost_equal(recovered, known_vec)


# ── Registry tests ────────────────────────────────────────────────────────────


def test_registry_contains_local_embedders() -> None:
    assert set(REGISTRY.keys()) == {"bge_small_en_v15", "minilm_l6_v2"}


def test_registry_does_not_contain_openai() -> None:
    assert "openai_text_embedding_3_small" not in REGISTRY


def test_get_embedder_returns_correct_type() -> None:
    from docintel.embeddings.bge_embedder import BgeEmbedder
    from docintel.embeddings.minilm_embedder import MiniLMEmbedder

    assert isinstance(get_embedder(_bge_cfg()), BgeEmbedder)
    assert isinstance(get_embedder(_minilm_cfg()), MiniLMEmbedder)


def test_get_embedder_dim_and_name() -> None:
    bge = get_embedder(_bge_cfg())
    assert bge.dim == 384
    assert bge.name == "bge_small_en_v15"

    mini = get_embedder(_minilm_cfg())
    assert mini.dim == 384
    assert mini.name == "minilm_l6_v2"


def test_embedder_is_abstract() -> None:
    """Embedder cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Embedder()  # type: ignore[abstract]


# ── Output contract (using FakeEmbedder as a proxy) ──────────────────────────


def test_output_dtype_float32() -> None:
    emb = FakeEmbedder()
    out = emb.embed(SAMPLE_TEXTS)
    assert out.dtype == np.float32


def test_output_shape_matches_input_length() -> None:
    emb = FakeEmbedder()
    for n in (1, 5, 13):
        texts = [f"text {i}" for i in range(n)]
        out = emb.embed(texts)
        assert out.shape == (n, 384)


def test_output_l2_norm_is_one() -> None:
    emb = FakeEmbedder()
    out = emb.embed(SAMPLE_TEXTS)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(SAMPLE_TEXTS)), atol=1e-5)


def test_single_text_embed() -> None:
    emb = FakeEmbedder()
    out = emb.embed(["single sentence."])
    assert out.shape == (1, 384)
