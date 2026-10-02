"""Tests for docintel.utils.config — Phase 1 acceptance criteria."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from docintel.exceptions import ConfigError
from docintel.utils.config import ExperimentConfig, load_config


# ── Fixtures ──────────────────────────────────────────────────────────────────


VALID_YAML = textwrap.dedent("""\
    chunking:
      - {name: fixed, size: 512, overlap: 64}
      - {name: recursive, size: 512, overlap: 64}
      - {name: semantic, threshold: 0.75, max_size: 800}

    embeddings:
      - {name: bge_small_en_v15, batch: 64}
      - {name: minilm_l6_v2, batch: 64}

    retrieval:
      k: 5
      mmr: false
      mmr_lambda: 0.5
      rerank: false

    generation:
      model: openai/gpt-4o-mini
      temperature: 0.0
      max_tokens: 512

    evaluation:
      judge_model: openai/gpt-4o-mini
      judge_prompt_version: v1
      metrics:
        - p_at_k
        - answer_accuracy
      k_values: [1, 3, 5]
      bootstrap_samples: 1000
      budget_usd: 5.00
      seed: 0
      partial_claim_weight: 0.5
""")


@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    p = tmp_path / "experiment.yaml"
    p.write_text(VALID_YAML)
    return p


@pytest.fixture
def canonical_config(valid_config_file: Path) -> ExperimentConfig:
    return load_config(valid_config_file)


# ── Test: loading valid config returns ExperimentConfig ──────────────────────


def test_load_valid_config_returns_experiment_config(valid_config_file: Path) -> None:
    cfg = load_config(valid_config_file)
    assert isinstance(cfg, ExperimentConfig)


def test_load_config_parses_chunking_strategies(canonical_config: ExperimentConfig) -> None:
    names = [c.name for c in canonical_config.chunking]
    assert names == ["fixed", "recursive", "semantic"]


def test_load_config_parses_embeddings(canonical_config: ExperimentConfig) -> None:
    # ADR-015: two local embedders only (openai_text_embedding_3_small removed)
    names = [e.name for e in canonical_config.embeddings]
    assert "bge_small_en_v15" in names
    assert "minilm_l6_v2" in names
    assert "openai_text_embedding_3_small" not in names


def test_load_config_parses_generation_model(canonical_config: ExperimentConfig) -> None:
    assert canonical_config.generation.model == "openai/gpt-4o-mini"


def test_load_config_parses_evaluation_budget(canonical_config: ExperimentConfig) -> None:
    assert canonical_config.evaluation.budget_usd == 5.00


# ── Test: malformed configs raise ConfigError ─────────────────────────────────


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nonexistent.yaml")


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("chunking: [unterminated")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_extra_field_raises_config_error(tmp_path: Path) -> None:
    """extra='forbid' on config models catches typos."""
    bad_yaml = VALID_YAML + "\nunknown_field: true\n"
    p = tmp_path / "extra.yaml"
    p.write_text(bad_yaml)
    with pytest.raises(ConfigError):
        load_config(p)


def test_wrong_chunking_name_raises_config_error(tmp_path: Path) -> None:
    bad_yaml = VALID_YAML.replace("name: fixed", "name: sliding")
    p = tmp_path / "wrong_name.yaml"
    p.write_text(bad_yaml)
    with pytest.raises(ConfigError):
        load_config(p)


# ── Test: config_hash() is stable ────────────────────────────────────────────


def test_hash_is_stable_for_same_config(valid_config_file: Path) -> None:
    cfg_a = load_config(valid_config_file)
    cfg_b = load_config(valid_config_file)
    assert cfg_a.hash() == cfg_b.hash()


def test_hash_differs_for_different_configs(tmp_path: Path) -> None:
    p1 = tmp_path / "a.yaml"
    p2 = tmp_path / "b.yaml"
    p1.write_text(VALID_YAML)
    p2.write_text(VALID_YAML.replace("k: 5", "k: 10"))
    assert load_config(p1).hash() != load_config(p2).hash()


def test_hash_is_hex_string_of_correct_length(canonical_config: ExperimentConfig) -> None:
    h = canonical_config.hash()
    assert len(h) == 64
    int(h, 16)  # raises ValueError if not valid hex


# ── Test: canonical config file loads without errors ─────────────────────────


def test_canonical_config_file_loads(tmp_path: Path) -> None:
    """configs/experiment_matrix.yaml must pass schema validation."""
    project_root = Path(__file__).parent.parent
    canonical = project_root / "configs" / "experiment_matrix.yaml"
    cfg = load_config(canonical)
    assert isinstance(cfg, ExperimentConfig)
    assert len(cfg.chunking) == 3
    assert len(cfg.embeddings) == 2  # ADR-015: two local embedders
