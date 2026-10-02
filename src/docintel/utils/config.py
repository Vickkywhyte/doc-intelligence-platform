"""YAML → Pydantic config loader with stable content-hash.

Usage:
    from docintel.utils.config import load_config

    cfg = load_config("configs/experiment_matrix.yaml")
    print(cfg.hash())

CLI (for make check-config):
    python -m docintel.utils.config [path/to/config.yaml]
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from docintel.exceptions import ConfigError

# ── Sub-models ────────────────────────────────────────────────────────────────

EmbeddingName = Literal[
    "openai_text_embedding_3_small",
    "bge_small_en_v15",
    "minilm_l6_v2",
]

ChunkingStrategyName = Literal["fixed", "recursive", "semantic"]


class ChunkingConfig(BaseModel):
    model_config = {"extra": "forbid"}

    name: ChunkingStrategyName
    size: int = 512
    overlap: int = 64
    threshold: float | None = None
    max_size: int | None = None


class EmbeddingConfig(BaseModel):
    model_config = {"extra": "forbid"}

    name: EmbeddingName
    batch: int = 64


class RetrievalConfig(BaseModel):
    model_config = {"extra": "forbid"}

    k: int = 5
    mmr: bool = False
    mmr_lambda: float = 0.5
    rerank: bool = False


class GenerationConfig(BaseModel):
    model_config = {"extra": "forbid"}

    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512


class EvaluationConfig(BaseModel):
    model_config = {"extra": "forbid"}

    judge_model: str = "openai/gpt-4o-mini"
    judge_prompt_version: str = "v1"
    metrics: list[str]
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])
    bootstrap_samples: int = 1000
    budget_usd: float = 5.00
    seed: int = 0
    partial_claim_weight: float = 0.5


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration. extra='forbid' catches typos early."""

    model_config = {"extra": "forbid"}

    chunking: list[ChunkingConfig]
    embeddings: list[EmbeddingConfig]
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig

    def hash(self) -> str:
        """Return SHA-256 of the canonicalized config (sorted keys, compact JSON)."""
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


# ── Loader ────────────────────────────────────────────────────────────────────


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load and validate an experiment config YAML.

    Raises ConfigError on missing file, invalid YAML, or schema violation.
    """
    if path is None:
        path = os.environ.get("CONFIG", "configs/experiment_matrix.yaml")

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Expected a YAML mapping at top level, got {type(raw).__name__}")

    try:
        return ExperimentConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"Config schema error in {path}:\n{exc}") from exc


# ── CLI entry-point (make check-config) ──────────────────────────────────────


def _main(argv: list[str]) -> int:
    path = argv[0] if argv else None
    try:
        cfg = load_config(path)
    except ConfigError as exc:
        print(f"ConfigError: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(cfg.model_dump(mode="json"), indent=2))
    print(f"\nconfig_hash={cfg.hash()}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
