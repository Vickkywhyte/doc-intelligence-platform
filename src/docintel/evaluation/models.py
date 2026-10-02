"""Pydantic models for the evaluation layer — DATA_MODEL §2.5."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    """One entry in data/eval/golden.jsonl.

    Composition (total 75):
      factual     — 45  single-chunk facts, all 3 strategies mapped
      multi_hop   — 15  require combining info across chunks/docs
      unanswerable — 15 not in corpus; gold_chunk_ids must be empty
    """

    model_config = {"extra": "forbid"}

    id: str = Field(min_length=1, description="Unique identifier, e.g. 'g001'")
    question: str = Field(min_length=5)
    gold_answer: str = Field(min_length=1)
    gold_chunk_ids: dict[str, list[str]] = Field(
        description="Strategy → list of chunk_id strings. Empty for unanswerable."
    )
    type: Literal["factual", "multi_hop", "unanswerable"]
    notes: str = Field(default="", description="Curation notes; not used in scoring")
