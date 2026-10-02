"""Pydantic models for generation output (DATA_MODEL §2.4)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from docintel.vectorstore.base import RetrievedChunk


class Citation(BaseModel):
    """A validated source reference extracted from an LLM answer."""

    model_config = {"extra": "ignore"}

    doc_id: str
    page: int
    chunk_id: str


class LLMResponse(BaseModel):
    """Raw response from a single LLM call."""

    model_config = {"extra": "ignore"}

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int


class RAGResponse(BaseModel):
    """Full RAG pipeline response for one question."""

    model_config = {"extra": "ignore"}

    query: str
    answer: str
    refused: bool
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    model: str
    latency_ms: int
    cost_usd: float
    warnings: list[str] = Field(default_factory=list)
