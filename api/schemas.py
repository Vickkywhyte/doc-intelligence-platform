"""Pydantic request/response schemas for the FastAPI layer (07_API_SPEC.md)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Error envelope ─────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    model_config = {"extra": "ignore"}

    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
    request_id: str


class ErrorResponse(BaseModel):
    model_config = {"extra": "ignore"}

    error: ErrorDetail


# ── Health ─────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    model_config = {"extra": "ignore"}

    status: Literal["ok", "degraded"]
    version: str
    git_sha: str
    python_version: str
    uptime_s: float
    reason: str | None = None


# ── Ingest ─────────────────────────────────────────────────────────────────────


class IngestedDoc(BaseModel):
    model_config = {"extra": "ignore"}

    doc_id: str
    filename: str
    doc_type: str
    pages: int
    chars: int
    chunks: dict[str, int]  # strategy → count


class FailedDoc(BaseModel):
    model_config = {"extra": "ignore"}

    filename: str
    error: str


class IngestSummary(BaseModel):
    model_config = {"extra": "ignore"}

    n_processed: int
    n_failed: int
    total_chars: int


class IngestResponse(BaseModel):
    model_config = {"extra": "ignore"}

    processed: list[IngestedDoc]
    failed: list[FailedDoc]
    summary: IngestSummary


# ── Query ──────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    model_config = {"extra": "ignore"}

    question: str = Field(min_length=0)  # length validated in route to return 400 not 422
    config_override: dict[str, Any] | None = None
    return_chunks: bool = True


class RetrievedChunkOut(BaseModel):
    model_config = {"extra": "ignore"}

    chunk_id: str
    score: float
    rank: int
    retriever: str
    text: str
    page: int


class CitationOut(BaseModel):
    model_config = {"extra": "ignore"}

    doc_id: str
    page: int
    chunk_id: str


class QueryResponse(BaseModel):
    model_config = {"extra": "ignore"}

    query: str
    answer: str
    refused: bool
    citations: list[CitationOut]
    retrieved: list[RetrievedChunkOut]
    model: str
    latency_ms: int
    cost_usd: float
    warnings: list[str]


# ── Eval ───────────────────────────────────────────────────────────────────────


class EvalRunRequest(BaseModel):
    model_config = {"extra": "ignore"}

    config_path: str = "configs/experiment_matrix.yaml"
    notes: str | None = None
    subset: list[str] | None = None


class EvalRunQueued(BaseModel):
    model_config = {"extra": "ignore"}

    run_id: str
    status: Literal["queued"]
    n_configs: int
    n_questions: int
    estimated_cost_usd: float
    estimated_duration_s: float


class BestMetric(BaseModel):
    model_config = {"extra": "ignore"}

    name: str
    value: float


class RunSummary(BaseModel):
    model_config = {"extra": "ignore"}

    run_id: str
    config_hash: str
    chunking: str
    embedding: str
    started_at: str
    ended_at: str
    n_questions: int
    best_metric: BestMetric
    total_cost_usd: float
    status: str


class RunsListResponse(BaseModel):
    model_config = {"extra": "ignore"}

    runs: list[RunSummary]
    total: int
