"""POST /query — ask a question against the indexed corpus."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from api.deps import get_pipeline
from api.schemas import CitationOut, QueryRequest, QueryResponse, RetrievedChunkOut

log = structlog.get_logger()

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    request: Request,
    pipeline=Depends(get_pipeline),
) -> QueryResponse:
    """Run the RAG pipeline for one question."""
    question = body.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "EMPTY_QUESTION",
                "message": "Question must be at least 1 character.",
                "detail": {"field": "question", "received_length": 0},
                "request_id": getattr(request.state, "request_id", ""),
            },
        )
    if len(body.question) > 2000:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "QUESTION_TOO_LONG",
                "message": "Question must be 2000 characters or fewer.",
                "detail": {"field": "question", "received_length": len(question)},
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    try:
        response = pipeline.query(question)
    except Exception as exc:
        log.error("query_failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": str(exc),
                "detail": {},
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    retrieved_out = [
        RetrievedChunkOut(
            chunk_id=rc.chunk.chunk_id,
            score=rc.score,
            rank=rc.rank,
            retriever=rc.retriever,
            text=rc.chunk.text,
            page=rc.chunk.page,
        )
        for rc in response.retrieved
    ] if body.return_chunks else []

    return QueryResponse(
        query=response.query,
        answer=response.answer,
        refused=response.refused,
        citations=[
            CitationOut(doc_id=c.doc_id, page=c.page, chunk_id=c.chunk_id)
            for c in response.citations
        ],
        retrieved=retrieved_out,
        model=response.model,
        latency_ms=response.latency_ms,
        cost_usd=response.cost_usd,
        warnings=response.warnings,
    )
