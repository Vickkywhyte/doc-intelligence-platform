"""RAGPipeline — end-to-end: question → retrieve → generate → parse."""

from __future__ import annotations

import time

import structlog

from docintel.generation.citation_parser import parse_citations
from docintel.generation.llm_client import LLMClient
from docintel.generation.models import RAGResponse
from docintel.generation.prompt_templates import build_user_prompt, is_refusal
from docintel.retrieval.retriever import Retriever
from docintel.utils.config import RetrievalConfig

log = structlog.get_logger()


class RAGPipeline:
    """Combines a Retriever with an LLMClient to answer questions with citations.

    Usage:
        pipeline = RAGPipeline(retriever, llm_client, k=5)
        response = pipeline.query("What are the payment terms?")
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        k: int = 5,
        score_floor: float = 0.0,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_client
        self._k = k
        self._score_floor = score_floor

    def query(self, question: str) -> RAGResponse:
        """Run the full RAG pipeline for one question.

        Steps:
        1. Retrieve top-k chunks.
        2. Build the user prompt from question + chunks.
        3. Call the LLM.
        4. Parse citations from the answer.
        5. Detect refusal.
        6. Return a fully-populated RAGResponse.
        """
        t0 = time.monotonic()

        retrieved = self._retriever.retrieve(question, k=self._k)
        user_prompt = build_user_prompt(question, retrieved)

        llm_response = self._llm.generate(user_prompt)

        valid_ids = {r.chunk.chunk_id for r in retrieved}
        citations = parse_citations(llm_response.text, valid_chunk_ids=valid_ids)
        refused = is_refusal(
            llm_response.text,
            citations,
            retrieved,
            score_floor=self._score_floor,
        )

        total_latency_ms = round((time.monotonic() - t0) * 1000)

        warnings: list[str] = []
        if not refused and not citations:
            warnings.append("answer_has_no_citations")

        log.info(
            "rag_query_complete",
            refused=refused,
            citations=len(citations),
            retrieved=len(retrieved),
            latency_ms=total_latency_ms,
            cost_usd=llm_response.cost_usd,
        )

        return RAGResponse(
            query=question,
            answer=llm_response.text,
            refused=refused,
            citations=citations,
            retrieved=retrieved,
            model=llm_response.model,
            latency_ms=total_latency_ms,
            cost_usd=llm_response.cost_usd,
            warnings=warnings,
        )
