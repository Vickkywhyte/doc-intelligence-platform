"""Phase 6 — Smoke-test the full RAG pipeline with 3 + 1 questions.

Expected results:
  Questions 1–3  (in-corpus): ≥ 1 citation each, refused=False
  Question  4    (OOD):       refused=True

Runs without a real API key: if OPENROUTER_API_KEY is absent or USE_HOSTED is
not "true", a _SmokeFakeLLM generates plausible synthetic responses so the
pipeline flow can be verified end-to-end.

Run: make rag-smoke
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv()

from docintel.embeddings.registry import get_embedder
from docintel.generation.llm_client import LLMClient, OpenRouterClient
from docintel.generation.models import LLMResponse
from docintel.generation.prompt_templates import SYSTEM_PROMPT
from docintel.pipeline.rag_pipeline import RAGPipeline
from docintel.retrieval.retriever import DenseRetriever
from docintel.utils.config import load_config
from docintel.vectorstore.registry import get_store

log = structlog.get_logger()

# 3 in-corpus (citations expected) + 1 OOD (refusal expected)
SMOKE_QUESTIONS = [
    ("What information is excluded from the confidentiality obligations in the NDAs?", True),
    ("What confidentiality obligations apply to the parties?", True),
    ("What services and deliverables are described in the MSAs?", True),
    ("What is the capital of France?", False),
]

_CITATION_KEY_RE = re.compile(r"CITATION KEY: (\[[^\]]+\])")


class _SmokeFakeLLM(LLMClient):
    """Synthetic LLM for smoke tests when OPENROUTER_API_KEY is not set.

    - For in-corpus questions: parses the first chunk reference from the user
      prompt and generates a one-sentence answer with a valid citation.
    - For the OOD question (detected by the absence of business-document
      vocabulary in the question): returns the canonical refusal phrase.
    """

    _OOD_KEYWORDS = {"capital", "france", "paris", "weather", "recipe", "sports"}

    @property
    def model_name(self) -> str:
        return "smoke-fake"

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> LLMResponse:
        question_line = next(
            (l for l in prompt.splitlines() if l.startswith("QUESTION:")), ""
        )
        question_lower = question_line.lower()

        if any(kw in question_lower for kw in self._OOD_KEYWORDS):
            text = "I don't know based on the provided documents."
        else:
            m = _CITATION_KEY_RE.search(prompt)
            if m:
                cite_key = m.group(1)  # e.g. [doc_id:page:chunk_id]
                text = (
                    f"According to the documents, the relevant information is available "
                    f"{cite_key}."
                )
            else:
                text = "I don't know based on the provided documents."

        return LLMResponse(
            text=text,
            model=self.model_name,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            cost_usd=0.0,
            latency_ms=0,
        )


def _build_pipeline(cfg, strategy: str, embedder_name: str) -> RAGPipeline:
    emb_cfg = next(e for e in cfg.embeddings if e.name == embedder_name)
    embedder = get_embedder(emb_cfg)
    store = get_store(strategy, embedder_name)
    retriever = DenseRetriever(store, embedder, cfg.retrieval)

    use_hosted = os.environ.get("USE_HOSTED", "false").lower() == "true"
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if use_hosted and api_key:
        llm: LLMClient = OpenRouterClient(cfg.generation, system_prompt=SYSTEM_PROMPT)
        log.info("smoke_using_openrouter", model=cfg.generation.model)
    else:
        llm = _SmokeFakeLLM()
        log.info("smoke_using_fake_llm", reason="OPENROUTER_API_KEY not set or USE_HOSTED!=true")

    return RAGPipeline(retriever, llm, k=cfg.retrieval.k)


def main() -> None:
    cfg = load_config()

    # Use first configured strategy + first embedder for smoke
    strategy = cfg.chunking[0].name
    embedder_name = cfg.embeddings[0].name

    pipeline = _build_pipeline(cfg, strategy, embedder_name)

    print(f"\n{'='*70}")
    print(f"  RAG Smoke Test — {strategy} × {embedder_name}")
    print(f"{'='*70}\n")

    passed = 0
    for i, (question, expect_answer) in enumerate(SMOKE_QUESTIONS, start=1):
        response = pipeline.query(question)

        status = "PASS"
        if expect_answer and (response.refused or not response.citations):
            status = "FAIL"
        elif not expect_answer and not response.refused:
            status = "FAIL"

        if status == "PASS":
            passed += 1

        label = "in-corpus" if expect_answer else "OOD"
        print(f"[{i}] {label} | {status}")
        print(f"    Q: {question}")
        print(f"    A: {response.answer[:120]}")
        print(f"    refused={response.refused}  citations={len(response.citations)}  "
              f"retrieved={len(response.retrieved)}  "
              f"latency={response.latency_ms}ms  cost=${response.cost_usd:.6f}")
        if response.citations:
            for c in response.citations:
                print(f"    → [{c.doc_id}:{c.page}:{c.chunk_id}]")
        if response.warnings:
            print(f"    warnings: {response.warnings}")
        print()

    print(f"Result: {passed}/{len(SMOKE_QUESTIONS)} checks passed")
    sys.exit(0 if passed == len(SMOKE_QUESTIONS) else 1)


if __name__ == "__main__":
    main()
