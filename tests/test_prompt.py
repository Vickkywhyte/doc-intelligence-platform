"""Tests for prompt_templates.py — system prompt, refusal detection (Phase 6)."""

from __future__ import annotations

import pytest

from docintel.chunking.base import Chunk
from docintel.generation.prompt_templates import (
    REFUSAL_PHRASES,
    SCORE_FLOOR,
    SYSTEM_PROMPT,
    build_user_prompt,
    is_refusal,
)
from docintel.vectorstore.base import RetrievedChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_rc(score: float = 0.8, index: int = 0) -> RetrievedChunk:
    doc_id = "abcdef1234567890"
    chunk = Chunk(
        chunk_id=f"{doc_id}:fixed:{index:05d}",
        doc_id=doc_id,
        strategy="fixed",
        index=index,
        text="The payment terms are net-30 from invoice date.",
        page=2,
        char_start=0,
        char_end=48,
        token_count=9,
    )
    return RetrievedChunk(chunk=chunk, score=score, rank=index + 1, retriever="topk")


# ── System prompt ─────────────────────────────────────────────────────────────

def test_system_prompt_non_empty() -> None:
    assert len(SYSTEM_PROMPT) > 50


def test_system_prompt_contains_citation_format() -> None:
    assert "[doc_id:page:chunk_id]" in SYSTEM_PROMPT


def test_system_prompt_contains_refusal_instruction() -> None:
    assert "I don't know based on the provided documents" in SYSTEM_PROMPT


def test_system_prompt_prompt_injection_warning() -> None:
    assert "not instructions" in SYSTEM_PROMPT.lower() or "data" in SYSTEM_PROMPT.lower()


# ── Refusal phrase list ───────────────────────────────────────────────────────

def test_refusal_phrases_non_empty() -> None:
    assert len(REFUSAL_PHRASES) >= 5


def test_refusal_phrases_all_lowercase() -> None:
    for phrase in REFUSAL_PHRASES:
        assert phrase == phrase.lower(), f"Not lowercase: {phrase!r}"


def test_refusal_phrases_canonical_present() -> None:
    canonical = "i don't know based on the provided documents"
    assert canonical in REFUSAL_PHRASES


# ── build_user_prompt ─────────────────────────────────────────────────────────

def test_build_user_prompt_contains_question() -> None:
    question = "What is the payment term?"
    result = build_user_prompt(question, [])
    assert question in result


def test_build_user_prompt_contains_chunk_id() -> None:
    rc = _make_rc()
    result = build_user_prompt("q?", [rc])
    assert rc.chunk.chunk_id in result


def test_build_user_prompt_contains_citation_key() -> None:
    rc = _make_rc()
    result = build_user_prompt("q?", [rc])
    assert "CITATION KEY:" in result


def test_build_user_prompt_contains_doc_id() -> None:
    rc = _make_rc()
    result = build_user_prompt("q?", [rc])
    assert rc.chunk.doc_id in result


def test_build_user_prompt_contains_page() -> None:
    rc = _make_rc()
    result = build_user_prompt("q?", [rc])
    assert str(rc.chunk.page) in result


def test_build_user_prompt_contains_text() -> None:
    rc = _make_rc()
    result = build_user_prompt("q?", [rc])
    assert rc.chunk.text in result


def test_build_user_prompt_empty_chunks() -> None:
    result = build_user_prompt("q?", [])
    assert "no chunks" in result.lower() or "CHUNKS" in result


def test_build_user_prompt_multiple_chunks() -> None:
    rcs = [_make_rc(index=i) for i in range(3)]
    result = build_user_prompt("q?", rcs)
    for rc in rcs:
        assert rc.chunk.chunk_id in result


# ── is_refusal — phrase detection (condition 1) ───────────────────────────────

@pytest.mark.parametrize("phrase", [
    "I don't know based on the provided documents.",
    "I cannot answer this question.",
    "Not in the provided documents.",
    "No relevant information was found.",
    "I don't know",
])
def test_is_refusal_phrase_detected(phrase: str) -> None:
    rc = _make_rc(score=0.9)
    assert is_refusal(phrase, [], [rc]) is True


def test_is_refusal_phrase_case_insensitive() -> None:
    assert is_refusal("I DON'T KNOW", [], []) is True
    assert is_refusal("I CANNOT ANSWER", [], []) is True


def test_is_refusal_real_answer_not_refused() -> None:
    from docintel.generation.models import Citation
    rc = _make_rc(score=0.9)
    citation = Citation(doc_id="abcdef1234567890", page=2, chunk_id="abcdef1234567890:fixed:00000")
    answer = "The payment terms are net-30 [abcdef1234567890:2:abcdef1234567890:fixed:00000]."
    assert is_refusal(answer, [citation], [rc]) is False


# ── is_refusal — score floor condition (condition 2) ─────────────────────────

def test_is_refusal_no_citations_and_low_scores() -> None:
    rc = _make_rc(score=0.0)  # score at floor
    assert is_refusal("Something irrelevant.", [], [rc], score_floor=0.0) is True


def test_is_refusal_no_citations_but_high_score() -> None:
    rc = _make_rc(score=0.9)
    assert is_refusal("Something irrelevant.", [], [rc], score_floor=0.0) is False


def test_is_refusal_has_citations_overrides_score() -> None:
    from docintel.generation.models import Citation
    rc = _make_rc(score=0.0)
    citation = Citation(doc_id="abcdef1234567890", page=2, chunk_id="x:fixed:00000")
    assert is_refusal("Some answer.", [citation], [rc], score_floor=0.5) is False


def test_is_refusal_empty_retrieved() -> None:
    assert is_refusal("Some answer.", [], [], score_floor=0.0) is True
