"""Tests for citation_parser.py — regex extraction and validation (Phase 6)."""

from __future__ import annotations

import pytest

from docintel.generation.citation_parser import parse_citations
from docintel.generation.models import Citation


# ── Basic parsing ─────────────────────────────────────────────────────────────

def test_parse_single_citation() -> None:
    text = "The contract is one year [ae958268:1:ae958268abcdef12:fixed:00002]."
    result = parse_citations(text)
    assert len(result) == 1
    c = result[0]
    assert c.doc_id == "ae958268"
    assert c.page == 1
    assert c.chunk_id == "ae958268abcdef12:fixed:00002"


def test_parse_multiple_citations() -> None:
    text = (
        "Payment is net-30 [abc123:2:abc123def456:fixed:00001]. "
        "Confidentiality applies [def456:3:def456abc789:recursive:00005]."
    )
    result = parse_citations(text)
    assert len(result) == 2
    assert result[0].page == 2
    assert result[1].page == 3


def test_parse_no_citations() -> None:
    assert parse_citations("I don't know.") == []
    assert parse_citations("") == []


def test_parse_deduplicates() -> None:
    text = (
        "Claim A [abc123:1:abc123:fixed:00000]. "
        "Claim B [abc123:1:abc123:fixed:00000]."
    )
    result = parse_citations(text)
    assert len(result) == 1


def test_parse_returns_citation_objects() -> None:
    text = "[abc:5:abc:fixed:00001] something."
    result = parse_citations(text)
    assert all(isinstance(c, Citation) for c in result)


def test_parse_preserves_order() -> None:
    text = "[a:1:a:f:00001] [b:2:b:f:00002] [c:3:c:f:00003]"
    result = parse_citations(text)
    assert [c.page for c in result] == [1, 2, 3]


def test_parse_chunk_id_with_multiple_colons() -> None:
    text = "[ae958268:3:ae958268abcdef12:recursive:00007] text."
    result = parse_citations(text)
    assert len(result) == 1
    assert result[0].chunk_id == "ae958268abcdef12:recursive:00007"


# ── Validation against retrieved chunk IDs ────────────────────────────────────

def test_validate_known_chunk_id_passes() -> None:
    chunk_id = "ae958268abcdef12:fixed:00002"
    text = f"[ae958268:1:{chunk_id}]."
    result = parse_citations(text, valid_chunk_ids={chunk_id})
    assert len(result) == 1


def test_validate_unknown_chunk_id_excluded(caplog) -> None:
    import logging
    text = "[ae958268:1:ae958268abcdef12:fixed:00099]."
    with caplog.at_level(logging.WARNING):
        result = parse_citations(text, valid_chunk_ids={"ae958268abcdef12:fixed:00000"})
    assert result == []


def test_validate_none_skips_validation() -> None:
    text = "[abc:1:any:chunk:id]."
    result = parse_citations(text, valid_chunk_ids=None)
    assert len(result) == 1


def test_validate_mixed_valid_invalid() -> None:
    valid_id = "abc:fixed:00001"
    invalid_id = "xyz:fixed:00099"
    text = f"[a:1:{valid_id}] and [b:2:{invalid_id}]."
    result = parse_citations(text, valid_chunk_ids={valid_id})
    assert len(result) == 1
    assert result[0].chunk_id == valid_id


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_no_match_on_malformed_no_page() -> None:
    text = "[only_chunk_id_no_page]"
    result = parse_citations(text)
    assert result == []


def test_no_match_on_text_without_brackets() -> None:
    text = "No citations here at all."
    assert parse_citations(text) == []


def test_citation_in_middle_of_sentence() -> None:
    text = "The agreement states [abc:2:abc:f:00003] that payment is due monthly."
    result = parse_citations(text)
    assert len(result) == 1
    assert result[0].page == 2
