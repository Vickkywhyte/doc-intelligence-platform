"""Unit tests for the ingestion package.

Covers:
- PdfPlumberLoader: text extraction, page count, char offsets
- clean_text / clean_pages: whitespace, dehyphenation, header/footer strip
- infer_doc_type: filename + text heuristics
- extract_file_metadata: size, extension, mtime
- load_document: round-trip, doc_id stability, OCRRequired edge case
- sample_document.json fixture: schema validity
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docintel.exceptions import IngestionError, OCRRequired
from docintel.ingestion.cleaner import clean_pages, clean_text, strip_headers_footers
from docintel.ingestion.loader import (
    Document,
    Page,
    PdfPlumberLoader,
    load_document,
)
from docintel.ingestion.metadata import extract_file_metadata, infer_doc_type

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_pdf_with_text(path: Path, pages_text: list[str]) -> None:
    """Write a minimal, text-extractable PDF using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    story = []
    for i, text in enumerate(pages_text):
        story.append(Paragraph(text, styles["Normal"]))
        if i < len(pages_text) - 1:
            story.append(PageBreak())
    doc.build(story)


# ── cleaner.clean_text ─────────────────────────────────────────────────────────


def test_clean_text_collapses_spaces() -> None:
    result = clean_text("hello   world  here")
    assert "  " not in result


def test_clean_text_dehyphenation() -> None:
    result = clean_text("docu-\nment processing")
    assert "document" in result
    assert "-\n" not in result


def test_clean_text_collapses_blank_lines() -> None:
    result = clean_text("line1\n\n\n\n\nline2")
    assert "\n\n\n" not in result


def test_clean_text_strips_trailing_whitespace() -> None:
    result = clean_text("hello   \nworld   ")
    for line in result.split("\n"):
        assert line == line.rstrip()


def test_clean_text_normalizes_crlf() -> None:
    result = clean_text("line1\r\nline2\rline3")
    assert "\r" not in result


def test_clean_text_preserves_content() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    assert clean_text(text) == text


# ── cleaner.strip_headers_footers ─────────────────────────────────────────────


def _make_pages(texts: list[str]) -> list[Page]:
    """Create Page objects with approximate char offsets for testing."""
    pages = []
    offset = 0
    for i, text in enumerate(texts):
        pages.append(
            Page(page_number=i + 1, text=text, char_start=offset, char_end=offset + len(text))
        )
        offset += len(text)
    return pages


def test_strip_headers_footers_removes_repeating_line() -> None:
    texts = [
        "TECHCORP CONFIDENTIAL\nContent of page one. This is important text.\nPage 1",
        "TECHCORP CONFIDENTIAL\nContent of page two. More important text here.\nPage 2",
        "TECHCORP CONFIDENTIAL\nContent of page three. Even more content here.\nPage 3",
        "TECHCORP CONFIDENTIAL\nContent of page four. Additional relevant content.\nPage 4",
    ]
    pages = _make_pages(texts)
    cleaned = strip_headers_footers(pages)
    for page in cleaned:
        assert "TECHCORP CONFIDENTIAL" not in page.text


def test_strip_headers_footers_preserves_unique_content() -> None:
    texts = [
        "Header\nSection 1: This is unique content for page one only.\nFooter",
        "Header\nSection 2: This is unique content for page two only.\nFooter",
        "Header\nSection 3: This is unique content for page three only.\nFooter",
        "Header\nSection 4: This is unique content for page four only.\nFooter",
    ]
    pages = _make_pages(texts)
    cleaned = strip_headers_footers(pages)
    assert any("unique content for page one" in p.text for p in cleaned)


def test_strip_headers_footers_noop_single_page() -> None:
    pages = _make_pages(["Header\nBody text here.\nFooter"])
    result = strip_headers_footers(pages)
    assert result[0].text == pages[0].text


def test_clean_pages_runs_both_steps() -> None:
    texts = [
        "HEADER\nHello   world.\nFOOTER",
        "HEADER\nSecond   page.\nFOOTER",
        "HEADER\nThird   page.\nFOOTER",
        "HEADER\nFourth   page.\nFOOTER",
    ]
    pages = _make_pages(texts)
    cleaned = clean_pages(pages)
    for page in cleaned:
        assert "HEADER" not in page.text
        assert "  " not in page.text


# ── metadata.infer_doc_type ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "filename, text_prefix, expected",
    [
        ("msa_acme_2024.pdf", "Master Services Agreement between parties", "contract"),
        ("nda_globex.pdf", "Non-Disclosure Agreement entered into", "nda"),
        ("dpa_vendor.pdf", "Data Processing Agreement pursuant to GDPR", "dpa"),
        ("policy_hr.pdf", "This policy establishes employment standards", "policy"),
        ("handbook_employee_2024.pdf", "Welcome to TechCorp employee handbook", "handbook"),
        ("rfp_cloud_infra.pdf", "Request for Proposal for cloud infrastructure", "rfp"),
        ("unknown_doc.pdf", "This document contains general information.", "other"),
    ],
)
def test_infer_doc_type(filename: str, text_prefix: str, expected: str) -> None:
    assert infer_doc_type(filename, text_prefix) == expected


# ── metadata.extract_file_metadata ────────────────────────────────────────────


def test_extract_file_metadata_fields(tmp_path: Path) -> None:
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 content")
    meta = extract_file_metadata(f)
    assert meta["file_size_bytes"] == f.stat().st_size
    assert meta["extension"] == ".pdf"
    assert "mtime_utc" in meta
    assert meta["mtime_utc"].endswith("+00:00")


# ── PdfPlumberLoader ───────────────────────────────────────────────────────────


def test_pdfplumber_loader_extracts_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"
    _make_pdf_with_text(pdf_path, ["Hello from page one.", "Hello from page two."])
    loader = PdfPlumberLoader()
    pages = loader.load(pdf_path)
    assert len(pages) == 2
    full = " ".join(p.text for p in pages)
    assert "Hello" in full


def test_pdfplumber_loader_page_numbers_sequential(tmp_path: Path) -> None:
    pdf_path = tmp_path / "multi.pdf"
    _make_pdf_with_text(pdf_path, ["Page A text.", "Page B text.", "Page C text."])
    pages = PdfPlumberLoader().load(pdf_path)
    assert [p.page_number for p in pages] == [1, 2, 3]


def test_pdfplumber_loader_char_offsets_accumulate(tmp_path: Path) -> None:
    pdf_path = tmp_path / "offsets.pdf"
    _make_pdf_with_text(pdf_path, ["First page text here.", "Second page text here."])
    pages = PdfPlumberLoader().load(pdf_path)
    # char_start of page 2 must equal char_end of page 1
    assert pages[1].char_start == pages[0].char_end


def test_pdfplumber_loader_raises_ocr_required_on_empty_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    # Patch pdfplumber to return pages with empty text (simulates scanned PDF)
    mock_page = MagicMock()
    mock_page.extract_text.return_value = None

    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]

    with patch("pdfplumber.open", return_value=mock_pdf):
        pdf_path.write_bytes(b"%PDF-1.4")
        with pytest.raises(OCRRequired):
            PdfPlumberLoader().load(pdf_path)


def test_pdfplumber_loader_raises_ingestion_error_on_missing_file() -> None:
    with pytest.raises(IngestionError):
        PdfPlumberLoader().load(Path("/nonexistent/file.pdf"))


# ── load_document ──────────────────────────────────────────────────────────────


def test_load_document_returns_document(tmp_path: Path) -> None:
    pdf_path = tmp_path / "contract.pdf"
    _make_pdf_with_text(pdf_path, ["Master Services Agreement content."])
    doc = load_document(pdf_path)
    assert isinstance(doc, Document)
    assert doc.filename == "contract.pdf"


def test_load_document_doc_id_is_stable(tmp_path: Path) -> None:
    pdf_path = tmp_path / "stable.pdf"
    _make_pdf_with_text(pdf_path, ["Stable content."])
    doc1 = load_document(pdf_path)
    doc2 = load_document(pdf_path)
    assert doc1.doc_id == doc2.doc_id


def test_load_document_doc_id_format(tmp_path: Path) -> None:
    pdf_path = tmp_path / "fmt.pdf"
    _make_pdf_with_text(pdf_path, ["Content."])
    doc = load_document(pdf_path)
    assert len(doc.doc_id) == 16
    int(doc.doc_id, 16)  # must be valid hex


def test_load_document_raises_for_missing_file() -> None:
    with pytest.raises(IngestionError, match="not found"):
        load_document("/nonexistent/path.pdf")


def test_load_document_ingest_at_is_utc(tmp_path: Path) -> None:
    pdf_path = tmp_path / "ts.pdf"
    _make_pdf_with_text(pdf_path, ["Timestamp test."])
    doc = load_document(pdf_path)
    assert doc.ingested_at.tzinfo is not None


def test_load_document_full_text_equals_concat_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "concat.pdf"
    _make_pdf_with_text(pdf_path, ["First page.", "Second page."])
    doc = load_document(pdf_path)
    assert doc.full_text == "".join(p.text for p in doc.pages)


# ── sample_document.json fixture ──────────────────────────────────────────────


def test_sample_document_json_is_valid_schema() -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_document.json"
    assert fixture.exists(), "tests/fixtures/sample_document.json must exist"
    raw = json.loads(fixture.read_text())
    doc = Document.model_validate(raw)
    assert doc.doc_id == "a1b2c3d4e5f6a7b8"
    assert len(doc.pages) == 3
    assert doc.doc_type == "nda"


def test_sample_document_char_offsets_consistent() -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_document.json"
    doc = Document.model_validate(json.loads(fixture.read_text()))
    for i in range(len(doc.pages) - 1):
        assert doc.pages[i + 1].char_start == doc.pages[i].char_end
