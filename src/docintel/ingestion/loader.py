"""Document loaders: ABC + PdfPlumberLoader (default) + UnstructuredLoader (optional).

Defines the Document and Page Pydantic models (DATA_MODEL.md §2.1) and the
DocumentLoader ABC. Implementations are registered in REGISTRY.

Usage:
    from docintel.ingestion.loader import load_document

    doc = load_document("data/raw/contract.pdf")
    print(doc.doc_id, doc.total_chars)
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

from docintel.exceptions import IngestionError, OCRRequired

log = structlog.get_logger()

# ── Core data models (DATA_MODEL.md §2.1) ─────────────────────────────────────

DocType = Literal["contract", "policy", "handbook", "rfp", "dpa", "nda", "other"]


class Page(BaseModel):
    """One page of a parsed document with character offsets into the full text."""

    model_config = {"extra": "ignore"}

    page_number: int = Field(ge=1)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)


class Document(BaseModel):
    """A parsed, cleaned document ready for chunking.

    doc_id is sha256(file_bytes)[:16] — stable across machines.
    """

    model_config = {"extra": "ignore"}

    doc_id: str
    source_path: str
    filename: str
    doc_type: DocType = "other"
    pages: list[Page]
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime

    @property
    def full_text(self) -> str:
        """Concatenation of all page texts."""
        return "".join(p.text for p in self.pages)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


# ── Abstract base ──────────────────────────────────────────────────────────────


class DocumentLoader(ABC):
    """Extract a list of Pages from a file path."""

    @abstractmethod
    def load(self, path: Path) -> list[Page]:
        """Return pages in reading order with accumulated char offsets.

        Raises IngestionError on format errors, OCRRequired when text is absent.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry key, e.g. 'pdfplumber'."""
        ...


# ── PdfPlumber implementation (always available) ───────────────────────────────


class PdfPlumberLoader(DocumentLoader):
    """Extract text from PDFs using pdfplumber. No system dependencies required."""

    @property
    def name(self) -> str:
        return "pdfplumber"

    def load(self, path: Path) -> list[Page]:
        import pdfplumber

        pages: list[Page] = []
        char_offset = 0

        try:
            with pdfplumber.open(str(path)) as pdf:
                if not pdf.pages:
                    raise IngestionError(f"No pages in {path.name}")

                for i, pdf_page in enumerate(pdf.pages):
                    text: str = pdf_page.extract_text() or ""
                    char_start = char_offset
                    char_end = char_start + len(text)
                    pages.append(
                        Page(
                            page_number=i + 1,
                            text=text,
                            char_start=char_start,
                            char_end=char_end,
                        )
                    )
                    char_offset = char_end
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"pdfplumber failed on {path.name}: {exc}") from exc

        if pages and not any(p.text.strip() for p in pages):
            raise OCRRequired(
                f"No extractable text in {path.name}; PDF may be scanned"
            )

        return pages


# ── Unstructured implementation (optional; requires system deps) ───────────────


class UnstructuredLoader(DocumentLoader):
    """Extract text via the unstructured library (richer layout parsing).

    Falls back gracefully if unstructured is not installed.
    """

    @property
    def name(self) -> str:
        return "unstructured"

    def load(self, path: Path) -> list[Page]:
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as exc:
            raise IngestionError(
                "unstructured is not installed; install it or use loader='pdfplumber'"
            ) from exc

        try:
            elements = partition_pdf(str(path))
        except Exception as exc:
            raise IngestionError(
                f"unstructured failed on {path.name}: {exc}"
            ) from exc

        # Group element texts by page number
        pages_dict: dict[int, list[str]] = {}
        for elem in elements:
            pg: int = getattr(getattr(elem, "metadata", None), "page_number", None) or 1
            pages_dict.setdefault(pg, []).append(str(elem))

        if not pages_dict:
            raise OCRRequired(
                f"No extractable text in {path.name}; PDF may be scanned"
            )

        pages: list[Page] = []
        char_offset = 0
        for pg_num in sorted(pages_dict):
            text = "\n".join(pages_dict[pg_num])
            char_start = char_offset
            char_end = char_start + len(text)
            pages.append(
                Page(
                    page_number=pg_num,
                    text=text,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            char_offset = char_end

        return pages


# ── Plain-text loader (TXT / MD) ───────────────────────────────────────────────


class PlainTextLoader(DocumentLoader):
    """Load plain text files as a single page."""

    @property
    def name(self) -> str:
        return "plaintext"

    def load(self, path: Path) -> list[Page]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise IngestionError(f"Cannot read {path.name}: {exc}") from exc

        return [Page(page_number=1, text=text, char_start=0, char_end=len(text))]


# ── Registry ───────────────────────────────────────────────────────────────────

REGISTRY: dict[str, type[DocumentLoader]] = {
    "pdfplumber": PdfPlumberLoader,
    "unstructured": UnstructuredLoader,
    "plaintext": PlainTextLoader,
}

_PDF_EXTENSIONS = {".pdf"}
_TEXT_EXTENSIONS = {".txt", ".md"}
_UNSTRUCTURED_EXTENSIONS = {".docx", ".doc", ".pptx", ".xlsx", ".eml"}

_UNSTRUCTURED_AVAILABLE: bool | None = None


def _is_unstructured_available() -> bool:
    """Return True only if unstructured.partition.pdf is fully importable."""
    global _UNSTRUCTURED_AVAILABLE
    if _UNSTRUCTURED_AVAILABLE is None:
        try:
            from unstructured.partition.pdf import partition_pdf  # noqa: F401

            _UNSTRUCTURED_AVAILABLE = True
        except Exception:
            # Catches ImportError and transitive ModuleNotFoundError (e.g. pi_heif)
            _UNSTRUCTURED_AVAILABLE = False
    return _UNSTRUCTURED_AVAILABLE


def _auto_select_loader(path: Path) -> DocumentLoader:
    """Choose the best available loader for the given file extension."""
    ext = path.suffix.lower()
    if ext in _TEXT_EXTENSIONS:
        return PlainTextLoader()
    if ext in _UNSTRUCTURED_EXTENSIONS:
        if _is_unstructured_available():
            return UnstructuredLoader()
        raise IngestionError(
            f"unstructured is required to load {ext!r} files but is not installed"
        )
    # Default: pdfplumber for PDFs; try unstructured first if available
    if ext in _PDF_EXTENSIONS and _is_unstructured_available():
        return UnstructuredLoader()
    return PdfPlumberLoader()


# ── Public orchestration function ──────────────────────────────────────────────


def load_document(
    path: str | Path,
    *,
    loader: str = "auto",
    clean: bool = True,
) -> Document:
    """Load, clean, and return a Document for the given file.

    Raises IngestionError on unrecoverable failures (bad file, missing loader).
    Raises OCRRequired when a PDF contains no extractable text.

    Args:
        path: Path to the source file.
        loader: Loader key ('auto', 'pdfplumber', 'unstructured', 'plaintext').
        clean: If True, run the cleaner over extracted pages.
    """
    from docintel.ingestion.cleaner import clean_pages
    from docintel.ingestion.metadata import extract_file_metadata, infer_doc_type

    path = Path(path)
    if not path.exists():
        raise IngestionError(f"File not found: {path}")

    file_bytes = path.read_bytes()
    doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]

    if loader == "auto":
        loader_instance = _auto_select_loader(path)
    else:
        loader_cls = REGISTRY.get(loader)
        if loader_cls is None:
            raise IngestionError(
                f"Unknown loader {loader!r}. Valid: {sorted(REGISTRY)}"
            )
        loader_instance = loader_cls()

    log.info(
        "ingest_document_start",
        filename=path.name,
        doc_id=doc_id,
        loader=loader_instance.name,
    )

    pages = loader_instance.load(path)

    if clean:
        pages = clean_pages(pages)

    full_text = "".join(p.text for p in pages)
    doc_type = infer_doc_type(path.name, full_text)
    file_meta = extract_file_metadata(path)

    return Document(
        doc_id=doc_id,
        source_path=str(path),
        filename=path.name,
        doc_type=doc_type,
        pages=pages,
        metadata=file_meta,
        ingested_at=datetime.now(timezone.utc),
    )
