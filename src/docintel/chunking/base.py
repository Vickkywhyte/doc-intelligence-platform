"""Chunk data model and ChunkStrategy ABC (DATA_MODEL.md §2.2, §3.1).

chunk_id = f"{doc_id}:{strategy}:{index:05d}" — sortable, traceable, stable
across machines (§5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import tiktoken
from pydantic import BaseModel, Field

from docintel.ingestion.loader import Document

# ── Token counter (shared across all strategies) ──────────────────────────────

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (GPT-4 compatible BPE)."""
    return len(_enc.encode(text))


def page_for_offset(doc: Document, char_offset: int) -> int:
    """Return the page number that contains the given character offset."""
    for page in doc.pages:
        if page.char_start <= char_offset < page.char_end:
            return page.page_number
    return doc.pages[-1].page_number if doc.pages else 1


# ── Chunk model (DATA_MODEL.md §2.2) ──────────────────────────────────────────


class Chunk(BaseModel):
    """A single chunk of a document, positioned within the source text.

    char_start/char_end are offsets into Document.full_text, i.e.
    doc.full_text[chunk.char_start:chunk.char_end] == chunk.text.
    """

    model_config = {"extra": "ignore"}

    chunk_id: str
    doc_id: str
    strategy: str
    index: int = Field(ge=0)
    text: str
    page: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    token_count: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Abstract strategy (DATA_MODEL.md §3.1) ────────────────────────────────────


class ChunkStrategy(ABC):
    """Base class for all chunking strategies.

    Concrete subclasses implement chunk() and register themselves in registry.py.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry key, e.g. 'fixed'."""
        ...

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a Document into Chunks.

        Every chunk must satisfy:
          doc.full_text[c.char_start:c.char_end] == c.text
          c.chunk_id == f"{c.doc_id}:{c.strategy}:{c.index:05d}"
        """
        ...
