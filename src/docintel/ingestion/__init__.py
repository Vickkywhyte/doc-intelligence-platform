"""Ingestion package: load raw files into Document objects.

Public API::

    from docintel.ingestion import Document, Page, DocType, load_document
"""

from __future__ import annotations

from docintel.ingestion.loader import (
    DocType,
    Document,
    DocumentLoader,
    Page,
    REGISTRY,
    load_document,
)

__all__ = [
    "DocType",
    "Document",
    "DocumentLoader",
    "Page",
    "REGISTRY",
    "load_document",
]
