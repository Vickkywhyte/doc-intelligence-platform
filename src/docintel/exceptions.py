"""Typed exceptions for the Document Intelligence Platform.

All exceptions raised by library code must subclass DocIntelError so
callers can catch the base type for blanket handling.
"""

from __future__ import annotations


class DocIntelError(Exception):
    """Base class for all docintel errors."""


class ConfigError(DocIntelError):
    """Raised when a config file is missing, malformed, or fails schema validation."""


class BudgetExceededError(DocIntelError):
    """Raised when cumulative spend would exceed evaluation.budget_usd."""

    def __init__(self, spent: float, budget: float) -> None:
        self.spent = spent
        self.budget = budget
        super().__init__(
            f"Budget exceeded: spent ${spent:.4f} of ${budget:.2f} limit"
        )


class OCRRequired(DocIntelError):
    """Raised when a PDF contains no extractable text and OCR is unavailable."""


class IngestionError(DocIntelError):
    """Raised for unrecoverable errors during document ingestion."""


class ChunkingError(DocIntelError):
    """Raised when a chunking strategy fails to produce valid chunks."""


class EmbeddingError(DocIntelError):
    """Raised when an embedder call fails (network, quota, shape mismatch)."""


class RetrievalError(DocIntelError):
    """Raised when the vector store or retriever returns an unexpected result."""


class GenerationError(DocIntelError):
    """Raised when the LLM client fails to produce a response."""


class EvalError(DocIntelError):
    """Raised for errors in the evaluation pipeline (metric computation, judge calls)."""
