"""Fixed-window chunker: token-aware sliding windows with overlap.

Windows are defined in token space using cl100k_base BPE. Each window is
[tok_start, tok_start + size) tokens; consecutive windows step by (size -
overlap) tokens so adjacent chunks share an overlap-token prefix/suffix.
"""

from __future__ import annotations

import structlog

from docintel.ingestion.loader import Document
from docintel.utils.config import ChunkingConfig

from .base import Chunk, ChunkStrategy, _enc, count_tokens, page_for_offset

log = structlog.get_logger()


class FixedChunker(ChunkStrategy):
    """Token-aware fixed-window chunker (ADR-004)."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._size = config.size
        self._overlap = config.overlap

    @property
    def name(self) -> str:
        return "fixed"

    def chunk(self, doc: Document) -> list[Chunk]:
        full_text = doc.full_text
        if not full_text.strip():
            return []

        tokens = _enc.encode(full_text)
        step = max(1, self._size - self._overlap)

        chunks: list[Chunk] = []
        index = 0
        tok_start = 0

        while tok_start < len(tokens):
            tok_end = min(tok_start + self._size, len(tokens))

            # Decode prefix to get exact char_start; prefix decoding is lossless.
            prefix_text = _enc.decode(tokens[:tok_start])
            chunk_text = _enc.decode(tokens[tok_start:tok_end])

            char_start = len(prefix_text)
            char_end = char_start + len(chunk_text)

            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:fixed:{index:05d}",
                    doc_id=doc.doc_id,
                    strategy="fixed",
                    index=index,
                    text=chunk_text,
                    page=page_for_offset(doc, char_start),
                    char_start=char_start,
                    char_end=char_end,
                    token_count=tok_end - tok_start,
                )
            )

            index += 1
            tok_start += step

        log.debug(
            "fixed_chunk_complete",
            doc_id=doc.doc_id,
            chunks=len(chunks),
            total_tokens=len(tokens),
        )
        return chunks
