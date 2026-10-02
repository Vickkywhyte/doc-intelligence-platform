"""Recursive chunker: separator-hierarchy splitting with overlap merging.

Algorithm:
  1. Recursively split text on a priority-ordered separator list until every
     atomic piece fits within `size` tokens.  Character offsets are tracked
     throughout so no re-searching is needed.
  2. Greedily merge adjacent atomic pieces into chunks up to `size` tokens,
     then back up by at most `overlap` tokens for the next chunk's start.

Separator hierarchy (highest to lowest priority):
  "\\n\\n"  →  "\\n"  →  ". "  →  "! "  →  "? "  →  " "  →  "" (token split)
"""

from __future__ import annotations

import structlog

from docintel.ingestion.loader import Document
from docintel.utils.config import ChunkingConfig

from .base import Chunk, ChunkStrategy, _enc, count_tokens, page_for_offset

log = structlog.get_logger()

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


class RecursiveChunker(ChunkStrategy):
    """Separator-hierarchy recursive chunker (ADR-004)."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._size = config.size
        self._overlap = config.overlap

    @property
    def name(self) -> str:
        return "recursive"

    def chunk(self, doc: Document) -> list[Chunk]:
        full_text = doc.full_text
        if not full_text.strip():
            return []

        # Step 1: split into atomic (char_start, text) pieces
        pieces = self._split(full_text, 0, _SEPARATORS)
        if not pieces:
            return []

        # Step 2: merge pieces into final chunks with overlap
        windows = self._merge(pieces, full_text)

        chunks: list[Chunk] = []
        for index, (char_start, char_end) in enumerate(windows):
            chunk_text = full_text[char_start:char_end]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:recursive:{index:05d}",
                    doc_id=doc.doc_id,
                    strategy="recursive",
                    index=index,
                    text=chunk_text,
                    page=page_for_offset(doc, char_start),
                    char_start=char_start,
                    char_end=char_end,
                    token_count=count_tokens(chunk_text),
                )
            )

        log.debug(
            "recursive_chunk_complete",
            doc_id=doc.doc_id,
            chunks=len(chunks),
            pieces=len(pieces),
        )
        return chunks

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _split(
        self, text: str, base_offset: int, separators: list[str]
    ) -> list[tuple[int, str]]:
        """Return (char_start, piece_text) pairs for each atomic piece.

        Pieces are guaranteed to be ≤ self._size tokens (or as small as
        possible when a single word exceeds the limit).
        """
        if not text.strip():
            return []

        if count_tokens(text) <= self._size:
            return [(base_offset, text)]

        for i, sep in enumerate(separators):
            if sep == "":
                # Last resort: split at token boundaries
                tokens = _enc.encode(text)
                pieces: list[tuple[int, str]] = []
                pos = base_offset
                for j in range(0, len(tokens), self._size):
                    piece = _enc.decode(tokens[j : j + self._size])
                    pieces.append((pos, piece))
                    pos += len(piece)
                return pieces

            if sep not in text:
                continue

            # Split on this separator, recurse on each part
            results: list[tuple[int, str]] = []
            pos = base_offset
            for part in text.split(sep):
                sub = self._split(part, pos, separators[i + 1 :])
                results.extend(sub)
                pos += len(part) + len(sep)
            return results if results else [(base_offset, text)]

        return [(base_offset, text)]

    def _merge(
        self, pieces: list[tuple[int, str]], full_text: str
    ) -> list[tuple[int, int]]:
        """Merge atomic pieces into (char_start, char_end) chunk windows.

        Greedily fills up to self._size tokens, then backs up by up to
        self._overlap tokens for the start of the next chunk.

        chunk_end is set to the start of the first EXCLUDED piece so that any
        separator characters consumed during splitting (e.g. ". ") are included
        in the chunk text.  The last chunk extends to len(full_text).
        """
        windows: list[tuple[int, int]] = []
        i = 0

        while i < len(pieces):
            # Build the current chunk
            chunk_start = pieces[i][0]
            chunk_tokens = count_tokens(pieces[i][1])
            j = i + 1

            while j < len(pieces):
                pt = count_tokens(pieces[j][1])
                if chunk_tokens + pt > self._size:
                    break
                chunk_tokens += pt
                j += 1

            # Extend chunk_end to the start of the next piece so separators
            # consumed during splitting are captured inside the chunk.
            chunk_end = pieces[j][0] if j < len(pieces) else len(full_text)

            windows.append((chunk_start, chunk_end))

            if j >= len(pieces):
                break

            # Find next chunk start: back up from j-1 until overlap budget exhausted
            overlap_tokens = 0
            next_i = j - 1
            while next_i > i:
                pt = count_tokens(pieces[next_i][1])
                if overlap_tokens + pt > self._overlap:
                    break
                overlap_tokens += pt
                next_i -= 1
            i = max(i + 1, next_i)

        return windows
