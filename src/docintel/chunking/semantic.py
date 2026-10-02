"""Semantic chunker: MiniLM embedding-similarity valley detection (ADR-004).

Algorithm:
  1. Split full_text into sentence-level spans using punctuation + newline
     boundaries.
  2. Embed each sentence with sentence-transformers/all-MiniLM-L6-v2
     (fixed internal model — NOT the matrix embedder; see ADR-004).
  3. Compute cosine similarity between consecutive sentence embeddings.
  4. Insert split points at "valleys" where similarity < threshold.
  5. Merge resulting segments up to max_size tokens; hard-split any segment
     that still exceeds max_size by token-level bisection.

The MiniLM model is loaded lazily on first chunk() call.
"""

from __future__ import annotations

import re

import numpy as np
import structlog

from docintel.ingestion.loader import Document
from docintel.utils.config import ChunkingConfig

from .base import Chunk, ChunkStrategy, _enc, count_tokens, page_for_offset

log = structlog.get_logger()

_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Sentence boundary: after .!? followed by whitespace + uppercase/digit/quote,
# or at a double-newline paragraph break.
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z\d\"\'])|(?<=\n)\s*\n')


class SemanticChunker(ChunkStrategy):
    """MiniLM-based semantic chunker (ADR-004).

    Uses all-MiniLM-L6-v2 exclusively for boundary detection; the model
    is never used for retrieval embeddings (those come from Phase 4).
    """

    def __init__(self, config: ChunkingConfig) -> None:
        self._threshold = config.threshold if config.threshold is not None else 0.75
        self._max_size = config.max_size if config.max_size is not None else 800
        self._model = None  # lazy load

    @property
    def name(self) -> str:
        return "semantic"

    def chunk(self, doc: Document) -> list[Chunk]:
        full_text = doc.full_text
        if not full_text.strip():
            return []

        # 1. Sentence spans: (char_start, char_end) in full_text
        spans = self._sentence_spans(full_text)
        if not spans:
            return []

        sentences = [full_text[s:e] for s, e in spans]

        # 2. Embed
        model = self._get_model()
        embeddings: np.ndarray = model.encode(sentences, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1.0)

        # 3. Cosine similarity between consecutive sentences
        if len(embeddings) > 1:
            sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1).tolist()
        else:
            sims = []

        # 4. Valley split points
        split_indices = {0, len(spans)}
        for i, sim in enumerate(sims):
            if sim < self._threshold:
                split_indices.add(i + 1)

        boundaries = sorted(split_indices)

        # 5. Build candidate segments from sentence groups
        segments: list[tuple[int, int]] = []
        for k in range(len(boundaries) - 1):
            seg_start = spans[boundaries[k]][0]
            seg_end = spans[boundaries[k + 1] - 1][1]
            segments.append((seg_start, seg_end))

        # 6. Enforce max_size by merging small and splitting large segments
        windows = self._enforce_size(segments, full_text)

        chunks: list[Chunk] = []
        for index, (char_start, char_end) in enumerate(windows):
            chunk_text = full_text[char_start:char_end]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:semantic:{index:05d}",
                    doc_id=doc.doc_id,
                    strategy="semantic",
                    index=index,
                    text=chunk_text,
                    page=page_for_offset(doc, char_start),
                    char_start=char_start,
                    char_end=char_end,
                    token_count=count_tokens(chunk_text),
                )
            )

        log.debug(
            "semantic_chunk_complete",
            doc_id=doc.doc_id,
            chunks=len(chunks),
            sentences=len(spans),
        )
        return chunks

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_model(self):  # type: ignore[return]
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("semantic_chunker_load_model", model=_MINILM_MODEL)
            self._model = SentenceTransformer(_MINILM_MODEL)
        return self._model

    def _sentence_spans(self, text: str) -> list[tuple[int, int]]:
        """Return (char_start, char_end) for each detected sentence."""
        boundaries = [0]
        for m in _SENT_BOUNDARY.finditer(text):
            boundaries.append(m.end())
        boundaries.append(len(text))

        spans: list[tuple[int, int]] = []
        for k in range(len(boundaries) - 1):
            start = boundaries[k]
            end = boundaries[k + 1]
            piece = text[start:end]
            # Strip leading whitespace to get the actual sentence start
            stripped = piece.lstrip()
            if stripped:
                actual_start = start + (len(piece) - len(stripped))
                spans.append((actual_start, end))
        return spans

    def _enforce_size(
        self, segments: list[tuple[int, int]], full_text: str
    ) -> list[tuple[int, int]]:
        """Merge tiny segments and split oversized ones to honour max_size."""
        result: list[tuple[int, int]] = []

        for seg_start, seg_end in segments:
            seg_tokens = count_tokens(full_text[seg_start:seg_end])

            if seg_tokens <= self._max_size:
                # Try to merge with previous if both are small
                if (
                    result
                    and count_tokens(full_text[result[-1][0] : seg_end])
                    <= self._max_size
                ):
                    result[-1] = (result[-1][0], seg_end)
                else:
                    result.append((seg_start, seg_end))
            else:
                # Hard-split at token boundaries
                tokens = _enc.encode(full_text[seg_start:seg_end])
                pos = seg_start
                for j in range(0, len(tokens), self._max_size):
                    piece = _enc.decode(tokens[j : j + self._max_size])
                    result.append((pos, pos + len(piece)))
                    pos += len(piece)

        return result
