"""Extract and validate [doc_id:page:chunk_id] citations from LLM output.

Format: [doc_id:page:chunk_id]
  doc_id   — one or more non-whitespace, non-bracket chars
  page     — integer
  chunk_id — everything up to the closing ] (may contain colons)

Malformed citations are logged and skipped; they are never silently dropped
without a warning (CLAUDE.md §4 rule: fail loud in dev).
"""

from __future__ import annotations

import re

import structlog

from docintel.generation.models import Citation

log = structlog.get_logger()

# Matches [doc_id:page:chunk_id] where chunk_id may itself contain colons.
# Group 1: doc_id (non-colon, non-bracket chars)
# Group 2: page   (digits)
# Group 3: chunk_id (non-bracket chars, including colons)
_CITATION_RE = re.compile(r"\[([^\]:]+):(\d+):([^\]]+)\]")


def parse_citations(
    text: str,
    valid_chunk_ids: set[str] | None = None,
) -> list[Citation]:
    """Extract validated Citation objects from LLM answer text.

    Args:
        text: raw answer string from the LLM.
        valid_chunk_ids: set of chunk_ids that were actually retrieved.
            If provided, any citation whose chunk_id is not in this set is
            logged as a mismatch and excluded from the returned list.

    Returns:
        Deduplicated list of Citation objects in order of first appearance.
    """
    citations: list[Citation] = []
    seen: set[str] = set()

    for match in _CITATION_RE.finditer(text):
        raw = match.group(0)
        doc_id = match.group(1).strip()
        page_str = match.group(2)
        chunk_id = match.group(3).strip()

        if raw in seen:
            continue
        seen.add(raw)

        try:
            page = int(page_str)
        except ValueError:
            log.warning("citation_bad_page", raw=raw, page_str=page_str)
            continue

        if valid_chunk_ids is not None and chunk_id not in valid_chunk_ids:
            log.warning(
                "citation_hallucinated",
                chunk_id=chunk_id,
                raw=raw,
                hint="chunk_id not in retrieved set",
            )
            continue

        citations.append(Citation(doc_id=doc_id, page=page, chunk_id=chunk_id))

    return citations
