"""Text cleaning pipeline for ingested pages.

Applies whitespace normalization, dehyphenation, and header/footer stripping.
All functions are pure (no I/O) and operate on Page objects or plain strings.
"""

from __future__ import annotations

import re
from collections import Counter

from docintel.ingestion.loader import Page

# Header/footer detection: a line appearing in >= this fraction of pages is a
# candidate for removal.
_HEADER_FOOTER_THRESHOLD = 0.60
_HEADER_FOOTER_LOOK_LINES = 3   # inspect this many lines at top and bottom of each page
_MIN_LINE_LENGTH = 3            # ignore very short lines (page numbers alone, etc.)


def clean_text(text: str) -> str:
    """Normalize whitespace and fix extraction artifacts in a single text string.

    Operations (in order):
    1. Normalize line endings to \\n.
    2. Dehyphenation: ``word-\\nword`` → ``wordword`` (soft-hyphen at line break).
    3. Collapse runs of spaces to a single space.
    4. Strip trailing whitespace from each line.
    5. Collapse 3+ consecutive blank lines to 2.
    6. Strip leading/trailing whitespace from the whole string.
    """
    # Normalize Windows / old Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Dehyphenate: letter-<newline>letter → letteletter (removes soft hyphen)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse multiple spaces (not newlines) within a line
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _detect_repeating_lines(pages: list[Page]) -> frozenset[str]:
    """Return lines that appear in >= threshold fraction of pages.

    Inspects the first and last N lines of each page as header/footer candidates.
    """
    n_pages = len(pages)
    if n_pages < 2:
        return frozenset()

    counts: Counter[str] = Counter()

    for page in pages:
        lines = page.text.split("\n")
        candidate_set: set[str] = set()

        top_lines = lines[:_HEADER_FOOTER_LOOK_LINES]
        bottom_lines = lines[-_HEADER_FOOTER_LOOK_LINES:]

        for line in top_lines + bottom_lines:
            stripped = line.strip()
            if len(stripped) >= _MIN_LINE_LENGTH:
                candidate_set.add(stripped)

        for candidate in candidate_set:
            counts[candidate] += 1

    threshold = max(2, int(n_pages * _HEADER_FOOTER_THRESHOLD))
    return frozenset(line for line, count in counts.items() if count >= threshold)


def strip_headers_footers(pages: list[Page]) -> list[Page]:
    """Remove lines identified as repeating headers or footers from all pages.

    Detection: lines appearing in >= 60% of pages (top/bottom N lines only)
    are treated as headers/footers and removed everywhere.
    """
    repeating = _detect_repeating_lines(pages)
    if not repeating:
        return pages

    result: list[Page] = []
    for page in pages:
        filtered_lines = [
            line for line in page.text.split("\n")
            if line.strip() not in repeating
        ]
        new_text = "\n".join(filtered_lines)
        result.append(
            Page(
                page_number=page.page_number,
                text=new_text,
                char_start=page.char_start,
                char_end=page.char_end,
            )
        )
    return result


def clean_pages(pages: list[Page]) -> list[Page]:
    """Full cleaning pipeline: header/footer strip then text normalization."""
    pages = strip_headers_footers(pages)
    cleaned: list[Page] = []
    for page in pages:
        cleaned.append(
            Page(
                page_number=page.page_number,
                text=clean_text(page.text),
                char_start=page.char_start,
                char_end=page.char_end,
            )
        )
    return cleaned
