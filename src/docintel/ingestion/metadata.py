"""Document metadata extraction and doc_type inference.

Provides two public functions:
- ``infer_doc_type``: classify a document by filename + text keywords.
- ``extract_file_metadata``: mtime, size, extension from the filesystem.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docintel.ingestion.loader import DocType

# Ordered: most-specific patterns first.  Each entry is
# (list-of-keyword-substrings, DocType).  The probe string is
# ``filename.lower() + " " + first_500_chars_of_text.lower()``.
_PATTERNS: list[tuple[list[str], DocType]] = [
    (
        ["master service agreement", "master services agreement", " msa "],
        "contract",
    ),
    (
        ["non-disclosure agreement", "non disclosure agreement", " nda ", "nondisclosure"],
        "nda",
    ),
    (
        ["data processing agreement", "data processing addendum", " dpa "],
        "dpa",
    ),
    (
        ["request for proposal", " rfp "],
        "rfp",
    ),
    (
        ["employee handbook", "contractor handbook", "staff handbook"],
        "handbook",
    ),
    (
        ["policy", "procedure", "guidelines"],
        "policy",
    ),
]


def infer_doc_type(filename: str, text: str) -> DocType:
    """Classify a document from its filename and leading text.

    Returns the first matching DocType, or 'other' if nothing matches.
    """
    probe = filename.lower() + " " + text[:500].lower()
    for keywords, doc_type in _PATTERNS:
        if any(kw in probe for kw in keywords):
            return doc_type
    return "other"


def extract_file_metadata(path: Path) -> dict[str, Any]:
    """Return filesystem metadata for the given file path.

    Keys: ``file_size_bytes``, ``mtime_utc``, ``extension``.
    """
    stat = path.stat()
    mtime_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return {
        "file_size_bytes": stat.st_size,
        "mtime_utc": mtime_utc,
        "extension": path.suffix.lower(),
    }
