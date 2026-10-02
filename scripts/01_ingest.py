"""Ingest all documents in data/raw/ and write Document JSON to data/processed/.

Reads every .pdf, .docx, .txt, and .md file in the raw directory,
loads/cleans each one, and writes a {doc_id}.json file for each.
Skips and logs failures; prints a structured summary at the end.

Usage:
    python scripts/01_ingest.py [raw_dir] [processed_dir]

Defaults:
    raw_dir       = data/raw
    processed_dir = data/processed
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import structlog

from docintel.exceptions import IngestionError, OCRRequired
from docintel.ingestion import load_document
from docintel.utils.logging import configure_logging

configure_logging()
log = structlog.get_logger()

_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}


def ingest_all(raw_dir: Path, processed_dir: Path) -> None:
    """Load every supported file in raw_dir and write JSON to processed_dir."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    if not candidates:
        log.warning("ingest_no_files_found", raw_dir=str(raw_dir))
        return

    processed = 0
    failed = 0
    total_chars = 0
    failed_files: list[str] = []

    for path in candidates:
        t0 = time.perf_counter()
        try:
            doc = load_document(path)
        except OCRRequired as exc:
            log.error(
                "ingest_document_failed",
                filename=path.name,
                error=str(exc),
                reason="ocr_required",
            )
            failed += 1
            failed_files.append(path.name)
            continue
        except IngestionError as exc:
            log.error(
                "ingest_document_failed",
                filename=path.name,
                error=str(exc),
            )
            failed += 1
            failed_files.append(path.name)
            continue
        except Exception as exc:
            log.error(
                "ingest_document_failed",
                filename=path.name,
                error=str(exc),
                exc_info=True,
            )
            failed += 1
            failed_files.append(path.name)
            continue

        out_path = processed_dir / f"{doc.doc_id}.json"
        out_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")

        latency_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "ingest_document_complete",
            doc_id=doc.doc_id,
            filename=doc.filename,
            doc_type=doc.doc_type,
            pages=len(doc.pages),
            chars=doc.total_chars,
            latency_ms=latency_ms,
        )

        processed += 1
        total_chars += doc.total_chars

    log.info(
        "ingest_complete",
        processed=processed,
        failed=failed,
        total_chars=total_chars,
        failed_files=failed_files,
    )


def main(argv: list[str]) -> int:
    raw_dir = Path(argv[0]) if len(argv) > 0 else Path("data/raw")
    processed_dir = Path(argv[1]) if len(argv) > 1 else Path("data/processed")

    if not raw_dir.exists():
        log.error("ingest_raw_dir_not_found", raw_dir=str(raw_dir))
        return 1

    ingest_all(raw_dir, processed_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
