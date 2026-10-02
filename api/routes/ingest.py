"""POST /ingest — upload and ingest files into the corpus."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.params import File, Form

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from docintel.exceptions import IngestionError, OCRRequired
from docintel.ingestion.loader import DocType, PdfPlumberLoader, load_document
from docintel.chunking.registry import get_chunker
from docintel.utils.config import ChunkingConfig, load_config

from api.schemas import FailedDoc, IngestedDoc, IngestResponse, IngestSummary

log = structlog.get_logger()

router = APIRouter()

_STRATEGIES: list[str] = ["fixed", "recursive", "semantic"]


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    files: list[UploadFile] = File(...),
    doc_type: str = Form(default="other"),
) -> IngestResponse:
    """Upload one or more files and ingest them into the corpus."""
    request_id = getattr(request.state, "request_id", "")

    if not files:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "NO_FILES",
                "message": "At least one file must be provided.",
                "detail": {},
                "request_id": request_id,
            },
        )

    valid_doc_types: tuple[str, ...] = (
        "contract", "policy", "handbook", "rfp", "dpa", "nda", "other"
    )
    if doc_type not in valid_doc_types:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_DOC_TYPE",
                "message": f"doc_type must be one of {valid_doc_types}.",
                "detail": {"received": doc_type},
                "request_id": request_id,
            },
        )

    try:
        cfg = load_config()
    except Exception:
        cfg = None

    max_bytes = 50 * 1024 * 1024  # 50 MB default
    if cfg is not None:
        try:
            max_bytes = int(getattr(cfg, "ingestion", None) and cfg.ingestion.max_file_mb * 1024 * 1024 or max_bytes)
        except Exception:
            pass

    processed: list[IngestedDoc] = []
    failed: list[FailedDoc] = []
    total_chars = 0

    for upload in files:
        filename = upload.filename or "unknown"
        try:
            raw = await upload.read()
            if len(raw) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "code": "FILE_TOO_LARGE",
                        "message": f"{filename} exceeds the maximum allowed size.",
                        "detail": {"filename": filename, "size_bytes": len(raw)},
                        "request_id": request_id,
                    },
                )

            with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
                tmp.write(raw)
                tmp_path = Path(tmp.name)

            try:
                doc = load_document(tmp_path, doc_type=doc_type)
            finally:
                tmp_path.unlink(missing_ok=True)

            chunk_counts: dict[str, int] = {}
            for strategy in _STRATEGIES:
                chunk_cfg = ChunkingConfig(name=strategy)
                chunker = get_chunker(chunk_cfg)
                chunks = chunker.chunk(doc)
                chunk_counts[strategy] = len(chunks)

            total_chars += doc.total_chars
            processed.append(
                IngestedDoc(
                    doc_id=doc.doc_id,
                    filename=filename,
                    doc_type=doc.doc_type,
                    pages=len(doc.pages),
                    chars=doc.total_chars,
                    chunks=chunk_counts,
                )
            )
            log.info("ingest_success", filename=filename, doc_id=doc.doc_id, chars=doc.total_chars)

        except HTTPException:
            raise
        except (IngestionError, OCRRequired) as exc:
            log.warning("ingest_failed", filename=filename, error=str(exc))
            failed.append(FailedDoc(filename=filename, error=str(exc)))
        except Exception as exc:
            log.error("ingest_unexpected_error", filename=filename, error=str(exc))
            failed.append(FailedDoc(filename=filename, error=f"Unexpected error: {exc}"))

    if processed == [] and failed:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INGEST_FAILED",
                "message": "All files failed to ingest.",
                "detail": {"failed": [f.model_dump() for f in failed]},
                "request_id": request_id,
            },
        )

    return IngestResponse(
        processed=processed,
        failed=failed,
        summary=IngestSummary(
            n_processed=len(processed),
            n_failed=len(failed),
            total_chars=total_chars,
        ),
    )
