"""FastAPI application entry point — Document Intelligence Platform API.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.utils.logging import configure_logging

from api.routes import eval as eval_router
from api.routes import ingest as ingest_router
from api.routes import query as query_router

log = structlog.get_logger()

_START_TIME = time.monotonic()


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info("api_startup", version="0.1.0")
    yield
    log.info("api_shutdown")


app = FastAPI(
    title="Document Intelligence Platform",
    version="0.1.0",
    description="RAG-based document analysis with first-class evaluation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request ID middleware ──────────────────────────────────────────────────────


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    import uuid
    request_id = f"req_{uuid.uuid4().hex[:20]}"
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Global exception handler ──────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    log.error("unhandled_exception", error=str(exc), request_id=request_id)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred.",
                "detail": {"exception": type(exc).__name__},
                "request_id": request_id,
            }
        },
    )


# Re-wrap HTTPException to match error envelope
from fastapi.exceptions import HTTPException, RequestValidationError


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    # If detail is already an error envelope dict, use it
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": detail},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": str(detail),
                "detail": {},
                "request_id": request_id,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "detail": {"errors": exc.errors()},
                "request_id": request_id,
            }
        },
    )


# ── Health ─────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    """Liveness + version probe."""
    try:
        version = importlib.metadata.version("docintel")
    except importlib.metadata.PackageNotFoundError:
        version = "0.1.0"

    return {
        "status": "ok",
        "version": version,
        "git_sha": _git_sha(),
        "python_version": ".".join(str(v) for v in sys.version_info[:3]),
        "uptime_s": round(time.monotonic() - _START_TIME, 1),
    }


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(ingest_router.router)
app.include_router(query_router.router)
app.include_router(eval_router.router)
