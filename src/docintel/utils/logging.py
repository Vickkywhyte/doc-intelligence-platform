"""Structured logging setup using structlog with JSON output.

Call configure_logging() once at process startup (main scripts, FastAPI lifespan).
Everywhere else, just:

    import structlog
    log = structlog.get_logger()
    log.info("event_name", key=value, ...)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path


def configure_logging(
    log_dir: str | Path = "logs",
    level: str | None = None,
) -> None:
    """Configure structlog with JSON output to stdout and a rotating file.

    Safe to call multiple times; re-configures on each call.
    """
    import structlog

    log_level_name = level or os.environ.get("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.jsonl"

    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove any pre-existing handlers to avoid duplicate output on re-call.
    root.handlers.clear()

    fmt = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    # 10 MB × 5 rotated files per OBSERVABILITY.md §2.4
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
