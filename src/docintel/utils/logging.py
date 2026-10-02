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

    # ProcessorFormatter handles both structlog events and plain stdlib records,
    # producing one JSON object per line on both stdout and the rotating file.
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(json_formatter)
    root.addHandler(stream_handler)

    # 10 MB × 5 rotated files per OBSERVABILITY.md §2.4
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(json_formatter)
    root.addHandler(file_handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Hands the event dict to ProcessorFormatter on each stdlib handler.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Suppress chatty third-party libs from flooding the log file.
    for noisy in ("httpx", "httpcore", "pikepdf", "spacy", "kaleido"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None):
    """Return a structlog bound logger. Thin wrapper around structlog.get_logger()."""
    import structlog
    return structlog.get_logger(name)
