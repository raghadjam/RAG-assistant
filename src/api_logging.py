"""Structured request logging: one JSON object per line.

Example line (pretty-printed here, one line in the file):

    {"time": "2026-11-03T10:15:02+00:00", "request_id": "5f0c...", "method": "POST",
     "path": "/ask", "status_code": 200, "duration_ms": 1432.1, "rag_status": "answered", ...}

JSON lines can be read with `jq`, loaded into pandas, or shipped to a log service.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ACCESS_LOGGER = "rag.api.access"
PREVIEW_CHARS = 80


def configure_api_logging(log_path: str | None, console: bool = True) -> logging.Logger:
    """Send request records to the console and/or a JSON-lines file.

    Calling this again replaces the handlers, so creating several apps (as the tests do)
    never produces duplicate lines.
    """
    logger = logging.getLogger(ACCESS_LOGGER)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(message)s")
    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:      # a log file problem must never stop the service
            logging.getLogger(__name__).warning("cannot write request log %s: %s", log_path, exc)
    return logger


def log_request(logger: logging.Logger, record: dict) -> None:
    line = {"time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **record}
    logger.info(json.dumps(line, ensure_ascii=False, default=str))


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Short single-line start of a user's text. The full text is never logged."""
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
