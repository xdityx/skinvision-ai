"""
phase0/src/utils/logging.py
Shared logging utilities for Phase 0 of the AI Skincare Assistant project.

Usage:
    from phase0.src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger = get_logger(__name__, log_file=Path("run.log"))
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Custom JSON formatter
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Formats a LogRecord as a single-line JSON string.

    Output schema:
        {"time": <ISO-8601 str>, "level": <str>, "logger": <str>,
         "message": <str>, "extra": {<any extra fields>}}
    """

    # Standard LogRecord attributes that are NOT treated as "extra"
    _STANDARD_ATTRS: frozenset[str] = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Build the structured payload
        payload: dict = {
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "extra": {},
        }

        # Attach any extra fields the caller passed via logging.debug(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in self._STANDARD_ATTRS:
                try:
                    json.dumps(value)          # only include JSON-serialisable values
                    payload["extra"][key] = value
                except (TypeError, ValueError):
                    payload["extra"][key] = str(value)

        # Attach exception traceback if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_logger(
    name: str,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    The function is **idempotent**: if the logger already has handlers attached
    (e.g. because it was retrieved earlier in the same process), it is returned
    unchanged so duplicate log lines are never emitted.

    Args:
        name:     Logger name, typically ``__name__`` of the calling module.
        log_file: Optional path for a DEBUG-level JSON log file.  The parent
                  directory is created automatically if it does not exist.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Idempotency guard — do nothing if handlers are already attached.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)   # capture everything; handlers filter

    # ------------------------------------------------------------------
    # Console handler — INFO level, human-readable
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # File handler — DEBUG level, JSON-structured  (optional)
    # ------------------------------------------------------------------
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger to avoid duplicate output
    logger.propagate = False

    return logger
