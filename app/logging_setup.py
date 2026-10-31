"""Central logging configuration. Never log secrets.

* Console (stderr) and, optionally, a rotating file (default ``logs/backend.log``; LOG_FILE in .env,
  empty to disable).
* Every handler uses ``RedactingFormatter``, which scrubs registered secrets (e.g. the Gemini API key)
  and common key patterns from the *final* text — including tracebacks and third-party messages.
* Application code must still never log prompts, retrieved passages or other sensitive content;
  the formatter only guards against secrets.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_SECRETS: set[str] = set()
_LOCK = threading.Lock()

_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),                                       # classic Google API keys
    re.compile(r"((?:api[_-]?key|key|x-goog-api-key)\s*[=:]\s*)[^\s&\"',;]+", re.I),  # key=..., key: ...
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{16,}"),                               # bearer tokens
]
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def register_secret(value: str | None) -> None:
    """Add a value that must never appear in any log output."""
    if value and len(value.strip()) >= 8:
        with _LOCK:
            _SECRETS.add(value.strip())


def redact(text: str) -> str:
    for secret in sorted(_SECRETS, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = _PATTERNS[0].sub("[REDACTED]", text)
    text = _PATTERNS[1].sub(r"\1[REDACTED]", text)
    return _PATTERNS[2].sub(r"\1[REDACTED]", text)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))

    def formatException(self, ei) -> str:  # noqa: N802 (logging API name)
        return redact(super().formatException(ei))


def setup_logging(level: str = "INFO", log_file: str | Path | None = "default") -> None:
    """Configure root logging once. ``log_file="default"`` reads LOG_FILE from settings."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    formatter = RedactingFormatter(LOG_FORMAT)
    root = logging.getLogger()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file == "default":
        from app.config import get_settings
        log_file = get_settings().log_file_path
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    root.setLevel(level)
    # Third-party libraries are chatty at INFO.
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "chromadb", "sentence_transformers", "urllib3",
                  "google_genai", "google.genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
