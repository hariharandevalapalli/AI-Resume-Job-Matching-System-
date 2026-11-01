"""Central logging setup.

Log records never contain resume text: modules log sizes, counts, scores and
request ids only.
"""

from __future__ import annotations

import logging
import os

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Level defaults to the LOG_LEVEL env var or INFO."""
    global _configured
    if _configured:
        return
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(level=resolved, format=_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "multipart", "python_multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
