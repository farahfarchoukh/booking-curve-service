"""
Structured logging setup, shared by every entrypoint (train/predict/api).

Why this exists: the original submission used bare `print()` for both
progress messages and data-quality warnings — indistinguishable severity,
no timestamps, no way to redirect or filter in a real deployment. A
container's stdout is scraped by whatever log pipeline runs it (CloudWatch,
in Ampliphi's case per DESIGN.md §6.5); that pipeline needs level and
timestamp to be useful for anything past "did it crash."

Usage: `log = get_logger(__name__)` at module import time, then
`log.info(...)` / `log.warning(...)`. Level is controlled by the
`LOG_LEVEL` env var (default INFO) so it's a deploy-time knob, not a
code change.
"""

from __future__ import annotations

import logging
import os


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03dZ %(levelname)-8s %(name)s: %(message)s",
                "%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger
