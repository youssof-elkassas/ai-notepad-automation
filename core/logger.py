"""Structured logging with failure screenshot support."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def setup_logger(
    name: str = "automation",
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_failure_screenshot(
    image: Image.Image,
    failures_dir: str | Path,
    reason: str,
    logger: logging.Logger | None = None,
) -> Path:
    """Persist a failure screenshot with timestamp and reason slug."""
    directory = Path(failures_dir)
    directory.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in reason.lower())[:80]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = directory / f"{timestamp}_{slug}.png"
    image.save(path)
    if logger:
        logger.error("Failure screenshot saved: %s (reason: %s)", path, reason)
    return path
