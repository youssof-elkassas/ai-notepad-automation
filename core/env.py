"""Load environment variables from .env files (Windows-friendly)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    return _PROJECT_ROOT


def load_project_dotenv() -> Path | None:
    """
    Load GEMINI_API_KEY from a .env file in the project.

    Tries several locations because on Windows users often save as `.env.txt`
    or run commands from a different working directory.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv not installed; .env files will not be loaded")
        return None

    candidates = [
        _PROJECT_ROOT / ".env",
        Path.cwd() / ".env",
        _PROJECT_ROOT / ".env.txt",
        Path.cwd() / ".env.txt",
    ]

    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False, encoding="utf-8")
            logger.debug("Loaded environment from %s", path)
            return path

    from dotenv import find_dotenv

    found_path = find_dotenv(usecwd=True)
    if found_path:
        load_dotenv(found_path, override=False, encoding="utf-8")
        logger.debug("Loaded environment via find_dotenv: %s", found_path)
        return Path(found_path)

    return None


def get_gemini_api_key(
    *,
    config_key: str = "",
    env_var: str = "GEMINI_API_KEY",
) -> str:
    """Resolve Gemini API key from config, .env, or environment."""
    load_project_dotenv()

    if config_key.strip():
        return _strip_quotes(config_key.strip())

    for name in (env_var, "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return _strip_quotes(value)

    return ""


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value
