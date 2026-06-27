"""Google GenAI SDK client factory (google-genai package)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from google import genai
from google.genai import types

from core.config import AppConfig
from core.env import get_gemini_api_key, load_project_dotenv
from core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Structured JSON schema for GUI grounding responses
GROUNDING_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "confidence": {"type": "number"},
        "bbox_1000": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        },
        "description": {"type": "string"},
    },
    "required": ["found", "confidence", "bbox_1000"],
}


def resolve_api_key(config: AppConfig) -> str:
    """Resolve API key from config file, .env, or environment."""
    load_project_dotenv()
    key = get_gemini_api_key(
        config_key=config.gemini.api_key,
        env_var=config.gemini.api_key_env,
    )
    if not key:
        raise ConfigurationError(_missing_key_message(config))
    return key


def _missing_key_message(config: AppConfig) -> str:
    from core.env import project_root

    root = project_root()
    return (
        "Google GenAI API key not found. The google-genai SDK reads "
        "GOOGLE_API_KEY or GEMINI_API_KEY.\n"
        f"  1) {root / '.env'} → GOOGLE_API_KEY=your_key\n"
        f"  2) config/secrets.yaml → gemini.api_key\n"
        "Get a key: https://aistudio.google.com/apikey"
    )


def _sync_env_vars(key: str) -> None:
    """Set both standard env names so the SDK and other tools agree."""
    if key:
        os.environ.setdefault("GOOGLE_API_KEY", key)
        os.environ.setdefault("GEMINI_API_KEY", key)


@lru_cache(maxsize=4)
def _cached_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def create_genai_client(config: AppConfig) -> genai.Client:
    """
    Create a google-genai Client using the official SDK.

    Uses explicit api_key when configured; otherwise relies on SDK env detection.
    """
    key = resolve_api_key(config)
    _sync_env_vars(key)
    logger.debug("GenAI client initialized (key length=%d)", len(key))
    return _cached_client(key)


def build_grounding_config(config: AppConfig) -> types.GenerateContentConfig:
    """Build GenerateContentConfig for structured grounding JSON."""
    return types.GenerateContentConfig(
        temperature=config.gemini.temperature,
        max_output_tokens=config.gemini.max_output_tokens,
        response_mime_type="application/json",
        response_json_schema=GROUNDING_JSON_SCHEMA,
    )
