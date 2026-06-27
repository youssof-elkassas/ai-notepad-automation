"""Tests for google-genai client factory."""

import pytest

from core.config import load_config
from core.env import get_gemini_api_key
from vision.genai_client import GROUNDING_JSON_SCHEMA, build_grounding_config, resolve_api_key


def test_grounding_json_schema_has_required_fields():
    assert "found" in GROUNDING_JSON_SCHEMA["properties"]
    assert "bbox_1000" in GROUNDING_JSON_SCHEMA["properties"]


def test_build_grounding_config():
    config = load_config("high")
    gen_config = build_grounding_config(config)
    assert gen_config.response_mime_type == "application/json"
    assert gen_config.response_json_schema == GROUNDING_JSON_SCHEMA


def test_get_gemini_api_key_prefers_google(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert get_gemini_api_key() == "google-key"


def test_resolve_api_key_from_config():
    config = load_config("high")
    config.gemini.api_key = "config-key"
    assert resolve_api_key(config) == "config-key"
