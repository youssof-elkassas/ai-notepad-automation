"""Tests for .env loading helpers."""

import os
from pathlib import Path

from core.env import _strip_quotes, get_gemini_api_key, load_project_dotenv


def test_strip_quotes():
    assert _strip_quotes('"abc"') == "abc"
    assert _strip_quotes("'abc'") == "abc"
    assert _strip_quotes("abc") == "abc"


def test_get_gemini_api_key_from_config():
    key = get_gemini_api_key(config_key="test-key-123")
    assert key == "test-key-123"


def test_load_project_dotenv_missing(monkeypatch, tmp_path):
    import core.env as env_module

    monkeypatch.setattr(env_module, "_PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    assert load_project_dotenv() is None


def test_get_gemini_api_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key-456")
    key = get_gemini_api_key(config_key="")
    assert key == "env-key-456"
