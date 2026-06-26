"""Tests for configuration loading."""

from core.config import load_config


def test_load_high_profile():
    config = load_config("high")
    assert config.profile == "high"
    assert config.gemini.model == "gemini-2.5-flash"


def test_load_low_profile():
    config = load_config("low")
    assert config.profile == "low"
    assert config.gemini.model == "gemini-2.5-flash"
    assert config.gemini.min_confidence == 0.4
