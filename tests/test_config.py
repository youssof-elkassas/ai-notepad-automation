"""Tests for configuration loading."""

from core.config import load_config


def test_load_high_profile():
    config = load_config("high")
    assert config.profile == "high"
    assert "7B" in config.models.grounder_id


def test_load_low_profile():
    config = load_config("low")
    assert config.profile == "low"
    assert config.models.load_in_4bit is True
    assert config.screenseeker.use_regound_fallback is True
