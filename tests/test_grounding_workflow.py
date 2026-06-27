"""Tests for shared grounding workflow."""

from unittest.mock import MagicMock

from PIL import Image

from core.config import load_config
from core.grounding_workflow import (
    refresh_from_cache,
    resolve_grounding_instruction,
    resolve_grounding_with_cache,
)
from vision.grounding import GroundingResult
from vision.gui_parser import Bbox


def test_resolve_grounding_instruction_uses_config():
    config = load_config("low")
    assert resolve_grounding_instruction(config) == "Notepad desktop icon"


def test_resolve_grounding_instruction_cli_override():
    config = load_config("low")
    assert resolve_grounding_instruction(config, "custom target") == "custom target"


def test_resolve_grounding_with_cache_uses_verify_on_second_call():
    config = load_config("low")
    screenshot = Image.new("RGB", (1920, 1080))
    cached = GroundingResult(
        bbox=Bbox(0.1, 0.1, 0.15, 0.2),
        center=(0.125, 0.15),
        confidence=0.9,
        raw_output="{}",
        click_point=(0.125, 0.135),
        image_size=(1920, 1080),
    )
    service = MagicMock()
    service.verify_cached.return_value = True

    result, stored = resolve_grounding_with_cache(
        service,
        "Notepad desktop icon",
        screenshot,
        config,
        cached,
    )

    service.verify_cached.assert_called_once()
    service.locate.assert_not_called()
    assert stored is cached
    assert "cache:hit" in result.search_trace[-1]


def test_resolve_grounding_with_cache_regounds_when_verify_fails():
    config = load_config("low")
    screenshot = Image.new("RGB", (1920, 1080))
    cached = GroundingResult(
        bbox=Bbox(0.1, 0.1, 0.15, 0.2),
        center=(0.125, 0.15),
        confidence=0.9,
        raw_output="{}",
        click_point=(0.125, 0.135),
        image_size=(1920, 1080),
    )
    fresh = GroundingResult(
        bbox=Bbox(0.11, 0.11, 0.16, 0.21),
        center=(0.135, 0.16),
        confidence=0.95,
        raw_output="{}",
        click_point=(0.135, 0.145),
        image_size=(1920, 1080),
    )
    service = MagicMock()
    service.verify_cached.return_value = False
    service.locate.return_value = fresh

    result, stored = resolve_grounding_with_cache(
        service,
        "Notepad desktop icon",
        screenshot,
        config,
        cached,
    )

    service.locate.assert_called_once()
    assert result is fresh
    assert stored is fresh


def test_refresh_from_cache_updates_trace():
    config = load_config("low")
    screenshot = Image.new("RGB", (1920, 1080))
    cached = GroundingResult(
        bbox=Bbox(0.1, 0.1, 0.15, 0.2),
        center=(0.125, 0.15),
        confidence=0.9,
        raw_output="{}",
        click_point=(0.125, 0.135),
        image_size=(1920, 1080),
        search_trace=["genai:flash"],
    )
    result = refresh_from_cache(cached, screenshot, config, "Notepad desktop icon")
    assert result.search_trace[-1] == "cache:hit"
    assert result.annotated_image is not None
