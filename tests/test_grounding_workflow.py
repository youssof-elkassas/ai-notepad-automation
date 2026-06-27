"""Tests for shared grounding workflow."""

from core.config import load_config
from core.grounding_workflow import resolve_grounding_instruction


def test_resolve_grounding_instruction_uses_config():
    config = load_config("low")
    assert resolve_grounding_instruction(config) == "Notepad desktop icon"


def test_resolve_grounding_instruction_cli_override():
    config = load_config("low")
    assert resolve_grounding_instruction(config, "custom target") == "custom target"
