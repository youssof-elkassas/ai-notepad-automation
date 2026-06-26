"""Tests for Gemini JSON grounding response parsing."""

import pytest

from core.exceptions import BboxParseError
from vision.gemini_grounding import (
    bbox_from_gemini_payload,
    parse_gemini_grounding_json,
    parse_grounding_response,
)


def test_parse_gemini_json_clean():
    raw = '{"found": true, "confidence": 0.9, "bbox_1000": [100, 200, 150, 280]}'
    data = parse_gemini_grounding_json(raw)
    assert data["found"] is True
    bbox = bbox_from_gemini_payload(data)
    assert bbox.x1 == pytest.approx(0.1)
    assert bbox.y2 == pytest.approx(0.28)


def test_parse_gemini_json_with_fence():
    raw = '```json\n{"found": true, "confidence": 0.8, "bbox_1000": [500, 500, 550, 550]}\n```'
    data = parse_gemini_grounding_json(raw)
    bbox = bbox_from_gemini_payload(data)
    assert bbox.center == pytest.approx((0.525, 0.525))


def test_parse_gemini_normalized_0_1():
    data = {"found": True, "confidence": 0.7, "bbox": [0.1, 0.2, 0.15, 0.25]}
    bbox = bbox_from_gemini_payload(data)
    assert bbox.x1 == pytest.approx(0.1)


def test_parse_invalid_json_raises():
    with pytest.raises(BboxParseError):
        parse_gemini_grounding_json("not json")


def test_repair_truncated_json_bbox():
    raw = '{"found": true, "confidence": 0.8, "bbox_1000": [120, 340, 180, 420'
    data = parse_gemini_grounding_json(raw)
    bbox = bbox_from_gemini_payload(data)
    assert data["found"] is True
    assert bbox.x1 == pytest.approx(0.12)


def test_parse_grounding_response_ignores_incomplete_parsed():
    class FakeResponse:
        parsed = {"found": True}
        text = '{"found": true, "confidence": 0.9, "bbox_1000": [100, 200, 150, 280]}'

    data = parse_grounding_response(FakeResponse())
    assert data["bbox_1000"] == [100, 200, 150, 280]


def test_refine_falls_back_on_parse_error():
    from unittest.mock import MagicMock, patch

    from core.config import load_config
    from vision.gemini_grounding import GeminiGrounder
    from vision.gui_parser import Bbox

    config = load_config("low")
    grounder = GeminiGrounder(config)
    rough = Bbox(0.1, 0.2, 0.15, 0.3)
    image = __import__("PIL").Image.new("RGB", (1920, 1080))

    with patch.object(
        grounder,
        "_generate_grounding",
        side_effect=BboxParseError("truncated"),
    ):
        bbox, _raw, _conf, found = grounder.refine(image, "Notepad icon", rough)

    assert found is False
    assert bbox == rough


def test_mock_grounding_service():
    from PIL import Image

    from core.config import load_config
    from vision.gemini_grounding import MockGroundingService
    from vision.gui_parser import Bbox

    config = load_config("high")
    service = MockGroundingService(config, bbox=Bbox(0.1, 0.1, 0.2, 0.2))
    image = Image.new("RGB", (1920, 1080))
    result = service.locate("Notepad desktop icon", image)
    assert result.confidence >= 0.5
    assert result.bbox.x1 == pytest.approx(0.1)
