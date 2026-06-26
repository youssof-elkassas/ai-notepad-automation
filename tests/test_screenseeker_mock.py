"""Tests for ScreenSeekeR with mocked models."""

import pytest
from PIL import Image

from core.config import load_config
from core.exceptions import LowConfidenceError
from vision.grounding import GroundingService, MockGrounder
from vision.gui_parser import Bbox
from vision.planner import MockPlanner, REGION_VIEWPORTS
from vision.screenseeker import ScreenSeekeR


@pytest.fixture
def config():
    return load_config("high")


@pytest.fixture
def screenshot():
    return Image.new("RGB", (1920, 1080), color=(30, 30, 30))


def test_screenseeker_mock_search(config, screenshot):
    bbox = Bbox(0.1, 0.1, 0.15, 0.18)
    grounder = MockGrounder(bbox=bbox)
    planner = MockPlanner(regions=[REGION_VIEWPORTS["full"]], verdict="is_target")
    seeker = ScreenSeekeR(config, grounder, planner)
    result = seeker.search("Notepad desktop icon", screenshot)

    assert result.confidence >= 0.5
    assert result.planner_verdict == "is_target"
    assert result.bbox.x1 == pytest.approx(0.1, abs=0.05)


def test_screenseeker_rejects_low_verdict(config, screenshot):
    config.grounding.require_planner_verdict = True
    bbox = Bbox(0.1, 0.1, 0.15, 0.18)
    grounder = MockGrounder(bbox=bbox)
    planner = MockPlanner(
        regions=[REGION_VIEWPORTS["full"]],
        verdict="target_not_found",
    )
    seeker = ScreenSeekeR(config, grounder, planner)

    with pytest.raises(LowConfidenceError):
        seeker.search("Notepad desktop icon", screenshot)


def test_grounding_service_locate(config, screenshot):
    bbox = Bbox(0.2, 0.2, 0.25, 0.28)
    grounder = MockGrounder(bbox=bbox)
    planner = MockPlanner(verdict="is_target")
    seeker = ScreenSeekeR(config, grounder, planner)
    service = GroundingService(grounder, config, screenseeker=seeker)
    result = service.locate("Notepad desktop icon", screenshot)

    assert result.annotated_image is not None
    assert len(result.search_trace) > 0
