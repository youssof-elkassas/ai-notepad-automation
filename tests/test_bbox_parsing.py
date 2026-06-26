"""Tests for OS-Atlas bbox parsing."""

import pytest

from core.exceptions import BboxParseError
from vision.grounding import parse_os_atlas_bbox
from vision.gui_parser import Bbox


def test_parse_os_atlas_bbox_tokens():
    raw = "<|box_start|>(100,200),(150,280)<|box_end|>"
    bbox = parse_os_atlas_bbox(raw, 1920, 1080)
    assert bbox.x1 == pytest.approx(0.1)
    assert bbox.y1 == pytest.approx(0.2)
    assert bbox.x2 == pytest.approx(0.15)
    assert bbox.y2 == pytest.approx(0.28)


def test_parse_os_atlas_bbox_alt_format():
    raw = "some text (576,12),(592,42) trailing"
    bbox = parse_os_atlas_bbox(raw, 1000, 1000)
    assert bbox.x1 == pytest.approx(0.576)
    assert bbox.y2 == pytest.approx(0.042)


def test_parse_invalid_raises():
    with pytest.raises(BboxParseError):
        parse_os_atlas_bbox("no bbox here", 1920, 1080)


def test_bbox_center():
    bbox = Bbox(0.1, 0.2, 0.3, 0.4)
    assert bbox.center == pytest.approx((0.2, 0.3))


def test_bbox_clamp():
    bbox = Bbox(-0.1, 0.0, 1.1, 0.5).clamp()
    assert bbox.x1 == 0.0
    assert bbox.x2 == 1.0
