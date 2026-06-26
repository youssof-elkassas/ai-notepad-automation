"""Tests for coordinate transforms between viewports."""

import pytest
from PIL import Image

from vision.gui_parser import Bbox, GuiParser, Viewport


@pytest.fixture
def parser():
    return GuiParser(1920, 1080)


def test_global_to_local(parser):
    viewport = Viewport(0.25, 0.25, 0.75, 0.75)
    global_bbox = Bbox(0.375, 0.375, 0.625, 0.625)
    local = parser.global_to_local(global_bbox, viewport)
    assert local.x1 == pytest.approx(0.25)
    assert local.x2 == pytest.approx(0.75)


def test_local_global_roundtrip(parser):
    viewport = Viewport(0.25, 0.25, 0.75, 0.75)
    local = Bbox(0.0, 0.0, 0.5, 0.5)
    global_bbox = parser.local_to_global(local, viewport)
    assert global_bbox.x1 == pytest.approx(0.25)
    assert global_bbox.y1 == pytest.approx(0.25)
    assert global_bbox.x2 == pytest.approx(0.5)
    assert global_bbox.y2 == pytest.approx(0.5)


def test_normalized_to_screen(parser):
    x, y = parser.normalized_to_screen((0.5, 0.5), monitor_offset=(0, 0), dpi_scale=1.0)
    assert x == 960
    assert y == 540


def test_normalized_to_screen_with_offset(parser):
    x, y = parser.normalized_to_screen((0.0, 0.0), monitor_offset=(100, 50), dpi_scale=1.0)
    assert x == 100
    assert y == 50


def test_crop_viewport(parser):
    image = Image.new("RGB", (1920, 1080), color=(255, 0, 0))
    vp = Viewport(0.0, 0.0, 0.5, 0.5)
    cropped = parser.crop_viewport(image, vp)
    assert cropped.size == (960, 540)


def test_viewport_around_point(parser):
    vp = parser.viewport_around_point((0.5, 0.5), crop_size_px=1024, img_w=1920, img_h=1080)
    assert vp.x1 < 0.5 < vp.x2
    assert vp.y1 < 0.5 < vp.y2
