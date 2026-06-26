"""Tests for patch scoring, NMS, and dilation."""

import pytest

from vision.gui_parser import Bbox, Viewport
from vision.patch_scoring import (
    bbox_to_viewport,
    dilate_viewport,
    gaussian_score,
    iou_viewports,
    nms_patches,
    score_patches,
    ScoredPatch,
)


def test_gaussian_score_center_high():
    vp = Viewport(0.0, 0.0, 1.0, 1.0)
    center_score = gaussian_score((0.5, 0.5), vp, sigma=0.3)
    edge_score = gaussian_score((0.0, 0.0), vp, sigma=0.3)
    assert center_score > edge_score
    assert center_score == pytest.approx(1.0, abs=0.01)


def test_gaussian_score_outside_zero():
    vp = Viewport(0.2, 0.2, 0.4, 0.4)
    assert gaussian_score((0.1, 0.1), vp) == 0.0


def test_iou_identical():
    vp = Viewport(0.1, 0.1, 0.5, 0.5)
    assert iou_viewports(vp, vp) == pytest.approx(1.0)


def test_iou_disjoint():
    a = Viewport(0.0, 0.0, 0.2, 0.2)
    b = Viewport(0.5, 0.5, 0.8, 0.8)
    assert iou_viewports(a, b) == 0.0


def test_nms_removes_overlap():
    vp1 = Viewport(0.0, 0.0, 0.5, 0.5)
    vp2 = Viewport(0.1, 0.1, 0.6, 0.6)
    vp3 = Viewport(0.7, 0.7, 1.0, 1.0)
    patches = [
        ScoredPatch(viewport=vp1, score=0.9),
        ScoredPatch(viewport=vp2, score=0.5),
        ScoredPatch(viewport=vp3, score=0.8),
    ]
    kept = nms_patches(patches, iou_threshold=0.3)
    assert len(kept) == 2
    assert kept[0].score == 0.9
    assert kept[1].score == 0.8


def test_dilate_expands_small_viewport():
    small = Viewport(0.4, 0.4, 0.42, 0.42)
    dilated = dilate_viewport(small, 1920, 1080, min_size_px=1280)
    assert dilated.width * 1920 >= 1280 * 0.5  # at least partially expanded


def test_score_patches_ranks_by_votes():
    bboxes = [Bbox(0.1, 0.1, 0.15, 0.15), Bbox(0.12, 0.12, 0.17, 0.17)]
    candidates = [bbox_to_viewport(b) for b in bboxes]
    scored = score_patches(bboxes, candidates, sigma=0.3)
    assert all(s.score > 0 for s in scored)
