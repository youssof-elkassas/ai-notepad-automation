"""Patch scoring, dilation, and NMS for ScreenSeekeR (paper Eq. 1-2)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vision.gui_parser import Bbox, Viewport


@dataclass
class ScoredPatch:
    """A search patch with accumulated score."""

    viewport: Viewport
    score: float
    source_bbox: Bbox | None = None


def gaussian_score(
    vote_center: tuple[float, float],
    candidate: Viewport,
    sigma: float = 0.3,
) -> float:
    """Paper Eq. 1-2: centrality-weighted score for a candidate patch."""
    x, y = vote_center
    if not (candidate.x1 <= x <= candidate.x2 and candidate.y1 <= y <= candidate.y2):
        return 0.0
    x_prime = (x - candidate.x1) / (candidate.x2 - candidate.x1)
    y_prime = (y - candidate.y1) / (candidate.y2 - candidate.y1)
    return math.exp(-((x_prime - 0.5) ** 2 + (y_prime - 0.5) ** 2) / (2 * sigma**2))


def bbox_to_viewport(bbox: Bbox) -> Viewport:
    return Viewport(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2)


def dilate_viewport(
    viewport: Viewport,
    img_w: int,
    img_h: int,
    min_size_px: int,
    max_ratio: float = 0.5,
) -> Viewport:
    """Expand viewport so both dimensions are at least min_size_px."""
    cx = (viewport.x1 + viewport.x2) / 2
    cy = (viewport.y1 + viewport.y2) / 2
    half_w = max(viewport.width / 2, min_size_px / (2 * img_w))
    half_h = max(viewport.height / 2, min_size_px / (2 * img_h))
    half_w = min(half_w, max_ratio)
    half_h = min(half_h, max_ratio)
    return Viewport(
        x1=max(0.0, cx - half_w),
        y1=max(0.0, cy - half_h),
        x2=min(1.0, cx + half_w),
        y2=min(1.0, cy + half_h),
    )


def iou_viewports(a: Viewport, b: Viewport) -> float:
    """Intersection over union for two normalized viewports."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_patches(
    patches: list[ScoredPatch],
    iou_threshold: float = 0.5,
) -> list[ScoredPatch]:
    """Non-maximum suppression on scored patches."""
    sorted_patches = sorted(patches, key=lambda p: p.score, reverse=True)
    kept: list[ScoredPatch] = []
    for patch in sorted_patches:
        if all(
            iou_viewports(patch.viewport, k.viewport) < iou_threshold for k in kept
        ):
            kept.append(patch)
    return kept


def score_patches(
    vote_bboxes: list[Bbox],
    candidate_viewports: list[Viewport],
    sigma: float = 0.3,
) -> list[ScoredPatch]:
    """Score candidate patches by summing Gaussian votes from grounding boxes."""
    scores: dict[int, float] = {i: 0.0 for i in range(len(candidate_viewports))}
    for bbox in vote_bboxes:
        center = bbox.center
        for i, vp in enumerate(candidate_viewports):
            scores[i] += gaussian_score(center, vp, sigma=sigma)

    return [
        ScoredPatch(viewport=candidate_viewports[i], score=scores[i])
        for i in range(len(candidate_viewports))
    ]
