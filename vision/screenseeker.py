"""ScreenSeekeR: agentic cascaded visual search (paper Algorithm 1)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from PIL import Image

from core.config import AppConfig
from core.exceptions import GroundingError, LowConfidenceError
from vision.grounding import (
    GroundingResult,
    OSAtlasGrounder,
    MockGrounder,
    compute_confidence,
)
from vision.gui_parser import Bbox, GuiParser, Viewport
from vision.patch_scoring import (
    bbox_to_viewport,
    dilate_viewport,
    nms_patches,
    score_patches,
)
from vision.planner import MockPlanner, PlannerBackend, QwenPlanner

logger = logging.getLogger(__name__)


@dataclass
class SearchState:
    """Tracks recursive search context."""

    depth: int = 0
    viewport: Viewport = field(
        default_factory=lambda: Viewport(0.0, 0.0, 1.0, 1.0)
    )
    trace: list[str] = field(default_factory=list)


class ScreenSeekeR:
    """Paper-faithful ScreenSeekeR visual search framework."""

    def __init__(
        self,
        config: AppConfig,
        grounder: OSAtlasGrounder | MockGrounder,
        planner: PlannerBackend | None = None,
    ) -> None:
        self.config = config
        self.grounder = grounder
        self.planner = planner or QwenPlanner(config)
        self._parser = GuiParser(config.screen.width, config.screen.height)

    def search(self, instruction: str, image: Image.Image) -> GroundingResult:
        """Run full ScreenSeekeR search and return grounding result."""
        image = self._parser.normalize_screenshot(image)
        trace: list[str] = []

        if self.config.screenseeker.use_regound_fallback:
            result = self._regound(instruction, image, trace)
            if result is not None:
                return result

        state = SearchState()
        bbox = self._visual_search(instruction, image, state)
        if bbox is None:
            raise GroundingError(f"ScreenSeekeR failed to locate: {instruction}")

        local_bbox = self._parser.global_to_local(bbox, state.viewport)
        verification = self.planner.verify_result(
            self._parser.crop_viewport(image, state.viewport),
            instruction,
            bbox_to_viewport(local_bbox),
        )
        trace.append(f"planner_verdict={verification.verdict}")

        confidence = compute_confidence(
            bbox,
            image.width,
            image.height,
            min_area=self.config.grounding.min_bbox_area_px,
            max_area=self.config.grounding.max_bbox_area_px,
            planner_verdict=verification.verdict,
        )

        if (
            self.config.grounding.require_planner_verdict
            and verification.verdict != "is_target"
        ):
            raise LowConfidenceError(
                f"Planner rejected grounding for: {instruction}",
                reason=f"planner_verdict={verification.verdict}",
                confidence=confidence,
            )

        if confidence < 0.5:
            raise LowConfidenceError(
                f"Low confidence grounding for: {instruction}",
                reason="confidence_below_threshold",
                confidence=confidence,
            )

        annotated = self._parser.annotate(image, bbox, label=instruction)
        return GroundingResult(
            bbox=bbox,
            center=bbox.center,
            confidence=confidence,
            raw_output=verification.raw_text,
            search_trace=trace,
            annotated_image=annotated,
            planner_verdict=verification.verdict,
        )

    def _visual_search(
        self,
        instruction: str,
        image: Image.Image,
        state: SearchState,
    ) -> Bbox | None:
        """Recursive visual search (Algorithm 1)."""
        cfg = self.config.screenseeker
        img_w, img_h = image.width, image.height
        crop = self._parser.crop_viewport(image, state.viewport)

        if state.depth >= cfg.max_depth or not self._parser.patch_too_large(
            state.viewport, img_w, img_h, cfg.min_patch_size_px
        ):
            state.trace.append(f"direct_ground depth={state.depth}")
            return self._direct_grounding(instruction, crop, state.viewport, image)

        planner_result = self.planner.infer_regions(image, instruction)
        state.trace.append(
            f"position_inference depth={state.depth} regions={len(planner_result.regions)}"
        )

        vote_bboxes: list[Bbox] = []
        for region in planner_result.regions[: cfg.max_candidates]:
            region_crop = self._parser.crop_viewport(image, region)
            try:
                bbox, _ = self.grounder.ground_bbox(
                    region_crop,
                    instruction,
                    viewport=region,
                    full_image=image,
                )
                vote_bboxes.append(bbox)
            except Exception as exc:
                logger.debug("Grounding failed for region %s: %s", region, exc)

        if not vote_bboxes:
            return self._direct_grounding(instruction, crop, state.viewport, image)

        candidate_vps = [
            dilate_viewport(
                bbox_to_viewport(b),
                img_w,
                img_h,
                cfg.min_patch_size_px,
            )
            for b in vote_bboxes
        ]
        scored = score_patches(vote_bboxes, candidate_vps, sigma=cfg.sigma)
        filtered = nms_patches(scored, iou_threshold=cfg.nms_iou_threshold)
        sorted_patches = sorted(filtered, key=lambda p: p.score, reverse=True)

        for patch in sorted_patches[: cfg.max_candidates]:
            child = SearchState(
                depth=state.depth + 1,
                viewport=patch.viewport,
                trace=list(state.trace),
            )
            result = self._visual_search(instruction, image, child)
            if result is not None:
                state.trace.extend(child.trace)
                return result

        return self._direct_grounding(instruction, crop, state.viewport, image)

    def _direct_grounding(
        self,
        instruction: str,
        crop: Image.Image,
        viewport: Viewport,
        full_image: Image.Image,
    ) -> Bbox | None:
        try:
            bbox, _ = self.grounder.ground_bbox(
                crop,
                instruction,
                viewport=viewport,
                full_image=full_image,
            )
            return bbox
        except Exception as exc:
            logger.warning("Direct grounding failed: %s", exc)
            return None

    def _regound(
        self,
        instruction: str,
        image: Image.Image,
        trace: list[str],
    ) -> GroundingResult | None:
        """ReGround fallback: crop around initial prediction and re-ground."""
        cfg = self.config.screenseeker
        trace.append("regound_fallback_start")
        try:
            initial, raw = self.grounder.ground_bbox(image, instruction)
        except Exception:
            return None

        vp = self._parser.viewport_around_point(
            initial.center,
            cfg.reground_crop_size,
            image.width,
            image.height,
        )
        crop = self._parser.crop_viewport(image, vp)
        refined, raw2 = self.grounder.ground_bbox(
            crop, instruction, viewport=vp, full_image=image
        )
        trace.append("regound_fallback_done")

        local_refined = self._parser.global_to_local(refined, vp)
        verification = self.planner.verify_result(
            self._parser.crop_viewport(image, vp),
            instruction,
            bbox_to_viewport(local_refined),
        )
        confidence = compute_confidence(
            refined,
            image.width,
            image.height,
            min_area=self.config.grounding.min_bbox_area_px,
            max_area=self.config.grounding.max_bbox_area_px,
            planner_verdict=verification.verdict,
        )
        if confidence < 0.5:
            return None

        return GroundingResult(
            bbox=refined,
            center=refined.center,
            confidence=confidence,
            raw_output=raw2 or raw,
            search_trace=trace,
            annotated_image=self._parser.annotate(image, refined, label=instruction),
            planner_verdict=verification.verdict,
        )
