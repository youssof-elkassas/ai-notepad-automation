"""Generate annotated screenshots for three icon positions."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from automation.windows import WindowsCapture
from core.config import AppConfig
from scripts.setup_desktop import ICON_POSITIONS
from vision.grounding import GroundingService, OSAtlasGrounder, MockGrounder
from vision.gui_parser import Bbox
from vision.planner import MockPlanner, QwenPlanner, REGION_VIEWPORTS
from vision.screenseeker import ScreenSeekeR

logger = logging.getLogger(__name__)

# Mock bboxes per position for non-GPU testing
MOCK_BBOXES = {
    "top_left": Bbox(0.02, 0.05, 0.06, 0.12),
    "center": Bbox(0.45, 0.40, 0.50, 0.48),
    "bottom_right": Bbox(0.88, 0.70, 0.93, 0.78),
}


def generate_annotated(config: AppConfig, *, use_mock: bool = False) -> None:
    """Capture, ground, and save annotated screenshots for each icon position."""
    capture = WindowsCapture(config)
    capture.ensure_directories()
    out_dir = Path(config.paths.annotated_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instruction = config.grounding.icon_instruction

    for position_name, grid in ICON_POSITIONS.items():
        logger.info("=== Annotating position: %s ===", position_name)
        logger.info(
            "Ensure Notepad icon is at grid (%d, %d), then capturing in 3s...",
            grid["grid_x"],
            grid["grid_y"],
        )
        time.sleep(3)

        screenshot = capture.capture_screenshot()

        if use_mock:
            bbox = MOCK_BBOXES[position_name]
            grounder = MockGrounder(bbox=bbox)
            regions = [REGION_VIEWPORTS.get("desktop_left", REGION_VIEWPORTS["full"])]
            if position_name == "center":
                regions = [REGION_VIEWPORTS["desktop_center"]]
            elif position_name == "bottom_right":
                regions = [REGION_VIEWPORTS["desktop_right"]]
            planner = MockPlanner(regions=regions)
        else:
            grounder = OSAtlasGrounder(config)
            planner = QwenPlanner(config)

        seeker = ScreenSeekeR(config, grounder, planner)
        service = GroundingService(grounder, config, screenseeker=seeker)

        try:
            result = service.locate(instruction, screenshot)
            annotated = result.annotated_image
        except Exception as exc:
            logger.error("Grounding failed for %s: %s", position_name, exc)
            from core.logger import save_failure_screenshot

            save_failure_screenshot(
                screenshot,
                config.paths.failures_dir,
                f"annotate_{position_name}",
                logger,
            )
            continue

        if annotated is None:
            from vision.gui_parser import GuiParser

            parser = GuiParser(config.screen.width, config.screen.height)
            annotated = parser.annotate(
                screenshot,
                result.bbox,
                label=f"{position_name}: {instruction}",
            )

        out_path = out_dir / f"notepad_{position_name}.png"
        annotated.save(out_path)
        logger.info(
            "Saved %s | bbox=%s confidence=%.2f",
            out_path,
            result.bbox.as_tuple(),
            result.confidence,
        )

    logger.info("Annotated screenshots written to %s", out_dir)
