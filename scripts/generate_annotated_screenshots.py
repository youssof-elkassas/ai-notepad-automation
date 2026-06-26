"""Generate annotated screenshots for three icon positions."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from automation.windows import WindowsCapture
from core.config import AppConfig
from scripts.setup_desktop import ICON_POSITIONS
from vision.gemini_grounding import create_grounding_service
from vision.gui_parser import Bbox, GuiParser

logger = logging.getLogger(__name__)

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
    parser = GuiParser(config.screen.width, config.screen.height)

    for position_name, grid in ICON_POSITIONS.items():
        logger.info("=== Annotating position: %s ===", position_name)
        logger.info(
            "Ensure Notepad icon is at grid (%d, %d), then capturing in 3s...",
            grid["grid_x"],
            grid["grid_y"],
        )
        time.sleep(3)

        screenshot = capture.capture_screenshot()
        mock_bbox = MOCK_BBOXES[position_name] if use_mock else None
        service = create_grounding_service(config, use_mock=use_mock, mock_bbox=mock_bbox)

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
