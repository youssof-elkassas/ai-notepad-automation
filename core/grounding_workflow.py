"""Shared grounding workflow used by demo and run commands."""

from __future__ import annotations

import logging
from pathlib import Path

from automation.windows import WindowsCapture
from core.config import AppConfig
from vision.gemini_grounding import GeminiGroundingService, MockGroundingService
from vision.grounding import GroundingResult
from vision.gui_parser import GuiParser

logger = logging.getLogger(__name__)


def resolve_grounding_instruction(
    config: AppConfig,
    override: str | None = None,
) -> str:
    """Return the grounding instruction (CLI override or config default)."""
    if override and override.strip():
        return override.strip()
    return config.grounding.icon_instruction


def locate_on_screenshot(
    capture: WindowsCapture,
    service: GeminiGroundingService | MockGroundingService,
    instruction: str,
) -> GroundingResult:
    """Capture a screenshot and ground the target — same path as demo."""
    screenshot = capture.capture_screenshot()
    logger.info("Grounding instruction: %s", instruction)
    return service.locate(instruction, screenshot)


def to_screen_coords(
    result: GroundingResult,
    capture: WindowsCapture,
    config: AppConfig,
) -> tuple[int, int]:
    """Map normalized click point to absolute screen pixels."""
    parser = GuiParser(config.screen.width, config.screen.height)
    point = result.click_point or result.center
    width, height = result.image_size
    screen_x, screen_y = parser.normalized_to_screen(
        point,
        image_width=width,
        image_height=height,
        monitor_offset=capture.monitor_offset,
    )
    logger.info(
        "Click point norm=%s → screen=(%d, %d) image=%dx%d",
        point,
        screen_x,
        screen_y,
        width,
        height,
    )
    return screen_x, screen_y


def save_grounding_debug(
    result: GroundingResult,
    config: AppConfig,
    filename: str,
) -> Path | None:
    """Save annotated grounding image (bbox + click crosshair)."""
    if not result.annotated_image:
        return None
    out_dir = Path(config.paths.annotated_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename}.png"
    result.annotated_image.save(out_path)
    logger.info("Grounding debug image saved: %s", out_path)
    return out_path


def log_grounding_result(result: GroundingResult) -> None:
    logger.info("Bbox: %s", result.bbox.as_tuple())
    logger.info("Click point: %s", result.click_point or result.center)
    logger.info("Confidence: %.2f", result.confidence)
    logger.info("Trace: %s", result.search_trace)
