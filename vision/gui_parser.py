"""GUI screenshot preprocessing, coordinate transforms, and annotation."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from PIL import Image

from utils.image import draw_bbox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Viewport:
    """Normalized viewport region within the full screenshot [0,1]."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_pixels(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        return (
            int(self.x1 * img_w),
            int(self.y1 * img_h),
            int(self.x2 * img_w),
            int(self.y2 * img_h),
        )


@dataclass(frozen=True)
class Bbox:
    """Bounding box in normalized [0,1] coordinates relative to full screenshot."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def area_px(self, width: int, height: int) -> float:
        return (self.x2 - self.x1) * width * (self.y2 - self.y1) * height

    def clamp(self) -> Bbox:
        return Bbox(
            x1=max(0.0, min(1.0, self.x1)),
            y1=max(0.0, min(1.0, self.y1)),
            x2=max(0.0, min(1.0, self.x2)),
            y2=max(0.0, min(1.0, self.y2)),
        )


class GuiParser:
    """Vision-side preprocessing and coordinate transformation."""

    def __init__(self, target_width: int = 1920, target_height: int = 1080) -> None:
        self.target_width = target_width
        self.target_height = target_height

    def get_dpi_scale(self) -> float:
        """Return Windows DPI scale factor (1.0 = 100%)."""
        if sys.platform != "win32":
            return 1.0
        try:
            import ctypes

            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            dpi = user32.GetDpiForSystem()
            return dpi / 96.0
        except Exception:
            logger.warning("Could not read DPI scale; assuming 1.0")
            return 1.0

    def normalize_screenshot(self, image: Image.Image) -> Image.Image:
        """Ensure screenshot is RGB and matches expected resolution."""
        rgb = image.convert("RGB")
        if rgb.size != (self.target_width, self.target_height):
            logger.warning(
                "Screenshot size %s differs from expected %dx%d",
                rgb.size,
                self.target_width,
                self.target_height,
            )
        return rgb

    def crop_viewport(self, image: Image.Image, viewport: Viewport) -> Image.Image:
        """Crop image to normalized viewport."""
        px1, py1, px2, py2 = viewport.to_pixels(image.width, image.height)
        return image.crop((px1, py1, px2, py2))

    def local_to_global(self, local_bbox: Bbox, viewport: Viewport) -> Bbox:
        """Convert bbox from viewport-local [0,1] to full-screenshot [0,1]."""
        return Bbox(
            x1=viewport.x1 + local_bbox.x1 * viewport.width,
            y1=viewport.y1 + local_bbox.y1 * viewport.height,
            x2=viewport.x1 + local_bbox.x2 * viewport.width,
            y2=viewport.y1 + local_bbox.y2 * viewport.height,
        ).clamp()

    def global_to_local(self, global_bbox: Bbox, viewport: Viewport) -> Bbox:
        """Convert bbox from full-screenshot [0,1] to viewport-local [0,1]."""
        return Bbox(
            x1=(global_bbox.x1 - viewport.x1) / viewport.width,
            y1=(global_bbox.y1 - viewport.y1) / viewport.height,
            x2=(global_bbox.x2 - viewport.x1) / viewport.width,
            y2=(global_bbox.y2 - viewport.y1) / viewport.height,
        ).clamp()

    def normalized_to_screen(
        self,
        point: tuple[float, float],
        monitor_offset: tuple[int, int] = (0, 0),
        dpi_scale: float = 1.0,
    ) -> tuple[int, int]:
        """Convert normalized center to absolute screen pixel coordinates."""
        x_norm, y_norm = point
        screen_x = int(x_norm * self.target_width * dpi_scale) + monitor_offset[0]
        screen_y = int(y_norm * self.target_height * dpi_scale) + monitor_offset[1]
        return screen_x, screen_y

    def viewport_around_point(
        self,
        center: tuple[float, float],
        crop_size_px: int,
        img_w: int,
        img_h: int,
    ) -> Viewport:
        """Create a normalized viewport centered on a point with fixed pixel size."""
        cx, cy = center
        half_w = (crop_size_px / 2) / img_w
        half_h = (crop_size_px / 2) / img_h
        return Viewport(
            x1=max(0.0, cx - half_w),
            y1=max(0.0, cy - half_h),
            x2=min(1.0, cx + half_w),
            y2=min(1.0, cy + half_h),
        )

    def annotate(
        self,
        image: Image.Image,
        bbox: Bbox,
        *,
        label: str | None = None,
    ) -> Image.Image:
        """Draw detection bbox on image."""
        return draw_bbox(image, bbox.as_tuple(), label=label)

    def patch_too_large(self, viewport: Viewport, img_w: int, img_h: int, min_size: int) -> bool:
        """Return True if viewport pixel dimensions exceed min patch size."""
        px_w = viewport.width * img_w
        px_h = viewport.height * img_h
        return px_w > min_size or px_h > min_size
