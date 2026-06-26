"""PIL image helpers."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def draw_bbox(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    *,
    color: str = "red",
    width: int = 3,
    label: str | None = None,
) -> Image.Image:
    """Draw a normalized [0,1] bbox on a copy of the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size
    x1, y1, x2, y2 = bbox
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    draw.rectangle([px1, py1, px2, py2], outline=color, width=width)
    if label:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        draw.text((px1, max(0, py1 - 20)), label, fill=color, font=font)
    return annotated


def draw_click_point(
    image: Image.Image,
    point: tuple[float, float],
    *,
    color: str = "lime",
    radius: int = 8,
) -> Image.Image:
    """Mark the normalized click target on a copy of the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size
    cx, cy = int(point[0] * w), int(point[1] * h)
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=color,
        width=3,
    )
    draw.line([cx - radius * 2, cy, cx + radius * 2, cy], fill=color, width=2)
    draw.line([cx, cy - radius * 2, cx, cy + radius * 2], fill=color, width=2)
    return annotated


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return normalized center of a bbox."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_area_px(bbox: tuple[float, float, float, float], width: int, height: int) -> float:
    """Return bbox area in pixels."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * width * (y2 - y1) * height
