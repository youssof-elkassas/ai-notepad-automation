"""GUI visual grounding via Google Gemini API (cloud vision-language model)."""

from __future__ import annotations

import io
import json
import logging
import os
import re
from typing import Any

from PIL import Image

from core.config import AppConfig
from core.exceptions import BboxParseError, ConfigurationError, GroundingError, LowConfidenceError
from vision.grounding import GroundingResult, MockGrounder, compute_confidence
from vision.gui_parser import Bbox, GuiParser

logger = logging.getLogger(__name__)

GEMINI_GROUNDING_PROMPT = """You are a GUI visual grounding assistant for Windows desktop automation.

Given a screenshot, locate the UI element that best matches this instruction:
"{instruction}"

Return ONLY valid JSON (no markdown) with this exact schema:
{{
  "found": boolean,
  "confidence": number between 0 and 1,
  "bbox_1000": [x1, y1, x2, y2],
  "description": "brief description of the matched element"
}}

Rules:
- bbox_1000 uses coordinates on a 0-1000 scale relative to the image (top-left is origin).
- x1,y1 is top-left corner; x2,y2 is bottom-right corner of the clickable region.
- If the element is not visible, set found=false, confidence=0, bbox_1000=[0,0,0,0].
- Target desktop icons including their label text below the icon.
- Be precise: the box should tightly wrap the interactable icon+label area.
"""


def parse_gemini_grounding_json(raw: str) -> dict[str, Any]:
    """Parse Gemini JSON response, tolerating optional markdown fences."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BboxParseError(f"Gemini returned invalid JSON: {raw[:300]}") from exc
    if not isinstance(data, dict):
        raise BboxParseError("Gemini JSON root must be an object")
    return data


def bbox_from_gemini_payload(data: dict[str, Any]) -> Bbox:
    """Convert Gemini bbox_1000 array to normalized Bbox."""
    coords = data.get("bbox_1000") or data.get("bbox")
    if not coords or len(coords) != 4:
        raise BboxParseError(f"Missing bbox_1000 in Gemini response: {data}")

    x1, y1, x2, y2 = (float(v) for v in coords)
    # Accept 0-1 floats as well as 0-1000 ints
    if max(x1, y1, x2, y2) <= 1.0:
        scale = 1.0
    else:
        scale = 1000.0

    return Bbox(
        x1=x1 / scale,
        y1=y1 / scale,
        x2=x2 / scale,
        y2=y2 / scale,
    ).clamp()


class GeminiGrounder:
    """Ground UI elements using the Gemini multimodal API."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._parser = GuiParser(config.screen.width, config.screen.height)
        self._client = None

    def _api_key(self) -> str:
        env_name = self.config.gemini.api_key_env
        key = os.environ.get(env_name, "").strip()
        if not key:
            raise ConfigurationError(
                f"Gemini API key not set. Export {env_name}=your_key "
                f"(get one at https://aistudio.google.com/apikey)"
            )
        return key

    def _client_instance(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise GroundingError(
                "google-genai not installed. Run: uv sync"
            ) from exc
        self._client = genai.Client(api_key=self._api_key())
        return self._client

    def _image_bytes(self, image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    def ground(self, image: Image.Image, instruction: str) -> tuple[Bbox, str, float, bool]:
        """Call Gemini and return (bbox, raw_text, confidence, found)."""
        from google.genai import types

        client = self._client_instance()
        prompt = GEMINI_GROUNDING_PROMPT.format(instruction=instruction)
        image_bytes = self._image_bytes(image)

        logger.info("Calling Gemini model=%s for: %s", self.config.gemini.model, instruction)

        response = client.models.generate_content(
            model=self.config.gemini.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=self.config.gemini.temperature,
                response_mime_type="application/json",
                max_output_tokens=self.config.gemini.max_output_tokens,
            ),
        )

        raw = response.text or ""
        if not raw.strip():
            raise GroundingError("Gemini returned an empty response")

        data = parse_gemini_grounding_json(raw)
        found = bool(data.get("found", False))
        confidence = float(data.get("confidence", 0.0))
        bbox = bbox_from_gemini_payload(data) if found else Bbox(0, 0, 0, 0)

        logger.info(
            "Gemini result: found=%s confidence=%.2f bbox=%s",
            found,
            confidence,
            bbox.as_tuple(),
        )
        return bbox, raw, confidence, found


class GeminiGroundingService:
    """High-level grounding API backed by Gemini."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._grounder = GeminiGrounder(config)
        self._parser = GuiParser(config.screen.width, config.screen.height)

    def locate(self, instruction: str, screenshot: Image.Image) -> GroundingResult:
        image = self._parser.normalize_screenshot(screenshot)
        bbox, raw, model_confidence, found = self._grounder.ground(image, instruction)

        if not found:
            raise GroundingError(f"Gemini could not find element: {instruction}")

        if model_confidence < self.config.gemini.min_confidence:
            raise LowConfidenceError(
                f"Gemini confidence too low for: {instruction}",
                reason=f"confidence={model_confidence:.2f}",
                confidence=model_confidence,
            )

        area = bbox.area_px(image.width, image.height)
        if area < self.config.grounding.min_bbox_area_px:
            raise LowConfidenceError(
                f"Bbox too small ({area:.0f}px) for: {instruction}",
                reason="bbox_area_below_minimum",
                confidence=model_confidence,
            )
        if area > self.config.grounding.max_bbox_area_px:
            raise LowConfidenceError(
                f"Bbox too large ({area:.0f}px) for: {instruction}",
                reason="bbox_area_above_maximum",
                confidence=model_confidence,
            )

        heuristic_confidence = compute_confidence(
            bbox,
            image.width,
            image.height,
            min_area=self.config.grounding.min_bbox_area_px,
            max_area=self.config.grounding.max_bbox_area_px,
            planner_verdict="is_target",
        )
        confidence = min(1.0, (model_confidence + heuristic_confidence) / 2)

        return GroundingResult(
            bbox=bbox,
            center=bbox.center,
            confidence=confidence,
            raw_output=raw,
            search_trace=[f"gemini:{self.config.gemini.model}"],
            annotated_image=self._parser.annotate(image, bbox, label=instruction),
            planner_verdict="is_target",
        )


class MockGroundingService:
    """Offline grounding for tests using deterministic bbox."""

    def __init__(self, config: AppConfig, bbox: Bbox | None = None) -> None:
        self.config = config
        self._mock = MockGrounder(bbox=bbox)
        self._parser = GuiParser(config.screen.width, config.screen.height)

    def locate(self, instruction: str, screenshot: Image.Image) -> GroundingResult:
        image = self._parser.normalize_screenshot(screenshot)
        bbox, raw = self._mock.ground_bbox(image, instruction)
        confidence = compute_confidence(
            bbox,
            image.width,
            image.height,
            min_area=self.config.grounding.min_bbox_area_px,
            max_area=self.config.grounding.max_bbox_area_px,
            planner_verdict="is_target",
        )
        return GroundingResult(
            bbox=bbox,
            center=bbox.center,
            confidence=confidence,
            raw_output=raw,
            search_trace=["mock"],
            annotated_image=self._parser.annotate(image, bbox, label=instruction),
            planner_verdict="is_target",
        )


def create_grounding_service(
    config: AppConfig,
    *,
    use_mock: bool = False,
    mock_bbox: Bbox | None = None,
) -> GeminiGroundingService | MockGroundingService:
    """Factory for the active grounding backend."""
    if use_mock:
        return MockGroundingService(config, bbox=mock_bbox)
    return GeminiGroundingService(config)
