"""GUI visual grounding via the official google-genai SDK."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from PIL import Image

from core.config import AppConfig
from core.exceptions import BboxParseError, GeminiQuotaError, GroundingError, LowConfidenceError
from vision.genai_client import build_grounding_config, create_genai_client
from vision.grounding import GroundingResult, MockGrounder, compute_confidence
from vision.gui_parser import Bbox, GuiParser

logger = logging.getLogger(__name__)

GEMINI_GROUNDING_PROMPT = """You are a GUI visual grounding assistant for Windows desktop automation.

Locate the UI element that best matches this instruction:
"{instruction}"

Return JSON with:
- found: true if the element is visible
- confidence: 0.0 to 1.0
- bbox_1000: [x1, y1, x2, y2] on a 0-1000 scale (top-left origin)
- description: brief label for the matched element

Rules:
- bbox must tightly wrap the clickable icon and its label text.
- If not visible, set found=false, confidence=0, bbox_1000=[0,0,0,0].
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


def parse_grounding_response(response) -> dict[str, Any]:
    """Extract grounding dict from a google-genai GenerateContentResponse."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    if parsed is not None:
        return dict(parsed)
    text = getattr(response, "text", None) or ""
    if not text.strip():
        raise GroundingError("GenAI returned an empty response")
    return parse_gemini_grounding_json(text)


def bbox_from_gemini_payload(data: dict[str, Any]) -> Bbox:
    """Convert Gemini bbox_1000 array to normalized Bbox."""
    coords = data.get("bbox_1000") or data.get("bbox")
    if not coords or len(coords) != 4:
        raise BboxParseError(f"Missing bbox_1000 in Gemini response: {data}")

    x1, y1, x2, y2 = (float(v) for v in coords)
    scale = 1.0 if max(x1, y1, x2, y2) <= 1.0 else 1000.0
    return Bbox(
        x1=x1 / scale,
        y1=y1 / scale,
        x2=x2 / scale,
        y2=y2 / scale,
    ).clamp()


class GeminiGrounder:
    """Ground UI elements using google.genai.Client."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._parser = GuiParser(config.screen.width, config.screen.height)
        self._client = create_genai_client(config)

    def _prepare_api_image(self, image: Image.Image) -> Image.Image:
        max_w = self.config.gemini.max_image_width
        if image.width <= max_w:
            return image.convert("RGB")
        ratio = max_w / image.width
        resized = image.resize((max_w, int(image.height * ratio)), Image.Resampling.LANCZOS)
        logger.info("Resized screenshot for GenAI: %s → %s", image.size, resized.size)
        return resized.convert("RGB")

    def _models_to_try(self) -> list[str]:
        models = [self.config.gemini.model]
        for m in self.config.gemini.fallback_models:
            if m not in models:
                models.append(m)
        return models

    def ground(self, image: Image.Image, instruction: str) -> tuple[Bbox, str, float, bool]:
        """Call GenAI and return (bbox, raw_text, confidence, found)."""
        prompt = GEMINI_GROUNDING_PROMPT.format(instruction=instruction)
        api_image = self._prepare_api_image(image)
        gen_config = build_grounding_config(self.config)
        models = self._models_to_try()
        last_quota_error: Exception | None = None

        for model in models:
            logger.info("GenAI generate_content model=%s instruction=%s", model, instruction)
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=[api_image, prompt],
                    config=gen_config,
                )
                data = parse_grounding_response(response)
                raw = json.dumps(data)
                found = bool(data.get("found", False))
                confidence = float(data.get("confidence", 0.0))
                bbox = bbox_from_gemini_payload(data) if found else Bbox(0, 0, 0, 0)

                logger.info(
                    "GenAI result (%s): found=%s confidence=%.2f bbox=%s",
                    model,
                    found,
                    confidence,
                    bbox.as_tuple(),
                )
                return bbox, raw, confidence, found

            except Exception as exc:
                if _is_rate_limit_error(exc):
                    last_quota_error = exc
                    logger.warning("GenAI rate limit on %s: %s", model, exc)
                    if model != models[-1]:
                        time.sleep(self.config.gemini.rate_limit_wait_seconds)
                        continue
                    raise _quota_error_message(exc) from exc
                if _is_model_not_found_error(exc):
                    logger.warning("GenAI model unavailable: %s — trying next", model)
                    continue
                raise

        if last_quota_error:
            raise _quota_error_message(last_quota_error) from last_quota_error
        raise GroundingError("GenAI grounding failed for all configured models")


class GeminiGroundingService:
    """High-level grounding API backed by google-genai."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._grounder = GeminiGrounder(config)
        self._parser = GuiParser(config.screen.width, config.screen.height)

    def locate(self, instruction: str, screenshot: Image.Image) -> GroundingResult:
        image = self._parser.normalize_screenshot(screenshot)
        bbox, raw, model_confidence, found = self._grounder.ground(image, instruction)

        if not found:
            raise GroundingError(f"GenAI could not find element: {instruction}")

        if model_confidence < self.config.gemini.min_confidence:
            raise LowConfidenceError(
                f"GenAI confidence too low for: {instruction}",
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

        heuristic = compute_confidence(
            bbox,
            image.width,
            image.height,
            min_area=self.config.grounding.min_bbox_area_px,
            max_area=self.config.grounding.max_bbox_area_px,
            planner_verdict="is_target",
        )
        confidence = min(1.0, (model_confidence + heuristic) / 2)

        return GroundingResult(
            bbox=bbox,
            center=bbox.center,
            confidence=confidence,
            raw_output=raw,
            search_trace=[f"genai:{self.config.gemini.model}"],
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
    if use_mock:
        return MockGroundingService(config, bbox=mock_bbox)
    return GeminiGroundingService(config)


def _is_rate_limit_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "quota" in text


def _is_model_not_found_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "404" in text or "not found" in text or "not supported for generatecontent" in text


def _quota_error_message(exc: Exception) -> GeminiQuotaError:
    return GeminiQuotaError(
        "GenAI API quota/rate limit exceeded (HTTP 429).\n"
        "Check usage: https://ai.dev/rate-limit\n"
        "Or wait and retry / enable billing in Google AI Studio."
    )
