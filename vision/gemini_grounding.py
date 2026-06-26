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
from vision.gui_parser import Bbox, GuiParser, Viewport

logger = logging.getLogger(__name__)

GEMINI_GROUNDING_PROMPT = """You are a GUI visual grounding assistant for Windows desktop automation.

Locate this UI element in the screenshot:
"{instruction}"

Return JSON with:
- found: true if the element is visible
- confidence: 0.0 to 1.0
- bbox_1000: [x1, y1, x2, y2] on a 0-1000 scale (origin top-left of the full image)
- description: brief label

Rules for Windows desktop icons:
- Box ONLY the square icon graphic (the colored symbol), NOT the text label below it.
- The clickable target is the icon image in the upper part of the desktop shortcut.
- bbox_1000 must be a tight rectangle around the icon graphic only.
- If not visible, set found=false, confidence=0, bbox_1000=[0,0,0,0].
"""

GEMINI_REFINE_PROMPT = """You are refining a crop of a Windows desktop screenshot.

Find the application icon GRAPHIC (colored symbol only) matching:
"{instruction}"

Return JSON with bbox_1000 tightly around the square icon image.
Do NOT include the text label under the icon.
Coordinates are relative to this cropped image (0-1000 scale).
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
        repaired = _repair_truncated_json(text)
        if repaired is None:
            raise BboxParseError(f"Gemini returned invalid JSON: {text[:300]}") from exc
        data = repaired
    if not isinstance(data, dict):
        raise BboxParseError("Gemini JSON root must be an object")
    return data


def _repair_truncated_json(text: str) -> dict[str, Any] | None:
    """Best-effort repair for truncated model JSON."""
    match = re.search(
        r'"bbox_1000"\s*:\s*\[([^\]]*)|"bbox"\s*:\s*\[([^\]]*)',
        text,
    )
    if not match:
        return None
    coords_raw = match.group(1) or match.group(2)
    numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", coords_raw)]
    if len(numbers) != 4:
        return None
    found_match = re.search(r'"found"\s*:\s*(true|false)', text, re.IGNORECASE)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    return {
        "found": found_match.group(1).lower() == "true" if found_match else True,
        "confidence": float(conf_match.group(1)) if conf_match else 0.7,
        "bbox_1000": numbers,
    }


def _response_text(response) -> str:
    """Collect full text from a GenerateContentResponse."""
    text = getattr(response, "text", None) or ""
    if text.strip():
        return text

    chunks: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)
    return "".join(chunks)


def parse_grounding_response(response) -> dict[str, Any]:
    """Extract grounding dict from a google-genai GenerateContentResponse."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict) and _payload_has_bbox(parsed):
        return parsed
    if parsed is not None and hasattr(parsed, "model_dump"):
        data = parsed.model_dump()
        if isinstance(data, dict) and _payload_has_bbox(data):
            return data

    text = _response_text(response)
    if not text.strip():
        raise GroundingError("GenAI returned an empty response")
    return parse_gemini_grounding_json(text)


def _payload_has_bbox(data: dict[str, Any]) -> bool:
    coords = data.get("bbox_1000") or data.get("bbox")
    return isinstance(coords, list) and len(coords) == 4


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


def refinement_viewport(rough_bbox: Bbox) -> Viewport:
    """Expand bbox upward to include icon graphic above a mis-detected label."""
    width = rough_bbox.x2 - rough_bbox.x1
    height = rough_bbox.y2 - rough_bbox.y1
    pad_x = max(width * 0.6, 0.02)
    pad_top = max(height * 1.5, 0.04)
    pad_bottom = max(height * 0.4, 0.01)
    expanded = rough_bbox.expand(
        pad_left=pad_x,
        pad_right=pad_x,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
    )
    return Viewport(*expanded.as_tuple())


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

    def _generate_grounding(
        self,
        image: Image.Image,
        prompt: str,
    ) -> tuple[Bbox, str, float, bool, str]:
        api_image = self._prepare_api_image(image)
        gen_config = build_grounding_config(self.config)
        models = self._models_to_try()
        last_quota_error: Exception | None = None

        for model in models:
            logger.info("GenAI generate_content model=%s", model)
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
                return bbox, raw, confidence, found, model

            except BboxParseError as exc:
                logger.warning("GenAI JSON parse failed on %s: %s", model, exc)
                if model != models[-1]:
                    continue
                raise
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

    def ground(self, image: Image.Image, instruction: str) -> tuple[Bbox, str, float, bool]:
        """Call GenAI and return (bbox, raw_text, confidence, found)."""
        prompt = GEMINI_GROUNDING_PROMPT.format(instruction=instruction)
        bbox, raw, confidence, found, _model = self._generate_grounding(image, prompt)
        return bbox, raw, confidence, found

    def refine(
        self,
        image: Image.Image,
        instruction: str,
        rough_bbox: Bbox,
    ) -> tuple[Bbox, str, float, bool]:
        """Re-ground inside an expanded crop; fall back to rough_bbox on failure."""
        try:
            viewport = refinement_viewport(rough_bbox)
            crop = self._parser.crop_viewport(image, viewport)
            prompt = GEMINI_REFINE_PROMPT.format(instruction=instruction)
            local_bbox, raw, confidence, found, _model = self._generate_grounding(crop, prompt)
            if not found:
                logger.warning("Refinement: element not found in crop; keeping initial bbox")
                return rough_bbox, raw, confidence, False
            global_bbox = self._parser.local_to_global(local_bbox, viewport)
            return global_bbox, raw, confidence, True
        except (BboxParseError, GroundingError) as exc:
            logger.warning("Refinement failed (%s); keeping initial bbox", exc)
            return rough_bbox, "", 0.0, False


class GeminiGroundingService:
    """High-level grounding API backed by google-genai."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._grounder = GeminiGrounder(config)
        self._parser = GuiParser(config.screen.width, config.screen.height)

    def locate(self, instruction: str, screenshot: Image.Image) -> GroundingResult:
        image = self._parser.normalize_screenshot(screenshot)
        bbox, raw, model_confidence, found = self._grounder.ground(image, instruction)
        trace = [f"genai:{self.config.gemini.model}"]

        if found and self.config.grounding.refine_grounding:
            refined_bbox, refine_raw, refine_conf, refine_found = self._grounder.refine(
                image,
                instruction,
                bbox,
            )
            if refine_found:
                bbox = refined_bbox
                raw = refine_raw
                model_confidence = max(model_confidence, refine_conf)
                trace.append("genai:refine")

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
        click_point = bbox.click_point(
            vertical_bias=self.config.grounding.click_vertical_bias,
        )

        return GroundingResult(
            bbox=bbox,
            center=bbox.center,
            confidence=confidence,
            raw_output=raw,
            search_trace=trace,
            annotated_image=self._parser.annotate(
                image,
                bbox,
                label=f"{instruction} → click",
                click_point=click_point,
            ),
            planner_verdict="is_target",
            click_point=click_point,
            image_size=(image.width, image.height),
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
        click_point = bbox.click_point(
            vertical_bias=self.config.grounding.click_vertical_bias,
        )
        return GroundingResult(
            bbox=bbox,
            center=bbox.center,
            confidence=confidence,
            raw_output=raw,
            search_trace=["mock"],
            annotated_image=self._parser.annotate(image, bbox, label=instruction),
            planner_verdict="is_target",
            click_point=click_point,
            image_size=(image.width, image.height),
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
