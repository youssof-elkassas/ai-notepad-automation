"""OS-Atlas grounder: model loading, inference, and bbox parsing."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from PIL import Image

from core.config import AppConfig
from core.exceptions import BboxParseError, GroundingError
from vision.gui_parser import Bbox, GuiParser, Viewport

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

GROUNDING_PROMPT = (
    'In this UI screenshot, what is the position of the element corresponding '
    'to the command "{instruction}" (with bbox)?'
)

BOX_PATTERN = re.compile(
    r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>"
)
BOX_PATTERN_ALT = re.compile(r"\((\d+),(\d+)\),\((\d+),(\d+)\)")


@dataclass
class GroundingResult:
    """Result of a grounding operation."""

    bbox: Bbox
    center: tuple[float, float]
    confidence: float
    raw_output: str
    search_trace: list[str] = field(default_factory=list)
    annotated_image: Image.Image | None = None
    planner_verdict: str | None = None


def parse_os_atlas_bbox(raw: str, img_w: int, img_h: int) -> Bbox:
    """Parse OS-Atlas 0-1000 normalized output to [0,1] Bbox."""
    match = BOX_PATTERN.search(raw) or BOX_PATTERN_ALT.search(raw)
    if not match:
        raise BboxParseError(f"Could not parse bbox from model output: {raw[:200]}")

    x1, y1, x2, y2 = (int(match.group(i)) for i in range(1, 5))
    return Bbox(
        x1=x1 / 1000.0,
        y1=y1 / 1000.0,
        x2=x2 / 1000.0,
        y2=y2 / 1000.0,
    ).clamp()


def compute_confidence(
    bbox: Bbox,
    img_w: int,
    img_h: int,
    *,
    min_area: int,
    max_area: int,
    parse_ok: bool = True,
    planner_verdict: str | None = None,
) -> float:
    """Heuristic confidence score in [0, 1]."""
    score = 0.0
    if parse_ok:
        score += 0.4
    area = bbox.area_px(img_w, img_h)
    if min_area <= area <= max_area:
        score += 0.3
    elif area > 0:
        score += 0.1
    if planner_verdict == "is_target":
        score += 0.3
    elif planner_verdict == "target_elsewhere":
        score += 0.1
    return min(1.0, score)


class GrounderBackend(Protocol):
    def ground(self, image: Image.Image, instruction: str) -> str: ...


class OSAtlasGrounder:
    """OS-Atlas vision-language grounding model wrapper."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None
        self._parser = GuiParser(config.screen.width, config.screen.height)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise GroundingError(
                "Vision dependencies not installed. Run: uv sync --extra vision"
            ) from exc

        model_id = self.config.models.grounder_id
        logger.info("Loading grounder model: %s", model_id)

        kwargs: dict = {"torch_dtype": "auto", "device_map": "auto"}
        if self.config.models.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except ImportError:
                logger.warning("bitsandbytes unavailable; loading full precision")

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._device = self.config.models.device
        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; grounder will use CPU (slow)")
            self._device = "cpu"

    def ground(self, image: Image.Image, instruction: str) -> str:
        """Run OS-Atlas inference and return raw text output."""
        self._ensure_loaded()
        from qwen_vl_utils import process_vision_info

        prompt = GROUNDING_PROMPT.format(instruction=instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        generated_ids = self._model.generate(**inputs, max_new_tokens=128)
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        output = self._processor.batch_decode(
            trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        return output[0] if output else ""

    def ground_bbox(
        self,
        image: Image.Image,
        instruction: str,
        *,
        viewport: Viewport | None = None,
        full_image: Image.Image | None = None,
    ) -> tuple[Bbox, str]:
        """Ground instruction in image; return local bbox and raw output."""
        raw = self.ground(image, instruction)
        local = parse_os_atlas_bbox(raw, image.width, image.height)
        if viewport and full_image:
            global_bbox = self._parser.local_to_global(local, viewport)
            return global_bbox, raw
        return local, raw


class MockGrounder:
    """Deterministic grounder for tests without GPU."""

    def __init__(self, bbox: Bbox | None = None) -> None:
        self.bbox = bbox or Bbox(0.1, 0.1, 0.15, 0.18)

    def ground(self, image: Image.Image, instruction: str) -> str:
        b = self.bbox
        return (
            f"<|box_start|>({int(b.x1*1000)},{int(b.y1*1000)}),"
            f"({int(b.x2*1000)},{int(b.y2*1000)})<|box_end|>"
        )

    def ground_bbox(
        self,
        image: Image.Image,
        instruction: str,
        *,
        viewport: Viewport | None = None,
        full_image: Image.Image | None = None,
    ) -> tuple[Bbox, str]:
        raw = self.ground(image, instruction)
        # Mock always returns the configured global bbox (ScreenSeekeR expects global coords)
        return self.bbox, raw


class GroundingService:
    """High-level generic grounding API."""

    def __init__(
        self,
        grounder: OSAtlasGrounder | MockGrounder,
        config: AppConfig,
        screenseeker: object | None = None,
    ) -> None:
        self.grounder = grounder
        self.config = config
        self.screenseeker = screenseeker
        self._parser = GuiParser(config.screen.width, config.screen.height)

    def locate(self, instruction: str, screenshot: Image.Image) -> GroundingResult:
        """Locate UI element by natural language instruction."""
        image = self._parser.normalize_screenshot(screenshot)

        if self.screenseeker is not None:
            return self.screenseeker.search(instruction, image)

        bbox, raw = self.grounder.ground_bbox(image, instruction)
        confidence = compute_confidence(
            bbox,
            image.width,
            image.height,
            min_area=self.config.grounding.min_bbox_area_px,
            max_area=self.config.grounding.max_bbox_area_px,
        )
        return GroundingResult(
            bbox=bbox,
            center=bbox.center,
            confidence=confidence,
            raw_output=raw,
            annotated_image=self._parser.annotate(image, bbox, label=instruction),
        )
