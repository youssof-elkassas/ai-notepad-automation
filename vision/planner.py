"""Planner VLM for ScreenSeekeR Position Inference and Result Checking."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from PIL import Image

from core.config import AppConfig
from core.exceptions import PlannerError
from vision.gui_parser import Viewport

logger = logging.getLogger(__name__)

# Paper Appendix C, Table 7
POSITION_INFERENCE_PROMPT = """I want to identify a UI element that best matches my instruction. Please help me determine which region(s) of the screenshot to focus on and list the UI elements that might appear next to the target.
If the target does not exist in the screenshot, please output "No target".
Output Requirements:
1. List the possible regions in descending order of probability.
2. Always make specific, clear and unique references to avoid ambiguity. References such as "Other icons" and "window" are NOT allowed.
3. Use the following XML tags to describe items in the screenshot:
- <element>: Wrap a specific UI element.
- <area>: Describe an area of the UI containing multiple elements.
- <neighbor>: Describe a UI element that may appear around the target.
Example Output:
The shortcut link is most likely to be found in the <area>Settings window</area>, in the <area>tools panel</area>, next to the <neighbor>Search button</neighbor>.
Important Notes:
- The target UI element is guaranteed to be present in the screenshot.
Do not speculate about operations that could change the screenshot.
Instruction:
{instruction}"""

# Paper Appendix C, Table 8
RESULT_CHECKING_PROMPT = """You are given a cropped screenshot. Your task is to evaluate whether the marked element in the red box matches the target described in my instruction.
Please follow these steps:
1. Analyze the screenshot by describing its visible content and functionalities.
2. Determine which of the following applies:
- 'is_target': The marked element is the target.
- 'target_elsewhere': The marked element is not the target, but it exists elsewhere.
- 'target_not_found': The marked element is not the target, and it does not exist.
3. If the target exists, rewrite the instruction to make it clearer.
After your analysis, provide the result in JSON format:
- "result": (str) One of 'is_target', 'target_elsewhere', or 'target_not_found'.
- "new_instruction": (str, default null) A clearer version of the instruction.
Here is my instruction:
{instruction}"""

PlannerVerdict = Literal["is_target", "target_elsewhere", "target_not_found"]

REGION_PATTERNS = [
    re.compile(r"<area>(.*?)</area>", re.DOTALL),
    re.compile(r"desktop", re.IGNORECASE),
    re.compile(r"taskbar", re.IGNORECASE),
    re.compile(r"left side", re.IGNORECASE),
    re.compile(r"right side", re.IGNORECASE),
    re.compile(r"center", re.IGNORECASE),
    re.compile(r"top", re.IGNORECASE),
    re.compile(r"bottom", re.IGNORECASE),
]

# Heuristic region → normalized viewport mapping for Windows desktop
REGION_VIEWPORTS: dict[str, Viewport] = {
    "desktop_left": Viewport(0.0, 0.0, 0.35, 0.85),
    "desktop_center": Viewport(0.25, 0.2, 0.75, 0.8),
    "desktop_right": Viewport(0.65, 0.0, 1.0, 0.85),
    "desktop_top": Viewport(0.0, 0.0, 1.0, 0.35),
    "desktop_bottom": Viewport(0.0, 0.65, 1.0, 0.95),
    "taskbar": Viewport(0.0, 0.92, 1.0, 1.0),
    "full": Viewport(0.0, 0.0, 1.0, 1.0),
}


@dataclass
class PlannerResult:
    """Structured planner output."""

    regions: list[Viewport]
    raw_text: str
    neighbors: list[str]


@dataclass
class VerificationResult:
    """Result checking output."""

    verdict: PlannerVerdict
    new_instruction: str | None
    raw_text: str


class PlannerBackend(Protocol):
    def infer_regions(self, image: Image.Image, instruction: str) -> PlannerResult: ...
    def verify_result(
        self, image: Image.Image, instruction: str, bbox_viewport: Viewport
    ) -> VerificationResult: ...


def _parse_regions_from_text(text: str) -> list[Viewport]:
    """Parse planner text into candidate search viewports."""
    regions: list[Viewport] = []
    lower = text.lower()

    if "no target" in lower:
        return [REGION_VIEWPORTS["full"]]

    if any(k in lower for k in ("left", "top-left", "upper left")):
        regions.append(REGION_VIEWPORTS["desktop_left"])
    if "center" in lower or "middle" in lower:
        regions.append(REGION_VIEWPORTS["desktop_center"])
    if any(k in lower for k in ("right", "bottom-right", "lower right")):
        regions.append(REGION_VIEWPORTS["desktop_right"])
    if "taskbar" in lower or "bottom" in lower:
        regions.append(REGION_VIEWPORTS["taskbar"])
    if "top" in lower and "taskbar" not in lower:
        regions.append(REGION_VIEWPORTS["desktop_top"])

    if not regions:
        regions = [
            REGION_VIEWPORTS["desktop_left"],
            REGION_VIEWPORTS["desktop_center"],
            REGION_VIEWPORTS["full"],
        ]
    return regions


def _parse_verification_json(text: str) -> VerificationResult:
    """Extract JSON verdict from planner response."""
    json_match = re.search(r"\{[^{}]*\"result\"[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            verdict = data.get("result", "target_not_found")
            if verdict not in ("is_target", "target_elsewhere", "target_not_found"):
                verdict = "target_not_found"
            return VerificationResult(
                verdict=verdict,
                new_instruction=data.get("new_instruction"),
                raw_text=text,
            )
        except json.JSONDecodeError:
            pass

    lower = text.lower()
    if "is_target" in lower:
        verdict: PlannerVerdict = "is_target"
    elif "target_elsewhere" in lower:
        verdict = "target_elsewhere"
    else:
        verdict = "target_not_found"
    return VerificationResult(verdict=verdict, new_instruction=None, raw_text=text)


class QwenPlanner:
    """Qwen2.5-VL planner substituting GPT-4o from the paper."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise PlannerError(
                "Vision dependencies not installed. Run: uv sync --extra vision"
            ) from exc

        model_id = self.config.models.planner_id
        logger.info("Loading planner model: %s", model_id)
        kwargs: dict = {"torch_dtype": "auto", "device_map": "auto"}
        if self.config.models.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except ImportError:
                logger.warning("bitsandbytes unavailable for planner")

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **kwargs
        )
        self._processor = AutoProcessor.from_pretrained(model_id)
        if self.config.models.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable for planner; using CPU")

    def _run_vlm(self, image: Image.Image, prompt: str) -> str:
        self._ensure_loaded()
        from qwen_vl_utils import process_vision_info

        # Downscale for planner overview on low-spec
        overview = image
        if self.config.profile == "low" and image.width > 1280:
            ratio = 1280 / image.width
            overview = image.resize(
                (1280, int(image.height * ratio)), Image.Resampling.LANCZOS
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": overview},
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
        generated_ids = self._model.generate(**inputs, max_new_tokens=512)
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        output = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output[0] if output else ""

    def infer_regions(self, image: Image.Image, instruction: str) -> PlannerResult:
        prompt = POSITION_INFERENCE_PROMPT.format(instruction=instruction)
        raw = self._run_vlm(image, prompt)
        regions = _parse_regions_from_text(raw)
        neighbors = re.findall(r"<neighbor>(.*?)</neighbor>", raw, re.DOTALL)
        return PlannerResult(regions=regions, raw_text=raw, neighbors=neighbors)

    def verify_result(
        self,
        image: Image.Image,
        instruction: str,
        bbox_viewport: Viewport,
    ) -> VerificationResult:
        from utils.image import draw_bbox

        annotated = draw_bbox(
            image,
            (bbox_viewport.x1, bbox_viewport.y1, bbox_viewport.x2, bbox_viewport.y2),
            color="red",
            width=4,
        )
        prompt = RESULT_CHECKING_PROMPT.format(instruction=instruction)
        raw = self._run_vlm(annotated, prompt)
        return _parse_verification_json(raw)


class MockPlanner:
    """Deterministic planner for unit tests."""

    def __init__(
        self,
        regions: list[Viewport] | None = None,
        verdict: PlannerVerdict = "is_target",
    ) -> None:
        self.regions = regions or [REGION_VIEWPORTS["full"]]
        self.verdict = verdict

    def infer_regions(self, image: Image.Image, instruction: str) -> PlannerResult:
        return PlannerResult(
            regions=self.regions,
            raw_text="mock planner output",
            neighbors=[],
        )

    def verify_result(
        self,
        image: Image.Image,
        instruction: str,
        bbox_viewport: Viewport,
    ) -> VerificationResult:
        return VerificationResult(
            verdict=self.verdict,
            new_instruction=None,
            raw_text='{"result": "' + self.verdict + '"}',
        )
