"""Configuration loading with profile merge."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from core.exceptions import ConfigurationError
from core.env import load_project_dotenv


class ScreenConfig(BaseModel):
    width: int = 1920
    height: int = 1080
    warn_on_dpi_mismatch: bool = True


class GroundingConfig(BaseModel):
    icon_instruction: str = "Notepad desktop icon"
    min_bbox_area_px: int = 400
    max_bbox_area_px: int = 20000
    require_planner_verdict: bool = True


class ScreenSeekeRConfig(BaseModel):
    max_depth: int = 3
    min_patch_size_px: int = 1280
    sigma: float = 0.3
    nms_iou_threshold: float = 0.5
    max_candidates: int = 5
    reground_crop_size: int = 1024
    use_regound_fallback: bool = False


class RetryConfig(BaseModel):
    screenshot_max_attempts: int = 3
    grounding_max_attempts: int = 3
    click_verify_max_attempts: int = 2
    backoff_base_seconds: float = 1.0
    backoff_jitter_seconds: float = 0.5


class TimeoutConfig(BaseModel):
    grounding_seconds: int = 120
    window_wait_seconds: int = 15
    save_dialog_seconds: int = 10
    notepad_launch_seconds: int = 15


class PathsConfig(BaseModel):
    project_folder: str = "tjm-project"
    screenshots_dir: str = "screenshots"
    failures_dir: str = "screenshots/failures"
    annotated_dir: str = "screenshots/annotated"
    models_dir: str = "models"


class ApiConfig(BaseModel):
    posts_url: str = "https://jsonplaceholder.typicode.com/posts"
    posts_limit: int = 10


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_file: str = "screenshots/automation.log"


class AutomationConfig(BaseModel):
    mouse_fail_safe: bool = False
    typing_interval_seconds: float = 0.02


class GeminiConfig(BaseModel):
    model: str = "gemini-2.5-flash"
    api_key: str = ""  # optional; prefer .env or config/secrets.yaml
    api_key_env: str = "GOOGLE_API_KEY"
    temperature: float = 0.1
    max_output_tokens: int = 512
    min_confidence: float = 0.5
    max_image_width: int = 1280  # shrink screenshot before API to save quota/tokens
    fallback_models: list[str] = Field(
        default_factory=lambda: ["gemini-2.0-flash-lite"]
    )
    rate_limit_wait_seconds: float = 45.0


class ModelsConfig(BaseModel):
    grounder_id: str = "OS-Copilot/OS-Atlas-Base-7B"
    planner_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    load_in_4bit: bool = False
    device: str = "cuda"


class AppConfig(BaseSettings):
    """Merged application configuration."""

    profile: Literal["high", "low"] = "high"
    screen: ScreenConfig = Field(default_factory=ScreenConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    screenseeker: ScreenSeekeRConfig = Field(default_factory=ScreenSeekeRConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    automation: AutomationConfig = Field(default_factory=AutomationConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    model_config = {"extra": "ignore"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_config(
    profile: Literal["high", "low"] = "high",
    config_dir: Path | None = None,
) -> AppConfig:
    """Load default config merged with the selected hardware profile."""
    load_project_dotenv()

    root = config_dir or Path(__file__).resolve().parent.parent / "config"
    default_data = _load_yaml(root / "default.yaml")
    profile_path = root / f"profile_{profile}.yaml"
    profile_data = _load_yaml(profile_path)
    merged = _deep_merge(default_data, profile_data)

    secrets_path = root / "secrets.yaml"
    if secrets_path.exists():
        secrets_data = _load_yaml(secrets_path)
        merged = _deep_merge(merged, secrets_data)

    merged["profile"] = profile
    return AppConfig(**merged)
