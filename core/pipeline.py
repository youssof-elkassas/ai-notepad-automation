"""Observe → Reason → Act pipeline orchestrator."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from api.posts import Post, fetch_posts
from automation.keyboard import KeyboardController
from automation.mouse import MouseController
from automation.windows import WindowsCapture
from core.config import AppConfig
from core.exceptions import GroundingError, LowConfidenceError, VerificationError
from core.grounding_workflow import (
    log_grounding_result,
    resolve_grounding_instruction,
    resolve_grounding_with_cache,
    save_grounding_debug,
    to_screen_coords,
)
from core.logger import save_failure_screenshot
from core.retry import retry_with_backoff
from vision.gemini_grounding import create_grounding_service
from vision.grounding import GroundingResult

logger = logging.getLogger(__name__)


class NotepadPipeline:
    """Full automation pipeline with cached icon coordinates after first grounding."""

    def __init__(
        self,
        config: AppConfig,
        *,
        use_mock: bool = False,
    ) -> None:
        self.config = config
        self.capture = WindowsCapture(config)
        self.mouse = MouseController(config)
        self.keyboard = KeyboardController(config)
        self.grounding = create_grounding_service(config, use_mock=use_mock)
        self._instruction = resolve_grounding_instruction(config)
        self._cached_grounding: GroundingResult | None = None

    def run(self, posts: list[Post] | None = None) -> None:
        """Execute the full Notepad workflow for all posts."""
        if sys.platform != "win32":
            logger.warning("Non-Windows platform detected; pipeline may fail at runtime")

        self.capture.ensure_directories()
        output_dir = self.capture.project_output_dir()
        posts = posts or fetch_posts(self.config)

        logger.info("Starting pipeline for %d posts → %s", len(posts), output_dir)
        logger.info("Grounding instruction: %s", self._instruction)
        if self.config.grounding.cache_coordinates:
            logger.info("Coordinate caching enabled (ground once, verify thereafter)")

        for post in posts:
            self._process_post(post, output_dir)

        logger.info("Pipeline completed successfully")

    def _process_post(self, post: Post, output_dir: Path) -> None:
        """Process a single post with cached or fresh grounding."""
        logger.info("=== Processing post %d ===", post.id)

        result = self._open_notepad_via_grounding(post.id)
        self._write_and_save(post, output_dir)
        self._close_notepad()

        outfile = output_dir / f"post_{post.id}.txt"
        if not outfile.exists() or outfile.stat().st_size == 0:
            raise VerificationError(f"Output file missing or empty: {outfile}")

        logger.info("Post %d saved to %s", post.id, outfile)

    def _open_notepad_via_grounding(self, post_id: int) -> GroundingResult:
        """Show desktop, ground (or verify cache), click, and open Notepad."""

        def _attempt() -> GroundingResult:
            self.keyboard.show_desktop()
            time.sleep(0.3)
            screenshot = self.capture.capture_screenshot()
            result, self._cached_grounding = resolve_grounding_with_cache(
                self.grounding,
                self._instruction,
                screenshot,
                self.config,
                self._cached_grounding,
            )
            log_grounding_result(result)
            save_grounding_debug(result, self.config, f"post_{post_id}_grounding")

            screen_x, screen_y = to_screen_coords(result, self.capture, self.config)
            self.mouse.double_click(screen_x, screen_y)
            time.sleep(1.0)
            self.capture.wait_for_window(
                "Notepad",
                timeout_seconds=self.config.timeouts.notepad_launch_seconds,
            )
            return result

        def on_retry(attempt: int, exc: Exception) -> None:
            logger.warning("Open Notepad attempt %d failed: %s", attempt, exc)
            self._cached_grounding = None
            try:
                img = self.capture.capture_screenshot()
                save_failure_screenshot(
                    img,
                    self.config.paths.failures_dir,
                    f"open_notepad_retry_{post_id}_{attempt}",
                    logger,
                )
            except Exception:
                pass

        try:
            return retry_with_backoff(
                _attempt,
                max_attempts=self.config.retry.click_verify_max_attempts,
                backoff_base_seconds=self.config.retry.backoff_base_seconds,
                backoff_jitter_seconds=self.config.retry.backoff_jitter_seconds,
                retryable_exceptions=(
                    GroundingError,
                    LowConfidenceError,
                    VerificationError,
                ),
                on_retry=on_retry,
            )
        except (GroundingError, LowConfidenceError, VerificationError) as exc:
            self._cached_grounding = None
            try:
                img = self.capture.capture_screenshot()
                reason = getattr(exc, "reason", str(exc))
                save_failure_screenshot(
                    img,
                    self.config.paths.failures_dir,
                    f"post_{post_id}_{reason}",
                    logger,
                )
            except Exception:
                pass
            logger.error("Failed to open Notepad for post %d; exiting", post_id)
            raise SystemExit(1) from exc

    def _write_and_save(self, post: Post, output_dir: Path) -> None:
        """Type post content and save to project folder."""
        self.keyboard.type_text(post.formatted_content())
        time.sleep(0.3)
        outfile = output_dir / f"post_{post.id}.txt"
        self.keyboard.save_file_as(
            outfile,
            timeout_seconds=self.config.timeouts.save_dialog_seconds,
        )

    def _close_notepad(self) -> None:
        self.keyboard.close_active_window()
        time.sleep(0.5)
        if self.capture.is_window_open("Notepad"):
            logger.warning("Notepad still open after Alt+F4; retrying close")
            self.keyboard.close_active_window()
