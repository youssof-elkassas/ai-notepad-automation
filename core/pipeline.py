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
from core.logger import save_failure_screenshot
from core.retry import retry_with_backoff
from vision.gemini_grounding import create_grounding_service
from vision.grounding import GroundingResult
from vision.gui_parser import GuiParser

logger = logging.getLogger(__name__)


class NotepadPipeline:
    """Full automation pipeline with per-iteration re-grounding via Gemini."""

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
        self._parser = GuiParser(config.screen.width, config.screen.height)
        self.grounding = create_grounding_service(config, use_mock=use_mock)

    def run(self, posts: list[Post] | None = None) -> None:
        """Execute the full Notepad workflow for all posts."""
        if sys.platform != "win32":
            logger.warning("Non-Windows platform detected; pipeline may fail at runtime")

        self.capture.ensure_directories()
        output_dir = self.capture.project_output_dir()
        posts = posts or fetch_posts(self.config)

        logger.info("Starting pipeline for %d posts → %s", len(posts), output_dir)

        for post in posts:
            self._process_post(post, output_dir)

        logger.info("Pipeline completed successfully")

    def _process_post(self, post: Post, output_dir: Path) -> None:
        """Process a single post with fresh screenshot and grounding."""
        logger.info("=== Processing post %d ===", post.id)

        result = self._ground_notepad_icon()
        screen_x, screen_y = self._to_screen_coords(result)

        self._launch_notepad(screen_x, screen_y)
        self._write_and_save(post, output_dir)
        self._close_notepad()

        outfile = output_dir / f"post_{post.id}.txt"
        if not outfile.exists() or outfile.stat().st_size == 0:
            raise VerificationError(f"Output file missing or empty: {outfile}")

        logger.info("Post %d saved to %s", post.id, outfile)

    def _ground_notepad_icon(self) -> GroundingResult:
        """Observe + Reason: fresh screenshot and Gemini grounding."""

        def _attempt() -> GroundingResult:
            screenshot = self.capture.capture_screenshot()
            instruction = self.config.grounding.icon_instruction
            logger.info("Grounding instruction: %s", instruction)
            return self.grounding.locate(instruction, screenshot)

        def on_retry(attempt: int, exc: Exception) -> None:
            logger.warning("Grounding attempt %d failed: %s", attempt, exc)
            try:
                img = self.capture.capture_screenshot()
                save_failure_screenshot(
                    img,
                    self.config.paths.failures_dir,
                    f"grounding_retry_{attempt}",
                    logger,
                )
            except Exception:
                pass

        try:
            return retry_with_backoff(
                _attempt,
                max_attempts=self.config.retry.grounding_max_attempts,
                backoff_base_seconds=self.config.retry.backoff_base_seconds,
                backoff_jitter_seconds=self.config.retry.backoff_jitter_seconds,
                retryable_exceptions=(GroundingError, LowConfidenceError),
                on_retry=on_retry,
            )
        except (GroundingError, LowConfidenceError) as exc:
            try:
                img = self.capture.capture_screenshot()
                reason = getattr(exc, "reason", str(exc))
                save_failure_screenshot(
                    img,
                    self.config.paths.failures_dir,
                    reason,
                    logger,
                )
            except Exception:
                pass
            logger.error("Grounding failed after retries; exiting gracefully")
            raise SystemExit(1) from exc

    def _to_screen_coords(self, result: GroundingResult) -> tuple[int, int]:
        dpi = self._parser.get_dpi_scale()
        return self._parser.normalized_to_screen(
            result.center,
            monitor_offset=self.capture.monitor_offset,
            dpi_scale=dpi,
        )

    def _launch_notepad(self, screen_x: int, screen_y: int) -> None:
        """Act: double-click grounded icon and verify Notepad opens."""

        def _click_and_verify() -> None:
            self.mouse.double_click(screen_x, screen_y)
            time.sleep(1.0)
            self.capture.wait_for_window(
                "Notepad",
                timeout_seconds=self.config.timeouts.notepad_launch_seconds,
            )

        retry_with_backoff(
            _click_and_verify,
            max_attempts=self.config.retry.click_verify_max_attempts,
            retryable_exceptions=(VerificationError,),
        )

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
