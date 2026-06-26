"""Windows screenshot capture, window management, and path helpers."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from PIL import Image

from core.config import AppConfig
from core.exceptions import PlatformError, ScreenshotError, WindowNotFoundError

logger = logging.getLogger(__name__)


def require_windows() -> None:
    if sys.platform != "win32":
        raise PlatformError(
            "This automation targets Windows 10/11. Run on a Windows machine."
        )


class WindowsCapture:
    """Capture desktop screenshots and verify window state."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._monitor_offset = (0, 0)

    def capture_screenshot(self) -> Image.Image:
        """Capture primary monitor screenshot."""
        require_windows()
        try:
            import mss
            import mss.tools
        except ImportError as exc:
            raise ScreenshotError("mss not installed") from exc

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            self._monitor_offset = (monitor["left"], monitor["top"])
            shot = sct.grab(monitor)
            image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            logger.debug(
                "Captured screenshot %dx%d offset=%s",
                image.width,
                image.height,
                self._monitor_offset,
            )
            if (
                image.width != self.config.screen.width
                or image.height != self.config.screen.height
            ):
                logger.warning(
                    "Screenshot resolution %s != expected %dx%d",
                    image.size,
                    self.config.screen.width,
                    self.config.screen.height,
                )
            return image

    @property
    def monitor_offset(self) -> tuple[int, int]:
        return self._monitor_offset

    def wait_for_window(
        self,
        title_substring: str,
        timeout_seconds: float | None = None,
    ) -> bool:
        """Wait until a window whose title contains *title_substring* appears."""
        require_windows()
        timeout = timeout_seconds or self.config.timeouts.window_wait_seconds
        deadline = time.time() + timeout

        while time.time() < deadline:
            if self._find_window(title_substring):
                logger.info("Window found: %s", title_substring)
                return True
            time.sleep(0.5)

        raise WindowNotFoundError(
            f"Window '{title_substring}' not found within {timeout}s"
        )

    def _find_window(self, title_substring: str) -> bool:
        try:
            from pywinauto import Desktop

            windows = Desktop(backend="uia").windows()
            for w in windows:
                title = w.window_text() or ""
                if title_substring.lower() in title.lower():
                    return True
        except Exception as exc:
            logger.debug("Window enumeration failed: %s", exc)
        return False

    def is_window_open(self, title_substring: str) -> bool:
        return self._find_window(title_substring)

    def close_window(self, title_substring: str) -> None:
        """Close window matching title via Alt+F4 after focusing."""
        require_windows()
        try:
            from pywinauto import Desktop

            for w in Desktop(backend="uia").windows():
                title = w.window_text() or ""
                if title_substring.lower() in title.lower():
                    w.set_focus()
                    break
        except Exception as exc:
            logger.warning("Could not focus window: %s", exc)

    @staticmethod
    def desktop_path() -> Path:
        return Path.home() / "Desktop"

    def project_output_dir(self) -> Path:
        path = self.desktop_path() / self.config.paths.project_folder
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_directories(self) -> None:
        """Create runtime directories."""
        for rel in (
            self.config.paths.screenshots_dir,
            self.config.paths.failures_dir,
            self.config.paths.annotated_dir,
        ):
            Path(rel).mkdir(parents=True, exist_ok=True)
