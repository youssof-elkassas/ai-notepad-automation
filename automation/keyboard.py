"""Keyboard automation for typing and Save-As dialog."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pynput.keyboard import Controller, Key

from core.config import AppConfig
from core.exceptions import SaveError

logger = logging.getLogger(__name__)


class KeyboardController:
    """Type text and navigate Save dialogs via keyboard shortcuts."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._keyboard = Controller()
        self._interval = config.automation.typing_interval_seconds

    def type_text(self, text: str) -> None:
        """Type multiline text into the active window."""
        logger.debug("Typing %d characters", len(text))
        for char in text:
            self._keyboard.type(char)
            time.sleep(self._interval)

    def hotkey(self, *keys: Key | str) -> None:
        """Press a key combination."""
        parsed: list[Key | str] = []
        for k in keys:
            if isinstance(k, str):
                attr = getattr(Key, k.lower(), None)
                parsed.append(attr if attr else k)
            else:
                parsed.append(k)

        for k in parsed:
            self._keyboard.press(k)
        for k in reversed(parsed):
            self._keyboard.release(k)
        time.sleep(0.2)

    def save_file_as(self, filepath: Path, timeout_seconds: float = 10.0) -> None:
        """Save current document via keyboard: Ctrl+Shift+S then type path."""
        logger.info("Saving file as: %s", filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Ctrl+Shift+S opens Save As in Notepad (Windows 11)
        self.hotkey(Key.ctrl, Key.shift, "s")
        time.sleep(0.8)

        # Clear filename field and type full path
        self.hotkey(Key.ctrl, "a")
        time.sleep(0.1)
        self.type_text(str(filepath))
        time.sleep(0.2)
        self._keyboard.press(Key.enter)
        self._keyboard.release(Key.enter)
        time.sleep(0.5)

        # Handle overwrite confirmation if present
        self._keyboard.press(Key.enter)
        self._keyboard.release(Key.enter)
        time.sleep(0.3)

        if not filepath.exists():
            raise SaveError(f"File was not created: {filepath}")
        if filepath.stat().st_size == 0:
            raise SaveError(f"Saved file is empty: {filepath}")
        logger.info("File saved successfully: %s", filepath)

    def show_desktop(self) -> None:
        """Show Windows desktop (Win+D) before grounding or clicking icons."""
        logger.debug("Showing desktop (Win+D)")
        self.hotkey(Key.cmd, "d")
        time.sleep(0.5)

    def close_active_window(self) -> None:
        """Close active window with Alt+F4."""
        self.hotkey(Key.alt, Key.f4)
        time.sleep(0.5)
