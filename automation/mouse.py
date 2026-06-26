"""Mouse automation — always uses freshly computed coordinates."""

from __future__ import annotations

import logging
import time

from pynput.mouse import Button, Controller

from core.config import AppConfig

logger = logging.getLogger(__name__)


class MouseController:
    """Double-click and move using absolute screen coordinates."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._mouse = Controller()

    def move_to(self, x: int, y: int) -> None:
        logger.debug("Moving mouse to (%d, %d)", x, y)
        self._mouse.position = (x, y)
        time.sleep(0.1)

    def double_click(self, x: int, y: int) -> None:
        """Double-click at absolute screen coordinates — never cached."""
        self.move_to(x, y)
        logger.info("Double-clicking at (%d, %d)", x, y)
        self._mouse.click(Button.left, 2)
        time.sleep(0.3)

    def single_click(self, x: int, y: int) -> None:
        self.move_to(x, y)
        self._mouse.click(Button.left, 1)
