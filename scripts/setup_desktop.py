"""Desktop setup: create output folder and prepare icon positions."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from core.config import AppConfig

logger = logging.getLogger(__name__)

ICON_POSITIONS = {
    "top_left": {"grid_x": 0, "grid_y": 0},
    "center": {"grid_x": 4, "grid_y": 6},
    "bottom_right": {"grid_x": 8, "grid_y": 12},
}


def setup_desktop(config: AppConfig) -> None:
    """Create tjm-project folder and print icon arrangement instructions."""
    desktop = Path.home() / "Desktop"
    project_dir = desktop / config.paths.project_folder
    project_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created project folder: %s", project_dir)

    for rel in (
        config.paths.screenshots_dir,
        config.paths.failures_dir,
        config.paths.annotated_dir,
    ):
        Path(rel).mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        _ensure_notepad_shortcut(desktop)
        _print_arrangement_guide()
    else:
        logger.warning("Icon arrangement script requires Windows; skipping COM setup")
        _print_arrangement_guide()


def _ensure_notepad_shortcut(desktop: Path) -> None:
    """Create a Notepad shortcut on the desktop if missing."""
    shortcut = desktop / "Notepad.lnk"
    if shortcut.exists():
        logger.info("Notepad shortcut already exists: %s", shortcut)
        return

    ps_script = """
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\\Desktop\\Notepad.lnk")
$Shortcut.TargetPath = "notepad.exe"
$Shortcut.WorkingDirectory = "$env:USERPROFILE"
$Shortcut.Save()
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Created Notepad desktop shortcut")
    except subprocess.CalledProcessError as exc:
        logger.warning("Could not create shortcut via PowerShell: %s", exc.stderr)


def _print_arrangement_guide() -> None:
    """Print instructions for the three annotated screenshot positions."""
    logger.info("=== Icon arrangement for annotated screenshots ===")
    for name, pos in ICON_POSITIONS.items():
        logger.info(
            "  %s: move Notepad icon to grid column %d, row %d",
            name,
            pos["grid_x"],
            pos["grid_y"],
        )
    logger.info(
        "After arranging, run: uv run python main.py annotate --profile high"
    )
