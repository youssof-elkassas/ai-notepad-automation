#!/usr/bin/env python3
"""CLI entry point for Gemini-based Notepad GUI automation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from core.config import load_config
from core.env import load_project_dotenv
from core.logger import setup_logger
from core.pipeline import NotepadPipeline
from vision.gemini_grounding import create_grounding_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemini-based Windows GUI grounding automation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run full Notepad automation pipeline")
    run_parser.add_argument(
        "--profile",
        choices=["high", "low"],
        default="high",
        help="Gemini model profile (default: high)",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock grounding (no API key, for structure testing)",
    )

    annotate_parser = sub.add_parser(
        "annotate",
        help="Generate annotated screenshots for icon positions",
    )
    annotate_parser.add_argument(
        "--profile",
        choices=["high", "low"],
        default="high",
    )
    annotate_parser.add_argument("--mock", action="store_true")

    demo_parser = sub.add_parser("demo", help="Run grounding once and print bbox")
    demo_parser.add_argument(
        "--profile",
        choices=["high", "low"],
        default="high",
    )
    demo_parser.add_argument(
        "--instruction",
        default="Notepad desktop icon",
    )
    demo_parser.add_argument("--mock", action="store_true")

    sub.add_parser("setup", help="Prepare desktop and output folder")

    return parser


def main(argv: list[str] | None = None) -> int:
    load_project_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    profile = getattr(args, "profile", "high")
    config = load_config(profile=profile)

    logger = setup_logger(level=config.logging.level, log_file=config.logging.log_file)
    logger.info("Profile: %s | Command: %s", config.profile, args.command)

    if args.command == "run":
        pipeline = NotepadPipeline(config, use_mock=getattr(args, "mock", False))
        pipeline.run()
        return 0

    if args.command == "annotate":
        from scripts.generate_annotated_screenshots import generate_annotated

        generate_annotated(config, use_mock=args.mock)
        return 0

    if args.command == "demo":
        from automation.windows import WindowsCapture

        capture = WindowsCapture(config)
        screenshot = capture.capture_screenshot()
        service = create_grounding_service(config, use_mock=args.mock)
        result = service.locate(args.instruction, screenshot)

        logger.info("Bbox: %s", result.bbox.as_tuple())
        logger.info("Center: %s", result.center)
        logger.info("Confidence: %.2f", result.confidence)
        logger.info("Trace: %s", result.search_trace)

        out_dir = Path(config.paths.annotated_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if result.annotated_image:
            out_path = out_dir / "demo_grounding.png"
            result.annotated_image.save(out_path)
            logger.info("Annotated image saved: %s", out_path)
        return 0

    if args.command == "setup":
        from scripts.setup_desktop import setup_desktop

        setup_desktop(config)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
