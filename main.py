#!/usr/bin/env python3
"""CLI entry point for ScreenSeekeR Notepad automation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from core.config import load_config
from core.logger import setup_logger
from core.pipeline import NotepadPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ScreenSeekeR-based Windows GUI grounding automation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run full Notepad automation pipeline")
    run_parser.add_argument(
        "--profile",
        choices=["high", "low"],
        default="high",
        help="Hardware profile (default: high)",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock models (no GPU, for structure testing)",
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
    annotate_parser.add_argument(
        "--mock",
        action="store_true",
    )

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

    setup_parser = sub.add_parser("setup", help="Prepare desktop and output folder")
    setup_parser.add_argument(
        "--profile",
        choices=["high", "low"],
        default="high",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(profile=args.profile)

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
        from vision.grounding import GroundingService, OSAtlasGrounder, MockGrounder
        from vision.planner import MockPlanner, QwenPlanner
        from vision.screenseeker import ScreenSeekeR

        capture = WindowsCapture(config)
        screenshot = capture.capture_screenshot()

        if args.mock:
            grounder = MockGrounder()
            planner = MockPlanner()
        else:
            grounder = OSAtlasGrounder(config)
            planner = QwenPlanner(config)

        seeker = ScreenSeekeR(config, grounder, planner)
        service = GroundingService(grounder, config, screenseeker=seeker)
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
