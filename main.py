"""
QuaRUM – CLI entry point.

Usage:
  python main.py --input dataset/Iot.txt
  python main.py --input dataset/Iot.txt --verbose
  python main.py --input_dir dataset/
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from pipeline import QuaRUMPipeline


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join("output", "logs", "quarum.log"), mode="a"),
        ],
    )


def main() -> None:
    os.makedirs("output/logs", exist_ok=True)

    parser = argparse.ArgumentParser(
        description="QuaRUM: QDA-based UML domain model generation from requirements."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", type=str,
        help="Path to a single requirements document (txt, pdf, docx, md)."
    )
    group.add_argument(
        "--input_dir", type=str,
        help="Directory containing multiple requirements documents."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging."
    )
    parser.add_argument(
        "--resume-from", type=int, choices=[1, 2, 3, 4], default=1,
        dest="resume_from",
        help=(
            "Resume from phase N without re-running earlier phases. "
            "1=document processing (default, full run), "
            "2=knowledge construction (load Phase I from checkpoint), "
            "3=model generation (load Phases I-II from checkpoint), "
            "4=UML generation (load Phases I-III from checkpoint)."
        ),
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("quarum.main")

    pipeline = QuaRUMPipeline()

    if args.input:
        files = [args.input]
    else:
        patterns = ["*.txt", "*.pdf", "*.docx", "*.md"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(args.input_dir, pattern)))
        files = sorted(set(files))

    if not files:
        logger.error("No input files found.")
        sys.exit(1)

    logger.info("Processing %d file(s).", len(files))

    for file_path in files:
        try:
            bundle = pipeline.run(file_path, resume_from=args.resume_from)
            doc_name = os.path.splitext(os.path.basename(file_path))[0]
            logger.info(
                "Completed %s: %d entities, %d relationships",
                doc_name,
                len(bundle.model.entities),
                len(bundle.model.relationships),
            )
        except Exception as exc:
            logger.error("Failed to process %s: %s", file_path, exc, exc_info=True)


if __name__ == "__main__":
    main()
