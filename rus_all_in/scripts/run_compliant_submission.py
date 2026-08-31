#!/usr/bin/env python3
"""Build the strict 700-image compliant submission and its attestation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiijc_puzzle.compliant_submission import run_production_submission

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=PROJECT_ROOT / "data/raw/test")
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=PROJECT_ROOT / "data/raw/archives/test.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/compliant-submission/predictions",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=PROJECT_ROOT / "outputs/compliant-submission/submission.zip",
    )
    parser.add_argument(
        "--attestation",
        type=Path,
        default=PROJECT_ROOT / "outputs/compliant-submission/compliance-attestation.json",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute the expensive 700-board job; omission is a no-write dry run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configuration = {
        "inputs": str(args.inputs.resolve()),
        "source_archive": str(args.source_archive.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "output_zip": str(args.output_zip.resolve()),
        "attestation": str(args.attestation.resolve()),
        "schema": str((PROJECT_ROOT / "configs/submission-compliance.schema.json").resolve()),
        "schema_override_allowed": False,
        "layout": "no-atlas bilateral buddies96",
        "tail": "RGB seam offsets -> bounded luma gains -> colored NLM h20/hColor20 x1",
        "run": args.run,
    }
    if not args.run:
        print(json.dumps({"status": "DRY_RUN_NO_WRITES", **configuration}, indent=2))
        return
    report = run_production_submission(
        inputs_dir=args.inputs,
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        output_zip=args.output_zip,
        attestation_path=args.attestation,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
