#!/usr/bin/env python3
"""Independently recompute and verify the DRUNet-protected 700-image bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiijc_puzzle.compliant_drunet_protected_submission import (
    DEFAULT_ATTESTATION,
    DEFAULT_OUTPUT_ZIP,
)
from aiijc_puzzle.compliant_drunet_protected_validation import validate_submission
from aiijc_puzzle.compliant_submission import atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=PROJECT_ROOT / "data/raw/test")
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=PROJECT_ROOT / "data/raw/archives/test.zip",
    )
    parser.add_argument("--submission-zip", type=Path, default=DEFAULT_OUTPUT_ZIP)
    parser.add_argument("--attestation", type=Path, default=DEFAULT_ATTESTATION)
    parser.add_argument(
        "--report",
        type=Path,
        help="optional new JSON report path; existing files are never overwritten",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_submission(
        inputs_dir=args.inputs,
        source_archive=args.source_archive,
        submission_zip=args.submission_zip,
        attestation_path=args.attestation,
    )
    if args.report is not None:
        if args.report.exists() or args.report.is_symlink():
            raise FileExistsError(f"refusing to overwrite validation report: {args.report}")
        atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
