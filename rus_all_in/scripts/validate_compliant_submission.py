#!/usr/bin/env python3
"""Independently verify a compliant submission ZIP and all board evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiijc_puzzle.compliant_submission import validate_submission

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
        "--submission-zip",
        type=Path,
        default=PROJECT_ROOT / "outputs/compliant-submission/submission.zip",
    )
    parser.add_argument(
        "--attestation",
        type=Path,
        default=PROJECT_ROOT / "outputs/compliant-submission/compliance-attestation.json",
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
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
