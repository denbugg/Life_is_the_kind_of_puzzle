#!/usr/bin/env python3
"""Build or verify the deterministic source snapshot beside the submission ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiijc_puzzle.source_snapshot import (
    PROJECT_ROOT,
    build_source_snapshot,
    reproducibility_check,
    validate_source_snapshot,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/compliant-submission"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "source-snapshot.zip",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "source-snapshot-manifest.json",
    )
    parser.add_argument(
        "--checksum",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "source-snapshot.sha256",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing artifacts without writing them",
    )
    parser.add_argument(
        "--check-reproducible",
        action="store_true",
        help="also build twice in temporary storage and require byte identity",
    )
    parser.add_argument(
        "--without-workspace-comparison",
        action="store_true",
        help="verify self-integrity without requiring current sources to match",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        report = validate_source_snapshot(
            archive_path=args.archive,
            manifest_path=args.manifest,
            checksum_path=args.checksum,
            compare_with_workspace=not args.without_workspace_comparison,
        )
        report = {"status": "PASS", **report}
    else:
        report = build_source_snapshot(
            archive_path=args.archive,
            manifest_path=args.manifest,
            checksum_path=args.checksum,
        )
    if args.check_reproducible:
        report["reproducibility_check"] = reproducibility_check()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
