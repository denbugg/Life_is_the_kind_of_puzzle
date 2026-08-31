#!/usr/bin/env python3
"""Build the separately versioned frozen DRUNet-protected 700-image bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiijc_puzzle.compliant_drunet_protected_submission import (
    dry_run_status,
    run_production_submission,
)

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
        "--run",
        action="store_true",
        help="execute canonical MPS production; omission performs a read-only dry run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run:
        print(json.dumps(dry_run_status(), indent=2, sort_keys=True))
        return
    report = run_production_submission(
        inputs_dir=args.inputs,
        source_archive=args.source_archive,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
