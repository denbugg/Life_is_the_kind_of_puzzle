#!/usr/bin/env python3
"""Fail-closed builder for the separately versioned fixed-B submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiijc_puzzle.compliant_fixed_b_standard_submission import (
    dry_run_status,
    freeze_production_runtime_preflight,
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
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--run",
        action="store_true",
        help="run only after the absent-by-default promotion config is authorized",
    )
    actions.add_argument(
        "--freeze-runtime-preflight",
        action="store_true",
        help="write the immutable current runtime manifest before root authorization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze_runtime_preflight:
        print(json.dumps(freeze_production_runtime_preflight(), indent=2, sort_keys=True))
        return
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
