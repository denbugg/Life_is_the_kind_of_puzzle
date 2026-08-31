#!/usr/bin/env python3
"""Independently validate/resume the official Union-v2 submission proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiijc_puzzle.union_v2_submission import DEFAULT_CONFIG, DEFAULT_VALIDATION_STATE
from aiijc_puzzle.union_v2_submission_validation import validate_union_v2_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submission-zip", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument(
        "--validation-state",
        type=Path,
        default=DEFAULT_VALIDATION_STATE,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--force-full-layout-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_union_v2_submission(
        source_dir=args.source_dir,
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        submission_zip=args.submission_zip,
        attestation_path=args.attestation,
        validation_state_path=args.validation_state,
        config_path=args.config,
        device_name=args.device,
        allow_nondeterministic_mps=args.allow_nondeterministic_mps,
        force_full_layout_recompute=args.force_full_layout_recompute,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
