#!/usr/bin/env python3
"""Inspect or run/resume the exact official Union-v2+h20 submission bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiijc_puzzle.union_v2_submission import (
    DEFAULT_ATTESTATION,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_ZIP,
    DEFAULT_VALIDATION_STATE,
    inspect_union_v2_submission,
    run_union_v2_submission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-zip", type=Path, default=DEFAULT_OUTPUT_ZIP)
    parser.add_argument("--attestation", type=Path, default=DEFAULT_ATTESTATION)
    parser.add_argument(
        "--validation-state",
        type=Path,
        default=DEFAULT_VALIDATION_STATE,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="open official pixels and run/resume; omission is metadata-only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = {
        "source_dir": args.source_dir,
        "source_archive": args.source_archive,
        "output_dir": args.output_dir,
        "output_zip": args.output_zip,
        "attestation_path": args.attestation,
        "config_path": args.config,
        "device_name": args.device,
        "allow_nondeterministic_mps": args.allow_nondeterministic_mps,
    }
    if args.run:
        result = run_union_v2_submission(
            **common,
            validation_state_path=args.validation_state,
        )
    else:
        _, _, result = inspect_union_v2_submission(**common)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
