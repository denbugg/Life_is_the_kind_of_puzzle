#!/usr/bin/env python3
"""Dry-run or execute the resumable SocketMatcher tile-sorter packager.

Omitting ``--run`` performs checkpoint/source validation and prints a no-write
plan.  The competition test directory is intentionally not a default: source,
checkpoint and output paths must all be chosen explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiijc_puzzle.socket_sorter_production import (
    PIXEL_TAILS,
    inspect_socket_sorter_run,
    run_socket_sorter_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default="cpu",
        help="explicit deterministic inference device; no implicit auto selection",
    )
    parser.add_argument(
        "--cyclic-border5",
        action="store_true",
        help="apply the fresh-exact-confirmed global cyclic border-weight-5 anchor",
    )
    parser.add_argument(
        "--pixel-tail",
        choices=tuple(PIXEL_TAILS),
        default="identity",
        help="separate registered post-layout hook; identity is the safe default",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="write/resume predictions; omission is a no-write validation plan",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run:
        result = run_socket_sorter_directory(
            checkpoint_path=args.checkpoint,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            cyclic_border5=args.cyclic_border5,
            pixel_tail_name=args.pixel_tail,
        )
    else:
        _, _, _, result = inspect_socket_sorter_run(
            checkpoint_path=args.checkpoint,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            cyclic_border5=args.cyclic_border5,
            pixel_tail_name=args.pixel_tail,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
