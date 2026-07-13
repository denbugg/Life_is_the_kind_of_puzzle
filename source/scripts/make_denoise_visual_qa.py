#!/usr/bin/env python3
"""Create a leakage-safe corrupt/restored/clean calibration contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzle_denoise_v2.visual_qa import VisualQAConfig, run_visual_qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--val-pairs", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--quarantine-artifact",
        default="configs/denoise_validation_quarantine_v1.json",
    )
    parser.add_argument("--output", required=True, help="contact-sheet PNG")
    parser.add_argument("--report", help="JSON provenance; defaults beside --output")
    parser.add_argument("--expected-val-pairs-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-quarantine-sha256", required=True)
    parser.add_argument("--selection-seed", type=int, default=20260710)
    parser.add_argument("--pairs", type=int, default=12, help="one pair per distinct source")
    parser.add_argument("--tile-scale", type=int, default=6)
    parser.add_argument("--state", choices=["ema", "model"], default="ema")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser()
    report_path = (
        Path(args.report).expanduser()
        if args.report
        else output_path.with_suffix(".json")
    )
    report = run_visual_qa(
        VisualQAConfig(
            data_root=args.data_root,
            manifest=args.manifest,
            val_pairs=args.val_pairs,
            checkpoint=args.checkpoint,
            quarantine_artifact=args.quarantine_artifact,
            output_png=str(output_path),
            report_json=str(report_path),
            expected_val_pairs_sha256=args.expected_val_pairs_sha256,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_quarantine_sha256=args.expected_quarantine_sha256,
            selection_seed=args.selection_seed,
            pair_count=args.pairs,
            tile_scale=args.tile_scale,
            state=args.state,
            device=args.device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    )
    print(
        json.dumps(
            {
                "event": "denoise_visual_qa_complete",
                "output": report["outputs"]["contact_sheet_png"],
                "report": report["outputs"]["report_json"],
                "pairs": report["source_partition"]["selected_calibration_source_count"],
                "selection_sha256": report["selection"]["selection_sha256"],
                "contact_sheet_png_sha256": report["outputs"][
                    "contact_sheet_png_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
