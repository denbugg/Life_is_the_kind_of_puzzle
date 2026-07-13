#!/usr/bin/env python3
"""Run the frozen CPU-only pre-fine-tune calibration benchmark."""

from __future__ import annotations

import argparse
import json

from puzzle_denoise_v2.prefinetune_benchmark import (
    PreFineTuneBenchmarkConfig,
    run_prefinetune_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--val-pairs", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument("--quarantine-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pairs-sha256", required=True)
    parser.add_argument("--expected-init-checkpoint-sha256", required=True)
    parser.add_argument("--expected-legacy-checkpoint-sha256", required=True)
    parser.add_argument("--expected-quarantine-sha256", required=True)
    parser.add_argument("--expected-validation-pixels-sha256", required=True)
    parser.add_argument("--expected-code-sha256", required=True)
    parser.add_argument("--expected-opencv-version", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-legacy-ssim-deficit", type=float, default=0.01)
    parser.add_argument("--gate-source-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_prefinetune_benchmark(
        PreFineTuneBenchmarkConfig(
            data_root=args.data_root,
            manifest=args.manifest,
            val_pairs=args.val_pairs,
            init_checkpoint=args.init_checkpoint,
            legacy_checkpoint=args.legacy_checkpoint,
            quarantine_artifact=args.quarantine_artifact,
            output=args.output,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_val_pairs_sha256=args.expected_val_pairs_sha256,
            expected_init_checkpoint_sha256=args.expected_init_checkpoint_sha256,
            expected_legacy_checkpoint_sha256=args.expected_legacy_checkpoint_sha256,
            expected_quarantine_sha256=args.expected_quarantine_sha256,
            expected_validation_pixels_sha256=args.expected_validation_pixels_sha256,
            expected_code_sha256=args.expected_code_sha256,
            expected_opencv_version=args.expected_opencv_version,
            batch_size=args.batch_size,
            bootstrap_resamples=args.bootstrap_resamples,
            torch_threads=args.torch_threads,
            max_legacy_ssim_deficit=args.max_legacy_ssim_deficit,
            gate_source_count=args.gate_source_count,
        )
    )
    print(
        json.dumps(
            {
                "event": "prefinetune_calibration_complete",
                "output": args.output,
                "diagnostic": report["diagnostic"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
