#!/usr/bin/env python3
"""Run the one-shot CPU audit of the pre-frozen selected denoiser."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json

from puzzle_denoise_v2.final_gate_audit import FinalGateAuditConfig, run_final_gate_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--val-pairs", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument(
        "--quarantine-artifact",
        default="configs/denoise_validation_quarantine_v1.json",
    )
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-val-pairs-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-legacy-checkpoint-sha256", required=True)
    parser.add_argument("--expected-quarantine-sha256", required=True)
    parser.add_argument("--expected-selection-manifest-sha256", required=True)
    parser.add_argument("--expected-code-sha256", required=True)
    parser.add_argument("--expected-opencv-version", default="4.11.0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_final_gate_audit(
        FinalGateAuditConfig(
            data_root=args.data_root,
            manifest=args.manifest,
            val_pairs=args.val_pairs,
            checkpoint=args.checkpoint,
            legacy_checkpoint=args.legacy_checkpoint,
            quarantine_artifact=args.quarantine_artifact,
            selection_manifest=args.selection_manifest,
            output=args.output,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_val_pairs_sha256=args.expected_val_pairs_sha256,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_legacy_checkpoint_sha256=args.expected_legacy_checkpoint_sha256,
            expected_quarantine_sha256=args.expected_quarantine_sha256,
            expected_selection_manifest_sha256=args.expected_selection_manifest_sha256,
            expected_code_sha256=args.expected_code_sha256,
            expected_opencv_version=args.expected_opencv_version,
            batch_size=args.batch_size,
            bootstrap_resamples=args.bootstrap_resamples,
            torch_threads=args.torch_threads,
        )
    )
    print(
        json.dumps(
            {
                "event": "selected_denoiser_final_gate_complete",
                "decision": report["assessment"]["decision"],
                "passes_final_gate": report["assessment"]["passes_final_gate"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
