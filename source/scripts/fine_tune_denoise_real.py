#!/usr/bin/env python3
"""Fine-tune a synthetic TileNAF checkpoint on strict real pairs with rollback gates."""

from __future__ import annotations

import argparse
import json

from puzzle_denoise_v2.real_training import FineTuneConfig, fine_tune


def parse_args() -> FineTuneConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--val-pairs", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument("--quarantine-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-train-pairs-sha256", required=True)
    parser.add_argument("--expected-val-pairs-sha256", required=True)
    parser.add_argument("--expected-init-checkpoint-sha256", required=True)
    parser.add_argument("--expected-legacy-checkpoint-sha256", required=True)
    parser.add_argument("--expected-quarantine-sha256", required=True)
    parser.add_argument("--expected-training-pixels-sha256", required=True)
    parser.add_argument("--expected-validation-pixels-sha256", required=True)
    parser.add_argument("--expected-opencv-version", required=True)
    parser.add_argument("--model", choices=["tile-naf"], default="tile-naf")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-real-source", type=int, default=32)
    parser.add_argument("--synthetic-train-images", type=int, default=512)
    parser.add_argument("--train-min-confidence", type=float, default=1.0)
    parser.add_argument("--val-sensitivity-confidence", type=float, default=1.0)
    parser.add_argument("--val-primary-confidence", type=float, default=1.5)
    parser.add_argument("--val-pairs-per-source", type=int, default=8)
    parser.add_argument("--peak-learning-rate", type=float, default=1e-5)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--early-real-period", type=int, default=8)
    parser.add_argument("--late-real-period", type=int, default=4)
    parser.add_argument("--schedule-switch-step", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--gate-source-count", type=int, required=True)
    parser.add_argument("--no-gain-patience", type=int, default=3)
    parser.add_argument("--no-gain-min-delta", type=float, default=1e-4)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto")
    return FineTuneConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    result = fine_tune(parse_args())
    print(json.dumps({"event": "real_fine_tune_complete", **result}, sort_keys=True), flush=True)
