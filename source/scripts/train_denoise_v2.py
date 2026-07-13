#!/usr/bin/env python3
"""Train a tile-only denoiser on exact synthetic pairs."""

from __future__ import annotations

import argparse
import json

from puzzle_denoise_v2.training import TrainConfig, train


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["tile-naf", "full-naf"], default="tile-naf")
    parser.add_argument("--train-images", type=int, default=256)
    parser.add_argument("--val-images", type=int, default=16)
    parser.add_argument("--val-tiles-per-image", type=int, default=576)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--ssim-start-fraction", type=float, default=0.75)
    parser.add_argument("--variant-weights", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument("--libjpeg-val-images", type=int, default=4)
    parser.add_argument("--resume")
    parser.add_argument("--init-weights")
    parser.add_argument("--loss-ssim", type=float, default=0.10)
    parser.add_argument("--loss-gradient", type=float, default=0.05)
    parser.add_argument("--loss-boundary-extra", type=float, default=0.50)
    args = parser.parse_args()
    values = vars(args)
    values["variant_weights"] = tuple(values["variant_weights"])
    return TrainConfig(**values)


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps({"event": "train_complete", **result}, sort_keys=True), flush=True)
