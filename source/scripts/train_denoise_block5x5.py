#!/usr/bin/env python3
"""Fine-tune TileNAF with true contiguous 5x5 block supervision."""

from __future__ import annotations

import argparse
import json

from puzzle_denoise_v2.block5x5 import Block5x5TrainConfig, train_block5x5


def parse_args() -> Block5x5TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--protocol", default="configs/denoise_block5x5_v1.json")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=["moderate", "strong"], required=True)
    parser.add_argument("--train-images", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--block-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-ssim", type=float, required=True)
    parser.add_argument("--tile-gradient", type=float, required=True)
    parser.add_argument("--tile-boundary-extra", type=float, required=True)
    parser.add_argument("--block-ssim", type=float, required=True)
    parser.add_argument("--block-gradient", type=float, required=True)
    parser.add_argument("--seam-gradient", type=float, required=True)
    parser.add_argument("--neighbour-mean", type=float, required=True)
    return Block5x5TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    result = train_block5x5(parse_args())
    print(json.dumps({"event": "block5x5_train_complete", **result}, sort_keys=True), flush=True)
