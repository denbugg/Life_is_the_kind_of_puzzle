"""Frozen smoke evaluator for block-preserving solver experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

from solver_candidate import solve_layout

GRID, TILE, N = 24, 20, 576


def assemble(tiles, layout):
    return tiles[layout].reshape(GRID, GRID, TILE, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


def adjacency(layout, truth):
    target_of = np.empty(N, np.int32)
    target_of[truth] = np.arange(N)
    board = target_of[layout].reshape(GRID, GRID)
    right = (board[:, 1:] == board[:, :-1] + 1) & (board[:, 1:] // GRID == board[:, :-1] // GRID)
    down = board[1:] == board[:-1] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def summarize(values):
    values = np.asarray(values, np.float64)
    folds = np.asarray([values[o::4].mean() for o in range(4)])
    return {"mean": float(values.mean()), "robust": float(values.mean() - 0.5 * folds.std()), "folds": folds.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=32)
    args = p.parse_args()
    data = np.load(args.cache, mmap_mode="r")
    methods = [item for item in os.getenv(
        "SOLVER_METHODS", "baseline,block2,segment4,mixed"
    ).split(",") if item]
    rows = {m: {"ssim": [], "adj": []} for m in methods}
    for index in range(min(args.limit, len(data["stems"]))):
        for method in methods:
            os.environ["BLOCK_MODE"] = method
            layout = np.asarray(solve_layout(data["right"][index], data["down"][index], data["pos"][index], 20260818 + index * 100), np.int32)
            if layout.shape != (N,) or len(np.unique(layout)) != N:
                raise ValueError(f"invalid permutation from {method}")
            rows[method]["ssim"].append(float(structural_similarity(data["target"][index], assemble(data["tiles"][index], layout), channel_axis=2, data_range=255)))
            rows[method]["adj"].append(adjacency(layout, data["truth"][index]))
        print(json.dumps({"done": index + 1, "total": args.limit, "stem": str(data["stems"][index])}), flush=True)
    baseline = np.asarray(rows["baseline"]["ssim"])
    report = {}
    for method in methods:
        scores = np.asarray(rows[method]["ssim"])
        report[method] = {"ssim": summarize(scores), "mean_adjacency": float(np.mean(rows[method]["adj"])), "wins_vs_baseline": int((scores > baseline).sum())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cases": args.limit, "methods": report}, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
