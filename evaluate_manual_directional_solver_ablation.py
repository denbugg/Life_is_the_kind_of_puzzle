"""Manual score-calibration ablations for the dense directional solver cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import log_softmax
from skimage.metrics import structural_similarity

from global_solver_candidate import solve_layout

GRID, TILE, N = 24, 20, 576


def assemble(tiles: np.ndarray, layout: np.ndarray) -> np.ndarray:
    return tiles[layout].reshape(GRID, GRID, TILE, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


def adjacency(layout: np.ndarray, truth: np.ndarray) -> float:
    target_of = np.empty(N, np.int32)
    target_of[truth] = np.arange(N)
    board = target_of[layout].reshape(GRID, GRID)
    right = (board[:, 1:] == board[:, :-1] + 1) & (board[:, 1:] // GRID == board[:, :-1] // GRID)
    down = board[1:] == board[:-1] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def calibrate(matrix: np.ndarray, method: str) -> np.ndarray:
    if method == "raw":
        return matrix
    if method.startswith("edge_scale_"):
        return matrix * float(method.rsplit("_", 1)[1])
    if method.startswith("column_"):
        alpha = float(method.rsplit("_", 1)[1])
        return matrix + alpha * log_softmax(matrix, axis=0)
    if method.startswith("margin_column_"):
        alpha = float(method.rsplit("_", 1)[1])
        top2 = np.partition(matrix, -2, axis=1)[:, -2:]
        confidence = np.maximum(top2[:, 1] - top2[:, 0], 0.0)[:, None]
        confidence /= np.median(confidence) + 1e-6
        confidence = np.clip(confidence, 0.25, 4.0)
        return matrix * confidence + alpha * log_softmax(matrix, axis=0)
    raise ValueError(method)


def summarize(values: list[float]) -> dict:
    scores = np.asarray(values, np.float64)
    folds = np.asarray([scores[offset::4].mean() for offset in range(4)])
    return {"mean": float(scores.mean()), "robust": float(scores.mean() - 0.5 * folds.std()), "folds": folds.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    data = np.load(args.cache, mmap_mode="r")
    methods = ["raw", "edge_scale_0.8", "edge_scale_1.25", "column_0.10", "column_0.25", "column_0.50", "margin_column_0.10"]
    rows = {name: {"ssim": [], "adjacency": []} for name in methods}
    for index in range(min(args.limit, len(data["stems"]))):
        for method in methods:
            right = calibrate(np.asarray(data["right"][index]), method)
            down = calibrate(np.asarray(data["down"][index]), method)
            layout = np.asarray(solve_layout(right, down, data["pos"][index], 20260818 + index * 100), np.int32)
            rows[method]["ssim"].append(float(structural_similarity(
                data["target"][index], assemble(data["tiles"][index], layout), channel_axis=2, data_range=255
            )))
            rows[method]["adjacency"].append(adjacency(layout, data["truth"][index]))
        print(json.dumps({"done": index + 1, "total": min(args.limit, len(data["stems"])), "stem": str(data["stems"][index])}), flush=True)
    baseline = np.asarray(rows["raw"]["ssim"])
    report = {}
    for method in methods:
        scores = np.asarray(rows[method]["ssim"])
        report[method] = {
            "ssim": summarize(rows[method]["ssim"]),
            "mean_adjacency": float(np.mean(rows[method]["adjacency"])),
            "wins_vs_raw": int((scores > baseline).sum()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cases": min(args.limit, len(data["stems"])), "methods": report}, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
