"""Frozen end-to-end SSIM evaluator for an isolated global solver candidate."""
import os
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

from optimize import solve_layout

GRID, TILE, N = 24, 20, 576
CASE_FILE = Path(os.getenv(
    "CASE_FILE",
    "/home/kva/pazzle_directional_transformer/border_solver_ssim_cases.npz",
))


def assemble(tiles, layout):
    return tiles[np.asarray(layout)].reshape(
        GRID, GRID, TILE, TILE, 3
    ).transpose(0, 2, 1, 3, 4).reshape(GRID * TILE, GRID * TILE, 3)


def adjacency(layout, true_layout):
    target_of = np.empty(N, np.int32)
    target_of[true_layout] = np.arange(N)
    board = target_of[np.asarray(layout)].reshape(GRID, GRID)
    right = (
        (board[:, 1:] == board[:, :-1] + 1)
        & (board[:, 1:] // GRID == board[:, :-1] // GRID)
    )
    down = board[1:] == board[:-1] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def main():
    data = np.load(CASE_FILE)
    ssim_scores, adjacency_scores = [], []
    for index, (right, down, pos, restored, target, truth) in enumerate(zip(
        data["right"], data["down"], data["pos"], data["restored"],
        data["target"], data["truth"],
    )):
        layout = np.asarray(
            solve_layout(right, down, pos, 20260818 + index * 100), np.int64
        )
        if (
            layout.shape != (N,)
            or len(np.unique(layout)) != N
            or layout.min() != 0
            or layout.max() != N - 1
        ):
            raise ValueError("solve_layout must return a permutation of 0..575")
        image = assemble(restored, layout)
        ssim_scores.append(float(structural_similarity(
            target, image, channel_axis=2, data_range=255
        )))
        adjacency_scores.append(adjacency(layout, truth))
    ssim_scores = np.asarray(ssim_scores, np.float64)
    fold_means = np.asarray([
        ssim_scores[offset::4].mean() for offset in range(4)
    ])
    # Prefer improvements that generalize across four fixed grouped folds.  The
    # dispersion penalty makes a narrow win on a few images less attractive.
    robust_ssim = float(ssim_scores.mean() - 0.5 * fold_means.std())
    print(f"mean_adjacency: {np.mean(adjacency_scores):.9f}")
    print(f"mean_ssim: {ssim_scores.mean():.9f}")
    print("fold_ssim: " + ",".join(f"{value:.9f}" for value in fold_means))
    print(f"robust_ssim: {robust_ssim:.9f}")


if __name__ == "__main__":
    main()
