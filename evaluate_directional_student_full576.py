"""Evaluate the real-noisy directional student as a dense 576x576 scorer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import log_softmax
from skimage.metrics import structural_similarity
import torch

from global_solver_candidate import solve_layout
from train_directional_jigsaw_transformer import DirectionalTransformer, structural_channels

GRID, TILE, N = 24, 20, 576


def split(image: np.ndarray) -> np.ndarray:
    return image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3)


def assemble(tiles: np.ndarray, layout: np.ndarray) -> np.ndarray:
    return tiles[layout].reshape(GRID, GRID, TILE, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


def adjacency(layout: np.ndarray, truth: np.ndarray) -> float:
    target_of = np.empty(N, np.int32)
    target_of[truth] = np.arange(N)
    board = target_of[layout].reshape(GRID, GRID)
    right = (board[:, 1:] == board[:, :-1] + 1) & (board[:, 1:] // GRID == board[:, :-1] // GRID)
    down = board[1:] == board[:-1] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


@torch.inference_mode()
def score(model: DirectionalTransformer, tiles: np.ndarray, device: torch.device, tau: float) -> tuple[np.ndarray, np.ndarray]:
    x = torch.stack([structural_channels(tile.astype(np.float32) / 255.0) for tile in tiles])
    sides = []
    for start in range(0, N, 192):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            side, _ = model(x[start:start + 192].to(device))
        sides.append(side.float().cpu())
    side = torch.cat(sides)
    right = (side[:, 1] @ side[:, 0].T).numpy() / tau
    down = (side[:, 3] @ side[:, 2].T).numpy() / tau
    np.fill_diagonal(right, -1e4)
    np.fill_diagonal(down, -1e4)
    return log_softmax(right, axis=1).astype(np.float32), log_softmax(down, axis=1).astype(np.float32)


def ranks(matrix: np.ndarray, truth: np.ndarray, delta: int) -> list[int]:
    result = []
    for position in range(N):
        if delta == 1 and position % GRID == GRID - 1:
            continue
        if delta == GRID and position >= N - GRID:
            continue
        anchor = int(truth[position])
        neighbour = int(truth[position + delta])
        result.append(1 + int((matrix[anchor] > matrix[anchor, neighbour]).sum()))
    return result


def summarize(values: list[float]) -> dict:
    scores = np.asarray(values, np.float64)
    folds = np.asarray([scores[offset::4].mean() for offset in range(4)])
    return {
        "mean": float(scores.mean()),
        "robust": float(scores.mean() - 0.5 * folds.std()),
        "folds": folds.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--raw-input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.10)
    args = parser.parse_args()

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DirectionalTransformer().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data = np.load(args.cases)
    new_ssim, old_ssim, new_adj, old_adj = [], [], [], []
    right_ranks, down_ranks, rows = [], [], []
    for index, (old_right, old_down, pos, target, truth, stem) in enumerate(zip(
        data["right"], data["down"], data["pos"], data["target"], data["truth"], data["stems"]
    )):
        raw_image = np.asarray(Image.open(args.raw_input_dir / f"{stem}.png").convert("RGB"), np.uint8)
        tiles = split(raw_image)
        right, down = score(model, tiles, device, args.tau)
        right_ranks.extend(ranks(right, truth, 1))
        down_ranks.extend(ranks(down, truth, GRID))
        seed = 20260818 + index * 100
        old_layout = np.asarray(solve_layout(old_right, old_down, pos, seed), np.int32)
        new_layout = np.asarray(solve_layout(right, down, pos, seed), np.int32)
        old_score = float(structural_similarity(target, assemble(tiles, old_layout), channel_axis=2, data_range=255))
        new_score = float(structural_similarity(target, assemble(tiles, new_layout), channel_axis=2, data_range=255))
        old_a = adjacency(old_layout, truth)
        new_a = adjacency(new_layout, truth)
        old_ssim.append(old_score); new_ssim.append(new_score); old_adj.append(old_a); new_adj.append(new_a)
        row = {"index": index, "stem": str(stem), "old_ssim": old_score, "new_ssim": new_score,
               "old_adjacency": old_a, "new_adjacency": new_a}
        rows.append(row)
        print(json.dumps({"done": index + 1, "total": len(data["stems"]), **row}), flush=True)

    rank_metrics = {}
    for name, values in (("right", right_ranks), ("down", down_ranks)):
        rank = np.asarray(values)
        rank_metrics[name] = {"r1": float((rank <= 1).mean()), "r5": float((rank <= 5).mean()),
                              "r25": float((rank <= 25).mean()), "median": float(np.median(rank))}
    report = {
        "checkpoint_epoch": checkpoint.get("epoch"), "tau": args.tau,
        "rank": rank_metrics,
        "old_ssim": summarize(old_ssim), "new_ssim": summarize(new_ssim),
        "old_adjacency": float(np.mean(old_adj)), "new_adjacency": float(np.mean(new_adj)),
        "ssim_wins": int((np.asarray(new_ssim) > np.asarray(old_ssim)).sum()),
        "adjacency_wins": int((np.asarray(new_adj) > np.asarray(old_adj)).sum()),
        "images": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
