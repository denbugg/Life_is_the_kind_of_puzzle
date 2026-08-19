"""Test light per-tile denoising before dense directional scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch

from evaluate_directional_student_full576 import adjacency, assemble, ranks, score, split, summarize
from global_solver_candidate import solve_layout
from train_directional_jigsaw_transformer import DirectionalTransformer


def denoise_tiles(tiles: np.ndarray, h: int) -> np.ndarray:
    output = np.empty_like(tiles)
    for index, tile in enumerate(tiles):
        bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        filtered = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 3, 7)
        output[index] = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--raw-input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DirectionalTransformer().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data = np.load(args.cases)
    count = min(args.count, len(data["stems"]))
    names = ("raw", "tile_nlm_h3", "tile_nlm_h5")
    ssims = {name: [] for name in names}
    adjacencies = {name: [] for name in names}
    rank_right = {name: [] for name in names}
    rank_down = {name: [] for name in names}
    rows = []
    for index in range(count):
        stem = str(data["stems"][index])
        raw_image = np.asarray(Image.open(args.raw_input_dir / f"{stem}.png").convert("RGB"), np.uint8)
        raw_tiles = split(raw_image)
        scorer_tiles = {
            "raw": raw_tiles,
            "tile_nlm_h3": denoise_tiles(raw_tiles, 3),
            "tile_nlm_h5": denoise_tiles(raw_tiles, 5),
        }
        row = {"index": index, "stem": stem}
        for name, tiles in scorer_tiles.items():
            right, down = score(model, tiles, device, 0.10)
            rank_right[name].extend(ranks(right, data["truth"][index], 1))
            rank_down[name].extend(ranks(down, data["truth"][index], 24))
            layout = np.asarray(solve_layout(right, down, data["pos"][index], 20260818 + index * 100), np.int32)
            image = assemble(raw_tiles, layout)
            ssim = float(structural_similarity(data["target"][index], image, channel_axis=2, data_range=255))
            adj = adjacency(layout, data["truth"][index])
            ssims[name].append(ssim); adjacencies[name].append(adj)
            row[f"{name}_ssim"] = ssim; row[f"{name}_adjacency"] = adj
        rows.append(row)
        print(json.dumps({"done": index + 1, "total": count, **row}), flush=True)

    summary = {}
    raw_ssim = np.asarray(ssims["raw"])
    for name in names:
        right_rank = np.asarray(rank_right[name]); down_rank = np.asarray(rank_down[name])
        summary[name] = {
            "ssim": summarize(ssims[name]),
            "mean_adjacency": float(np.mean(adjacencies[name])),
            "ssim_wins_vs_raw": int((np.asarray(ssims[name]) > raw_ssim).sum()),
            "right": {"r1": float((right_rank <= 1).mean()), "r5": float((right_rank <= 5).mean()),
                      "r25": float((right_rank <= 25).mean()), "median": float(np.median(right_rank))},
            "down": {"r1": float((down_rank <= 1).mean()), "r5": float((down_rank <= 5).mean()),
                     "r25": float((down_rank <= 25).mean()), "median": float(np.median(down_rank))},
        }
    report = {"count": count, "summary": summary, "images": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"count": count, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
