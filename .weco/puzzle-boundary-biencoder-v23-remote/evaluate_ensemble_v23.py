"""Calibrate and evaluate a V23 small/XL/handcrafted boundary ensemble."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/kva/pazzle_boundary_biencoder_v23_xl")
import train_boundary_biencoder_v23 as v23


SMALL = Path("/home/kva/pazzle_boundary_biencoder_v23/outputs/boundary_biencoder_best.pt")
XL = Path("/home/kva/pazzle_boundary_biencoder_v23_xl/outputs/boundary_biencoder_best.pt")
OUT = Path("/home/kva/pazzle_boundary_biencoder_v23_ensemble")
CALIBRATION = range(6736, 6756)
HOLDOUT = range(6957, 6973)


def load_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = v23.ModelConfig(**state["model_config"])
    model = v23.BoundaryBiEncoder(config)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), state


def row_z(matrix):
    result = matrix.copy()
    np.fill_diagonal(result, np.nan)
    mean = np.nanmean(result, axis=1, keepdims=True)
    std = np.nanstd(result, axis=1, keepdims=True) + 1e-6
    result = (result - mean) / std
    np.fill_diagonal(result, -1e4)
    return result


@torch.inference_mode()
def learned_scores(model, tiles):
    e = model(tiles)
    return [row_z((e["right"] @ e["left"].t()).float().cpu().numpy()),
            row_z((e["bottom"] @ e["top"].t()).float().cpu().numpy())]


@torch.inference_mode()
def handcrafted_scores(model, tiles):
    output = []
    for source_side, target_side in (("right", "left"), ("bottom", "top")):
        source = model.side_features(tiles, source_side).flatten(1)
        target = model.side_features(tiles, target_side).flatten(1)
        source = F.normalize(source - source.mean(1, keepdim=True), dim=1)
        target = F.normalize(target - target.mean(1, keepdim=True), dim=1)
        output.append(row_z((source @ target.t()).float().cpu().numpy()))
    return output


def metrics(matrices):
    values = [v23.retrieval(matrix.copy(), v23.GRID, direction)
              for matrix, direction in zip(matrices, ("right", "down"))]
    return {key: float(np.mean([item[key] for item in values])) for key in values[0]}


def objective(value):
    return (0.50 * value["top32"] + 0.20 * value["top5"]
            + 0.20 * value["top1"] + 0.10 * value["mrr"])


def aggregate(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def union_recall(first, second, k_each, direction):
    side = v23.GRID
    grid = np.arange(side * side).reshape(side, side)
    if direction == "right":
        sources = grid[:, :-1].reshape(-1); targets = grid[:, 1:].reshape(-1)
    else:
        sources = grid[:-1].reshape(-1); targets = grid[1:].reshape(-1)
    a = np.argpartition(-first[sources], k_each - 1, axis=1)[:, :k_each]
    b = np.argpartition(-second[sources], k_each - 1, axis=1)[:, :k_each]
    found = ((a == targets[:, None]).any(1) | (b == targets[:, None]).any(1))
    return float(found.mean())


@torch.inference_mode()
def score_scene(path, small, xl, device):
    tiles = v23.load_board(path).reshape(v23.GRID ** 2, 3, v23.TILE, v23.TILE).to(device)
    return learned_scores(small, tiles), learned_scores(xl, tiles), handcrafted_scores(small, tiles)


def blend(scores, weights):
    return [weights[0] * scores[0][d] + weights[1] * scores[1][d] + weights[2] * scores[2][d]
            for d in range(2)]


def main():
    device = torch.device("cuda")
    small, small_state = load_model(SMALL, device)
    xl, xl_state = load_model(XL, device)
    data = Path("/home/kva/pazzle_directional_transformer/data/real/restored_target_order")
    calibration_cache = []
    for scene in CALIBRATION:
        calibration_cache.append(score_scene(data / f"img_{scene:06d}.png", small, xl, device))
    trials = []
    for small_weight, seam_weight in itertools.product(np.linspace(0, 1, 9), (0.0, 0.1, 0.2, 0.35, 0.5)):
        weights = (float(small_weight), float(1.0 - small_weight), float(seam_weight))
        value = aggregate([metrics(blend(scores, weights)) for scores in calibration_cache])
        trials.append({"weights": weights, **value, "objective": objective(value)})
    selected = max(trials, key=lambda item: item["objective"])
    holdout_rows = []; small_rows = []; xl_rows = []; seam_rows = []
    union32 = []; union64 = []
    for scene in HOLDOUT:
        scores = score_scene(data / f"img_{scene:06d}.png", small, xl, device)
        small_rows.append(metrics(scores[0])); xl_rows.append(metrics(scores[1])); seam_rows.append(metrics(scores[2]))
        holdout_rows.append(metrics(blend(scores, selected["weights"])))
        union32.append(np.mean([union_recall(scores[0][d], scores[1][d], 16, direction)
                                for d, direction in enumerate(("right", "down"))]))
        union64.append(np.mean([union_recall(scores[0][d], scores[1][d], 32, direction)
                                for d, direction in enumerate(("right", "down"))]))
    report = {
        "schema": "puzzle-boundary-biencoder-v23-ensemble",
        "small_step": small_state["step"], "xl_step": xl_state["step"],
        "calibration_scenes": [min(CALIBRATION), max(CALIBRATION)],
        "holdout_scenes": [min(HOLDOUT), max(HOLDOUT)],
        "selected": selected,
        "holdout": {
            "small": aggregate(small_rows), "xl": aggregate(xl_rows),
            "handcrafted": aggregate(seam_rows), "blend": aggregate(holdout_rows),
            "union_recall_at_32_budget": float(np.mean(union32)),
            "union_recall_at_64_budget": float(np.mean(union64))},
        "trials": sorted(trials, key=lambda item: item["objective"], reverse=True)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
