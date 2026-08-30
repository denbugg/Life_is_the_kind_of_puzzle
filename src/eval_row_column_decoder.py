"""Evaluate the discrete row/column decoder on held-out full boards."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import CACHE_DIR, CKPT_DIR, GRID, TRAIN_INP
from discrete_field_decoder import solve_discrete
from infer_coarse_field import load_rgb
from restore_tile import to_frags
from row_column_decoder import solve_rows_then_columns
from seam_cost import costs_from_models
from seam_embed import SeamEmbed


def load_matchers(names, device="cuda"):
    models = []
    for name in names:
        ck = torch.load(Path(CKPT_DIR) / name, map_location=device,
                        weights_only=False)
        args = ck["args"]
        model = SeamEmbed(args["ch"], args["blocks"], args["dim"],
                          args["strip"], args.get("head", "global"),
                          predict=any(k.startswith("pred.")
                                      for k in ck["model"])).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        models.append(model)
    return models


def metrics(layout):
    board = np.asarray(layout).reshape(GRID, GRID)
    placement = float(np.mean(board.reshape(-1) == np.arange(GRID * GRID)))
    horizontal = float(np.mean(board[:, 1:] == board[:, :-1] + 1))
    vertical = float(np.mean(board[1:] == board[:-1] + GRID))
    return placement, horizontal, vertical, 0.5 * (horizontal + vertical)


def objective(layout, right, down):
    board = np.asarray(layout).reshape(GRID, GRID)
    return float(right[board[:, :-1], board[:, 1:]].sum()
                 + down[board[:-1], board[1:]].sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matcher", nargs="+",
                        default=["seam_embed_v3.pt", "seam_embed_local.pt"])
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--trim", type=float, default=0.2)
    parser.add_argument("--mutual-weight", type=float, default=0.15)
    parser.add_argument("--qap-iters", type=int, default=0)
    args = parser.parse_args()

    models = load_matchers(args.matcher)
    labels = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names = [str(x) for x in labels["names"][-300:]][:args.n]
    inv = labels["inv"][-300:][:args.n]
    rows = []
    for k, name in enumerate(names):
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / name)).astype(np.float32)
        tiles = tiles[inv[k].astype(np.int64)]
        cost_h, cost_v = costs_from_models(models, tiles)
        right, down = -cost_h, -cost_v
        candidates = []
        for first in ("rows", "columns"):
            layout = solve_rows_then_columns(
                right, down, GRID, first, args.trim, args.mutual_weight)
            candidates.append((objective(layout, right, down), first, layout))
        selected = max(candidates, key=lambda x: x[0])
        current = []
        for value, first, layout in candidates:
            current.extend(metrics(layout))
        current.extend(metrics(selected[2]))
        if args.qap_iters:
            zero = np.zeros((GRID * GRID, GRID * GRID), np.float64)
            polished, _ = solve_discrete(
                right, down, zero, selected[2], GRID, unary_weight=0.0,
                iters=args.qap_iters, restarts=1, seed=1234 + k, sweeps=1)
            current.extend(metrics(polished))
        rows.append(current)
        print(k, name, "pick", selected[1],
              "rows", np.round(current[:4], 4),
              "columns", np.round(current[4:8], 4), flush=True)
    mean = np.mean(rows, axis=0)
    report = {
        "n": len(rows),
        "rows": dict(zip(("placement", "horizontal", "vertical", "adjacency"),
                         np.round(mean[:4], 5))),
        "columns": dict(zip(("placement", "horizontal", "vertical", "adjacency"),
                            np.round(mean[4:8], 5))),
        "objective_pick": dict(zip(
            ("placement", "horizontal", "vertical", "adjacency"),
            np.round(mean[8:12], 5))),
    }
    if args.qap_iters:
        report["qap_polished"] = dict(zip(
            ("placement", "horizontal", "vertical", "adjacency"),
            np.round(mean[12:16], 5)))
    print(report)


if __name__ == "__main__":
    main()
