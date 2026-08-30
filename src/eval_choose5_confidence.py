"""Calibrate a board-adaptive chooser confidence floor on frozen validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from choose5 import K, seam_patch
from config import CACHE_DIR, CKPT_DIR
from train_choose5 import board_batch
from train_choose5_full import load_model, torch_packs
from train_verify_full import Cache


@torch.no_grad()
def board_records(model, tiles, packs, device):
    confidence, correct = [], []
    for axis in ("h", "v"):
        idx, val, lab = (value.to(device) for value in packs[axis])
        keep, src, dst, values, target = board_batch(
            tiles, idx, val, lab, model.strip, device)
        patch = seam_patch(tiles, src, dst, axis, model.strip).reshape(
            len(keep), K, 3, 20, 2 * model.strip)
        rank = torch.arange(K, device=device, dtype=torch.float32)
        relative = values - values[:, :1]
        scalars = torch.stack(
            [values / 10.0, relative, rank.expand(len(keep), K),
             (relative == 0).float()], -1)
        logits = model(patch, scalars)
        pick = logits.argmax(1)
        top2 = logits.topk(2, dim=1).values
        emitted = pick < K
        confidence.append((top2[:, 0] - top2[:, 1])[emitted].cpu().numpy())
        correct.append(((pick == target) & emitted)[emitted].cpu().numpy())
    return np.concatenate(confidence), np.concatenate(correct)


def fixed_metrics(records, volume):
    counts, hits = [], []
    for confidence, correct in records:
        order = np.argsort(-confidence)[:volume]
        counts.append(len(order)); hits.append(int(correct[order].sum()))
    return counts, hits


def floor_metrics(records, floor, cap):
    counts, hits = [], []
    for confidence, correct in records:
        order = np.argsort(-confidence)
        order = order[confidence[order] >= floor][:cap]
        counts.append(len(order)); hits.append(int(correct[order].sum()))
    return counts, hits


def summary(counts, hits):
    return {"mean_edges": float(np.mean(counts)),
            "edge_sd": float(np.std(counts)),
            "correct": float(np.mean(hits)),
            "precision": float(np.sum(hits) / max(np.sum(counts), 1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path,
                        default=Path(CACHE_DIR) / "verify_top5_v2")
    parser.add_argument("--checkpoint",
                        default="choose5_full_none0_best_bonds.pt")
    parser.add_argument("--held-start", type=int, default=6700)
    parser.add_argument("--boards", type=int, default=300)
    parser.add_argument("--calibration-boards", type=int, default=150)
    parser.add_argument("--volumes", default="350,390,430,470,510")
    parser.add_argument("--cap", type=int, default=576)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        checkpoint = Path(CKPT_DIR) / checkpoint
    model, _ = load_model(checkpoint, device)
    model.eval()
    cache = Cache(args.cache)
    stop = min(len(cache.names), args.held_start + args.boards)
    records = []
    for index in range(args.held_start, stop):
        tiles, packs = cache.board(index)
        tensor = torch.from_numpy(tiles.astype(np.float32)).to(device)
        records.append(board_records(model, tensor, torch_packs(packs), device))
        if len(records) % 25 == 0:
            print(f"boards {len(records)}/{stop-args.held_start}", flush=True)
    split = min(args.calibration_boards, len(records) - 1)
    calibration, evaluation = records[:split], records[split:]
    pooled = np.concatenate([confidence for confidence, _ in calibration])
    result = []
    for volume in (int(v) for v in args.volumes.split(",")):
        take = min(volume * len(calibration), len(pooled))
        floor = float(np.partition(pooled, len(pooled) - take)[len(pooled)-take])
        fixed = summary(*fixed_metrics(evaluation, volume))
        adaptive = summary(*floor_metrics(evaluation, floor, args.cap))
        result.append({"target": volume, "floor": floor,
                       "fixed": fixed, "adaptive": adaptive})
    print(json.dumps({"calibration_boards": len(calibration),
                      "evaluation_boards": len(evaluation),
                      "cap": args.cap, "results": result}, indent=2))


if __name__ == "__main__":
    main()
