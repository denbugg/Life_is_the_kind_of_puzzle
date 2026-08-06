"""Measure how much position signal an ideal coarse canvas would provide.

This is deliberately a lower-level diagnostic, not a claimed solver: it supplies
the target's clean low-frequency canvas to an analytic tile/cell matcher.  If this
ceiling is weak, training a CanvasNet at that resolution cannot help.
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from canvas_data import CanvasDataset
from canvas_metrics import decoded_geometry, rank_summary
from config import FS
from imgio import train_val_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=4, help="clean descriptor side per 20px tile")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if FS % args.patch:
        ap.error("--patch must divide 20")

    _, val = train_val_split()
    ds = CanvasDataset(val[:args.n], patch=args.patch, real_prob=0.0)
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers)
    rows = []
    for batch in dl:
        tiles = batch["tiles"]
        b, n, c, h, w = tiles.shape
        # Exact area pooling makes the corrupted tile descriptor live in the
        # same 4x4 (or requested) coordinate system as the clean canvas cell.
        td = F.avg_pool2d(tiles.reshape(b * n, c, h, w), FS // args.patch)
        td = td.reshape(b, n, -1)
        cd = batch["target_patches"].permute(0, 1, 4, 2, 3).reshape(b, n, -1)
        td = F.normalize(td - td.mean(-1, keepdim=True), dim=-1)
        cd = F.normalize(cd - cd.mean(-1, keepdim=True), dim=-1)
        logits = td @ cd.transpose(1, 2)
        row = rank_summary(logits, batch["perm"])
        row.update(decoded_geometry(logits, batch["perm"]))
        rows.append(row)

    for key in ("r1", "r5", "r20", "median_rank", "place_acc", "neighbour_acc"):
        print(f"{key}={sum(r[key] for r in rows) / len(rows):.4f}")


if __name__ == "__main__":
    sys.exit(main())
