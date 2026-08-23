"""Build the (dirty tile -> clean tile) label cache for the restoration model.

Matches each train input's 576 distorted tiles to its target's clean tiles with
a full 20x20 photometrically-normalised descriptor plus Hungarian assignment.
Both the per-tile brightness b and contrast a are scalar per tile, so
normalising each tile by its own mean/std removes them exactly.

Measured on synthetic boards with a known permutation: raw accuracy 0.825,
but the assignment margin is strongly calibrated -- keeping the top 50% of
positions by margin gives 0.996 accuracy.  Training therefore uses a margin
threshold rather than the whole board.

Output: E:/pazzle_work/cache/restore_labels.npz
  names (M,)      train filenames
  inv   (M,576)   inv[k,p] = index of the input tile belonging at grid pos p
  margin(M,576)   assignment margin at grid pos p (higher = more reliable)
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import GRID as G, FS, NFRAG as N, TRAIN_INP, TRAIN_TGT, CACHE_DIR


def to_frags(img: np.ndarray) -> np.ndarray:
    return img.reshape(G, FS, G, FS, 3).transpose(0, 2, 1, 3, 4).reshape(N, FS, FS, 3)


def from_frags(frags: np.ndarray) -> np.ndarray:
    return frags.reshape(G, G, FS, FS, 3).transpose(0, 2, 1, 3, 4).reshape(G * FS, G * FS, 3)


def blur3(x: np.ndarray) -> np.ndarray:
    """Separable 3x3 Gaussian with reflect pad, matching the generator."""
    xp = np.pad(x, ((0, 0), (1, 1), (0, 0), (0, 0)), "reflect")
    x = .25 * xp[:, :-2] + .5 * xp[:, 1:-1] + .25 * xp[:, 2:]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (0, 0)), "reflect")
    return .25 * xp[:, :, :-2] + .5 * xp[:, :, 1:-1] + .25 * xp[:, :, 2:]


def normalised(frags: np.ndarray) -> np.ndarray:
    """Per-tile mean/std normalisation: exactly inverts the generator's scalar a,b."""
    x = frags.astype(np.float32).reshape(len(frags), -1)
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def match_board(dirty: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return inv[pos] = dirty tile index, and the per-position assignment margin."""
    di = normalised(dirty)
    dt = normalised(blur3(clean.astype(np.float32)))
    cost = (di * di).sum(1)[:, None] + (dt * dt).sum(1)[None, :] - 2.0 * di @ dt.T
    rows, cols = linear_sum_assignment(cost)
    inv = np.empty(N, np.int64)
    inv[cols] = rows
    second = np.partition(cost, 1, axis=1)
    margin = (second[:, 1] - second[:, 0]) / (np.abs(second[:, 0]) + 1e-6)
    per_pos = np.empty(N, np.float32)
    per_pos[cols] = margin[rows]
    return inv, per_pos


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.shape != (G * FS, G * FS, 3):
        raise RuntimeError(f"bad image: {path}")
    return img[:, :, ::-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(CACHE_DIR) / "restore_labels.npz")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(TRAIN_INP) if n.endswith(".png"))
    if args.limit:
        names = names[: args.limit]
    inv_all = np.zeros((len(names), N), np.int16)
    margin_all = np.zeros((len(names), N), np.float32)

    started = time.perf_counter()
    for k, nm in enumerate(names):
        dirty = to_frags(load_rgb(Path(TRAIN_INP) / nm))
        clean = to_frags(load_rgb(Path(TRAIN_TGT) / nm))
        inv, margin = match_board(dirty, clean)
        inv_all[k] = inv.astype(np.int16)
        margin_all[k] = margin
        if (k + 1) % 250 == 0:
            rate = (k + 1) / (time.perf_counter() - started)
            print(f"  {k+1}/{len(names)}  {rate:.1f} img/s  "
                  f"eta {(len(names)-k-1)/rate/60:.1f} min", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, names=np.array(names), inv=inv_all, margin=margin_all)
    thr = np.median(margin_all)
    print(f"saved {args.out}  images={len(names)}  median_margin={thr:.4f}  "
          f"elapsed={(time.perf_counter()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
