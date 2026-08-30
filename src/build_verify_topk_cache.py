"""Build a compact full-train hard-candidate cache for seam verification.

The cache contains only corrupted input tiles' candidate IDs/scores and the
permutation-derived candidate label.  Clean target pixels are never stored or
consumed.  Arrays are memory-mapped and every completed board is marked, so a
long all-7000 build can resume without repeating work.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import CACHE_DIR, CKPT_DIR, GRID, TRAIN_INP
from infer_coarse_field import load_rgb
from restore_tile import to_frags
from seam_cost import costs_from_models
from seam_embed import SeamEmbed

N = GRID * GRID


def _open(path: Path, dtype, shape):
    if path.exists():
        arr = np.load(path, mmap_mode="r+")
        if arr.dtype != np.dtype(dtype) or arr.shape != shape:
            raise ValueError(f"cache schema mismatch: {path}")
        return arr
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def load_matchers(names, device):
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
        model.eval().requires_grad_(False)
        models.append(model)
    return models


def shortlist(cost, k):
    cost = np.asarray(cost, np.float32).copy()
    np.fill_diagonal(cost, np.inf)
    ids = np.argpartition(cost, k - 1, axis=1)[:, :k]
    values = np.take_along_axis(cost, ids, axis=1)
    order = np.argsort(values, axis=1)
    ids = np.take_along_axis(ids, order, axis=1)
    # The verifier consumes compatibility, higher is better.
    score = -np.take_along_axis(cost, ids, axis=1)
    return ids.astype(np.uint16), score.astype(np.float16)


def candidate_labels(ids, offset, valid):
    labels = np.full(len(ids), -1, np.int8)
    rows = np.flatnonzero(valid)
    target = rows + offset
    hit = ids[rows] == target[:, None]
    labels[rows] = np.where(hit.any(1), hit.argmax(1), ids.shape[1]).astype(np.int8)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matcher", nargs="+",
                        default=["seam_embed_v3.pt", "seam_embed_local.pt"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out", type=Path,
                        default=Path(CACHE_DIR) / "verify_top5_v2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=0,
                        help="0 means through all 7000 boards")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 1 <= args.topk <= 32:
        parser.error("--topk must be in 1..32")

    labels = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names = np.asarray(labels["names"])
    inverse = np.asarray(labels["inv"], np.uint16)
    total = len(names)
    args.out.mkdir(parents=True, exist_ok=True)
    names_path = args.out / "names.npy"
    if names_path.exists():
        old_names = np.load(names_path)
        if not np.array_equal(old_names, names):
            raise ValueError("cache names disagree with restore_labels")
    else:
        np.save(names_path, names)
    inv = _open(args.out / "inv.npy", np.uint16, (total, N))
    inv[:] = inverse
    idx = _open(args.out / "idx.npy", np.uint16,
                (total, 2, N, args.topk))
    val = _open(args.out / "val.npy", np.float16,
                (total, 2, N, args.topk))
    lab = _open(args.out / "lab.npy", np.int8, (total, 2, N))
    done = _open(args.out / "done.npy", np.uint8, (total,))
    manifest = {
        "version": 2, "boards": total, "grid": GRID, "topk": args.topk,
        "matchers": args.matcher, "train_end": total - 300,
        "validation_start": total - 300,
        "pixel_source": "train/inputs only",
        "label_source": "restore_labels permutation only",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                             encoding="utf-8")

    models = load_matchers(args.matcher, args.device)
    end = total if not args.count else min(total, args.start + args.count)
    todo = [i for i in range(args.start, end) if not done[i]]
    horizontal = np.arange(N) % GRID != GRID - 1
    vertical = np.arange(N) // GRID != GRID - 1
    t0 = time.time()
    for number, i in enumerate(todo, 1):
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / str(names[i]))).astype(
            np.float32)[inverse[i].astype(np.int64)]
        ch, cv = costs_from_models(models, tiles)
        hi, hv = shortlist(ch, args.topk)
        vi, vv = shortlist(cv, args.topk)
        idx[i, 0], val[i, 0] = hi, hv
        idx[i, 1], val[i, 1] = vi, vv
        lab[i, 0] = candidate_labels(hi, 1, horizontal)
        lab[i, 1] = candidate_labels(vi, GRID, vertical)
        done[i] = 1
        if number % 20 == 0 or number == len(todo):
            idx.flush(); val.flush(); lab.flush(); done.flush()
            elapsed = time.time() - t0
            rate = elapsed / number
            print(f"{number}/{len(todo)} new; total {int(done.sum())}/{total}; "
                  f"{rate:.3f}s/board; eta {(len(todo)-number)*rate/60:.1f}m",
                  flush=True)
    print(json.dumps({**manifest, "complete": int(done.sum())}, indent=2))


if __name__ == "__main__":
    main()
