"""The metric that actually matters: edges harvested at a precision floor.

bb_prec is a summary; the solver has a hard requirement.  Feeding it synthetic
edge sets shows assembly needs precision >=0.95 at ~900 edges (574/576 tiles at
precision 1.00, but only 162 correct tiles at 0.95 and 92 at 0.85).  Conflict
pruning does not rescue a lower precision, because at 900 edges over 576 tiles
the graph has degree 1.5 and almost no cycles to cross-check against.

So the scorer should be judged by: sorting all candidate edges by confidence,
how many can we take before precision drops below the floor?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from config import CACHE_DIR, CKPT_DIR, GRID as G, NFRAG as N, TRAIN_INP, VAL_COUNT
from pair_compat import PairCompat, dense_scores
from restore_tile import TileRestorer, ridge_cost, to_frags


def load_tiles(name: str) -> np.ndarray:
    img = cv2.imread(str(Path(TRAIN_INP) / name), cv2.IMREAD_COLOR)
    return to_frags(np.ascontiguousarray(img[:, :, ::-1])).astype(np.float32)


def yield_curve(score: np.ndarray, axis: str, floors=(0.99, 0.95, 0.90, 0.80)) -> dict[float, int]:
    """score[i,j] higher = better.  Rank mutual-best edges by margin, walk down."""
    s = score.copy()
    np.fill_diagonal(s, -np.inf)
    step = 1 if axis == "h" else G
    on_grid = (lambda p: p % G != G - 1) if axis == "h" else (lambda p: p < N - G)
    best = s.argmax(1)
    rev = s.argmax(0)
    part = np.partition(s, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    cand = [(margin[a], a, int(best[a])) for a in range(N) if rev[best[a]] == a]
    cand.sort(reverse=True)
    hits = np.cumsum([1 if (on_grid(a) and b == a + step) else 0 for _, a, b in cand])
    out = {}
    for floor in floors:
        n = 0
        for k in range(len(cand)):
            if hits[k] / (k + 1) >= floor:
                n = k + 1
        out[floor] = n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path(CACHE_DIR) / "restore_labels.npz")
    ap.add_argument("--restorer", type=Path, default=Path(CKPT_DIR) / "tile_restorer_seam.pt")
    ap.add_argument("--pair", type=Path, default=None)
    ap.add_argument("--boards", type=int, default=6)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.restorer, map_location=dev, weights_only=False)
    rest = TileRestorer(ck["args"]["ch"], ck["args"]["blocks"],
                       ck["args"].get("residual", False)).to(dev)
    rest.load_state_dict(ck["model"]); rest.eval()

    pair = None
    if args.pair is not None and args.pair.is_file():
        pk = torch.load(args.pair, map_location=dev, weights_only=False)
        pair = PairCompat(pk["args"]["ch"]).to(dev)
        pair.load_state_dict(pk["model"]); pair.eval()
        print(f"pair_compat: step {pk['step']} bb_prec {pk['bb_prec']:.4f}", flush=True)

    blob = np.load(args.labels, allow_pickle=True)
    names, inv_all = blob["names"][-VAL_COUNT:], blob["inv"][-VAL_COUNT:]
    agg: dict[str, dict[float, list]] = {}

    for k in range(args.boards):
        tiles = load_tiles(str(names[k]))[inv_all[k].astype(np.int64)]
        x = torch.from_numpy(tiles).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=dev == "cuda"):
            xr = torch.cat([rest(x[i:i + 288]) for i in range(0, len(x), 288)])
        npr = xr.float().permute(0, 2, 3, 1).clamp(0, 255).cpu().numpy()
        for axis in ("h", "v"):
            arms = {"ridge_raw": -ridge_cost(tiles, axis=axis),
                    "ridge_restored": -ridge_cost(npr, axis=axis)}
            if pair is not None:
                sl = np.argsort(ridge_cost(npr, axis=axis), axis=1)[:, :64]
                arms["pair_compat"] = dense_scores(pair, xr.float().clamp(0, 255), axis, shortlist=sl)
            for tag, sc in arms.items():
                for floor, n in yield_curve(sc, axis).items():
                    agg.setdefault(tag, {}).setdefault(floor, []).append(n)
        print(f"  board {k+1}/{args.boards}", flush=True)

    print(f"\n{'arm':16s} " + " ".join(f"p>={f:.2f}" for f in (0.99, 0.95, 0.90, 0.80)))
    print("(edges per board per axis; solver needs ~450/axis at p>=0.95)")
    for tag, d in agg.items():
        print(f"{tag:16s} " + " ".join(f"{np.mean(d[f]):7.1f}" for f in (0.99, 0.95, 0.90, 0.80)))


if __name__ == "__main__":
    main()
