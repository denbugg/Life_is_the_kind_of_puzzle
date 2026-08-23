"""The EM loop: restore -> solve -> re-restore with context -> re-solve.

Rationale (all measured, see restore_context.py and pazzle-seam-budget):
  * a lone 20x20 tile denoises poorly, so the context-free restorer caps around
    ridge bb_prec 0.40, i.e. ~0.30 SSIM;
  * a tile with placed neighbours denoises over 60x60 instead, and the context
    model beat the context-free one after 40 steps versus 8000;
  * placement only has to reach ~0.30 to beat the current 0.23749 submission,
    and 0.64 to match the leader, so iteration 1 does not need to be good.

Each iteration re-scores with BOTH measures and keeps whichever ranks the true
neighbour better on this board: the ridge cost dominates while tiles are noisy
(0.396 vs 0.113) and MGC takes over once they are clean (0.994 vs 0.796), and
the crossover is exactly what the loop is trying to cross.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim

from config import CACHE_DIR, CKPT_DIR, GRID as G, NFRAG as N, TRAIN_INP, TRAIN_TGT, VAL_COUNT
from mgc import mgc_cost
from restore_context import ContextRestorer, build_blocks
from restore_tile import TileRestorer, ridge_cost, to_frags
from solve_loop import solve as solve_loop
from torus_origin import fix_origin


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(img[:, :, ::-1])


def assemble(frags: np.ndarray, order: np.ndarray) -> np.ndarray:
    f = np.clip(frags[np.asarray(order)], 0, 255).astype(np.uint8)
    return f.reshape(G, G, f.shape[1], f.shape[2], 3).transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


def ssim(a, b) -> float:
    return float(sk_ssim(a, b, channel_axis=2, data_range=255))


@torch.no_grad()
def restore_free(model, tiles, dev):
    x = torch.from_numpy(tiles).permute(0, 3, 1, 2).to(dev)
    with torch.autocast("cuda", torch.float16, enabled=dev == "cuda"):
        o = torch.cat([model(x[i:i + 288]) for i in range(0, len(x), 288)])
    return o.float().permute(0, 2, 3, 1).clamp(0, 255).cpu().numpy()


@torch.no_grad()
def restore_ctx(model, tiles, board, dev):
    block, mask = build_blocks(tiles, board, G)
    b = torch.from_numpy(block).permute(0, 3, 1, 2).to(dev)
    m = torch.from_numpy(mask).permute(0, 3, 1, 2).to(dev)
    with torch.autocast("cuda", torch.float16, enabled=dev == "cuda"):
        o = torch.cat([model(b[i:i + 128], m[i:i + 128]) for i in range(0, N, 128)])
    out = o.float().permute(0, 2, 3, 1).clamp(0, 255).cpu().numpy()
    # out[p] is the restored tile for POSITION p; map back to tile indexing
    res = tiles.copy()
    for p in range(N):
        if board[p] >= 0:
            res[board[p]] = out[p]
    return res


def solve_best(tiles):
    """Solve with both measures, keep the layout whose own objective is tighter."""
    best = None
    for metric in ("ridge", "mgc"):
        if metric == "mgc":
            ch, cv_ = mgc_cost(tiles, "h"), mgc_cost(tiles, "v")
        else:
            ch, cv_ = ridge_cost(tiles, axis="h"), ridge_cost(tiles, axis="v")
        board, comps = solve_loop(ch, cv_)
        board = fix_origin(board, tiles, metric=metric)
        # label-free quality proxy: mean cost along the realised seams
        grid = board.reshape(G, G)
        val = (ch[grid[:, :-1], grid[:, 1:]].mean() + cv_[grid[:-1], grid[1:]].mean()) / 2
        norm = val / (ch.mean() + cv_.mean()) * 2                # scale-free
        if best is None or norm < best[0]:
            best = (norm, board, metric)
    return best[1], best[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path(CACHE_DIR) / "restore_labels.npz")
    ap.add_argument("--free", type=Path, default=Path(CKPT_DIR) / "tile_restorer_seam.pt")
    ap.add_argument("--ctx", type=Path, default=Path(CKPT_DIR) / "restore_context.pt")
    ap.add_argument("--boards", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckf = torch.load(args.free, map_location=dev, weights_only=False)
    free = TileRestorer(ckf["args"]["ch"], ckf["args"]["blocks"],
                        ckf["args"].get("residual", False)).to(dev)
    free.load_state_dict(ckf["model"]); free.eval()
    ckc = torch.load(args.ctx, map_location=dev, weights_only=False)
    ctx = ContextRestorer(ckc["args"]["ch"], ckc["args"]["blocks"]).to(dev)
    ctx.load_state_dict(ckc["model"]); ctx.eval()
    print(f"free: step {ckf['step']} bb {ckf['bb_prec']:.3f} | "
          f"ctx: step {ckc['step']} bb {ckc['bb_prec']:.3f}", flush=True)

    blob = np.load(args.labels, allow_pickle=True)
    names, inv_all = blob["names"][-VAL_COUNT:], blob["inv"][-VAL_COUNT:]
    hist: dict[int, list] = {}
    started = time.perf_counter()

    for k in range(args.boards):
        nm = str(names[k])
        shuffled = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)
        target = load_rgb(Path(TRAIN_TGT) / nm)
        inv = inv_all[k].astype(np.int64)

        tiles = restore_free(free, shuffled, dev)
        for it in range(args.iters):
            board, metric = solve_best(tiles)
            acc = float(np.mean(board == inv))
            img = assemble(shuffled, board)                      # raw pixels, honest SSIM
            hist.setdefault(it, []).append((acc, ssim(img, target), metric == "mgc"))
            tiles = restore_ctx(ctx, tiles, board, dev)
        print(f"  {k+1}/{args.boards} {nm}", flush=True)

    print(f"\n{'iter':>5} {'place_acc':>10} {'ssim_raw':>10} {'used_mgc':>9}")
    for it in sorted(hist):
        v = np.array(hist[it], float).mean(0)
        print(f"{it:5d} {v[0]:10.4f} {v[1]:10.4f} {v[2]:9.2f}")
    print(f"\nplatform S1 = 0.23749 ; leader = 0.40 ; elapsed "
          f"{(time.perf_counter()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
