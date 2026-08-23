"""End-to-end gate: dirty board in, SSIM out, nothing oracular anywhere.

Every intermediate metric in this project has misled at least once -- residual
sigma pointed the wrong way three times, R@1 turned out not to be what solvers
consume, and place_acc read at chance for a layout that was correct up to a
cyclic shift.  This runs the whole chain the way a submission would and reports
the only number the platform cares about.

Chain:
  tiles -> learned descriptors -> log-probability at the model's temperature
        -> Sinkhorn -> cycle consistency          (src/seam_cost.py)
        -> greedy loop-verified construction      (src/solve_loop.py)
        -> origin anchored to the LP layout       (M96; the LP has no torus
                                                   ambiguity, greedy has the
                                                   better relative structure)
        -> assemble from RAW pixels, optional non-local denoise

Assembly uses raw pixels rather than restored ones on purpose: the restorer
makes tiles worse as an image and worse as matcher input (M91).

Reference points: the platform submission scores 0.23748, the leader 0.40, a
perfect layout of these dirty tiles 0.4734, and a random layout 0.0829.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

import infer_rank96 as rank96
from config import CACHE_DIR, CKPT_DIR, FS, GRID as G, NFRAG as N, TRAIN_INP, TRAIN_TGT
from models import RestoreNet
from distort import distort_frags_scaled
from restore_tile import to_frags
from rerank_fuse import fused_costs
from seam_cost import costs_from_model
from seam_embed import SeamEmbed
from seam_rerank import SeamRerank
from solve_lp import build_matches, solve_lp
from solve_loop import solve as solve_loop
from solve_relax import solve_relax
from torus_origin import best_possible_shift, fix_origin


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


def assemble(tiles, lay):
    x = np.clip(tiles[np.asarray(lay)], 0, 255).astype(np.uint8)
    return x.reshape(G, G, FS, FS, 3).transpose(0, 2, 1, 3, 4).reshape(G * FS, G * FS, 3)


def anchor_to(board, ref, grid=G):
    """Roll `board` to whichever cyclic shift agrees most with `ref`.

    Greedy's layout is only defined up to a cyclic shift and it picks that shift
    badly -- worth 0.40 at the best one against 0.001 at its own (M88).  The LP
    solves for absolute translations and so has no such ambiguity, which makes
    it a usable anchor even when its own placement is worse.
    """
    b = board.reshape(grid, grid)
    r = ref.reshape(grid, grid)
    best, arg = -1, (0, 0)
    for dr in range(grid):
        rolled = np.roll(b, -dr, axis=0)
        for dc in range(grid):
            k = int((np.roll(rolled, -dc, axis=1) == r).sum())
            if k > best:
                best, arg = k, (dr, dc)
    return np.roll(b, (-arg[0], -arg[1]), axis=(0, 1)).ravel()


def correct_adjacencies(lay, grid=G):
    """Fraction of the layout's 1104 adjacencies that are true neighbours.

    This, not place_acc, is what the post-processing is paid for.  Greedy scores
    +0.02 SSIM at a strict place_acc of 0.007 (M104) because it glues true
    neighbours together even when the whole block sits in the wrong slot, and a
    restoration net sees a locally coherent image either way.  Unlike place_acc
    it also has no cliff -- every recovered adjacency counts on its own.
    """
    b = np.asarray(lay).reshape(grid, grid)
    ok = int((b[:, 1:] == b[:, :-1] + 1).sum() + (b[1:] == b[:-1] + grid).sum())
    return ok / (2 * grid * (grid - 1))


def layout_cost(lay, cost_h, cost_v, grid=G):
    """Summed seam cost of a finished layout -- the objective, not a proxy."""
    b = np.asarray(lay).reshape(grid, grid)
    return float(cost_h[b[:, :-1], b[:, 1:]].sum() + cost_v[b[:-1], b[1:]].sum())


def edge_precision(cost_h, cost_v):
    """Mutual top-1 count and precision -- the quantity that gates assembly."""
    n = ok = 0
    for C, step, valid in ((cost_h, 1, lambda p: p % G != G - 1),
                           (cost_v, G, lambda p: p < N - G)):
        D = C.copy()
        np.fill_diagonal(D, np.inf)
        fwd, back = D.argmin(1), D.argmin(0)
        for i in range(N):
            if back[fwd[i]] == i:
                n += 1
                ok += int(valid(i) and fwd[i] == i + step)
    return n, ok / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="seam_embed_v1.pt")
    ap.add_argument("--boards", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--weight", type=float, default=0.35,
                    help="cycle-consistency weight; re-tuned from 0.50, "
                         "worth +0.015 edge precision")
    ap.add_argument("--rerank", default="",
                    help="second-stage checkpoint; splices its scores into "
                         "the shortlist before calibration")
    ap.add_argument("--blend", type=float, default=0.7,
                    help="how much of the re-ranker to trust on its shortlist")
    ap.add_argument("--severity", type=float, default=-1.0,
                    help="re-corrupt clean targets at this strength instead "
                         "of using the real inputs; -1 means real")
    ap.add_argument("--post", default="r5nlm",
                    choices=["none", "nlm", "r5nlm"],
                    help="post-processing; r5nlm is what the 0.23748 submission uses")
    ap.add_argument("--r5", default="E:/pazzle_work/pazzle_fixed_orientation_20260813/R5_restore_unet/r5_capacity_fp32.pt")
    a = ap.parse_args()

    dev = "cuda"
    ck = torch.load(Path(CKPT_DIR) / a.ckpt, map_location=dev, weights_only=False)
    ta = ck["args"]
    model = SeamEmbed(ta["ch"], ta["blocks"], ta["dim"], ta["strip"],
                      ta.get("head", "global"),
                      predict=ta.get("predict_weight", 0) > 0).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"{a.ckpt}: step {ck.get('step')}, its eval {ck.get('eval')}", flush=True)

    rerank = None
    if a.rerank:
        rk = torch.load(Path(CKPT_DIR) / a.rerank, map_location=dev, weights_only=False)
        ra = rk["args"]
        rerank = SeamRerank(ra["ch"], ra["blocks"], ra["width"]).to(dev)
        rerank.load_state_dict(rk["model"])
        rerank.eval()
        print(f"{a.rerank}: step {rk.get('step')}, its eval {rk.get('eval')}", flush=True)

    r5 = None
    if a.post == "r5nlm":
        payload = torch.load(a.r5, map_location=dev, weights_only=False)
        st = (payload.get("model") or payload.get("model_state_dict")
              or payload.get("state_dict") or payload)
        r5 = RestoreNet(base=st["stem.weight"].shape[0],
                        depth=1 + sum(1 for k in st
                                      if k.startswith("down.")
                                      and k.endswith(".weight"))).to(dev)
        r5.load_state_dict(st, strict=True)
        r5.eval()

    def post(img):
        """Exactly the arm behind the 0.23748 score: R5 U-Net then NLM."""
        if r5 is not None:
            with torch.no_grad():
                t = torch.from_numpy(img).to(dev, torch.float32)
                t = t.permute(2, 0, 1)[None] / 255.0
                o = r5(t).clamp_(0, 1).squeeze(0).permute(1, 2, 0)
            img = np.rint(o.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if a.post != "none":
            img = rank96.fixed_nlm(img)
        return img

    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names, inv_all = blob["names"][-300:], blob["inv"][-300:]
    truth = np.arange(N)
    rows = []
    t0 = time.time()
    for k in range(a.boards):
        nm = str(names[k])
        iv = inv_all[k].astype(np.int64)
        target = load_rgb(Path(TRAIN_TGT) / nm)
        if a.severity >= 0.0:
            tiles = distort_frags_scaled(to_frags(target).astype(np.uint8),
                                         np.random.default_rng(k),
                                         a.severity).astype(np.float32)
        else:
            tiles = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[iv]

        if rerank is None:
            CH, CV = costs_from_model(model, tiles, rounds=a.rounds, weight=a.weight)
        else:
            CH, CV = fused_costs(model, rerank, tiles, k=ra["k"], width=ra["width"],
                                 rounds=a.rounds, weight=a.weight,
                                 blend=a.blend)
        n_edge, prec = edge_precision(CH, CV)
        lp = solve_lp(CH, CV)
        lp = np.arange(N) if lp is None else lp
        greedy = solve_loop(CH, CV)[0]
        hybrid = anchor_to(greedy, lp)

        # a random layout is the control: it is what the pipeline scores with no
        # assembly at all, so the gap to it is exactly what our solver is worth
        rnd = np.random.default_rng(k).permutation(N)
        gr_fixed = fix_origin(greedy, tiles, metric="mgc")
        # relaxation labelling cannot break the symmetry of a uniform start --
        # from scratch it scores 0.003 even on clean tiles with MGC costs -- but
        # seeded with a construction it refines: adjacency 0.4882 -> 0.5512 at
        # severity 0.3 and 0.2124 -> 0.2405 on real boards
        relax = solve_relax(CH, CV, rounds=200, init=gr_fixed)
        arms = [("lp", lp), ("greedy", gr_fixed), ("hybrid", hybrid),
                ("relax", relax), ("random", rnd)]
        # Pick between arms by the objective itself.  The solvers are erratic
        # near the activation knee -- at severity 0.0 greedy scored place_acc
        # 0.004 on costs BETTER than the severity 0.3 run where it scored 0.131
        # -- and M82 established that under the learned cost the true layout is
        # the minimum, so total cost is a usable selector with nothing oracular
        # in it.
        cheapest = min(arms, key=lambda kv: layout_cost(kv[1], CH, CV))
        arms.append(("selected", cheapest[1]))

        best = None
        for tag, lay in arms:
            img = post(assemble(tiles, lay))
            s = float(ssim_fn(img, target, channel_axis=2, data_range=255))
            acc = float(np.mean(lay == truth))
            if best is None or s > best[2]:
                best = (tag, acc, s)
            rows.append([tag, n_edge, prec, acc, s, correct_adjacencies(lay),
                         best_possible_shift(greedy, truth)[0]])
        print(f"  board {k}: {n_edge} edges at precision {prec:.3f}; "
              f"best arm {best[0]} place {best[1]:.4f} SSIM {best[2]:.4f}", flush=True)

    print(f"\n{a.boards} boards in {time.time()-t0:.0f} s")
    print(f"{'arm':10s} {'edges':>7} {'precision':>10} {'place_acc':>10} "
          f"{'adjacency':>10} {'SSIM':>8} {'ceiling':>8}")
    for tag in ("random", "lp", "greedy", "hybrid", "relax", "selected"):
        v = np.mean([r[1:] for r in rows if r[0] == tag], axis=0)
        print(f"{tag:10s} {v[0]:7.0f} {v[1]:10.3f} {v[2]:10.4f} "
              f"{v[4]:10.4f} {v[3]:8.4f} {v[5]:8.4f}")
    print()
    print(f"post-processing {a.post}: on these boards a chance layout scores 0.192,")
    print("place_acc 0.10 -> 0.228, 0.20 -> 0.267, a perfect layout 0.631 (M103).")
    print("Platform submission 0.23748; the LP switches on near edge precision 0.72.")


if __name__ == "__main__":
    main()
