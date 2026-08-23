"""Honest gate for the learned seam matcher: recall, objective soundness, assembly.

Three questions, in the order that matters:

1. R@1 / R@20 against MGC on the same boards -- is the cost better at all.
2. Is the true layout now the MINIMUM of the summed objective?  With MGC costs
   it is not (annealing reaches 0.858x the true cost while placing nothing), and
   that single fact invalidated every solver in the repo.  If the learned cost
   fixes it, the solver work becomes usable again; if not, recall gains cannot
   convert and we would be measuring the wrong thing.
3. Only then: actual place_acc and SSIM, assembled from real dirty pixels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

from config import CACHE_DIR, CKPT_DIR, FS, GRID as G, NFRAG as N, TRAIN_INP, TRAIN_TGT
from mgc import mgc_cost
from restore_tile import to_frags
from seam_cost import costs_from_model
from seam_embed import SeamEmbed
from solve_anneal import solve_anneal, total_cost
from solve_lp import solve_lp
from solve_loop import solve as solve_loop
from torus_origin import fix_origin


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(img[:, :, ::-1])


# cost construction lives in seam_cost: log-probability at the model's own
# temperature, Sinkhorn, then cycle consistency between the two axes.  The naive
# 1 - cosine form has margins four times too flat for the LP weighting (M85/M86)


def recall(CH, CV):
    r = []
    for C, step, ok in ((CH, 1, lambda p: p % G != G - 1), (CV, G, lambda p: p < N - G)):
        D = C.copy(); np.fill_diagonal(D, np.inf)
        idx = np.array([p for p in range(N) if ok(p)])
        o = np.argsort(D[idx], axis=1)
        rk = np.array([np.where(o[i] == idx[i] + step)[0][0] for i in range(len(idx))])
        r.append([(rk == 0).mean(), (rk < 20).mean()])
    return np.mean(r, axis=0)


def assemble(frags, lay):
    x = np.clip(frags[np.asarray(lay)], 0, 255).astype(np.uint8)
    return x.reshape(G, G, FS, FS, 3).transpose(0, 2, 1, 3, 4).reshape(G * FS, G * FS, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="seam_embed_v1.pt")
    ap.add_argument("--boards", type=int, default=6)
    ap.add_argument("--anneal-iters", type=int, default=6_000_000)
    a = ap.parse_args()

    dev = "cuda"
    ck = torch.load(Path(CKPT_DIR) / a.ckpt, map_location=dev, weights_only=False)
    ta = ck["args"]
    model = SeamEmbed(ta["ch"], ta["blocks"], ta["dim"], ta["strip"],
                  ta.get("head", "global")).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"checkpoint from step {ck.get('step')}, its own eval {ck.get('eval')}")

    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names, inv_all = blob["names"][-300:], blob["inv"][-300:]
    truth = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(0)
    rows = []
    for k in range(a.boards):
        nm = str(names[k]); iv = inv_all[k].astype(np.int64)
        raw = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[iv]
        tgt = load_rgb(Path(TRAIN_TGT) / nm)

        CH, CV = costs_from_model(model, raw)
        r_learn = recall(CH, CV)
        r_mgc = recall(mgc_cost(raw, "h"), mgc_cost(raw, "v"))

        # objective soundness on the LEARNED cost
        ct = total_cost(truth, CH, CV, G, N)
        cr = np.mean([total_cost(rng.permutation(N).astype(np.int64), CH, CV, G, N)
                      for _ in range(10)])
        lay_an, ca = solve_anneal(CH, CV, iters=a.anneal_iters, restarts=2)

        lp = solve_lp(CH, CV)
        lp_acc = float(np.mean(lp == truth)) if lp is not None else 0.0
        gr = fix_origin(solve_loop(CH, CV)[0], raw, metric="mgc")
        rows.append([
            r_learn[0], r_learn[1], r_mgc[0],
            cr / ct, ca / ct, float(np.mean(lay_an == truth)),
            lp_acc, float(np.mean(gr == truth)),
            float(ssim_fn(assemble(raw, lay_an), tgt, channel_axis=2, data_range=255)),
            float(ssim_fn(assemble(raw, lp), tgt, channel_axis=2, data_range=255))
            if lp is not None else 0.0,
        ])
        print(f"  board {k}: R@1 {r_learn[0]:.3f} (mgc {r_mgc[0]:.3f}), "
              f"annealed/true {ca/ct:.3f}", flush=True)

    v = np.mean(rows, axis=0)
    print(f"\n{'learned R@1':28s} {v[0]:8.4f}   (MGC on same boards {v[2]:.4f})")
    print(f"{'learned R@20':28s} {v[1]:8.4f}")
    print(f"{'random / true cost':28s} {v[3]:8.4f}   (higher = objective separates)")
    print(f"{'annealed / true cost':28s} {v[4]:8.4f}   (ABOVE 1.0 = truth is the minimum)")
    print(f"{'place_acc, annealing':28s} {v[5]:8.4f}")
    print(f"{'place_acc, LP':28s} {v[6]:8.4f}")
    print(f"{'place_acc, greedy+origin':28s} {v[7]:8.4f}   (chance 0.0017)")
    print(f"{'SSIM, annealed layout':28s} {v[8]:8.4f}")
    print(f"{'SSIM, LP layout':28s} {v[9]:8.4f}   (submission 0.2375, leader 0.40)")


if __name__ == "__main__":
    main()
