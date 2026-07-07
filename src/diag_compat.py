"""Diagnose compat retrieval quality on real val fragments:
top-1 accuracy and recall@K for right/down neighbors. recall@K predicts whether
top-K pairwise re-scoring can recover accuracy."""
import os, argparse
import numpy as np
import torch
from config import GRID, CKPT_DIR, TRAIN_INP, CACHE_DIR
from imgio import load, to_frags, train_val_split
from pipeline import load_compat
from solve import compat_scores

DEV = "cuda"


def ranks(R, anchors, trues):
    """For each (anchor, true) return rank of true among candidates (0=best)."""
    out = np.empty(len(anchors), np.int64)
    for k, (a, t) in enumerate(zip(anchors, trues)):
        row = R[a].copy(); row[a] = -1e30
        out[k] = int((row > row[t]).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--tag", default="compat")
    args = ap.parse_args()
    model, ck = load_compat(args.tag)
    print(f"compat step={ck.get('step')} val={ck.get('val')}")
    z = np.load(os.path.join(CACHE_DIR, "perms.npz"), allow_pickle=True)
    inv_ = {n: z["inv"][i].astype(np.int64) for i, n in enumerate(z["names"])}
    _, val = train_val_split()
    Ks = [1, 3, 5, 10, 25, 50]
    hr, vr = [], []
    for nm in val[:args.n]:
        frags = to_frags(load(os.path.join(TRAIN_INP, nm)))
        R, D = compat_scores(model, frags, DEV)
        inv = inv_[nm]
        # right edges: position p -> p+1 ; down: p -> p+GRID
        ph = [p for p in range(GRID * GRID) if p % GRID != GRID - 1]
        pv = [p for p in range(GRID * GRID) if p // GRID != GRID - 1]
        hr.append(ranks(R, inv[ph], inv[[p + 1 for p in ph]]))
        vr.append(ranks(D, inv[pv], inv[[p + GRID for p in pv]]))
    hr = np.concatenate(hr); vr = np.concatenate(vr)
    print(f"\nN={args.n} images   right edges={len(hr)} down edges={len(vr)}")
    print("        " + "".join(f"  R@{k:<4}" for k in Ks))
    print("right  " + "".join(f"  {(hr < k).mean():.3f}" for k in Ks))
    print("down   " + "".join(f"  {(vr < k).mean():.3f}" for k in Ks))
    print(f"\nmedian rank right={np.median(hr):.0f} down={np.median(vr):.0f} (of 575)")


if __name__ == "__main__":
    main()
