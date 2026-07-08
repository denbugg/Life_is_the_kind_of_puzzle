"""Solver-independent quality of the pairwise compatibility scores on val.
Reports true-neighbour rank (R@1/R@5/median) and best-buddy precision -- the
metric that actually predicts whether the puzzle is assemblable. A jigsaw solver
needs bb_prec ~0.9; the v1 scorer sat at ~0.48. This is the run-#2 success gauge."""
import os, argparse, numpy as np, torch
from config import GRID, NFRAG, TRAIN_INP, CACHE_DIR
from imgio import load, to_frags, train_val_split
from pipeline import load_pair
from solve import pairwise_scores_full

DEV = "cuda"; G = GRID


def true_nbrs(inv):
    f2p = np.empty(NFRAG, int); f2p[inv] = np.arange(NFRAG)
    tr = -np.ones(NFRAG, int); td = -np.ones(NFRAG, int)
    for a in range(NFRAG):
        p = f2p[a]
        if p % G != G - 1: tr[a] = inv[p + 1]
        if p // G != G - 1: td[a] = inv[p + G]
    return tr, td


def ranks(M, tn):
    keep = np.where(tn >= 0)[0]
    Mk = M[keep].copy()
    Mk[np.arange(len(keep)), keep] = -1e30                 # mask self
    tn_score = M[keep, tn[keep]]
    return (Mk > tn_score[:, None]).sum(1) + 1


def bb_precision(M, tn):
    Mm = M.copy(); np.fill_diagonal(Mm, -1e30)
    bf = Mm.argmax(1); bb = np.where(Mm.argmax(0)[bf] == np.arange(NFRAG))[0]
    bv = bb[tn[bb] >= 0]
    return (np.mean(bf[bv] == tn[bv]) if len(bv) else 0.0), len(bb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    pair, pck = load_pair()
    print(f"pair step={pck.get('step')} val={pck.get('val')}", flush=True)
    z = np.load(os.path.join(CACHE_DIR, "perms.npz"), allow_pickle=True)
    names_, inv_ = z["names"], z["inv"]
    gt = {n: inv_[i].astype(np.int64) for i, n in enumerate(names_)}
    _, val = train_val_split()
    r1, r5, med, bbp = [], [], [], []
    for nm in val[:args.n]:
        inv = gt[nm]; tr, td = true_nbrs(inv)
        frags = to_frags(load(os.path.join(TRAIN_INP, nm)))
        R, D = pairwise_scores_full(pair, frags, DEV)
        rk = np.concatenate([ranks(R, tr), ranks(D, td)])
        bR, _ = bb_precision(R, tr); bD, _ = bb_precision(D, td)
        r1.append(np.mean(rk == 1)); r5.append(np.mean(rk <= 5))
        med.append(np.median(rk)); bbp.append((bR + bD) / 2)
        print(f"{nm} R@1 {np.mean(rk==1):.3f} R@5 {np.mean(rk<=5):.3f} "
              f"med {np.median(rk):.0f} bb_prec {(bR+bD)/2:.3f}", flush=True)
    print(f"\nDIAG scores: R@1_mean={np.mean(r1):.3f} R@5_mean={np.mean(r5):.3f} "
          f"med_mean={np.mean(med):.1f} bb_prec_mean={np.mean(bbp):.3f}", flush=True)
    print("(need bb_prec_mean >> 0.48 to make placement solvable)", flush=True)


if __name__ == "__main__":
    main()
