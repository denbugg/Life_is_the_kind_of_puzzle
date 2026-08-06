"""Compare scorer quality on raw/normalized/denoised fragments."""
import os
import argparse
import numpy as np
from config import GRID, NFRAG, TRAIN_INP, CACHE_DIR
from imgio import load, to_frags, train_val_split
from pipeline import load_pair
from solve import pairwise_scores_full
from match_preprocess import load_match_denoiser, preprocess_frags_np

DEV = "cuda"


def true_nbrs(inv):
    f2p = np.empty(NFRAG, int)
    f2p[inv] = np.arange(NFRAG)
    tr = -np.ones(NFRAG, int)
    td = -np.ones(NFRAG, int)
    for a in range(NFRAG):
        p = f2p[a]
        if p % GRID != GRID - 1:
            tr[a] = inv[p + 1]
        if p // GRID != GRID - 1:
            td[a] = inv[p + GRID]
    return tr, td


def ranks(M, tn):
    keep = np.where(tn >= 0)[0]
    Mk = M[keep].copy()
    Mk[np.arange(len(keep)), keep] = -1e30
    tn_score = M[keep, tn[keep]]
    return (Mk > tn_score[:, None]).sum(1) + 1


def bb_precision(M, tn):
    Mm = M.copy()
    np.fill_diagonal(Mm, -1e30)
    bf = Mm.argmax(1)
    bb = np.where(Mm.argmax(0)[bf] == np.arange(NFRAG))[0]
    bv = bb[tn[bb] >= 0]
    return (float(np.mean(bf[bv] == tn[bv])) if len(bv) else 0.0), len(bb)


def diag_one(R, D, inv):
    tr, td = true_nbrs(inv)
    rk = np.concatenate([ranks(R, tr), ranks(D, td)])
    bR, nR = bb_precision(R, tr)
    bD, nD = bb_precision(D, td)
    return dict(r1=float(np.mean(rk == 1)), r5=float(np.mean(rk <= 5)),
                r25=float(np.mean(rk <= 25)), med=float(np.median(rk)),
                bb=float((bR + bD) * 0.5), bbR=bR, bbD=bD, nbbR=nR, nbbD=nD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--modes", default="raw,norm")
    ap.add_argument("--denoise_tag", default="matchden")
    ap.add_argument("--pair_tag", default="pair")
    ap.add_argument("--bs_score", type=int, default=4096)
    args = ap.parse_args()

    pair, pck = load_pair(args.pair_tag)
    if pair is None:
        raise FileNotFoundError("no pair checkpoint found")
    print(f"pair step={pck.get('step')} val={pck.get('val')}", flush=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    denoiser = None
    if any(m in ("denoise", "denoise_norm") for m in modes):
        denoiser, _ = load_match_denoiser(args.denoise_tag, device=DEV)
        if denoiser is None:
            raise FileNotFoundError("no matching denoiser checkpoint found")

    z = np.load(os.path.join(CACHE_DIR, "perms.npz"), allow_pickle=True)
    names_, inv_ = z["names"], z["inv"]  # materialize once; npz is lazy
    gt = {n: inv_[i].astype(np.int64) for i, n in enumerate(names_)}
    _, val = train_val_split()
    acc = {m: [] for m in modes}

    for nm in val[:args.n]:
        inv = gt[nm]
        frags = to_frags(load(os.path.join(TRAIN_INP, nm)))
        parts = []
        for mode in modes:
            sf = preprocess_frags_np(frags, mode, denoiser, DEV)
            R, D = pairwise_scores_full(pair, sf, DEV, bs=args.bs_score)
            d = diag_one(R, D, inv)
            acc[mode].append(d)
            parts.append(f"{mode}:R@1={d['r1']:.3f} R@5={d['r5']:.3f} "
                         f"med={d['med']:.0f} bb={d['bb']:.3f}")
        print(nm + "  " + " | ".join(parts), flush=True)

    print("\n== SCORE SUMMARY ==")
    for mode in modes:
        rows = acc[mode]
        print(f"{mode:12s} R@1={np.mean([r['r1'] for r in rows]):.3f} "
              f"R@5={np.mean([r['r5'] for r in rows]):.3f} "
              f"R@25={np.mean([r['r25'] for r in rows]):.3f} "
              f"med={np.mean([r['med'] for r in rows]):.1f} "
              f"bb_prec={np.mean([r['bb'] for r in rows]):.3f}", flush=True)


if __name__ == "__main__":
    main()



