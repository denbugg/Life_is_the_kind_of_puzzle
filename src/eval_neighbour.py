"""Evaluate placement and neighbour accuracy for SA or buddies solver."""
import os
import time
import argparse
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
from config import TRAIN_INP, TRAIN_TGT, CACHE_DIR
from imgio import load, to_frags, assemble, train_val_split
from pipeline import load_pair
from solve import pairwise_scores_full, solve_from_scores
from placement_metrics import placement_accuracy, neighbour_accuracy
from match_preprocess import load_match_denoiser, preprocess_frags_np

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--solver", choices=("sa", "buddies"), default="sa")
    ap.add_argument("--preprocess", choices=("raw", "norm", "denoise", "denoise_norm"), default="raw")
    ap.add_argument("--denoise_tag", default="matchden")
    ap.add_argument("--pair_tag", default="pair")
    ap.add_argument("--iters", type=int, default=500_000)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--bs_score", type=int, default=4096)
    args = ap.parse_args()

    pair, pck = load_pair(args.pair_tag)
    if pair is None:
        raise FileNotFoundError("no pair checkpoint found")
    print(f"pair step={pck.get('step')} val={pck.get('val')}", flush=True)
    denoiser = None
    if args.preprocess in ("denoise", "denoise_norm"):
        denoiser, _ = load_match_denoiser(args.denoise_tag, device=DEV)
        if denoiser is None:
            raise FileNotFoundError("no matching denoiser checkpoint found")

    z = np.load(os.path.join(CACHE_DIR, "perms.npz"), allow_pickle=True)
    names_, inv_, conf_ = z["names"], z["inv"], z["conf"]  # materialize once; npz is lazy
    gt = {n: (inv_[i].astype(np.int64), conf_[i]) for i, n in enumerate(names_)}
    _, val = train_val_split()

    if args.solver == "buddies":
        from solve_buddies import solve_buddies_from_scores

    rows = []
    t0 = time.time()
    for nm in val[:args.n]:
        frags = to_frags(load(os.path.join(TRAIN_INP, nm)))
        score_frags = preprocess_frags_np(frags, args.preprocess, denoiser, DEV)
        R, D = pairwise_scores_full(pair, score_frags, DEV, bs=args.bs_score)
        if args.solver == "buddies":
            place, obj = solve_buddies_from_scores(R, D)
        else:
            place, obj = solve_from_scores(R, D, iters=args.iters, restarts=args.restarts)
        inv, conf = gt[nm]
        pacc, hi = placement_accuracy(place, inv, conf)
        nacc, nr, nd = neighbour_accuracy(place, inv)
        tgt = load(os.path.join(TRAIN_TGT, nm))
        ss = sk_ssim(tgt, assemble(frags, place), channel_axis=2, data_range=255)
        rows.append((pacc, hi, nacc, nr, nd, ss))
        print(f"{nm} place={pacc:.3f} hi={hi:.3f} neigh={nacc:.3f} "
              f"R={nr:.3f} D={nd:.3f} SSIM={ss:.3f} obj={obj:.0f}", flush=True)

    a = np.array(rows, np.float32)
    print(f"\n== N={len(rows)} solver={args.solver} preprocess={args.preprocess} ==")
    print(f"place_acc   {a[:,0].mean():.4f}")
    print(f"hi_acc      {a[:,1].mean():.4f}")
    print(f"neigh_acc   {a[:,2].mean():.4f}  right={a[:,3].mean():.4f} down={a[:,4].mean():.4f}")
    print(f"SSIM_solve  {a[:,5].mean():.4f}")
    print(f"time/img    {(time.time()-t0)/max(1,len(rows)):.2f}s")


if __name__ == "__main__":
    main()



