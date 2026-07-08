"""Measure puzzle-solver placement accuracy + SSIM on held-out train images,
using recovered ground-truth arrangement. Also reports the perfect-solve ceiling."""
import os, time, argparse
import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim
from config import CKPT_DIR, TRAIN_INP, TRAIN_TGT, CACHE_DIR, NFRAG
from imgio import load, to_frags, assemble, train_val_split
from models import CompatNet
from solve import solve_image

DEV = "cuda"


def load_compat(tag="compat"):
    for name in (f"{tag}_best.pt", f"{tag}_last.pt"):
        p = os.path.join(CKPT_DIR, name)
        if os.path.exists(p):
            ck = torch.load(p, map_location=DEV)
            m = CompatNet().to(DEV); m.load_state_dict(ck["model"]); m.eval()
            print(f"loaded {name} step={ck.get('step')} val={ck.get('val')}")
            return m
    raise FileNotFoundError("no compat checkpoint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--iters", type=int, default=3_000_000)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--tscale", type=float, default=1.0)
    ap.add_argument("--tag", default="compat")
    ap.add_argument("--use_pair", action="store_true")
    ap.add_argument("--full_pair", action="store_true")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=3.0)
    args = ap.parse_args()

    model = None
    if args.full_pair:
        try:
            model = load_compat(args.tag)
        except FileNotFoundError:
            print("no compat checkpoint; using full_pair pairwise only")
    else:
        model = load_compat(args.tag)
    pair = None
    if args.use_pair or args.full_pair:
        from pipeline import load_pair
        pair, pck = load_pair()
        print(f"pair step={pck.get('step')} val={pck.get('val')}")
    z = np.load(os.path.join(CACHE_DIR, "perms.npz"), allow_pickle=True)
    names_, inv_, conf_ = z["names"], z["inv"], z["conf"]  # materialize once
    gt = {n: (inv_[i], conf_[i]) for i, n in enumerate(names_)}
    _, val = train_val_split()

    accs, hi_accs, s_solve, s_ceil = [], [], [], []
    t0 = time.time()
    for k, nm in enumerate(val[:args.n]):
        frags = to_frags(load(os.path.join(TRAIN_INP, nm)))
        tgt = load(os.path.join(TRAIN_TGT, nm))
        place, R, D, v = solve_image(frags, model, DEV, pair_model=pair,
                                     rescore_kw=dict(K=args.K, alpha=args.alpha),
                                     full_pair=args.full_pair, iters=args.iters,
                                     restarts=args.restarts, T_scale=args.tscale)
        inv, conf = gt[nm]
        inv = inv.astype(np.int64)
        acc = float(np.mean(place == inv))
        hi = conf > 0.6
        hi_acc = float(np.mean(place[hi] == inv[hi])) if hi.sum() else 0.0
        solved = assemble(frags, place)
        ceil = assemble(frags, inv)
        ss = sk_ssim(tgt, solved, channel_axis=2, data_range=255)
        sc = sk_ssim(tgt, ceil, channel_axis=2, data_range=255)
        accs.append(acc); hi_accs.append(hi_acc); s_solve.append(ss); s_ceil.append(sc)
        print(f"{nm} place_acc={acc:.3f} hi_acc={hi_acc:.3f} SSIM_solve={ss:.3f} "
              f"ceil={sc:.3f} obj={v:.0f}", flush=True)
    print(f"\n== N={len(accs)} ==")
    print(f"place_acc   mean={np.mean(accs):.3f}")
    print(f"hi_acc      mean={np.mean(hi_accs):.3f}  (conf>0.6 positions)")
    print(f"SSIM solve  mean={np.mean(s_solve):.4f}  (no restore)")
    print(f"SSIM ceil   mean={np.mean(s_ceil):.4f}  (perfect solve, no restore)")
    print(f"time {(time.time()-t0)/max(1,len(accs)):.2f}s/img")


if __name__ == "__main__":
    main()
