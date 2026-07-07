"""End-to-end SSIM on held-out train images (honest leaderboard estimate).
Reports: solve+restore (final), and ceilings (perfect placement +/- restore)."""
import os, time, argparse
import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim
from config import TRAIN_INP, TRAIN_TGT, CACHE_DIR
from imgio import load, to_frags, assemble, train_val_split
from pipeline import load_compat, load_restore, restore_full, process

DEV = "cuda"


def ssim(a, b):
    return sk_ssim(a, b, channel_axis=2, data_range=255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--iters", type=int, default=4_000_000)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--no_restore", action="store_true")
    args = ap.parse_args()

    compat, cck = load_compat()
    restore, rck = (None, None) if args.no_restore else load_restore()
    print(f"compat step={cck.get('step')} val={cck.get('val')}; "
          f"restore step={rck.get('step') if rck else None} val={rck.get('val') if rck else None}")

    z = np.load(os.path.join(CACHE_DIR, "perms.npz"), allow_pickle=True)
    names_, inv_ = z["names"], z["inv"]
    gt = {n: inv_[i].astype(np.int64) for i, n in enumerate(names_)}
    _, val = train_val_split()

    fin, sol, ceil_r, ceil_nr, accs = [], [], [], [], []
    t0 = time.time()
    for nm in val[:args.n]:
        frags = to_frags(load(os.path.join(TRAIN_INP, nm)))
        tgt = load(os.path.join(TRAIN_TGT, nm))
        out, place, assembled = process(frags, compat, restore,
                                        dict(iters=args.iters, restarts=args.restarts))
        inv = gt[nm]
        accs.append(float(np.mean(place == inv)))
        fin.append(ssim(tgt, out))
        sol.append(ssim(tgt, assembled))
        ceil_asm = assemble(frags, inv)
        ceil_nr.append(ssim(tgt, ceil_asm))
        ceil_r.append(ssim(tgt, restore_full(restore, ceil_asm)))
    n = len(fin)
    print(f"\n== N={n}  ({(time.time()-t0)/n:.1f}s/img) ==")
    print(f"placement acc          : {np.mean(accs):.3f}")
    print(f"FINAL solve+restore    : {np.mean(fin):.4f}   <-- leaderboard estimate")
    print(f"  solve only (no rest) : {np.mean(sol):.4f}")
    print(f"ceiling perfect+restore: {np.mean(ceil_r):.4f}")
    print(f"ceiling perfect no-rest: {np.mean(ceil_nr):.4f}")


if __name__ == "__main__":
    main()
