"""Generate submission.zip: solve + restore every test image."""
import os, time, argparse, zipfile
import numpy as np
from PIL import Image
from config import TEST_DIR, SUB_DIR
from imgio import load, to_frags, list_test
from pipeline import load_compat, load_restore, load_pair, process


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit images (0=all)")
    ap.add_argument("--iters", type=int, default=5_000_000)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--no_restore", action="store_true")
    ap.add_argument("--use_pair", action="store_true", help="siamese top-K re-scored by pairwise")
    ap.add_argument("--full_pair", action="store_true", help="full NxN pairwise scoring (best)")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--save_dir", default="")
    args = ap.parse_args()

    compat, cck = load_compat()
    restore, rck = (None, None) if args.no_restore else load_restore()
    pair, pck = (load_pair() if (args.use_pair or args.full_pair) else (None, None))
    print(f"compat step={cck.get('step')}; restore step={rck.get('step') if rck else None}; "
          f"pair step={pck.get('step') if pck else None}", flush=True)

    names = list_test()
    if args.n:
        names = names[:args.n]
    out_zip = os.path.join(SUB_DIR, args.out)
    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    t0 = time.time()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, nm in enumerate(names):
            frags = to_frags(load(os.path.join(TEST_DIR, nm)))
            out, place, _ = process(
                frags, compat, restore,
                dict(iters=args.iters, restarts=args.restarts, full_pair=args.full_pair),
                pair=pair, rescore_kw=dict(K=args.K, alpha=args.alpha))
            img = Image.fromarray(out)
            tmp = os.path.join(SUB_DIR, "_tmp.png")
            img.save(tmp)
            zf.write(tmp, nm)
            if save_dir:
                img.save(os.path.join(save_dir, nm))
            if (i + 1) % 25 == 0 or i == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(names)}  {el/(i+1):.2f}s/img  eta {el/(i+1)*(len(names)-i-1)/60:.1f}min",
                      flush=True)
    os.path.exists(os.path.join(SUB_DIR, "_tmp.png")) and os.remove(os.path.join(SUB_DIR, "_tmp.png"))
    sz = os.path.getsize(out_zip) / 1e6
    print(f"wrote {out_zip}  ({len(names)} imgs, {sz:.1f} MB, {(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
