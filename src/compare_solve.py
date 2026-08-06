"""Render side-by-side comparison panels for the current solve pipeline.

For each held-out validation image this saves one wide PNG:

    input | SA solve | buddies solve | denoise->solve (clean-first) |
    solve->NLM (assemble-first) | target

Every panel is annotated with its SSIM against the clean target, so two
solvers (ours vs a teammate's) can be compared frame by frame.  Ground truth
comes from the recovered permutation cache, exactly like eval_neighbour.py.

    python src/compare_solve.py --n 8
"""
import argparse
import os
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim

from config import CACHE_DIR, TRAIN_INP, TRAIN_TGT
from imgio import assemble, load, to_frags, train_val_split
from match_preprocess import apply_match_denoiser_np, load_match_denoiser
from pipeline import load_pair, nlm_restore
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve import pairwise_scores_full, solve_from_scores
from solve_buddies import solve_buddies_from_scores

DEV = "cuda"
HEADER = 30


def _panel(image: np.ndarray, label: str) -> np.ndarray:
    strip = np.zeros((HEADER, image.shape[1], 3), np.uint8)
    cv2.putText(strip, label, (8, HEADER - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, image])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--iters", type=int, default=500_000)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--bs_score", type=int, default=4096)
    ap.add_argument("--pair_tag", default="pair")
    ap.add_argument("--denoise_tag", default="matchden")
    ap.add_argument("--nlm_h", type=int, default=10)
    ap.add_argument("--out_dir", default=r"E:/pazzle_work/compare")
    args = ap.parse_args()

    pair, pck = load_pair(args.pair_tag)
    if pair is None:
        raise FileNotFoundError("no pair checkpoint found")
    print(f"pair step={pck.get('step')} val={pck.get('val')}", flush=True)
    denoiser, _ = load_match_denoiser(args.denoise_tag, device=DEV)
    if denoiser is None:
        print("no MatchDenoiser checkpoint; clean-first panel will reuse raw tiles", flush=True)

    z = np.load(os.path.join(CACHE_DIR, "perms.npz"), allow_pickle=True)
    names_, inv_, conf_ = z["names"], z["inv"], z["conf"]
    gt = {n: (inv_[i].astype(np.int64), conf_[i]) for i, n in enumerate(names_)}
    _, val = train_val_split()
    os.makedirs(args.out_dir, exist_ok=True)

    stats: dict[str, list[float]] = {k: [] for k in
                                     ("input", "sa", "buddies", "clean_first", "solve_nlm")}
    place_stats: dict[str, list[float]] = {k: [] for k in ("sa", "buddies", "clean_first")}

    t0 = time.time()
    for nm in val[: args.n]:
        inp = load(os.path.join(TRAIN_INP, nm))
        tgt = load(os.path.join(TRAIN_TGT, nm))
        frags = to_frags(inp)
        inv, conf = gt[nm]

        def solved(place: np.ndarray, tiles: np.ndarray) -> tuple[np.ndarray, float, float]:
            img = assemble(tiles, place)
            ss = sk_ssim(tgt, img, channel_axis=2, data_range=255)
            nacc = neighbour_accuracy(place, inv)[0]
            return img, ss, nacc

        # Our current pipeline scores RAW tiles (denoise-first measurably does
        # not help matching), and the two solvers share one score matrix.
        R, D = pairwise_scores_full(pair, frags, DEV, bs=args.bs_score)
        place_sa, _ = solve_from_scores(R, D, iters=args.iters, restarts=args.restarts)
        place_bud, _ = solve_buddies_from_scores(R, D)
        img_sa, ss_sa, na_sa = solved(place_sa, frags)
        img_bud, ss_bud, na_bud = solved(place_bud, frags)

        # Teammate's order: clean the tiles first, then match+glue the CLEAN tiles.
        if denoiser is not None:
            frags_dn = apply_match_denoiser_np(frags, denoiser, device=DEV)
        else:
            frags_dn = frags
        R2, D2 = pairwise_scores_full(pair, frags_dn, DEV, bs=args.bs_score)
        place_dn, _ = solve_from_scores(R2, D2, iters=args.iters, restarts=args.restarts)
        img_dn, ss_dn, na_dn = solved(place_dn, frags_dn)

        # Our order: glue raw tiles with the better solver, then NLM-restore.
        best_place = place_sa if ss_sa >= ss_bud else place_bud
        img_nlm = nlm_restore(assemble(frags, best_place), h=args.nlm_h)
        ss_nlm = sk_ssim(tgt, img_nlm, channel_axis=2, data_range=255)

        ss_inp = sk_ssim(tgt, inp, channel_axis=2, data_range=255)
        pa_sa = placement_accuracy(place_sa, inv, conf)[0]
        pa_bud = placement_accuracy(place_bud, inv, conf)[0]
        pa_dn = placement_accuracy(place_dn, inv, conf)[0]

        panels = [
            _panel(inp, f"input  SSIM={ss_inp:.3f}"),
            _panel(img_sa, f"SA solve  SSIM={ss_sa:.3f} place={pa_sa:.3f}"),
            _panel(img_bud, f"buddies  SSIM={ss_bud:.3f} place={pa_bud:.3f}"),
            _panel(img_dn, f"clean->solve  SSIM={ss_dn:.3f} place={pa_dn:.3f}"),
            _panel(img_nlm, f"solve->NLM  SSIM={ss_nlm:.3f}"),
            _panel(tgt, "target"),
        ]
        sheet = np.hstack(panels)
        out_path = os.path.join(args.out_dir, f"{os.path.splitext(nm)[0]}_compare.png")
        cv2.imwrite(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

        for key, value in (("input", ss_inp), ("sa", ss_sa), ("buddies", ss_bud),
                           ("clean_first", ss_dn), ("solve_nlm", ss_nlm)):
            stats[key].append(value)
        for key, value in (("sa", pa_sa), ("buddies", pa_bud), ("clean_first", pa_dn)):
            place_stats[key].append(value)
        print(f"{nm}: input={ss_inp:.3f} sa={ss_sa:.3f} (p={pa_sa:.3f} n={na_sa:.3f}) "
              f"buddies={ss_bud:.3f} (p={pa_bud:.3f} n={na_bud:.3f}) "
              f"clean_first={ss_dn:.3f} (p={pa_dn:.3f} n={na_dn:.3f}) "
              f"solve_nlm={ss_nlm:.3f} -> {out_path}", flush=True)

    print(f"\n== mean over {len(stats['input'])} val images "
          f"({(time.time() - t0) / max(1, len(stats['input'])):.1f}s/img) ==", flush=True)
    for key in ("input", "sa", "buddies", "clean_first", "solve_nlm"):
        line = f"SSIM {key:12s} {np.mean(stats[key]):.4f}"
        if key in place_stats:
            line += f"   place_acc {np.mean(place_stats[key]):.4f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
