"""Score the field diffusion end to end, in SSIM, against the bars that matter.

Placement during training is measured on 4x4 descriptors and is the right
training signal, but it is not the deliverable. This assembles the actual board:
the sampled picture assigns the 576 REAL fragments to cells by Hungarian, every
fragment is placed exactly once and none is altered, and the result is scored
against the clean target.

The bars, all measured on the same boards:
    flat fill        0.3514   what a constant beats us with today
    the competitor   0.38     reached honestly, by someone the user knows
    M471 at 32 RMSE  0.3902   what this model needs the picture to be worth
    M428 oracle      0.4292   the true picture at this resolution
    true layout      0.4734   every fragment in its own cell, still corrupted

Two arms. FREE is the prior on its own -- the sampler never sees the bag as a
constraint, only as conditioning. SNAPPED projects each denoising step onto the
bag once the picture has form, which is the data-consistency step of a diffusion
solver for an inverse problem; `prior_projection.py` measured that snapping from
the very start locks in an arbitrary permutation, so it begins late.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from build_field_cache import load_rgb, to_frags
from config import CACHE_DIR, CKPT_DIR, GRID as G, TRAIN_INP, TRAIN_TGT
from field_diffusion import (N, RES, FieldDiffusion, Schedule, cell_desc,
                             project, sample)

S = 20


def board(frags, order):
    img = np.zeros((G * S, G * S, 3), np.float32)
    for cell, f in enumerate(order):
        img[cell // G * S:(cell // G + 1) * S,
            cell % G * S:(cell % G + 1) * S] = frags[int(f)]
    return np.clip(img, 0, 255).astype(np.uint8)


def conformant(order, frags, drawn):
    """The organisers check by hand: all 576 present, once each, unaltered."""
    if sorted(int(x) for x in order) != list(range(N)):
        return False, "not a permutation of the 576 fragments"
    for cell, f in enumerate(order):
        y, x = divmod(cell, G)
        tile = drawn[y * S:(y + 1) * S, x * S:(x + 1) * S]
        if not np.array_equal(tile, np.clip(frags[int(f)], 0, 255
                                            ).astype(np.uint8)):
            return False, f"fragment {int(f)} altered at cell {cell}"
    return True, "all 576 placed once, pixels untouched"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="field_diff_best.pt")
    ap.add_argument("--cache", default="field_cache.npz")
    ap.add_argument("--boards", type=int, default=12)
    ap.add_argument("--sample-steps", type=int, default=100)
    ap.add_argument("--snap-from", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(Path(CKPT_DIR) / a.ckpt, map_location=dev,
                    weights_only=False)
    ka = ck["args"]
    model = FieldDiffusion(ka["d"], ka["layers"], ka["heads"],
                           ka["base"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    sched = Schedule(ka["steps"], dev)
    print(f"{a.ckpt}: trained placement {ck.get('placed', float('nan')):.4f}, "
          f"RMSE {ck.get('rmse', float('nan')):.2f}", flush=True)

    z = np.load(Path(CACHE_DIR) / a.cache)
    names, pic = z["names"], z["pic"]
    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    inv = blob["inv"]
    ev = np.arange(len(pic) - 300, len(pic))[:a.boards]

    rows = {k: [] for k in ("free", "snapped", "oracle picture", "true layout",
                            "flat fill")}
    place = {k: [] for k in rows}
    checked = None
    for i in ev:
        nm = str(names[i])
        frags = to_frags(load_rgb(Path(TRAIN_INP) / nm))[
            inv[i].astype(np.int64)].astype(np.float32)
        clean = load_rgb(Path(TRAIN_TGT) / nm)
        tiles = torch.from_numpy(frags[None]).to(dev)

        def score(tag, order):
            drawn = board(frags, order)
            rows[tag].append(ssim(clean, drawn, channel_axis=2,
                                  data_range=255))
            place[tag].append(float((np.asarray(order)
                                     == np.arange(N)).mean()))
            return drawn

        with torch.no_grad():
            for tag, snap in (("free", 1.01), ("snapped", a.snap_from)):
                x = sample(model, tiles, sched, a.sample_steps,
                           frags=[frags] if snap <= 1.0 else None,
                           snap_from=snap, device=dev)
                img = (x[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5
                drawn = score(tag, project(img, frags))
                if tag == "snapped" and checked is None:
                    checked = conformant(project(img, frags), frags, drawn)
        score("oracle picture", project(pic[i].astype(np.float64), frags))
        score("true layout", np.arange(N))
        flat = np.zeros_like(clean)
        flat[:] = clean.reshape(-1, 3).mean(0).astype(np.uint8)
        rows["flat fill"].append(ssim(clean, flat, channel_axis=2,
                                      data_range=255))
        place["flat fill"].append(0.0)
        print(f"board {i} done", flush=True)

    print(f"\n{len(ev)} held-out boards, {a.sample_steps} sampling steps")
    print(f"{'arm':>22} {'SSIM':>8} {'vs flat':>9} {'placed':>8}")
    bar = np.mean(rows["flat fill"])
    for tag in ("true layout", "oracle picture", "snapped", "free",
                "flat fill"):
        m = np.mean(rows[tag])
        print(f"{tag:>22} {m:8.4f} {m-bar:+9.4f} {np.mean(place[tag]):8.4f}")
    print(f"\nconformance: {checked[1] if checked else 'not checked'}")
    print("bars: competitor 0.38, M471 at 32 RMSE 0.3902, M428 oracle 0.4292")


if __name__ == "__main__":
    main()
