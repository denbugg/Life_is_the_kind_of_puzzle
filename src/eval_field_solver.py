"""Gate the absolute-field assembly before spending a long diffusion run.

The cache stores fragments in true-cell order only for evaluation.  The solver
is not given that order: it receives the predicted field and a bag of fragment
descriptors and must recover a Hungarian permutation.  The reported number is
therefore exact placement, with chance equal to 1/576.

Examples
--------
Oracle/tolerance curve used to verify M429/M471::

    python src/eval_field_solver.py --boards 8 --noise 0 16 32 64 \
        --modes raw zscore blend:0.25

The important deployment gate is not oracle performance but a learned field:
its held-out RMSE should be below roughly 32 and the resulting placement must
rise materially above the current 0.01 assembly path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import CACHE_DIR, CKPT_DIR, TRAIN_INP
from field_solver import components_from_score_tail, solve_field
from imgio import load, to_frags
from seam_cost import costs_from_models
from seam_embed import SeamEmbed


def expand_bag4(bag4: np.ndarray) -> np.ndarray:
    """Exact 4x4 cache descriptors as compatible synthetic 20x20 fragments."""
    return np.repeat(np.repeat(bag4.astype(np.float32), 5, axis=1), 5, axis=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="field_cache.npz")
    ap.add_argument("--boards", type=int, default=8)
    ap.add_argument("--noise", type=float, nargs="+", default=[0, 16, 32, 64])
    ap.add_argument("--modes", nargs="+", default=["raw", "zscore"])
    ap.add_argument("--matchers", nargs="*", default=[])
    ap.add_argument("--edge-keep", type=int, default=0)
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--offsets", type=int, default=96)
    ap.add_argument("--compare-unconstrained", action="store_true",
                    help="paired A/B against plain Hungarian on each field")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    z = np.load(Path(CACHE_DIR) / a.cache)
    pic, bag4, names = z["pic"], z["bag4"], z["names"]
    first = max(0, len(pic) - 300)
    ids = np.arange(first, min(len(pic), first + a.boards))
    rng = np.random.default_rng(a.seed)
    print(f"{len(ids)} held-out boards; chance={1/576:.6f}")

    models = []
    if a.matchers:
        for name in a.matchers:
            ck = torch.load(Path(CKPT_DIR) / name, map_location=a.device,
                            weights_only=False)
            ma = ck["args"]
            model = SeamEmbed(ma["ch"], ma["blocks"], ma["dim"], ma["strip"],
                              ma.get("head", "global"),
                              predict=any(k.startswith("pred.")
                                          for k in ck["model"])).to(a.device)
            model.load_state_dict(ck["model"])
            model.eval()
            models.append(model)
    if a.edge_keep and not models:
        ap.error("--edge-keep requires --matchers")

    restore = None
    if models:
        restore = np.load(Path(CACHE_DIR) / "restore_labels.npz",
                          allow_pickle=True)
    fragments, components = [], []
    for i in ids:
        if models:
            frags = to_frags(load(Path(TRAIN_INP) / str(names[i]))).astype(
                np.float32)[restore["inv"][i].astype(np.int64)]
            right, down = costs_from_models(models, frags)
            comps = components_from_score_tail(right, down, a.edge_keep)
        else:
            frags = expand_bag4(bag4[i])
            comps = []
        fragments.append(frags)
        components.append(comps)
    if models:
        print(f"edge_keep={a.edge_keep}; mean rigid components="
              f"{np.mean([len(x) for x in components]):.1f}")

    for sigma in a.noise:
        fields = []
        for i in ids:
            noise = rng.normal(0, sigma, pic[i].shape) if sigma else 0.0
            fields.append(np.clip(pic[i].astype(np.float32) + noise, 0, 255))
        actual_rmse = float(np.mean([
            np.sqrt(np.mean((f - pic[i].astype(np.float32)) ** 2))
            for f, i in zip(fields, ids)]))
        for mode in a.modes:
            placed, control = [], []
            for field, i, frags, comps in zip(fields, ids, fragments, components):
                layout, _ = solve_field(field, frags, mode=mode,
                                        components=comps, beam=a.beam,
                                        offsets=a.offsets)
                placed.append(float((layout == np.arange(576)).mean()))
                if a.compare_unconstrained and comps:
                    base, _ = solve_field(field, frags, mode=mode)
                    control.append(float((base == np.arange(576)).mean()))
            msg = (f"noise={sigma:5.1f} actual_rmse={actual_rmse:6.2f} "
                   f"mode={mode:>10s} placed={np.mean(placed):.4f} "
                   f"range=[{np.min(placed):.4f},{np.max(placed):.4f}]")
            if control:
                delta = np.asarray(placed) - np.asarray(control)
                se = (float(delta.std(ddof=1) / np.sqrt(len(delta)))
                      if len(delta) > 1 else float("nan"))
                msg += (f" control={np.mean(control):.4f} delta={delta.mean():+.4f} "
                        f"SE={se:.4f} up/down="
                        f"{int((delta > 0).sum())}/{int((delta < 0).sum())}")
            print(msg)


if __name__ == "__main__":
    main()
