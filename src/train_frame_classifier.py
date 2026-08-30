"""Fit a bag-relative frame classifier on frozen seam evidence."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from config import CACHE_DIR, TRAIN_INP
from eval_strong_crop_solver import fragments, load_matcher, load_rgb
from frame_classifier import frame_features, frame_labels
from seam_cost import costs_from_models


def make_crop(labels, index, side, rng):
    name = str(labels["names"][index])
    inv = labels["inv"][index].astype(np.int64)
    tiles = fragments(load_rgb(Path(TRAIN_INP) / name))[inv]
    y, x = int(rng.integers(25 - side)), int(rng.integers(25 - side))
    cells = ((np.arange(side)[:, None] + y) * 24
             + np.arange(side)[None, :] + x).reshape(-1)
    perm = rng.permutation(side * side)
    return tiles[cells][perm].astype(np.float32), perm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--boards", type=int, default=384)
    ap.add_argument("--eval-boards", type=int, default=64)
    ap.add_argument("--matchers", default="seam_embed_v3.pt,seam_embed_local.pt")
    ap.add_argument("--iterations", type=int, default=160)
    ap.add_argument("--seed", type=int, default=1357)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_matcher(v.strip(), dev) for v in a.matchers.split(",")]
    labels = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    cut = len(labels["names"]) - 300
    rng = np.random.default_rng(a.seed)

    def collect(indices, tag):
        xx, yy = [], []
        base_y = frame_labels(a.side).reshape(4, -1)
        for k, idx in enumerate(indices):
            raw, perm = make_crop(labels, int(idx), a.side, rng)
            ch, cv = costs_from_models(models, raw, device=dev)
            score_h, score_v = -ch, -cv
            stats = np.concatenate([raw.mean((1, 2)), raw.std((1, 2))], 1) / 255.0
            xx.append(frame_features(score_h, score_v, stats))
            # Feature rows are direction-major and tiles are permuted.  Label
            # for input tile i is the boundary status of original crop cell perm[i].
            yy.append(np.concatenate([base_y[d, perm] for d in range(4)]))
            if (k + 1) % 32 == 0:
                print(f"{tag} {k+1}/{len(indices)}", flush=True)
        return np.concatenate(xx), np.concatenate(yy)

    train_ids = rng.choice(np.arange(cut), min(a.boards, cut), replace=False)
    eval_ids = np.arange(cut, cut + min(a.eval_boards, 300))
    xtr, ytr = collect(train_ids, "train")
    xev, yev = collect(eval_ids, "eval")
    model = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=a.iterations, max_leaf_nodes=31,
        l2_regularization=1.0, class_weight="balanced", random_state=a.seed)
    model.fit(xtr, ytr)
    prob = model.predict_proba(xev)[:, 1]
    print(f"held-out ROC-AUC {roc_auc_score(yev, prob):.6f}  "
          f"AP {average_precision_score(yev, prob):.6f}  "
          f"base rate {yev.mean():.6f}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"model": model, "side": a.side,
                     "matchers": a.matchers, "topk": 16}, f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
