"""Train-only existence gate for position signal in real corruption parameters.

This diagnostic aligns every real dirty input tile with its clean target tile
using ``perms.npz``.  It then removes as much content as practical and asks a
small tabular classifier whether the remaining affine/noise/blur/JPEG
fingerprint predicts the tile's clean row or column.  Clean targets are an
oracle unavailable at test time: a pass establishes that a generator signal
exists, not that it is already deployable.

Example::

    python src/eval_generator_forensics.py --images 120 --val-images 24
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from config import CACHE_DIR, FS, GRID, NFRAG, SEED, TRAIN_INP, TRAIN_TGT
from imgio import load, to_frags


FEATURE_NAMES = (
    "affine_slope_all", "affine_intercept_all",
    "affine_slope_r", "affine_slope_g", "affine_slope_b",
    "affine_intercept_r", "affine_intercept_g", "affine_intercept_b",
    "residual_std_all", "residual_std_r", "residual_std_g", "residual_std_b",
    "residual_mad", "residual_high_frequency_rms", "laplacian_log_ratio",
    "clip_low", "clip_high", "clip_low_r", "clip_low_g", "clip_low_b",
    "clip_high_r", "clip_high_g", "clip_high_b",
    "jpeg8_blockiness_delta", "jpeg8_blockiness_log_ratio",
)


def _affine(x: np.ndarray, y: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mx = x.mean(axis=axes)
    my = y.mean(axis=axes)
    cov = ((x - np.expand_dims(mx, axes[0])) * (y - np.expand_dims(my, axes[0]))).mean(axis=axes)
    var = ((x - np.expand_dims(mx, axes[0])) ** 2).mean(axis=axes)
    slope = cov / (var + 1e-4)
    return slope, my - slope * mx


def _blockiness(gray: np.ndarray) -> np.ndarray:
    """Eight-pixel boundary excess after per-tile contrast normalization."""
    z = (gray - gray.mean((1, 2), keepdims=True)) / (gray.std((1, 2), keepdims=True) + 1e-4)
    dx = np.abs(np.diff(z, axis=2))
    dy = np.abs(np.diff(z, axis=1))
    boundary = np.concatenate((dx[:, :, [7, 15]].reshape(len(z), -1),
                               dy[:, [7, 15], :].reshape(len(z), -1)), axis=1).mean(1)
    keep = np.ones(FS - 1, dtype=bool)
    keep[[7, 15]] = False
    interior = np.concatenate((dx[:, :, keep].reshape(len(z), -1),
                               dy[:, keep, :].reshape(len(z), -1)), axis=1).mean(1)
    return boundary - interior


def distortion_features(clean: np.ndarray, dirty: np.ndarray) -> np.ndarray:
    """Return content-minimal aligned dirty-vs-clean features for all tiles."""
    x = clean.astype(np.float32)
    y = dirty.astype(np.float32)
    flat_x, flat_y = x.reshape(len(x), -1), y.reshape(len(y), -1)
    slope_all, intercept_all = _affine(flat_x, flat_y, (1,))

    xc, yc = x.reshape(len(x), -1, 3), y.reshape(len(y), -1, 3)
    mx, my = xc.mean(1), yc.mean(1)
    slope_rgb = ((xc - mx[:, None]) * (yc - my[:, None])).mean(1)
    slope_rgb /= ((xc - mx[:, None]) ** 2).mean(1) + 1e-4
    intercept_rgb = my - slope_rgb * mx
    residual = y - (x * slope_rgb[:, None, None, :] + intercept_rgb[:, None, None, :])
    residual_flat = residual.reshape(len(x), -1)
    residual_median = np.median(residual_flat, axis=1)
    residual_mad = np.median(np.abs(residual_flat - residual_median[:, None]), axis=1)
    hf_sq = np.concatenate((np.diff(residual, axis=1).reshape(len(x), -1),
                            np.diff(residual, axis=2).reshape(len(x), -1)), axis=1) ** 2

    weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    gx, gy = x @ weights, y @ weights
    gx = (gx - gx.mean((1, 2), keepdims=True)) / (gx.std((1, 2), keepdims=True) + 1e-4)
    gy = (gy - gy.mean((1, 2), keepdims=True)) / (gy.std((1, 2), keepdims=True) + 1e-4)
    lap_x = (-4 * gx[:, 1:-1, 1:-1] + gx[:, :-2, 1:-1] + gx[:, 2:, 1:-1]
             + gx[:, 1:-1, :-2] + gx[:, 1:-1, 2:])
    lap_y = (-4 * gy[:, 1:-1, 1:-1] + gy[:, :-2, 1:-1] + gy[:, 2:, 1:-1]
             + gy[:, 1:-1, :-2] + gy[:, 1:-1, 2:])
    lap_ratio = np.log((np.sqrt(np.mean(lap_y ** 2, axis=(1, 2))) + 1e-4) /
                       (np.sqrt(np.mean(lap_x ** 2, axis=(1, 2))) + 1e-4))

    block_x, block_y = _blockiness(x @ weights), _blockiness(y @ weights)
    features = np.column_stack((
        slope_all, intercept_all, slope_rgb, intercept_rgb,
        residual_flat.std(1), residual.reshape(len(x), -1, 3).std(1),
        residual_mad, np.sqrt(hf_sq.mean(1)), lap_ratio,
        (y <= 0.5).mean((1, 2, 3)), (y >= 254.5).mean((1, 2, 3)),
        (y <= 0.5).mean((1, 2)), (y >= 254.5).mean((1, 2)),
        block_y - block_x, np.log((np.abs(block_y) + 1e-4) / (np.abs(block_x) + 1e-4)),
    ))
    return np.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)


def _extract(name: str, perm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if perm.shape != (NFRAG,) or np.unique(perm).size != NFRAG:
        raise ValueError(f"{name}: cache row is not a valid {NFRAG}-tile permutation")
    dirty = to_frags(load(os.path.join(TRAIN_INP, name)))
    clean = to_frags(load(os.path.join(TRAIN_TGT, name)))[perm.astype(np.int64)]
    return distortion_features(clean, dirty), perm // GRID, perm % GRID


def _prob_metrics(row_p: np.ndarray, col_p: np.ndarray,
                  row: np.ndarray, col: np.ndarray) -> dict[str, float]:
    n = len(row)
    row_rank = 1 + (row_p > row_p[np.arange(n), row, None]).sum(1)
    col_rank = 1 + (col_p > col_p[np.arange(n), col, None]).sum(1)
    joint = row_p[:, :, None] * col_p[:, None, :]
    true_joint = joint[np.arange(n), row, col]
    cell_rank = 1 + (joint > true_joint[:, None, None]).sum((1, 2))
    return {
        "row_top1": float(np.mean(row_rank <= 1)),
        "row_top3": float(np.mean(row_rank <= 3)),
        "col_top1": float(np.mean(col_rank <= 1)),
        "col_top3": float(np.mean(col_rank <= 3)),
        "cell_top1": float(np.mean(cell_rank <= 1)),
        "cell_recall_at_25": float(np.mean(cell_rank <= 25)),
        "cell_mrr": float(np.mean(1.0 / cell_rank)),
    }


def _full_proba(model: Any, x: np.ndarray) -> np.ndarray:
    out = np.zeros((len(x), GRID), dtype=np.float64)
    out[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(x)
    return out


def _slot_lookup(train_perm: np.ndarray, val_perm: np.ndarray) -> dict[str, float]:
    row_counts = np.ones((NFRAG, GRID), dtype=np.float64)
    col_counts = np.ones((NFRAG, GRID), dtype=np.float64)
    slots = np.arange(NFRAG)
    for perm in train_perm:
        np.add.at(row_counts, (slots, perm // GRID), 1)
        np.add.at(col_counts, (slots, perm % GRID), 1)
    row_p = row_counts / row_counts.sum(1, keepdims=True)
    col_p = col_counts / col_counts.sum(1, keepdims=True)
    tiled_row = np.tile(row_p, (len(val_perm), 1))
    tiled_col = np.tile(col_p, (len(val_perm), 1))
    return _prob_metrics(tiled_row, tiled_col,
                         (val_perm // GRID).reshape(-1), (val_perm % GRID).reshape(-1))


def _permutation_diagnostics(perms: np.ndarray, train_perm: np.ndarray,
                             val_perm: np.ndarray) -> dict[str, Any]:
    delta = np.diff(perms.astype(np.int32), axis=1)
    r, c = perms // GRID, perms % GRID
    manhattan = np.abs(np.diff(r, axis=1)) + np.abs(np.diff(c, axis=1))
    hashes = {p.tobytes() for p in perms}
    return {
        "valid_unique_permutation_fraction": float(np.mean([np.unique(p).size == NFRAG for p in perms])),
        "unique_permutation_rows": len(hashes),
        "duplicate_permutation_rows": int(len(perms) - len(hashes)),
        "forward_consecutive_input_slots_fraction": float(np.mean(delta == 1)),
        "either_direction_consecutive_input_slots_fraction": float(np.mean(np.abs(delta) == 1)),
        "grid_adjacent_input_slots_fraction": float(np.mean(manhattan == 1)),
        "input_slot_lookup": _slot_lookup(train_perm, val_perm),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=120, help="total train images")
    parser.add_argument("--val-images", type=int, default=24, help="held-out image groups")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=Path,
                        default=Path("E:/pazzle_work/gates/generator_forensics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.images <= args.val_images or args.val_images < 1:
        raise ValueError("--images must be greater than --val-images >= 1")
    from sklearn.ensemble import ExtraTreesClassifier

    cache_path = os.path.join(CACHE_DIR, "perms.npz")
    with np.load(cache_path, allow_pickle=True) as z:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in z["names"]]
        perms = np.asarray(z["perm"], dtype=np.int16)
        conf = np.asarray(z["conf"], dtype=np.float32) if "conf" in z else None
    cache_index = {name: i for i, name in enumerate(names)}
    available = [(n, p) for n, p in zip(names, perms)
                 if os.path.isfile(os.path.join(TRAIN_INP, n)) and os.path.isfile(os.path.join(TRAIN_TGT, n))]
    if len(available) < args.images:
        raise ValueError(f"requested {args.images} images, only {len(available)} cached pairs exist")
    rng = np.random.default_rng(args.seed)
    chosen_idx = rng.permutation(len(available))[:args.images]
    chosen = [available[int(i)] for i in chosen_idx]
    val = chosen[:args.val_images]
    train = chosen[args.val_images:]
    print(f"extracting {len(train)} train + {len(val)} val image groups", flush=True)
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        extracted = list(pool.map(lambda item: _extract(*item), train + val))
    cut = len(train)
    x_train = np.concatenate([e[0] for e in extracted[:cut]])
    row_train = np.concatenate([e[1] for e in extracted[:cut]]).astype(int)
    col_train = np.concatenate([e[2] for e in extracted[:cut]]).astype(int)
    x_val = np.concatenate([e[0] for e in extracted[cut:]])
    row_val = np.concatenate([e[1] for e in extracted[cut:]]).astype(int)
    col_val = np.concatenate([e[2] for e in extracted[cut:]]).astype(int)

    kwargs = dict(n_estimators=320, min_samples_leaf=2, max_features="sqrt",
                  class_weight="balanced", random_state=args.seed, n_jobs=workers)
    row_model = ExtraTreesClassifier(**kwargs).fit(x_train, row_train)
    col_model = ExtraTreesClassifier(**kwargs).fit(x_train, col_train)
    metrics = _prob_metrics(_full_proba(row_model, x_val), _full_proba(col_model, x_val),
                            row_val, col_val)
    passed = (metrics["row_top1"] >= 0.10 or metrics["col_top1"] >= 0.10
              or metrics["cell_recall_at_25"] >= 0.20)
    selected_perms = np.stack([p for _, p in chosen]).astype(np.int16)
    train_perms, val_perms = selected_perms[args.val_images:], selected_perms[:args.val_images]
    report = {
        "experiment": "generator_corruption_position_signal",
        "oracle_clean_features": True,
        "deployable": False,
        "interpretation": "existence gate only; clean aligned features are unavailable on test",
        "data": {
            "cache": cache_path, "images": args.images, "train_images": len(train),
            "val_images": len(val), "train_tiles": len(x_train), "val_tiles": len(x_val),
            "group_split": True, "train_names": [n for n, _ in train],
            "val_names": [n for n, _ in val],
            "cache_confidence_mean": None if conf is None else float(
                np.mean([conf[cache_index[n]] for n, _ in chosen])
            ),
        },
        "features": list(FEATURE_NAMES),
        "model": {"kind": "ExtraTreesClassifier", **kwargs, "n_jobs": workers},
        "chance": {"row_top1": 1 / GRID, "row_top3": 3 / GRID,
                   "col_top1": 1 / GRID, "col_top3": 3 / GRID,
                   "cell_top1": 1 / NFRAG, "cell_recall_at_25": 25 / NFRAG},
        "metrics": metrics,
        "permutation_diagnostics": _permutation_diagnostics(selected_perms, train_perms, val_perms),
        "gate": {
            "rule": "row_top1>=0.10 OR col_top1>=0.10 OR cell_recall_at_25>=0.20",
            "pass": bool(passed),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "gate": report["gate"], "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
