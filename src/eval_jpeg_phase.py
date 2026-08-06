"""Group-held-out gate for absolute JPEG/resampling phase in real tiles.

The dirty and clean branches are deliberately independent.  In particular,
the dirty classifier receives features of one dirty tile only; it never sees
the aligned clean tile or a paired residual.  The clean branch is a diagnostic
upper/comparator for phase already present in the source image.
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


CHANNELS = ("y", "cb", "cr")
TASKS = {"row_mod2": (0, 2), "col_mod2": (1, 2),
         "row_mod4": (0, 4), "col_mod4": (1, 4)}


def _feature_names() -> list[str]:
    names: list[str] = []
    for derivative in ("d1", "d2"):
        for axis in ("x", "y"):
            for boundary in range(1, FS):
                names.extend(f"{derivative}_{axis}_b{boundary}_{ch}" for ch in CHANNELS)
    for period in (8, 16):
        for axis in ("x", "y"):
            names.extend(f"block_p{period}_{axis}_phase{phase}" for phase in range(period))
    for axis in ("x", "y"):
        for k in range(1, 7):
            names.extend((f"boundary_fft_{axis}_k{k}_real", f"boundary_fft_{axis}_k{k}_imag"))
    for axis in ("x", "y"):
        for lag in (1, 2, 4, 8):
            names.extend(f"hf_autocorr_{axis}_lag{lag}_{ch}" for ch in CHANNELS)
    dct_bands = ("x_high", "y_high", "xy_high", "radial_high", "period8", "period16")
    for band in dct_bands:
        names.extend(f"dct_energy_{band}_{ch}" for ch in CHANNELS)
    for y, x in ((2, 0), (3, 0), (5, 0), (0, 2), (0, 3), (0, 5), (5, 5)):
        names.extend(f"dct_{y}_{x}_{ch}" for ch in CHANNELS)
    return names


FEATURE_NAMES = _feature_names()


def _normalized_ycbcr(tiles: np.ndarray) -> np.ndarray:
    rgb = tiles.astype(np.float32)
    transform = np.asarray(((0.299, 0.587, 0.114),
                            (-0.168736, -0.331264, 0.5),
                            (0.5, -0.418688, -0.081312)), dtype=np.float32)
    ycc = rgb @ transform.T
    return (ycc - ycc.mean((1, 2), keepdims=True)) / (ycc.std((1, 2), keepdims=True) + 1e-4)


def _phase_aggregates(profile: np.ndarray, period: int) -> np.ndarray:
    """Boundary strength for every phase, normalized by tile-wide strength."""
    coordinates = np.arange(1, FS)
    overall = profile.mean(1, keepdims=True) + 1e-4
    return np.column_stack([
        profile[:, coordinates % period == phase].mean(1) / overall[:, 0] - 1.0
        for phase in range(period)
    ])


def tile_features(tiles: np.ndarray) -> np.ndarray:
    """Single-tile phase features; no paired clean/dirty quantities enter."""
    from scipy.fft import dctn

    z = _normalized_ycbcr(tiles)
    d1x = np.abs(np.diff(z, axis=2)).mean(1)          # N, 19, C
    d1y = np.abs(np.diff(z, axis=1)).mean(2)
    d2x = np.abs(np.diff(np.gradient(z, axis=2), axis=2)).mean(1)
    d2y = np.abs(np.diff(np.gradient(z, axis=1), axis=1)).mean(2)
    parts: list[np.ndarray] = [d1x.reshape(len(z), -1), d1y.reshape(len(z), -1),
                               d2x.reshape(len(z), -1), d2y.reshape(len(z), -1)]

    for period in (8, 16):
        parts.extend((_phase_aggregates(d1x[:, :, 0], period),
                      _phase_aggregates(d1y[:, :, 0], period)))
    for profile in (d1x[:, :, 0], d1y[:, :, 0]):
        spectrum = np.fft.rfft(profile - profile.mean(1, keepdims=True), axis=1)
        scale = np.sqrt(np.sum(np.abs(spectrum[:, 1:]) ** 2, axis=1, keepdims=True)) + 1e-4
        selected = spectrum[:, 1:7] / scale
        parts.append(np.stack((selected.real, selected.imag), axis=2).reshape(len(z), -1))

    center = z[:, 1:-1, 1:-1]
    smooth = (center + z[:, :-2, 1:-1] + z[:, 2:, 1:-1]
              + z[:, 1:-1, :-2] + z[:, 1:-1, 2:]) / 5.0
    high = center - smooth
    for axis in (2, 1):
        for lag in (1, 2, 4, 8):
            if axis == 2:
                a, b = high[:, :, :-lag], high[:, :, lag:]
            else:
                a, b = high[:, :-lag], high[:, lag:]
            numerator = (a * b).mean((1, 2))
            denominator = np.sqrt((a * a).mean((1, 2)) * (b * b).mean((1, 2))) + 1e-4
            parts.append(numerator / denominator)

    dct = dctn(z, axes=(1, 2), norm="ortho")
    power = dct * dct
    yy, xx = np.meshgrid(np.arange(FS), np.arange(FS), indexing="ij")
    masks = (
        xx >= 8, yy >= 8, (xx >= 8) & (yy >= 8), xx + yy >= 16,
        ((xx >= 4) & (xx <= 6)) | ((yy >= 4) & (yy <= 6)),
        ((xx >= 2) & (xx <= 3)) | ((yy >= 2) & (yy <= 3)),
    )
    total = power[:, 1:, 1:].sum((1, 2)) + 1e-4
    parts.extend(power[:, mask].sum(1) / total for mask in masks)
    coefficient_scale = np.sqrt(power[:, 1:, 1:].mean((1, 2))) + 1e-4
    parts.extend(dct[:, y, x] / coefficient_scale
                 for y, x in ((2, 0), (3, 0), (5, 0), (0, 2), (0, 3), (0, 5), (5, 5)))
    features = np.concatenate(parts, axis=1)
    if features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f"feature schema mismatch: {features.shape[1]} != {len(FEATURE_NAMES)}")
    return np.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)


def _extract(name: str, perm: np.ndarray, conf_by_position: np.ndarray) -> dict[str, Any]:
    perm = perm.astype(np.int64)
    if perm.shape != (NFRAG,) or np.unique(perm).size != NFRAG:
        raise ValueError(f"{name}: invalid permutation")
    confidence = conf_by_position[perm]  # cache confidence is indexed by target cell
    keep = confidence >= 0.70
    dirty = to_frags(load(os.path.join(TRAIN_INP, name)))[keep]
    clean = to_frags(load(os.path.join(TRAIN_TGT, name)))[perm[keep]]
    cell = perm[keep]
    return {"dirty": tile_features(dirty), "clean": tile_features(clean),
            "row": cell // GRID, "col": cell % GRID, "kept": int(keep.sum())}


def _full_probability(model: Any, x: np.ndarray, classes: int) -> np.ndarray:
    probability = np.zeros((len(x), classes), dtype=np.float64)
    probability[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(x)
    return probability


def _metrics(model: Any, x: np.ndarray, labels: np.ndarray, classes: int) -> dict[str, float]:
    probability = _full_probability(model, x, classes)
    true_probability = probability[np.arange(len(labels)), labels]
    rank = 1 + (probability > true_probability[:, None]).sum(1)
    return {"accuracy": float(np.mean(rank == 1)), "top2": float(np.mean(rank <= 2))}


def _fit_branch(kind: str, x_train: np.ndarray, x_val: np.ndarray,
                row_train: np.ndarray, col_train: np.ndarray,
                row_val: np.ndarray, col_val: np.ndarray,
                workers: int, seed: int) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier

    results: dict[str, Any] = {}
    for offset, (task, (coordinate, modulo)) in enumerate(TASKS.items()):
        train_labels = (row_train if coordinate == 0 else col_train) % modulo
        val_labels = (row_val if coordinate == 0 else col_val) % modulo
        model = ExtraTreesClassifier(
            n_estimators=280, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", n_jobs=workers, random_state=seed + offset,
        ).fit(x_train, train_labels)
        importance = np.asarray(model.feature_importances_)
        top = np.argsort(importance)[-12:][::-1]
        results[task] = {
            **_metrics(model, x_val, val_labels, modulo),
            "classes": modulo,
            "chance": {"accuracy": 1.0 / modulo, "top2": min(1.0, 2.0 / modulo)},
            "top_features": [{"name": FEATURE_NAMES[int(i)], "importance": float(importance[i])}
                             for i in top],
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=120)
    parser.add_argument("--val-images", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--report", type=Path,
                        default=Path("E:/pazzle_work/gates/jpeg_phase_gate.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.images <= args.val_images or args.val_images < 1:
        raise ValueError("--images must be greater than --val-images >= 1")
    cache_path = os.path.join(CACHE_DIR, "perms.npz")
    with np.load(cache_path, allow_pickle=True) as z:
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in z["names"]]
        perms = np.asarray(z["perm"], dtype=np.int16)
        confidence = np.asarray(z["conf"], dtype=np.float32)
    available = [(n, p, c) for n, p, c in zip(names, perms, confidence)
                 if os.path.isfile(os.path.join(TRAIN_INP, n)) and os.path.isfile(os.path.join(TRAIN_TGT, n))]
    if len(available) < args.images:
        raise ValueError(f"requested {args.images} images; only {len(available)} aligned pairs exist")
    rng = np.random.default_rng(args.seed)
    chosen = [available[int(i)] for i in rng.permutation(len(available))[:args.images]]
    val_items, train_items = chosen[:args.val_images], chosen[args.val_images:]
    ordered_items = train_items + val_items
    workers = max(1, args.workers)
    print(f"extracting {len(train_items)} train + {len(val_items)} validation groups", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        extracted = list(pool.map(lambda item: _extract(*item), ordered_items))
    cut = len(train_items)

    def joined(key: str, items: list[dict[str, Any]]) -> np.ndarray:
        return np.concatenate([np.asarray(item[key]) for item in items])

    train, val = extracted[:cut], extracted[cut:]
    row_train, col_train = joined("row", train).astype(int), joined("col", train).astype(int)
    row_val, col_val = joined("row", val).astype(int), joined("col", val).astype(int)
    branches: dict[str, Any] = {}
    for kind in ("dirty", "clean"):
        branches[kind] = _fit_branch(
            kind, joined(kind, train), joined(kind, val), row_train, col_train,
            row_val, col_val, workers, args.seed + (0 if kind == "dirty" else 100),
        )
    dirty = branches["dirty"]
    passed = (dirty["row_mod2"]["accuracy"] >= 0.60 or dirty["col_mod2"]["accuracy"] >= 0.60
              or dirty["row_mod4"]["accuracy"] >= 0.35 or dirty["col_mod4"]["accuracy"] >= 0.35)
    report = {
        "experiment": "absolute_jpeg_resampling_phase",
        "train_only": True,
        "dirty_features_use_clean": False,
        "paired_residual_features": False,
        "confidence_filter": {"threshold": 0.70, "cache_indexing": "target_cell",
                              "mapping_to_input_tile": "confidence[perm[input_tile]]"},
        "data": {"cache": cache_path, "images": args.images,
                 "train_images": len(train_items), "val_images": len(val_items),
                 "group_split": True, "train_tiles": int(len(row_train)),
                 "val_tiles": int(len(row_val)),
                 "kept_tiles_per_image": {n: item["kept"]
                                          for (n, _, _), item in zip(ordered_items, extracted)},
                 "train_names": [n for n, _, _ in train_items],
                 "val_names": [n for n, _, _ in val_items]},
        "feature_count": len(FEATURE_NAMES), "feature_names": FEATURE_NAMES,
        "branches": branches,
        "gate": {"rule": "any DIRTY mod2 accuracy>=0.60 OR any DIRTY mod4 accuracy>=0.35",
                 "pass": bool(passed)},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dirty": dirty, "gate": report["gate"], "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
