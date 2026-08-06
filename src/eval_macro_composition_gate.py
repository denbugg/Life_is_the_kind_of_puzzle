"""Branch F: radial/optical macro-position prior (group-held-out gate).

Branches A-E all tested *content* signals: generator-forensics parameters,
JPEG/resampling phase, local seam continuity, absolute-cell classification,
and 2D neighbour context.  All either found no transferable signal or found
one too weak to place tiles (content at 20x20/24x24-grid scale is locally too
homogeneous, see NEW_SOLUTION_RESEARCH.md branch E).

This tests a fundamentally different mechanism: not "what is drawn on this
tile" but "where in the physical camera frame was this tile, independent of
scene content".  This dataset is real event/meetup photography (confirmed by
source forensics): such photos are overwhelmingly subject-centred (a person
in focus near the middle, background toward the corners/edges), and any
photo taken with a real lens carries some vignetting and chromatic
aberration that both increase radially from the optical centre. None of
these are properties of *what* is depicted -- they are properties of *where*
a patch sits in the frame, so they might transfer across different photos in
a way absolute-cell content classification provably does not.

The target is deliberately coarser than exact row/col: a 4-way radial
quartile (distance from grid centre) and a border-vs-interior binary split.
Both label sets are a fixed function of grid geometry only (identical set of
cells every image), so there is no per-image leakage to worry about.

Two independent branches, exactly like the JPEG-phase gate: DIRTY features
come from one independent synthetic corruption only; CLEAN is a diagnostic
upper bound/comparator and never leaks into the dirty branch.

Continuation threshold: dirty radial-quartile accuracy >= 0.35 (chance 0.25)
OR dirty border balanced-accuracy >= 0.60 (chance 0.50).  This is a
deliberately modest bar for what is meant to be a *weak macro prior* --
useful only combined with the existing high-precision seam seeds, never a
placement solver on its own.

    python src/eval_macro_composition_gate.py --images 120 --val-images 24 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from config import GRID, NFRAG, SEED, TRAIN_TGT
from distort import distort_frags
from imgio import load, to_frags, train_val_split


FEATURE_NAMES = (
    "mean_luminance", "std_luminance", "sharpness_laplacian_energy",
    "chromatic_aberration_abs_diff", "chromatic_aberration_decorrelation",
    "saturation_mean", "saturation_std", "skin_tone_fraction",
)


def radial_labels() -> dict[str, np.ndarray]:
    """Fixed, content-independent labels: identical cell sets every image."""
    rows, cols = np.divmod(np.arange(NFRAG), GRID)
    centre = (GRID - 1) / 2.0
    distance = np.sqrt((rows - centre) ** 2 + (cols - centre) ** 2)
    edges = np.quantile(distance, (0.25, 0.5, 0.75))
    quartile = np.digitize(distance, edges).astype(np.int64)
    border = ((rows < 2) | (rows >= GRID - 2) | (cols < 2) | (cols >= GRID - 2))
    return {"distance": distance, "quartile": quartile, "border": border.astype(np.int64)}


LABELS = radial_labels()


def tile_features(tiles: np.ndarray) -> np.ndarray:
    """Per-tile scalar optical/compositional descriptors; content-agnostic where possible."""
    if tiles.ndim != 4 or tiles.shape[1:] != (20, 20, 3):
        raise ValueError(f"expected (N,20,20,3) tiles, got {tiles.shape}")
    x = tiles.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    mean_luminance = luminance.mean((1, 2))
    std_luminance = luminance.std((1, 2))

    def laplacian(channel: np.ndarray) -> np.ndarray:
        centre = channel[:, 1:-1, 1:-1]
        return (
            -4.0 * centre
            + channel[:, :-2, 1:-1] + channel[:, 2:, 1:-1]
            + channel[:, 1:-1, :-2] + channel[:, 1:-1, 2:]
        )

    lap_luminance = laplacian(luminance)
    sharpness = (lap_luminance ** 2).mean((1, 2))

    lap_r, lap_b = laplacian(r), laplacian(b)
    ca_diff = np.abs(lap_r - lap_b).mean((1, 2))
    numerator = (lap_r * lap_b).mean((1, 2))
    denominator = np.sqrt((lap_r ** 2).mean((1, 2)) * (lap_b ** 2).mean((1, 2))) + 1.0e-4
    ca_decorrelation = 1.0 - numerator / denominator

    max_channel = x.max(-1)
    min_channel = x.min(-1)
    saturation = np.where(max_channel > 1.0, (max_channel - min_channel) / max_channel, 0.0)
    saturation_mean = saturation.mean((1, 2))
    saturation_std = saturation.std((1, 2))

    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    skin = ((cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)).astype(np.float32).mean((1, 2))

    features = np.stack(
        (mean_luminance, std_luminance, sharpness, ca_diff, ca_decorrelation,
         saturation_mean, saturation_std, skin),
        axis=1,
    )
    if features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("feature schema mismatch")
    return np.nan_to_num(features, nan=0.0, posinf=1.0e6, neginf=-1.0e6).astype(np.float32)


def _extract(name: str, seed: int) -> dict[str, Any]:
    """Exact synthetic dirty/clean tile features at their true, known cell ids."""
    clean_tiles = to_frags(load(os.path.join(TRAIN_TGT, name)))
    rng = np.random.default_rng(seed)
    dirty_tiles = distort_frags(clean_tiles, rng)
    return {
        "dirty": tile_features(dirty_tiles),
        "clean": tile_features(clean_tiles),
        "quartile": LABELS["quartile"],
        "border": LABELS["border"],
    }


def _full_probability(model: Any, x: np.ndarray, classes: int) -> np.ndarray:
    probability = np.zeros((len(x), classes), dtype=np.float64)
    probability[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(x)
    return probability


def _metrics(model: Any, x: np.ndarray, labels: np.ndarray, classes: int) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score

    probability = _full_probability(model, x, classes)
    true_probability = probability[np.arange(len(labels)), labels]
    rank = 1 + (probability > true_probability[:, None]).sum(1)
    predicted = probability.argmax(1)
    return {
        "accuracy": float(np.mean(rank == 1)),
        "top2": float(np.mean(rank <= 2)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }


def _fit_branch(
    x_train: np.ndarray, x_val: np.ndarray,
    quartile_train: np.ndarray, quartile_val: np.ndarray,
    border_train: np.ndarray, border_val: np.ndarray,
    workers: int, seed: int,
) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier

    results: dict[str, Any] = {}
    tasks = {"quartile": (quartile_train, quartile_val, 4), "border": (border_train, border_val, 2)}
    for offset, (task, (train_labels, val_labels, classes)) in enumerate(tasks.items()):
        model = ExtraTreesClassifier(
            n_estimators=280, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", n_jobs=workers, random_state=seed + offset,
        ).fit(x_train, train_labels)
        importance = np.asarray(model.feature_importances_)
        order = np.argsort(importance)[::-1]
        results[task] = {
            **_metrics(model, x_val, val_labels, classes),
            "classes": classes,
            "chance": {"accuracy": 1.0 / classes, "top2": min(1.0, 2.0 / classes),
                       "balanced_accuracy": 1.0 / classes},
            "feature_importance": [
                {"name": FEATURE_NAMES[int(i)], "importance": float(importance[i])} for i in order
            ],
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=120)
    parser.add_argument("--val-images", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--report", type=Path, default=Path("E:/pazzle_work/gates/macro_composition_gate.json")
    )
    args = parser.parse_args()
    if args.images <= args.val_images or args.val_images < 1:
        raise ValueError("--images must be greater than --val-images >= 1")
    return args


def main() -> None:
    args = parse_args()
    train_names, val_names = train_val_split()
    all_names = train_names + val_names
    rng = np.random.default_rng(args.seed)
    if len(all_names) < args.images:
        raise ValueError(f"requested {args.images} images; only {len(all_names)} available")
    chosen = [all_names[int(i)] for i in rng.permutation(len(all_names))[: args.images]]
    val_items, train_items = chosen[: args.val_images], chosen[args.val_images :]
    ordered_names = train_items + val_items
    workers = max(1, args.workers)
    print(f"extracting {len(train_items)} train + {len(val_items)} validation images", flush=True)
    seeds = [args.seed + 1_000_003 * index for index in range(len(ordered_names))]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        extracted = list(pool.map(lambda pair: _extract(*pair), zip(ordered_names, seeds)))
    cut = len(train_items)

    def joined(key: str, items: list[dict[str, Any]]) -> np.ndarray:
        return np.concatenate([np.asarray(item[key]) for item in items])

    train, val = extracted[:cut], extracted[cut:]
    quartile_train, quartile_val = joined("quartile", train), joined("quartile", val)
    border_train, border_val = joined("border", train), joined("border", val)
    branches: dict[str, Any] = {}
    for kind in ("dirty", "clean"):
        branches[kind] = _fit_branch(
            joined(kind, train), joined(kind, val), quartile_train, quartile_val,
            border_train, border_val, workers, args.seed + (0 if kind == "dirty" else 100),
        )
    dirty = branches["dirty"]
    passed = dirty["quartile"]["accuracy"] >= 0.35 or dirty["border"]["balanced_accuracy"] >= 0.60
    report = {
        "experiment": "radial_optical_compositional_macro_prior",
        "hypothesis": (
            "subject-centred composition, depth-of-field focus falloff, vignetting, and "
            "chromatic aberration all increase radially from frame centre independent of "
            "scene content, unlike any content-based signal tested in branches A-E"
        ),
        "labels": "fixed function of grid geometry only (identical cell sets every image)",
        "train_only": False,
        "dirty_features_use_clean": False,
        "data": {
            "images": args.images, "train_images": len(train_items), "val_images": len(val_items),
            "group_split": True, "train_tiles": int(len(quartile_train)), "val_tiles": int(len(quartile_val)),
            "train_names": train_items, "val_names": val_items,
        },
        "feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
        "branches": branches,
        "gate": {
            "rule": "dirty quartile accuracy >= 0.35 (chance 0.25) OR dirty border balanced_accuracy >= 0.60 (chance 0.50)",
            "pass": bool(passed),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    verdict = "PASSED -> weak macro prior usable to anchor seed components" if passed else "FAILED -> close branch F"
    print(f"\n=== branch F gate {verdict} ===", flush=True)
    print(json.dumps({"dirty": dirty, "clean": branches["clean"], "gate": report["gate"]}, indent=2), flush=True)
    print(f"report saved to {args.report}", flush=True)


if __name__ == "__main__":
    main()
