"""Test whether full-graph row statistics identify missing board neighbours."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import GRID, WORK_ROOT


def load_rows(cache_dir: Path, images: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    image_ids: list[np.ndarray] = []
    for image in images:
        stored = np.load(cache_dir / f"image_{image:04d}_k64.npz")
        anchors = stored["anchors"].astype(np.int64)
        directions = stored["directions"].astype(np.int64)
        cells = stored["permutation"].astype(np.int64)[anchors]
        rows, cols = np.divmod(cells, GRID)
        missing = (
            ((directions == 0) & (rows == 0))
            | ((directions == 1) & (rows == GRID - 1))
            | ((directions == 2) & (cols == 0))
            | ((directions == 3) & (cols == GRID - 1))
        )
        features.append(stored["features"].astype(np.float32))
        labels.append(missing.astype(np.int64))
        image_ids.append(np.full(len(missing), image, dtype=np.int64))
    return np.concatenate(features), np.concatenate(labels), np.concatenate(image_ids)


def metrics(probability: np.ndarray, labels: np.ndarray, image_ids: np.ndarray) -> dict[str, float]:
    predicted = probability >= 0.5
    per_image_auc = [
        roc_auc_score(labels[image_ids == image], probability[image_ids == image])
        for image in np.unique(image_ids)
    ]
    return {
        "roc_auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "positive_rate": float(labels.mean()),
        "worst_image_roc_auc": float(min(per_image_auc)),
        "mean_image_roc_auc": float(np.mean(per_image_auc)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument("--fit-images", default="0,1")
    parser.add_argument("--heldout-images", default="50,51,52,53,54,55")
    parser.add_argument("--seed", type=int, default=9341)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "graph_boundary_signal_gate.json",
    )
    args = parser.parse_args()
    fit_images = [int(value) for value in args.fit_images.split(",")]
    heldout_images = [int(value) for value in args.heldout_images.split(",")]
    fit_x, fit_y, _ = load_rows(args.cache_dir, fit_images)
    held_x, held_y, held_image = load_rows(args.cache_dir, heldout_images)
    scaler = StandardScaler().fit(fit_x)
    logistic = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        max_iter=1000,
        random_state=args.seed,
    ).fit(scaler.transform(fit_x), fit_y)
    forest = ExtraTreesClassifier(
        n_estimators=320,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    ).fit(fit_x, fit_y)
    results = {
        "logistic": metrics(
            logistic.predict_proba(scaler.transform(held_x))[:, 1],
            held_y,
            held_image,
        ),
        "extra_trees": metrics(
            forest.predict_proba(held_x)[:, 1],
            held_y,
            held_image,
        ),
    }
    best_key = max(
        results,
        key=lambda key: (
            results[key]["roc_auc"],
            results[key]["average_precision"],
        ),
    )
    best = results[best_key]
    thresholds = {
        "roc_auc": 0.70,
        "average_precision": 0.12,
        "worst_image_roc_auc": 0.60,
    }
    checks = {key: best[key] >= value for key, value in thresholds.items()}
    report = {
        "experiment": "directional_missing_neighbour_boundary_signal",
        "status": "pass" if all(checks.values()) else "fail",
        "fit_images": fit_images,
        "heldout_images": heldout_images,
        "best_model": best_key,
        "best": best,
        "results": results,
        "thresholds": thresholds,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
