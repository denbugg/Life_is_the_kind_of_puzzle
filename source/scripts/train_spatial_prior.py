#!/usr/bin/env python3
"""Train and validate a weak content-to-position prior on whole-source splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesRegressor

from puzzle_assembly.spatial_prior import tile_spatial_features
from puzzle_denoise_v2.tiles import split_tiles_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--train-sources", type=int, default=512)
    parser.add_argument("--val-sources", type=int, default=64)
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_tiles(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return split_tiles_numpy(values)


def _dataset(root: Path, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    feature_blocks = []
    label_blocks = []
    positions = np.arange(576, dtype=np.int32)
    labels = np.stack(
        [positions // 24, positions % 24], axis=1
    ).astype(np.float32) / 23.0
    for name in names:
        tiles = _read_tiles(root / "train" / "targets" / name)
        feature_blocks.append(tile_spatial_features(tiles))
        label_blocks.append(labels)
    return np.concatenate(feature_blocks), np.concatenate(label_blocks)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction_cells = np.clip(np.rint(prediction * 23.0), 0, 23)
    target_cells = target * 23.0
    error = np.abs(prediction_cells - target_cells)
    predicted_border = (
        (prediction_cells[:, 0] == 0)
        | (prediction_cells[:, 0] == 23)
        | (prediction_cells[:, 1] == 0)
        | (prediction_cells[:, 1] == 23)
    )
    target_border = (
        (target_cells[:, 0] == 0)
        | (target_cells[:, 0] == 23)
        | (target_cells[:, 1] == 0)
        | (target_cells[:, 1] == 23)
    )
    return {
        "row_mae_cells": float(error[:, 0].mean()),
        "column_mae_cells": float(error[:, 1].mean()),
        "row_within_2": float(np.mean(error[:, 0] <= 2)),
        "column_within_2": float(np.mean(error[:, 1] <= 2)),
        "exact_position": float(np.mean(np.all(error == 0, axis=1))),
        "border_accuracy": float(np.mean(predicted_border == target_border)),
        "border_recall": float(np.mean(predicted_border[target_border])) if np.any(target_border) else 0.0,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    report_path = Path(args.report)
    if (output.exists() or report_path.exists()) and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_names = list(manifest["splits"]["train"][: args.train_sources])
    val_names = list(manifest["splits"]["val"][: args.val_sources])
    if len(train_names) != args.train_sources or len(val_names) != args.val_sources:
        raise SystemExit("requested source count exceeds manifest split")
    started = time.perf_counter()
    train_x, train_y = _dataset(Path(args.data_root), train_names)
    val_x, val_y = _dataset(Path(args.data_root), val_names)
    model = ExtraTreesRegressor(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=0.75,
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(train_x, train_y)
    prediction = model.predict(val_x)
    baseline = np.full_like(val_y, 0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output, compress=3)
    artifact_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "kind": "spatial_tile_prior",
        "seed": args.seed,
        "train_sources": train_names,
        "val_sources": val_names,
        "train_tiles": len(train_x),
        "val_tiles": len(val_x),
        "feature_count": train_x.shape[1],
        "model": {
            "class": type(model).__name__,
            "trees": args.trees,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
        },
        "validation": _metrics(prediction, val_y),
        "center_baseline": _metrics(baseline, val_y),
        "artifact": str(output),
        "artifact_sha256": artifact_hash,
        "seconds": time.perf_counter() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "spatial_prior_complete", **report["validation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
