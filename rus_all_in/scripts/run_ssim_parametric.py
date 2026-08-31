#!/usr/bin/env python3
"""Train and evaluate target-free SSIM-parametric constant predictors.

The runner is intentionally unable to select ``holdout`` or ``test``.  It fits
only on the frozen manifest's train split and evaluates a predeclared roster on
calibration-48.  For every calibration board all prediction colours and hashes
are frozen before the target file is decoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    IMAGE_SIZE,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
)
from aiijc_puzzle.ssim_parametric import (
    extract_invariant_features,
    feature_names,
    input_median_rgb,
    paired_bootstrap_interval,
    render_constant_rgb,
    ssim_optimal_constant_rgb,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_INPUTS = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
DEFAULT_TARGETS = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ssim-parametric"

BASELINE = "input_median_rgb"
TARGET_FREE_ROSTER = (
    BASELINE,
    "ridge_oracle_residual",
    "extra_trees_leaf2_residual",
    "extra_trees_leaf8_residual",
    "histgb_oracle_residual",
    "ensemble_oracle_residual",
    "ensemble_oracle_residual_shrink50",
    "ensemble_oracle_residual_guarded",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-limit", type=int, default=5_600)
    parser.add_argument("--calibration-limit", type=int, default=48)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--trees", type=int, default=240)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser.parse_args()


def load_rgb_strict(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError(f"expected strict RGB PNG, got {image.format} {image.mode}: {path}")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected 480x480 image, got {image.size}: {path}")
        return np.asarray(image, dtype=np.uint8)


def raw_prediction_sha256(prediction: np.ndarray) -> str:
    value = np.asarray(prediction)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError(f"invalid prediction {value.dtype} {value.shape}")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def records_digest(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{record['filename']}\0{record['input_sha256']}\0{record['target_sha256']}"
        for record in records
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def training_row(
    record: dict[str, Any], inputs: Path, targets: Path
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    filename = str(record["filename"])
    input_path = inputs / filename
    target_path = targets / filename
    if sha256_file(input_path) != record["input_sha256"]:
        raise ValueError(f"training input hash mismatch: {filename}")
    input_image = load_rgb_strict(input_path)
    # Target decoding is allowed here because this record belongs to manifest train.
    if sha256_file(target_path) != record["target_sha256"]:
        raise ValueError(f"training target hash mismatch: {filename}")
    target = load_rgb_strict(target_path)
    return (
        filename,
        extract_invariant_features(input_image),
        input_median_rgb(input_image),
        ssim_optimal_constant_rgb(target),
    )


def build_training_cache(
    records: list[dict[str, Any]],
    *,
    inputs: Path,
    targets: Path,
    workers: int,
) -> dict[str, np.ndarray]:
    rows: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(training_row, record, inputs, targets): str(record["filename"])
            for record in records
        }
        for done, future in enumerate(as_completed(futures), start=1):
            filename, features, baseline, oracle = future.result()
            rows[filename] = (features, baseline, oracle)
            if done % 100 == 0 or done == len(futures):
                print(
                    json.dumps(
                        {
                            "phase": "training_cache",
                            "done": done,
                            "total": len(futures),
                            "runtime_seconds": perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    names = np.asarray([str(record["filename"]) for record in records])
    return {
        "filenames": names,
        "features": np.stack([rows[name][0] for name in names]),
        "baseline_rgb": np.stack([rows[name][1] for name in names]),
        "oracle_rgb": np.stack([rows[name][2] for name in names]),
    }


def load_or_build_cache(
    path: Path,
    records: list[dict[str, Any]],
    *,
    inputs: Path,
    targets: Path,
    workers: int,
    force: bool,
) -> dict[str, np.ndarray]:
    expected_records_digest = records_digest(records)
    expected_features_digest = hashlib.sha256("\n".join(feature_names()).encode()).hexdigest()
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as archive:
            cached = {name: archive[name] for name in archive.files}
        metadata = json.loads(str(cached.pop("metadata")))
        if metadata != {
            "schema": "aiijc-ssim-parametric-train-cache-v1",
            "records_digest": expected_records_digest,
            "feature_schema_sha256": expected_features_digest,
        }:
            raise ValueError(f"training cache contract mismatch: {path}")
        return cached

    cache = build_training_cache(records, inputs=inputs, targets=targets, workers=workers)
    metadata = {
        "schema": "aiijc-ssim-parametric-train-cache-v1",
        "records_digest": expected_records_digest,
        "feature_schema_sha256": expected_features_digest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata, sort_keys=True), **cache)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return cache


def fit_models(
    features: np.ndarray,
    baseline_rgb: np.ndarray,
    oracle_rgb: np.ndarray,
    *,
    trees: int,
) -> dict[str, Any]:
    residual = oracle_rgb - baseline_rgb
    ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 5, 15), cv=5, scoring="neg_mean_squared_error"),
    )
    extra_leaf2 = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=0.75,
        n_jobs=-1,
        random_state=20260829,
    )
    extra_leaf8 = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=8,
        max_features=1.0,
        n_jobs=-1,
        random_state=20260830,
    )
    histgb = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            learning_rate=0.055,
            max_iter=180,
            max_leaf_nodes=23,
            min_samples_leaf=18,
            l2_regularization=5.0,
            early_stopping=True,
            random_state=20260829,
        ),
        n_jobs=3,
    )
    models = {
        "ridge": ridge,
        "extra_leaf2": extra_leaf2,
        "extra_leaf8": extra_leaf8,
        "histgb": histgb,
    }
    for name, model in models.items():
        started = perf_counter()
        model.fit(features, residual)
        print(
            json.dumps(
                {"phase": "fit", "model": name, "runtime_seconds": perf_counter() - started}
            ),
            flush=True,
        )
    return models


def predict_roster(
    features: np.ndarray, baseline_rgb: np.ndarray, models: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Predict the complete predeclared target-free colour roster."""

    features_2d = np.asarray(features, dtype=np.float64).reshape(1, -1)
    baseline = np.asarray(baseline_rgb, dtype=np.float64)
    residuals = {
        name: np.asarray(model.predict(features_2d)[0], dtype=np.float64)
        for name, model in models.items()
    }
    component = np.stack(
        (residuals["extra_leaf2"], residuals["extra_leaf8"], residuals["histgb"])
    )
    ensemble = component.mean(axis=0)
    disagreement = component.std(axis=0)
    # The guard is fixed before calibration: uncertain channel corrections are
    # attenuated, while consensus corrections are left almost intact.
    reliability = np.clip(1.0 - disagreement / 24.0, 0.20, 1.0)
    colors = {
        BASELINE: baseline,
        "ridge_oracle_residual": baseline + residuals["ridge"],
        "extra_trees_leaf2_residual": baseline + residuals["extra_leaf2"],
        "extra_trees_leaf8_residual": baseline + residuals["extra_leaf8"],
        "histgb_oracle_residual": baseline + residuals["histgb"],
        "ensemble_oracle_residual": baseline + ensemble,
        "ensemble_oracle_residual_shrink50": baseline + 0.50 * ensemble,
        "ensemble_oracle_residual_guarded": baseline + reliability * ensemble,
    }
    if tuple(colors) != TARGET_FREE_ROSTER:
        raise RuntimeError("target-free roster drifted from its frozen declaration")
    return {name: np.clip(color, 0.0, 255.0) for name, color in colors.items()}


def calibration_evaluation(
    records: list[dict[str, Any]],
    *,
    inputs: Path,
    targets: Path,
    models: dict[str, Any],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        filename = str(record["filename"])
        input_path = inputs / filename
        target_path = targets / filename
        if sha256_file(input_path) != record["input_sha256"]:
            raise ValueError(f"calibration input hash mismatch: {filename}")
        input_image = load_rgb_strict(input_path)

        # Leakage firewall: the complete roster is materialised and hashed
        # before even validating or decoding the paired calibration target.
        features = extract_invariant_features(input_image)
        baseline = input_median_rgb(input_image)
        colors = predict_roster(features, baseline, models)
        predictions = {name: render_constant_rgb(color) for name, color in colors.items()}
        prediction_hashes = {
            name: raw_prediction_sha256(prediction) for name, prediction in predictions.items()
        }

        if sha256_file(target_path) != record["target_sha256"]:
            raise ValueError(f"calibration target hash mismatch: {filename}")
        target = load_rgb_strict(target_path)
        scores = {
            name: contest_ssim(target, prediction) for name, prediction in predictions.items()
        }
        oracle_rgb = ssim_optimal_constant_rgb(target)
        oracle_score = contest_ssim(target, render_constant_rgb(oracle_rgb))
        rows.append(
            {
                "index": index,
                "filename": filename,
                "input_sha256": record["input_sha256"],
                "target_sha256": record["target_sha256"],
                "predictions_frozen_before_target_decode": True,
                "feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                "baseline_rgb": baseline.tolist(),
                "predicted_rgb_continuous": {
                    name: color.tolist() for name, color in colors.items()
                },
                "predicted_rgb_uint8": {
                    name: prediction[0, 0].tolist() for name, prediction in predictions.items()
                },
                "prediction_sha256": prediction_hashes,
                "ssim": scores,
                "posthoc_oracle": {
                    "not_inference_eligible": True,
                    "rgb_continuous": oracle_rgb.tolist(),
                    "ssim": oracle_score,
                },
            }
        )
        print(
            json.dumps(
                {
                    "phase": "calibration",
                    "done": index + 1,
                    "total": len(records),
                    "filename": filename,
                    "baseline_ssim": scores[BASELINE],
                }
            ),
            flush=True,
        )

    baseline_scores = np.asarray([row["ssim"][BASELINE] for row in rows])
    aggregate: dict[str, Any] = {}
    for variant in TARGET_FREE_ROSTER:
        scores = np.asarray([row["ssim"][variant] for row in rows])
        interval = paired_bootstrap_interval(
            scores - baseline_scores,
            replicates=bootstrap_replicates,
            seed=EXPERIMENT_SUBSET_SEED,
        )
        aggregate[variant] = {
            "mean_ssim": float(scores.mean()),
            "std_ssim": float(scores.std()),
            "median_ssim": float(np.median(scores)),
            "paired_vs_input_median": {
                "mean": interval.mean,
                "lower_95": interval.lower_95,
                "upper_95": interval.upper_95,
                "wins": interval.wins,
                "count": interval.count,
            },
            "promotion_gate": bool(interval.mean > 0.005 and interval.lower_95 > 0.0),
        }
    oracle_scores = np.asarray([row["posthoc_oracle"]["ssim"] for row in rows])
    oracle_interval = paired_bootstrap_interval(
        oracle_scores - baseline_scores,
        replicates=bootstrap_replicates,
        seed=EXPERIMENT_SUBSET_SEED,
    )
    aggregate["posthoc_oracle_constant_upper_bound"] = {
        "not_inference_eligible": True,
        "mean_ssim": float(oracle_scores.mean()),
        "paired_vs_input_median": {
            "mean": oracle_interval.mean,
            "lower_95": oracle_interval.lower_95,
            "upper_95": oracle_interval.upper_95,
            "wins": oracle_interval.wins,
            "count": oracle_interval.count,
        },
    }
    eligible = [name for name in TARGET_FREE_ROSTER if name != BASELINE]
    champion = max(eligible, key=lambda name: aggregate[name]["mean_ssim"])
    return {
        "champion": champion,
        "champion_passes_promotion_gate": aggregate[champion]["promotion_gate"],
        "aggregate": aggregate,
        "per_board": rows,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.train_limit <= 0 or args.train_limit > 5_600:
        raise ValueError("train-limit must be in [1, 5600]")
    if args.calibration_limit <= 0 or args.calibration_limit > 700:
        raise ValueError("calibration-limit must be in [1, 700]")
    if args.workers <= 0 or args.trees <= 0:
        raise ValueError("workers and trees must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if compute_protocol_digest(manifest) != manifest.get("protocol_digest"):
        raise ValueError("validation manifest digest mismatch")
    train_records = [
        dict(record)
        for record in select_manifest_records(
            manifest,
            "train",
            limit=args.train_limit,
            seed=EXPERIMENT_SUBSET_SEED,
            namespace=EXPERIMENT_SUBSET_NAMESPACE,
        )
    ]
    calibration_records = [
        dict(record)
        for record in select_manifest_records(
            manifest,
            "calibration",
            limit=args.calibration_limit,
            seed=EXPERIMENT_SUBSET_SEED,
            namespace=EXPERIMENT_SUBSET_NAMESPACE,
        )
    ]
    output_dir = args.output_dir.resolve()
    cache_path = output_dir / f"train-cache-{args.train_limit}.npz"
    cache = load_or_build_cache(
        cache_path,
        train_records,
        inputs=args.inputs.resolve(),
        targets=args.targets.resolve(),
        workers=args.workers,
        force=args.force_cache,
    )
    started = perf_counter()
    models = fit_models(
        cache["features"],
        cache["baseline_rgb"],
        cache["oracle_rgb"],
        trees=args.trees,
    )
    result = calibration_evaluation(
        calibration_records,
        inputs=args.inputs.resolve(),
        targets=args.targets.resolve(),
        models=models,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    ridge_alpha = float(models["ridge"].named_steps["ridgecv"].alpha_)
    model_path = output_dir / f"models-train{args.train_limit}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema": "aiijc-ssim-parametric-model-v1",
            "feature_names": feature_names(),
            "train_records_digest": records_digest(train_records),
            "models": models,
        },
        model_path,
        compress=3,
    )
    report = {
        "schema": "aiijc-ssim-parametric-calibration-v1",
        "method_family": "permutation-invariant input statistics to SSIM-optimal constant RGB",
        "inference_target_access": False,
        "fit_target_scope": "manifest train only",
        "evaluation_split": "calibration",
        "holdout_access": False,
        "test_access": False,
        "protocol_digest": manifest["protocol_digest"],
        "selection_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selection_seed": EXPERIMENT_SUBSET_SEED,
        "train_count": len(train_records),
        "calibration_count": len(calibration_records),
        "train_records_digest": records_digest(train_records),
        "calibration_records_digest": records_digest(calibration_records),
        "feature_count": len(feature_names()),
        "feature_schema_sha256": hashlib.sha256("\n".join(feature_names()).encode()).hexdigest(),
        "target_free_roster": list(TARGET_FREE_ROSTER),
        "roster_frozen_before_calibration_target_decode": True,
        "promotion_gate": "paired mean > +0.005 and paired bootstrap lower95 > 0",
        "model_parameters": {
            "trees": args.trees,
            "ridge_selected_alpha_train_only": ridge_alpha,
        },
        "training_cache": str(cache_path),
        "model_artifact": str(model_path),
        **result,
        "runtime_seconds_excluding_cache_build": perf_counter() - started,
    }
    report_path = output_dir / f"calibration48-train{args.train_limit}.json"
    atomic_write_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "champion": report["champion"],
                "gate": report["champion_passes_promotion_gate"],
                "aggregate": report["aggregate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
