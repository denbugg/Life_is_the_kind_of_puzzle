#!/usr/bin/env python3
"""Run the noncompliant research-only low-frequency prior analysis."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from aiijc_puzzle.low_frequency_prior import (
    BLEND_STRENGTHS,
    CLUSTER_COUNT,
    GRID_SIZES,
    HUNGARIAN_BLUR_SIGMAS,
    MODEL_SCHEMA_VERSION,
    RIDGE_ALPHA,
    FrozenLowFrequencyPrior,
    dirty_board_features,
    fit_low_frequency_prior,
    target_grid,
)
from aiijc_puzzle.novel_analog_layout import tile_semantic_features
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "low-frequency-prior" / "train5600-v1.npz"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "low-frequency-prior"
    / "calibration48-noncompliant-research-only.json"
)
PRIMARY_BASELINE = "constant_input_channel_median"
MINIMUM_GAIN = 0.005
BOOTSTRAP_SAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-limit", type=int, default=5_600)
    parser.add_argument("--eval-size", type=int, default=48)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError(f"validation manifest digest mismatch: {path}")
    return manifest


def selection_digest(records: list[Any] | tuple[Any, ...]) -> str:
    names = [str(record["filename"]) for record in records]
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def load_rgb_verified(path: Path, expected_hash: str) -> np.ndarray:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_hash:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    seed: int = EXPERIMENT_SUBSET_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired differences must be a non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), (0.025, 0.975))
    return float(low), float(high)


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "low_frequency_prior.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "novel_analog_layout.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "protocol.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def fit_model(
    records: list[Any] | tuple[Any, ...],
    *,
    inputs_dir: Path,
    targets_dir: Path,
    protocol_digest: str,
) -> FrozenLowFrequencyPrior:
    features: list[np.ndarray] = []
    grids: dict[int, list[np.ndarray]] = {size: [] for size in GRID_SIZES}
    generic_sum: np.ndarray | None = None
    for index, record in enumerate(records, start=1):
        name = str(record["filename"])
        dirty = load_rgb_verified(inputs_dir / name, str(record["input_sha256"]))
        target = load_rgb_verified(targets_dir / name, str(record["target_sha256"]))
        features.append(dirty_board_features(dirty))
        for size in GRID_SIZES:
            grids[size].append(target_grid(target, size))
        target_tile_features = tile_semantic_features(split_tiles(target)).astype(np.float64)
        if generic_sum is None:
            generic_sum = np.zeros_like(target_tile_features, dtype=np.float64)
        generic_sum += target_tile_features
        if index % 100 == 0 or index == len(records):
            print(
                json.dumps({"phase": "fit_extract", "done": index, "total": len(records)}),
                flush=True,
            )
    if generic_sum is None:
        raise RuntimeError("empty fit record set")
    metadata: dict[str, object] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "protocol_digest": protocol_digest,
        "train_selection_digest": selection_digest(records),
        "train_filenames_sha256": selection_digest(records),
        "feature_contract": "permutation-invariant semantic distribution plus channel moments",
        "target_contract": "manifest train split only",
    }
    return fit_low_frequency_prior(
        np.stack(features),
        {size: np.stack(values) for size, values in grids.items()},
        (generic_sum / len(records)).astype(np.float32),
        ridge_alpha=RIDGE_ALPHA,
        cluster_count=CLUSTER_COUNT,
        seed=EXPERIMENT_SUBSET_SEED,
        metadata=metadata,
    )


def summarize(per_board: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    names = tuple(per_board[0]["variants"])
    baseline = np.asarray(
        [row["variants"][PRIMARY_BASELINE]["ssim"] for row in per_board], dtype=np.float64
    )
    summary: dict[str, Any] = {}
    for name in names:
        values = np.asarray([row["variants"][name]["ssim"] for row in per_board])
        differences = values - baseline
        low, high = paired_bootstrap_ci(differences)
        summary[name] = {
            "mean_ssim": float(values.mean()),
            "median_ssim": float(np.median(values)),
            "mean_gain_vs_constant_median": float(differences.mean()),
            "paired_bootstrap_95_ci": [low, high],
            "wins_vs_constant_median": int(np.sum(differences > 0)),
            "ties_vs_constant_median": int(np.sum(differences == 0)),
            "boards": int(len(values)),
        }
    candidates = [name for name in names if name != PRIMARY_BASELINE]
    winner = max(candidates, key=lambda name: summary[name]["mean_ssim"])
    winner_summary = summary[winner]
    gate = {
        "baseline": PRIMARY_BASELINE,
        "selected_variant": winner,
        "selection_rule": "maximum mean SSIM among the preregistered non-baseline arms",
        "minimum_required_gain": MINIMUM_GAIN,
        "mean_gain_at_least_0_005": winner_summary["mean_gain_vs_constant_median"]
        >= MINIMUM_GAIN,
        "bootstrap_lower_bound_above_zero": winner_summary["paired_bootstrap_95_ci"][0] > 0,
        "holdout_opened": False,
    }
    gate["passed"] = bool(
        gate["mean_gain_at_least_0_005"] and gate["bootstrap_lower_bound_above_zero"]
    )
    return summary, gate


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("Refusing the expensive run without --run")
    if args.eval_size != 48:
        raise ValueError("the frozen calibration gate requires --eval-size 48")
    manifest = load_manifest(args.manifest)
    train_records_all = tuple(manifest["splits"]["train"])
    if not 1 <= args.train_limit <= len(train_records_all):
        raise ValueError(f"train-limit must be in [1, {len(train_records_all)}]")
    if args.train_limit == len(train_records_all):
        train_records = train_records_all
    else:
        train_records = select_manifest_records(
            manifest,
            "train",
            limit=args.train_limit,
            namespace="low-frequency-prior-smoke-v1",
        )
    calibration_records = select_manifest_records(
        manifest,
        "calibration",
        limit=args.eval_size,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
        seed=EXPERIMENT_SUBSET_SEED,
    )
    train_names = {str(record["filename"]) for record in train_records}
    calibration_names = {str(record["filename"]) for record in calibration_records}
    if train_names & calibration_names:
        raise RuntimeError("train and calibration records overlap")

    started = perf_counter()
    expected_model_metadata = {
        "protocol_digest": str(manifest["protocol_digest"]),
        "train_selection_digest": selection_digest(train_records),
        "train_records": len(train_records),
    }
    if args.model.exists() and not args.refit:
        model = FrozenLowFrequencyPrior.load(args.model)
        for key, expected in expected_model_metadata.items():
            if model.metadata.get(key) != expected:
                raise ValueError(f"model metadata mismatch for {key}: {model.metadata.get(key)!r}")
        fit_seconds = 0.0
    else:
        fit_started = perf_counter()
        model = fit_model(
            train_records,
            inputs_dir=args.inputs,
            targets_dir=args.targets,
            protocol_digest=str(manifest["protocol_digest"]),
        )
        model.save(args.model)
        fit_seconds = perf_counter() - fit_started
    model_hash = sha256_file(args.model)

    per_board: list[dict[str, Any]] = []
    for index, record in enumerate(calibration_records, start=1):
        board_started = perf_counter()
        name = str(record["filename"])
        dirty = load_rgb_verified(args.inputs / name, str(record["input_sha256"]))

        # Leakage boundary: all pixels and hashes are fixed before this record's
        # target path is read or decoded.
        prediction_started = perf_counter()
        predictions = model.predict_all(dirty)
        prediction_seconds = perf_counter() - prediction_started
        prediction_hashes = {
            arm: hashlib.sha256(image.tobytes(order="C")).hexdigest()
            for arm, image in predictions.items()
        }
        target = load_rgb_verified(args.targets / name, str(record["target_sha256"]))
        variants = {
            arm: {
                "ssim": contest_ssim(target, image),
                "prediction_sha256_raw_rgb": prediction_hashes[arm],
            }
            for arm, image in predictions.items()
        }
        per_board.append(
            {
                "filename": name,
                "prediction_frozen_before_target_load": True,
                "variants": variants,
                "prediction_seconds": prediction_seconds,
                "runtime_seconds": perf_counter() - board_started,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "calibration",
                    "done": index,
                    "total": len(calibration_records),
                    "baseline": variants[PRIMARY_BASELINE]["ssim"],
                }
            ),
            flush=True,
        )

    summary, gate = summarize(per_board)
    result = {
        "schema_version": 1,
        "experiment": "low-frequency-prior-v1",
        "status": "noncompliant_research_only",
        "competition_eligible": False,
        "ineligibility_reason": "does not place all 576 dirty fragments in a 24x24 grid",
        "protocol_digest": manifest["protocol_digest"],
        "split": "calibration",
        "train_records": len(train_records),
        "train_selection_digest": selection_digest(train_records),
        "evaluation_records": len(calibration_records),
        "evaluation_selection_digest": selection_digest(calibration_records),
        "selector_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selector_seed": EXPERIMENT_SUBSET_SEED,
        "leakage_contract": {
            "fit_split": "train",
            "selection_split": "calibration",
            "holdout_opened": False,
            "per_record_prediction_frozen_before_target_load": True,
            "competition_test_used": False,
        },
        "roster": {
            "grid_sizes": list(GRID_SIZES),
            "blend_strengths": list(BLEND_STRENGTHS),
            "ridge_alpha": RIDGE_ALPHA,
            "cluster_count": CLUSTER_COUNT,
            "hungarian_blur_sigmas": list(HUNGARIAN_BLUR_SIGMAS),
            "arms": list(per_board[0]["variants"]),
        },
        "model": {
            "path": str(args.model.resolve()),
            "sha256": model_hash,
            "metadata": model.metadata,
        },
        "source_hashes": source_hashes(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "runtime": {
            "fit_seconds": fit_seconds,
            "total_seconds": perf_counter() - started,
        },
        "summary": summary,
        "gate": gate,
        "per_board": per_board,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_variant": gate["selected_variant"],
                "selected": summary[str(gate["selected_variant"])],
                "baseline": summary[PRIMARY_BASELINE],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
