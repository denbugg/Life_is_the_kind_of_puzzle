#!/usr/bin/env python3
"""Evaluate or package the strongest fully runnable historical fallback stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.legacy_upgrade import (
    atomic_write_png,
    constant_prediction,
    deterministic_submission_zip,
    directional_scores,
    layout_digest,
    low_frequency_prediction,
    solve_buddies,
)
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    IMAGE_SIZE,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
EXPECTED_TEST_IMAGES = 700
CHAMPION_VARIANT = "constant_median_rgb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "holdout", "test"), required=True)
    parser.add_argument("--suite", choices=("champion", "controls", "layout"), default="controls")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--test-inputs", type=Path, default=Path("data/raw/test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=EXPERIMENT_SUBSET_SEED)
    parser.add_argument("--sample-namespace", default=EXPERIMENT_SUBSET_NAMESPACE)
    return parser.parse_args()


def load_rgb_strict(path: Path) -> np.ndarray:
    """Load one exact contest RGB PNG without silent mode conversion."""

    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError(
                f"expected strict RGB PNG, got format={image.format} mode={image.mode}: {path}"
            )
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected {IMAGE_SIZE}x{IMAGE_SIZE} image, got {image.size}: {path}")
        return np.asarray(image, dtype=np.uint8)


def prediction_sha256(image: np.ndarray) -> str:
    """Hash raw RGB bytes after enforcing the contest prediction shape."""

    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError(f"invalid prediction dtype/shape: {value.dtype} {value.shape}")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def build_predictions(
    input_image: np.ndarray,
    *,
    suite: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build inference-only predictions; there is deliberately no target argument."""

    started = perf_counter()
    median = constant_prediction(input_image, statistic="median", per_channel=True)
    predictions = {CHAMPION_VARIANT: median}
    diagnostics: dict[str, Any] = {
        "champion": CHAMPION_VARIANT,
        "layout": None,
    }
    if suite in {"controls", "layout"}:
        predictions.update(
            {
                "constant_mean_rgb": constant_prediction(
                    input_image, statistic="mean", per_channel=True
                ),
                "constant_mean_gray": constant_prediction(
                    input_image, statistic="mean", per_channel=False
                ),
                "low_frequency_gaussian_sigma100": low_frequency_prediction(
                    input_image, sigma=100.0
                ),
                "constant_median_rgb_nlm9": apply_nlm_color(median, h=9).image,
            }
        )
    if suite == "layout":
        tiles = split_tiles(input_image)
        score_started = perf_counter()
        right, down = directional_scores(tiles, views=("bilateral",))["bilateral"]
        score_seconds = perf_counter() - score_started
        result = solve_buddies(right, down, max_edges=96)
        raw = assemble_tiles(tiles[result.layout])
        predictions.update(
            {
                "bilateral_buddies96_raw": raw,
                "bilateral_buddies96_nlm9": apply_nlm_color(raw, h=9).image,
                "bilateral_buddies96_gaussian_sigma100": low_frequency_prediction(raw, sigma=100.0),
            }
        )
        diagnostics["layout"] = {
            "method": "historical ORBIT rank-96 geometry with artifact-free bilateral E14 scores",
            "tile_at_position": result.layout,
            "layout_sha256": layout_digest(result.layout),
            "objective": result.objective,
            "score_runtime_seconds": score_seconds,
            "solve_runtime_seconds": result.runtime_seconds,
        }
    diagnostics["prediction_runtime_seconds"] = perf_counter() - started
    return predictions, diagnostics


def posthoc_layout_metrics(
    layout: np.ndarray, input_image: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Score a frozen layout against approximate target-assisted train labels."""

    recovered = recover_layout(split_tiles(input_image), split_tiles(target))
    truth = recovered.dirty_at_position
    position_of_dirty = recovered.position_of_dirty
    predicted_positions = np.empty_like(layout)
    predicted_positions[layout] = np.arange(len(layout))
    shifts: dict[tuple[int, int], int] = {}
    for tile, predicted in enumerate(predicted_positions):
        true = int(position_of_dirty[tile])
        predicted_row, predicted_column = divmod(int(predicted), 24)
        true_row, true_column = divmod(true, 24)
        shift = (true_row - predicted_row, true_column - predicted_column)
        shifts[shift] = shifts.get(shift, 0) + 1
    grid = layout.reshape(24, 24)
    left = position_of_dirty[grid[:, :-1]]
    right = position_of_dirty[grid[:, 1:]]
    top = position_of_dirty[grid[:-1]]
    bottom = position_of_dirty[grid[1:]]
    right_accuracy = np.mean((right - left == 1) & (right // 24 == left // 24))
    down_accuracy = np.mean(bottom - top == 24)
    return {
        "direct_placement": float(np.mean(layout == truth)),
        "translation_aligned_placement": float(max(shifts.values()) / len(layout)),
        "right_adjacency": float(right_accuracy),
        "down_adjacency": float(down_accuracy),
        "adjacency": float(0.5 * (right_accuracy + down_accuracy)),
        "label_mapping_mean_margin": float(recovered.margin_at_position.mean()),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one JSON report."""

    path = path.resolve()
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


def evaluation_records(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    digest = compute_protocol_digest(manifest)
    if manifest.get("protocol_digest") != digest:
        raise ValueError("validation manifest digest mismatch")
    limit = args.limit or 48
    selected = select_manifest_records(
        manifest,
        args.split,
        limit=limit,
        seed=args.sample_seed,
        namespace=args.sample_namespace,
    )
    return manifest, [dict(record) for record in selected]


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    manifest, records = evaluation_records(args)
    per_board: list[dict[str, Any]] = []
    started = perf_counter()
    for index, record in enumerate(records):
        filename = str(record["filename"])
        input_path = args.train_inputs / filename
        target_path = args.targets / filename
        if sha256_file(input_path) != record["input_sha256"]:
            raise ValueError(f"input hash mismatch: {filename}")
        input_image = load_rgb_strict(input_path)

        # Freeze all predictions and hashes before the target is decoded.  This
        # ordering is part of the leakage audit, not just a performance detail.
        predictions, diagnostics = build_predictions(input_image, suite=args.suite)
        frozen_hashes = {name: prediction_sha256(value) for name, value in predictions.items()}

        if sha256_file(target_path) != record["target_sha256"]:
            raise ValueError(f"target hash mismatch: {filename}")
        target = load_rgb_strict(target_path)
        metrics = {
            name: {
                "ssim": contest_ssim(target, value),
                "prediction_sha256": frozen_hashes[name],
            }
            for name, value in predictions.items()
        }
        layout_record = diagnostics["layout"]
        if layout_record is not None:
            layout = layout_record.pop("tile_at_position")
            layout_record["posthoc_metrics"] = posthoc_layout_metrics(layout, input_image, target)
        flat = input_image.reshape(-1, 3)
        row = {
            "index": index,
            "filename": filename,
            "input_sha256": record["input_sha256"],
            "target_sha256": record["target_sha256"],
            "input_rgb_mean": flat.mean(axis=0).tolist(),
            "input_rgb_median": np.median(flat, axis=0).tolist(),
            "target_rgb_mean_posthoc": target.reshape(-1, 3).mean(axis=0).tolist(),
            "predictions_frozen_before_target_decode": True,
            "metrics": metrics,
            "diagnostics": diagnostics,
        }
        per_board.append(row)
        print(
            json.dumps(
                {
                    "done": index + 1,
                    "total": len(records),
                    "filename": filename,
                    "champion_ssim": metrics[CHAMPION_VARIANT]["ssim"],
                }
            ),
            flush=True,
        )
    variants = tuple(per_board[0]["metrics"])
    aggregate: dict[str, Any] = {}
    for variant in variants:
        scores = np.asarray([row["metrics"][variant]["ssim"] for row in per_board])
        aggregate[variant] = {
            "mean_ssim": float(scores.mean()),
            "std_ssim": float(scores.std()),
            "median_ssim": float(np.median(scores)),
            "minimum_ssim": float(scores.min()),
            "maximum_ssim": float(scores.max()),
        }
    return {
        "schema": "aiijc-legacy-upgrade-evaluation-v1",
        "split": args.split,
        "suite": args.suite,
        "champion": CHAMPION_VARIANT,
        "inference_contract": "each prediction is frozen before target decode",
        "targets_used": "post-hoc metrics and approximate layout diagnostics only",
        "protocol_digest": manifest["protocol_digest"],
        "selection_namespace": args.sample_namespace,
        "selection_seed": args.sample_seed,
        "selection_sha256": hashlib.sha256(
            "\n".join(row["filename"] for row in per_board).encode()
        ).hexdigest(),
        "count": len(per_board),
        "aggregate": aggregate,
        "per_board": per_board,
        "runtime_seconds": perf_counter() - started,
    }


def run_test(args: argparse.Namespace) -> dict[str, Any]:
    if args.suite != "champion":
        raise ValueError("test packaging is locked to --suite champion")
    paths = sorted(args.test_inputs.glob("*.png"))
    if len(paths) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"expected {EXPECTED_TEST_IMAGES} test PNGs, found {len(paths)}")
    if args.limit not in (0, EXPECTED_TEST_IMAGES):
        raise ValueError("a test submission must contain all 700 images")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    rows = []
    for index, path in enumerate(paths):
        input_image = load_rgb_strict(path)
        prediction = constant_prediction(input_image, statistic="median", per_channel=True)
        output_hash = atomic_write_png(output_dir / path.name, prediction)
        rows.append(
            {
                "index": index,
                "filename": path.name,
                "input_sha256": sha256_file(path),
                "prediction_array_sha256": prediction_sha256(prediction),
                "output_png_sha256": output_hash,
                "input_rgb_median": np.median(input_image.reshape(-1, 3), axis=0).tolist(),
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(paths):
            print(json.dumps({"done": index + 1, "total": len(paths)}), flush=True)
    names = [path.name for path in paths]
    output_zip = args.output_zip or output_dir.with_suffix(".zip")
    zip_hash = deterministic_submission_zip(output_dir, names, output_zip)
    return {
        "schema": "aiijc-legacy-upgrade-test-package-v1",
        "method": CHAMPION_VARIANT,
        "target_access": False,
        "count": len(rows),
        "filenames_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "output_zip": str(output_zip.resolve()),
        "output_zip_sha256": zip_hash,
        "per_board": rows,
        "runtime_seconds": perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    report = run_test(args) if args.split == "test" else run_evaluation(args)
    report_path = args.output_dir / "report.json"
    atomic_write_json(report_path, report)
    print(
        json.dumps({"report": str(report_path.resolve()), "summary": report.get("aggregate")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
