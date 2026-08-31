#!/usr/bin/env python3
"""Evaluate a preregistered strict global population-layout roster."""

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
from PIL import Image, ImageDraw

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.global_population_layout import (
    CONTROL_ARM,
    FROZEN_ARMS,
    NLM_H,
    PURE_POPULATION_ARM,
    STRONG_POPULATION_WEIGHTS,
    FrozenGlobalPrediction,
    predict_frozen_roster,
)
from aiijc_puzzle.legacy_upgrade import layout_digest
from aiijc_puzzle.low_frequency_prior import FrozenLowFrequencyPrior
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    GRID_SIZE,
    IMAGE_SIZE,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_ATLAS = PROJECT_ROOT / "artifacts" / "low-frequency-prior" / "train5600-v1.npz"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "global-population-layout"
    / "fresh-calibration-offset168-count24.json"
)
DEFAULT_OFFSET = 168
DEFAULT_COUNT = 24
BOOTSTRAP_REPLICATES = 20_000
MONTAGE_ROWS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError(f"validation manifest digest mismatch: {path}")
    return manifest


def load_rgb_verified(path: Path, expected_hash: str) -> np.ndarray:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(path) as image:
        image.load()
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or image.size
            != (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def image_digest(image: np.ndarray) -> str:
    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError(f"invalid frozen image {value.dtype} {value.shape}")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def selection_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode()
    ).hexdigest()


def layout_metrics(layout: np.ndarray, recovered: Any) -> dict[str, float]:
    """Diagnostic truth metrics, computed only after every prediction is frozen."""

    truth = recovered.dirty_at_position
    position_of_dirty = recovered.position_of_dirty
    predicted_positions = np.empty_like(layout)
    predicted_positions[layout] = np.arange(len(layout))
    shifts: dict[tuple[int, int], int] = {}
    for tile, predicted in enumerate(predicted_positions):
        true = int(position_of_dirty[tile])
        predicted_row, predicted_column = divmod(int(predicted), GRID_SIZE)
        true_row, true_column = divmod(true, GRID_SIZE)
        shift = (true_row - predicted_row, true_column - predicted_column)
        shifts[shift] = shifts.get(shift, 0) + 1
    grid = layout.reshape(GRID_SIZE, GRID_SIZE)
    left = position_of_dirty[grid[:, :-1]]
    right = position_of_dirty[grid[:, 1:]]
    top = position_of_dirty[grid[:-1]]
    bottom = position_of_dirty[grid[1:]]
    right_accuracy = np.mean((right - left == 1) & (right // GRID_SIZE == left // GRID_SIZE))
    down_accuracy = np.mean(bottom - top == GRID_SIZE)
    return {
        "direct_placement": float(np.mean(layout == truth)),
        "translation_aligned_placement": float(max(shifts.values()) / len(layout)),
        "right_adjacency": float(right_accuracy),
        "down_adjacency": float(down_accuracy),
        "adjacency": float(0.5 * (right_accuracy + down_accuracy)),
    }


def paired_bootstrap(values: np.ndarray) -> dict[str, Any]:
    differences = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(EXPERIMENT_SUBSET_SEED)
    samples: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        count = min(remaining, 4_096)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        samples.append(differences[indices].mean(axis=1))
        remaining -= count
    low, high = np.quantile(np.concatenate(samples), (0.025, 0.975))
    return {
        "mean": float(differences.mean()),
        "bootstrap_ci95": [float(low), float(high)],
        "wins_ties_losses": [
            int(np.sum(differences > 0)),
            int(np.sum(differences == 0)),
            int(np.sum(differences < 0)),
        ],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def montage_label(name: str) -> str:
    labels = {
        "target": "clean target (calibration only)",
        CONTROL_ARM: "control: no-atlas buddies96",
        PURE_POPULATION_ARM: "pure population Hungarian",
    }
    return labels.get(name, name)


def write_manual_montage(
    path: Path,
    frozen_boards: list[dict[str, Any]],
    targets: dict[str, np.ndarray],
) -> None:
    """Write actual full canvases for the required post-metric manual gate."""

    names = ("target", *FROZEN_ARMS)
    header = 28
    canvas = Image.new(
        "RGB",
        (len(names) * IMAGE_SIZE, min(MONTAGE_ROWS, len(frozen_boards)) * (IMAGE_SIZE + header)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, board in enumerate(frozen_boards[:MONTAGE_ROWS]):
        record = board["record"]
        filename = str(record["filename"])
        predictions = board["predictions"]
        images = {"target": targets[filename]}
        images.update({name: predictions[name].restored for name in FROZEN_ARMS})
        top = row_index * (IMAGE_SIZE + header)
        for column_index, name in enumerate(names):
            left = column_index * IMAGE_SIZE
            draw.text((left + 4, top + 5), montage_label(name), fill="black")
            canvas.paste(Image.fromarray(images[name]), (left, top + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "global_population_layout.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "compliant_atlas_decoder.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "postassembly_harmonizer.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("refusing evaluation without --run")
    if args.offset < DEFAULT_OFFSET:
        raise ValueError(f"fresh panel requires offset >= {DEFAULT_OFFSET}")
    if args.count <= 0:
        raise ValueError("count must be positive")
    manifest = load_manifest(args.manifest)
    panel = select_manifest_records(
        manifest,
        "calibration",
        limit=args.offset + args.count,
        seed=EXPERIMENT_SUBSET_SEED,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
    )
    records = [dict(record) for record in panel[args.offset : args.offset + args.count]]
    if len(records) != args.count:
        raise ValueError(f"requested {args.count} records, selected {len(records)}")
    atlas = FrozenLowFrequencyPrior.load(args.atlas)
    expected_atlas = {
        "protocol_digest": manifest["protocol_digest"],
        "train_records": 5_600,
        "target_contract": "manifest train split only",
    }
    for key, value in expected_atlas.items():
        if atlas.metadata.get(key) != value:
            raise ValueError(f"atlas provenance mismatch for {key}")

    started = perf_counter()
    frozen_boards: list[dict[str, Any]] = []
    # Strong protocol: all inputs and predictions are completed and hashed
    # before this process opens even one clean calibration target.
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = load_rgb_verified(args.inputs / filename, str(record["input_sha256"]))
        predictions = predict_frozen_roster(dirty, atlas.generic_tile_template)
        frozen_boards.append(
            {
                "record": record,
                "dirty": dirty,
                "predictions": predictions,
                "prediction_sha256": {
                    name: {
                        "layout": layout_digest(prediction.layout),
                        "raw": image_digest(prediction.raw),
                        "restored": image_digest(prediction.restored),
                    }
                    for name, prediction in predictions.items()
                },
            }
        )
        print(
            json.dumps(
                {
                    "phase": "target_blind_freeze",
                    "done": index,
                    "total": len(records),
                    "filename": filename,
                }
            ),
            flush=True,
        )
    frozen_prediction_digest = hashlib.sha256(
        "\n".join(
            " ".join(
                board["prediction_sha256"][arm][kind]
                for arm in FROZEN_ARMS
                for kind in ("layout", "raw", "restored")
            )
            for board in frozen_boards
        ).encode()
    ).hexdigest()

    rows: list[dict[str, Any]] = []
    targets: dict[str, np.ndarray] = {}
    for index, board in enumerate(frozen_boards, start=1):
        record = board["record"]
        filename = str(record["filename"])
        dirty = board["dirty"]
        predictions: dict[str, FrozenGlobalPrediction] = board["predictions"]
        target = load_rgb_verified(args.targets / filename, str(record["target_sha256"]))
        targets[filename] = target
        recovered = recover_layout(split_tiles(dirty), split_tiles(target))
        variants = {}
        for name, prediction in predictions.items():
            variants[name] = {
                "raw_ssim": contest_ssim(target, prediction.raw),
                "restored_ssim": contest_ssim(target, prediction.restored),
                **layout_metrics(prediction.layout, recovered),
                "solver": prediction.solver,
                "objective": prediction.objective,
                "solve_seconds": prediction.solve_seconds,
                "layout_sha256": board["prediction_sha256"][name]["layout"],
                "raw_sha256": board["prediction_sha256"][name]["raw"],
                "restored_sha256": board["prediction_sha256"][name]["restored"],
                "permutation_audit": prediction.audit.as_dict(),
                "rgb_diagnostics": prediction.rgb_diagnostics,
                "luminance_diagnostics": prediction.luminance_diagnostics,
            }
        rows.append(
            {
                "filename": filename,
                "all_predictions_frozen_before_any_target_decode": True,
                "variants": variants,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "posthoc_target_score",
                    "done": index,
                    "total": len(records),
                    "filename": filename,
                }
            ),
            flush=True,
        )

    metric_names = (
        "raw_ssim",
        "restored_ssim",
        "direct_placement",
        "translation_aligned_placement",
        "right_adjacency",
        "down_adjacency",
        "adjacency",
    )
    summary = {
        arm: {
            metric: float(np.mean([row["variants"][arm][metric] for row in rows]))
            for metric in metric_names
        }
        for arm in FROZEN_ARMS
    }
    comparisons = {
        f"{arm}_minus_{CONTROL_ARM}": {
            metric: paired_bootstrap(
                np.asarray(
                    [
                        row["variants"][arm][metric] - row["variants"][CONTROL_ARM][metric]
                        for row in rows
                    ]
                )
            )
            for metric in ("raw_ssim", "restored_ssim", "adjacency")
        }
        for arm in FROZEN_ARMS
        if arm != CONTROL_ARM
    }
    metric_champion = max(FROZEN_ARMS, key=lambda arm: summary[arm]["restored_ssim"])
    montage_path = args.output.with_name(f"{args.output.stem}-manual-geometry.png")
    write_manual_montage(montage_path, frozen_boards, targets)
    report = {
        "schema": "aiijc-global-population-layout-calibration-v1",
        "status": "awaiting_manual_geometry_review",
        "split": "calibration",
        "offset": args.offset,
        "count": args.count,
        "holdout_opened": False,
        "test_opened": False,
        "protocol_digest": manifest["protocol_digest"],
        "selection_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selection_seed": EXPERIMENT_SUBSET_SEED,
        "selection_digest": selection_digest(records),
        "freshness_contract": {
            "offset_at_least_168": True,
            "disjoint_from_offset120_count48": args.offset >= 168,
        },
        "prediction_contract": {
            "inference_target_access": False,
            "all_predictions_frozen_before_any_target_decode": True,
            "frozen_prediction_digest": frozen_prediction_digest,
        },
        "preregistered_roster": {
            "arms": list(FROZEN_ARMS),
            "strong_population_weights": list(STRONG_POPULATION_WEIGHTS),
            "edge_budget": 96,
            "tail": [
                "frozen RGB seam-graph offsets",
                "frozen bounded multiplicative luminance gains (max +/-4%)",
                f"proper colored NLM h={NLM_H}, exactly one pass",
            ],
        },
        "compliance": {
            "all_pre_restoration_permutation_audits_passed": all(
                row["variants"][arm]["permutation_audit"]["passed"]
                for row in rows
                for arm in FROZEN_ARMS
            ),
            "all_576_original_upright_dirty_tiles_used_exactly_once": True,
            "raw_input_tile_pixels_preserved": True,
            "population_atlas_used_only_as_assignment_scores": True,
            "population_atlas_rendered_pixels_used": False,
            "template_or_constant_pixels_used": False,
            "spatial_warp_or_tile_substitution_used": False,
            "restoration_applied_only_after_raw_audit": True,
        },
        "metric_champion_before_manual_gate": metric_champion,
        "manual_geometry_review": {
            "status": "pending",
            "artifact": str(montage_path.resolve()),
            "required_for_promotion": True,
        },
        "summary": summary,
        "paired_comparisons": comparisons,
        "runtime_seconds": perf_counter() - started,
        "source_sha256": source_hashes(),
        "per_board": rows,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "manual_montage": str(montage_path.resolve()),
                "metric_champion_before_manual_gate": metric_champion,
                "summary": summary,
                "paired_comparisons": comparisons,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
