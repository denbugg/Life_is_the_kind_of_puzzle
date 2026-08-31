#!/usr/bin/env python3
"""Run the preregistered train-analog global layout calibration gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

from aiijc_puzzle.novel_analog_layout import (
    board_signature,
    consensus_layout,
    fit_signature_bridge,
    generic_template_features,
    render_layout,
    retrieve_analogs,
    tile_semantic_features,
)
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.protocol import (
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
LIBRARY_NAMESPACE = "novel-analog-layout-library-v1"
CALIBRATION_NAMESPACE = "novel-analog-layout-calibration-v1"
SEED = 20260829
PRIMARY_VARIANT = "analog_consensus4_nlm_h9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--library-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=24)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/novel-analog-layout/calibration24.json")
    )
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError(f"validation manifest digest mismatch: {path}")
    return manifest


def selection_digest(filenames: list[str]) -> str:
    return hashlib.sha256("\n".join(filenames).encode()).hexdigest()


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    seed: int = SEED,
    samples: int = 10_000,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "novel_analog_layout.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "pixel_tails.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "protocol.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def summarize(per_board: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    variants = tuple(per_board[0]["variants"])
    summary: dict[str, Any] = {}
    baseline = np.asarray(
        [row["variants"]["input_nlm_h9"]["ssim"] for row in per_board], dtype=np.float64
    )
    for name in variants:
        values = np.asarray([row["variants"][name]["ssim"] for row in per_board])
        differences = values - baseline
        low, high = paired_bootstrap_ci(differences)
        summary[name] = {
            "mean_ssim": float(values.mean()),
            "median_ssim": float(np.median(values)),
            "mean_gain_vs_input_nlm_h9": float(differences.mean()),
            "paired_bootstrap_95_ci": [low, high],
            "wins_vs_input_nlm_h9": int((differences > 0).sum()),
            "ties_vs_input_nlm_h9": int((differences == 0).sum()),
            "boards": len(values),
        }
    primary = summary[PRIMARY_VARIANT]
    gate = {
        "primary_variant": PRIMARY_VARIANT,
        "mean_gain_at_least_0_015": primary["mean_gain_vs_input_nlm_h9"] >= 0.015,
        "mean_ssim_at_least_0_18": primary["mean_ssim"] >= 0.18,
        "bootstrap_lower_bound_above_zero": primary["paired_bootstrap_95_ci"][0] > 0,
        "wins_at_least_16_of_24": primary["wins_vs_input_nlm_h9"] >= 16,
    }
    gate["passed"] = all(value for key, value in gate.items() if key != "primary_variant")
    gate["holdout_opened"] = False
    return summary, gate


def main() -> None:
    args = parse_args()
    if args.eval_size != 24:
        raise ValueError("the preregistered gate requires --eval-size 24")
    if args.ridge_alpha != 10.0:
        raise ValueError("the preregistered gate requires --ridge-alpha 10")
    manifest = load_manifest(args.manifest)
    library_records = select_manifest_records(
        manifest,
        "train",
        limit=args.library_size,
        seed=SEED,
        namespace=LIBRARY_NAMESPACE,
    )
    eval_records = select_manifest_records(
        manifest,
        "calibration",
        limit=args.eval_size,
        seed=SEED,
        namespace=CALIBRATION_NAMESPACE,
    )
    library_names = [str(record["filename"]) for record in library_records]
    eval_names = [str(record["filename"]) for record in eval_records]
    if set(library_names) & set(eval_names):
        raise RuntimeError("train library and calibration panel overlap")

    run_started = perf_counter()
    dirty_signatures: list[np.ndarray] = []
    clean_signatures: list[np.ndarray] = []
    library_clean_features: list[np.ndarray] = []
    for index, record in enumerate(library_records, start=1):
        filename = str(record["filename"])
        input_path = args.inputs / filename
        target_path = args.targets / filename
        if sha256_file(input_path) != record["input_sha256"]:
            raise ValueError(f"input hash mismatch: {filename}")
        if sha256_file(target_path) != record["target_sha256"]:
            raise ValueError(f"target hash mismatch: {filename}")
        dirty_features = tile_semantic_features(split_tiles(load_rgb(input_path)))
        clean_features = tile_semantic_features(split_tiles(load_rgb(target_path)))
        dirty_signatures.append(board_signature(dirty_features))
        clean_signatures.append(board_signature(clean_features))
        library_clean_features.append(clean_features)
        if index % 64 == 0 or index == len(library_records):
            print(
                json.dumps({"phase": "library", "done": index, "total": len(library_records)}),
                flush=True,
            )

    dirty_signature_array = np.stack(dirty_signatures)
    clean_signature_array = np.stack(clean_signatures)
    library_feature_array = np.stack(library_clean_features)
    bridge = fit_signature_bridge(
        dirty_signature_array,
        clean_signature_array,
        alpha=args.ridge_alpha,
    )
    generic_features = generic_template_features(library_feature_array)
    library_seconds = perf_counter() - run_started

    per_board: list[dict[str, Any]] = []
    for index, record in enumerate(eval_records, start=1):
        board_started = perf_counter()
        filename = str(record["filename"])
        input_path = args.inputs / filename
        target_path = args.targets / filename
        if sha256_file(input_path) != record["input_sha256"]:
            raise ValueError(f"input hash mismatch: {filename}")
        if sha256_file(target_path) != record["target_sha256"]:
            raise ValueError(f"target hash mismatch: {filename}")
        input_image = load_rgb(input_path)
        target_image = load_rgb(target_path)
        query_features = tile_semantic_features(split_tiles(input_image))
        predicted_signature = bridge.transform(board_signature(query_features)[None])[0]
        analog_indices, analog_distances = retrieve_analogs(
            predicted_signature,
            clean_signature_array,
            k=8,
        )
        layouts: dict[str, np.ndarray] = {"input": input_image}
        generic_mapping, _ = consensus_layout(
            query_features,
            generic_features[None],
            np.zeros(1, dtype=np.float32),
        )
        layouts["generic_template"] = render_layout(input_image, generic_mapping)
        for count in (1, 4, 8):
            mapping, _ = consensus_layout(
                query_features,
                library_feature_array[analog_indices[:count]],
                analog_distances[:count],
            )
            layouts[f"analog_top{count}" if count == 1 else f"analog_consensus{count}"] = (
                render_layout(input_image, mapping)
            )

        variants: dict[str, Any] = {}
        for layout_name, image in layouts.items():
            variants[f"{layout_name}_raw"] = {
                "ssim": contest_ssim(target_image, image),
                "tail_seconds": 0.0,
            }
            filtered = apply_nlm_color(image, h=9)
            variants[f"{layout_name}_nlm_h9"] = {
                "ssim": contest_ssim(target_image, filtered.image),
                "tail_seconds": filtered.seconds,
            }
            if args.save_images:
                output_path = (
                    args.output.parent / "images" / Path(filename).stem / f"{layout_name}.png"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(filtered.image, mode="RGB").save(output_path)

        row = {
            "filename": filename,
            "retrieved_analogs": [library_names[int(item)] for item in analog_indices],
            "analog_distances": [float(value) for value in analog_distances],
            "variants": variants,
            "runtime_seconds": perf_counter() - board_started,
        }
        per_board.append(row)
        print(
            json.dumps(
                {
                    "phase": "evaluation",
                    "done": index,
                    "total": len(eval_records),
                    "filename": filename,
                    "input_nlm": variants["input_nlm_h9"]["ssim"],
                    "primary": variants[PRIMARY_VARIANT]["ssim"],
                }
            ),
            flush=True,
        )

    summary, gate = summarize(per_board)
    report = {
        "experiment": "novel scene-analog global layout",
        "status": "promote_to_holdout" if gate["passed"] else "reject_as_tested",
        "protocol": {
            "manifest": str(args.manifest.resolve()),
            "protocol_digest": manifest["protocol_digest"],
            "library_split": "train",
            "library_namespace": LIBRARY_NAMESPACE,
            "library_seed": SEED,
            "library_size": args.library_size,
            "library_selection_digest": selection_digest(library_names),
            "library_filenames": library_names,
            "evaluation_split": "calibration",
            "evaluation_namespace": CALIBRATION_NAMESPACE,
            "evaluation_seed": SEED,
            "evaluation_size": args.eval_size,
            "evaluation_selection_digest": selection_digest(eval_names),
            "evaluation_filenames": eval_names,
            "train_calibration_disjoint": True,
            "file_hashes_verified": True,
            "target_pixels_used_for_inference": False,
            "competition_test_accessed": False,
            "ridge_alpha": args.ridge_alpha,
            "primary_variant": PRIMARY_VARIANT,
            "metric": "RGB SSIM(channel_axis=2, data_range=255)",
            "source_hashes": source_hashes(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "opencv": cv2.__version__,
        },
        "runtime": {
            "library_and_bridge_seconds": library_seconds,
            "total_seconds": perf_counter() - run_started,
        },
        "summary": summary,
        "gate": gate,
        "per_board": per_board,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps({"output": str(args.output), "gate": gate, "summary": summary[PRIMARY_VARIANT]})
    )


if __name__ == "__main__":
    main()
