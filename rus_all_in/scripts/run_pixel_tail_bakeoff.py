#!/usr/bin/env python3
"""Run a target-assisted fixed-layout pixel-tail bakeoff on train pairs only."""

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
import skimage
from PIL import Image

from aiijc_puzzle.pixel_tails import (
    assemble_tiles,
    contest_ssim,
    pixel_tail_variants,
    recover_layout,
    split_tiles,
    summarize_variant_rows,
)
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("calibration", "holdout"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=EXPERIMENT_SUBSET_SEED)
    parser.add_argument("--sample-namespace", default=EXPERIMENT_SUBSET_NAMESPACE)
    parser.add_argument("--descriptor-bins", type=int, default=5)
    parser.add_argument("--reference-bins", type=int, default=20)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_digest = compute_protocol_digest(manifest)
    if manifest.get("protocol_digest") != expected_digest:
        raise ValueError(f"validation manifest digest mismatch: {path}")
    return manifest


def evaluate_layout(
    assembled: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, Any], dict[str, int]]:
    variants, audit = pixel_tail_variants(assembled, target_for_oracle=target)
    metrics: dict[str, Any] = {}
    for name, timed in variants.items():
        metrics[name] = {
            "ssim": contest_ssim(target, timed.image),
            "runtime_seconds": timed.seconds,
            "deployable": timed.deployable,
        }
    return metrics, audit


def aggregate_layout_recovery(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "assignment_agreement",
        "primary_raw_ssim",
        "reference_raw_ssim",
        "reference_minus_primary_raw_ssim",
    )
    aggregate = {f"mean_{key}": float(np.mean([row[key] for row in rows])) for key in keys}
    aggregate["min_assignment_agreement"] = float(min(row["assignment_agreement"] for row in rows))
    for recovery_name in ("primary", "reference"):
        diagnostic_keys = rows[0][recovery_name]
        for key in diagnostic_keys:
            if key == "descriptor_bins":
                continue
            aggregate[f"mean_{recovery_name}_{key}"] = float(
                np.mean([row[recovery_name][key] for row in rows])
            )
    return aggregate


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    selected = select_manifest_records(
        manifest,
        args.split,
        limit=args.limit,
        seed=args.sample_seed,
        namespace=args.sample_namespace,
    )
    filenames = [str(record["filename"]) for record in selected]
    selection_digest = hashlib.sha256("\n".join(filenames).encode()).hexdigest()
    run_started = perf_counter()
    per_board: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []

    for board_index, record in enumerate(selected):
        filename = str(record["filename"])
        stem = Path(filename).stem
        input_path = args.inputs / filename
        target_path = args.targets / filename
        if sha256_file(input_path) != record["input_sha256"]:
            raise ValueError(f"input hash mismatch: {filename}")
        if sha256_file(target_path) != record["target_sha256"]:
            raise ValueError(f"target hash mismatch: {filename}")
        input_image = load_rgb(input_path)
        target_image = load_rgb(target_path)

        recovery_started = perf_counter()
        primary = recover_layout(
            input_image,
            target_image,
            descriptor_bins=args.descriptor_bins,
        )
        primary_seconds = perf_counter() - recovery_started
        reference_started = perf_counter()
        reference = recover_layout(
            input_image,
            target_image,
            descriptor_bins=args.reference_bins,
        )
        reference_seconds = perf_counter() - reference_started

        input_tiles = split_tiles(input_image)
        primary_raw = assemble_tiles(input_tiles, primary.slot_to_input)
        reference_raw = assemble_tiles(input_tiles, reference.slot_to_input)
        primary_raw_ssim = contest_ssim(target_image, primary_raw)
        reference_raw_ssim = contest_ssim(target_image, reference_raw)
        agreement = float((primary.slot_to_input == reference.slot_to_input).mean())
        recovery_row = {
            "stem": stem,
            "assignment_agreement": agreement,
            "primary_raw_ssim": primary_raw_ssim,
            "reference_raw_ssim": reference_raw_ssim,
            "reference_minus_primary_raw_ssim": reference_raw_ssim - primary_raw_ssim,
            "primary_runtime_seconds": primary_seconds,
            "reference_runtime_seconds": reference_seconds,
            "primary": primary.diagnostics(),
            "reference": reference.diagnostics(),
        }
        recovery_rows.append(recovery_row)

        primary_metrics, primary_audit = evaluate_layout(primary_raw, target_image)
        reference_metrics, reference_audit = evaluate_layout(reference_raw, target_image)
        per_board.append(
            {
                "index": board_index,
                "filename": filename,
                "stem": stem,
                "layout_recovery": recovery_row,
                "layouts": {
                    "lowres_hungarian": {
                        "variants": primary_metrics,
                        "gray_audit": primary_audit,
                    },
                    "fullres_reference_hungarian": {
                        "variants": reference_metrics,
                        "gray_audit": reference_audit,
                    },
                },
            }
        )
        print(
            json.dumps(
                {
                    "done": board_index + 1,
                    "total": len(selected),
                    "filename": filename,
                    "layout_agreement": agreement,
                    "lowres_raw_ssim": primary_raw_ssim,
                    "reference_raw_ssim": reference_raw_ssim,
                    "lowres_best": max(
                        primary_metrics,
                        key=lambda name: primary_metrics[name]["ssim"],
                    ),
                }
            ),
            flush=True,
        )

    layout_summaries: dict[str, Any] = {}
    for layout_name in ("lowres_hungarian", "fullres_reference_hungarian"):
        rows = [{"variants": board["layouts"][layout_name]["variants"]} for board in per_board]
        summary = summarize_variant_rows(rows)
        deployable = {name: values for name, values in summary.items() if values["deployable"]}
        layout_summaries[layout_name] = {
            "best_deployable_mean": max(deployable, key=lambda name: deployable[name]["mean_ssim"]),
            "best_deployable_robust": max(
                deployable,
                key=lambda name: deployable[name]["robust_ssim"],
            ),
            "best_including_oracle_mean": max(
                summary,
                key=lambda name: summary[name]["mean_ssim"],
            ),
            "variants": summary,
        }

    report = {
        "experiment": "target-assisted fixed-layout classical pixel-tail bakeoff",
        "protocol": {
            "inputs": str(args.inputs.resolve()),
            "targets": str(args.targets.resolve()),
            "manifest_path": str(manifest_path),
            "protocol_digest": manifest["protocol_digest"],
            "explicitly_excluded": "data/raw/test and every test-derived artifact",
            "selection": "shared deterministic subset inside frozen manifest split",
            "split": args.split,
            "sample_seed": args.sample_seed,
            "sample_namespace": args.sample_namespace,
            "selection_digest": selection_digest,
            "limit": args.limit,
            "filenames": filenames,
            "file_hashes_verified": True,
            "lowres_layout": {
                "descriptor": "per-tile normalized RGB block means",
                "bins": args.descriptor_bins,
                "assignment": "one-to-one scipy Hungarian",
                "target_assisted": True,
                "deployable": False,
            },
            "reference_layout": {
                "descriptor": "per-tile normalized RGB full-resolution pixels",
                "bins": args.reference_bins,
                "assignment": "one-to-one scipy Hungarian",
                "target_assisted": True,
                "deployable": False,
            },
            "metric": "skimage RGB structural_similarity(channel_axis=2, data_range=255)",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "skimage": skimage.__version__,
        },
        "layout_recovery": {
            "aggregate": aggregate_layout_recovery(recovery_rows),
            "boards": recovery_rows,
        },
        "layouts": layout_summaries,
        "wall_seconds": perf_counter() - run_started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_board.json").write_text(
        json.dumps(per_board, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    console_report = {key: value for key, value in report.items() if key != "layout_recovery"}
    print(json.dumps(console_report, indent=2))


if __name__ == "__main__":
    main()
