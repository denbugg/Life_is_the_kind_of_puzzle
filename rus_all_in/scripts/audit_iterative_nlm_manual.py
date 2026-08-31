#!/usr/bin/env python3
"""Audit repeated NLM on a frozen strict decoder without target-time prediction access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.legacy_upgrade import layout_digest
from aiijc_puzzle.protocol import IMAGE_SIZE, assemble_tiles, contest_ssim, sha256_file, split_tiles
from aiijc_puzzle.restoration_r6 import nlm_color

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "outputs/restoration-r6/compliant-iterative-nlm-fresh-calibration24.json"
)
LAYOUT_VARIANT = "bilateral_buddies96_atlas_w0p03"
SAFETY_THRESHOLDS = {
    "phase_shift_pixels_max": 0.25,
    "raw_structural_ssim_min": 0.75,
    "tile_mean_correlation_min": 0.98,
    "same_position_descriptor_top1_min": 0.50,
    "global_std_ratio_min": 0.50,
    "tile_mean_std_ratio_min": 0.80,
    "dynamic_range_ratio_min": 0.70,
    "entropy_bits_min": 4.50,
    "near_constant_tile_fraction_max": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--passes", type=int, nargs="+", default=(1, 5, 10))
    parser.add_argument("--nlm-h", type=int, default=10)
    parser.add_argument("--representatives", type=int, default=6)
    return parser.parse_args()


def load_rgb(path: Path, expected_hash: str | None = None) -> np.ndarray:
    if expected_hash is not None and sha256_file(path) != expected_hash:
        raise ValueError(f"hash mismatch: {path}")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)


def entropy_bits(image: np.ndarray) -> float:
    histogram = np.bincount(grayscale(image).reshape(-1), minlength=256).astype(np.float64)
    probability = histogram[histogram > 0] / histogram.sum()
    return float(-(probability * np.log2(probability)).sum())


def gradient_energy(image: np.ndarray) -> float:
    gray = grayscale(image).astype(np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(dx * dx + dy * dy)))


def tile_descriptors(image: np.ndarray) -> np.ndarray:
    tiles = split_tiles(image)
    descriptors = np.stack(
        [cv2.resize(tile, (5, 5), interpolation=cv2.INTER_AREA).reshape(-1) for tile in tiles]
    ).astype(np.float32)
    mean = descriptors.mean(axis=1, keepdims=True)
    std = descriptors.std(axis=1, keepdims=True)
    return (descriptors - mean) / np.maximum(std, 1e-4)


def descriptor_identity_rate(raw: np.ndarray, restored: np.ndarray) -> float:
    source = tile_descriptors(raw)
    target = tile_descriptors(restored)
    distance = (
        np.square(target).sum(axis=1)[:, None]
        + np.square(source).sum(axis=1)[None]
        - 2.0 * target @ source.T
    )
    return float(np.mean(np.argmin(distance, axis=1) == np.arange(len(source))))


def tile_mean_correlation(raw: np.ndarray, restored: np.ndarray) -> float:
    source = split_tiles(raw).astype(np.float32).mean(axis=(1, 2)).reshape(-1)
    target = split_tiles(restored).astype(np.float32).mean(axis=(1, 2)).reshape(-1)
    if source.std() < 1e-6 or target.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(source, target)[0, 1])


def diagnostics(raw: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    raw_f = raw.astype(np.float32)
    restored_f = restored.astype(np.float32)
    raw_gray = grayscale(raw).astype(np.float32)
    restored_gray = grayscale(restored).astype(np.float32)
    shift, phase_response = cv2.phaseCorrelate(raw_gray, restored_gray)
    phase_shift = float(np.hypot(*shift))
    raw_tiles = split_tiles(raw_f)
    restored_tiles = split_tiles(restored_f)
    raw_tile_means = raw_tiles.mean(axis=(1, 2))
    restored_tile_means = restored_tiles.mean(axis=(1, 2))
    raw_range = float(np.percentile(raw_f, 99) - np.percentile(raw_f, 1))
    restored_range = float(np.percentile(restored_f, 99) - np.percentile(restored_f, 1))
    raw_gradient = gradient_energy(raw)
    restored_gradient = gradient_energy(restored)
    near_constant = np.mean(restored_tiles.std(axis=(1, 2, 3)) < 2.0)
    return {
        "phase_shift_pixels": phase_shift,
        "phase_response": float(phase_response),
        "raw_structural_ssim": contest_ssim(raw, restored),
        "mean_absolute_change": float(np.mean(np.abs(restored_f - raw_f))),
        "global_std_ratio": float(restored_f.std() / max(raw_f.std(), 1e-6)),
        "tile_mean_std_ratio": float(restored_tile_means.std() / max(raw_tile_means.std(), 1e-6)),
        "dynamic_range_ratio": restored_range / max(raw_range, 1e-6),
        "gradient_energy_ratio": restored_gradient / max(raw_gradient, 1e-6),
        "entropy_bits": entropy_bits(restored),
        "entropy_delta_bits": entropy_bits(restored) - entropy_bits(raw),
        "near_constant_tile_fraction": float(near_constant),
        "tile_mean_correlation": tile_mean_correlation(raw, restored),
        "same_position_descriptor_top1": descriptor_identity_rate(raw, restored),
    }


def safety_checks(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "phase_alignment": (
            metrics["phase_shift_pixels"] <= SAFETY_THRESHOLDS["phase_shift_pixels_max"]
        ),
        "raw_structure": (
            metrics["raw_structural_ssim"] >= SAFETY_THRESHOLDS["raw_structural_ssim_min"]
        ),
        "tile_mean_geometry": (
            metrics["tile_mean_correlation"] >= SAFETY_THRESHOLDS["tile_mean_correlation_min"]
        ),
        "tile_descriptor_identity": (
            metrics["same_position_descriptor_top1"]
            >= SAFETY_THRESHOLDS["same_position_descriptor_top1_min"]
        ),
        "global_variance": (
            metrics["global_std_ratio"] >= SAFETY_THRESHOLDS["global_std_ratio_min"]
        ),
        "tile_mean_variance": (
            metrics["tile_mean_std_ratio"] >= SAFETY_THRESHOLDS["tile_mean_std_ratio_min"]
        ),
        "dynamic_range": (
            metrics["dynamic_range_ratio"] >= SAFETY_THRESHOLDS["dynamic_range_ratio_min"]
        ),
        "entropy": metrics["entropy_bits"] >= SAFETY_THRESHOLDS["entropy_bits_min"],
        "nonconstant_tiles": (
            metrics["near_constant_tile_fraction"]
            <= SAFETY_THRESHOLDS["near_constant_tile_fraction_max"]
        ),
    }


def representative_filenames(inference_records: list[dict[str, Any]], count: int) -> list[str]:
    if not 1 <= count <= len(inference_records):
        raise ValueError("representative count outside panel size")
    ranked = sorted(
        inference_records,
        key=lambda record: (record["input_gradient_energy"], record["filename"]),
    )
    indices = np.rint(np.linspace(0, len(ranked) - 1, count)).astype(int)
    return [ranked[int(index)]["filename"] for index in indices]


def font(size: int) -> ImageFont.ImageFont:
    paths = (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def contact_sheet(
    images: dict[str, dict[str, np.ndarray]],
    filenames: list[str],
    columns: list[str],
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    panel = 210
    label_width = 130
    title_height = 40
    row_height = panel + 28
    canvas = Image.new(
        "RGB",
        (label_width + panel * len(columns), title_height + row_height * len(filenames)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(17)
    body_font = font(14)
    for column, name in enumerate(columns):
        draw.text((label_width + column * panel + 8, 10), name, fill="black", font=title_font)
    for row, filename in enumerate(filenames):
        top = title_height + row * row_height
        draw.text((8, top + 8), filename, fill="black", font=body_font)
        for column, name in enumerate(columns):
            value = Image.fromarray(images[filename][name], mode="RGB")
            if crop is not None:
                value = value.crop(crop)
            value = value.resize((panel, panel), Image.Resampling.LANCZOS)
            canvas.paste(value, (label_width + column * panel, top))
    return canvas


def aggregate(records: list[dict[str, Any]], passes: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metric_names = tuple(records[0]["passes"][str(passes[0])]["diagnostics"])
    for pass_count in passes:
        key = str(pass_count)
        result[key] = {
            "boards": len(records),
            "safety_passes": int(
                sum(all(record["passes"][key]["checks"].values()) for record in records)
            ),
            "check_pass_counts": {
                check: int(sum(record["passes"][key]["checks"][check] for record in records))
                for check in records[0]["passes"][key]["checks"]
            },
            "metrics_mean": {
                metric: float(
                    np.mean([record["passes"][key]["diagnostics"][metric] for record in records])
                )
                for metric in metric_names
            },
            "metrics_min": {
                metric: float(
                    np.min([record["passes"][key]["diagnostics"][metric] for record in records])
                )
                for metric in metric_names
            },
            "metrics_max": {
                metric: float(
                    np.max([record["passes"][key]["diagnostics"][metric] for record in records])
                )
                for metric in metric_names
            },
        }
    return result


def main() -> None:
    args = parse_args()
    passes = sorted(set(args.passes))
    if not passes or passes[0] < 1 or passes[-1] > 100:
        raise ValueError("passes must be unique positive integers at most 100")
    if args.nlm_h < 1:
        raise ValueError("nlm-h must be positive")
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source.get("split") != "calibration" or source.get("offset") != 48:
        raise ValueError("audit is bound to fresh calibration offset 48")
    if source.get("inference_target_access") is not False:
        raise ValueError("source report does not assert target-free inference")
    if source.get("predictions_frozen_before_target_decode") is not True:
        raise ValueError("source report did not freeze predictions before targets")
    if source.get("configuration", {}).get("nlm_h") != args.nlm_h:
        raise ValueError("NLM strength differs from source report")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_records: list[dict[str, Any]] = []
    representative_images: dict[str, dict[str, np.ndarray]] = {}

    # Phase 1: whitelist only target-free fields from the authoritative report.
    # All NLM outputs and hashes for all boards are frozen before Phase 2 opens
    # any target image.
    for source_board in source["per_board"]:
        filename = str(source_board["filename"])
        variant = source_board["variants"][LAYOUT_VARIANT]
        layout = np.asarray(variant["tile_at_position"], dtype=np.int32)
        if layout_digest(layout) != variant["layout_sha256"]:
            raise ValueError(f"layout digest mismatch for {filename}")
        input_image = load_rgb(args.inputs / filename, source_board["input_sha256"])
        raw = assemble_tiles(split_tiles(input_image)[layout])
        permutation = audit_raw_permutation(
            input_image, raw, layout, restoration_applied_after_audit=True
        )
        if not permutation.passed:
            raise RuntimeError(f"strict permutation audit failed for {filename}")
        restored = raw
        pass_records: dict[str, Any] = {}
        saved_images: dict[str, np.ndarray] = {"raw": raw.copy()}
        for pass_count in range(1, passes[-1] + 1):
            restored = nlm_color(restored, args.nlm_h)
            if pass_count not in passes:
                continue
            metrics = diagnostics(raw, restored)
            pass_records[str(pass_count)] = {
                "prediction_sha256": array_digest(restored),
                "diagnostics": metrics,
                "checks": safety_checks(metrics),
            }
            saved_images[f"nlm {pass_count}x"] = restored.copy()
        inference_records.append(
            {
                "filename": filename,
                "input_sha256": source_board["input_sha256"],
                "layout_sha256": variant["layout_sha256"],
                "raw_prediction_sha256": array_digest(raw),
                "permutation_audit": permutation.as_dict(),
                "input_gradient_energy": gradient_energy(input_image),
                "passes": pass_records,
            }
        )
        representative_images[filename] = saved_images

    representatives = representative_filenames(inference_records, args.representatives)
    representative_images = {
        filename: representative_images[filename] for filename in representatives
    }
    frozen_digest = hashlib.sha256(
        "\n".join(
            record["raw_prediction_sha256"]
            + " "
            + " ".join(
                record["passes"][str(pass_count)]["prediction_sha256"] for pass_count in passes
            )
            for record in inference_records
        ).encode()
    ).hexdigest()

    # Phase 2: targets are used only for the manual side-by-side contact sheets.
    target_hashes: dict[str, str] = {}
    for filename in representatives:
        target = load_rgb(args.targets / filename)
        representative_images[filename]["target"] = target
        target_hashes[filename] = sha256_file(args.targets / filename)
        frozen_dir = output_dir / "frozen" / Path(filename).stem
        for name, image in representative_images[filename].items():
            write_png(frozen_dir / f"{name.replace(' ', '_')}.png", image)

    columns = ["raw", *[f"nlm {pass_count}x" for pass_count in passes], "target"]
    full = contact_sheet(representative_images, representatives, columns)
    zoom = contact_sheet(
        representative_images,
        representatives,
        columns,
        crop=(140, 140, 340, 340),
    )
    full_path = output_dir / "contact-sheet-full.png"
    zoom_path = output_dir / "contact-sheet-center-zoom.png"
    full.save(full_path)
    zoom.save(zoom_path)

    summary = aggregate(inference_records, passes)
    safe_counts = [
        pass_count
        for pass_count in passes
        if summary[str(pass_count)]["safety_passes"] == len(inference_records)
    ]
    report = {
        "schema": "iterative-nlm-manual-compliance-v1",
        "source_report": str(args.source_report.resolve()),
        "source_report_sha256": sha256_file(args.source_report),
        "layout_variant": LAYOUT_VARIANT,
        "split": "calibration",
        "offset": source["offset"],
        "count": len(inference_records),
        "selection_digest": source["selection_digest"],
        "passes": passes,
        "nlm_h": args.nlm_h,
        "prediction_contract": {
            "input_only": True,
            "layout_source_fields_whitelist": [
                "filename",
                "input_sha256",
                "tile_at_position",
                "layout_sha256",
            ],
            "all_predictions_frozen_before_any_target_load": True,
            "frozen_prediction_digest": frozen_digest,
            "holdout_opened": False,
            "test_opened": False,
        },
        "strict_compliance": {
            "permutation_audits_passed": int(
                sum(record["permutation_audit"]["passed"] for record in inference_records)
            ),
            "boards": len(inference_records),
            "raw_assembly_uses_each_input_tile_once": True,
            "nlm_applied_only_after_raw_audit": True,
            "spatial_warp_or_tile_substitution": False,
        },
        "safety_thresholds": SAFETY_THRESHOLDS,
        "summary": summary,
        "highest_fully_safe_evaluated_pass": max(safe_counts) if safe_counts else None,
        "representative_selection": {
            "policy": "six evenly spaced quantiles of input-only gradient energy",
            "filenames": representatives,
            "target_sha256": target_hashes,
        },
        "contact_sheets": {
            "full": {"path": str(full_path), "sha256": sha256_file(full_path)},
            "center_zoom": {"path": str(zoom_path), "sha256": sha256_file(zoom_path)},
        },
        "per_board": inference_records,
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "representatives": representatives,
                "highest_fully_safe_evaluated_pass": report["highest_fully_safe_evaluated_pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
