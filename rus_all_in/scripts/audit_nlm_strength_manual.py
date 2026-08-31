#!/usr/bin/env python3
"""Audit NLM strength/pass grids for non-collapse and manual-rule safety.

All layouts and restored images are reconstructed from dirty inputs and frozen
before any clean target is decoded.  Target images are used only afterwards to
report validation SSIM and to build manual side-by-side sheets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.denoise_safety import (
    cross_board_diagnostics,
    gradient_energy,
    restoration_diagnostics,
)
from aiijc_puzzle.legacy_upgrade import layout_digest
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    assemble_tiles,
    contest_ssim,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.restoration_r6 import nlm_color

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATTERN = "outputs/restoration-r6/nlm-strength-screen-cal12-offset96-h{h}.json"
DEFAULT_STRENGTHS = (10, 12, 15, 20, 30, 50, 80, 120)
DEFAULT_PASSES = (1, 2, 3, 5, 10)
LAYOUTS = (
    "bilateral_buddies96",
    "bilateral_buddies96_atlas_w0p03",
)
VISUAL_LAYOUT = "bilateral_buddies96_atlas_w0p03"
CORE_STRENGTHS = (10, 12, 15, 20, 30, 50)
COLLAPSE_BOUNDARY_STRENGTHS = (80, 120)
CORE_THRESHOLDS = {
    "phase_shift_pixels_max": 0.25,
    "global_std_ratio_min": 0.50,
    "tile_mean_std_ratio_min": 0.70,
    "dynamic_range_ratio_min": 0.65,
    "entropy_bits_min": 4.50,
    "near_constant_tile_fraction_std_lt_2_max": 0.50,
    "pairwise_board_distance_ratio_min": 0.35,
    "cross_board_pixel_variance_ratio_min": 0.35,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/manual-compliance/nlm-strength-grid-cal12-offset96"),
    )
    parser.add_argument("--strengths", type=int, nargs="+", default=DEFAULT_STRENGTHS)
    parser.add_argument("--passes", type=int, nargs="+", default=DEFAULT_PASSES)
    parser.add_argument("--representatives", type=int, default=6)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def source_path(strength: int) -> Path:
    return PROJECT_ROOT / DEFAULT_SOURCE_PATTERN.format(h=strength)


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


def tail_name(pass_count: int) -> str:
    if pass_count == 1:
        return "nlm"
    if pass_count == 2:
        return "nlm_twice"
    return f"nlm_{pass_count}x"


def validate_sources(
    strengths: list[int],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    reports: dict[int, dict[str, Any]] = {}
    for strength in strengths:
        path = source_path(strength)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "completed":
            raise ValueError(f"source report is not completed: {path}")
        if report.get("split") != "calibration" or report.get("offset") != 96:
            raise ValueError(f"source report is not bound to calibration offset 96: {path}")
        if report.get("count") != 12:
            raise ValueError(f"source report count is not 12: {path}")
        if report.get("inference_target_access") is not False:
            raise ValueError(f"source report does not assert target-free inference: {path}")
        if report.get("predictions_frozen_before_target_decode") is not True:
            raise ValueError(f"source report did not freeze predictions: {path}")
        if report.get("configuration", {}).get("nlm_h") != strength:
            raise ValueError(f"source report strength mismatch: {path}")
        if report.get("configuration", {}).get("max_nlm_passes") != 10:
            raise ValueError(f"source report does not contain ten NLM passes: {path}")
        if report.get("configuration", {}).get("r6_evaluated") is not False:
            raise ValueError(f"source report unexpectedly evaluated R6: {path}")
        if not report.get("compliance", {}).get("all_permutation_audits_passed"):
            raise ValueError(f"source report failed strict permutation audit: {path}")
        reports[strength] = report

    reference = reports[strengths[0]]
    names = [row["filename"] for row in reference["per_board"]]
    sanitized: list[dict[str, Any]] = []
    for index, reference_row in enumerate(reference["per_board"]):
        record: dict[str, Any] = {
            "filename": str(reference_row["filename"]),
            "input_sha256": str(reference_row["input_sha256"]),
            "layouts": {},
        }
        for layout_name in LAYOUTS:
            variant = reference_row["variants"][layout_name]
            record["layouts"][layout_name] = {
                "tile_at_position": list(variant["tile_at_position"]),
                "layout_sha256": str(variant["layout_sha256"]),
            }
        sanitized.append(record)

        for strength, report in reports.items():
            row = report["per_board"][index]
            if row["filename"] != names[index] or row["input_sha256"] != record["input_sha256"]:
                raise ValueError(f"record roster drift at h={strength}, index={index}")
            for layout_name in LAYOUTS:
                variant = row["variants"][layout_name]
                expected = record["layouts"][layout_name]
                if (
                    variant["tile_at_position"] != expected["tile_at_position"]
                    or variant["layout_sha256"] != expected["layout_sha256"]
                ):
                    raise ValueError(
                        f"layout drift at h={strength}, index={index}, layout={layout_name}"
                    )
    selection_digests = {report["selection_digest"] for report in reports.values()}
    if len(selection_digests) != 1:
        raise ValueError("selection digest drifted across source reports")
    return reports, sanitized


def representative_names(
    records: list[dict[str, Any]],
    predictions: dict[str, dict[str, dict[int, dict[int, np.ndarray]]]],
    count: int,
    max_strength: int,
    max_pass: int,
) -> list[str]:
    if not 2 <= count <= len(records):
        raise ValueError("representative count outside panel size")
    ranked = sorted(records, key=lambda row: (row["input_gradient_energy"], row["filename"]))
    quantile_count = max(count - 1, 1)
    indices = np.rint(np.linspace(0, len(ranked) - 1, quantile_count)).astype(int)
    selected = [ranked[int(index)]["filename"] for index in indices]
    worst = max(
        records,
        key=lambda row: row["diagnostics"][VISUAL_LAYOUT][str(max_strength)][str(max_pass)][
            "near_constant_tile_fraction_std_lt_4"
        ],
    )["filename"]
    if worst not in selected:
        selected.append(worst)
    for row in ranked:
        if len(selected) >= count:
            break
        if row["filename"] not in selected:
            selected.append(row["filename"])
    del predictions
    return selected[:count]


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
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
    panel = 165
    label_width = 145
    title_height = 44
    row_height = panel + 30
    canvas = Image.new(
        "RGB",
        (label_width + panel * len(columns), title_height + row_height * len(filenames)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(15)
    body_font = font(14)
    for column, name in enumerate(columns):
        draw.text((label_width + column * panel + 6, 11), name, fill="black", font=title_font)
    for row, filename in enumerate(filenames):
        top = title_height + row * row_height
        draw.text((7, top + 7), filename, fill="black", font=body_font)
        for column, name in enumerate(columns):
            value = Image.fromarray(images[filename][name], mode="RGB")
            if crop is not None:
                value = value.crop(crop)
            value = value.resize((panel, panel), Image.Resampling.LANCZOS)
            canvas.paste(value, (label_width + column * panel, top))
    return canvas


def metric_statistics(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    names = tuple(values[0])
    return {
        "mean": {name: float(np.mean([value[name] for value in values])) for name in names},
        "min": {name: float(np.min([value[name] for value in values])) for name in names},
        "max": {name: float(np.max([value[name] for value in values])) for name in names},
    }


def core_checks(
    statistics: dict[str, dict[str, float]],
    cross_board: dict[str, float | int],
    distinct_hashes: int,
    boards: int,
) -> dict[str, bool]:
    return {
        "phase_alignment": (
            statistics["max"]["phase_shift_pixels"] <= CORE_THRESHOLDS["phase_shift_pixels_max"]
        ),
        "global_variance": (
            statistics["min"]["global_std_ratio"] >= CORE_THRESHOLDS["global_std_ratio_min"]
        ),
        "tile_mean_variance": (
            statistics["min"]["tile_mean_std_ratio"] >= CORE_THRESHOLDS["tile_mean_std_ratio_min"]
        ),
        "dynamic_range": (
            statistics["min"]["dynamic_range_ratio"] >= CORE_THRESHOLDS["dynamic_range_ratio_min"]
        ),
        "entropy": statistics["min"]["entropy_bits"] >= CORE_THRESHOLDS["entropy_bits_min"],
        "no_majority_constant_tiles": (
            statistics["max"]["near_constant_tile_fraction_std_lt_2"]
            <= CORE_THRESHOLDS["near_constant_tile_fraction_std_lt_2_max"]
        ),
        "own_board_identity": cross_board["own_raw_board_top1_count"] == boards,
        "pairwise_board_diversity": (
            cross_board["pairwise_board_distance_ratio"]
            >= CORE_THRESHOLDS["pairwise_board_distance_ratio_min"]
        ),
        "cross_board_pixel_variance": (
            cross_board["cross_board_pixel_variance_ratio"]
            >= CORE_THRESHOLDS["cross_board_pixel_variance_ratio_min"]
        ),
        "all_outputs_distinct": distinct_hashes == boards,
    }


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("refusing audit without --run")
    strengths = sorted(set(args.strengths))
    passes = sorted(set(args.passes))
    if strengths != sorted(DEFAULT_STRENGTHS):
        raise ValueError(f"audit strength roster must be exactly {DEFAULT_STRENGTHS}")
    if passes != sorted(DEFAULT_PASSES):
        raise ValueError(f"audit pass roster must be exactly {DEFAULT_PASSES}")

    reports, sanitized = validate_sources(strengths)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, dict[str, dict[int, dict[int, np.ndarray]]]] = {
        layout_name: {} for layout_name in LAYOUTS
    }
    raw_images: dict[str, dict[str, np.ndarray]] = {layout_name: {} for layout_name in LAYOUTS}
    inference_records: list[dict[str, Any]] = []

    # Phase 1: reconstruct strict raw layouts, NLM outputs and every diagnostic.
    # The clean target directory is not touched until this loop completes.
    for board_index, source_record in enumerate(sanitized):
        filename = source_record["filename"]
        dirty = load_rgb(args.inputs / filename, source_record["input_sha256"])
        record: dict[str, Any] = {
            "filename": filename,
            "input_sha256": source_record["input_sha256"],
            "input_gradient_energy": gradient_energy(dirty),
            "layouts": {},
            "diagnostics": {},
            "prediction_sha256": {},
        }
        for layout_name in LAYOUTS:
            metadata = source_record["layouts"][layout_name]
            layout = np.asarray(metadata["tile_at_position"], dtype=np.int32)
            if layout_digest(layout) != metadata["layout_sha256"]:
                raise ValueError(f"layout digest mismatch for {filename}, {layout_name}")
            raw = assemble_tiles(split_tiles(dirty)[layout])
            audit = audit_raw_permutation(dirty, raw, layout, restoration_applied_after_audit=True)
            if not audit.passed:
                raise RuntimeError(f"strict permutation audit failed for {filename}, {layout_name}")
            raw_images[layout_name][filename] = raw
            predictions[layout_name][filename] = {}
            record["layouts"][layout_name] = {
                "tile_at_position": layout.tolist(),
                "layout_sha256": metadata["layout_sha256"],
                "raw_prediction_sha256": array_digest(raw),
                "permutation_audit": audit.as_dict(),
            }
            record["diagnostics"][layout_name] = {}
            record["prediction_sha256"][layout_name] = {}
            for strength in strengths:
                iterative = raw
                selected_predictions: dict[int, np.ndarray] = {}
                selected_diagnostics: dict[str, dict[str, float]] = {}
                selected_hashes: dict[str, str] = {}
                expected_variant = reports[strength]["per_board"][board_index]["variants"][
                    layout_name
                ]
                for pass_count in range(1, max(passes) + 1):
                    iterative = nlm_color(iterative, strength)
                    digest = array_digest(iterative)
                    expected_digest = expected_variant["prediction_sha256"][tail_name(pass_count)]
                    if digest != expected_digest:
                        raise ValueError(
                            f"prediction digest mismatch for {filename}, {layout_name}, "
                            f"h={strength}, pass={pass_count}"
                        )
                    if pass_count in passes:
                        selected_predictions[pass_count] = iterative.copy()
                        selected_diagnostics[str(pass_count)] = restoration_diagnostics(
                            raw, iterative
                        )
                        selected_hashes[str(pass_count)] = digest
                predictions[layout_name][filename][strength] = selected_predictions
                record["diagnostics"][layout_name][str(strength)] = selected_diagnostics
                record["prediction_sha256"][layout_name][str(strength)] = selected_hashes
        inference_records.append(record)
        print(
            json.dumps({"phase": "target-free-freeze", "done": board_index + 1, "total": 12}),
            flush=True,
        )

    frozen_digest = hashlib.sha256(
        "\n".join(
            digest
            for record in inference_records
            for layout_name in LAYOUTS
            for strength in strengths
            for digest in record["prediction_sha256"][layout_name][str(strength)].values()
        ).encode()
    ).hexdigest()

    # Phase 2: with predictions frozen, decode targets for validation SSIM and
    # manual side-by-side review only.  Targets never influence a prediction.
    targets: dict[str, np.ndarray] = {}
    target_hashes: dict[str, str] = {}
    base_report_rows = {row["filename"]: row for row in reports[strengths[0]]["per_board"]}
    for record in inference_records:
        filename = record["filename"]
        expected_hash = str(base_report_rows[filename]["target_sha256"])
        target = load_rgb(args.targets / filename, expected_hash)
        targets[filename] = target
        target_hashes[filename] = expected_hash
        record["official_ssim"] = {}
        for layout_name in LAYOUTS:
            raw = raw_images[layout_name][filename]
            record["official_ssim"][layout_name] = {
                "raw": contest_ssim(target, raw),
                "nlm": {
                    str(strength): {
                        str(pass_count): contest_ssim(
                            target, predictions[layout_name][filename][strength][pass_count]
                        )
                        for pass_count in passes
                    }
                    for strength in strengths
                },
            }

    summary: dict[str, Any] = {}
    for layout_name in LAYOUTS:
        raw_scores = [record["official_ssim"][layout_name]["raw"] for record in inference_records]
        summary[layout_name] = {
            "raw_mean_ssim": float(np.mean(raw_scores)),
            "strengths": {},
        }
        raw_list = [raw_images[layout_name][record["filename"]] for record in inference_records]
        for strength in strengths:
            summary[layout_name]["strengths"][str(strength)] = {}
            for pass_count in passes:
                diagnostics = [
                    record["diagnostics"][layout_name][str(strength)][str(pass_count)]
                    for record in inference_records
                ]
                outputs = [
                    predictions[layout_name][record["filename"]][strength][pass_count]
                    for record in inference_records
                ]
                scores = [
                    record["official_ssim"][layout_name]["nlm"][str(strength)][str(pass_count)]
                    for record in inference_records
                ]
                hashes = [array_digest(output) for output in outputs]
                statistics = metric_statistics(diagnostics)
                cross_board = cross_board_diagnostics(raw_list, outputs)
                checks = core_checks(
                    statistics, cross_board, len(set(hashes)), len(inference_records)
                )
                summary[layout_name]["strengths"][str(strength)][str(pass_count)] = {
                    "mean_ssim": float(np.mean(scores)),
                    "gain_vs_raw": float(np.mean(np.asarray(scores) - np.asarray(raw_scores))),
                    "wins_vs_raw": int(np.sum(np.asarray(scores) > np.asarray(raw_scores))),
                    "diagnostics": statistics,
                    "cross_board": cross_board,
                    "distinct_prediction_hashes": len(set(hashes)),
                    "core_noncollapse_checks": checks,
                    "core_noncollapse_passed": all(checks.values()),
                }

    metric_candidates = [
        (
            metrics["mean_ssim"],
            layout_name,
            strength,
            pass_count,
        )
        for layout_name, layout_metrics in summary.items()
        for strength, strength_metrics in layout_metrics["strengths"].items()
        for pass_count, metrics in strength_metrics.items()
    ]
    metric_score, metric_layout, metric_strength, metric_pass = max(metric_candidates)

    representatives = representative_names(
        inference_records,
        predictions,
        args.representatives,
        max(strengths),
        max(passes),
    )
    visual_images: dict[str, dict[str, np.ndarray]] = {}
    safe_columns = [
        "raw",
        "h10 x1",
        "h10 x5",
        "h12 x1",
        "h15 x1",
        "h15 x3",
        "h20 x1",
        "h20 x3",
        "h30 x1",
        "h30 x3",
        "target",
    ]
    boundary_columns = [
        "raw",
        "h30 x3",
        "h30 x10",
        "h50 x1",
        "h50 x3",
        "h50 x10",
        "h80 x1",
        "h80 x3",
        "h80 x10",
        "h120 x1",
        "h120 x3",
        "h120 x10",
        "target",
    ]
    for filename in representatives:
        values: dict[str, np.ndarray] = {
            "raw": raw_images[VISUAL_LAYOUT][filename],
            "target": targets[filename],
        }
        for strength in strengths:
            for pass_count in passes:
                values[f"h{strength} x{pass_count}"] = predictions[VISUAL_LAYOUT][filename][
                    strength
                ][pass_count]
        visual_images[filename] = values
        frozen_dir = output_dir / "frozen" / Path(filename).stem
        for column in sorted(set(safe_columns + boundary_columns)):
            write_png(frozen_dir / f"{column.replace(' ', '_')}.png", values[column])

    sheets: dict[str, dict[str, str]] = {}
    sheet_rosters = (("safe-grid", safe_columns), ("collapse-boundary", boundary_columns))
    for sheet_name, columns in sheet_rosters:
        for view_name, crop in (("full", None), ("center-zoom", (140, 140, 340, 340))):
            sheet = contact_sheet(visual_images, representatives, columns, crop=crop)
            path = output_dir / f"{sheet_name}-{view_name}.png"
            sheet.save(path)
            sheets[f"{sheet_name}-{view_name}"] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }

    report = {
        "schema": "aiijc-nlm-strength-manual-safety-v1",
        "status": "completed",
        "split": "calibration",
        "offset": 96,
        "count": len(inference_records),
        "strengths": strengths,
        "core_strengths": list(CORE_STRENGTHS),
        "collapse_boundary_strengths": list(COLLAPSE_BOUNDARY_STRENGTHS),
        "passes": passes,
        "layout_variants": list(LAYOUTS),
        "visual_layout": VISUAL_LAYOUT,
        "source_reports": {
            str(strength): {
                "path": str(source_path(strength).resolve()),
                "sha256": sha256_file(source_path(strength)),
            }
            for strength in strengths
        },
        "prediction_contract": {
            "input_only": True,
            "layout_source_fields_whitelist": [
                "filename",
                "input_sha256",
                "tile_at_position",
                "layout_sha256",
            ],
            "all_predictions_and_diagnostics_frozen_before_any_target_load": True,
            "all_reconstructed_prediction_hashes_match_source_reports": True,
            "frozen_prediction_digest": frozen_digest,
            "holdout_opened": False,
            "test_opened": False,
        },
        "strict_compliance": {
            "permutation_audits_passed": int(
                sum(
                    record["layouts"][layout_name]["permutation_audit"]["passed"]
                    for record in inference_records
                    for layout_name in LAYOUTS
                )
            ),
            "permutation_audits_expected": len(inference_records) * len(LAYOUTS),
            "all_576_input_tiles_used_exactly_once": True,
            "raw_assembly_pixel_preserving": True,
            "nlm_applied_only_after_raw_permutation_audit": True,
            "nlm_changes_pixel_values_but_not_coordinates_or_canvas_shape": True,
            "spatial_warp_resize_crop_or_tile_substitution": False,
        },
        "core_thresholds": CORE_THRESHOLDS,
        "metric_max": {
            "layout": metric_layout,
            "nlm_h": int(metric_strength),
            "passes": int(metric_pass),
            "mean_ssim": float(metric_score),
            "manual_safe": None,
            "note": "metric maximum is not a manual-compliance verdict",
        },
        "summary": summary,
        "representative_selection": {
            "policy": (
                "input-gradient quantiles plus worst target-free near-constant-tile board "
                "at h120 x10"
            ),
            "filenames": representatives,
            "target_sha256": {name: target_hashes[name] for name in representatives},
        },
        "contact_sheets": sheets,
        "per_board": inference_records,
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "metric_max": report["metric_max"],
                "representatives": representatives,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
