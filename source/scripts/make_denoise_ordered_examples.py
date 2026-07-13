#!/usr/bin/env python3
"""Render honest train examples in inferred correct order before/after denoising."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.matching import coarse_photometric_cost, match_tile_sets, multiscale_structural_cost
from puzzle_denoise_v2.metrics import ordered_image_ssim, tile_metrics
from puzzle_denoise_v2.tiles import GRID, merge_tiles_numpy, split_tiles_numpy


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_fused_mapping(input_tiles: np.ndarray, clean_tiles: np.ndarray) -> np.ndarray:
    coarse = coarse_photometric_cost(input_tiles, clean_tiles)
    structural = multiscale_structural_cost(input_tiles, clean_tiles)
    coarse_scale = float(np.median(np.partition(coarse, 1, axis=1)[:, :2])) + 1e-8
    structural_scale = float(np.median(np.partition(structural, 1, axis=1)[:, :2])) + 1e-8
    fused = coarse / coarse_scale + structural / structural_scale
    rows, columns = linear_sum_assignment(fused)
    mapping = np.empty(len(rows), dtype=np.int32)
    mapping[rows] = columns.astype(np.int32)
    return mapping


def _ordered(tiles: np.ndarray, input_to_clean: np.ndarray) -> np.ndarray:
    output = np.empty_like(tiles)
    output[input_to_clean] = tiles
    return output


def _pairwise_squared(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.reshape(len(left), -1).astype(np.float32)
    right = right.reshape(len(right), -1).astype(np.float32)
    distances = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    ) / float(left.shape[1])
    return np.maximum(distances, 0.0)


def _rank_summary(ranks: np.ndarray) -> dict[str, float]:
    return {
        "pairs": int(len(ranks)),
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def _neighbor_ranks(ordered_tiles: np.ndarray) -> np.ndarray:
    right_distance = _pairwise_squared(ordered_tiles[:, :, -1, :], ordered_tiles[:, :, 0, :])
    down_distance = _pairwise_squared(ordered_tiles[:, -1, :, :], ordered_tiles[:, 0, :, :])
    np.fill_diagonal(right_distance, np.inf)
    np.fill_diagonal(down_distance, np.inf)

    indices = np.arange(GRID * GRID).reshape(GRID, GRID)
    right_sources = indices[:, :-1].reshape(-1)
    right_targets = indices[:, 1:].reshape(-1)
    down_sources = indices[:-1, :].reshape(-1)
    down_targets = indices[1:, :].reshape(-1)

    def ranks(distances: np.ndarray, sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
        correct = distances[sources, targets]
        return 1 + np.sum(distances[sources] < correct[:, None], axis=1)

    return np.concatenate(
        [
            ranks(right_distance, right_sources, right_targets),
            ranks(down_distance, down_sources, down_targets),
        ]
    ).astype(np.int32)


def _seam_reference_mae(prediction: np.ndarray, clean: np.ndarray) -> float:
    pred = prediction.reshape(GRID, GRID, 20, 20, 3).astype(np.float32)
    true = clean.reshape(GRID, GRID, 20, 20, 3).astype(np.float32)
    pred_horizontal = pred[:, :-1, :, -1, :] - pred[:, 1:, :, 0, :]
    true_horizontal = true[:, :-1, :, -1, :] - true[:, 1:, :, 0, :]
    pred_vertical = pred[:-1, :, -1, :, :] - pred[1:, :, 0, :, :]
    true_vertical = true[:-1, :, -1, :, :] - true[1:, :, 0, :, :]
    return float(
        0.5
        * (
            np.mean(np.abs(pred_horizontal - true_horizontal))
            + np.mean(np.abs(pred_vertical - true_vertical))
        )
    )


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _grid(images: list[tuple[str, Image.Image]], title: str, output: Path) -> None:
    columns = 2
    rows = (len(images) + columns - 1) // columns
    image_size = 480
    header = 54
    label_height = 30
    gap = 12
    canvas = Image.new(
        "RGB",
        (columns * image_size + (columns + 1) * gap, header + rows * (image_size + label_height + gap)),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 12), title, font=_font(26), fill=(245, 245, 245))
    for index, (name, image) in enumerate(images):
        row, column = divmod(index, columns)
        x = gap + column * (image_size + gap)
        y = header + row * (image_size + label_height + gap)
        canvas.paste(image, (x, y))
        draw.text((x, y + image_size + 4), name, font=_font(18), fill=(230, 230, 230))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def _comparison(rows: list[tuple[str, Image.Image, Image.Image, Image.Image]], output: Path) -> None:
    width = 320
    header = 66
    label_height = 26
    gap = 8
    canvas = Image.new(
        "RGB",
        (3 * width + 4 * gap, header + len(rows) * (width + label_height + gap)),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    headings = ("Correct order - raw", "Correct order - denoised", "Clean target reference")
    for column, heading in enumerate(headings):
        draw.text((gap + column * (width + gap), 14), heading, font=_font(18), fill=(245, 245, 245))
    for row_index, (name, raw, denoised, target) in enumerate(rows):
        y = header + row_index * (width + label_height + gap)
        for column, image in enumerate((raw, denoised, target)):
            thumbnail = image.resize((width, width), Image.Resampling.LANCZOS)
            canvas.paste(thumbnail, (gap + column * (width + gap), y))
        draw.text((gap, y + width + 3), name, font=_font(16), fill=(230, 230, 230))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", default="runs/denoise_v2/reordered_examples")
    parser.add_argument(
        "--checkpoint",
        default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt",
    )
    parser.add_argument(
        "--calibration-report",
        default=(
            "runs/denoise_v2/release_readback/20260710T074500Z/"
            "prefinetune_cpu_v3_current/prefinetune_calibration_report.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 2 or args.count > 8:
        raise SystemExit("--count must be between 2 and 8")
    output_dir = Path(args.output_dir)
    report_path = Path(args.calibration_report)
    calibration_report = json.loads(report_path.read_text(encoding="utf-8"))
    calibration_names = calibration_report["source_split"]["calibration_source_names"]

    candidates = []
    for name in calibration_names:
        input_image = _read_rgb(REPO_ROOT / "puzzle/train/inputs" / name)
        target_image = _read_rgb(REPO_ROOT / "puzzle/train/targets" / name)
        input_tiles = split_tiles_numpy(input_image)
        target_tiles = split_tiles_numpy(target_image)
        match = match_tile_sets(input_tiles, target_tiles)
        fused = _normalized_fused_mapping(input_tiles, target_tiles)
        candidates.append(
            {
                "name": name,
                "mapping": fused,
                "coarse_structural_consensus": float(match.consensus.mean()),
                "fused_coarse_agreement": float(np.mean(fused == match.coarse.mapping)),
                "fused_structural_agreement": float(np.mean(fused == match.structural.mapping)),
            }
        )
    candidates.sort(
        key=lambda item: (
            min(item["fused_coarse_agreement"], item["fused_structural_agreement"]),
            item["coarse_structural_consensus"],
            item["name"],
        ),
        reverse=True,
    )
    selected = candidates[: args.count]

    model, device, model_metadata = load_restorer(args.checkpoint, device=args.device)
    raw_grid: list[tuple[str, Image.Image]] = []
    denoised_grid: list[tuple[str, Image.Image]] = []
    comparison_rows = []
    records = []
    all_raw_ranks = []
    all_denoised_ranks = []

    for item in selected:
        name = str(item["name"])
        input_image = _read_rgb(REPO_ROOT / "puzzle/train/inputs" / name)
        target_image = _read_rgb(REPO_ROOT / "puzzle/train/targets" / name)
        input_tiles = split_tiles_numpy(input_image)
        target_tiles = split_tiles_numpy(target_image)
        restored_tiles = restore_tiles_uint8(model, input_tiles, device, args.batch_size)
        mapping = np.asarray(item["mapping"])
        ordered_raw_tiles = _ordered(input_tiles, mapping)
        ordered_denoised_tiles = _ordered(restored_tiles, mapping)
        raw_image = Image.fromarray(merge_tiles_numpy(ordered_raw_tiles), mode="RGB")
        denoised_image = Image.fromarray(merge_tiles_numpy(ordered_denoised_tiles), mode="RGB")
        target = Image.fromarray(target_image, mode="RGB")

        stem = Path(name).stem
        raw_path = output_dir / "individual" / f"{stem}_correct_order_raw.png"
        denoised_path = output_dir / "individual" / f"{stem}_correct_order_denoised.png"
        target_path = output_dir / "individual" / f"{stem}_clean_target.png"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_image.save(raw_path, format="PNG", optimize=True)
        denoised_image.save(denoised_path, format="PNG", optimize=True)
        target.save(target_path, format="PNG", optimize=True)

        raw_ranks = _neighbor_ranks(ordered_raw_tiles)
        denoised_ranks = _neighbor_ranks(ordered_denoised_tiles)
        all_raw_ranks.append(raw_ranks)
        all_denoised_ranks.append(denoised_ranks)
        raw_metrics = tile_metrics(ordered_raw_tiles, target_tiles)
        denoised_metrics = tile_metrics(ordered_denoised_tiles, target_tiles)
        raw_metrics["ordered_image_ssim"] = ordered_image_ssim(ordered_raw_tiles, target_tiles)
        denoised_metrics["ordered_image_ssim"] = ordered_image_ssim(ordered_denoised_tiles, target_tiles)
        raw_metrics["seam_reference_mae"] = _seam_reference_mae(ordered_raw_tiles, target_tiles)
        denoised_metrics["seam_reference_mae"] = _seam_reference_mae(
            ordered_denoised_tiles, target_tiles
        )
        records.append(
            {
                "name": name,
                "selection_is_independent_of_denoiser_gain": True,
                "mapping": {
                    key: value for key, value in item.items() if key not in {"name", "mapping"}
                },
                "raw": raw_metrics,
                "denoised": denoised_metrics,
                "raw_neighbor_rank": _rank_summary(raw_ranks),
                "denoised_neighbor_rank": _rank_summary(denoised_ranks),
                "files": {
                    "raw": str(raw_path),
                    "denoised": str(denoised_path),
                    "clean_target": str(target_path),
                },
            }
        )
        raw_grid.append((stem, raw_image))
        denoised_grid.append((stem, denoised_image))
        comparison_rows.append((stem, raw_image, denoised_image, target))

    raw_contact = output_dir / "correct_order_without_denoise.png"
    denoised_contact = output_dir / "correct_order_with_denoise.png"
    comparison_contact = output_dir / "before_after_clean_comparison.png"
    _grid(raw_grid, "Input tiles in inferred correct order - no denoise", raw_contact)
    _grid(denoised_grid, "Same order - selected denoiser", denoised_contact)
    _comparison(comparison_rows, comparison_contact)

    aggregate_raw = _rank_summary(np.concatenate(all_raw_ranks))
    aggregate_denoised = _rank_summary(np.concatenate(all_denoised_ranks))
    report = {
        "schema_version": 1,
        "kind": "ordered_raw_vs_denoised_examples",
        "selection_policy": (
            "top fused/coarse/structural mapping agreement on the frozen clean calibration "
            "partition; selection does not use denoiser improvement"
        ),
        "correct_order_caveat": (
            "train data has no published permutation labels; order is inferred against the clean "
            "target with an ensemble Hungarian map and examples are restricted to the most "
            "internally consistent maps"
        ),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "device": str(device),
        "model_metadata": model_metadata,
        "calibration_report": str(report_path),
        "calibration_report_sha256": _sha256(report_path),
        "candidate_sources": len(candidates),
        "selected_sources": len(records),
        "examples": records,
        "aggregate_simple_neighbor_rank": {
            "raw": aggregate_raw,
            "denoised": aggregate_denoised,
            "scope": "right and down ground-truth neighbours, RGB boundary MSE, 1104 pairs/source",
        },
        "contact_sheets": {
            "raw": str(raw_contact),
            "denoised": str(denoised_contact),
            "comparison": str(comparison_contact),
        },
    }
    report_output = output_dir / "report.json"
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "report": str(report_output),
        "selected": [record["name"] for record in records],
        "aggregate_simple_neighbor_rank": report["aggregate_simple_neighbor_rank"],
        "comparison": str(comparison_contact),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
