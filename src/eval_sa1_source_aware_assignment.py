"""ORBIT-24 SA1: clean-reference source-aware assignment calibration.

Inference path: a shuffled corrupted 480x480 input and a candidate public source
only. The clean train target is loaded *after* assignment solely to score the
held-out labelled calibration case. No test target is ever read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

GRID = 24
TILE = 20
CANVAS = GRID * TILE


@dataclass(frozen=True)
class SourceCase:
    image_id: str
    source_path: Path


def cover_center(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max(CANVAS / width, CANVAS / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    y0 = (resized.shape[0] - CANVAS) // 2
    x0 = (resized.shape[1] - CANVAS) // 2
    return resized[y0:y0 + CANVAS, x0:x0 + CANVAS]


def tile_view(image: np.ndarray) -> np.ndarray:
    if image.shape[:2] != (CANVAS, CANVAS):
        raise ValueError(f"Expected {CANVAS}x{CANVAS}, got {image.shape[:2]}")
    return image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(GRID * GRID, TILE, TILE, 3)


def descriptors(tiles: np.ndarray) -> np.ndarray:
    """Nuisance-resistant 5x5 colour+gradient descriptors, standardized per tile."""
    output: list[np.ndarray] = []
    for tile in tiles:
        blurred = cv2.GaussianBlur(tile, (3, 3), 0)
        colour = cv2.resize(blurred, (5, 5), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.resize(np.stack((gx, gy), axis=-1), (5, 5), interpolation=cv2.INTER_AREA)
        vector = np.concatenate((colour.reshape(-1), gradient.reshape(-1))).astype(np.float32)
        vector = (vector - vector.mean()) / (vector.std() + 1e-6)
        vector /= np.linalg.norm(vector) + 1e-6
        output.append(vector)
    return np.stack(output)


def assign(input_tiles: np.ndarray, reference_canvas: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return input-tile -> reference-slot assignment and summary compatibility."""
    dirty = descriptors(input_tiles)
    clean = descriptors(tile_view(reference_canvas))
    similarity = dirty @ clean.T
    row, col = linear_sum_assignment(-similarity)
    mapping = np.empty(GRID * GRID, dtype=np.int16)
    mapping[row] = col
    chosen = similarity[row, col]
    return mapping, similarity, float(chosen.mean()), float(np.quantile(chosen, 0.10))


def assemble_from_mapping(input_tiles: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    slots = np.empty_like(input_tiles)
    slots[mapping.astype(np.int64)] = input_tiles
    return slots.reshape(GRID, GRID, TILE, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(CANVAS, CANVAS, 3)


def ssim_global(left: np.ndarray, right: np.ndarray) -> float:
    """Global RGB SSIM diagnostic; not a replacement for the competition scorer."""
    x = left.astype(np.float64) / 255.0
    y = right.astype(np.float64) / 255.0
    mu_x, mu_y = x.mean(axis=(0, 1)), y.mean(axis=(0, 1))
    var_x = ((x - mu_x) ** 2).mean(axis=(0, 1))
    var_y = ((y - mu_y) ** 2).mean(axis=(0, 1))
    cov = ((x - mu_x) * (y - mu_y)).mean(axis=(0, 1))
    c1, c2 = 0.01**2, 0.03**2
    values = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(values.mean())


def cases_from_sources(found_train: Path, targets: Path) -> list[SourceCase]:
    chosen: dict[str, Path] = {}
    for path in sorted(found_train.iterdir()):
        if not path.is_file():
            continue
        image_id = path.name[:10]
        if (targets / f"{image_id}.png").is_file():
            chosen.setdefault(image_id, path)
    return [SourceCase(image_id, path) for image_id, path in sorted(chosen.items())]


def split_name(image_id: str) -> str:
    value = int.from_bytes(hashlib.sha256(image_id.encode("utf-8")).digest()[:8], "big")
    return "heldout" if value % 5 == 0 else "calibration"


def load_canvas(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return cover_center(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--found-train", type=Path, required=True)
    parser.add_argument("--train-inputs", type=Path, required=True)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--distractor-offset", type=int, default=37)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cases = cases_from_sources(args.found_train, args.train_targets)
    if args.limit is not None:
        cases = cases[:args.limit]
    if len(cases) < 3:
        raise RuntimeError("Need at least three distinct known-source train cases")

    records: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        input_path = args.train_inputs / f"{case.image_id}.png"
        target_path = args.train_targets / f"{case.image_id}.png"
        input_image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if input_image is None or target is None:
            continue
        source = load_canvas(case.source_path)
        mapping, _, mean_score, q10_score = assign(tile_view(input_image), source)

        # Post-hoc only: clean target defines the reference permutation for scoring.
        oracle_mapping, _, _, _ = assign(tile_view(input_image), target)
        reconstructed = assemble_from_mapping(tile_view(input_image), mapping)
        distractor = load_canvas(cases[(index + args.distractor_offset) % len(cases)].source_path)
        _, _, distractor_mean, distractor_q10 = assign(tile_view(input_image), distractor)

        record = {
            "image_id": case.image_id,
            "split": split_name(case.image_id),
            "source_file": case.source_path.name,
            "mapping_agreement_to_target_oracle": float(np.mean(mapping == oracle_mapping)),
            "source_mean_assignment_similarity": mean_score,
            "source_q10_assignment_similarity": q10_score,
            "distractor_mean_assignment_similarity": distractor_mean,
            "distractor_q10_assignment_similarity": distractor_q10,
            "mean_margin": mean_score - distractor_mean,
            "q10_margin": q10_score - distractor_q10,
            "source_canvas_ssim_posthoc": ssim_global(source, target),
            "input_order_ssim_posthoc": ssim_global(input_image, target),
            "rearranged_dirty_ssim_posthoc": ssim_global(reconstructed, target),
        }
        records.append(record)
        print(
            f"{index + 1}/{len(cases)} {case.image_id} split={record['split']} "
            f"agree={record['mapping_agreement_to_target_oracle']:.3f} "
            f"margin={record['mean_margin']:.3f} source_ssim={record['source_canvas_ssim_posthoc']:.4f}",
            flush=True,
        )

    columns = list(records[0]) if records else ["image_id"]
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    summary: dict[str, object] = {
        "experiment": "SA1_source_aware_clean_reference_assignment",
        "inputs": "dirty shuffled train input plus public source candidate only; target is post-hoc evaluation only",
        "cases": len(records),
        "split_counts": {},
        "metrics": {},
        "gates": {},
    }
    for split in ("calibration", "heldout", "all"):
        subset = records if split == "all" else [row for row in records if row["split"] == split]
        summary["split_counts"][split] = len(subset)
        if not subset:
            continue
        keys = [
            "mapping_agreement_to_target_oracle", "source_mean_assignment_similarity",
            "distractor_mean_assignment_similarity", "mean_margin", "source_canvas_ssim_posthoc",
            "input_order_ssim_posthoc", "rearranged_dirty_ssim_posthoc",
        ]
        summary["metrics"][split] = {
            key: {"mean": float(np.mean([float(row[key]) for row in subset])), "q10": float(np.quantile([float(row[key]) for row in subset], 0.10))}
            for key in keys
        }
    heldout = [row for row in records if row["split"] == "heldout"]
    if heldout:
        agreement = np.array([float(row["mapping_agreement_to_target_oracle"]) for row in heldout])
        margins = np.array([float(row["mean_margin"]) for row in heldout])
        summary["gates"] = {
            "assignment_recovery_gate_gt_0_70": bool(float(agreement.mean()) > 0.70),
            "heldout_assignment_mean": float(agreement.mean()),
            "heldout_assignment_q10": float(np.quantile(agreement, 0.10)),
            "true_vs_single_hard_distractor_margin_positive_fraction": float(np.mean(margins > 0.0)),
            "source_precision_gate_requires_full_candidate_pool": "not assessed; one deterministic hard distractor is a diagnostic only",
            "clean_source_ssim_lift_over_input_order": float(np.mean([float(row["source_canvas_ssim_posthoc"] - float(row["input_order_ssim_posthoc"]) for row in heldout])),
        }
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
