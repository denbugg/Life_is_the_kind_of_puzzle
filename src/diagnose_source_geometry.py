"""Read-only source geometry audit for ORBIT-24 SA1 calibration.

For each public-source training original, compare label-blind candidate normalizations
against its withheld clean training target. This does not change data or artifacts.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

SIZE = 480


def normalized_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (gray - gray.mean()) / (gray.std() + 1e-6)


def resize_stretch(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def resize_cover_center(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max(SIZE / width, SIZE / height)
    scaled = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    y0 = (scaled.shape[0] - SIZE) // 2
    x0 = (scaled.shape[1] - SIZE) // 2
    return scaled[y0:y0 + SIZE, x0:x0 + SIZE]


def resize_contain_reflect(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(SIZE / width, SIZE / height)
    scaled = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    pad_y = SIZE - scaled.shape[0]
    pad_x = SIZE - scaled.shape[1]
    return cv2.copyMakeBorder(
        scaled, pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2,
        cv2.BORDER_REFLECT_101,
    )


def score(candidate: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    cand = normalized_gray(candidate)
    truth = normalized_gray(target)
    corr = float(np.mean(cand * truth))
    mae = float(np.mean(np.abs(candidate.astype(np.float32) - target.astype(np.float32))))
    mse = float(np.mean((candidate.astype(np.float32) - target.astype(np.float32)) ** 2))
    return corr, mae, mse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--found-train", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for source_path in sorted(args.found_train.iterdir()):
        if not source_path.is_file():
            continue
        image_id = source_path.name[:10]
        target_path = args.targets / f"{image_id}.png"
        if not target_path.is_file():
            continue
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if source is None or target is None or target.shape[:2] != (SIZE, SIZE):
            continue
        variants = {
            "stretch": resize_stretch(source),
            "cover_center": resize_cover_center(source),
            "contain_reflect": resize_contain_reflect(source),
        }
        result: dict[str, object] = {
            "image_id": image_id,
            "source_file": source_path.name,
            "source_h": source.shape[0],
            "source_w": source.shape[1],
        }
        best_name = ""
        best_corr = -2.0
        for name, variant in variants.items():
            corr, mae, mse = score(variant, target)
            result[f"{name}_corr"] = corr
            result[f"{name}_mae"] = mae
            result[f"{name}_mse"] = mse
            if corr > best_corr:
                best_name, best_corr = name, corr
        result["best_variant"] = best_name
        result["best_corr"] = best_corr
        rows.append(result)
        if args.limit is not None and len(rows) >= args.limit:
            break
    fields = list(rows[0]) if rows else ["image_id"]
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    correlations = np.array([float(row["best_corr"]) for row in rows], dtype=np.float64)
    print(f"rows={len(rows)}")
    if len(rows):
        for quantile in (0.0, 0.1, 0.5, 0.9, 1.0):
            print(f"best_corr_q{int(quantile * 100):02d}={np.quantile(correlations, quantile):.6f}")
        print("best_variant_counts=" + ",".join(
            f"{name}:{sum(row['best_variant'] == name for row in rows)}"
            for name in ("stretch", "cover_center", "contain_reflect")
        ))
        for row in sorted(rows, key=lambda item: float(item["best_corr"]), reverse=True)[:12]:
            print(
                f"top image={row['image_id']} best={row['best_variant']} corr={float(row['best_corr']):.6f} "
                f"size={row['source_w']}x{row['source_h']} file={row['source_file']}"
            )


if __name__ == "__main__":
    main()
