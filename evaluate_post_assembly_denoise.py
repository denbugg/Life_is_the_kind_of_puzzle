"""Honest post-assembly denoising ablation on a frozen no-source holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from global_solver_candidate import solve_layout

GRID, TILE, N = 24, 20, 576


def split(image: np.ndarray) -> np.ndarray:
    return image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3)


def assemble(tiles: np.ndarray, layout: np.ndarray) -> np.ndarray:
    return tiles[layout].reshape(GRID, GRID, TILE, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


def nlm(rgb: np.ndarray, h: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def bilateral(rgb: np.ndarray, sigma: int) -> np.ndarray:
    return cv2.bilateralFilter(rgb, 5, sigma, sigma)


def blend(raw: np.ndarray, filtered: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(
        raw.astype(np.float32) * (1.0 - alpha) + filtered.astype(np.float32) * alpha,
        0,
        255,
    ).round().astype(np.uint8)


def robust(scores: np.ndarray) -> tuple[float, list[float]]:
    folds = np.asarray([scores[offset::4].mean() for offset in range(4)])
    return float(scores.mean() - 0.5 * folds.std()), folds.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--raw-input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.cases)
    methods: dict[str, list[float]] = {
        "raw": [],
        "nlm_h3": [], "nlm_h5": [], "nlm_h7": [], "nlm_h9": [],
        "nlm_h5_a50": [], "nlm_h7_a50": [], "nlm_h9_a50": [],
        "bilateral_s15": [], "bilateral_s25": [],
    }
    per_image = []
    for index, (right, down, pos, target, stem) in enumerate(zip(
        data["right"], data["down"], data["pos"], data["target"], data["stems"]
    )):
        raw_image = np.asarray(Image.open(args.raw_input_dir / f"{stem}.png").convert("RGB"), np.uint8)
        layout = np.asarray(solve_layout(right, down, pos, 20260818 + index * 100), np.int32)
        assembled = assemble(split(raw_image), layout)
        n3, n5, n7, n9 = (nlm(assembled, h) for h in (3, 5, 7, 9))
        outputs = {
            "raw": assembled,
            "nlm_h3": n3, "nlm_h5": n5, "nlm_h7": n7, "nlm_h9": n9,
            "nlm_h5_a50": blend(assembled, n5, 0.5),
            "nlm_h7_a50": blend(assembled, n7, 0.5),
            "nlm_h9_a50": blend(assembled, n9, 0.5),
            "bilateral_s15": bilateral(assembled, 15),
            "bilateral_s25": bilateral(assembled, 25),
        }
        row = {"index": index, "stem": str(stem)}
        for name, image in outputs.items():
            score = float(structural_similarity(target, image, channel_axis=2, data_range=255))
            methods[name].append(score)
            row[name] = score
        per_image.append(row)
        print(json.dumps({"done": index + 1, "total": len(data["stems"]), **row}), flush=True)

    summary = {}
    raw_scores = np.asarray(methods["raw"], np.float64)
    for name, values in methods.items():
        scores = np.asarray(values, np.float64)
        robust_score, folds = robust(scores)
        summary[name] = {
            "mean_ssim": float(scores.mean()),
            "robust_ssim": robust_score,
            "fold_ssim": folds,
            "mean_gain": float(scores.mean() - raw_scores.mean()),
            "wins_vs_raw": int((scores > raw_scores).sum()),
        }
    report = {
        "count": len(per_image),
        "best_mean": max(summary, key=lambda name: summary[name]["mean_ssim"]),
        "best_robust": max(summary, key=lambda name: summary[name]["robust_ssim"]),
        "summary": summary,
        "images": per_image,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
