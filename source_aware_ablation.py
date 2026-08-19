"""Evaluate source retrieval + clean-tile Hungarian against the frozen 64 cases.

The retrieval descriptor is invariant to tile order.  Ground-truth targets are used
only for reporting; candidate selection and the fallback gate use query/gallery data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity

from global_solver_candidate import solve_layout

GRID, TILE, N = 24, 20, 576


def split(image: np.ndarray) -> np.ndarray:
    return image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3)


def assemble(tiles: np.ndarray, layout: np.ndarray) -> np.ndarray:
    return tiles[layout].reshape(GRID, GRID, TILE, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


def tile_desc(tiles: np.ndarray) -> np.ndarray:
    """Affine-resistant structure plus weak absolute colour statistics."""
    x = tiles.astype(np.float32)
    pooled = x.reshape(N, 5, 4, 5, 4, 3).mean((2, 4)).reshape(N, -1)
    norm = (pooled - pooled.mean(1, keepdims=True)) / (pooled.std(1, keepdims=True) + 1e-4)
    mean = x.mean((1, 2)) / 255.0
    std = x.std((1, 2)) / 128.0
    return np.concatenate((norm, mean, std), axis=1).astype(np.float32)


def set_desc(tiles: np.ndarray) -> np.ndarray:
    """Permutation-invariant colour/texture distribution descriptor."""
    x = tiles.astype(np.float32) / 255.0
    means = x.mean((1, 2))
    stds = x.std((1, 2))
    gray = x.mean(3)
    grad_x = np.abs(np.diff(gray, axis=2)).mean((1, 2))[:, None]
    grad_y = np.abs(np.diff(gray, axis=1)).mean((1, 2))[:, None]
    feats = np.concatenate((means, stds, grad_x, grad_y), axis=1)
    quantiles = np.quantile(feats, (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98), axis=0)
    hist = []
    for channel in range(3):
        h, _ = np.histogram(x[..., channel], bins=32, range=(0.0, 1.0), density=True)
        hist.append(h / (h.sum() + 1e-8))
    out = np.concatenate((quantiles.ravel(), *hist)).astype(np.float32)
    return out


def build_gallery(target_dir: Path, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    files = np.asarray(sorted(target_dir.glob("*.png")), dtype=object)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        cached_names = z["names"]
        if len(cached_names) == len(files) and all(Path(a).name == Path(b).name for a, b in zip(cached_names, files)):
            return files, z["descriptors"].astype(np.float32)
    descriptors = np.empty((len(files), 152), np.float32)
    for index, path in enumerate(files):
        image = np.asarray(Image.open(path).convert("RGB"), np.uint8)
        descriptors[index] = set_desc(split(image))
        if (index + 1) % 500 == 0:
            print(json.dumps({"gallery": index + 1, "total": len(files)}), flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, names=np.asarray([p.name for p in files]), descriptors=descriptors)
    return files, descriptors


def standardize_gallery(gallery: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = np.median(gallery, axis=0)
    scale = np.quantile(np.abs(gallery - centre), 0.75, axis=0) + 1e-4
    return (gallery - centre) / scale, (query - centre) / scale


def hungarian_layout(query_tiles: np.ndarray, source_image: np.ndarray) -> tuple[np.ndarray, float, float]:
    q = tile_desc(query_tiles)
    s = tile_desc(split(source_image))
    structure = ((q[:, None, :75] - s[None, :, :75]) ** 2).mean(2)
    colour = ((q[:, None, 75:] - s[None, :, 75:]) ** 2).mean(2)
    cost = structure + 0.20 * colour
    rows, positions = linear_sum_assignment(cost)
    layout = np.empty(N, np.int32)
    layout[positions] = rows
    assigned = cost[rows, positions]
    row_second = np.partition(cost, 1, axis=1)[:, 1]
    margin = float(np.mean(row_second[rows] - assigned))
    return layout, float(assigned.mean()), margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--gallery-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--fixed-threshold", type=float, default=0.10403286346600248)
    parser.add_argument("--raw-input-dir", type=Path)
    parser.add_argument("--output-mode", choices=("restored", "raw", "guarded"), default="restored")
    args = parser.parse_args()

    files, gallery_raw = build_gallery(args.targets, args.gallery_cache)
    data = np.load(args.cases)
    rows = []
    stems = data["stems"] if "stems" in data.files else np.asarray([str(i) for i in range(len(data["right"]))])
    for index, (right, down, pos, restored, target, truth, stem) in enumerate(zip(
        data["right"], data["down"], data["pos"], data["restored"], data["target"], data["truth"]
        , stems
    )):
        output_tiles = restored
        reverted_tiles = 0
        if args.raw_input_dir is not None:
            raw_image = np.asarray(Image.open(args.raw_input_dir / f"{stem}.png").convert("RGB"), np.uint8)
            raw_tiles = split(raw_image)
            if args.output_mode == "raw":
                output_tiles = raw_tiles
                reverted_tiles = N
            elif args.output_mode == "guarded":
                raw_std = raw_tiles.astype(np.float32).std((1, 2, 3))
                restored_std = restored.astype(np.float32).std((1, 2, 3))
                raw_mean = raw_tiles.astype(np.float32).mean((1, 2, 3))
                restored_mean = restored.astype(np.float32).mean((1, 2, 3))
                restored_rgb = restored.astype(np.float32).mean((1, 2))
                restored_sat = restored_rgb.max(1) - restored_rgb.min(1)
                bad = (
                    (restored_std < np.maximum(10.0, 0.72 * raw_std))
                    | (np.abs(restored_mean - raw_mean) > 24.0)
                    | ((restored_sat < 10.0) & (restored_std < 25.0) & (raw_std >= 10.0))
                )
                output_tiles = restored.copy()
                output_tiles[bad] = raw_tiles[bad]
                reverted_tiles = int(bad.sum())
        query_raw = set_desc(restored)
        gallery, query = standardize_gallery(gallery_raw, query_raw)
        distances = ((gallery - query) ** 2).mean(1)
        candidate_ids = np.argpartition(distances, args.topk)[:args.topk]
        candidate_ids = candidate_ids[np.argsort(distances[candidate_ids])]

        candidates = []
        for candidate_id in candidate_ids:
            source = np.asarray(Image.open(files[candidate_id]).convert("RGB"), np.uint8)
            layout, assignment_cost, assignment_margin = hungarian_layout(restored, source)
            candidates.append((assignment_cost, int(candidate_id), layout, assignment_margin))
        candidates.sort(key=lambda item: item[0])
        cost, source_id, source_layout, assignment_margin = candidates[0]
        cost2 = candidates[1][0]

        baseline_layout = np.asarray(solve_layout(right, down, pos, 20260818 + index * 100), np.int32)
        baseline_ssim = float(structural_similarity(target, assemble(output_tiles, baseline_layout), channel_axis=2, data_range=255))
        source_ssim = float(structural_similarity(target, assemble(output_tiles, source_layout), channel_axis=2, data_range=255))
        source_image = np.asarray(Image.open(files[source_id]).convert("RGB"), np.uint8)
        source_target_ssim = float(structural_similarity(target, source_image, channel_axis=2, data_range=255))
        retrieval_margin = float((cost2 - cost) / (abs(cost) + 1e-8))
        row = {
            "index": index,
            "source": files[source_id].name,
            "retrieval_rank_distance": float(distances[source_id]),
            "assignment_cost": cost,
            "assignment_margin": assignment_margin,
            "retrieval_margin": retrieval_margin,
            "source_target_ssim": source_target_ssim,
            "baseline_ssim": baseline_ssim,
            "source_ssim": source_ssim,
            "reverted_tiles": reverted_tiles,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    baseline = np.asarray([row["baseline_ssim"] for row in rows])
    source = np.asarray([row["source_ssim"] for row in rows])
    margins = np.asarray([row["retrieval_margin"] for row in rows])
    gates = []
    for threshold in np.quantile(margins, np.linspace(0, 1, 21)):
        use = margins >= threshold
        scores = np.where(use, source, baseline)
        folds = np.asarray([scores[offset::4].mean() for offset in range(4)])
        gates.append({
            "threshold": float(threshold), "coverage": float(use.mean()),
            "mean_ssim": float(scores.mean()),
            "robust_ssim": float(scores.mean() - 0.5 * folds.std()),
            "fold_ssim": folds.tolist(),
        })
    report = {
        "count": len(rows),
        "output_pixels": args.output_mode,
        "mean_reverted_tiles": float(np.mean([row["reverted_tiles"] for row in rows])),
        "baseline_mean_ssim": float(baseline.mean()),
        "source_mean_ssim": float(source.mean()),
        "source_wins": int((source > baseline).sum()),
        "retrieval_exact_like": int(sum(row["source_target_ssim"] > 0.99 for row in rows)),
        "best_gate": max(gates, key=lambda item: item["robust_ssim"]),
        "fixed_gate": next(
            {
                "threshold": float(args.fixed_threshold),
                "coverage": float((margins >= args.fixed_threshold).mean()),
                "mean_ssim": float(np.where(margins >= args.fixed_threshold, source, baseline).mean()),
                "robust_ssim": float(
                    np.where(margins >= args.fixed_threshold, source, baseline).mean()
                    - 0.5 * np.asarray([
                        np.where(margins >= args.fixed_threshold, source, baseline)[offset::4].mean()
                        for offset in range(4)
                    ]).std()
                ),
                "fold_ssim": [
                    float(np.where(margins >= args.fixed_threshold, source, baseline)[offset::4].mean())
                    for offset in range(4)
                ],
            }
            for _ in (0,)
        ),
        "gates": gates,
        "images": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in ("gates", "images")}, indent=2))


if __name__ == "__main__":
    main()
