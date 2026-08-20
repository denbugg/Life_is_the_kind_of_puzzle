"""Paired E14-vs-E15 evaluator with raw-layout and guarded-pixel reporting."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(HERE)]

import kaggle_e14_solver as e14
from e15_multiplex_solver import (
    DISAGREEMENT_PENALTY,
    GUARDED_SUPPORT_WEIGHT,
    RAW_SUPPORT_WEIGHT,
    solve_layout,
)

EXPECTED_CACHE_SHA256 = "74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df"
EXPECTED_CHECKPOINT_SHA256 = "6fcc7de2cf8063b4f2f45d4b96b8999d5eb9c29a071ff2c0031d2703c70d6695"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble(tiles: np.ndarray, layout: np.ndarray) -> np.ndarray:
    return (tiles[layout].reshape(e14.GRID, e14.GRID, e14.TILE, e14.TILE, 3)
            .transpose(0, 2, 1, 3, 4).reshape(480, 480, 3))


def adjacency(layout: np.ndarray, truth: np.ndarray) -> float:
    target_of = np.empty(e14.N, np.int32)
    target_of[truth] = np.arange(e14.N)
    board = target_of[layout].reshape(e14.GRID, e14.GRID)
    right = ((board[:, 1:] == board[:, :-1] + 1)
             & (board[:, 1:] // e14.GRID == board[:, :-1] // e14.GRID))
    down = board[1:] == board[:-1] + e14.GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def gray_count(tiles: np.ndarray) -> int:
    mean = tiles.astype(np.float32).mean((1, 2))
    std = tiles.astype(np.float32).std((1, 2, 3))
    return int(((mean.max(1) - mean.min(1) < 10.0) & (std < 25.0)).sum())


def summarize(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, np.float64)
    folds = np.asarray([array[offset::4].mean() for offset in range(4)])
    return {
        "mean": float(array.mean()),
        "robust": float(array.mean() - 0.5 * folds.std()),
        "folds": folds.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    cache_hash = sha256(args.cache)
    if cache_hash != EXPECTED_CACHE_SHA256:
        raise ValueError(f"cache hash mismatch: {cache_hash}")
    data = np.load(args.cache, mmap_mode="r", allow_pickle=False)
    sidecar = np.load(args.sidecar, mmap_mode="r", allow_pickle=False)
    provenance = json.loads(str(sidecar["provenance_json"]))
    if provenance["source_cache_sha256"] != cache_hash:
        raise ValueError("sidecar source-cache provenance mismatch")
    if provenance["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("sidecar checkpoint provenance mismatch")
    if not np.array_equal(data["stems"], sidecar["stems"]):
        raise ValueError("sidecar stem order mismatch")
    if sidecar["restored"].shape != data["tiles"].shape:
        raise ValueError("sidecar restored shape mismatch")
    if sidecar["bad_mask"].shape != data["tiles"].shape[:2]:
        raise ValueError("sidecar bad-mask shape mismatch")

    stop = min(args.start + args.limit, len(data["stems"]))
    methods = ("e14", "e15_multiplex")
    raw_ssim = {name: [] for name in methods}
    guarded_ssim = {name: [] for name in methods}
    adjacencies = {name: [] for name in methods}
    runtimes = {name: [] for name in methods}
    rows, failures = [], []
    raw_preprocessing = []
    guard_preprocessing = []
    gray_deltas = []
    gray_excess_images = 0

    for index in range(args.start, stop):
        seed = 20260818 + index * 100 + args.seed_offset
        row = {"index": index, "stem": str(data["stems"][index])}
        try:
            raw_tiles = np.asarray(data["tiles"][index], np.uint8)
            restored = np.asarray(sidecar["restored"][index], np.uint8)
            bad = np.asarray(sidecar["bad_mask"][index], np.bool_)
            guarded = restored.copy()
            guarded[bad] = raw_tiles[bad]
            raw_gray = gray_count(raw_tiles)
            guarded_gray = gray_count(guarded)
            gray_delta = guarded_gray - raw_gray
            gray_deltas.append(gray_delta)
            gray_excess_images += int(gray_delta > 0)
            row.update({
                "reverted_tiles": int(bad.sum()),
                "raw_gray_tiles": raw_gray,
                "guarded_gray_tiles": guarded_gray,
                "gray_delta": gray_delta,
            })

            started = time.perf_counter()
            raw_classical_right, raw_classical_down = e14.classical_mgc_ssd_scores(raw_tiles)
            raw_right = e14.fuse_scores(data["right"][index], raw_classical_right)
            raw_down = e14.fuse_scores(data["down"][index], raw_classical_down)
            raw_prep = time.perf_counter() - started
            raw_preprocessing.append(raw_prep)

            started = time.perf_counter()
            guarded_classical_right, guarded_classical_down = e14.classical_mgc_ssd_scores(guarded)
            guard_prep = time.perf_counter() - started
            guard_preprocessing.append(guard_prep)
            row.update({"raw_preprocessing_seconds": raw_prep,
                        "guard_preprocessing_seconds": guard_prep})

            for method in methods:
                started = time.perf_counter()
                if method == "e14":
                    layout = np.asarray(
                        e14.solve_layout(raw_right, raw_down, data["pos"][index], seed),
                        np.int32,
                    )
                else:
                    layout = np.asarray(
                        solve_layout(
                            raw_right, raw_down,
                            guarded_classical_right, guarded_classical_down,
                            data["pos"][index], seed,
                        ),
                        np.int32,
                    )
                elapsed = time.perf_counter() - started
                if not e14.is_valid_layout(layout):
                    raise ValueError(f"{method} returned invalid permutation")
                raw_score = float(structural_similarity(
                    data["target"][index], assemble(raw_tiles, layout),
                    channel_axis=2, data_range=255,
                ))
                guarded_score = float(structural_similarity(
                    data["target"][index], assemble(guarded, layout),
                    channel_axis=2, data_range=255,
                ))
                adj = adjacency(layout, data["truth"][index])
                if not all(np.isfinite(value) for value in
                           (elapsed, raw_score, guarded_score, adj)):
                    raise FloatingPointError(f"non-finite metric from {method}")
                raw_ssim[method].append(raw_score)
                guarded_ssim[method].append(guarded_score)
                adjacencies[method].append(adj)
                runtimes[method].append(elapsed)
                row[method] = {
                    "raw_ssim": raw_score,
                    "guarded_ssim": guarded_score,
                    "adjacency": adj,
                    "solver_seconds": elapsed,
                    "valid_permutation": True,
                }
        except Exception as exc:
            failures.append({"index": index, "error": repr(exc),
                             "traceback": traceback.format_exc()})
            row["failure"] = repr(exc)
        rows.append(row)
        print(json.dumps({"done": index - args.start + 1,
                          "total": stop - args.start, "stem": row["stem"]}), flush=True)

    cases = stop - args.start
    report = {
        "experiment": "E15 no-gray raw/guarded multiplex relaxation",
        "cache_sha256": cache_hash,
        "sidecar": str(args.sidecar.resolve()),
        "sidecar_provenance": provenance,
        "cases": cases,
        "start": args.start,
        "seed_offset": args.seed_offset,
        "weights": {
            "raw_support": RAW_SUPPORT_WEIGHT,
            "guarded_support": GUARDED_SUPPORT_WEIGHT,
            "disagreement_penalty": DISAGREEMENT_PENALTY,
        },
        "selection_inputs": ["raw tiles", "guarded restored tiles", "right", "down", "pos"],
        "selection_excludes": ["target", "truth", "SSIM", "adjacency"],
        "gray_audit": {
            "gray_delta": int(np.sum(gray_deltas)),
            "images_with_gray_excess": int(gray_excess_images),
        },
        "methods": {},
        "failures": failures,
        "images": rows,
    }
    for method in methods:
        if len(raw_ssim[method]) != cases:
            report["methods"][method] = {"completed": len(raw_ssim[method])}
            continue
        preprocessing = np.sum(raw_preprocessing)
        if method == "e15_multiplex":
            preprocessing += np.sum(guard_preprocessing)
        report["methods"][method] = {
            "raw_ssim": summarize(raw_ssim[method]),
            "guarded_ssim": summarize(guarded_ssim[method]),
            "mean_adjacency": float(np.mean(adjacencies[method])),
            "solver_runtime_seconds": float(np.sum(runtimes[method])),
            "preprocessing_seconds": float(preprocessing),
            "end_to_end_runtime_seconds": float(preprocessing + np.sum(runtimes[method])),
            "valid_permutations": cases,
        }
    if not failures:
        baseline = report["methods"]["e14"]
        candidate = report["methods"]["e15_multiplex"]
        comparison = {
            "raw_robust_ssim_delta": (
                candidate["raw_ssim"]["robust"] - baseline["raw_ssim"]["robust"]
            ),
            "raw_mean_ssim_delta": (
                candidate["raw_ssim"]["mean"] - baseline["raw_ssim"]["mean"]
            ),
            "guarded_robust_ssim_delta": (
                candidate["guarded_ssim"]["robust"]
                - baseline["guarded_ssim"]["robust"]
            ),
            "guarded_mean_ssim_delta": (
                candidate["guarded_ssim"]["mean"]
                - baseline["guarded_ssim"]["mean"]
            ),
            "mean_adjacency_delta": (
                candidate["mean_adjacency"] - baseline["mean_adjacency"]
            ),
            "raw_ssim_wins": int((np.asarray(raw_ssim["e15_multiplex"])
                                  > np.asarray(raw_ssim["e14"])).sum()),
            "adjacency_wins": int((np.asarray(adjacencies["e15_multiplex"])
                                   > np.asarray(adjacencies["e14"])).sum()),
            "runtime_ratio": (
                candidate["end_to_end_runtime_seconds"]
                / baseline["end_to_end_runtime_seconds"]
            ),
        }
        report["comparison"] = comparison
        report["predeclared_smoke_gate"] = bool(
            comparison["raw_robust_ssim_delta"] >= 0.002
            and comparison["mean_adjacency_delta"] >= 0.005
            and comparison["raw_mean_ssim_delta"] > 0
            and comparison["runtime_ratio"] <= 2.0
            and report["gray_audit"]["gray_delta"] <= 0
            and report["gray_audit"]["images_with_gray_excess"] == 0
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "images"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
