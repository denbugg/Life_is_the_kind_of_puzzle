"""Verify the E3 compiled kernel against the Python baseline on frozen smoke-32."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

from global_solver_candidate import POSITION_WEIGHT, objective, solve_layout

GRID, TILE, N = 24, 20, 576
EXPECTED_CACHE_SHA256 = "74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df"


def assemble(tiles, layout):
    return (tiles[layout].reshape(GRID, GRID, TILE, TILE, 3)
            .transpose(0, 2, 1, 3, 4).reshape(480, 480, 3))


def adjacency(layout, truth):
    target_of = np.empty(N, np.int32)
    target_of[truth] = np.arange(N)
    board = target_of[layout].reshape(GRID, GRID)
    right = ((board[:, 1:] == board[:, :-1] + 1)
             & (board[:, 1:] // GRID == board[:, :-1] // GRID))
    down = board[1:] == board[:-1] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def summarize(values):
    values = np.asarray(values, np.float64)
    folds = np.asarray([values[offset::4].mean() for offset in range(4)])
    return {
        "mean": float(values.mean()),
        "robust": float(values.mean() - 0.5 * folds.std()),
        "folds": folds.tolist(),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    cache_hash = sha256(args.cache)
    if cache_hash != EXPECTED_CACHE_SHA256:
        raise ValueError(f"cache hash mismatch: {cache_hash}")

    # Import before timing: report warm end-to-end solver time, excluding build/import.
    import global_solver_kernel  # noqa: F401

    data = np.load(args.cache, mmap_mode="r")
    limit = min(args.limit, len(data["stems"]))
    rows, failures = [], []
    python_times, cython_times, scores, adjacencies = [], [], [], []
    exact_layouts, exact_ssim, exact_adjacency = 0, 0, 0
    max_objective_delta = 0.0
    for index in range(limit):
        row = {"index": index, "stem": str(data["stems"][index])}
        try:
            layouts = {}
            for backend, times in (("python", python_times), ("cython", cython_times)):
                os.environ["SOLVER_BACKEND"] = backend
                started = time.perf_counter()
                layout = np.asarray(solve_layout(
                    data["right"][index], data["down"][index], data["pos"][index],
                    20260818 + index * 100,
                ), np.int32)
                times.append(time.perf_counter() - started)
                if layout.shape != (N,) or not np.array_equal(np.sort(layout), np.arange(N)):
                    raise ValueError(f"{backend} returned an invalid permutation")
                layouts[backend] = layout

            py_layout, cy_layout = layouts["python"], layouts["cython"]
            layout_equal = bool(np.array_equal(py_layout, cy_layout))
            exact_layouts += int(layout_equal)
            weighted_pos = POSITION_WEIGHT * data["pos"][index]
            py_objective = objective(py_layout, data["right"][index], data["down"][index], weighted_pos)
            cy_objective = objective(cy_layout, data["right"][index], data["down"][index], weighted_pos)
            objective_delta = float(cy_objective - py_objective)
            max_objective_delta = max(max_objective_delta, abs(objective_delta))
            py_ssim = float(structural_similarity(
                data["target"][index], assemble(data["tiles"][index], py_layout),
                channel_axis=2, data_range=255,
            ))
            cy_ssim = float(structural_similarity(
                data["target"][index], assemble(data["tiles"][index], cy_layout),
                channel_axis=2, data_range=255,
            ))
            py_adj = adjacency(py_layout, data["truth"][index])
            cy_adj = adjacency(cy_layout, data["truth"][index])
            exact_ssim += int(py_ssim == cy_ssim)
            exact_adjacency += int(py_adj == cy_adj)
            scores.append(cy_ssim)
            adjacencies.append(cy_adj)
            row.update({
                "layout_equal": layout_equal,
                "objective_delta": objective_delta,
                "python_ssim": py_ssim,
                "cython_ssim": cy_ssim,
                "python_adjacency": py_adj,
                "cython_adjacency": cy_adj,
                "python_seconds": python_times[-1],
                "cython_seconds": cython_times[-1],
            })
        except Exception as exc:
            failure = {"index": index, "error": repr(exc), "traceback": traceback.format_exc()}
            failures.append(failure)
            row["failure"] = failure["error"]
        rows.append(row)
        print(json.dumps({"done": index + 1, "total": limit, "stem": row["stem"]}), flush=True)

    report = {
        "cache_sha256": cache_hash,
        "cases": limit,
        "exact_layouts": exact_layouts,
        "exact_ssim": exact_ssim,
        "exact_adjacency": exact_adjacency,
        "max_abs_objective_delta": max_objective_delta,
        "python_runtime_seconds": float(np.sum(python_times)),
        "cython_runtime_seconds": float(np.sum(cython_times)),
        "runtime_ratio": float(np.sum(cython_times) / np.sum(python_times)),
        "speedup": float(np.sum(python_times) / np.sum(cython_times)),
        "metrics": {
            "ssim": summarize(scores) if len(scores) == limit else None,
            "mean_adjacency": float(np.mean(adjacencies)) if len(adjacencies) == limit else None,
        },
        "failures": failures,
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
