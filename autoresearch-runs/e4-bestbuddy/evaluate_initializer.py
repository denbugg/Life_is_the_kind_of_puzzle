"""Frozen paired smoke evaluator for the E4 best-buddy initializer."""
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

from global_solver_candidate import solve_layout

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
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    cache_hash = None if args.skip_hash else sha256(args.cache)
    if cache_hash is not None and cache_hash != EXPECTED_CACHE_SHA256:
        raise ValueError(f"cache hash mismatch: {cache_hash}")
    data = np.load(args.cache, mmap_mode="r")
    limit = min(args.limit, len(data["stems"]))
    methods = ("hungarian", "best_buddy")
    scores = {method: [] for method in methods}
    adjacencies = {method: [] for method in methods}
    runtimes = {method: [] for method in methods}
    failures = []
    rows = []

    for index in range(limit):
        row = {"index": index, "stem": str(data["stems"][index])}
        for method in methods:
            try:
                os.environ["INITIALIZER_MODE"] = method
                started = time.perf_counter()
                layout = np.asarray(solve_layout(
                    data["right"][index], data["down"][index], data["pos"][index],
                    20260818 + args.seed_offset + index * 100,
                ), np.int32)
                elapsed = time.perf_counter() - started
                if layout.shape != (N,) or not np.array_equal(np.sort(layout), np.arange(N)):
                    raise ValueError("solver returned an invalid permutation")
                score = float(structural_similarity(
                    data["target"][index], assemble(data["tiles"][index], layout),
                    channel_axis=2, data_range=255,
                ))
                adj = adjacency(layout, data["truth"][index])
                scores[method].append(score)
                adjacencies[method].append(adj)
                runtimes[method].append(elapsed)
                row[method] = {"ssim": score, "adjacency": adj, "seconds": elapsed}
            except Exception as exc:  # persist the failure before aborting the gate
                failure = {
                    "index": index,
                    "method": method,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                row[method] = {"failure": failure["error"]}
        rows.append(row)
        print(json.dumps({"done": index + 1, "total": limit, "stem": row["stem"]}), flush=True)

    report = {
        "cache": str(args.cache),
        "cache_sha256": cache_hash or "skipped_after_verified_run",
        "cases": limit,
        "seed_offset": args.seed_offset,
        "methods": {},
        "failures": failures,
        "images": rows,
    }
    for method in methods:
        if len(scores[method]) != limit:
            report["methods"][method] = {"completed": len(scores[method])}
            continue
        report["methods"][method] = {
            "ssim": summarize(scores[method]),
            "mean_adjacency": float(np.mean(adjacencies[method])),
            "runtime_seconds": float(np.sum(runtimes[method])),
            "mean_runtime_seconds": float(np.mean(runtimes[method])),
        }
    if all(len(scores[method]) == limit for method in methods):
        baseline = np.asarray(scores["hungarian"])
        candidate = np.asarray(scores["best_buddy"])
        baseline_adj = np.asarray(adjacencies["hungarian"])
        candidate_adj = np.asarray(adjacencies["best_buddy"])
        report["comparison"] = {
            "robust_ssim_delta": (
                report["methods"]["best_buddy"]["ssim"]["robust"]
                - report["methods"]["hungarian"]["ssim"]["robust"]
            ),
            "mean_ssim_delta": float(candidate.mean() - baseline.mean()),
            "mean_adjacency_delta": float(candidate_adj.mean() - baseline_adj.mean()),
            "ssim_wins": int((candidate > baseline).sum()),
            "ssim_ties": int((candidate == baseline).sum()),
            "adjacency_wins": int((candidate_adj > baseline_adj).sum()),
            "adjacency_ties": int((candidate_adj == baseline_adj).sum()),
            "runtime_ratio": float(
                np.sum(runtimes["best_buddy"]) / np.sum(runtimes["hungarian"])
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
