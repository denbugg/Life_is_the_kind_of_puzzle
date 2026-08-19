"""Paired frozen-cache evaluator for E11 sparse relaxation labeling."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from global_solver_candidate import POSITION_WEIGHT, objective, solve_layout

GRID, TILE, N = 24, 20, 576
BASELINE_COMMIT = "ceea9ca234d8700bfeef5a9392f1ef31d6dfe4b7"
EXPECTED_CACHE_SHA256 = "74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df"


def load_baseline_solver(repo: Path):
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:global_solver_candidate.py"],
        cwd=repo,
        text=True,
    )
    module = types.ModuleType("e11_frozen_baseline")
    exec(compile(source, f"{BASELINE_COMMIT}:global_solver_candidate.py", "exec"), module.__dict__)
    return module.solve_layout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble(tiles: np.ndarray, layout: np.ndarray) -> np.ndarray:
    return (tiles[layout].reshape(GRID, GRID, TILE, TILE, 3)
            .transpose(0, 2, 1, 3, 4).reshape(480, 480, 3))


def adjacency(layout: np.ndarray, truth: np.ndarray) -> float:
    target_of = np.empty(N, np.int32)
    target_of[truth] = np.arange(N)
    board = target_of[layout].reshape(GRID, GRID)
    right = ((board[:, 1:] == board[:, :-1] + 1)
             & (board[:, 1:] // GRID == board[:, :-1] // GRID))
    down = board[1:] == board[:-1] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    cache_hash = "skipped_after_verified_run" if args.skip_hash else sha256(args.cache)
    if not args.skip_hash and cache_hash != EXPECTED_CACHE_SHA256:
        raise ValueError(f"cache hash mismatch: {cache_hash}")
    baseline_solver = load_baseline_solver(REPO)
    data = np.load(args.cache, mmap_mode="r")
    stop = min(args.start + args.limit, len(data["stems"]))
    methods = ("baseline_sa", "e11_relaxation")
    solvers = (baseline_solver, solve_layout)
    scores = {method: [] for method in methods}
    adjacencies = {method: [] for method in methods}
    objectives = {method: [] for method in methods}
    runtimes = {method: [] for method in methods}
    rows, failures = [], []

    for index in range(args.start, stop):
        row = {"index": index, "stem": str(data["stems"][index])}
        for method, solver in zip(methods, solvers):
            try:
                started = time.perf_counter()
                layout = np.asarray(solver(
                    data["right"][index], data["down"][index], data["pos"][index],
                    20260818 + index * 100 + args.seed_offset,
                ), np.int32)
                elapsed = time.perf_counter() - started
                if layout.shape != (N,) or not np.array_equal(np.sort(layout), np.arange(N)):
                    raise ValueError("solver returned an invalid permutation")
                score = float(structural_similarity(
                    data["target"][index], assemble(data["tiles"][index], layout),
                    channel_axis=2, data_range=255,
                ))
                adj = adjacency(layout, data["truth"][index])
                cached_objective = objective(
                    layout, data["right"][index], data["down"][index],
                    POSITION_WEIGHT * data["pos"][index],
                )
                scores[method].append(score)
                adjacencies[method].append(adj)
                objectives[method].append(cached_objective)
                runtimes[method].append(elapsed)
                row[method] = {
                    "ssim": score,
                    "adjacency": adj,
                    "cached_objective": cached_objective,
                    "seconds": elapsed,
                    "valid_permutation": True,
                }
            except Exception as exc:
                failure = {
                    "index": index,
                    "method": method,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                row[method] = {"failure": failure["error"]}
        rows.append(row)
        print(json.dumps({"done": index - args.start + 1, "total": stop - args.start,
                          "stem": row["stem"]}), flush=True)

    cases = stop - args.start
    report = {
        "experiment": "E11 sparse multi-phase relaxation labeling",
        "baseline_commit": BASELINE_COMMIT,
        "cache": str(args.cache),
        "cache_sha256": cache_hash,
        "cases": cases,
        "start": args.start,
        "seed_offset": args.seed_offset,
        "seed_formula": "20260818 + index * 100 + seed_offset",
        "selection_inputs": ["right", "down", "pos"],
        "selection_excludes": ["target", "truth", "SSIM", "adjacency"],
        "methods": {},
        "failures": failures,
        "images": rows,
    }
    for method in methods:
        if len(scores[method]) != cases:
            report["methods"][method] = {"completed": len(scores[method])}
            continue
        report["methods"][method] = {
            "ssim": summarize(scores[method]),
            "mean_adjacency": float(np.mean(adjacencies[method])),
            "mean_cached_objective": float(np.mean(objectives[method])),
            "runtime_seconds": float(np.sum(runtimes[method])),
            "mean_runtime_seconds": float(np.mean(runtimes[method])),
            "valid_permutations": cases,
        }
    if not failures:
        baseline_scores = np.asarray(scores["baseline_sa"])
        candidate_scores = np.asarray(scores["e11_relaxation"])
        baseline_adj = np.asarray(adjacencies["baseline_sa"])
        candidate_adj = np.asarray(adjacencies["e11_relaxation"])
        report["comparison"] = {
            "robust_ssim_delta": (
                report["methods"]["e11_relaxation"]["ssim"]["robust"]
                - report["methods"]["baseline_sa"]["ssim"]["robust"]
            ),
            "mean_ssim_delta": float(candidate_scores.mean() - baseline_scores.mean()),
            "mean_adjacency_delta": float(candidate_adj.mean() - baseline_adj.mean()),
            "mean_cached_objective_delta": float(
                np.mean(objectives["e11_relaxation"])
                - np.mean(objectives["baseline_sa"])
            ),
            "ssim_wins": int((candidate_scores > baseline_scores).sum()),
            "adjacency_wins": int((candidate_adj > baseline_adj).sum()),
            "runtime_ratio": float(
                np.sum(runtimes["e11_relaxation"]) / np.sum(runtimes["baseline_sa"])
            ),
        }
        delta = report["comparison"]
        report["dual_metric_gate"] = bool(
            delta["robust_ssim_delta"] > 0
            and delta["mean_ssim_delta"] > 0
            and delta["mean_adjacency_delta"] >= 0
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
