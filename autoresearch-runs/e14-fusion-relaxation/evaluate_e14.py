"""Paired frozen-cache evaluator for E14 E2-score -> E11-relaxation."""
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
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(HERE)]
from e2_raw_fusion import ALPHA, classical_mgc_ssd_scores, fuse_scores
from global_solver_candidate import POSITION_WEIGHT, objective, solve_layout

GRID, TILE, N = 24, 20, 576
BASELINE_COMMIT = "ceea9ca234d8700bfeef5a9392f1ef31d6dfe4b7"
E2_COMMIT = "63c14562ce6caa7228ccb902160259531b8fbab2"
E11_COMMIT = "4d677494a6ec4532caba64aa35d72db018644a8c"
EXPECTED_CACHE_SHA256 = "74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df"


def load_baseline_solver(repo: Path):
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:global_solver_candidate.py"],
        cwd=repo, text=True,
    )
    module = types.ModuleType("e14_frozen_baseline")
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
    return {"mean": float(array.mean()),
            "robust": float(array.mean() - 0.5 * folds.std()),
            "folds": folds.tolist()}


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
    methods = ("baseline_sa", "e14_fusion_relaxation")
    scores = {method: [] for method in methods}
    adjacencies = {method: [] for method in methods}
    learned_objectives = {method: [] for method in methods}
    fused_objectives = {method: [] for method in methods}
    runtimes = {method: [] for method in methods}
    preprocessing, rows, failures = [], [], []

    for index in range(args.start, stop):
        seed = 20260818 + index * 100 + args.seed_offset
        row = {"index": index, "stem": str(data["stems"][index])}
        try:
            prep_started = time.perf_counter()
            classical_right, classical_down = classical_mgc_ssd_scores(data["tiles"][index])
            fused_right = fuse_scores(data["right"][index], classical_right)
            fused_down = fuse_scores(data["down"][index], classical_down)
            prep_seconds = time.perf_counter() - prep_started
            preprocessing.append(prep_seconds)
            row["fusion_preprocessing_seconds"] = prep_seconds
            for method, solver, right, down in (
                ("baseline_sa", baseline_solver, data["right"][index], data["down"][index]),
                ("e14_fusion_relaxation", solve_layout, fused_right, fused_down),
            ):
                started = time.perf_counter()
                layout = np.asarray(solver(right, down, data["pos"][index], seed), np.int32)
                elapsed = time.perf_counter() - started
                if layout.shape != (N,) or not np.array_equal(np.sort(layout), np.arange(N)):
                    raise ValueError(f"{method} returned an invalid permutation")
                score = float(structural_similarity(
                    data["target"][index], assemble(data["tiles"][index], layout),
                    channel_axis=2, data_range=255,
                ))
                adj = adjacency(layout, data["truth"][index])
                learned_obj = objective(layout, data["right"][index], data["down"][index],
                                        POSITION_WEIGHT * data["pos"][index])
                fused_obj = objective(layout, fused_right, fused_down,
                                      POSITION_WEIGHT * data["pos"][index])
                if not all(np.isfinite(v) for v in (score, adj, learned_obj, fused_obj, elapsed)):
                    raise FloatingPointError(f"non-finite result from {method}")
                scores[method].append(score); adjacencies[method].append(adj)
                learned_objectives[method].append(learned_obj)
                fused_objectives[method].append(fused_obj); runtimes[method].append(elapsed)
                row[method] = {"ssim": score, "adjacency": adj,
                               "learned_objective": learned_obj,
                               "fused_objective": fused_obj, "seconds": elapsed,
                               "valid_permutation": True}
        except Exception as exc:
            failures.append({"index": index, "error": repr(exc),
                             "traceback": traceback.format_exc()})
            row["failure"] = repr(exc)
        rows.append(row)
        print(json.dumps({"done": index - args.start + 1, "total": stop - args.start,
                          "stem": row["stem"]}), flush=True)

    cases = stop - args.start
    report = {
        "experiment": "E14 fixed E2 raw fusion into unchanged E11 relaxation",
        "baseline_commit": BASELINE_COMMIT, "e2_commit": E2_COMMIT,
        "e11_commit": E11_COMMIT, "cache_sha256": cache_hash,
        "cases": cases, "start": args.start, "seed_offset": args.seed_offset,
        "alpha": ALPHA,
        "selection_inputs": ["raw tiles", "right", "down", "pos"],
        "selection_excludes": ["target", "truth", "SSIM", "adjacency"],
        "fusion_preprocessing_seconds": float(np.sum(preprocessing)),
        "methods": {}, "failures": failures, "images": rows,
    }
    for method in methods:
        if len(scores[method]) != cases:
            report["methods"][method] = {"completed": len(scores[method])}
            continue
        report["methods"][method] = {
            "ssim": summarize(scores[method]),
            "mean_adjacency": float(np.mean(adjacencies[method])),
            "mean_learned_objective": float(np.mean(learned_objectives[method])),
            "mean_fused_objective": float(np.mean(fused_objectives[method])),
            "runtime_seconds": float(np.sum(runtimes[method])),
            "mean_runtime_seconds": float(np.mean(runtimes[method])),
            "valid_permutations": cases,
        }
    if not failures:
        bs, cs = np.asarray(scores[methods[0]]), np.asarray(scores[methods[1]])
        ba, ca = np.asarray(adjacencies[methods[0]]), np.asarray(adjacencies[methods[1]])
        comparison = {
            "robust_ssim_delta": report["methods"][methods[1]]["ssim"]["robust"]
                                 - report["methods"][methods[0]]["ssim"]["robust"],
            "mean_ssim_delta": float(cs.mean() - bs.mean()),
            "mean_adjacency_delta": float(ca.mean() - ba.mean()),
            "ssim_wins": int((cs > bs).sum()),
            "adjacency_wins": int((ca > ba).sum()),
            "candidate_end_to_end_runtime_seconds": float(
                np.sum(preprocessing) + np.sum(runtimes[methods[1]])
            ),
            "candidate_end_to_end_runtime_ratio": float(
                (np.sum(preprocessing) + np.sum(runtimes[methods[1]]))
                / np.sum(runtimes[methods[0]])
            ),
        }
        report["comparison"] = comparison
        report["dual_metric_gate"] = bool(
            comparison["robust_ssim_delta"] > 0
            and comparison["mean_ssim_delta"] > 0
            and comparison["mean_adjacency_delta"] >= 0
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "images"}, indent=2))


if __name__ == "__main__":
    main()
