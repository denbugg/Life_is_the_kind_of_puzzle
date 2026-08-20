"""Paired frozen-cache evaluator for E19 against the verified E14 layout solver."""
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
E14 = REPO / "autoresearch-runs" / "e14-fusion-relaxation"
sys.path[:0] = [str(REPO), str(E14), str(HERE)]

from e19_nlm_dual_view import (  # noqa: E402
    ALPHA,
    CLASSICAL_RAW_WEIGHT,
    NLM_H,
    NLM_SEARCH_WINDOW,
    NLM_TEMPLATE_WINDOW,
    dual_view_classical_scores,
)
from e2_raw_fusion import classical_mgc_ssd_scores, fuse_scores  # noqa: E402
from global_solver_candidate import POSITION_WEIGHT, objective, solve_layout  # noqa: E402

GRID, TILE, N = 24, 20, 576
E14_COMMIT = "2087f8d4025d6aede1593b8f72506b2b9ed135a0"
EXPECTED_CACHE_SHA256 = "74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df"


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
    data = np.load(args.cache, mmap_mode="r")
    stop = min(args.start + args.limit, len(data["stems"]))
    methods = ("e14_raw_classical", "e19_raw_nlm_dual_view")
    scores = {method: [] for method in methods}
    adjacencies = {method: [] for method in methods}
    learned_objectives = {method: [] for method in methods}
    fused_objectives = {method: [] for method in methods}
    runtimes = {method: [] for method in methods}
    raw_preprocessing, nlm_preprocessing, rows, failures = [], [], [], []

    for index in range(args.start, stop):
        seed = 20260818 + index * 100 + args.seed_offset
        row = {"index": index, "stem": str(data["stems"][index])}
        try:
            raw_started = time.perf_counter()
            raw_right, raw_down = classical_mgc_ssd_scores(data["tiles"][index])
            e14_right = fuse_scores(data["right"][index], raw_right)
            e14_down = fuse_scores(data["down"][index], raw_down)
            raw_seconds = time.perf_counter() - raw_started
            raw_preprocessing.append(raw_seconds)

            nlm_started = time.perf_counter()
            dual_right, dual_down = dual_view_classical_scores(
                data["tiles"][index], (raw_right, raw_down)
            )
            e19_right = fuse_scores(data["right"][index], dual_right)
            e19_down = fuse_scores(data["down"][index], dual_down)
            nlm_seconds = time.perf_counter() - nlm_started
            nlm_preprocessing.append(nlm_seconds)
            row["raw_classical_seconds"] = raw_seconds
            row["nlm_view_seconds"] = nlm_seconds

            for method, right, down in (
                (methods[0], e14_right, e14_down),
                (methods[1], e19_right, e19_down),
            ):
                started = time.perf_counter()
                layout = np.asarray(solve_layout(right, down, data["pos"][index], seed), np.int32)
                elapsed = time.perf_counter() - started
                if layout.shape != (N,) or not np.array_equal(np.sort(layout), np.arange(N)):
                    raise ValueError(f"{method} returned an invalid permutation")
                # Target and truth appear only below, after target-free layout selection.
                score = float(structural_similarity(
                    data["target"][index], assemble(data["tiles"][index], layout),
                    channel_axis=2, data_range=255,
                ))
                adj = adjacency(layout, data["truth"][index])
                learned_obj = objective(layout, data["right"][index], data["down"][index],
                                        POSITION_WEIGHT * data["pos"][index])
                fused_obj = objective(layout, right, down,
                                      POSITION_WEIGHT * data["pos"][index])
                if not all(np.isfinite(v) for v in (score, adj, learned_obj, fused_obj, elapsed)):
                    raise FloatingPointError(f"non-finite result from {method}")
                scores[method].append(score)
                adjacencies[method].append(adj)
                learned_objectives[method].append(learned_obj)
                fused_objectives[method].append(fused_obj)
                runtimes[method].append(elapsed)
                row[method] = {
                    "ssim": score,
                    "adjacency": adj,
                    "learned_objective": learned_obj,
                    "fused_objective": fused_obj,
                    "seconds": elapsed,
                    "valid_permutation": True,
                }
        except Exception as exc:
            failures.append({
                "index": index,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            row["failure"] = repr(exc)
        rows.append(row)
        print(json.dumps({
            "done": index - args.start + 1,
            "total": stop - args.start,
            "stem": row["stem"],
        }), flush=True)

    cases = stop - args.start
    report = {
        "experiment": "E19 fixed raw/NLM-h9 dual classical view into unchanged E14",
        "e14_commit": E14_COMMIT,
        "cache_sha256": cache_hash,
        "cases": cases,
        "start": args.start,
        "seed_offset": args.seed_offset,
        "alpha": ALPHA,
        "classical_raw_weight": CLASSICAL_RAW_WEIGHT,
        "nlm": {
            "h": NLM_H,
            "template_window": NLM_TEMPLATE_WINDOW,
            "search_window": NLM_SEARCH_WINDOW,
            "scope": "independent raw tiles",
        },
        "selection_inputs": ["raw tiles", "right", "down", "pos", "seed"],
        "selection_excludes": ["target", "truth", "SSIM", "adjacency"],
        "output_pixels": "raw tiles",
        "raw_classical_seconds": float(np.sum(raw_preprocessing)),
        "nlm_view_seconds": float(np.sum(nlm_preprocessing)),
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
            "mean_learned_objective": float(np.mean(learned_objectives[method])),
            "mean_fused_objective": float(np.mean(fused_objectives[method])),
            "solver_runtime_seconds": float(np.sum(runtimes[method])),
            "valid_permutations": cases,
        }
    if not failures:
        baseline, candidate = methods
        bs = np.asarray(scores[baseline]); cs = np.asarray(scores[candidate])
        ba = np.asarray(adjacencies[baseline]); ca = np.asarray(adjacencies[candidate])
        baseline_e2e = float(np.sum(raw_preprocessing) + np.sum(runtimes[baseline]))
        candidate_e2e = float(
            np.sum(raw_preprocessing) + np.sum(nlm_preprocessing) + np.sum(runtimes[candidate])
        )
        comparison = {
            "robust_ssim_delta": (
                report["methods"][candidate]["ssim"]["robust"]
                - report["methods"][baseline]["ssim"]["robust"]
            ),
            "mean_ssim_delta": float(cs.mean() - bs.mean()),
            "mean_adjacency_delta": float(ca.mean() - ba.mean()),
            "ssim_wins": int((cs > bs).sum()),
            "adjacency_wins": int((ca > ba).sum()),
            "baseline_end_to_end_runtime_seconds": baseline_e2e,
            "candidate_end_to_end_runtime_seconds": candidate_e2e,
            "candidate_end_to_end_runtime_ratio": candidate_e2e / baseline_e2e,
        }
        report["comparison"] = comparison
        report["smoke16_gate"] = bool(
            cases == 16
            and comparison["robust_ssim_delta"] >= 0.0005
            and comparison["mean_ssim_delta"] > 0
            and comparison["mean_adjacency_delta"] >= 0
            and comparison["candidate_end_to_end_runtime_ratio"] <= 2.0
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
