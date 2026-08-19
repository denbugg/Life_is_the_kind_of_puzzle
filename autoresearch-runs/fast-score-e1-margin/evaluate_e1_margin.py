"""Frozen smoke-32 evaluator for the E1 reciprocal-margin score bonus.

This is an experiment-only wrapper around the unchanged production solver.  E1
changes exactly one input to that solver: it adds a fixed bonus to sufficiently
confident reciprocal top-1 directional edges.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from global_solver_candidate import solve_layout

GRID, TILE, N = 24, 20, 576


def assemble(tiles: np.ndarray, layout: np.ndarray) -> np.ndarray:
    return (
        tiles[layout]
        .reshape(GRID, GRID, TILE, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(GRID * TILE, GRID * TILE, 3)
    )


def adjacency(layout: np.ndarray, truth: np.ndarray) -> float:
    target_of = np.empty(N, np.int32)
    target_of[truth] = np.arange(N)
    board = target_of[layout].reshape(GRID, GRID)
    right = (board[:, 1:] == board[:, :-1] + 1) & (
        board[:, 1:] // GRID == board[:, :-1] // GRID
    )
    down = board[1:] == board[:-1] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def reciprocal_margin_bonus(
    matrix: np.ndarray, *, beta: float = 0.5, threshold: float = 0.5
) -> tuple[np.ndarray, int]:
    """Bonus mutual row/column maxima when both top-1 margins are confident."""
    scores = np.asarray(matrix)
    if scores.shape != (N, N):
        raise ValueError(f"expected {(N, N)} score matrix, got {scores.shape}")

    # Self-edges must not contribute to either a maximum or a margin, regardless
    # of which sentinel value the cache happens to use on its diagonal.
    rank_scores = scores.copy()
    np.fill_diagonal(rank_scores, -np.inf)
    row_best = np.argmax(rank_scores, axis=1)
    col_best = np.argmax(rank_scores, axis=0)
    row_top2 = np.partition(rank_scores, -2, axis=1)[:, -2:]
    col_top2 = np.partition(rank_scores, -2, axis=0)[-2:, :]
    row_margin = row_top2[:, 1] - row_top2[:, 0]
    col_margin = col_top2[1, :] - col_top2[0, :]

    sources = np.arange(N)
    targets = row_best
    keep = (
        (col_best[targets] == sources)
        & (row_margin >= threshold)
        & (col_margin[targets] >= threshold)
    )
    confident_sources = sources[keep]
    confident_targets = targets[keep]

    calibrated = scores.copy()
    calibrated[confident_sources, confident_targets] += beta
    return calibrated, int(keep.sum())


def summarize(values: list[float]) -> dict[str, object]:
    scores = np.asarray(values, np.float64)
    folds = np.asarray([scores[offset::4].mean() for offset in range(4)])
    return {
        "mean": float(scores.mean()),
        "robust": float(scores.mean() - 0.5 * folds.std()),
        "folds": folds.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.cache, mmap_mode="r")
    if args.start < 0 or args.start >= len(data["stems"]):
        raise ValueError(f"invalid start index {args.start}")
    cases = min(args.limit, len(data["stems"]) - args.start)
    rows = {
        "baseline": {"ssim": [], "adjacency": [], "runtime_seconds": []},
        "e1_margin": {"ssim": [], "adjacency": [], "runtime_seconds": []},
    }
    images: list[dict[str, object]] = []
    bonus_edges = {"right": [], "down": []}

    for index in range(args.start, args.start + cases):
        right = np.asarray(data["right"][index])
        down = np.asarray(data["down"][index])
        candidate_right, right_count = reciprocal_margin_bonus(
            right, beta=args.beta, threshold=args.threshold
        )
        candidate_down, down_count = reciprocal_margin_bonus(
            down, beta=args.beta, threshold=args.threshold
        )
        bonus_edges["right"].append(right_count)
        bonus_edges["down"].append(down_count)

        case_row: dict[str, object] = {
            "index": index,
            "stem": str(data["stems"][index]),
            "bonus_edges_right": right_count,
            "bonus_edges_down": down_count,
        }
        for method, method_right, method_down in (
            ("baseline", right, down),
            ("e1_margin", candidate_right, candidate_down),
        ):
            started = time.perf_counter()
            layout = np.asarray(
                solve_layout(
                    method_right,
                    method_down,
                    data["pos"][index],
                    20260818 + index * 100 + args.seed_offset,
                ),
                np.int32,
            )
            elapsed = time.perf_counter() - started
            if layout.shape != (N,) or len(np.unique(layout)) != N:
                raise ValueError(f"invalid permutation from {method} at case {index}")
            score = float(
                structural_similarity(
                    data["target"][index],
                    assemble(data["tiles"][index], layout),
                    channel_axis=2,
                    data_range=255,
                )
            )
            adj = adjacency(layout, data["truth"][index])
            if not np.isfinite(score) or not np.isfinite(adj):
                raise FloatingPointError(f"non-finite metric from {method} at case {index}")
            rows[method]["ssim"].append(score)
            rows[method]["adjacency"].append(adj)
            rows[method]["runtime_seconds"].append(elapsed)
            case_row[f"{method}_ssim"] = score
            case_row[f"{method}_adjacency"] = adj
            case_row[f"{method}_runtime_seconds"] = elapsed
        images.append(case_row)
        print(
            json.dumps(
                {
                    "done": index - args.start + 1,
                    "total": cases,
                    "stem": case_row["stem"],
                    "baseline_ssim": case_row["baseline_ssim"],
                    "e1_margin_ssim": case_row["e1_margin_ssim"],
                }
            ),
            flush=True,
        )

    baseline_ssim = np.asarray(rows["baseline"]["ssim"])
    baseline_adj = np.asarray(rows["baseline"]["adjacency"])
    report: dict[str, object] = {
        "experiment": "E1 reciprocal top-1 margin bonus",
        "cases": cases,
        "start": args.start,
        "beta": args.beta,
        "threshold": args.threshold,
        "seed_offset": args.seed_offset,
        "seed_formula": "20260818 + index * 100 + seed_offset",
        "bonus_edges": {
            direction: {
                "mean": float(np.mean(counts)),
                "min": int(np.min(counts)),
                "max": int(np.max(counts)),
            }
            for direction, counts in bonus_edges.items()
        },
        "methods": {},
        "images": images,
    }
    methods = report["methods"]
    assert isinstance(methods, dict)
    for method in ("baseline", "e1_margin"):
        scores = np.asarray(rows[method]["ssim"])
        adjacencies = np.asarray(rows[method]["adjacency"])
        runtimes = np.asarray(rows[method]["runtime_seconds"])
        methods[method] = {
            "ssim": summarize(rows[method]["ssim"]),
            "mean_adjacency": float(adjacencies.mean()),
            "ssim_wins_vs_baseline": int((scores > baseline_ssim).sum()),
            "adjacency_wins_vs_baseline": int((adjacencies > baseline_adj).sum()),
            "runtime_seconds": {
                "total": float(runtimes.sum()),
                "mean": float(runtimes.mean()),
            },
        }

    baseline_metrics = methods["baseline"]
    candidate_metrics = methods["e1_margin"]
    assert isinstance(baseline_metrics, dict) and isinstance(candidate_metrics, dict)
    baseline_summary = baseline_metrics["ssim"]
    candidate_summary = candidate_metrics["ssim"]
    assert isinstance(baseline_summary, dict) and isinstance(candidate_summary, dict)
    report["delta"] = {
        "mean_ssim": candidate_summary["mean"] - baseline_summary["mean"],
        "robust_ssim": candidate_summary["robust"] - baseline_summary["robust"],
        "mean_adjacency": candidate_metrics["mean_adjacency"]
        - baseline_metrics["mean_adjacency"],
        "runtime_seconds": candidate_metrics["runtime_seconds"]["total"]
        - baseline_metrics["runtime_seconds"]["total"],
    }
    report["promotion_gate_rule"] = (
        "robust_ssim_delta > 0.0005 and mean_ssim_delta > 0 "
        "and mean_adjacency_delta >= 0"
    )
    report["dual_metric_gate"] = bool(
        report["delta"]["robust_ssim"] > 0.0005
        and report["delta"]["mean_ssim"] > 0
        and report["delta"]["mean_adjacency"] >= 0
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
