"""Frozen evaluator for E2: learned score + fixed MGC/SSD edge auxiliary.

The only solver input changed by E2 is the right/down score matrix. Classical
scores use raw cached tiles (an inference-visible input), never target/truth.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import log_softmax
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from global_solver_candidate import solve_layout

GRID, TILE, N = 24, 20, 576
DUMMY_DIFFS = np.asarray(
    [
        [0, 0, 0],
        [1, 1, 1],
        [-1, -1, -1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ],
    dtype=np.float32,
)


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


def _mahalanobis_gradient_cost(
    source_boundary: np.ndarray,
    source_inner: np.ndarray,
    target_boundary: np.ndarray,
    *,
    batch_size: int = 24,
) -> np.ndarray:
    """Asymmetric MGC for every source/target pair in one orientation."""
    source_boundary = np.asarray(source_boundary, np.float32)
    source_inner = np.asarray(source_inner, np.float32)
    target_boundary = np.asarray(target_boundary, np.float32)
    gradients = source_boundary - source_inner
    means = gradients.mean(axis=1)
    dummy = np.broadcast_to(DUMMY_DIFFS, (N, *DUMMY_DIFFS.shape))
    samples = np.concatenate((gradients, dummy), axis=1).astype(np.float64)
    centered = samples - samples.mean(axis=1, keepdims=True)
    covariance = np.einsum(
        "nki,nkj->nij", centered, centered, optimize=True
    ) / (samples.shape[1] - 1)
    # Dummy differences make the 3x3 covariances invertible, matching the
    # reference MGC implementation while allowing one batched inversion.
    precisions = np.linalg.inv(covariance).astype(np.float32)

    costs = np.empty((N, N), np.float32)
    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        residual = (
            target_boundary[None, :, :, :]
            - source_boundary[start:stop, None, :, :]
            - means[start:stop, None, None, :]
        )
        costs[start:stop] = np.einsum(
            "btkc,bcd,btkd->bt",
            residual,
            precisions[start:stop],
            residual,
            optimize=True,
        )
    return costs


def _ssd_cost(
    source_boundary: np.ndarray,
    target_boundary: np.ndarray,
    *,
    batch_size: int = 24,
) -> np.ndarray:
    source_boundary = np.asarray(source_boundary, np.float32)
    target_boundary = np.asarray(target_boundary, np.float32)
    costs = np.empty((N, N), np.float32)
    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        residual = (
            source_boundary[start:stop, None, :, :]
            - target_boundary[None, :, :, :]
        )
        costs[start:stop] = np.einsum(
            "btkc,btkc->bt", residual, residual, optimize=True
        )
    return costs


def _row_robust_dissimilarity(cost: np.ndarray) -> np.ndarray:
    """Put a cost on a row-wise median/MAD scale, excluding self-edge."""
    cost = np.asarray(cost, np.float32)
    mask = ~np.eye(N, dtype=bool)
    off_diagonal = cost[mask].reshape(N, N - 1)
    median = np.median(off_diagonal, axis=1, keepdims=True)
    mad = np.median(np.abs(off_diagonal - median), axis=1, keepdims=True)
    scaled = (cost - median) / np.maximum(mad, 1e-6)
    np.fill_diagonal(scaled, np.inf)
    return scaled


def _dissimilarity_logp(dissimilarity: np.ndarray) -> np.ndarray:
    """Apply the critic-locked robust calibration to a dissimilarity matrix."""
    dissimilarity = np.asarray(dissimilarity, np.float32)
    mask = ~np.eye(N, dtype=bool)
    off_diagonal = dissimilarity[mask].reshape(N, N - 1)
    median = np.median(off_diagonal, axis=1, keepdims=True)
    mad = np.median(np.abs(off_diagonal - median), axis=1, keepdims=True)
    z = -(dissimilarity - median) / np.maximum(mad, 1e-6)
    np.fill_diagonal(z, -1e4)
    return log_softmax(z, axis=1).astype(np.float32)


def classical_mgc_ssd_scores(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed 50/50 MGC+SSD log-scores for right and down edges."""
    tiles = np.asarray(tiles)
    if tiles.shape != (N, TILE, TILE, 3):
        raise ValueError(f"expected {(N, TILE, TILE, 3)} tiles, got {tiles.shape}")
    pixel = tiles.astype(np.float32)

    left, left_inner = pixel[:, :, 0, :], pixel[:, :, 1, :]
    right, right_inner = pixel[:, :, -1, :], pixel[:, :, -2, :]
    top, top_inner = pixel[:, 0, :, :], pixel[:, 1, :, :]
    bottom, bottom_inner = pixel[:, -1, :, :], pixel[:, -2, :, :]

    right_mgc = _mahalanobis_gradient_cost(right, right_inner, left)
    right_mgc += _mahalanobis_gradient_cost(left, left_inner, right).T
    down_mgc = _mahalanobis_gradient_cost(bottom, bottom_inner, top)
    down_mgc += _mahalanobis_gradient_cost(top, top_inner, bottom).T
    right_ssd = _ssd_cost(right, left)
    down_ssd = _ssd_cost(bottom, top)

    right_dissimilarity = 0.5 * (
        _row_robust_dissimilarity(right_mgc)
        + _row_robust_dissimilarity(right_ssd)
    )
    down_dissimilarity = 0.5 * (
        _row_robust_dissimilarity(down_mgc)
        + _row_robust_dissimilarity(down_ssd)
    )
    return (
        _dissimilarity_logp(right_dissimilarity),
        _dissimilarity_logp(down_dissimilarity),
    )


def fuse_scores(
    learned: np.ndarray, classical: np.ndarray, *, alpha: float = 0.2
) -> np.ndarray:
    """Geometric-probability fusion of two row-wise log-score matrices."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    fused = (1.0 - alpha) * np.asarray(learned) + alpha * np.asarray(classical)
    fused = np.asarray(fused, np.float32)
    np.fill_diagonal(fused, -1e4)
    return fused


def ranks(matrix: np.ndarray, truth: np.ndarray, delta: int) -> list[int]:
    result = []
    for position in range(N):
        if delta == 1 and position % GRID == GRID - 1:
            continue
        if delta == GRID and position >= N - GRID:
            continue
        anchor = int(truth[position])
        neighbour = int(truth[position + delta])
        result.append(1 + int((matrix[anchor] > matrix[anchor, neighbour]).sum()))
    return result


def summarize(values: list[float]) -> dict[str, object]:
    scores = np.asarray(values, np.float64)
    folds = np.asarray([scores[offset::4].mean() for offset in range(4)])
    return {
        "mean": float(scores.mean()),
        "robust": float(scores.mean() - 0.5 * folds.std()),
        "folds": folds.tolist(),
    }


def summarize_ranks(values: list[int]) -> dict[str, float]:
    rank = np.asarray(values)
    return {
        "r1": float((rank <= 1).mean()),
        "r5": float((rank <= 5).mean()),
        "r25": float((rank <= 25).mean()),
        "median": float(np.median(rank)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.cache, mmap_mode="r")
    if args.start < 0 or args.start >= len(data["stems"]):
        raise ValueError(f"invalid start index {args.start}")
    cases = min(args.limit, len(data["stems"]) - args.start)
    rows = {
        "baseline": {"ssim": [], "adjacency": [], "runtime_seconds": []},
        "e2_fusion": {"ssim": [], "adjacency": [], "runtime_seconds": []},
    }
    rank_rows = {
        method: {direction: [] for direction in ("right", "down")}
        for method in rows
    }
    preprocessing_seconds: list[float] = []
    images: list[dict[str, object]] = []

    for index in range(args.start, args.start + cases):
        learned_right = np.asarray(data["right"][index])
        learned_down = np.asarray(data["down"][index])
        preprocess_started = time.perf_counter()
        classical_right, classical_down = classical_mgc_ssd_scores(
            data["tiles"][index]
        )
        fused_right = fuse_scores(learned_right, classical_right, alpha=args.alpha)
        fused_down = fuse_scores(learned_down, classical_down, alpha=args.alpha)
        preprocess_elapsed = time.perf_counter() - preprocess_started
        preprocessing_seconds.append(preprocess_elapsed)

        case_row: dict[str, object] = {
            "index": index,
            "stem": str(data["stems"][index]),
            "e2_preprocessing_seconds": preprocess_elapsed,
        }
        for method, method_right, method_down in (
            ("baseline", learned_right, learned_down),
            ("e2_fusion", fused_right, fused_down),
        ):
            rank_rows[method]["right"].extend(
                ranks(method_right, data["truth"][index], 1)
            )
            rank_rows[method]["down"].extend(
                ranks(method_down, data["truth"][index], GRID)
            )
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
            if (
                layout.shape != (N,)
                or len(np.unique(layout)) != N
                or layout.min() != 0
                or layout.max() != N - 1
            ):
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
                    "e2_fusion_ssim": case_row["e2_fusion_ssim"],
                    "preprocessing_seconds": preprocess_elapsed,
                }
            ),
            flush=True,
        )

    baseline_ssim = np.asarray(rows["baseline"]["ssim"])
    baseline_adj = np.asarray(rows["baseline"]["adjacency"])
    report: dict[str, object] = {
        "experiment": "E2 fixed 50/50 MGC+SSD score fusion",
        "cases": cases,
        "start": args.start,
        "alpha": args.alpha,
        "seed_offset": args.seed_offset,
        "seed_formula": "20260818 + index * 100 + seed_offset",
        "classical_input": "raw cached inference-visible tiles",
        "classical_mix": "d = 0.5 row-robust(MGC) + 0.5 row-robust(SSD)",
        "classical_calibration": "z = -(d-row_median)/max(row_MAD,1e-6); log_softmax(z)",
        "fusion": "(1-alpha) * learned_logp + alpha * classical_logp",
        "preprocessing_seconds": {
            "total": float(np.sum(preprocessing_seconds)),
            "mean": float(np.mean(preprocessing_seconds)),
        },
        "methods": {},
        "images": images,
    }
    methods = report["methods"]
    assert isinstance(methods, dict)
    for method in rows:
        scores = np.asarray(rows[method]["ssim"])
        adjacencies = np.asarray(rows[method]["adjacency"])
        runtimes = np.asarray(rows[method]["runtime_seconds"])
        methods[method] = {
            "ssim": summarize(rows[method]["ssim"]),
            "mean_adjacency": float(adjacencies.mean()),
            "ssim_wins_vs_baseline": int((scores > baseline_ssim).sum()),
            "adjacency_wins_vs_baseline": int((adjacencies > baseline_adj).sum()),
            "rank": {
                direction: summarize_ranks(rank_rows[method][direction])
                for direction in ("right", "down")
            },
            "solver_runtime_seconds": {
                "total": float(runtimes.sum()),
                "mean": float(runtimes.mean()),
            },
        }

    baseline_metrics = methods["baseline"]
    candidate_metrics = methods["e2_fusion"]
    assert isinstance(baseline_metrics, dict) and isinstance(candidate_metrics, dict)
    baseline_summary = baseline_metrics["ssim"]
    candidate_summary = candidate_metrics["ssim"]
    assert isinstance(baseline_summary, dict) and isinstance(candidate_summary, dict)
    report["delta"] = {
        "mean_ssim": candidate_summary["mean"] - baseline_summary["mean"],
        "robust_ssim": candidate_summary["robust"] - baseline_summary["robust"],
        "mean_adjacency": candidate_metrics["mean_adjacency"]
        - baseline_metrics["mean_adjacency"],
        "solver_runtime_seconds": candidate_metrics["solver_runtime_seconds"]["total"]
        - baseline_metrics["solver_runtime_seconds"]["total"],
    }
    report["candidate_end_to_end_runtime_seconds"] = float(
        report["preprocessing_seconds"]["total"]
        + candidate_metrics["solver_runtime_seconds"]["total"]
    )
    report["candidate_end_to_end_runtime_ratio"] = float(
        report["candidate_end_to_end_runtime_seconds"]
        / baseline_metrics["solver_runtime_seconds"]["total"]
    )
    report["promotion_gate_rule"] = (
        "robust_ssim_delta > 0.0005 and mean_ssim_delta > 0 "
        "and mean_adjacency_delta >= 0 and candidate end-to-end runtime <= 1.1x"
    )
    report["joint_metric_gate"] = bool(
        report["delta"]["robust_ssim"] > 0.0005
        and report["delta"]["mean_ssim"] > 0
        and report["delta"]["mean_adjacency"] >= 0
        and report["candidate_end_to_end_runtime_ratio"] <= 1.1
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
