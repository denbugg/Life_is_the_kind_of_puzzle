"""Cached-score gate for atomic two-side plaquette growth.

This evaluator never trains or re-scores tiles.  It consumes the existing
``full_graph_cache/image_XXXX_k64.npz`` files, verifies their row/permutation
contracts, runs the deterministic solver, and reports the predeclared early
precision/coverage/runtime kill gates.

Example:
    python src/eval_two_side_growth.py --images 50,51,52 --top-k 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from config import GRID, NFRAG, WORK_ROOT
from placement_metrics import neighbour_accuracy, placement_accuracy
from two_side_growth import (
    DirectionalTopK,
    early_gate_metrics,
    enumerate_plaquettes,
    fixed_gate_checks,
    grow_plaquettes,
    pack_components,
)


def _parse_images(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("--images must select at least one cached image")
    if len(result) != len(set(result)):
        raise ValueError("--images contains duplicates")
    return result


def _reshape_scores(stored: Any, count: int) -> np.ndarray:
    values = np.asarray(stored["candidate_scores"], dtype=np.float64)
    candidate_ids = np.asarray(stored["candidate_ids"])
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != count:
        raise ValueError("candidate_ids must have shape (N,K)")
    width = int(candidate_ids.shape[1])
    if values.shape == (count * 4, width):
        values = values.reshape(count, 4, width)
    elif values.shape != (count, 4, width):
        raise ValueError(
            "candidate_scores must have shape (N*4,K) or (N,4,K), "
            f"got {values.shape}"
        )
    return values


def _validate_cache_contract(stored: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_ids = np.asarray(stored["candidate_ids"], dtype=np.int64)
    count = int(candidate_ids.shape[0])
    if count != NFRAG:
        raise ValueError(f"cache has {count} tiles, expected {NFRAG}")
    scores = _reshape_scores(stored, count)
    permutation = np.asarray(stored["permutation"], dtype=np.int64)
    if permutation.shape != (count,) or not np.array_equal(
        np.sort(permutation), np.arange(count, dtype=np.int64)
    ):
        raise ValueError("cache permutation is not tile->clean-cell bijection")

    # Full-graph caches record all rows in anchor-major/direction-minor order.
    # Refuse silently transposed U/D/L/R rows when metadata is available.
    if "anchors" in stored.files and "directions" in stored.files:
        anchors = np.asarray(stored["anchors"], dtype=np.int64)
        directions = np.asarray(stored["directions"], dtype=np.int64)
        expected_anchors = np.repeat(np.arange(count, dtype=np.int64), 4)
        expected_directions = np.tile(np.arange(4, dtype=np.int64), count)
        if not np.array_equal(anchors, expected_anchors):
            raise ValueError("cache rows are not anchor-major")
        if not np.array_equal(directions, expected_directions):
            raise ValueError("cache rows are not ordered UP,DOWN,LEFT,RIGHT")
    return candidate_ids, scores, permutation


def evaluate_cache(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    with np.load(path) as stored:
        candidate_ids, scores, permutation = _validate_cache_contract(stored)

    start = time.perf_counter()
    graph = DirectionalTopK.from_candidate_rows(
        candidate_ids,
        scores,
        top_k=args.top_k,
        missing_logp=args.missing_logp,
    )
    graph_seconds = time.perf_counter() - start

    start = time.perf_counter()
    motifs = enumerate_plaquettes(graph, max_per_elbow=args.max_per_elbow)
    enumerate_seconds = time.perf_counter() - start

    start = time.perf_counter()
    growth = grow_plaquettes(
        NFRAG,
        GRID,
        motifs,
        minimum_edge=args.minimum_edge,
        growth_min_corners=args.growth_min_corners,
    )
    growth_seconds = time.perf_counter() - start

    start = time.perf_counter()
    packed = pack_components(growth.dsu, graph)
    packing_seconds = time.perf_counter() - start

    truth_board = np.argsort(permutation)
    neighbour, right, down = neighbour_accuracy(packed.placement, truth_board)
    placement, _ = placement_accuracy(packed.placement, truth_board)
    metrics = early_gate_metrics(motifs, growth, permutation)
    solver_seconds = graph_seconds + enumerate_seconds + growth_seconds + packing_seconds
    metrics.update(
        {
            "placement": float(placement),
            "neighbour": float(neighbour),
            "right": float(right),
            "down": float(down),
            "rigid_components_placed": float(packed.rigid_components_placed),
            "rigid_tiles_placed": float(packed.rigid_tiles_placed),
            "hungarian_tiles": float(packed.hungarian_tiles),
            "graph_seconds": graph_seconds,
            "enumerate_seconds": enumerate_seconds,
            "growth_seconds": growth_seconds,
            "packing_seconds": packing_seconds,
            "solver_seconds": solver_seconds,
        }
    )
    return {
        "cache": str(path),
        "metrics": metrics,
        "rejections": growth.rejection_counts,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([row["metrics"][key] for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument("--images", default="50,51,52,53,54,55")
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-per-elbow", type=int, default=64)
    parser.add_argument("--minimum-edge", type=float, default=-20.0)
    parser.add_argument("--missing-logp", type=float, default=-20.0)
    parser.add_argument("--growth-min-corners", type=int, default=2)
    parser.add_argument("--maximum-runtime", type=float, default=2.0)
    parser.add_argument("--minimum-proposal-recall", type=float, default=0.20)
    parser.add_argument("--minimum-motif-precision", type=float, default=0.95)
    parser.add_argument("--minimum-seed-coverage", type=float, default=0.15)
    parser.add_argument("--minimum-grown-coverage", type=float, default=0.25)
    parser.add_argument("--minimum-edge-precision", type=float, default=0.97)
    parser.add_argument("--minimum-largest-pure", type=float, default=12.0)
    parser.add_argument("--minimum-worst-motif-precision", type=float, default=0.85)
    parser.add_argument("--maximum-motifs", type=int, default=150_000)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "two_side_growth_gate.json",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= args.candidate_k:
        raise ValueError("--top-k must lie in [1,--candidate-k]")

    rows = []
    for image in _parse_images(args.images):
        path = args.cache_dir / f"image_{image:04d}_k{args.candidate_k}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        row = evaluate_cache(path, args)
        rows.append(row)
        print(json.dumps({"image": image, **row["metrics"]}), flush=True)

    aggregate = {
        key: _mean(rows, key)
        for key in (
            "true_plaquette_proposal_recall",
            "exact_seed_motif_precision",
            "seed_tile_coverage",
            "certified_edge_precision",
            "pure_nontrivial_tile_coverage",
            "largest_pure_component",
            "placement",
            "neighbour",
            "solver_seconds",
            "enumerated_motifs",
        )
    }
    # The predeclared size gate is a median-scene contract, not a mean that a
    # single giant component can dominate.
    aggregate["largest_pure_component"] = float(
        np.median([row["metrics"]["largest_pure_component"] for row in rows])
    )
    structural_checks = fixed_gate_checks(
        aggregate,
        motif_precision=args.minimum_motif_precision,
        seed_coverage=args.minimum_seed_coverage,
        grown_pure_coverage=args.minimum_grown_coverage,
        edge_precision=args.minimum_edge_precision,
        largest_pure=args.minimum_largest_pure,
    )
    checks = {
        "proposal_recall": aggregate["true_plaquette_proposal_recall"]
        >= args.minimum_proposal_recall,
        **structural_checks,
        "worst_motif_precision": min(
            row["metrics"]["exact_seed_motif_precision"] for row in rows
        )
        >= args.minimum_worst_motif_precision,
        "runtime": max(row["metrics"]["solver_seconds"] for row in rows)
        <= args.maximum_runtime,
        "motif_cap": max(row["metrics"]["enumerated_motifs"] for row in rows)
        <= args.maximum_motifs,
    }
    report = {
        "experiment": "atomic_two_side_plaquette_growth",
        "status": "pass" if all(checks.values()) else "fail",
        "config": {
            "images": args.images,
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "max_per_elbow": args.max_per_elbow,
            "minimum_edge": args.minimum_edge,
            "growth_min_corners": args.growth_min_corners,
        },
        "aggregate": aggregate,
        "checks": checks,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
