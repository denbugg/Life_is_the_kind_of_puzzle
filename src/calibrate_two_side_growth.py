"""Fail-closed structure-only calibration for atomic two-side growth.

Only the predeclared calibration caches ``image_0010`` through ``image_0017``
are accepted.  For every top-k value, each scene is enumerated exactly once;
the resulting motifs are then replayed through a small deterministic grid of
label-free structural cutoffs.  Labels are used only to measure and select a
configuration *inside this calibration split*.

No board packing, Hungarian completion, neighbour metric, or SSIM is run here.
If no operating point reaches both 0.95 exact seed precision and 0.15 seed-tile
coverage, the report contains no selected configuration and the process exits
with status 2.

Example:
    python src/calibrate_two_side_growth.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config import NFRAG, WORK_ROOT
from two_side_growth import (
    DirectionalTopK,
    Plaquette,
    enumerate_plaquettes,
    plaquette_sort_key,
    true_plaquette_keys,
)


CALIBRATION_IMAGES = tuple(range(10, 18))


@dataclass(frozen=True, order=True)
class CalibrationConfig:
    top_k: int
    minimum_edge: float
    maximum_reciprocal_rank_sum: int

    @property
    def key(self) -> str:
        edge = format(self.minimum_edge, ".6g")
        return f"k{self.top_k}:edge{edge}:rank{self.maximum_reciprocal_rank_sum}"


def parse_int_grid(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item < 1 for item in result):
        raise ValueError("integer grid must contain positive values")
    return result


def parse_float_grid(value: str, *, allow_inf: bool = False) -> list[float]:
    parsed = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in ("inf", "+inf", "infinity"):
            if not allow_inf:
                raise ValueError("infinity is not allowed in this grid")
            parsed.append(float("inf"))
        else:
            parsed.append(float(item))
    if not parsed:
        raise ValueError("float grid must not be empty")
    return sorted(set(parsed))


def build_config_grid(
    top_ks: Sequence[int],
    minimum_edges: Sequence[float],
    maximum_mean_reciprocal_ranks: Sequence[float],
) -> list[CalibrationConfig]:
    """Build and deduplicate a fixed structural threshold grid."""

    result = set()
    for top_k in top_ks:
        if top_k < 1:
            raise ValueError("top-k values must be positive")
        maximum_possible = 8 * int(top_k)
        for minimum_edge in minimum_edges:
            if not math.isfinite(float(minimum_edge)):
                raise ValueError("minimum-edge cutoffs must be finite")
            for mean_rank in maximum_mean_reciprocal_ranks:
                if mean_rank <= 0:
                    raise ValueError("reciprocal-rank cutoffs must be positive")
                rank_sum = (
                    maximum_possible
                    if math.isinf(float(mean_rank))
                    else min(maximum_possible, int(round(8 * float(mean_rank))))
                )
                result.add(
                    CalibrationConfig(
                        top_k=int(top_k),
                        minimum_edge=float(minimum_edge),
                        maximum_reciprocal_rank_sum=int(rank_sum),
                    )
                )
    return sorted(
        result,
        key=lambda item: (
            item.top_k,
            -item.minimum_edge,
            item.maximum_reciprocal_rank_sum,
        ),
    )


def select_strict_tier_a_seeds(
    motifs: Sequence[Plaquette],
    count: int,
    config: CalibrationConfig,
    *,
    motifs_are_sorted: bool = False,
) -> list[Plaquette]:
    """Greedy fresh-seed replay without constructing or packing a board.

    A fresh seed is an atomic 2x2 and therefore claims all four tiles.  A motif
    which overlaps an earlier seed is not a fresh seed; conditional growth is
    deliberately outside this precision/coverage calibration.
    """

    claimed = np.zeros(int(count), dtype=bool)
    accepted: list[Plaquette] = []
    ordered = motifs if motifs_are_sorted else sorted(motifs, key=plaquette_sort_key)
    for motif in ordered:
        if not motif.tier_a:
            continue
        if motif.min_edge < config.minimum_edge:
            continue
        if motif.reciprocal_rank_sum > config.maximum_reciprocal_rank_sum:
            continue
        indices = np.asarray(motif.tiles, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= count):
            raise ValueError("motif tile id lies outside the bag")
        if bool(claimed[indices].any()):
            continue
        claimed[indices] = True
        accepted.append(motif)
    return accepted


def evaluate_seed_selection(
    motifs: Sequence[Plaquette],
    truth: set[tuple[int, int, int, int]],
    count: int,
    config: CalibrationConfig,
    *,
    motifs_are_sorted: bool = False,
    proposal_recall: float | None = None,
) -> dict[str, float]:
    start = time.perf_counter()
    accepted = select_strict_tier_a_seeds(
        motifs,
        count,
        config,
        motifs_are_sorted=motifs_are_sorted,
    )
    selection_seconds = time.perf_counter() - start
    exact = sum(motif.tiles in truth for motif in accepted)
    if proposal_recall is None:
        proposed = {motif.tiles for motif in motifs}
        proposal_recall = len(proposed & truth) / max(1, len(truth))
    return {
        "accepted": float(len(accepted)),
        "exact": float(exact),
        "precision": exact / max(1, len(accepted)),
        "seed_tile_coverage": 4 * len(accepted) / count,
        "accepted_true_seed_recall": exact / max(1, len(truth)),
        "proposal_recall": float(proposal_recall),
        "selection_seconds": selection_seconds,
    }


def aggregate_config_rows(
    config: CalibrationConfig,
    rows: Sequence[dict[str, float]],
    *,
    shared_seconds: Sequence[float],
    minimum_precision: float,
    minimum_coverage: float,
) -> dict[str, Any]:
    accepted = float(sum(row["accepted"] for row in rows))
    exact = float(sum(row["exact"] for row in rows))
    precision = exact / max(1.0, accepted)
    coverage = float(np.mean([row["seed_tile_coverage"] for row in rows]))
    proposal_recall = float(np.mean([row["proposal_recall"] for row in rows]))
    selection_seconds = float(np.mean([row["selection_seconds"] for row in rows]))
    shared_mean = float(np.mean(shared_seconds))
    nonempty_precision = [row["precision"] for row in rows if row["accepted"] > 0]
    result: dict[str, Any] = {
        "key": config.key,
        "config": asdict(config),
        "accepted": accepted,
        "exact": exact,
        "precision": precision,
        "mean_seed_tile_coverage": coverage,
        "mean_proposal_recall": proposal_recall,
        "mean_accepted_true_seed_recall": float(
            np.mean([row["accepted_true_seed_recall"] for row in rows])
        ),
        "worst_nonempty_scene_precision": float(min(nonempty_precision))
        if nonempty_precision
        else 0.0,
        "nonempty_scenes": int(sum(row["accepted"] > 0 for row in rows)),
        "mean_shared_graph_enumeration_seconds": shared_mean,
        "mean_selection_seconds": selection_seconds,
        "estimated_mean_seconds": shared_mean + selection_seconds,
        "passes_precision_coverage": precision >= minimum_precision
        and coverage >= minimum_coverage,
    }
    return result


def pareto_frontier(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-dominated precision/coverage/recall/runtime operating points."""

    frontier = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = (
                other["precision"] >= candidate["precision"]
                and other["mean_seed_tile_coverage"]
                >= candidate["mean_seed_tile_coverage"]
                and other["mean_proposal_recall"] >= candidate["mean_proposal_recall"]
                and other["estimated_mean_seconds"]
                <= candidate["estimated_mean_seconds"]
            )
            strictly_better = (
                other["precision"] > candidate["precision"]
                or other["mean_seed_tile_coverage"]
                > candidate["mean_seed_tile_coverage"]
                or other["mean_proposal_recall"] > candidate["mean_proposal_recall"]
                or other["estimated_mean_seconds"]
                < candidate["estimated_mean_seconds"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda row: (
            -row["precision"],
            -row["mean_seed_tile_coverage"],
            -row["mean_proposal_recall"],
            row["estimated_mean_seconds"],
            row["key"],
        ),
    )


def select_passing_config(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose only among points which passed both predeclared gates."""

    passing = [row for row in rows if row["passes_precision_coverage"]]
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            -row["mean_seed_tile_coverage"],
            -row["precision"],
            -row["mean_proposal_recall"],
            row["estimated_mean_seconds"],
            row["config"]["top_k"],
            -row["config"]["minimum_edge"],
            row["config"]["maximum_reciprocal_rank_sum"],
        ),
    )


def _reshape_cache(stored: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(stored["candidate_ids"], dtype=np.int64)
    if ids.ndim != 2 or ids.shape[0] != NFRAG:
        raise ValueError("candidate_ids must have shape (576,K)")
    width = int(ids.shape[1])
    scores = np.asarray(stored["candidate_scores"], dtype=np.float64)
    if scores.shape == (NFRAG * 4, width):
        scores = scores.reshape(NFRAG, 4, width)
    elif scores.shape != (NFRAG, 4, width):
        raise ValueError("candidate_scores has an unsupported row layout")
    permutation = np.asarray(stored["permutation"], dtype=np.int64)
    if permutation.shape != (NFRAG,) or not np.array_equal(
        np.sort(permutation), np.arange(NFRAG, dtype=np.int64)
    ):
        raise ValueError("permutation must map every input tile to one clean cell")
    if "anchors" in stored.files and "directions" in stored.files:
        expected_anchors = np.repeat(np.arange(NFRAG, dtype=np.int64), 4)
        expected_directions = np.tile(np.arange(4, dtype=np.int64), NFRAG)
        if not np.array_equal(stored["anchors"], expected_anchors):
            raise ValueError("cache rows are not anchor-major")
        if not np.array_equal(stored["directions"], expected_directions):
            raise ValueError("cache directions are not UP,DOWN,LEFT,RIGHT")
    return ids, scores, permutation


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    top_ks = parse_int_grid(args.top_ks)
    minimum_edges = parse_float_grid(args.minimum_edges)
    mean_ranks = parse_float_grid(args.maximum_mean_reciprocal_ranks, allow_inf=True)
    if max(top_ks) > args.candidate_k:
        raise ValueError("top-k grid exceeds --candidate-k")
    configs = build_config_grid(top_ks, minimum_edges, mean_ranks)
    configs_by_k: dict[int, list[CalibrationConfig]] = {
        top_k: [config for config in configs if config.top_k == top_k]
        for top_k in top_ks
    }
    per_config: dict[str, list[dict[str, float]]] = {
        config.key: [] for config in configs
    }
    shared_runtime: dict[str, list[float]] = {config.key: [] for config in configs}
    enumeration_rows = []
    total_start = time.perf_counter()

    for image in CALIBRATION_IMAGES:
        path = args.cache_dir / f"image_{image:04d}_k{args.candidate_k}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as stored:
            candidate_ids, scores, permutation = _reshape_cache(stored)
        truth = true_plaquette_keys(permutation, 24)
        for top_k in top_ks:
            start = time.perf_counter()
            graph = DirectionalTopK.from_candidate_rows(
                candidate_ids,
                scores,
                top_k=top_k,
                missing_logp=args.missing_logp,
            )
            motifs = enumerate_plaquettes(graph, max_per_elbow=args.max_per_elbow)
            shared_seconds = time.perf_counter() - start
            if len(motifs) > args.maximum_motifs:
                raise RuntimeError(
                    f"image {image} top-k {top_k} produced {len(motifs)} motifs, "
                    f"above cap {args.maximum_motifs}"
                )
            proposal_recall = len({motif.tiles for motif in motifs} & truth) / len(truth)
            enumeration_rows.append(
                {
                    "image": image,
                    "top_k": top_k,
                    "motifs": len(motifs),
                    "tier_a_motifs": sum(motif.tier_a for motif in motifs),
                    "proposal_recall": proposal_recall,
                    "seconds": shared_seconds,
                }
            )
            for config in configs_by_k[top_k]:
                row = evaluate_seed_selection(
                    motifs,
                    truth,
                    NFRAG,
                    config,
                    motifs_are_sorted=True,
                    proposal_recall=proposal_recall,
                )
                per_config[config.key].append(row)
                shared_runtime[config.key].append(shared_seconds)
        print(json.dumps({"calibration_image": image, "of": list(CALIBRATION_IMAGES)}), flush=True)

    summaries = [
        aggregate_config_rows(
            config,
            per_config[config.key],
            shared_seconds=shared_runtime[config.key],
            minimum_precision=args.minimum_precision,
            minimum_coverage=args.minimum_coverage,
        )
        for config in configs
    ]
    selected = select_passing_config(summaries)
    return {
        "experiment": "two_side_structure_only_calibration",
        "status": "pass" if selected is not None else "fail_closed",
        "calibration_images": list(CALIBRATION_IMAGES),
        "outside_calibration_labels_used": False,
        "enumerations": len(enumeration_rows),
        "expected_enumerations": len(CALIBRATION_IMAGES) * len(top_ks),
        "thresholds": {
            "minimum_precision": args.minimum_precision,
            "minimum_seed_tile_coverage": args.minimum_coverage,
        },
        "selected_config": selected["config"] if selected is not None else None,
        "selected_metrics": selected,
        "pareto": pareto_frontier(summaries),
        "grid": summaries,
        "enumeration_rows": enumeration_rows,
        "total_seconds": time.perf_counter() - total_start,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--top-ks", default="4,8,12")
    parser.add_argument("--minimum-edges", default="-6,-4,-3,-2,-1,-0.5")
    parser.add_argument(
        "--maximum-mean-reciprocal-ranks", default="1,1.5,2,3,inf"
    )
    parser.add_argument("--missing-logp", type=float, default=-20.0)
    parser.add_argument("--max-per-elbow", type=int, default=64)
    parser.add_argument("--maximum-motifs", type=int, default=150_000)
    parser.add_argument("--minimum-precision", type=float, default=0.95)
    parser.add_argument("--minimum-coverage", type=float, default=0.15)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "two_side_growth_calibration.json",
    )
    args = parser.parse_args()
    try:
        report = run_calibration(args)
    except Exception as error:
        report = {
            "experiment": "two_side_structure_only_calibration",
            "status": "fail_closed",
            "calibration_images": list(CALIBRATION_IMAGES),
            "outside_calibration_labels_used": False,
            "selected_config": None,
            "error": f"{type(error).__name__}: {error}",
        }
        _write_report(args.report, report)
        raise SystemExit(2) from error
    _write_report(args.report, report)
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
