"""Gate seed-conditioned joint photometric calibration before seam scoring.

Each fragment is independently brightness/contrast corrupted.  A tempting
explanation for the weak seam models is therefore that a *small set of very
reliable physical edges* could estimate one RGB affine correction per tile,
after which ordinary seam evidence becomes useful.  This file tests that
claim, rather than assuming it.

For each connected component of the supplied seed graph it fits

``corrected[tile, channel] = scale[tile, channel] * raw + bias[tile, channel]``

from only two-pixel strips on the selected seams.  The fit is a bounded Huber
IRLS solve, with an identity prior, and components never exchange estimates.
The two-sided linear boundary extrapolations are important: equating two
adjacent pixels directly would mistake a real image gradient for exposure
error.  The solver is deliberately *not* a learned restoration model.

The diagnostic has two evidence levels:

* ``oracle`` supplies exact physical right/down edges as an upper bound.  Use
  ``--oracle-fit-fraction 0.5`` for a non-leaking edge holdout.
* ``v2`` obtains the existing ``rank_v2w64`` reciprocal high-confidence seeds
  (and applies RSCM slot capacity).  It reports all-edge and seed-excluded
  results, so a fit cannot be credited merely for reusing its own edges.

Every source is evaluated with an untrained full-bag linear-seam rank across
all 575 candidates.  In the ``v2`` route the frozen candidate ranker is also
re-run on the *same raw affinity candidate graph* before and after correction;
its reciprocal precision/coverage is an optional operational check.

This is a bounded gate, not a reconstruction attempt.  Keep this direction
only if a 50%-oracle heldout edge split gives a material full-bag improvement
and the high-confidence v2 route repeats it on non-seed edges.  In practical
terms the recommended minimum is +0.02 absolute non-seed R@1 for oracle and
+0.01 for v2, plus no loss of reciprocal precision at comparable coverage.

Examples
--------

    # No data or model inference: validates the constrained solver itself.
    python src/eval_seed_photometric_calibration.py --smoke

    # CPU-only oracle upper-bound / holdout test (no affinity or ranker).
    python src/eval_seed_photometric_calibration.py --seed-sources oracle --n 4 \
        --oracle-fit-fraction 0.5 --device cpu

    # Run only after the GPU is free.  v2 uses the frozen raw candidate graph.
    python src/eval_seed_photometric_calibration.py --seed-sources oracle,v2 --n 4 \
        --oracle-fit-fraction 0.5 --device cuda \
        --ranker-ckpt artifacts/candidate_rank/rank_v2w64_best.pt
"""
from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from torch import Tensor, nn

from canvas_data import CanvasDataset
from candidate_rank import NUM_DIRECTIONS, neighbor_targets
from config import FS, GRID, NFRAG, SEED
from eval_candidate_rank import load_ranker, mutual_argmax_relations, score_full_graph
from eval_rscm_gate import INVERSE_DIRECTION, PhysicalRelation, rscm_greedy
from imgio import train_val_split
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RANKER = os.path.join(WORKSPACE, "artifacts", "candidate_rank", "rank_v2w64_best.pt")
DEFAULT_AFFINITY_1 = os.path.join(
    WORKSPACE, "artifacts", "macro_affinity", "affinity_r1_1200_best.pt"
)
DEFAULT_AFFINITY_2 = os.path.join(
    WORKSPACE, "artifacts", "macro_affinity", "affinity_r3_1000_best.pt"
)

# Direction of target relative to source.  Keep this independent from a
# particular checkpoint; it is the shared contract used by candidate_rank.
OFFSETS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
CANONICAL_ROTATIONS: tuple[int, ...] = (3, 1, 2, 0)


@dataclass(frozen=True)
class DirectedSeed:
    """A one-way physical proposal used only as a calibration constraint."""

    source: int
    target: int
    direction: int
    weight: float = 1.0

    @property
    def pair(self) -> tuple[int, int]:
        return (self.source, self.target) if self.source < self.target else (self.target, self.source)


@dataclass(frozen=True)
class CalibrationConfig:
    """Numerical safeguards for the per-component Huber IRLS problem."""

    iterations: int = 6
    huber: float = 0.055
    scale_min: float = 0.65
    scale_max: float = 1.45
    bias_limit: float = 0.25
    scale_prior: float = 8.0
    bias_prior: float = 24.0


@dataclass(frozen=True)
class CalibrationResult:
    scale: np.ndarray  # (tiles, 3)
    bias: np.ndarray  # (tiles, 3)
    components: int
    fitted_components: int
    fitted_tiles: int
    fitted_edges: int
    raw_seam_rmse: float
    calibrated_seam_rmse: float
    scale_saturation: float
    bias_saturation: float


@dataclass(frozen=True)
class RankMetrics:
    edges: int
    r1: float
    r5: float
    mrr: float
    mean_rank: float


@dataclass(frozen=True)
class RelationMetrics:
    selected: int
    correct: int
    precision: float
    coverage: float
    denominator: int


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def _parse_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return device


def _parse_sources(value: str) -> tuple[str, ...]:
    sources = tuple(dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip()))
    allowed = {"oracle", "v2"}
    if not sources or any(source not in allowed for source in sources):
        raise argparse.ArgumentTypeError("--seed-sources must be a non-empty subset of oracle,v2")
    return sources


def _component_edges(tile_count: int, seeds: Sequence[DirectedSeed]) -> list[tuple[list[int], list[DirectedSeed]]]:
    """Partition the seed graph; singleton/no-edge tiles intentionally vanish.

    Calibration must not infer a brightness offset from an unrelated component.
    Returning only components with at least one edge makes that property
    explicit; all other output parameters stay exactly at identity/zero.
    """
    union_find = _UnionFind(tile_count)
    for seed in seeds:
        if not (0 <= seed.source < tile_count and 0 <= seed.target < tile_count):
            raise ValueError("seed endpoint outside tile bag")
        if seed.source == seed.target:
            raise ValueError("self seed is not a physical seam")
        if not 0 <= seed.direction < NUM_DIRECTIONS:
            raise ValueError("seed direction must lie in [0,3]")
        union_find.union(seed.source, seed.target)
    by_root: dict[int, list[int]] = defaultdict(list)
    for tile in range(tile_count):
        by_root[union_find.find(tile)].append(tile)
    seed_by_root: dict[int, list[DirectedSeed]] = defaultdict(list)
    for seed in seeds:
        seed_by_root[union_find.find(seed.source)].append(seed)
    return [(by_root[root], seed_by_root[root]) for root in sorted(seed_by_root)]


def _oriented_boundary_pairs(
    tiles: np.ndarray,
    seed: DirectedSeed,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 60 robust strip equations per RGB channel for one seed.

    Each orientation puts ``source`` on the left and ``target`` on the right.
    We use the direct border comparison plus extrapolation in both directions:

    * ``left[-1] == right[0]``;
    * ``2*left[-1] - left[-2] == right[0]``;
    * ``left[-1] == 2*right[0] - right[1]``.

    The second and third rows are the two-pixel seam strips; their affine
    intercept remains exactly one ``bias`` term, so they stay linear in the
    unknown per-tile RGB corrections.
    """
    turns = CANONICAL_ROTATIONS[seed.direction]
    left = np.rot90(tiles[seed.source], turns, axes=(-2, -1))
    right = np.rot90(tiles[seed.target], turns, axes=(-2, -1))
    left_edge = left[:, :, -1]
    left_inner = left[:, :, -2]
    right_edge = right[:, :, 0]
    right_inner = right[:, :, 1]
    left_values = np.concatenate(
        (left_edge, 2.0 * left_edge - left_inner, left_edge), axis=1
    )
    right_values = np.concatenate(
        (right_edge, right_edge, 2.0 * right_edge - right_inner), axis=1
    )
    return left_values.astype(np.float64, copy=False), right_values.astype(np.float64, copy=False)


def _component_system(
    tiles: np.ndarray,
    nodes: Sequence[int],
    edges: Sequence[DirectedSeed],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse linear rows for one connected calibration component."""
    local = {node: index for index, node in enumerate(nodes)}
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    for seed in edges:
        left, right = _oriented_boundary_pairs(tiles, seed)
        count = left.shape[1]
        source_parts.append(np.full(count, local[seed.source], dtype=np.int64))
        target_parts.append(np.full(count, local[seed.target], dtype=np.int64))
        left_parts.append(left)
        right_parts.append(right)
    if not source_parts:
        raise ValueError("fitted component has no seam equations")
    return (
        np.concatenate(source_parts),
        np.concatenate(target_parts),
        np.concatenate(left_parts, axis=1),
        np.concatenate(right_parts, axis=1),
    )


def _solve_component(
    source: np.ndarray,
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    node_count: int,
    config: CalibrationConfig,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit one component independently with projected Huber IRLS.

    ``theta = [scale - 1, bias]`` keeps the identity correction at zero.  A
    diagonal prior resolves the unavoidable global affine gauge without taking
    information from another component.  Projection after each sparse solve
    gives actual hard calibration bounds instead of an aspirational penalty.
    """
    samples = int(source.size)
    if not (target.shape == source.shape and left.shape == (3, samples) and right.shape == left.shape):
        raise ValueError("inconsistent component system shapes")
    # ``cols`` is concatenated by coefficient block (all source scales, then
    # all target scales, ...), so row ids must use the same block layout.
    # ``repeat`` would silently connect coefficients from different pixels.
    rows = np.tile(np.arange(samples, dtype=np.int64), 4)
    cols = np.concatenate((source, target, source + node_count, target + node_count))
    data_common = np.concatenate(
        (np.ones(samples), -np.ones(samples), np.ones(samples), -np.ones(samples))
    )
    prior = diags(
        np.concatenate(
            (
                np.full(node_count, float(config.scale_prior)),
                np.full(node_count, float(config.bias_prior)),
            )
        ),
        format="csc",
    )
    scales = np.ones((node_count, 3), dtype=np.float64)
    biases = np.zeros((node_count, 3), dtype=np.float64)
    raw_residuals: list[np.ndarray] = []
    corrected_residuals: list[np.ndarray] = []
    for channel in range(3):
        x_left = left[channel]
        x_right = right[channel]
        # (x_left - x_right) + x_left*(scale_i-1) - x_right*(scale_j-1)
        # + bias_i - bias_j == 0.
        values = data_common.copy()
        values[:samples] = x_left
        values[samples : 2 * samples] = -x_right
        matrix = coo_matrix((values, (rows, cols)), shape=(samples, 2 * node_count)).tocsr()
        base = x_left - x_right
        theta = np.zeros(2 * node_count, dtype=np.float64)
        for _ in range(config.iterations):
            residual = base + matrix @ theta
            weights = np.minimum(1.0, config.huber / np.maximum(np.abs(residual), 1.0e-8))
            weighted = matrix.multiply(weights[:, None])
            normal = (matrix.T @ weighted).tocsc() + prior
            rhs = -np.asarray(matrix.T @ (weights * base)).reshape(-1)
            solution = np.asarray(spsolve(normal, rhs), dtype=np.float64).reshape(-1)
            if not np.isfinite(solution).all():
                raise FloatingPointError("photometric IRLS produced non-finite parameters")
            solution[:node_count] = np.clip(
                solution[:node_count], config.scale_min - 1.0, config.scale_max - 1.0
            )
            solution[node_count:] = np.clip(
                solution[node_count:], -config.bias_limit, config.bias_limit
            )
            if np.max(np.abs(solution - theta)) < 1.0e-6:
                theta = solution
                break
            theta = solution
        scales[:, channel] = 1.0 + theta[:node_count]
        biases[:, channel] = theta[node_count:]
        raw_residuals.append(base)
        corrected_residuals.append(base + matrix @ theta)
    raw = np.concatenate(raw_residuals)
    corrected = np.concatenate(corrected_residuals)
    return (
        scales,
        biases,
        float(np.sqrt(np.mean(np.square(raw)))),
        float(np.sqrt(np.mean(np.square(corrected)))),
    )


def calibrate_tiles(
    tiles: np.ndarray,
    seeds: Sequence[DirectedSeed],
    config: CalibrationConfig,
) -> CalibrationResult:
    """Fit all seed components separately and return tile-wise RGB correction."""
    if tiles.ndim != 4 or tiles.shape[1:] != (3, FS, FS):
        raise ValueError(f"tiles must have shape (N,3,{FS},{FS}), got {tiles.shape}")
    tile_count = int(tiles.shape[0])
    scale = np.ones((tile_count, 3), dtype=np.float64)
    bias = np.zeros((tile_count, 3), dtype=np.float64)
    raw_errors: list[float] = []
    corrected_errors: list[float] = []
    components = _component_edges(tile_count, seeds)
    fitted_tiles = 0
    fitted_edges = 0
    for nodes, edges in components:
        source, target, left, right = _component_system(tiles, nodes, edges)
        local_scale, local_bias, raw_error, calibrated_error = _solve_component(
            source, target, left, right, len(nodes), config
        )
        node_array = np.asarray(nodes, dtype=np.int64)
        scale[node_array] = local_scale
        bias[node_array] = local_bias
        fitted_tiles += len(nodes)
        fitted_edges += len(edges)
        raw_errors.append(raw_error)
        corrected_errors.append(calibrated_error)
    scale_saturation = np.logical_or(
        np.isclose(scale, config.scale_min, atol=1.0e-5),
        np.isclose(scale, config.scale_max, atol=1.0e-5),
    )
    bias_saturation = np.isclose(np.abs(bias), config.bias_limit, atol=1.0e-5)
    return CalibrationResult(
        scale=scale,
        bias=bias,
        components=len(components),
        fitted_components=len(components),
        fitted_tiles=fitted_tiles,
        fitted_edges=fitted_edges,
        raw_seam_rmse=float(np.mean(raw_errors)) if raw_errors else float("nan"),
        calibrated_seam_rmse=float(np.mean(corrected_errors)) if corrected_errors else float("nan"),
        scale_saturation=float(scale_saturation.mean()),
        bias_saturation=float(bias_saturation.mean()),
    )


def apply_calibration(tiles: np.ndarray, result: CalibrationResult) -> np.ndarray:
    """Apply the bounded correction and retain the ranker's [0,1] contract."""
    corrected = tiles * result.scale[:, :, None, None] + result.bias[:, :, None, None]
    return np.clip(corrected, 0.0, 1.0).astype(np.float32, copy=False)


def oracle_edges(perm: np.ndarray) -> list[DirectedSeed]:
    """Exact right/down edges in shuffled input-tile coordinates."""
    if perm.shape != (NFRAG,):
        raise ValueError(f"perm must be ({NFRAG},), got {perm.shape}")
    inverse = np.empty(NFRAG, dtype=np.int64)
    inverse[perm] = np.arange(NFRAG, dtype=np.int64)
    output: list[DirectedSeed] = []
    for source, cell in enumerate(perm.tolist()):
        row, col = divmod(int(cell), GRID)
        if row < GRID - 1:
            output.append(DirectedSeed(source, int(inverse[cell + GRID]), 1))
        if col < GRID - 1:
            output.append(DirectedSeed(source, int(inverse[cell + 1]), 3))
    return output


def deterministic_edge_subset(
    edges: Sequence[DirectedSeed], fraction: float, *, seed: int
) -> list[DirectedSeed]:
    """Choose a reproducible oracle fit subset without looking at image pixels."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0,1]")
    if fraction == 1.0:
        return list(edges)
    count = max(1, int(round(len(edges) * fraction)))
    generator = np.random.default_rng(seed)
    selected = generator.choice(len(edges), size=count, replace=False)
    return [edges[int(index)] for index in sorted(selected.tolist())]


def _direct_targets(perm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact U/D/L/R targets and board-valid mask in input-tile order."""
    inverse = np.empty(NFRAG, dtype=np.int64)
    inverse[perm] = np.arange(NFRAG, dtype=np.int64)
    targets = np.full((NFRAG, NUM_DIRECTIONS), -1, dtype=np.int64)
    exists = np.zeros((NFRAG, NUM_DIRECTIONS), dtype=bool)
    for source, cell in enumerate(perm.tolist()):
        row, col = divmod(int(cell), GRID)
        for direction, (drow, dcol) in enumerate(OFFSETS):
            target_row, target_col = row + drow, col + dcol
            if 0 <= target_row < GRID and 0 <= target_col < GRID:
                exists[source, direction] = True
                targets[source, direction] = inverse[target_row * GRID + target_col]
    return targets, exists


def full_bag_linear_scores(tiles: np.ndarray, *, chunk: int = 96) -> np.ndarray:
    """Untrained all-575-candidate seam proxy using the calibration's strips."""
    if tiles.shape != (NFRAG, 3, FS, FS):
        raise ValueError(f"tiles must be ({NFRAG},3,{FS},{FS}), got {tiles.shape}")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    scores = np.full((NUM_DIRECTIONS, NFRAG, NFRAG), -np.inf, dtype=np.float32)
    for direction, turns in enumerate(CANONICAL_ROTATIONS):
        oriented = np.rot90(tiles, turns, axes=(-2, -1)).astype(np.float32, copy=False)
        left_edge = oriented[:, :, :, -1]
        left_inner = oriented[:, :, :, -2]
        right_edge = oriented[:, :, :, 0]
        right_inner = oriented[:, :, :, 1]
        left_predict = 2.0 * left_edge - left_inner
        right_predict = 2.0 * right_edge - right_inner
        for start in range(0, NFRAG, chunk):
            stop = min(start + chunk, NFRAG)
            direct = left_edge[start:stop, None] - right_edge[None]
            forward = left_predict[start:stop, None] - right_edge[None]
            backward = left_edge[start:stop, None] - right_predict[None]
            mse = (
                np.square(direct) + np.square(forward) + np.square(backward)
            ).mean(axis=(2, 3)) / 3.0
            scores[direction, start:stop] = -mse.astype(np.float32, copy=False)
        np.fill_diagonal(scores[direction], -np.inf)
    return scores


def rank_metrics(
    scores: np.ndarray,
    perm: np.ndarray,
    *,
    excluded_pairs: set[tuple[int, int]] | None = None,
) -> RankMetrics:
    """Exact all-bag neighbour rank, optionally excluding fitted physical pairs."""
    if scores.shape != (NUM_DIRECTIONS, NFRAG, NFRAG):
        raise ValueError("full-bag scores must have shape (4,576,576)")
    targets, exists = _direct_targets(perm)
    ranks: list[int] = []
    excluded = excluded_pairs or set()
    for direction in range(NUM_DIRECTIONS):
        for source in range(NFRAG):
            if not exists[source, direction]:
                continue
            target = int(targets[source, direction])
            pair = (source, target) if source < target else (target, source)
            if pair in excluded:
                continue
            truth = float(scores[direction, source, target])
            if not math.isfinite(truth):
                raise RuntimeError("true neighbour received a non-finite full-bag score")
            ranks.append(1 + int(np.sum(scores[direction, source] > truth)))
    if not ranks:
        return RankMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"))
    array = np.asarray(ranks, dtype=np.float64)
    return RankMetrics(
        edges=int(array.size),
        r1=float(np.mean(array <= 1.0)),
        r5=float(np.mean(array <= 5.0)),
        mrr=float(np.mean(1.0 / array)),
        mean_rank=float(np.mean(array)),
    )


def _expected_direction(perm: np.ndarray, source: int, target: int) -> int | None:
    source_row, source_col = divmod(int(perm[source]), GRID)
    target_row, target_col = divmod(int(perm[target]), GRID)
    try:
        return OFFSETS.index((target_row - source_row, target_col - source_col))
    except ValueError:
        return None


def _physical_pair_set(perm: np.ndarray) -> set[tuple[int, int]]:
    return {edge.pair for edge in oracle_edges(perm)}


def relation_metrics(
    relations: Iterable[PhysicalRelation],
    perm: np.ndarray,
    *,
    excluded_pairs: set[tuple[int, int]] | None = None,
) -> RelationMetrics:
    """RSCM/mutual precision and true-edge coverage, with seed exclusion."""
    excluded = excluded_pairs or set()
    kept = [
        relation
        for relation in relations
        if (relation.anchor, relation.target) not in excluded
    ]
    correct = sum(
        _expected_direction(perm, relation.anchor, relation.target) == relation.direction
        for relation in kept
    )
    denominator = len(_physical_pair_set(perm) - excluded)
    return RelationMetrics(
        selected=len(kept),
        correct=correct,
        precision=float(correct / len(kept)) if kept else 0.0,
        coverage=float(correct / denominator) if denominator else float("nan"),
        denominator=denominator,
    )


def _v2_seed_graph(
    ranker: nn.Module,
    affinity: nn.Module,
    affinity_secondary: nn.Module | None,
    tiles: Tensor,
    *,
    candidate_k: int,
    pair_batch: int,
    confidence: float,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, list[DirectedSeed]]:
    """Build v2 mutual/RSCM seeds from exactly the frozen raw candidate graph."""
    candidates_batched, valid_batched = mine_affinity_candidates(
        affinity,
        tiles.unsqueeze(0),
        candidate_k=candidate_k,
        device=device,
        affinity_secondary=affinity_secondary,
    )
    candidates, valid = candidates_batched[0], valid_batched[0]
    scores = score_full_graph(ranker, tiles, candidates, valid, pair_batch=pair_batch, device=device)
    mutual = mutual_argmax_relations(candidates, scores)
    selected = rscm_greedy([relation for relation in mutual if relation.weight >= confidence])
    seeds = [
        DirectedSeed(
            source=relation.anchor,
            target=relation.target,
            direction=relation.direction,
            weight=relation.weight,
        )
        for relation in selected
    ]
    return candidates, valid, scores, seeds


def _resolve_affinity_paths(
    payload: Mapping[str, object], primary_override: str, secondary_override: str
) -> tuple[str, str, int]:
    recorded = payload.get("candidate_graph", {})
    encoders = list(recorded.get("encoders", ())) if isinstance(recorded, Mapping) else []
    args = payload.get("args", {})
    training_args = args if isinstance(args, Mapping) else {}
    primary = primary_override or (str(encoders[0].get("path", "")) if encoders else DEFAULT_AFFINITY_1)
    secondary = secondary_override or (
        str(encoders[1].get("path", "")) if len(encoders) > 1 else DEFAULT_AFFINITY_2
    )
    candidate_k = int(training_args.get("candidate_k", 64))
    if not 1 <= candidate_k < NFRAG:
        raise RuntimeError(f"invalid candidate_k in ranker checkpoint: {candidate_k}")
    return primary, secondary, candidate_k


def _format_rank(label: str, metrics: RankMetrics) -> str:
    if not metrics.edges:
        return f"{label}: no eligible held-out physical edges"
    return (
        f"{label}: edges={metrics.edges} R1={metrics.r1:.4f} R5={metrics.r5:.4f} "
        f"MRR={metrics.mrr:.4f} mean_rank={metrics.mean_rank:.1f}"
    )


def _format_relations(label: str, metrics: RelationMetrics) -> str:
    return (
        f"{label}: selected={metrics.selected} correct={metrics.correct} "
        f"p={metrics.precision:.4f} coverage={metrics.coverage:.4f} "
        f"denom={metrics.denominator}"
    )


def _print_calibration(label: str, result: CalibrationResult) -> None:
    print(
        f"  [{label}] components={result.components} fitted_edges={result.fitted_edges} "
        f"fitted_tiles={result.fitted_tiles}/{NFRAG} "
        f"strip_RMSE={result.raw_seam_rmse:.5f}->{result.calibrated_seam_rmse:.5f} "
        f"scale_sat={result.scale_saturation:.3f} bias_sat={result.bias_saturation:.3f}",
        flush=True,
    )


def _mean_metric(rows: Sequence[RankMetrics], key: str) -> float:
    values = [float(getattr(row, key)) for row in rows if row.edges and math.isfinite(float(getattr(row, key)))]
    return float(np.mean(values)) if values else float("nan")


def _gate_text() -> str:
    return (
        "GATE: keep photometric calibration only if oracle 50%-fit heldout R1 gains >=0.020 "
        "and v2 seed-excluded R1 gains >=0.010, while frozen reciprocal precision does not fall "
        "at comparable coverage.  A lower fit-strip RMSE alone is not evidence."
    )


def smoke() -> dict[str, float]:
    """Data-free solver test with a smooth image and known tile affines."""
    rng = np.random.default_rng(88)
    side = FS * 2
    yy, xx = np.mgrid[:side, :side].astype(np.float64)
    clean = np.stack(
        (
            0.18 + 0.004 * xx + 0.003 * yy,
            0.66 - 0.002 * xx + 0.0025 * yy,
            0.34 + 0.0015 * xx - 0.002 * yy,
        ),
        axis=0,
    )
    clean += rng.normal(0.0, 0.002, size=clean.shape)
    clean_tiles = clean.reshape(3, 2, FS, 2, FS).transpose(1, 3, 0, 2, 4).reshape(4, 3, FS, FS)
    true_scale = np.asarray(
        ((0.80, 1.16, 0.91), (1.19, 0.86, 1.10), (0.92, 1.08, 0.82), (1.11, 0.89, 1.20)),
        dtype=np.float64,
    )
    true_bias = np.asarray(
        ((0.05, -0.04, 0.02), (-0.03, 0.02, -0.05), (0.04, 0.03, -0.02), (-0.04, -0.02, 0.04)),
        dtype=np.float64,
    )
    raw = (clean_tiles - true_bias[:, :, None, None]) / true_scale[:, :, None, None]
    seeds = [
        DirectedSeed(0, 1, 3), DirectedSeed(2, 3, 3),
        DirectedSeed(0, 2, 1), DirectedSeed(1, 3, 1),
    ]
    config = CalibrationConfig(iterations=8, huber=0.02, scale_prior=0.15, bias_prior=0.15)
    result = calibrate_tiles(raw, seeds, config)
    corrected = apply_calibration(raw, result).astype(np.float64)
    raw_rmse = float(np.sqrt(np.mean(np.square(raw - clean_tiles))))
    corrected_rmse = float(np.sqrt(np.mean(np.square(corrected - clean_tiles))))
    if not corrected_rmse < raw_rmse * 0.55:
        raise AssertionError(
            f"photometric smoke did not recover smooth-tile affine corruption: {raw_rmse} -> {corrected_rmse}"
        )
    return {
        "raw_rmse": raw_rmse,
        "corrected_rmse": corrected_rmse,
        "fit_strip_rmse": result.calibrated_seam_rmse,
        "components": float(result.components),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-sources", default="oracle", type=_parse_sources)
    parser.add_argument("--n", type=int, default=1, help="fresh exact synthetic held-out puzzles")
    parser.add_argument("--seed", type=int, default=SEED + 6211)
    parser.add_argument("--device", default="cpu", help="CPU by default; pass cuda only when coordinated")
    parser.add_argument(
        "--oracle-fit-fraction",
        type=float,
        default=0.5,
        help="fraction of exact oracle seams used for fitting; 0.5 preserves a physical-edge holdout",
    )
    parser.add_argument("--seed-conf", type=float, default=0.70, help="v2 reciprocal min confidence")
    parser.add_argument("--ranker-ckpt", default=DEFAULT_RANKER)
    parser.add_argument("--affinity-ckpt", default="")
    parser.add_argument("--affinity-ckpt2", default="")
    parser.add_argument("--pair-batch", type=int, default=4096)
    parser.add_argument("--irls-iters", type=int, default=6)
    parser.add_argument("--huber", type=float, default=0.055)
    parser.add_argument("--scale-min", type=float, default=0.65)
    parser.add_argument("--scale-max", type=float, default=1.45)
    parser.add_argument("--bias-limit", type=float, default=0.25)
    parser.add_argument("--scale-prior", type=float, default=8.0)
    parser.add_argument("--bias-prior", type=float, default=24.0)
    parser.add_argument("--score-chunk", type=int, default=96)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.n < 1 or args.pair_batch < 1 or args.irls_iters < 1 or args.score_chunk < 1:
        parser.error("--n, --pair-batch, --irls-iters, and --score-chunk must be positive")
    if not 0.0 < args.oracle_fit_fraction <= 1.0:
        parser.error("--oracle-fit-fraction must lie in (0,1]")
    if not 0.0 <= args.seed_conf <= 1.0:
        parser.error("--seed-conf must lie in [0,1]")
    if args.huber <= 0.0 or args.scale_min <= 0.0 or args.scale_max < args.scale_min:
        parser.error("invalid robust/bound configuration")
    if args.bias_limit <= 0.0 or args.scale_prior <= 0.0 or args.bias_prior <= 0.0:
        parser.error("priors and --bias-limit must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if args.smoke:
        print(f"[seed-photometric smoke] {smoke()}", flush=True)
        return
    device = _parse_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    config = CalibrationConfig(
        iterations=args.irls_iters,
        huber=args.huber,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        bias_limit=args.bias_limit,
        scale_prior=args.scale_prior,
        bias_prior=args.bias_prior,
    )

    ranker: nn.Module | None = None
    affinity: nn.Module | None = None
    affinity_secondary: nn.Module | None = None
    candidate_k = 64
    if "v2" in args.seed_sources:
        ranker, payload = load_ranker(args.ranker_ckpt, device)
        affinity_path, affinity_path2, candidate_k = _resolve_affinity_paths(
            payload, args.affinity_ckpt, args.affinity_ckpt2
        )
        affinity, _, _ = load_frozen_affinity(affinity_path, device)
        if affinity_path2:
            affinity_secondary, _, _ = load_frozen_affinity(affinity_path2, device)
        print(
            f"v2 source: ranker={os.path.abspath(args.ranker_ckpt)} "
            f"top{candidate_k}/encoder affinity={os.path.basename(affinity_path)}"
            + (f" + {os.path.basename(affinity_path2)}" if affinity_path2 else ""),
            flush=True,
        )

    _, validation_names = train_val_split()
    if args.n > len(validation_names):
        raise ValueError(f"--n exceeds held-out pool ({len(validation_names)})")
    dataset = CanvasDataset(validation_names[: args.n], real_prob=0.0, seed=args.seed)
    print("== SEED-CONDITIONED PHOTOMETRIC CALIBRATION GATE ==", flush=True)
    print(
        f"device={device} n={args.n} sources={','.join(args.seed_sources)} "
        f"oracle_fit_fraction={args.oracle_fit_fraction:.2f} v2_conf={args.seed_conf:.2f}",
        flush=True,
    )
    print(_gate_text(), flush=True)

    pooled: dict[str, list[RankMetrics]] = defaultdict(list)
    for index in range(args.n):
        sample = dataset[index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("exact synthetic labels are required")
        raw_tiles = sample["tiles"].numpy().astype(np.float32, copy=False)
        perm = sample["perm"].numpy().astype(np.int64, copy=False)
        raw_full_scores = full_bag_linear_scores(raw_tiles, chunk=args.score_chunk)
        print(f"\nimage {index + 1}/{args.n}", flush=True)

        v2_inputs: tuple[Tensor, Tensor, Tensor, list[DirectedSeed]] | None = None
        if "v2" in args.seed_sources:
            assert ranker is not None and affinity is not None
            tile_tensor = sample["tiles"].to(device, non_blocking=device.type == "cuda")
            v2_inputs = _v2_seed_graph(
                ranker, affinity, affinity_secondary, tile_tensor,
                candidate_k=candidate_k, pair_batch=args.pair_batch,
                confidence=args.seed_conf, device=device,
            )
            _, _, _, v2_seeds = v2_inputs
            raw_relations = [
                PhysicalRelation(seed.source, seed.target, seed.direction, 0, 0, seed.weight, seed.weight, seed.weight)
                for seed in v2_seeds
            ]
            print(
                "  " + _format_relations("v2 calibration seeds", relation_metrics(raw_relations, perm)),
                flush=True,
            )

        for source_name in args.seed_sources:
            if source_name == "oracle":
                seeds = deterministic_edge_subset(
                    oracle_edges(perm), args.oracle_fit_fraction, seed=args.seed + index * 97
                )
            else:
                assert v2_inputs is not None
                seeds = v2_inputs[3]
            seed_pairs = {seed.pair for seed in seeds}
            result = calibrate_tiles(raw_tiles, seeds, config)
            corrected = apply_calibration(raw_tiles, result)
            corrected_scores = full_bag_linear_scores(corrected, chunk=args.score_chunk)
            _print_calibration(source_name, result)
            # All-edge scores are an optimistic upper-bound view for oracle;
            # seed-excluded scores are the decisive view whenever such edges exist.
            raw_all = rank_metrics(raw_full_scores, perm)
            cal_all = rank_metrics(corrected_scores, perm)
            raw_nonseed = rank_metrics(raw_full_scores, perm, excluded_pairs=seed_pairs)
            cal_nonseed = rank_metrics(corrected_scores, perm, excluded_pairs=seed_pairs)
            print("  " + _format_rank(f"{source_name} full-bag raw all", raw_all), flush=True)
            print("  " + _format_rank(f"{source_name} full-bag cal all", cal_all), flush=True)
            print("  " + _format_rank(f"{source_name} full-bag raw nonseed", raw_nonseed), flush=True)
            print("  " + _format_rank(f"{source_name} full-bag cal nonseed", cal_nonseed), flush=True)
            pooled[f"{source_name}:raw_all"].append(raw_all)
            pooled[f"{source_name}:cal_all"].append(cal_all)
            pooled[f"{source_name}:raw_nonseed"].append(raw_nonseed)
            pooled[f"{source_name}:cal_nonseed"].append(cal_nonseed)

            if source_name != "v2":
                continue
            assert ranker is not None and v2_inputs is not None
            candidates, valid, raw_rank_scores, _ = v2_inputs
            corrected_tensor = torch.from_numpy(corrected).to(device)
            calibrated_rank_scores = score_full_graph(
                ranker, corrected_tensor, candidates, valid, pair_batch=args.pair_batch, device=device
            )
            raw_graph = rscm_greedy(
                [relation for relation in mutual_argmax_relations(candidates, raw_rank_scores) if relation.weight >= args.seed_conf]
            )
            calibrated_graph = rscm_greedy(
                [relation for relation in mutual_argmax_relations(candidates, calibrated_rank_scores) if relation.weight >= args.seed_conf]
            )
            print("  " + _format_relations("v2 RSCM raw all", relation_metrics(raw_graph, perm)), flush=True)
            print("  " + _format_relations("v2 RSCM cal all", relation_metrics(calibrated_graph, perm)), flush=True)
            print(
                "  " + _format_relations(
                    "v2 RSCM raw nonseed", relation_metrics(raw_graph, perm, excluded_pairs=seed_pairs)
                ),
                flush=True,
            )
            print(
                "  " + _format_relations(
                    "v2 RSCM cal nonseed", relation_metrics(calibrated_graph, perm, excluded_pairs=seed_pairs)
                ),
                flush=True,
            )

    print(f"\n== pooled full-bag ranks over {args.n} images ==", flush=True)
    for source_name in args.seed_sources:
        for scope in ("all", "nonseed"):
            raw_rows = pooled[f"{source_name}:raw_{scope}"]
            cal_rows = pooled[f"{source_name}:cal_{scope}"]
            if not raw_rows:
                continue
            raw_r1 = _mean_metric(raw_rows, "r1")
            cal_r1 = _mean_metric(cal_rows, "r1")
            raw_r5 = _mean_metric(raw_rows, "r5")
            cal_r5 = _mean_metric(cal_rows, "r5")
            print(
                f"{source_name} {scope}: R1 {raw_r1:.4f}->{cal_r1:.4f} "
                f"(delta={cal_r1 - raw_r1:+.4f}); R5 {raw_r5:.4f}->{cal_r5:.4f} "
                f"(delta={cal_r5 - raw_r5:+.4f})",
                flush=True,
            )
    print(_gate_text(), flush=True)


if __name__ == "__main__":
    main()
