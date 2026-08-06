"""Exact-synthetic diagnostic for global graph-to-grid geometry.

This is deliberately *not* another local seam scorer.  It asks whether the
frozen tile-proximity embeddings contain enough globally consistent geometry
to lay all 576 shuffled tiles onto a 24 x 24 grid.

For each fresh, exactly labelled synthetic puzzle the evaluator:

1. builds a rank-calibrated, weighted mutual-kNN graph from raw pixels or a
   frozen affinity encoder;
2. converts that graph into a full weighted geodesic distance matrix;
3. tries two cheap relative layouts (Laplacian eigenmaps and classical MDS);
4. optionally runs a small entropic *fused Gromov--Wasserstein* (FGW) match
   between the tile-graph distance matrix and the 24 x 24 grid distance
   matrix; and
5. reports both graph-distance signal and a Hungarian grid assignment.

There are two intentionally oracle-labelled measurements:

* ``oracle_affine_*`` fits a 2-D affine transform from a continuous layout to
  the known synthetic coordinates *only for diagnostic evaluation*, then
  Hungarian-rounds it.  It answers whether the layout has recovered relative
  geometry, independent of frame/scale/shear ambiguity.  It is not a solver.
* ``fgw_d4_*`` selects the best of the eight grid dihedral frames only for
  evaluation.  FGW itself sees no clean coordinates; a graph has no canonical
  orientation, so this merely removes its unavoidable D4 ambiguity.

The optional ``--include_oracle`` source uses exact clean Euclidean proximity.
It is a harness sanity check: if it succeeds while learned/raw sources fail,
the negative result is evidence about signal, rather than about the graph
layout implementation.

The default is intentionally small: one held-out synthetic image, all cheap
baselines, and FGW only for the rank-percentile ensemble.  Increase ``--n``
only after inspecting the first report.

Examples:

    python src/eval_gromov_grid_gate.py --n 1 --device cuda
    python src/eval_gromov_grid_gate.py --n 2 --include_oracle ^
      --fgw_sources ensemble,oracle --graph_k 16 --device cuda
    python src/eval_gromov_grid_gate.py --sources ensemble --methods spectral,mds ^
      --n 4 --device cpu
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import spearmanr

from canvas_data import CanvasDataset
from config import GRID, NFRAG, SEED
from eval_affinity_graph import (
    _parse_device,
    learned_affinity,
    load_model,
    raw_zpixel_affinity,
)
from imgio import train_val_split
from macro_affinity import count_params
from placement_metrics import neighbour_accuracy, placement_accuracy


DEFAULT_CKPT_A = os.path.join("artifacts", "macro_affinity", "affinity_r1_1200_best.pt")
DEFAULT_CKPT_B = os.path.join("artifacts", "macro_affinity", "affinity_r3_1000_best.pt")
ALL_SOURCES = ("raw", "a", "b", "ensemble", "oracle")
ALL_METHODS = ("spectral", "mds", "fgw")


@dataclass(frozen=True)
class GraphGeometry:
    """A mutual-kNN graph and its finite working geodesic metric."""

    weights: np.ndarray
    distance: np.ndarray
    components: int
    largest_fraction: float
    disconnected_pair_fraction: float
    undirected_edges: int
    mean_degree: float


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def _parse_csv(value: str, *, allowed: Iterable[str], name: str) -> tuple[str, ...]:
    allowed_set = set(allowed)
    parsed = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one comma-separated item")
    unknown = [item for item in parsed if item not in allowed_set]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {name}: {', '.join(unknown)}; choose from {', '.join(sorted(allowed_set))}"
        )
    # Preserve user order while removing accidental duplicates.
    return tuple(dict.fromkeys(parsed))


def _truth_coordinates(perm: torch.Tensor) -> np.ndarray:
    """Return clean ``(row, col)`` coordinates in input-tile order."""
    labels = perm.detach().cpu().numpy().astype(np.int64, copy=False)
    if labels.shape != (NFRAG,) or not np.array_equal(np.sort(labels), np.arange(NFRAG)):
        raise ValueError("synthetic perm must be a full valid clean-cell permutation")
    return np.stack((labels // GRID, labels % GRID), axis=1).astype(np.float64)


def _grid_coordinates() -> np.ndarray:
    return np.stack(np.divmod(np.arange(NFRAG, dtype=np.int64), GRID), axis=1).astype(np.float64)


GRID_COORDS = _grid_coordinates()


def _pairwise_euclidean(coordinates: np.ndarray) -> np.ndarray:
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.maximum(np.square(delta).sum(axis=-1), 0.0))


def _pairwise_chebyshev(coordinates: np.ndarray) -> np.ndarray:
    delta = np.abs(coordinates[:, None, :] - coordinates[None, :, :])
    return delta.max(axis=-1)


GRID_DISTANCE = _pairwise_euclidean(GRID_COORDS)


def _row_percentile_affinity(affinity: np.ndarray) -> np.ndarray:
    """Convert each row to a comparable descending non-self affinity percentile.

    Cosine ranges from separately trained encoders need not be calibrated.  A
    row percentile preserves each encoder's ranking while making their mean a
    robust ensemble.  The matrix is intentionally not symmetrized here: the
    mutual-kNN constructor later handles directionality explicitly.
    """
    values = np.asarray(affinity, dtype=np.float64)
    if values.shape != (NFRAG, NFRAG):
        raise ValueError(f"affinity must have shape ({NFRAG},{NFRAG})")
    work = values.copy()
    np.fill_diagonal(work, -np.inf)
    order = np.argsort(-work, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(NFRAG)[:, None], order] = np.arange(NFRAG)[None, :]
    # Non-self ranks are 0..574.  The artificially last self entry is ignored.
    percentile = 1.0 - ranks.astype(np.float64) / float(NFRAG - 2)
    np.fill_diagonal(percentile, 0.0)
    return percentile


def _rank_percentile_ensemble(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Mean row-percentiles, the scale-robust learned-affinity ensemble."""
    return 0.5 * (_row_percentile_affinity(first) + _row_percentile_affinity(second))


def _weighted_knn_geodesics(
    affinity: np.ndarray,
    *,
    k: int,
    rank_temperature: float,
    mutual: bool,
) -> GraphGeometry:
    """Build a rank-calibrated weighted kNN graph and all-pairs geodesics.

    Edges combine rank decay and within-row affinity contrast.  This makes the
    graph use the full learned affinity matrix for candidate selection while
    avoiding arbitrary cross-model cosine scale.  ``mutual=True`` is the
    conservative default: it rejects one-sided long shortcuts that are fatal
    to geodesic geometry.

    If the selected graph is disconnected, non-finite distances are replaced
    only in the *working* metric by a transparent large penalty.  Connectivity
    statistics are reported alongside every layout, so such a result cannot be
    mistaken for real global geometry.
    """
    values = np.asarray(affinity, dtype=np.float64)
    if values.shape != (NFRAG, NFRAG):
        raise ValueError(f"affinity must have shape ({NFRAG},{NFRAG})")
    if not 1 <= k < NFRAG:
        raise ValueError(f"k must be in [1,{NFRAG - 1}]")
    if rank_temperature <= 0.0:
        raise ValueError("rank_temperature must be positive")

    work = values.copy()
    work[~np.isfinite(work)] = -np.inf
    np.fill_diagonal(work, -np.inf)
    order = np.argsort(-work, axis=1, kind="stable")[:, :k]
    selected = np.take_along_axis(work, order, axis=1)
    if not np.isfinite(selected).all():
        raise RuntimeError("affinity has fewer than k finite non-self entries in a row")

    ranks = np.arange(k, dtype=np.float64)[None, :]
    rank_strength = np.exp(-ranks / float(rank_temperature))
    best = selected[:, :1]
    cutoff = selected[:, -1:]
    contrast = (selected - cutoff) / np.maximum(best - cutoff, 1.0e-8)
    # The weak floor avoids treating a valid rank-k edge as an infinite-length
    # link, while rank still dominates if a row is nearly flat.
    strength = 0.55 * rank_strength + 0.45 * (0.10 + 0.90 * np.clip(contrast, 0.0, 1.0))

    directed = np.zeros((NFRAG, NFRAG), dtype=np.float64)
    directed[np.arange(NFRAG)[:, None], order] = strength
    if mutual:
        both = (directed > 0.0) & (directed.T > 0.0)
        weights = np.where(both, np.minimum(directed, directed.T), 0.0)
    else:
        weights = np.maximum(directed, directed.T)
    np.fill_diagonal(weights, 0.0)

    mask = weights > 0.0
    # Strong rank-1 edges have cost near one; low-ranked links are several
    # times longer, so Dijkstra favours coherent local chains over shortcuts.
    costs = np.zeros_like(weights)
    costs[mask] = 1.0 / (0.15 + 0.85 * weights[mask])
    graph = csr_matrix(costs)
    component_count, labels = connected_components(graph, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=component_count)
    finite_distance = dijkstra(graph, directed=False, unweighted=False)
    off_diagonal = ~np.eye(NFRAG, dtype=bool)
    finite_pairs = np.isfinite(finite_distance) & off_diagonal
    disconnected_pair_fraction = float(1.0 - finite_pairs.sum() / off_diagonal.sum())
    finite_values = finite_distance[finite_pairs]
    if finite_values.size == 0:
        raise RuntimeError("mutual-kNN graph contains no finite off-diagonal geodesics")
    scale = float(np.median(finite_values))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("invalid geodesic distance scale")
    working_distance = finite_distance.copy()
    # This is intentionally a large penalty rather than a fabricated bridge.
    # It allows MDS/FGW to fail visibly on a disconnected graph.
    working_distance[~np.isfinite(working_distance)] = float(finite_values.max() * 4.0)
    np.fill_diagonal(working_distance, 0.0)
    working_distance /= scale

    edge_count = int(np.triu(mask, k=1).sum())
    return GraphGeometry(
        weights=weights,
        distance=working_distance,
        components=int(component_count),
        largest_fraction=float(sizes.max() / NFRAG),
        disconnected_pair_fraction=disconnected_pair_fraction,
        undirected_edges=edge_count,
        mean_degree=float(mask.sum(axis=1).mean()),
    )


def _corr(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return Pearson and Spearman correlations with finite-safe fallbacks."""
    if left.size < 3 or right.size != left.size:
        return float("nan"), float("nan")
    if np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
        return 0.0, 0.0
    pearson = float(np.corrcoef(left, right)[0, 1])
    rho = spearmanr(left, right).statistic
    return pearson, float(rho) if rho is not None else float("nan")


def _distance_signal_metrics(graph: GraphGeometry, truth_coordinates: np.ndarray) -> dict[str, float]:
    """Frame-free evidence that graph geodesics reflect actual board distance."""
    mask = np.triu(np.ones((NFRAG, NFRAG), dtype=bool), k=1)
    truth_euclid = _pairwise_euclidean(truth_coordinates)[mask]
    truth_chebyshev = _pairwise_chebyshev(truth_coordinates)[mask]
    source = graph.distance[mask]
    euclid_pearson, euclid_spearman = _corr(source, truth_euclid)
    cheb_pearson, cheb_spearman = _corr(source, truth_chebyshev)
    return {
        "components": float(graph.components),
        "largest_fraction": graph.largest_fraction,
        "disconnected_pair_fraction": graph.disconnected_pair_fraction,
        "undirected_edges": float(graph.undirected_edges),
        "mean_degree": graph.mean_degree,
        "geodesic_euclid_pearson": euclid_pearson,
        "geodesic_euclid_spearman": euclid_spearman,
        "geodesic_cheb_pearson": cheb_pearson,
        "geodesic_cheb_spearman": cheb_spearman,
    }


def _spectral_layout(weights: np.ndarray) -> np.ndarray:
    """Two non-trivial normalized-Laplacian eigenvectors (Laplacian eigenmaps)."""
    degree = weights.sum(axis=1)
    if np.any(degree <= 0.0):
        # Isolated vertices should not occur at k=16, but returning a finite
        # layout keeps the diagnostic report usable on a pathological source.
        return np.zeros((NFRAG, 2), dtype=np.float64)
    normalized = weights / np.sqrt(degree[:, None] * degree[None, :])
    laplacian = np.eye(NFRAG, dtype=np.float64) - normalized
    values, vectors = np.linalg.eigh(laplacian)
    nontrivial = np.flatnonzero(values > 1.0e-8)
    if nontrivial.size < 2:
        return np.zeros((NFRAG, 2), dtype=np.float64)
    # Dividing by sqrt(degree) gives the standard random-walk-compatible
    # embedding rather than raw normalized-Laplacian coordinates.
    return vectors[:, nontrivial[:2]] / np.sqrt(degree)[:, None]


def _classical_mds(distance: np.ndarray) -> np.ndarray:
    """Classical MDS of the weighted graph geodesic matrix."""
    if distance.shape != (NFRAG, NFRAG):
        raise ValueError("distance must be a full puzzle square matrix")
    centered = np.eye(NFRAG, dtype=np.float64) - np.full((NFRAG, NFRAG), 1.0 / NFRAG)
    gram = -0.5 * centered @ np.square(distance) @ centered
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    positive = [index for index in order if values[index] > 1.0e-9][:2]
    if not positive:
        return np.zeros((NFRAG, 2), dtype=np.float64)
    coordinates = vectors[:, positive] * np.sqrt(values[positive])[None, :]
    if coordinates.shape[1] == 1:
        coordinates = np.concatenate((coordinates, np.zeros((NFRAG, 1))), axis=1)
    return coordinates.astype(np.float64, copy=False)


def _affine_fit(coordinates: np.ndarray, truth_coordinates: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Oracle affine alignment used only to score a relative continuous layout."""
    points = np.nan_to_num(np.asarray(coordinates, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if points.shape != (NFRAG, 2):
        raise ValueError(f"coordinates must have shape ({NFRAG},2)")
    design = np.concatenate((points, np.ones((NFRAG, 1), dtype=np.float64)), axis=1)
    coefficients, *_ = np.linalg.lstsq(design, truth_coordinates, rcond=None)
    aligned = design @ coefficients
    residual = np.square(aligned - truth_coordinates).sum(axis=0)
    total = np.square(truth_coordinates - truth_coordinates.mean(axis=0, keepdims=True)).sum(axis=0)
    r2_dimensions = 1.0 - residual / np.maximum(total, 1.0e-12)
    return aligned, {
        "oracle_affine_r2": float(r2_dimensions.mean()),
        "oracle_affine_r2_min_axis": float(r2_dimensions.min()),
        "oracle_affine_rmse": float(np.sqrt(np.square(aligned - truth_coordinates).mean())),
    }


def _placement_from_slot_for_tile(slot_for_tile: np.ndarray) -> np.ndarray:
    slots = np.asarray(slot_for_tile, dtype=np.int64)
    if slots.shape != (NFRAG,) or not np.array_equal(np.sort(slots), np.arange(NFRAG)):
        raise ValueError("slot_for_tile must be a 576-element grid permutation")
    placement = np.empty(NFRAG, dtype=np.int64)
    placement[slots] = np.arange(NFRAG, dtype=np.int64)
    return placement


def _hungarian_slots(coordinates: np.ndarray) -> np.ndarray:
    """Assign a continuous point for every input tile to unique grid cells."""
    points = np.nan_to_num(np.asarray(coordinates, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    squared_cost = np.square(points[:, None, :] - GRID_COORDS[None, :, :]).sum(axis=-1)
    rows, cols = linear_sum_assignment(squared_cost)
    if rows.size != NFRAG:
        raise RuntimeError("Hungarian assignment did not cover every tile")
    slots = np.empty(NFRAG, dtype=np.int64)
    slots[rows] = cols
    return slots


def _truth_placement(perm: torch.Tensor) -> np.ndarray:
    # ``perm[tile] -> clean grid slot``; metric helper wants ``place[slot] -> tile``.
    labels = perm.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.argsort(labels)


def _assignment_metrics(slot_for_tile: np.ndarray, perm: torch.Tensor) -> dict[str, float]:
    placement = _placement_from_slot_for_tile(slot_for_tile)
    truth = _truth_placement(perm)
    placement_acc, _ = placement_accuracy(placement, truth)
    neighbour, right, down = neighbour_accuracy(placement, truth)
    return {
        "placement": float(placement_acc),
        "neighbour": float(neighbour),
        "right_neighbour": float(right),
        "down_neighbour": float(down),
    }


def _continuous_layout_metrics(coordinates: np.ndarray, perm: torch.Tensor) -> dict[str, float]:
    """Score a continuous layout after explicit oracle affine frame alignment."""
    truth = _truth_coordinates(perm)
    aligned, result = _affine_fit(coordinates, truth)
    slots = _hungarian_slots(aligned)
    result.update({f"oracle_affine_hungarian_{key}": value for key, value in _assignment_metrics(slots, perm).items()})

    # This rank statistic does not choose a coordinate frame and is useful when
    # an affine fit is visually suspicious (for example, collapsed layouts).
    mask = np.triu(np.ones((NFRAG, NFRAG), dtype=bool), k=1)
    source_distance = _pairwise_euclidean(coordinates)[mask]
    truth_distance = _pairwise_euclidean(truth)[mask]
    _, coordinate_spearman = _corr(source_distance, truth_distance)
    result["coordinate_distance_spearman"] = coordinate_spearman
    return result


def _d4_slot_maps() -> tuple[np.ndarray, ...]:
    """Return the eight grid-slot permutations induced by D4 symmetry."""
    matrices = (
        np.array(((1, 0), (0, 1)), dtype=np.int64),
        np.array(((0, -1), (1, 0)), dtype=np.int64),
        np.array(((-1, 0), (0, -1)), dtype=np.int64),
        np.array(((0, 1), (-1, 0)), dtype=np.int64),
        np.array(((-1, 0), (0, 1)), dtype=np.int64),
        np.array(((1, 0), (0, -1)), dtype=np.int64),
        np.array(((0, 1), (1, 0)), dtype=np.int64),
        np.array(((0, -1), (-1, 0)), dtype=np.int64),
    )
    center = (GRID - 1) / 2.0
    centered = GRID_COORDS - center
    maps: list[np.ndarray] = []
    for matrix in matrices:
        transformed = centered @ matrix.T + center
        rounded = np.rint(transformed).astype(np.int64)
        if np.any(rounded < 0) or np.any(rounded >= GRID):
            raise RuntimeError("internal D4 transform left grid bounds")
        maps.append(rounded[:, 0] * GRID + rounded[:, 1])
    return tuple(maps)


D4_SLOT_MAPS = _d4_slot_maps()
D4_MATRICES = (
    np.array(((1, 0), (0, 1)), dtype=np.float64),
    np.array(((0, -1), (1, 0)), dtype=np.float64),
    np.array(((-1, 0), (0, -1)), dtype=np.float64),
    np.array(((0, 1), (-1, 0)), dtype=np.float64),
    np.array(((-1, 0), (0, 1)), dtype=np.float64),
    np.array(((1, 0), (0, -1)), dtype=np.float64),
    np.array(((0, 1), (1, 0)), dtype=np.float64),
    np.array(((0, -1), (-1, 0)), dtype=np.float64),
)


def _best_d4_metrics(slot_for_tile: np.ndarray, perm: torch.Tensor) -> dict[str, float]:
    """Oracle-frame score of a discrete graph match, resolving only D4 ambiguity."""
    best: dict[str, float] | None = None
    best_index = -1
    for index, mapping in enumerate(D4_SLOT_MAPS):
        candidate = _assignment_metrics(mapping[np.asarray(slot_for_tile, dtype=np.int64)], perm)
        if best is None or (candidate["placement"], candidate["neighbour"]) > (
            best["placement"],
            best["neighbour"],
        ):
            best = candidate
            best_index = index
    if best is None:
        raise RuntimeError("no D4 grid frame was evaluated")
    return {
        "fgw_d4_placement": best["placement"],
        "fgw_d4_neighbour": best["neighbour"],
        "fgw_d4_right_neighbour": best["right_neighbour"],
        "fgw_d4_down_neighbour": best["down_neighbour"],
        "fgw_oracle_d4_frame": float(best_index),
    }


def _normalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Centre and use one shared RMS scale, retaining aspect information."""
    values = np.nan_to_num(np.asarray(coordinates, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    values = values - values.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.square(values).sum(axis=1).mean()))
    return values / max(scale, 1.0e-8)


def _sinkhorn_log(cost: torch.Tensor, *, epsilon: float, iterations: int) -> torch.Tensor:
    """Balanced entropic transport in the log domain for uniform marginals."""
    if epsilon <= 0.0 or iterations < 1:
        raise ValueError("Sinkhorn epsilon and iterations must be positive")
    count = cost.shape[0]
    if tuple(cost.shape) != (count, count):
        raise ValueError("Sinkhorn cost must be square")
    log_mass = -float(np.log(count))
    log_kernel = -cost / float(epsilon)
    log_u = torch.zeros(count, dtype=cost.dtype, device=cost.device)
    log_v = torch.zeros_like(log_u)
    for _ in range(iterations):
        log_u = log_mass - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_mass - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    log_transport = log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    return torch.exp(log_transport).clamp_min(0.0)


def _fgw_match(
    source_distance: np.ndarray,
    initial_coordinates: np.ndarray,
    *,
    device: torch.device,
    alpha: float,
    epsilon: float,
    init_epsilon: float,
    outer_iterations: int,
    sinkhorn_iterations: int,
    starts: int,
    relaxation: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Small practical entropic fused-GW match from a tile graph to the grid.

    The structural term is the standard squared-loss GW tensor contraction
    ``C_x^2 p + C_y^2 q - 2 C_x T C_y``.  The light feature term is a
    D4-enumerated MDS initialization; it prevents uniform transport from being
    the symmetric fixed point of entropic GW.  Start selection uses the FGW
    objective only -- never the known clean permutation.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0,1]")
    if starts < 1 or starts > len(D4_MATRICES):
        raise ValueError(f"starts must be in [1,{len(D4_MATRICES)}]")

    # Scale both relational costs identically around a median non-self value.
    source = np.asarray(source_distance, dtype=np.float64)
    grid = GRID_DISTANCE.copy()
    off_diagonal = ~np.eye(NFRAG, dtype=bool)
    source /= max(float(np.median(source[off_diagonal])), 1.0e-8)
    grid /= max(float(np.median(grid[off_diagonal])), 1.0e-8)
    source_t = torch.as_tensor(source, dtype=torch.float32, device=device)
    grid_t = torch.as_tensor(grid, dtype=torch.float32, device=device)
    probability = torch.full((NFRAG,), 1.0 / NFRAG, dtype=torch.float32, device=device)
    source_constant = (source_t.square() @ probability).unsqueeze(1)
    grid_constant = (grid_t.square() @ probability).unsqueeze(0)

    source_features = _normalize_coordinates(initial_coordinates)
    grid_features = _normalize_coordinates(GRID_COORDS)
    best_transport: torch.Tensor | None = None
    best_objective = float("inf")
    best_start = -1

    with torch.inference_mode(), _autocast(device):
        # Do not run arbitrary random restarts: D4 is the only intrinsic frame
        # family for a square grid and keeps this a genuinely bounded gate.
        for start, matrix in enumerate(D4_MATRICES[:starts]):
            transformed = source_features @ matrix.T
            feature = np.square(transformed[:, None, :] - grid_features[None, :, :]).sum(axis=-1)
            feature_t = torch.as_tensor(feature, dtype=torch.float32, device=device)
            # A feature-informed transport is an unsupervised initialization,
            # not an oracle coordinate alignment.
            transport = _sinkhorn_log(feature_t, epsilon=init_epsilon, iterations=sinkhorn_iterations)
            for _ in range(outer_iterations):
                structural = source_constant + grid_constant - 2.0 * (source_t @ transport @ grid_t.T)
                cost = alpha * structural + (1.0 - alpha) * feature_t
                proposal = _sinkhorn_log(cost, epsilon=epsilon, iterations=sinkhorn_iterations)
                transport = (1.0 - relaxation) * transport + relaxation * proposal
            structural = source_constant + grid_constant - 2.0 * (source_t @ transport @ grid_t.T)
            cost = alpha * structural + (1.0 - alpha) * feature_t
            objective = float((cost * transport).sum().float().cpu())
            if np.isfinite(objective) and objective < best_objective:
                best_objective = objective
                best_transport = transport.float().cpu()
                best_start = start

    if best_transport is None:
        raise RuntimeError("all FGW starts produced a non-finite objective")
    # One-to-one rounding is deliberate: plain per-row argmax would allow many
    # tiles to collapse onto one grid cell and would not test an actual layout.
    score = best_transport.numpy()
    rows, cols = linear_sum_assignment(-score)
    slots = np.empty(NFRAG, dtype=np.int64)
    slots[rows] = cols
    entropy = -float((score * np.log(np.maximum(score, 1.0e-30))).sum())
    return slots, {
        "fgw_objective": best_objective,
        "fgw_best_start": float(best_start),
        "fgw_transport_entropy": entropy,
    }


def _mean_dict(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set().union(*(row.keys() for row in rows))
    return {
        key: float(np.mean([row[key] for row in rows if key in row and np.isfinite(row[key])]))
        if any(key in row and np.isfinite(row[key]) for row in rows)
        else float("nan")
        for key in keys
    }


def _fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def _print_graph_report(label: str, metrics: Mapping[str, float]) -> None:
    print(
        f"[{label}] graph: components={metrics['components']:.0f} "
        f"largest={metrics['largest_fraction']:.3f} "
        f"disc_pairs={metrics['disconnected_pair_fraction']:.4f} "
        f"edges={metrics['undirected_edges']:.0f} degree={metrics['mean_degree']:.2f}",
        flush=True,
    )
    print(
        "  geodesic vs truth: "
        f"Cheb pearson/rho={_fmt(metrics['geodesic_cheb_pearson'])}/"
        f"{_fmt(metrics['geodesic_cheb_spearman'])} "
        f"Euclid pearson/rho={_fmt(metrics['geodesic_euclid_pearson'])}/"
        f"{_fmt(metrics['geodesic_euclid_spearman'])}",
        flush=True,
    )


def _print_layout_report(label: str, method: str, metrics: Mapping[str, float]) -> None:
    prefix = f"[{label}/{method}]"
    if method in {"spectral", "mds"}:
        print(
            f"{prefix} continuous: coord_dist_rho={_fmt(metrics['coordinate_distance_spearman'])} "
            f"ORACLE-affine R2={_fmt(metrics['oracle_affine_r2'])} "
            f"min_axis={_fmt(metrics['oracle_affine_r2_min_axis'])} "
            f"RMSE={_fmt(metrics['oracle_affine_rmse'])}",
            flush=True,
        )
        print(
            f"  ORACLE-affine Hungarian: placement={_fmt(metrics['oracle_affine_hungarian_placement'])} "
            f"neighbour={_fmt(metrics['oracle_affine_hungarian_neighbour'])} "
            f"right/down={_fmt(metrics['oracle_affine_hungarian_right_neighbour'])}/"
            f"{_fmt(metrics['oracle_affine_hungarian_down_neighbour'])}",
            flush=True,
        )
    else:
        print(
            f"{prefix} unsupervised FGW: objective={_fmt(metrics['fgw_objective'])} "
            f"start={metrics['fgw_best_start']:.0f} entropy={_fmt(metrics['fgw_transport_entropy'])}",
            flush=True,
        )
        print(
            f"  ORACLE-D4 frame only: placement={_fmt(metrics['fgw_d4_placement'])} "
            f"neighbour={_fmt(metrics['fgw_d4_neighbour'])} "
            f"right/down={_fmt(metrics['fgw_d4_right_neighbour'])}/"
            f"{_fmt(metrics['fgw_d4_down_neighbour'])}",
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_a", default=DEFAULT_CKPT_A, help="r=1 affinity checkpoint")
    parser.add_argument("--ckpt_b", default=DEFAULT_CKPT_B, help="r=3 affinity checkpoint")
    parser.add_argument("--n", type=int, default=1, help="held-out exact-synthetic images (default: 1)")
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument("--seed", type=int, default=SEED, help="fresh synthetic distortion seed")
    parser.add_argument(
        "--sources",
        default="raw,a,b,ensemble",
        help="comma-separated: raw,a,b,ensemble (oracle requires --include_oracle)",
    )
    parser.add_argument(
        "--methods",
        default="spectral,mds,fgw",
        help="comma-separated: spectral,mds,fgw",
    )
    parser.add_argument(
        "--fgw_sources",
        default="ensemble",
        help="sources for slower FGW; cheap spectral/MDS still run for all sources",
    )
    parser.add_argument("--include_oracle", action="store_true", help="add exact-proximity harness sanity source")
    parser.add_argument("--graph_k", type=int, default=16, help="directed candidate degree before mutual filter")
    parser.add_argument(
        "--union_graph",
        action="store_true",
        help="use union-kNN instead of conservative mutual-kNN graph",
    )
    parser.add_argument(
        "--rank_temperature",
        type=float,
        default=5.0,
        help="rank-decay temperature for weighted graph edges",
    )
    parser.add_argument("--fgw_alpha", type=float, default=0.85, help="structural GW weight in fused GW")
    parser.add_argument("--fgw_epsilon", type=float, default=0.08, help="outer FGW Sinkhorn entropy")
    parser.add_argument("--fgw_init_epsilon", type=float, default=0.12, help="feature-initialization Sinkhorn entropy")
    parser.add_argument("--fgw_iterations", type=int, default=12, help="outer FGW updates per D4 start")
    parser.add_argument("--sinkhorn_iterations", type=int, default=25, help="Sinkhorn updates per FGW update")
    parser.add_argument("--fgw_starts", type=int, default=8, help="bounded D4 starts for FGW (1..8)")
    parser.add_argument("--fgw_relaxation", type=float, default=0.7, help="FGW transport update relaxation")
    args = parser.parse_args()
    try:
        args.sources = _parse_csv(args.sources, allowed=ALL_SOURCES, name="sources")
        args.methods = _parse_csv(args.methods, allowed=ALL_METHODS, name="methods")
        args.fgw_sources = _parse_csv(args.fgw_sources, allowed=ALL_SOURCES, name="fgw_sources")
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.include_oracle and "oracle" not in args.sources:
        args.sources = (*args.sources, "oracle")
    if "oracle" in args.sources and not args.include_oracle:
        parser.error("source 'oracle' requires --include_oracle")
    if any(source not in args.sources for source in args.fgw_sources):
        parser.error("--fgw_sources must be a subset of --sources (after --include_oracle)")
    if args.n < 1:
        parser.error("--n must be positive")
    if not 1 <= args.graph_k < NFRAG:
        parser.error(f"--graph_k must be in [1,{NFRAG - 1}]")
    if args.rank_temperature <= 0.0:
        parser.error("--rank_temperature must be positive")
    if not 0.0 <= args.fgw_alpha <= 1.0:
        parser.error("--fgw_alpha must be in [0,1]")
    if args.fgw_epsilon <= 0.0 or args.fgw_init_epsilon <= 0.0:
        parser.error("FGW epsilons must be positive")
    if args.fgw_iterations < 1 or args.sinkhorn_iterations < 1:
        parser.error("FGW and Sinkhorn iterations must be positive")
    if not 1 <= args.fgw_starts <= len(D4_MATRICES):
        parser.error("--fgw_starts must be in [1,8]")
    if not 0.0 < args.fgw_relaxation <= 1.0:
        parser.error("--fgw_relaxation must be in (0,1]")
    return args


def _checkpoint_line(label: str, path: str, model: torch.nn.Module, metadata: Mapping[str, Any]) -> str:
    step = metadata.get("step") if isinstance(metadata, Mapping) else None
    return (
        f"{label}={os.path.abspath(path)} params={count_params(model):,}"
        + (f" step={step}" if step is not None else "")
    )


def main() -> None:
    args = _parse_args()
    device = _parse_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    needs_a = any(source in {"a", "ensemble"} for source in args.sources)
    needs_b = any(source in {"b", "ensemble"} for source in args.sources)
    model_a: torch.nn.Module | None = None
    model_b: torch.nn.Module | None = None
    metadata_a: Mapping[str, Any] = {}
    metadata_b: Mapping[str, Any] = {}
    if needs_a:
        model_a, metadata_a = load_model(args.ckpt_a, device)
    if needs_b:
        model_b, metadata_b = load_model(args.ckpt_b, device)

    print(
        f"device={device} images={args.n} sources={','.join(args.sources)} "
        f"methods={','.join(args.methods)} graph={'union' if args.union_graph else 'mutual'} "
        f"k={args.graph_k}",
        flush=True,
    )
    if model_a is not None:
        print(_checkpoint_line("model_a", args.ckpt_a, model_a, metadata_a), flush=True)
    if model_b is not None:
        print(_checkpoint_line("model_b", args.ckpt_b, model_b, metadata_b), flush=True)
    print(
        "Continuous-layout Hungarian metrics are ORACLE-affine diagnostics; FGW uses no truth "
        "and is scored only after an ORACLE D4 frame choice.  Neither is a deployable orientation step.",
        flush=True,
    )

    train_names, val_names = train_val_split()
    del train_names
    if args.n > len(val_names):
        raise ValueError(f"--n={args.n} exceeds held-out split size {len(val_names)}")
    dataset = CanvasDataset(val_names[: args.n], real_prob=0.0, seed=args.seed)

    graph_totals: dict[str, list[dict[str, float]]] = defaultdict(list)
    layout_totals: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    elapsed_start = time.perf_counter()

    for image_index in range(args.n):
        sample = dataset[image_index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("this diagnostic requires exact synthetic samples")
        tiles = sample["tiles"]
        perm = sample["perm"]
        truth = _truth_coordinates(perm)
        source_affinities: dict[str, np.ndarray] = {}
        first: np.ndarray | None = None
        second: np.ndarray | None = None
        if "raw" in args.sources:
            source_affinities["raw"] = raw_zpixel_affinity(tiles).detach().cpu().numpy().astype(np.float64)
        if needs_a:
            if model_a is None:
                raise RuntimeError("model_a was required but not loaded")
            first = learned_affinity(model_a, tiles, device).detach().cpu().numpy().astype(np.float64)
            if "a" in args.sources:
                source_affinities["a"] = first
        if needs_b:
            if model_b is None:
                raise RuntimeError("model_b was required but not loaded")
            second = learned_affinity(model_b, tiles, device).detach().cpu().numpy().astype(np.float64)
            if "b" in args.sources:
                source_affinities["b"] = second
        if "ensemble" in args.sources:
            if first is None or second is None:
                raise RuntimeError("ensemble requires both frozen affinity models")
            source_affinities["ensemble"] = _rank_percentile_ensemble(first, second)
        if "oracle" in args.sources:
            # This deliberately leaks synthetic labels solely to prove that the
            # exact same graph/layout machinery can recover a true grid metric.
            source_affinities["oracle"] = -_pairwise_euclidean(truth)

        print(f"\nimage {image_index + 1}/{args.n}", flush=True)
        for source in args.sources:
            affinity = source_affinities[source]
            graph = _weighted_knn_geodesics(
                affinity,
                k=args.graph_k,
                rank_temperature=args.rank_temperature,
                mutual=not args.union_graph,
            )
            graph_metrics = _distance_signal_metrics(graph, truth)
            graph_totals[source].append(graph_metrics)
            _print_graph_report(source, graph_metrics)

            mds_coordinates: np.ndarray | None = None
            if "spectral" in args.methods:
                metrics = _continuous_layout_metrics(_spectral_layout(graph.weights), perm)
                layout_totals[(source, "spectral")].append(metrics)
                _print_layout_report(source, "spectral", metrics)
            if "mds" in args.methods or ("fgw" in args.methods and source in args.fgw_sources):
                mds_coordinates = _classical_mds(graph.distance)
            if "mds" in args.methods:
                if mds_coordinates is None:
                    raise RuntimeError("internal MDS layout was not computed")
                metrics = _continuous_layout_metrics(mds_coordinates, perm)
                layout_totals[(source, "mds")].append(metrics)
                _print_layout_report(source, "mds", metrics)
            if "fgw" in args.methods and source in args.fgw_sources:
                if mds_coordinates is None:
                    raise RuntimeError("FGW needs MDS initialization coordinates")
                slots, metrics = _fgw_match(
                    graph.distance,
                    mds_coordinates,
                    device=device,
                    alpha=args.fgw_alpha,
                    epsilon=args.fgw_epsilon,
                    init_epsilon=args.fgw_init_epsilon,
                    outer_iterations=args.fgw_iterations,
                    sinkhorn_iterations=args.sinkhorn_iterations,
                    starts=args.fgw_starts,
                    relaxation=args.fgw_relaxation,
                )
                metrics.update(_best_d4_metrics(slots, perm))
                layout_totals[(source, "fgw")].append(metrics)
                _print_layout_report(source, "fgw", metrics)

    if args.n > 1:
        print("\n===== mean over exact-synthetic images =====", flush=True)
        for source in args.sources:
            _print_graph_report(source, _mean_dict(graph_totals[source]))
            for method in args.methods:
                rows = layout_totals.get((source, method), [])
                if rows:
                    _print_layout_report(source, method, _mean_dict(rows))
    elapsed = time.perf_counter() - elapsed_start
    print(f"\ncompleted images={args.n} elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
