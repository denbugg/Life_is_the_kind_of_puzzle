"""Robust translation synchronization followed by exact grid projection.

Each directed edge encodes the canonical displacement ``(column, row)`` from
``edges_src[e]`` to ``edges_dst[e]``.  The continuous problem is solved by a
graduated non-convex IRLS schedule over a weighted graph Laplacian.  Every GNC
stage is projected to the finite grid with a full Hungarian assignment, so the
public result is always a one-tile-per-cell permutation.

Candidate selection and restart selection use only the supplied edges,
confidences, offsets, and initial grid.  No target layout is accepted by this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class GncTlsConfig:
    """Configuration for robust graph synchronization and grid projection."""

    grid_size: int = 24
    gnc_stages: int = 8
    irls_iterations: int = 4
    # Upper bound on the data-derived convex starting value.  The actual
    # start follows the GNC-TLS initialization from the largest residual.
    gnc_mu_initial: float = 0.05
    gnc_mu_final: float = 100.0
    robust_cutoff: float = 0.75
    initial_anchor_weight: float = 1e-3
    current_anchor_weight: float = 0.0
    regularization: float = 1e-8
    max_candidates_per_tile: int = 64
    max_candidate_radius: float = 4.0
    restarts: int = 2
    start_perturbation: float = 0.05
    assignment_jitter: float = 1e-9

    def __post_init__(self) -> None:
        integer_fields = {
            "grid_size": self.grid_size,
            "gnc_stages": self.gnc_stages,
            "irls_iterations": self.irls_iterations,
            "max_candidates_per_tile": self.max_candidates_per_tile,
            "restarts": self.restarts,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        finite_positive = {
            "gnc_mu_initial": self.gnc_mu_initial,
            "gnc_mu_final": self.gnc_mu_final,
            "robust_cutoff": self.robust_cutoff,
            "regularization": self.regularization,
            "max_candidate_radius": self.max_candidate_radius,
        }
        for name, value in finite_positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.gnc_mu_final < self.gnc_mu_initial:
            raise ValueError("gnc_mu_final must be at least gnc_mu_initial")
        if self.initial_anchor_weight <= 0.0:
            raise ValueError("initial_anchor_weight must be positive to fix every component gauge")
        finite_nonnegative = {
            "initial_anchor_weight": self.initial_anchor_weight,
            "current_anchor_weight": self.current_anchor_weight,
            "start_perturbation": self.start_perturbation,
            "assignment_jitter": self.assignment_jitter,
        }
        for name, value in finite_nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class GncTlsResult:
    """Best input-only projected solution found across stages and restarts."""

    tile_to_cell: np.ndarray
    grid: np.ndarray
    continuous_positions: np.ndarray
    diagnostics: dict[str, Any]


def _validate_inputs(
    edges_src: np.ndarray,
    edges_dst: np.ndarray,
    offsets: np.ndarray,
    confidence: np.ndarray,
    initial_grid: np.ndarray,
    config: GncTlsConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(config, GncTlsConfig):
        raise TypeError("config must be a GncTlsConfig")
    grid = np.asarray(initial_grid)
    expected_grid = (config.grid_size, config.grid_size)
    if grid.shape != expected_grid:
        raise ValueError(f"initial_grid must have shape {expected_grid}")
    if not np.issubdtype(grid.dtype, np.integer):
        raise TypeError("initial_grid must contain integer tile ids")
    tile_count = config.grid_size**2
    flat = grid.astype(np.int64, copy=False).ravel()
    if not np.array_equal(np.sort(flat), np.arange(tile_count, dtype=np.int64)):
        raise ValueError("initial_grid must be a permutation of all tile ids")

    source = np.asarray(edges_src)
    destination = np.asarray(edges_dst)
    if source.ndim != 1 or destination.ndim != 1 or source.shape != destination.shape:
        raise ValueError("edges_src and edges_dst must be equal-length one-dimensional arrays")
    if len(source) == 0:
        raise ValueError("at least one directed edge is required")
    if not np.issubdtype(source.dtype, np.integer) or not np.issubdtype(
        destination.dtype, np.integer
    ):
        raise TypeError("edge endpoints must be integer tile ids")
    source = source.astype(np.int64, copy=False)
    destination = destination.astype(np.int64, copy=False)
    if (
        int(source.min()) < 0
        or int(destination.min()) < 0
        or int(source.max()) >= tile_count
        or int(destination.max()) >= tile_count
    ):
        raise ValueError("edge endpoint is outside the tile-id range")
    if np.any(source == destination):
        raise ValueError("self edges are not valid translation constraints")

    displacement = np.asarray(offsets, dtype=np.float64)
    if displacement.shape != (len(source), 2):
        raise ValueError(f"offsets must have shape {(len(source), 2)}")
    if not np.all(np.isfinite(displacement)):
        raise ValueError("offsets must be finite")
    maximum_offset = config.grid_size - 1
    if np.any(np.abs(displacement) > maximum_offset):
        raise ValueError("offset lies outside the finite grid extent")

    edge_confidence = np.asarray(confidence, dtype=np.float64)
    if edge_confidence.shape != (len(source),):
        raise ValueError(f"confidence must have shape {(len(source),)}")
    if not np.all(np.isfinite(edge_confidence)):
        raise ValueError("confidence must be finite")
    if np.any((edge_confidence < 0.0) | (edge_confidence > 1.0)):
        raise ValueError("confidence must lie in [0, 1]")
    if not np.any(edge_confidence > 0.0):
        raise ValueError("at least one edge must have positive confidence")
    return source, destination, displacement, edge_confidence, grid.astype(np.int32, copy=True)


def _tile_positions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = int(grid.shape[0])
    tile_count = size**2
    tile_to_cell = np.empty(tile_count, dtype=np.int32)
    tile_to_cell[grid.ravel()] = np.arange(tile_count, dtype=np.int32)
    positions = np.column_stack(
        [tile_to_cell % size, tile_to_cell // size]
    ).astype(np.float64)
    return tile_to_cell, positions


def _residual_vectors(
    positions: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    return positions[destination] - positions[source] - offsets


def _gnc_tls_weights(
    residual_squared: np.ndarray, cutoff: float, mu: float
) -> np.ndarray:
    """Return the exact piecewise GNC-TLS weight update for one ``mu``."""

    r2 = np.asarray(residual_squared, dtype=np.float64)
    c2 = float(cutoff) ** 2
    lower = (float(mu) / (float(mu) + 1.0)) * c2
    upper = ((float(mu) + 1.0) / float(mu)) * c2
    output = np.empty_like(r2)
    low = r2 <= lower
    high = r2 >= upper
    middle = ~(low | high)
    output[low] = 1.0
    output[high] = 0.0
    if np.any(middle):
        output[middle] = (
            np.sqrt(c2 * float(mu) * (float(mu) + 1.0) / r2[middle])
            - float(mu)
        )
    return np.clip(output, 0.0, 1.0)


def _deduplicate_and_normalize(
    source: np.ndarray,
    destination: np.ndarray,
    offsets: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Max-deduplicate exact edges and normalize each query-side group."""

    best: dict[tuple[int, int, float, float], float] = {}
    for src, dst, offset, value in zip(
        source.tolist(), destination.tolist(), offsets.tolist(), confidence.tolist(), strict=True
    ):
        key = (int(src), int(dst), float(offset[0]), float(offset[1]))
        best[key] = max(best.get(key, 0.0), float(value))
    keys = sorted(best)
    deduplicated_source = np.asarray([key[0] for key in keys], dtype=np.int64)
    deduplicated_destination = np.asarray([key[1] for key in keys], dtype=np.int64)
    deduplicated_offsets = np.asarray([[key[2], key[3]] for key in keys], dtype=np.float64)
    deduplicated_confidence = np.asarray([best[key] for key in keys], dtype=np.float64)

    group_total: dict[tuple[int, float, float], float] = {}
    for key, value in zip(keys, deduplicated_confidence.tolist(), strict=True):
        group = (key[0], key[2], key[3])
        group_total[group] = group_total.get(group, 0.0) + float(value)
    normalized = np.empty_like(deduplicated_confidence)
    zero_groups: set[tuple[int, float, float]] = set()
    for index, key in enumerate(keys):
        group = (key[0], key[2], key[3])
        total = group_total[group]
        if total > 0.0:
            normalized[index] = deduplicated_confidence[index] / total
        else:
            normalized[index] = 0.0
            zero_groups.add(group)
    return (
        deduplicated_source,
        deduplicated_destination,
        deduplicated_offsets,
        normalized,
        {
            "deduplicated_edge_count": len(keys),
            "query_side_group_count": len(group_total),
            "zero_confidence_group_count": len(zero_groups),
        },
    )


def _solve_laplacian(
    source: np.ndarray,
    destination: np.ndarray,
    offsets: np.ndarray,
    effective_weight: np.ndarray,
    initial_positions: np.ndarray,
    current_anchor: np.ndarray,
    config: GncTlsConfig,
) -> np.ndarray:
    tile_count = len(initial_positions)
    rows = np.concatenate([source, destination, source, destination])
    columns = np.concatenate([source, destination, destination, source])
    values = np.concatenate(
        [effective_weight, effective_weight, -effective_weight, -effective_weight]
    )
    laplacian = coo_matrix(
        (values, (rows, columns)), shape=(tile_count, tile_count), dtype=np.float64
    ).tocsr()
    # Every node receives a weak initial anchor.  This fixes every connected
    # component without privileging an arbitrary tile as a point gauge.
    diagonal_weight = config.initial_anchor_weight + config.current_anchor_weight + config.regularization
    matrix = laplacian + eye(tile_count, format="csr", dtype=np.float64) * diagonal_weight
    right_hand_side = np.zeros((tile_count, 2), dtype=np.float64)
    weighted_offsets = effective_weight[:, None] * offsets
    np.add.at(right_hand_side, source, -weighted_offsets)
    np.add.at(right_hand_side, destination, weighted_offsets)
    right_hand_side += config.initial_anchor_weight * initial_positions
    right_hand_side += config.current_anchor_weight * current_anchor
    right_hand_side += config.regularization * initial_positions
    solved = np.asarray(spsolve(matrix, right_hand_side), dtype=np.float64)
    if solved.shape != (tile_count, 2) or not np.all(np.isfinite(solved)):
        raise RuntimeError("Laplacian synchronization produced invalid positions")
    return solved


def _project_hungarian(
    positions: np.ndarray,
    initial_tile_to_cell: np.ndarray,
    current_tile_to_cell: np.ndarray,
    config: GncTlsConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    size = config.grid_size
    tile_count = size**2
    cell_coordinates = np.column_stack(
        [np.arange(tile_count) % size, np.arange(tile_count) // size]
    ).astype(np.float64)
    squared_distance = np.sum(
        (positions[:, None, :] - cell_coordinates[None, :, :]) ** 2, axis=2
    )
    candidate_count = min(config.max_candidates_per_tile, tile_count)
    allowed = squared_distance <= config.max_candidate_radius**2
    nearest = np.argsort(squared_distance, axis=1, kind="stable")[:, :candidate_count]
    allowed[np.arange(tile_count)[:, None], nearest] = True
    allowed[np.arange(tile_count), initial_tile_to_cell] = True
    allowed[np.arange(tile_count), current_tile_to_cell] = True

    maximum_allowed = float(np.max(squared_distance[allowed]))
    forbidden_cost = (tile_count + 1.0) * (
        maximum_allowed + config.assignment_jitter + 1.0
    )
    assignment_cost = squared_distance.copy()
    assignment_cost[~allowed] = forbidden_cost
    if config.assignment_jitter > 0.0:
        jitter = rng.uniform(0.0, config.assignment_jitter, size=assignment_cost.shape)
        assignment_cost[allowed] += jitter[allowed]
    row_indices, cell_indices = linear_sum_assignment(assignment_cost)
    if not np.array_equal(row_indices, np.arange(tile_count)):
        raise RuntimeError("Hungarian projection did not assign every tile")
    tile_to_cell = cell_indices.astype(np.int32)
    if not np.array_equal(np.sort(tile_to_cell), np.arange(tile_count)):
        raise RuntimeError("Hungarian projection did not produce a permutation")
    grid = np.empty(tile_count, dtype=np.int32)
    grid[tile_to_cell] = np.arange(tile_count, dtype=np.int32)
    outside_assignments = int(np.sum(~allowed[np.arange(tile_count), tile_to_cell]))
    if outside_assignments != 0:
        raise RuntimeError("candidate-restricted Hungarian projection used a forbidden assignment")
    geometric_cost = float(squared_distance[np.arange(tile_count), tile_to_cell].sum())
    return tile_to_cell, grid.reshape(size, size), geometric_cost, outside_assignments


def _projected_edge_score(
    tile_to_cell: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    offsets: np.ndarray,
    confidence: np.ndarray,
    config: GncTlsConfig,
) -> tuple[float, float, float]:
    size = config.grid_size
    projected = np.column_stack(
        [tile_to_cell % size, tile_to_cell // size]
    ).astype(np.float64)
    residual = np.linalg.norm(
        _residual_vectors(projected, source, destination, offsets), axis=1
    )
    denominator = float(confidence.sum())
    truncated_squared = np.minimum(residual**2, config.robust_cutoff**2)
    score = float(np.dot(confidence, truncated_squared) / denominator)
    mean_residual = float(np.dot(confidence, residual) / denominator)
    consistent_fraction = float(
        np.dot(confidence, residual <= config.robust_cutoff) / denominator
    )
    return score, mean_residual, consistent_fraction


def solve_gnc_tls(
    edges_src: np.ndarray,
    edges_dst: np.ndarray,
    offsets: np.ndarray,
    confidence: np.ndarray,
    initial_grid: np.ndarray,
    config: GncTlsConfig = GncTlsConfig(),
    seed: int = 0,
) -> GncTlsResult:
    """Synchronize edge translations and return the best exact grid permutation."""

    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    source, destination, displacement, edge_confidence, grid = _validate_inputs(
        edges_src, edges_dst, offsets, confidence, initial_grid, config
    )
    raw_edge_count = len(source)
    source, destination, displacement, edge_confidence, deduplication = (
        _deduplicate_and_normalize(
            source, destination, displacement, edge_confidence
        )
    )
    if not np.any(edge_confidence > 0.0):
        raise ValueError("deduplicated query-side groups contain no positive confidence")
    initial_tile_to_cell, initial_positions = _tile_positions(grid)
    # Standard GNC initialization: first solve the confidence-weighted convex
    # problem with every robust weight equal to one.  The discrete initial grid
    # remains the fail-safe candidate, but it must not determine the GNC scale.
    convex_positions = _solve_laplacian(
        source,
        destination,
        displacement,
        edge_confidence,
        initial_positions,
        initial_positions,
        config,
    )
    grid_center = (config.grid_size - 1.0) / 2.0
    convex_positions += np.asarray([grid_center, grid_center]) - np.mean(
        convex_positions, axis=0
    )
    convex_residual_squared = np.sum(
        _residual_vectors(convex_positions, source, destination, displacement) ** 2,
        axis=1,
    )
    c2 = config.robust_cutoff**2
    positive_confidence = edge_confidence > 0.0
    maximum_convex_residual = float(
        np.max(convex_residual_squared[positive_confidence])
    )
    denominator = 2.0 * maximum_convex_residual / c2 - 1.0
    derived_mu_initial = (
        1.0 / denominator if denominator > 0.0 else config.gnc_mu_final
    )
    mu_initial = min(config.gnc_mu_initial, derived_mu_initial)
    mu_initial = max(mu_initial, np.finfo(np.float64).eps)
    mu_schedule = (
        np.asarray([config.gnc_mu_final], dtype=np.float64)
        if config.gnc_stages == 1
        else np.geomspace(mu_initial, config.gnc_mu_final, config.gnc_stages)
    )
    seed_sequence = np.random.SeedSequence(int(seed))
    restart_sequences = seed_sequence.spawn(config.restarts)

    initial_score, initial_mean_residual, initial_consistent_fraction = _projected_edge_score(
        initial_tile_to_cell,
        source,
        destination,
        displacement,
        edge_confidence,
        config,
    )
    best_score = initial_score
    best_tile_to_cell: np.ndarray | None = initial_tile_to_cell.copy()
    best_grid: np.ndarray | None = grid.copy()
    best_continuous: np.ndarray | None = initial_positions.copy()
    best_restart = -1
    best_stage = -1
    stage_diagnostics: list[dict[str, Any]] = []

    for restart, restart_sequence in enumerate(restart_sequences):
        rng = np.random.default_rng(restart_sequence)
        continuous = convex_positions.copy()
        if restart > 0 and config.start_perturbation > 0.0:
            continuous += rng.normal(
                0.0, config.start_perturbation, size=continuous.shape
            )
            continuous += np.asarray([grid_center, grid_center]) - np.mean(
                continuous, axis=0
            )
        current_tile_to_cell = initial_tile_to_cell.copy()
        current_anchor = initial_positions.copy()
        previous_robust_weight = np.ones(len(source), dtype=np.float64)

        for stage, mu in enumerate(mu_schedule.tolist()):
            update_records: list[dict[str, int | float]] = []
            for iteration in range(config.irls_iterations):
                residual_squared = np.sum(
                    _residual_vectors(
                        continuous, source, destination, displacement
                    )
                    ** 2,
                    axis=1,
                )
                # Fresh update from the current residual: weights may recover as
                # an edge becomes geometrically consistent.
                robust_weight = _gnc_tls_weights(
                    residual_squared, config.robust_cutoff, mu
                )
                delta = robust_weight - previous_robust_weight
                update_records.append(
                    {
                        "iteration": iteration,
                        "increases": int(np.sum(delta > 1e-12)),
                        "decreases": int(np.sum(delta < -1e-12)),
                        "unchanged": int(np.sum(np.abs(delta) <= 1e-12)),
                        "mean_weight": float(np.mean(robust_weight)),
                    }
                )
                previous_robust_weight = robust_weight.copy()
                effective_weight = edge_confidence * robust_weight
                continuous = _solve_laplacian(
                    source,
                    destination,
                    displacement,
                    effective_weight,
                    initial_positions,
                    current_anchor,
                    config,
                )
                continuous += np.asarray([grid_center, grid_center]) - np.mean(
                    continuous, axis=0
                )

            residual_squared = np.sum(
                _residual_vectors(continuous, source, destination, displacement) ** 2,
                axis=1,
            )
            final_robust_weight = _gnc_tls_weights(
                residual_squared, config.robust_cutoff, mu
            )
            final_delta = final_robust_weight - previous_robust_weight
            update_records.append(
                {
                    "iteration": config.irls_iterations,
                    "increases": int(np.sum(final_delta > 1e-12)),
                    "decreases": int(np.sum(final_delta < -1e-12)),
                    "unchanged": int(np.sum(np.abs(final_delta) <= 1e-12)),
                    "mean_weight": float(np.mean(final_robust_weight)),
                }
            )
            robust_weight = final_robust_weight
            previous_robust_weight = robust_weight.copy()
            residual = np.sqrt(residual_squared)

            tile_to_cell, projected_grid, projection_cost, outside_assignments = (
                _project_hungarian(
                    continuous,
                    initial_tile_to_cell,
                    current_tile_to_cell,
                    config,
                    rng,
                )
            )
            projected_score, projected_mean_residual, consistent_fraction = (
                _projected_edge_score(
                    tile_to_cell,
                    source,
                    destination,
                    displacement,
                    edge_confidence,
                    config,
                )
            )
            current_tile_to_cell = tile_to_cell
            current_anchor = np.column_stack(
                [tile_to_cell % config.grid_size, tile_to_cell // config.grid_size]
            ).astype(np.float64)
            stage_diagnostics.append(
                {
                    "restart": restart,
                    "stage": stage,
                    "mu": float(mu),
                    "robust_cutoff": config.robust_cutoff,
                    "irls_iterations": config.irls_iterations,
                    "projected_edge_score": projected_score,
                    "projected_mean_residual": projected_mean_residual,
                    "consistent_confidence_fraction": consistent_fraction,
                    "continuous_mean_residual": float(np.mean(residual)),
                    "continuous_max_residual": float(np.max(residual)),
                    "mean_robust_weight": float(np.mean(robust_weight)),
                    "min_robust_weight": float(np.min(robust_weight)),
                    "max_robust_weight": float(np.max(robust_weight)),
                    "projection_squared_distance": projection_cost,
                    "outside_candidate_assignments": outside_assignments,
                    "weight_updates": update_records,
                    "weight_increases": int(
                        sum(record["increases"] for record in update_records)
                    ),
                    "weight_decreases": int(
                        sum(record["decreases"] for record in update_records)
                    ),
                }
            )
            selection_score = projected_score
            if selection_score < best_score - 1e-12:
                best_score = selection_score
                best_tile_to_cell = tile_to_cell.copy()
                best_grid = projected_grid.copy()
                best_continuous = continuous.copy()
                best_restart = restart
                best_stage = stage

    if best_tile_to_cell is None or best_grid is None or best_continuous is None:
        raise RuntimeError("GNC/TLS did not produce a projected solution")
    if not (
        np.all(np.isfinite(best_continuous))
        and np.isfinite(best_score)
        and np.array_equal(np.sort(best_grid.ravel()), np.arange(config.grid_size**2))
    ):
        raise RuntimeError("GNC/TLS result failed final integrity validation")
    diagnostics = {
        "seed": int(seed),
        "tile_count": int(config.grid_size**2),
        "raw_edge_count": int(raw_edge_count),
        "deduplicated_edge_count": int(len(source)),
        "query_side_group_count": deduplication["query_side_group_count"],
        "zero_confidence_group_count": deduplication["zero_confidence_group_count"],
        "positive_confidence_edges": int(np.sum(edge_confidence > 0.0)),
        "mu_schedule": [float(value) for value in mu_schedule],
        "derived_mu_initial": float(derived_mu_initial),
        "used_mu_initial": float(mu_initial),
        # Keep the legacy field name for evaluator compatibility, but bind it to
        # the scientifically correct convex initialization residual.
        "maximum_initial_residual_squared": maximum_convex_residual,
        "mu_derived_from": "confidence_weighted_convex_all_robust_weights_one",
        "convex_initialization": {
            "all_robust_weights_one": True,
            "positions_sha256": hashlib.sha256(
                np.ascontiguousarray(convex_positions, dtype=np.float64).tobytes()
            ).hexdigest(),
            "residual_squared_sha256": hashlib.sha256(
                np.ascontiguousarray(convex_residual_squared, dtype=np.float64).tobytes()
            ).hexdigest(),
            "maximum_residual_squared": maximum_convex_residual,
            "mean_residual_squared": float(np.mean(convex_residual_squared)),
            "confidence_weighted_mean_residual_squared": float(
                np.dot(edge_confidence, convex_residual_squared)
                / edge_confidence.sum()
            ),
            "centroid_column_row": [
                float(value) for value in np.mean(convex_positions, axis=0)
            ],
        },
        "robust_cutoff": config.robust_cutoff,
        "restarts": config.restarts,
        "best_restart": best_restart,
        "best_stage": best_stage,
        "best_input_edge_score": float(best_score),
        "initial_input_edge_score": initial_score,
        "initial_mean_residual": initial_mean_residual,
        "initial_consistent_confidence_fraction": initial_consistent_fraction,
        "initial_grid_selected": best_restart == -1,
        "selection_policy": "minimum confidence-weighted truncated input-edge residual",
        "stages": stage_diagnostics,
    }
    return GncTlsResult(
        tile_to_cell=best_tile_to_cell,
        grid=best_grid,
        continuous_positions=best_continuous,
        diagnostics=diagnostics,
    )


__all__ = ["GncTlsConfig", "GncTlsResult", "solve_gnc_tls"]
