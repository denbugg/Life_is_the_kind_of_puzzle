"""Categorical group-switch synchronization for ambiguous puzzle edges.

Candidates sharing ``(source, dx, dy)`` are mutually exclusive hypotheses.
Each group owns one categorical posterior over its candidates plus an explicit
null state, so increasing top-k cannot increase the total constraint mass of a
query side.  Continuous coordinates alternate with posterior updates and are
projected to the finite grid by an exact Hungarian assignment after every
temperature stage.
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
class GroupSwitchConfig:
    grid_size: int = 24
    stages: int = 8
    iterations_per_stage: int = 4
    temperature_initial: float = 4.0
    temperature_final: float = 0.05
    prior_power: float = 1.0
    prior_epsilon: float = 1e-12
    null_prior: float = 0.10
    null_cost: float = 0.75
    selection_prior_penalty: float = 0.02
    initial_anchor_weight: float = 1e-3
    current_anchor_weight: float = 0.0
    regularization: float = 1e-8
    max_candidates_per_tile: int = 64
    max_candidate_radius: float = 4.0
    restarts: int = 3
    gumbel_scale: float = 1.0
    start_perturbation: float = 0.02
    assignment_jitter: float = 1e-9

    def __post_init__(self) -> None:
        integer_fields = {
            "grid_size": self.grid_size,
            "stages": self.stages,
            "iterations_per_stage": self.iterations_per_stage,
            "max_candidates_per_tile": self.max_candidates_per_tile,
            "restarts": self.restarts,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        positive = {
            "temperature_initial": self.temperature_initial,
            "temperature_final": self.temperature_final,
            "prior_power": self.prior_power,
            "prior_epsilon": self.prior_epsilon,
            "null_prior": self.null_prior,
            "null_cost": self.null_cost,
            "initial_anchor_weight": self.initial_anchor_weight,
            "regularization": self.regularization,
            "max_candidate_radius": self.max_candidate_radius,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.temperature_initial < self.temperature_final:
            raise ValueError("temperature_initial must be at least temperature_final")
        if not 0.0 < self.null_prior <= 1.0:
            raise ValueError("null_prior must lie in (0, 1]")
        nonnegative = {
            "selection_prior_penalty": self.selection_prior_penalty,
            "current_anchor_weight": self.current_anchor_weight,
            "gumbel_scale": self.gumbel_scale,
            "start_perturbation": self.start_perturbation,
            "assignment_jitter": self.assignment_jitter,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class GroupSwitchResult:
    tile_to_cell: np.ndarray
    grid: np.ndarray
    continuous_positions: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _PreparedEdges:
    source: np.ndarray
    destination: np.ndarray
    offsets: np.ndarray
    prior: np.ndarray
    groups: tuple[np.ndarray, ...]
    group_keys: tuple[tuple[int, float, float], ...]
    raw_edge_count: int
    zero_prior_groups: int


@dataclass(frozen=True)
class _CategoricalPriors:
    candidate_probability: np.ndarray
    candidate_log_probability: np.ndarray
    null_probability: np.ndarray
    null_log_probability: np.ndarray
    max_normalization_error: float


def _validate_and_prepare(
    edges_src: np.ndarray,
    edges_dst: np.ndarray,
    offsets: np.ndarray,
    confidence: np.ndarray,
    initial_grid: np.ndarray,
    config: GroupSwitchConfig,
) -> tuple[_PreparedEdges, np.ndarray]:
    if not isinstance(config, GroupSwitchConfig):
        raise TypeError("config must be a GroupSwitchConfig")
    grid = np.asarray(initial_grid)
    expected = (config.grid_size, config.grid_size)
    if grid.shape != expected:
        raise ValueError(f"initial_grid must have shape {expected}")
    if not np.issubdtype(grid.dtype, np.integer):
        raise TypeError("initial_grid must contain integer tile ids")
    tile_count = config.grid_size**2
    flat_grid = grid.astype(np.int64, copy=False).ravel()
    if not np.array_equal(np.sort(flat_grid), np.arange(tile_count)):
        raise ValueError("initial_grid must be a permutation of every tile id")

    source = np.asarray(edges_src)
    destination = np.asarray(edges_dst)
    if source.ndim != 1 or destination.ndim != 1 or source.shape != destination.shape:
        raise ValueError("edges_src and edges_dst must be equal-length vectors")
    if len(source) == 0:
        raise ValueError("at least one candidate edge is required")
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
        raise ValueError("edge endpoint lies outside the tile-id range")
    if np.any(source == destination):
        raise ValueError("self edges are invalid group-switch candidates")
    displacement = np.asarray(offsets, dtype=np.float64)
    if displacement.shape != (len(source), 2) or not np.all(np.isfinite(displacement)):
        raise ValueError(f"offsets must be finite with shape {(len(source), 2)}")
    if np.any(np.abs(displacement) > config.grid_size - 1):
        raise ValueError("offset lies outside the finite grid extent")
    values = np.asarray(confidence, dtype=np.float64)
    if values.shape != (len(source),) or not np.all(np.isfinite(values)):
        raise ValueError(f"confidence must be finite with shape {(len(source),)}")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("confidence must lie in [0, 1]")

    # Exact duplicate candidates are one hypothesis; repeated producers cannot
    # multiply their posterior mass.
    deduplicated: dict[tuple[int, int, float, float], float] = {}
    for src, dst, offset, value in zip(
        source.tolist(), destination.tolist(), displacement.tolist(), values.tolist(), strict=True
    ):
        key = (int(src), int(dst), float(offset[0]), float(offset[1]))
        deduplicated[key] = max(deduplicated.get(key, 0.0), float(value))
    keys = sorted(deduplicated)
    source = np.asarray([key[0] for key in keys], dtype=np.int64)
    destination = np.asarray([key[1] for key in keys], dtype=np.int64)
    displacement = np.asarray([[key[2], key[3]] for key in keys], dtype=np.float64)
    values = np.asarray([deduplicated[key] for key in keys], dtype=np.float64)

    group_lookup: dict[tuple[int, float, float], list[int]] = {}
    for index, key in enumerate(keys):
        group_lookup.setdefault((key[0], key[2], key[3]), []).append(index)
    group_keys = tuple(sorted(group_lookup))
    groups = tuple(np.asarray(group_lookup[key], dtype=np.int64) for key in group_keys)
    prior = np.zeros_like(values)
    zero_prior_groups = 0
    for indices in groups:
        total = float(values[indices].sum())
        if total > 0.0:
            prior[indices] = values[indices] / total
        else:
            zero_prior_groups += 1
    prepared = _PreparedEdges(
        source=source,
        destination=destination,
        offsets=displacement,
        prior=prior,
        groups=groups,
        group_keys=group_keys,
        raw_edge_count=len(edges_src),
        zero_prior_groups=zero_prior_groups,
    )
    return prepared, grid.astype(np.int32, copy=True)


def _tile_positions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = int(grid.shape[0])
    count = size**2
    tile_to_cell = np.empty(count, dtype=np.int32)
    tile_to_cell[grid.ravel()] = np.arange(count, dtype=np.int32)
    positions = np.column_stack([tile_to_cell % size, tile_to_cell // size]).astype(
        np.float64
    )
    return tile_to_cell, positions


def _residual_squared(positions: np.ndarray, edges: _PreparedEdges) -> np.ndarray:
    residual = (
        positions[edges.destination] - positions[edges.source] - edges.offsets
    )
    return np.sum(residual**2, axis=1)


def _categorical_priors(
    edges: _PreparedEdges, config: GroupSwitchConfig
) -> _CategoricalPriors:
    """Build the shared normalized candidate-plus-null prior for every group."""

    candidate_probability = np.zeros(len(edges.source), dtype=np.float64)
    null_probability = np.ones(len(edges.groups), dtype=np.float64)
    errors = []
    for group_index, indices in enumerate(edges.groups):
        positive = edges.prior[indices] > 0.0
        if np.any(positive) and config.null_prior < 1.0:
            active = indices[positive]
            transformed = edges.prior[active] ** config.prior_power
            transformed_total = float(transformed.sum())
            candidate_probability[active] = (
                (1.0 - config.null_prior) * transformed / transformed_total
            )
            null_probability[group_index] = config.null_prior
        # An all-zero group, or null_prior=1, is exactly all-null.
        errors.append(
            abs(
                float(candidate_probability[indices].sum())
                + float(null_probability[group_index])
                - 1.0
            )
        )
    candidate_log_probability = np.full(len(edges.source), -np.inf, dtype=np.float64)
    positive_candidate = candidate_probability > 0.0
    candidate_log_probability[positive_candidate] = np.log(
        candidate_probability[positive_candidate]
    )
    null_log_probability = np.log(null_probability)
    maximum_error = float(max(errors, default=0.0))
    if maximum_error > 1e-12:
        raise RuntimeError("categorical group priors are not normalized")
    return _CategoricalPriors(
        candidate_probability=candidate_probability,
        candidate_log_probability=candidate_log_probability,
        null_probability=null_probability,
        null_log_probability=null_log_probability,
        max_normalization_error=maximum_error,
    )


def _solve_laplacian(
    edges: _PreparedEdges,
    edge_weight: np.ndarray,
    initial_positions: np.ndarray,
    current_anchor: np.ndarray,
    config: GroupSwitchConfig,
) -> np.ndarray:
    count = len(initial_positions)
    rows = np.concatenate(
        [edges.source, edges.destination, edges.source, edges.destination]
    )
    columns = np.concatenate(
        [edges.source, edges.destination, edges.destination, edges.source]
    )
    data = np.concatenate([edge_weight, edge_weight, -edge_weight, -edge_weight])
    matrix = coo_matrix((data, (rows, columns)), shape=(count, count)).tocsr()
    anchor = (
        config.initial_anchor_weight
        + config.current_anchor_weight
        + config.regularization
    )
    matrix = matrix + eye(count, format="csr") * anchor
    rhs = np.zeros((count, 2), dtype=np.float64)
    weighted_offset = edge_weight[:, None] * edges.offsets
    np.add.at(rhs, edges.source, -weighted_offset)
    np.add.at(rhs, edges.destination, weighted_offset)
    rhs += (config.initial_anchor_weight + config.regularization) * initial_positions
    rhs += config.current_anchor_weight * current_anchor
    output = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
    if output.shape != (count, 2) or not np.all(np.isfinite(output)):
        raise RuntimeError("group-switch Laplacian produced invalid coordinates")
    return output


def _recenter(positions: np.ndarray, size: int) -> np.ndarray:
    center = (size - 1.0) / 2.0
    return positions + np.asarray([center, center]) - np.mean(positions, axis=0)


def _posterior(
    residual_squared: np.ndarray,
    edges: _PreparedEdges,
    priors: _CategoricalPriors,
    temperature: float,
    config: GroupSwitchConfig,
) -> tuple[np.ndarray, np.ndarray]:
    edge_mass = np.zeros(len(edges.source), dtype=np.float64)
    null_mass = np.empty(len(edges.groups), dtype=np.float64)
    for group_index, indices in enumerate(edges.groups):
        candidate_logits = priors.candidate_log_probability[indices] - (
            residual_squared[indices] / temperature
        )
        null_logit = (
            priors.null_log_probability[group_index]
            - config.null_cost / temperature
        )
        maximum = max(float(np.max(candidate_logits)), float(null_logit))
        candidate_exp = np.exp(candidate_logits - maximum)
        null_exp = float(np.exp(null_logit - maximum))
        denominator = float(candidate_exp.sum()) + null_exp
        edge_mass[indices] = candidate_exp / denominator
        null_mass[group_index] = null_exp / denominator
    if not (
        np.all(np.isfinite(edge_mass))
        and np.all(np.isfinite(null_mass))
        and np.all(edge_mass >= 0.0)
        and np.all(null_mass >= 0.0)
    ):
        raise RuntimeError("group-switch posterior is invalid")
    return edge_mass, null_mass


def _initial_switches(
    edges: _PreparedEdges,
    priors: _CategoricalPriors,
    restart: int,
    rng: np.random.Generator,
    config: GroupSwitchConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Choose a direction-wise injective hard candidate/null initialization.

    Independent group argmax can select the same destination for many query
    tiles, creating a collapsed and scientifically unhelpful Laplacian start.
    For each exact offset direction, a rectangular assignment instead gives
    every destination tile capacity one and every row its own private null.
    """

    edge_mass = np.zeros(len(edges.source), dtype=np.float64)
    selected = np.full(len(edges.groups), -1, dtype=np.int64)
    direction_lookup: dict[tuple[float, float], list[int]] = {}
    for group_index, (_source, dx, dy) in enumerate(edges.group_keys):
        direction_lookup.setdefault((dx, dy), []).append(group_index)

    destination_collisions = 0
    forbidden_assignments = 0
    for direction in sorted(direction_lookup):
        group_indices = direction_lookup[direction]
        row_count = len(group_indices)
        tile_count = config.grid_size**2
        column_count = tile_count + row_count
        allowed = np.zeros((row_count, column_count), dtype=bool)
        allowed_cost = np.zeros((row_count, column_count), dtype=np.float64)

        for row, group_index in enumerate(group_indices):
            indices = edges.groups[group_index]
            positive = priors.candidate_probability[indices] > 0.0
            active = indices[positive]
            candidate_utility = priors.candidate_log_probability[active].copy()
            null_utility = float(priors.null_log_probability[group_index])
            if restart > 0 and config.gumbel_scale > 0.0:
                uniforms = np.clip(
                    rng.random(len(active) + 1), config.prior_epsilon, 1.0
                )
                gumbel = -np.log(-np.log(uniforms))
                candidate_utility += config.gumbel_scale * gumbel[:-1]
                null_utility += config.gumbel_scale * float(gumbel[-1])
            if len(active):
                columns = edges.destination[active]
                allowed[row, columns] = True
                allowed_cost[row, columns] = -candidate_utility
            private_null = tile_count + row
            allowed[row, private_null] = True
            allowed_cost[row, private_null] = -null_utility

        finite_allowed = allowed_cost[allowed]
        minimum_allowed = float(np.min(finite_allowed))
        maximum_allowed = float(np.max(finite_allowed))
        # Any assignment containing even one unavailable cell costs more than
        # the most expensive complete all-allowed assignment, including when
        # perturbed utilities make some allowed costs negative.
        forbidden = (
            row_count * maximum_allowed
            - (row_count - 1) * minimum_allowed
            + 1.0
        )
        cost = np.full((row_count, column_count), forbidden, dtype=np.float64)
        cost[allowed] = allowed_cost[allowed]
        rows, columns = linear_sum_assignment(cost)
        if not np.array_equal(rows, np.arange(row_count)):
            raise RuntimeError("direction-wise initialization omitted a group")
        unavailable = ~allowed[rows, columns]
        forbidden_assignments += int(np.sum(unavailable))
        if np.any(unavailable):
            raise RuntimeError("direction-wise initialization selected a forbidden cell")
        real = columns < tile_count
        real_destinations = columns[real]
        destination_collisions += int(
            len(real_destinations) - len(np.unique(real_destinations))
        )
        for row, column in zip(rows[real].tolist(), columns[real].tolist(), strict=True):
            group_index = group_indices[row]
            indices = edges.groups[group_index]
            matches = indices[edges.destination[indices] == column]
            if len(matches) != 1:
                raise RuntimeError("assigned destination is not a unique group candidate")
            edge_index = int(matches[0])
            if priors.candidate_probability[edge_index] <= 0.0:
                raise RuntimeError("zero-prior candidate entered initialization")
            edge_mass[edge_index] = 1.0
            selected[group_index] = edge_index

    return edge_mass, selected, {
        "direction_count": len(direction_lookup),
        "destination_collisions": destination_collisions,
        "forbidden_assignments": forbidden_assignments,
    }


def _project(
    positions: np.ndarray,
    initial_tile_to_cell: np.ndarray,
    current_tile_to_cell: np.ndarray,
    config: GroupSwitchConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    size = config.grid_size
    count = size**2
    cells = np.column_stack([np.arange(count) % size, np.arange(count) // size]).astype(
        np.float64
    )
    distance = np.sum((positions[:, None] - cells[None]) ** 2, axis=2)
    allowed = distance <= config.max_candidate_radius**2
    nearest = np.argsort(distance, axis=1, kind="stable")[
        :, : min(config.max_candidates_per_tile, count)
    ]
    allowed[np.arange(count)[:, None], nearest] = True
    allowed[np.arange(count), initial_tile_to_cell] = True
    allowed[np.arange(count), current_tile_to_cell] = True
    maximum_allowed = float(np.max(distance[allowed]))
    forbidden = (count + 1.0) * (
        maximum_allowed + config.assignment_jitter + 1.0
    )
    cost = distance.copy()
    cost[~allowed] = forbidden
    if config.assignment_jitter > 0.0:
        jitter = rng.uniform(0.0, config.assignment_jitter, size=cost.shape)
        cost[allowed] += jitter[allowed]
    rows, columns = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(count)):
        raise RuntimeError("Hungarian projection omitted a tile")
    tile_to_cell = columns.astype(np.int32)
    if not np.array_equal(np.sort(tile_to_cell), np.arange(count)):
        raise RuntimeError("Hungarian projection is not a permutation")
    outside = int(np.sum(~allowed[np.arange(count), tile_to_cell]))
    if outside:
        raise RuntimeError("Hungarian projection selected a forbidden cell")
    nearest_cells = np.argmin(distance, axis=1)
    collisions = int(count - len(np.unique(nearest_cells)))
    grid = np.empty(count, dtype=np.int32)
    grid[tile_to_cell] = np.arange(count, dtype=np.int32)
    projection_distance = float(distance[np.arange(count), tile_to_cell].sum())
    return tile_to_cell, grid.reshape(size, size), projection_distance, collisions, outside


def _group_objective(
    tile_to_cell: np.ndarray,
    edges: _PreparedEdges,
    priors: _CategoricalPriors,
    config: GroupSwitchConfig,
) -> float:
    size = config.grid_size
    positions = np.column_stack([tile_to_cell % size, tile_to_cell // size]).astype(
        np.float64
    )
    residual = _residual_squared(positions, edges)
    values = []
    for group_index, indices in enumerate(edges.groups):
        positive = priors.candidate_probability[indices] > 0.0
        if np.any(positive):
            active_indices = indices[positive]
            prior_penalty = config.selection_prior_penalty * (
                -priors.candidate_log_probability[active_indices]
            )
            candidate = float(np.min(residual[active_indices] + prior_penalty))
        else:
            candidate = np.inf
        null_energy = config.null_cost + config.selection_prior_penalty * (
            -priors.null_log_probability[group_index]
        )
        values.append(min(float(null_energy), candidate))
    return float(np.mean(values))


def solve_group_switch(
    edges_src: np.ndarray,
    edges_dst: np.ndarray,
    offsets: np.ndarray,
    confidence: np.ndarray,
    initial_grid: np.ndarray,
    config: GroupSwitchConfig = GroupSwitchConfig(),
    seed: int = 0,
) -> GroupSwitchResult:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    edges, grid = _validate_and_prepare(
        edges_src, edges_dst, offsets, confidence, initial_grid, config
    )
    priors = _categorical_priors(edges, config)
    initial_tile_to_cell, initial_positions = _tile_positions(grid)
    temperatures = (
        np.asarray([config.temperature_final], dtype=np.float64)
        if config.stages == 1
        else np.geomspace(
            config.temperature_initial, config.temperature_final, config.stages
        )
    )
    best_score = _group_objective(initial_tile_to_cell, edges, priors, config)
    best_tile_to_cell = initial_tile_to_cell.copy()
    best_grid = grid.copy()
    best_continuous = initial_positions.copy()
    best_restart = -1
    best_stage = -1
    stage_diagnostics: list[dict[str, Any]] = []
    restart_diagnostics: list[dict[str, Any]] = []
    restart_hashes: list[str] = []
    sequences = np.random.SeedSequence(int(seed)).spawn(config.restarts)

    for restart, sequence in enumerate(sequences):
        rng = np.random.default_rng(sequence)
        posterior, selected, initialization = _initial_switches(
            edges, priors, restart, rng, config
        )
        switch_hash = hashlib.sha256(selected.tobytes()).hexdigest()
        restart_hashes.append(switch_hash)
        continuous = _solve_laplacian(
            edges, posterior, initial_positions, initial_positions, config
        )
        continuous = _recenter(continuous, config.grid_size)
        if restart > 0 and config.start_perturbation > 0.0:
            continuous += rng.normal(
                0.0, config.start_perturbation, size=continuous.shape
            )
            continuous = _recenter(continuous, config.grid_size)
        current_tile_to_cell = initial_tile_to_cell.copy()
        current_anchor = initial_positions.copy()
        restart_diagnostics.append(
            {
                "restart": restart,
                "initial_switch_method": "per_direction_one_to_one_assignment",
                "initial_switch_sha256": switch_hash,
                "initial_candidate_switches": int(np.sum(selected >= 0)),
                "initial_null_switches": int(np.sum(selected < 0)),
                "initial_direction_count": initialization["direction_count"],
                "initial_direction_destination_collisions": initialization[
                    "destination_collisions"
                ],
                "initial_forbidden_assignments": initialization[
                    "forbidden_assignments"
                ],
            }
        )

        for stage, temperature in enumerate(temperatures.tolist()):
            for _ in range(config.iterations_per_stage):
                residual = _residual_squared(continuous, edges)
                posterior, null_mass = _posterior(
                    residual, edges, priors, temperature, config
                )
                continuous = _solve_laplacian(
                    edges, posterior, initial_positions, current_anchor, config
                )
                continuous = _recenter(continuous, config.grid_size)
            residual = _residual_squared(continuous, edges)
            posterior, null_mass = _posterior(
                residual, edges, priors, temperature, config
            )
            group_candidate_mass = np.asarray(
                [posterior[indices].sum() for indices in edges.groups],
                dtype=np.float64,
            )
            if np.any(group_candidate_mass > 1.0 + 1e-12):
                raise RuntimeError("candidate posterior mass exceeded one")
            tile_to_cell, projected_grid, projection_distance, collisions, outside = _project(
                continuous,
                initial_tile_to_cell,
                current_tile_to_cell,
                config,
                rng,
            )
            score = _group_objective(tile_to_cell, edges, priors, config)
            current_tile_to_cell = tile_to_cell
            current_anchor = np.column_stack(
                [tile_to_cell % config.grid_size, tile_to_cell // config.grid_size]
            ).astype(np.float64)
            stage_diagnostics.append(
                {
                    "restart": restart,
                    "stage": stage,
                    "temperature": float(temperature),
                    "group_objective": score,
                    "candidate_mass_mean": float(np.mean(group_candidate_mass)),
                    "candidate_mass_min": float(np.min(group_candidate_mass)),
                    "candidate_mass_max": float(np.max(group_candidate_mass)),
                    "null_mass_mean": float(np.mean(null_mass)),
                    "null_mass_min": float(np.min(null_mass)),
                    "null_mass_max": float(np.max(null_mass)),
                    "posterior_normalization_max_error": float(
                        np.max(np.abs(group_candidate_mass + null_mass - 1.0))
                    ),
                    "posterior_active_edges": int(np.sum(posterior > 1e-8)),
                    "continuous_nearest_cell_collisions": collisions,
                    "projection_squared_distance": projection_distance,
                    "outside_candidate_assignments": outside,
                }
            )
            if score < best_score - 1e-12:
                best_score = score
                best_tile_to_cell = tile_to_cell.copy()
                best_grid = projected_grid.copy()
                best_continuous = continuous.copy()
                best_restart = restart
                best_stage = stage

    if not (
        np.array_equal(np.sort(best_tile_to_cell), np.arange(config.grid_size**2))
        and np.array_equal(np.sort(best_grid.ravel()), np.arange(config.grid_size**2))
        and np.all(np.isfinite(best_continuous))
        and np.isfinite(best_score)
    ):
        raise RuntimeError("group-switch result failed integrity validation")
    diagnostics = {
        "seed": int(seed),
        "tile_count": int(config.grid_size**2),
        "raw_edge_count": int(edges.raw_edge_count),
        "deduplicated_edge_count": int(len(edges.source)),
        "group_count": int(len(edges.groups)),
        "zero_prior_group_count": int(edges.zero_prior_groups),
        "categorical_prior_normalization_max_error": priors.max_normalization_error,
        "temperature_schedule": [float(value) for value in temperatures],
        "initial_group_objective": float(
            _group_objective(initial_tile_to_cell, edges, priors, config)
        ),
        "best_group_objective": float(best_score),
        "best_restart": best_restart,
        "best_stage": best_stage,
        "initial_grid_selected": best_restart == -1,
        "initial_switch_method": "per_direction_one_to_one_assignment",
        "selection_policy": "mean_group_min_of_null_or_prior_penalized_edge_residual",
        "restart_unique_initial_switches": len(set(restart_hashes)),
        "restart_diversity_fraction": float(len(set(restart_hashes)) / len(restart_hashes)),
        "posterior_normalization_max_error": float(
            max(
                (
                    record["posterior_normalization_max_error"]
                    for record in stage_diagnostics
                ),
                default=0.0,
            )
        ),
        "restarts": restart_diagnostics,
        "stages": stage_diagnostics,
    }
    return GroupSwitchResult(
        tile_to_cell=best_tile_to_cell,
        grid=best_grid,
        continuous_positions=best_continuous,
        diagnostics=diagnostics,
    )


__all__ = ["GroupSwitchConfig", "GroupSwitchResult", "solve_group_switch"]
