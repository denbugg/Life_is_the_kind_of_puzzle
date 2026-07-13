"""Incremental, seed-protected annealing for 24x24 tile layouts.

The promoted FAQ/QAP solver ends with a small deterministic tile-swap search.
This module explores a meaningfully larger neighbourhood while keeping the
same input-only pairwise objective:

* arbitrary tile swaps;
* within-row/within-column segment relocation (preserves internal runs);
* equal-shape small-block swaps; and
* whole-row/whole-column swaps.

Only grid edges incident to changed cells are rescored for a proposal.  An
optional input-only regularizer protects reciprocal, high-margin HBT edges
from the component seed.  The protection term is a penalty for breaking those
edges, not a target-derived label.  Fixed seeds and stable tie breaking make
the bounded search reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE_COUNT, validate_permutation
from .solvers import placement_unary


_POSITIONS = np.arange(TILE_COUNT, dtype=np.int32).reshape(GRID, GRID)
_H_SOURCE = _POSITIONS[:, :-1].ravel()
_H_TARGET = _POSITIONS[:, 1:].ravel()
_V_SOURCE = _POSITIONS[:-1, :].ravel()
_V_TARGET = _POSITIONS[1:, :].ravel()
_EDGE_SOURCE = np.concatenate([_H_SOURCE, _V_SOURCE]).astype(np.int32)
_EDGE_TARGET = np.concatenate([_H_TARGET, _V_TARGET]).astype(np.int32)
_EDGE_DIRECTION = np.concatenate(
    [
        np.zeros(len(_H_SOURCE), dtype=np.int8),
        np.ones(len(_V_SOURCE), dtype=np.int8),
    ]
)
_INCIDENT_EDGES = tuple(
    np.flatnonzero((_EDGE_SOURCE == position) | (_EDGE_TARGET == position)).astype(
        np.int32
    )
    for position in range(TILE_COUNT)
)


@dataclass(frozen=True)
class ProtectedEdges:
    """Confidence bonuses for directed right/down seed adjacencies."""

    right: np.ndarray
    down: np.ndarray
    count_right: int
    count_down: int
    total_confidence: float
    raw_margin_threshold: float

    @property
    def count(self) -> int:
        return int(self.count_right + self.count_down)


@dataclass(frozen=True)
class AnnealRefineResult:
    """Layout plus deterministic diagnostics for an annealing run."""

    position_to_slot: np.ndarray
    seed: int
    best_restart: int
    evaluations: int
    accepted: int
    improving_accepted: int
    proposed_by_move: dict[str, int]
    accepted_by_move: dict[str, int]
    polished_swaps: int
    base_objective_before: float
    base_objective_after: float
    augmented_energy_before: float
    augmented_energy_after: float
    edge_scale: float
    protection_weight: float
    protected_edge_count: int
    protected_confidence_total: float
    protected_confidence_before: float
    protected_confidence_after: float


@dataclass(frozen=True)
class _Move:
    candidate: np.ndarray
    changed_positions: np.ndarray
    kind: str


def _finite_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError(f"{name} must have shape {(TILE_COUNT, TILE_COUNT)}")
    off_diagonal = ~np.eye(TILE_COUNT, dtype=bool)
    if not np.all(np.isfinite(matrix[off_diagonal])):
        raise ValueError(f"{name} has non-finite off-diagonal entries")
    return matrix


def _edge_scale(compatibility: CompatibilityMatrices) -> float:
    off_diagonal = ~np.eye(TILE_COUNT, dtype=bool)
    finite = np.concatenate(
        [
            np.asarray(compatibility.right, dtype=np.float64)[off_diagonal],
            np.asarray(compatibility.down, dtype=np.float64)[off_diagonal],
        ]
    )
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    median = float(np.median(np.abs(finite)))
    return float(max(q75 - q25, 0.05 * median, 1e-6))


def _layout_edge_pairs(
    position_to_slot: np.ndarray,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    grid = position_to_slot.reshape(GRID, GRID)
    right = set(
        zip(grid[:, :-1].ravel().tolist(), grid[:, 1:].ravel().tolist(), strict=True)
    )
    down = set(
        zip(grid[:-1, :].ravel().tolist(), grid[1:, :].ravel().tolist(), strict=True)
    )
    return right, down


def _reciprocal_margin_candidates(
    values: np.ndarray,
    *,
    allowed: set[tuple[int, int]] | None,
    direction: int,
) -> list[tuple[float, int, int, int]]:
    matrix = _finite_matrix(values, name="seed compatibility")
    matrix = matrix.copy()
    np.fill_diagonal(matrix, np.inf)
    row_order = np.argsort(matrix, axis=1, kind="stable")[:, :2]
    column_order = np.argsort(matrix, axis=0, kind="stable")[:2, :]
    candidates: list[tuple[float, int, int, int]] = []
    for first in range(TILE_COUNT):
        second = int(row_order[first, 0])
        if int(column_order[0, second]) != first:
            continue
        if allowed is not None and (first, second) not in allowed:
            continue
        best = float(matrix[first, second])
        margin_out = float(matrix[first, int(row_order[first, 1])] - best)
        margin_in = float(matrix[int(column_order[1, second]), second] - best)
        margin = min(margin_out, margin_in)
        if np.isfinite(margin) and margin > 0.0:
            candidates.append((margin, direction, first, second))
    return candidates


def build_protected_edges(
    seed_compatibility: CompatibilityMatrices,
    *,
    protected_layout: np.ndarray | None = None,
    confidence_quantile: float = 0.75,
    max_edges: int = 384,
) -> ProtectedEdges:
    """Select reciprocal, high-margin HBT edges using inputs only.

    When ``protected_layout`` is provided, candidates must also be oriented
    adjacencies of that component seed.  This is the intended QAP-refinement
    use: the annealer may move fragments broadly, but it pays a calibrated
    penalty when it breaks the seed's strongest mutually supported joins.
    """

    if not 0.0 <= confidence_quantile <= 1.0:
        raise ValueError("confidence_quantile must lie in [0, 1]")
    if max_edges <= 0:
        raise ValueError("max_edges must be positive")
    allowed_right: set[tuple[int, int]] | None = None
    allowed_down: set[tuple[int, int]] | None = None
    if protected_layout is not None:
        layout = validate_permutation(
            protected_layout, name="protected_position_to_slot"
        )
        allowed_right, allowed_down = _layout_edge_pairs(layout)

    candidates = _reciprocal_margin_candidates(
        seed_compatibility.right,
        allowed=allowed_right,
        direction=0,
    )
    candidates.extend(
        _reciprocal_margin_candidates(
            seed_compatibility.down,
            allowed=allowed_down,
            direction=1,
        )
    )
    right = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    down = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    if not candidates:
        return ProtectedEdges(right, down, 0, 0, 0.0, float("inf"))

    margins = np.asarray([candidate[0] for candidate in candidates], dtype=np.float64)
    threshold = float(np.quantile(margins, confidence_quantile))
    selected = [candidate for candidate in candidates if candidate[0] >= threshold]
    selected.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    selected = selected[:max_edges]
    reference = max(float(np.quantile([item[0] for item in selected], 0.9)), 1e-12)
    count_right = 0
    count_down = 0
    for margin, direction, first, second in selected:
        confidence = float(np.clip(margin / reference, 0.05, 1.0))
        if direction == 0:
            right[first, second] = confidence
            count_right += 1
        else:
            down[first, second] = confidence
            count_down += 1
    total = float(right.sum(dtype=np.float64) + down.sum(dtype=np.float64))
    return ProtectedEdges(
        right=right,
        down=down,
        count_right=count_right,
        count_down=count_down,
        total_confidence=total,
        raw_margin_threshold=threshold,
    )


def _edge_sum(
    layout: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    edge_ids: np.ndarray | None = None,
) -> float:
    if edge_ids is None:
        edge_ids = np.arange(len(_EDGE_SOURCE), dtype=np.int32)
    source = _EDGE_SOURCE[edge_ids]
    target = _EDGE_TARGET[edge_ids]
    first = layout[source]
    second = layout[target]
    horizontal = _EDGE_DIRECTION[edge_ids] == 0
    value = np.empty(len(edge_ids), dtype=np.float64)
    value[horizontal] = right[first[horizontal], second[horizontal]]
    value[~horizontal] = down[first[~horizontal], second[~horizontal]]
    return float(value.sum(dtype=np.float64))


def _retained_confidence(layout: np.ndarray, protected: ProtectedEdges) -> float:
    return _edge_sum(
        layout,
        np.asarray(protected.right, dtype=np.float64),
        np.asarray(protected.down, dtype=np.float64),
    )


def layout_energy(
    position_to_slot: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    boundary_weight: float = 0.05,
    protected: ProtectedEdges | None = None,
    protection_weight: float = 0.0,
) -> float:
    """Return seam + boundary + broken-protected-edge energy."""

    if boundary_weight < 0.0 or protection_weight < 0.0:
        raise ValueError("energy weights must be non-negative")
    layout = validate_permutation(position_to_slot, name="position_to_slot")
    right = _finite_matrix(compatibility.right, name="compatibility.right")
    down = _finite_matrix(compatibility.down, name="compatibility.down")
    unary = placement_unary(compatibility).astype(np.float64, copy=False)
    value = _edge_sum(layout, right, down)
    value += boundary_weight * float(
        unary[np.arange(TILE_COUNT, dtype=np.int32), layout].sum(dtype=np.float64)
    )
    if protected is not None and protection_weight:
        retained = _retained_confidence(layout, protected)
        value += protection_weight * (protected.total_confidence - retained)
    return float(value)


def _affected_edges(changed_positions: np.ndarray) -> np.ndarray:
    return np.unique(
        np.concatenate([_INCIDENT_EDGES[int(position)] for position in changed_positions])
    ).astype(np.int32, copy=False)


def _incremental_delta(
    current: np.ndarray,
    candidate: np.ndarray,
    changed_positions: np.ndarray,
    *,
    augmented_right: np.ndarray,
    augmented_down: np.ndarray,
    unary: np.ndarray,
    boundary_weight: float,
) -> tuple[float, int]:
    edge_ids = _affected_edges(changed_positions)
    delta = _edge_sum(candidate, augmented_right, augmented_down, edge_ids)
    delta -= _edge_sum(current, augmented_right, augmented_down, edge_ids)
    old_unary = unary[changed_positions, current[changed_positions]].sum(dtype=np.float64)
    new_unary = unary[changed_positions, candidate[changed_positions]].sum(dtype=np.float64)
    delta += boundary_weight * float(new_unary - old_unary)
    return float(delta), int(len(edge_ids))


def _local_costs(
    layout: np.ndarray,
    *,
    right: np.ndarray,
    down: np.ndarray,
    unary: np.ndarray,
    boundary_weight: float,
) -> np.ndarray:
    costs = boundary_weight * unary[
        np.arange(TILE_COUNT, dtype=np.int32), layout
    ].astype(np.float64, copy=True)
    grid = layout.reshape(GRID, GRID)
    horizontal = right[grid[:, :-1], grid[:, 1:]]
    vertical = down[grid[:-1, :], grid[1:, :]]
    costs.reshape(GRID, GRID)[:, :-1] += 0.5 * horizontal
    costs.reshape(GRID, GRID)[:, 1:] += 0.5 * horizontal
    costs.reshape(GRID, GRID)[:-1, :] += 0.5 * vertical
    costs.reshape(GRID, GRID)[1:, :] += 0.5 * vertical
    return costs


def _changed_move(current: np.ndarray, candidate: np.ndarray, kind: str) -> _Move | None:
    changed = np.flatnonzero(candidate != current).astype(np.int32)
    if len(changed) < 2:
        return None
    return _Move(candidate=candidate, changed_positions=changed, kind=kind)


def _swap_move(
    current: np.ndarray,
    rng: np.random.Generator,
    weak_pool: np.ndarray,
    weak_bias: float,
) -> _Move:
    if rng.random() < weak_bias:
        first = int(weak_pool[int(rng.integers(len(weak_pool)))])
    else:
        first = int(rng.integers(TILE_COUNT))
    if rng.random() < 0.5 * weak_bias:
        candidates = weak_pool[weak_pool != first]
        second = int(candidates[int(rng.integers(len(candidates)))])
    else:
        second = int(rng.integers(TILE_COUNT - 1))
        if second >= first:
            second += 1
    candidate = current.copy()
    candidate[first], candidate[second] = candidate[second], candidate[first]
    return _Move(
        candidate=candidate,
        changed_positions=np.asarray(sorted((first, second)), dtype=np.int32),
        kind="swap",
    )


def _segment_move(
    current: np.ndarray,
    rng: np.random.Generator,
    weak_pool: np.ndarray,
    weak_bias: float,
    max_segment: int,
) -> _Move | None:
    horizontal = bool(rng.integers(2) == 0)
    if rng.random() < weak_bias:
        anchor = int(weak_pool[int(rng.integers(len(weak_pool)))])
        line_index = anchor // GRID if horizontal else anchor % GRID
    else:
        line_index = int(rng.integers(GRID))
    length = int(rng.integers(2, max_segment + 1))
    start = int(rng.integers(GRID - length + 1))
    insertion_options = np.asarray(
        [value for value in range(GRID - length + 1) if value != start],
        dtype=np.int32,
    )
    insertion = int(insertion_options[int(rng.integers(len(insertion_options)))])
    grid = current.reshape(GRID, GRID)
    line = grid[line_index].copy() if horizontal else grid[:, line_index].copy()
    segment = line[start : start + length]
    remaining = np.concatenate([line[:start], line[start + length :]])
    reordered = np.concatenate(
        [remaining[:insertion], segment, remaining[insertion:]]
    )
    candidate_grid = grid.copy()
    if horizontal:
        candidate_grid[line_index] = reordered
    else:
        candidate_grid[:, line_index] = reordered
    return _changed_move(current, candidate_grid.ravel(), "segment")


def _block_move(
    current: np.ndarray,
    rng: np.random.Generator,
    weak_pool: np.ndarray,
    weak_bias: float,
    block_shapes: tuple[tuple[int, int], ...],
) -> _Move | None:
    height, width = block_shapes[int(rng.integers(len(block_shapes)))]
    for _ in range(12):
        if rng.random() < weak_bias:
            anchor = int(weak_pool[int(rng.integers(len(weak_pool)))])
            first_row = min(anchor // GRID, GRID - height)
            first_column = min(anchor % GRID, GRID - width)
        else:
            first_row = int(rng.integers(GRID - height + 1))
            first_column = int(rng.integers(GRID - width + 1))
        second_row = int(rng.integers(GRID - height + 1))
        second_column = int(rng.integers(GRID - width + 1))
        rows_overlap = not (
            first_row + height <= second_row or second_row + height <= first_row
        )
        columns_overlap = not (
            first_column + width <= second_column
            or second_column + width <= first_column
        )
        if rows_overlap and columns_overlap:
            continue
        grid = current.reshape(GRID, GRID).copy()
        first = grid[
            first_row : first_row + height,
            first_column : first_column + width,
        ].copy()
        second = grid[
            second_row : second_row + height,
            second_column : second_column + width,
        ].copy()
        grid[
            first_row : first_row + height,
            first_column : first_column + width,
        ] = second
        grid[
            second_row : second_row + height,
            second_column : second_column + width,
        ] = first
        return _changed_move(current, grid.ravel(), "block")
    return None


def _band_move(
    current: np.ndarray,
    rng: np.random.Generator,
    weak_pool: np.ndarray,
    weak_bias: float,
) -> _Move:
    rows = bool(rng.integers(2) == 0)
    if rng.random() < weak_bias:
        anchor = int(weak_pool[int(rng.integers(len(weak_pool)))])
        first = anchor // GRID if rows else anchor % GRID
    else:
        first = int(rng.integers(GRID))
    second = int(rng.integers(GRID - 1))
    if second >= first:
        second += 1
    grid = current.reshape(GRID, GRID).copy()
    if rows:
        grid[[first, second]] = grid[[second, first]]
    else:
        grid[:, [first, second]] = grid[:, [second, first]]
    return _changed_move(current, grid.ravel(), "band")  # type: ignore[return-value]


def _choose_move(
    current: np.ndarray,
    rng: np.random.Generator,
    weak_pool: np.ndarray,
    *,
    weak_bias: float,
    max_segment: int,
    block_shapes: tuple[tuple[int, int], ...],
    move_names: tuple[str, ...],
    cumulative_weights: np.ndarray,
) -> _Move:
    selected = move_names[
        int(np.searchsorted(cumulative_weights, rng.random(), side="right"))
    ]
    move: _Move | None
    if selected == "segment":
        move = _segment_move(current, rng, weak_pool, weak_bias, max_segment)
    elif selected == "block":
        move = _block_move(current, rng, weak_pool, weak_bias, block_shapes)
    elif selected == "band":
        move = _band_move(current, rng, weak_pool, weak_bias)
    else:
        move = _swap_move(current, rng, weak_pool, weak_bias)
    return move if move is not None else _swap_move(current, rng, weak_pool, weak_bias)


def _calibrate_temperature(
    layout: np.ndarray,
    rng: np.random.Generator,
    *,
    right: np.ndarray,
    down: np.ndarray,
    unary: np.ndarray,
    boundary_weight: float,
    edge_scale: float,
    samples: int,
) -> float:
    weak = np.arange(TILE_COUNT, dtype=np.int32)
    positive: list[float] = []
    for _ in range(samples):
        move = _swap_move(layout, rng, weak, 0.0)
        delta, _ = _incremental_delta(
            layout,
            move.candidate,
            move.changed_positions,
            augmented_right=right,
            augmented_down=down,
            unary=unary,
            boundary_weight=boundary_weight,
        )
        if delta > 0.0 and np.isfinite(delta):
            positive.append(delta)
    if not positive:
        return float(edge_scale)
    # Median worsening has 50% acceptance at the first temperature.
    return float(max(np.median(positive) / np.log(2.0), 1e-8))


def _polish_swaps(
    layout: np.ndarray,
    *,
    right: np.ndarray,
    down: np.ndarray,
    unary: np.ndarray,
    boundary_weight: float,
    weak_cells: int,
    moves: int,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, int]:
    current = layout.copy()
    accepted = 0
    for _ in range(moves):
        local = _local_costs(
            current,
            right=right,
            down=down,
            unary=unary,
            boundary_weight=boundary_weight,
        )
        weak = np.argsort(-local, kind="stable")[: min(weak_cells, TILE_COUNT)]
        best_delta = 0.0
        best_pair: tuple[int, int] | None = None
        for index, first_value in enumerate(weak[:-1]):
            first = int(first_value)
            for second_value in weak[index + 1 :]:
                second = int(second_value)
                candidate = current.copy()
                candidate[first], candidate[second] = candidate[second], candidate[first]
                changed = np.asarray((first, second), dtype=np.int32)
                delta, _ = _incremental_delta(
                    current,
                    candidate,
                    changed,
                    augmented_right=right,
                    augmented_down=down,
                    unary=unary,
                    boundary_weight=boundary_weight,
                )
                pair = (min(first, second), max(first, second))
                if delta < best_delta - tolerance or (
                    abs(delta - best_delta) <= tolerance
                    and best_pair is not None
                    and pair < best_pair
                ):
                    best_delta = delta
                    best_pair = pair
        if best_pair is None:
            break
        first, second = best_pair
        current[first], current[second] = current[second], current[first]
        accepted += 1
    return current, accepted


def anneal_refine(
    position_to_slot: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    seed: int,
    seed_compatibility: CompatibilityMatrices | None = None,
    protected_layout: np.ndarray | None = None,
    evaluations_per_restart: int = 10_000,
    restarts: int = 2,
    boundary_weight: float = 0.05,
    protection_strength: float = 0.15,
    confidence_quantile: float = 0.75,
    max_protected_edges: int = 384,
    weak_bias: float = 0.70,
    weak_pool_size: int = 64,
    weak_refresh: int = 64,
    max_segment: int = 8,
    block_shapes: tuple[tuple[int, int], ...] = (
        (1, 2),
        (2, 1),
        (1, 4),
        (4, 1),
        (2, 2),
        (3, 3),
    ),
    move_weights: Mapping[str, float] | None = None,
    calibration_samples: int = 64,
    final_temperature_ratio: float = 0.005,
    polish_moves: int = 6,
    polish_weak_cells: int = 40,
    audit_interval: int = 512,
) -> AnnealRefineResult:
    """Run deterministic multi-move annealing with incremental energy deltas.

    ``evaluations_per_restart`` is a hard proposal budget.  Each restart begins
    from the supplied layout, making the comparison with QAP paired.  The
    returned layout is the best *augmented* state seen across restarts followed
    by a bounded deterministic swap polish.
    """

    if evaluations_per_restart < 0 or restarts <= 0:
        raise ValueError("invalid annealing budget or restart count")
    if boundary_weight < 0.0 or protection_strength < 0.0:
        raise ValueError("objective weights must be non-negative")
    if not 0.0 <= weak_bias <= 1.0:
        raise ValueError("weak_bias must lie in [0, 1]")
    if not 2 <= weak_pool_size <= TILE_COUNT:
        raise ValueError("weak_pool_size must lie in [2, 576]")
    if weak_refresh <= 0 or audit_interval <= 0:
        raise ValueError("refresh/audit intervals must be positive")
    if not 2 <= max_segment <= GRID:
        raise ValueError("max_segment must lie in [2, 24]")
    if calibration_samples <= 0:
        raise ValueError("calibration_samples must be positive")
    if not 0.0 < final_temperature_ratio <= 1.0:
        raise ValueError("final_temperature_ratio must lie in (0, 1]")
    if polish_moves < 0 or not 2 <= polish_weak_cells <= TILE_COUNT:
        raise ValueError("invalid polish settings")
    if not block_shapes or any(
        height <= 0 or width <= 0 or height > GRID or width > GRID
        for height, width in block_shapes
    ):
        raise ValueError("block_shapes must contain valid positive grid shapes")

    weights = dict(
        move_weights
        if move_weights is not None
        else {"swap": 0.42, "segment": 0.25, "block": 0.23, "band": 0.10}
    )
    if set(weights) != {"swap", "segment", "block", "band"}:
        raise ValueError("move_weights must define swap, segment, block, and band")
    if any(not np.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("move weights must be finite and non-negative")
    total_weight = float(sum(weights.values()))
    if total_weight <= 0.0:
        raise ValueError("at least one move weight must be positive")
    move_names = tuple(weights)
    cumulative = np.cumsum([weights[name] / total_weight for name in move_names])
    cumulative[-1] = 1.0

    initial = validate_permutation(position_to_slot, name="position_to_slot").copy()
    if protected_layout is not None:
        protected_layout = validate_permutation(
            protected_layout, name="protected_position_to_slot"
        ).copy()
    seed_scores = seed_compatibility if seed_compatibility is not None else compatibility
    protected = build_protected_edges(
        seed_scores,
        protected_layout=protected_layout,
        confidence_quantile=confidence_quantile,
        max_edges=max_protected_edges,
    )
    right = _finite_matrix(compatibility.right, name="compatibility.right")
    down = _finite_matrix(compatibility.down, name="compatibility.down")
    unary = placement_unary(compatibility).astype(np.float64, copy=False)
    scale = _edge_scale(compatibility)
    protection_weight = float(protection_strength * scale)
    augmented_right = right - protection_weight * protected.right
    augmented_down = down - protection_weight * protected.down
    constant = protection_weight * protected.total_confidence

    def augmented_value(layout: np.ndarray) -> float:
        return float(
            _edge_sum(layout, augmented_right, augmented_down)
            + boundary_weight
            * unary[np.arange(TILE_COUNT, dtype=np.int32), layout].sum(dtype=np.float64)
            + constant
        )

    initial_augmented = augmented_value(initial)
    best = initial.copy()
    best_value = initial_augmented
    best_restart = -1
    total_accepted = 0
    total_improving = 0
    proposed_by_move = {name: 0 for name in move_names}
    accepted_by_move = {name: 0 for name in move_names}

    for restart in range(restarts):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(restart), 0xA11EA1])
        )
        current = initial.copy()
        current_value = initial_augmented
        start_temperature = _calibrate_temperature(
            current,
            rng,
            right=augmented_right,
            down=augmented_down,
            unary=unary,
            boundary_weight=boundary_weight,
            edge_scale=scale,
            samples=calibration_samples,
        )
        weak_pool = np.arange(weak_pool_size, dtype=np.int32)
        for evaluation in range(evaluations_per_restart):
            if evaluation % weak_refresh == 0:
                local = _local_costs(
                    current,
                    right=augmented_right,
                    down=augmented_down,
                    unary=unary,
                    boundary_weight=boundary_weight,
                )
                weak_pool = np.argsort(-local, kind="stable")[:weak_pool_size]
            move = _choose_move(
                current,
                rng,
                weak_pool,
                weak_bias=weak_bias,
                max_segment=max_segment,
                block_shapes=block_shapes,
                move_names=move_names,
                cumulative_weights=cumulative,
            )
            proposed_by_move[move.kind] += 1
            delta, affected_edges = _incremental_delta(
                current,
                move.candidate,
                move.changed_positions,
                augmented_right=augmented_right,
                augmented_down=augmented_down,
                unary=unary,
                boundary_weight=boundary_weight,
            )
            progress = evaluation / max(evaluations_per_restart - 1, 1)
            temperature = start_temperature * final_temperature_ratio**progress
            # Larger segment/band moves touch more boundary edges.  A capped
            # square-root correction gives them a nonzero chance without
            # allowing whole-row swaps to dominate the Markov chain.
            move_scale = min(4.0, max(1.0, np.sqrt(affected_edges / 8.0)))
            accept = delta <= 0.0 or rng.random() < np.exp(
                -delta / max(temperature * move_scale, 1e-12)
            )
            if accept:
                current = move.candidate
                current_value += delta
                total_accepted += 1
                accepted_by_move[move.kind] += 1
                if delta < 0.0:
                    total_improving += 1
                tolerance = 1e-12 * max(1.0, abs(best_value))
                better = current_value < best_value - tolerance
                if (
                    not better
                    and abs(current_value - best_value) <= tolerance
                    and tuple(current.tolist()) < tuple(best.tolist())
                ):
                    better = True
                if better:
                    best = current.copy()
                    best_value = float(current_value)
                    best_restart = restart
            if (evaluation + 1) % audit_interval == 0:
                audited = augmented_value(current)
                tolerance = 1e-8 * max(1.0, abs(audited))
                if abs(audited - current_value) > tolerance:
                    raise RuntimeError("incremental annealing energy drift detected")
                current_value = audited

    polished, polished_swaps = _polish_swaps(
        best,
        right=augmented_right,
        down=augmented_down,
        unary=unary,
        boundary_weight=boundary_weight,
        weak_cells=polish_weak_cells,
        moves=polish_moves,
    )
    polished_value = augmented_value(polished)
    if polished_value <= best_value + 1e-8 * max(1.0, abs(best_value)):
        best = polished
        best_value = polished_value
    else:
        raise RuntimeError("deterministic polish increased augmented energy")
    best = validate_permutation(best, name="annealed_position_to_slot")

    before_base = layout_energy(
        initial, compatibility, boundary_weight=boundary_weight
    )
    after_base = layout_energy(best, compatibility, boundary_weight=boundary_weight)
    retained_before = _retained_confidence(initial, protected)
    retained_after = _retained_confidence(best, protected)
    exact_augmented = layout_energy(
        best,
        compatibility,
        boundary_weight=boundary_weight,
        protected=protected,
        protection_weight=protection_weight,
    )
    if abs(exact_augmented - best_value) > 1e-8 * max(1.0, abs(best_value)):
        raise RuntimeError("final augmented energy verification failed")
    return AnnealRefineResult(
        position_to_slot=best.copy(),
        seed=int(seed),
        best_restart=int(best_restart),
        evaluations=int(evaluations_per_restart * restarts),
        accepted=int(total_accepted),
        improving_accepted=int(total_improving),
        proposed_by_move=proposed_by_move,
        accepted_by_move=accepted_by_move,
        polished_swaps=int(polished_swaps),
        base_objective_before=float(before_base),
        base_objective_after=float(after_base),
        augmented_energy_before=float(initial_augmented),
        augmented_energy_after=float(best_value),
        edge_scale=float(scale),
        protection_weight=float(protection_weight),
        protected_edge_count=int(protected.count),
        protected_confidence_total=float(protected.total_confidence),
        protected_confidence_before=float(retained_before),
        protected_confidence_after=float(retained_after),
    )


__all__ = [
    "AnnealRefineResult",
    "ProtectedEdges",
    "anneal_refine",
    "build_protected_edges",
    "layout_energy",
]
