"""Deterministic valid-permutation assembly baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from .compatibility import CompatibilityMatrices, rank_normalize
from .geometry import GRID, TILE_COUNT, validate_permutation


def identity_layout() -> np.ndarray:
    return np.arange(TILE_COUNT, dtype=np.int32)


def random_layout(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(TILE_COUNT).astype(np.int32)


def _unit_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int32)
    ranks[order] = np.arange(len(values), dtype=np.int32)
    return ranks.astype(np.float32) / float(max(len(values) - 1, 1))


def outside_evidence(compatibility: CompatibilityMatrices) -> np.ndarray:
    """Return heuristic p(outside) for left/right/up/down sides of each tile."""
    left = _unit_ranks(np.min(compatibility.right, axis=0))
    right = _unit_ranks(np.min(compatibility.right, axis=1))
    up = _unit_ranks(np.min(compatibility.down, axis=0))
    down = _unit_ranks(np.min(compatibility.down, axis=1))
    return np.stack([left, right, up, down], axis=1).astype(np.float32)


def placement_unary(compatibility: CompatibilityMatrices) -> np.ndarray:
    """Return position x slot boundary cost in [0, 1]."""
    outside = outside_evidence(compatibility)
    unary = np.empty((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    for position in range(TILE_COUNT):
        row, column = divmod(position, GRID)
        outer = np.asarray(
            [column == 0, column == GRID - 1, row == 0, row == GRID - 1], dtype=bool
        )
        side_cost = np.where(outer[None, :], 1.0 - outside, outside)
        unary[position] = side_cost.mean(axis=1)
    return unary


def outside_logits_placement_unary(outside_logits: np.ndarray) -> np.ndarray:
    """Convert learned left/right/up/down outside logits into position costs."""
    logits = np.asarray(outside_logits, dtype=np.float32)
    if logits.shape != (TILE_COUNT, 4) or not np.all(np.isfinite(logits)):
        raise ValueError("outside_logits must be a finite 576x4 array")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    probabilities = np.clip(probabilities, 1e-4, 1.0 - 1e-4)
    unary = np.empty((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    for position in range(TILE_COUNT):
        row, column = divmod(position, GRID)
        outer = np.asarray(
            [column == 0, column == GRID - 1, row == 0, row == GRID - 1], dtype=bool
        )
        side_cost = np.where(
            outer[None, :], -np.log(probabilities), -np.log1p(-probabilities)
        )
        unary[position] = side_cost.mean(axis=1)
    return unary


def position_logits_placement_unary(
    row_logits: np.ndarray, column_logits: np.ndarray
) -> np.ndarray:
    """Convert per-slot 24-way row/column logits into position x slot costs."""
    row_values = np.asarray(row_logits, dtype=np.float64)
    column_values = np.asarray(column_logits, dtype=np.float64)
    if (
        row_values.shape != (TILE_COUNT, GRID)
        or column_values.shape != (TILE_COUNT, GRID)
        or not np.all(np.isfinite(row_values))
        or not np.all(np.isfinite(column_values))
    ):
        raise ValueError("row and column logits must be finite 576x24 arrays")

    def negative_log_probabilities(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max(axis=1, keepdims=True)
        log_normalizer = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return -(shifted - log_normalizer)

    row_cost = negative_log_probabilities(row_values)
    column_cost = negative_log_probabilities(column_values)
    unary = np.empty((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    for position in range(TILE_COUNT):
        row, column = divmod(position, GRID)
        unary[position] = 0.5 * (row_cost[:, row] + column_cost[:, column])
    return unary


def _candidate_increment(
    position: int,
    order: np.ndarray,
    candidates: np.ndarray,
    compatibility: CompatibilityMatrices,
    unary: np.ndarray,
    boundary_weight: float,
) -> np.ndarray:
    row, column = divmod(position, GRID)
    value = boundary_weight * unary[position, candidates]
    neighbour_count = 0
    if column > 0:
        left_slot = int(order[position - 1])
        value = value + compatibility.right[left_slot, candidates]
        neighbour_count += 1
    if row > 0:
        up_slot = int(order[position - GRID])
        value = value + compatibility.down[up_slot, candidates]
        neighbour_count += 1
    if neighbour_count > 1:
        value = value / float(neighbour_count)
    return value.astype(np.float32, copy=False)


def greedy_row_major(
    compatibility: CompatibilityMatrices,
    *,
    boundary_weight: float = 0.2,
) -> np.ndarray:
    if boundary_weight < 0:
        raise ValueError("boundary_weight must be non-negative")
    unary = placement_unary(compatibility)
    order = np.full(TILE_COUNT, -1, dtype=np.int32)
    unused = np.ones(TILE_COUNT, dtype=bool)
    for position in range(TILE_COUNT):
        candidates = np.flatnonzero(unused)
        costs = _candidate_increment(
            position, order, candidates, compatibility, unary, boundary_weight
        )
        best_local = np.lexsort((candidates, costs))[0]
        selected = int(candidates[best_local])
        order[position] = selected
        unused[selected] = False
    return validate_permutation(order, name="greedy_position_to_slot")


@dataclass
class _BeamState:
    cost: float
    order: np.ndarray
    unused: np.ndarray


def beam_row_major(
    compatibility: CompatibilityMatrices,
    *,
    width: int = 8,
    candidate_pool: int = 4,
    boundary_weight: float = 0.2,
) -> np.ndarray:
    if width <= 0 or candidate_pool <= 0:
        raise ValueError("width and candidate_pool must be positive")
    if boundary_weight < 0:
        raise ValueError("boundary_weight must be non-negative")
    unary = placement_unary(compatibility)
    initial = _BeamState(
        cost=0.0,
        order=np.full(TILE_COUNT, -1, dtype=np.int32),
        unused=np.ones(TILE_COUNT, dtype=bool),
    )
    beam = [initial]
    for position in range(TILE_COUNT):
        expanded: list[_BeamState] = []
        for state in beam:
            candidates = np.flatnonzero(state.unused)
            increments = _candidate_increment(
                position,
                state.order,
                candidates,
                compatibility,
                unary,
                boundary_weight,
            )
            take = min(candidate_pool, len(candidates))
            if take < len(candidates):
                local = np.argpartition(increments, take - 1)[:take]
            else:
                local = np.arange(len(candidates))
            local = local[np.lexsort((candidates[local], increments[local]))]
            for candidate_index in local.tolist():
                selected = int(candidates[candidate_index])
                order = state.order.copy()
                unused = state.unused.copy()
                order[position] = selected
                unused[selected] = False
                expanded.append(
                    _BeamState(
                        cost=state.cost + float(increments[candidate_index]),
                        order=order,
                        unused=unused,
                    )
                )
        expanded.sort(key=lambda state: (state.cost, tuple(state.order[: position + 1].tolist())))
        beam = expanded[:width]
    return validate_permutation(beam[0].order, name="beam_position_to_slot")


def _layout_edge_cost(order: np.ndarray, compatibility: CompatibilityMatrices) -> float:
    grid = order.reshape(GRID, GRID)
    right = compatibility.right[grid[:, :-1], grid[:, 1:]].sum(dtype=np.float64)
    down = compatibility.down[grid[:-1, :], grid[1:, :]].sum(dtype=np.float64)
    return float(right + down)


def layout_objective(
    position_to_slot: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    boundary_weight: float = 0.2,
) -> float:
    order = validate_permutation(position_to_slot, name="position_to_slot")
    unary = placement_unary(compatibility)
    return _objective_with_unary(order, compatibility, unary, boundary_weight)


def _objective_with_unary(
    order: np.ndarray,
    compatibility: CompatibilityMatrices,
    unary: np.ndarray,
    boundary_weight: float,
) -> float:
    positions = np.arange(TILE_COUNT, dtype=np.int32)
    return _layout_edge_cost(order, compatibility) + float(
        boundary_weight * unary[positions, order].sum(dtype=np.float64)
    )


def _local_cell_costs(
    order: np.ndarray,
    compatibility: CompatibilityMatrices,
    unary: np.ndarray,
    boundary_weight: float,
) -> np.ndarray:
    costs = boundary_weight * unary[np.arange(TILE_COUNT), order]
    grid = order.reshape(GRID, GRID)
    for row in range(GRID):
        for column in range(GRID):
            position = row * GRID + column
            slot = int(grid[row, column])
            if column > 0:
                costs[position] += 0.5 * compatibility.right[int(grid[row, column - 1]), slot]
            if column + 1 < GRID:
                costs[position] += 0.5 * compatibility.right[slot, int(grid[row, column + 1])]
            if row > 0:
                costs[position] += 0.5 * compatibility.down[int(grid[row - 1, column]), slot]
            if row + 1 < GRID:
                costs[position] += 0.5 * compatibility.down[slot, int(grid[row + 1, column])]
    return costs


def swap_refine(
    position_to_slot: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    boundary_weight: float = 0.2,
    weak_cells: int = 48,
    max_swaps: int = 32,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Deterministic bounded swaps over the currently weakest cells."""
    order = validate_permutation(position_to_slot, name="position_to_slot").copy()
    if weak_cells < 2 or max_swaps < 0:
        raise ValueError("weak_cells must be >=2 and max_swaps non-negative")
    unary = placement_unary(compatibility)
    current = _objective_with_unary(order, compatibility, unary, boundary_weight)
    for _ in range(max_swaps):
        local = _local_cell_costs(order, compatibility, unary, boundary_weight)
        count = min(weak_cells, TILE_COUNT)
        weak = np.argsort(-local, kind="stable")[:count]
        best_delta = 0.0
        best_pair: tuple[int, int] | None = None
        for first_index in range(len(weak)):
            first = int(weak[first_index])
            for second in weak[first_index + 1 :].tolist():
                candidate = order.copy()
                candidate[first], candidate[second] = candidate[second], candidate[first]
                objective = _objective_with_unary(
                    candidate, compatibility, unary, boundary_weight
                )
                delta = objective - current
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
        order[first], order[second] = order[second], order[first]
        current += best_delta
    return validate_permutation(order, name="refined_position_to_slot")


def _swap_blocks(
    order: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray | None:
    first_row, first_column = first
    second_row, second_column = second
    height, width = shape
    first_cells = {
        (row, column)
        for row in range(first_row, first_row + height)
        for column in range(first_column, first_column + width)
    }
    second_cells = {
        (row, column)
        for row in range(second_row, second_row + height)
        for column in range(second_column, second_column + width)
    }
    if first_cells & second_cells:
        return None
    grid = order.reshape(GRID, GRID).copy()
    first_values = grid[
        first_row : first_row + height, first_column : first_column + width
    ].copy()
    second_values = grid[
        second_row : second_row + height, second_column : second_column + width
    ].copy()
    grid[first_row : first_row + height, first_column : first_column + width] = second_values
    grid[second_row : second_row + height, second_column : second_column + width] = first_values
    return grid.ravel()


def segment_block_refine(
    position_to_slot: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    boundary_weight: float = 0.2,
    weak_cells: int = 20,
    max_moves: int = 4,
    block_shapes: tuple[tuple[int, int], ...] = ((1, 2), (2, 1), (2, 2), (4, 4)),
    tolerance: float = 1e-8,
) -> np.ndarray:
    """R3: deterministic equal-shape swaps around weak layout regions."""
    order = validate_permutation(position_to_slot, name="position_to_slot").copy()
    if weak_cells < 2 or max_moves < 0:
        raise ValueError("weak_cells must be >=2 and max_moves non-negative")
    unary = placement_unary(compatibility)
    current = _objective_with_unary(order, compatibility, unary, boundary_weight)
    for _ in range(max_moves):
        local = _local_cell_costs(order, compatibility, unary, boundary_weight)
        weak = np.argsort(-local, kind="stable")[: min(weak_cells, TILE_COUNT)]
        best_objective = current
        best_candidate = None
        best_key = None
        for height, width in block_shapes:
            if height <= 0 or width <= 0 or height > GRID or width > GRID:
                raise ValueError(f"invalid block shape {(height, width)}")
            anchors = sorted(
                {
                    (
                        min(position // GRID, GRID - height),
                        min(position % GRID, GRID - width),
                    )
                    for position in weak.tolist()
                }
            )
            for first_index, first in enumerate(anchors):
                for second in anchors[first_index + 1 :]:
                    candidate = _swap_blocks(order, first, second, (height, width))
                    if candidate is None:
                        continue
                    objective = _objective_with_unary(
                        candidate, compatibility, unary, boundary_weight
                    )
                    key = (height * width, height, width, first, second)
                    if objective < best_objective - tolerance or (
                        abs(objective - best_objective) <= tolerance
                        and best_candidate is not None
                        and key < best_key
                    ):
                        best_objective = objective
                        best_candidate = candidate
                        best_key = key
        if best_candidate is None:
            break
        order = best_candidate
        current = best_objective
    return validate_permutation(order, name="segment_refined_position_to_slot")


def simulated_anneal_swaps(
    position_to_slot: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    seed: int,
    evaluations: int = 2000,
    boundary_weight: float = 0.2,
    weak_bias: float = 0.8,
) -> np.ndarray:
    """R4: fixed-budget seeded annealing over tile swaps."""
    if evaluations < 0 or not 0.0 <= weak_bias <= 1.0:
        raise ValueError("invalid annealing budget or weak_bias")
    order = validate_permutation(position_to_slot, name="position_to_slot").copy()
    unary = placement_unary(compatibility)
    current = _objective_with_unary(order, compatibility, unary, boundary_weight)
    best = order.copy()
    best_objective = current
    rng = np.random.default_rng(seed)
    finite = np.concatenate(
        [
            compatibility.right[np.isfinite(compatibility.right)],
            compatibility.down[np.isfinite(compatibility.down)],
        ]
    )
    scale = max(float(np.median(finite)), 1e-4)
    for evaluation in range(evaluations):
        progress = evaluation / max(evaluations - 1, 1)
        temperature = scale * (0.5 * (1.0 - progress) + 0.005)
        if rng.random() < weak_bias:
            local = _local_cell_costs(order, compatibility, unary, boundary_weight)
            pool = np.argsort(-local, kind="stable")[:48]
            first, second = rng.choice(pool, size=2, replace=False).tolist()
        else:
            first, second = rng.choice(TILE_COUNT, size=2, replace=False).tolist()
        candidate = order.copy()
        candidate[first], candidate[second] = candidate[second], candidate[first]
        objective = _objective_with_unary(candidate, compatibility, unary, boundary_weight)
        delta = objective - current
        if delta < 0.0 or rng.random() < np.exp(-delta / max(temperature, 1e-8)):
            order = candidate
            current = objective
            if objective < best_objective:
                best = candidate.copy()
                best_objective = objective
    return validate_permutation(best, name="annealed_position_to_slot")


def simulated_anneal_mixed(
    position_to_slot: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    seed: int,
    evaluations: int = 20000,
    boundary_weight: float = 0.2,
    block_probability: float = 0.45,
) -> np.ndarray:
    """Anneal tile swaps and segment-preserving equal-shape block moves."""
    if evaluations < 0 or not 0.0 <= block_probability <= 1.0:
        raise ValueError("invalid mixed annealing parameters")
    order = validate_permutation(position_to_slot, name="position_to_slot").copy()
    unary = placement_unary(compatibility)
    current = _objective_with_unary(order, compatibility, unary, boundary_weight)
    best = order.copy()
    best_objective = current
    rng = np.random.default_rng(seed)
    finite = np.concatenate(
        [
            compatibility.right[np.isfinite(compatibility.right)],
            compatibility.down[np.isfinite(compatibility.down)],
        ]
    )
    scale = max(float(np.median(finite)), 1e-4)
    block_shapes = (
        (1, 2),
        (2, 1),
        (1, 4),
        (4, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (1, GRID),
        (GRID, 1),
    )
    for evaluation in range(evaluations):
        progress = evaluation / max(evaluations - 1, 1)
        temperature = scale * (0.8 * (1.0 - progress) + 0.003)
        move_scale = 1.0
        candidate = None
        if rng.random() < block_probability:
            height, width = block_shapes[int(rng.integers(len(block_shapes)))]
            first = (
                int(rng.integers(GRID - height + 1)),
                int(rng.integers(GRID - width + 1)),
            )
            second = (
                int(rng.integers(GRID - height + 1)),
                int(rng.integers(GRID - width + 1)),
            )
            candidate = _swap_blocks(order, first, second, (height, width))
            move_scale = max(1.0, np.sqrt(float(height * width)))
        if candidate is None:
            if rng.random() < 0.8:
                local = _local_cell_costs(
                    order, compatibility, unary, boundary_weight
                )
                pool = np.argsort(-local, kind="stable")[:64]
                first, second = rng.choice(pool, size=2, replace=False).tolist()
            else:
                first, second = rng.choice(TILE_COUNT, size=2, replace=False).tolist()
            candidate = order.copy()
            candidate[first], candidate[second] = candidate[second], candidate[first]
        objective = _objective_with_unary(
            candidate, compatibility, unary, boundary_weight
        )
        delta = objective - current
        if delta < 0.0 or rng.random() < np.exp(
            -delta / max(temperature * move_scale, 1e-8)
        ):
            order = candidate
            current = objective
            if objective < best_objective:
                best = candidate.copy()
                best_objective = objective
    return validate_permutation(best, name="mixed_annealed_position_to_slot")


def _parent_neighbours(order: np.ndarray) -> np.ndarray:
    """Return tile x (left,right,up,down) neighbours for one parent layout."""
    grid = order.reshape(GRID, GRID)
    neighbours = np.full((TILE_COUNT, 4), -1, dtype=np.int32)
    for row in range(GRID):
        for column in range(GRID):
            tile = int(grid[row, column])
            if column > 0:
                neighbours[tile, 0] = int(grid[row, column - 1])
            if column + 1 < GRID:
                neighbours[tile, 1] = int(grid[row, column + 1])
            if row > 0:
                neighbours[tile, 2] = int(grid[row - 1, column])
            if row + 1 < GRID:
                neighbours[tile, 3] = int(grid[row + 1, column])
    return neighbours


def _kernel_growing_crossover(
    first_parent: np.ndarray,
    second_parent: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    rng: np.random.Generator,
    candidate_top_k: int = 2,
) -> np.ndarray:
    """Grow a child while preserving parent-supported frontier segments."""
    parents = (first_parent, second_parent)
    parent_neighbours = tuple(_parent_neighbours(parent) for parent in parents)
    right = rank_normalize(compatibility.right)
    down = rank_normalize(compatibility.down)
    np.fill_diagonal(right, 2.0)
    np.fill_diagonal(down, 2.0)
    right_out = np.argsort(right, axis=1, kind="stable")[:, :candidate_top_k]
    right_in = np.argsort(right, axis=0, kind="stable")[:candidate_top_k, :].T
    down_out = np.argsort(down, axis=1, kind="stable")[:, :candidate_top_k]
    down_in = np.argsort(down, axis=0, kind="stable")[:candidate_top_k, :].T
    unary = placement_unary(compatibility)
    grid = np.full((GRID, GRID), -1, dtype=np.int32)
    unused = np.ones(TILE_COUNT, dtype=bool)
    center = (GRID // 2 - 1, GRID // 2 - 1)
    center_position = center[0] * GRID + center[1]
    start_parent = int(rng.integers(2))
    start = int(parents[start_parent][center_position])
    grid[center] = start
    unused[start] = False
    frontier: set[tuple[int, int]] = set()

    def add_frontier(row: int, column: int) -> None:
        for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nr, nc = row + dr, column + dc
            if 0 <= nr < GRID and 0 <= nc < GRID and grid[nr, nc] < 0:
                frontier.add((nr, nc))

    add_frontier(*center)
    while np.any(unused):
        # Prefer cells already constrained by multiple placed sides.
        cell = min(
            frontier,
            key=lambda rc: (
                -sum(
                    0 <= rc[0] + dr < GRID
                    and 0 <= rc[1] + dc < GRID
                    and grid[rc[0] + dr, rc[1] + dc] >= 0
                    for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0))
                ),
                abs(rc[0] - center[0]) + abs(rc[1] - center[1]),
                rc[0],
                rc[1],
            ),
        )
        frontier.remove(cell)
        row, column = cell
        suggestions: list[int] = []
        # Absolute-position inheritance is a weak vote; adjacency agreement
        # from either parent is added once per supporting relation.
        position = row * GRID + column
        suggestions.extend(int(parent[position]) for parent in parents)
        neighbour_specs = []
        if column > 0 and grid[row, column - 1] >= 0:
            neighbour_specs.append((int(grid[row, column - 1]), 1, right_out))
        if column + 1 < GRID and grid[row, column + 1] >= 0:
            neighbour_specs.append((int(grid[row, column + 1]), 0, right_in))
        if row > 0 and grid[row - 1, column] >= 0:
            neighbour_specs.append((int(grid[row - 1, column]), 3, down_out))
        if row + 1 < GRID and grid[row + 1, column] >= 0:
            neighbour_specs.append((int(grid[row + 1, column]), 2, down_in))
        for neighbour, parent_direction, score_candidates in neighbour_specs:
            for parent_lookup in parent_neighbours:
                proposal = int(parent_lookup[neighbour, parent_direction])
                if proposal >= 0:
                    suggestions.append(proposal)
            suggestions.extend(int(value) for value in score_candidates[neighbour])
        votes: dict[int, int] = {}
        for tile in suggestions:
            if unused[tile]:
                votes[tile] = votes.get(tile, 0) + 1
        candidates = np.asarray(sorted(votes), dtype=np.int32)
        if len(candidates) == 0:
            candidates = np.flatnonzero(unused).astype(np.int32)
        costs = 0.05 * unary[position, candidates].astype(np.float64)
        contacts = 0
        if column > 0 and grid[row, column - 1] >= 0:
            costs += right[int(grid[row, column - 1]), candidates]
            contacts += 1
        if column + 1 < GRID and grid[row, column + 1] >= 0:
            costs += right[candidates, int(grid[row, column + 1])]
            contacts += 1
        if row > 0 and grid[row - 1, column] >= 0:
            costs += down[int(grid[row - 1, column]), candidates]
            contacts += 1
        if row + 1 < GRID and grid[row + 1, column] >= 0:
            costs += down[candidates, int(grid[row + 1, column])]
            contacts += 1
        costs /= max(contacts, 1)
        vote_values = np.asarray([votes.get(int(tile), 0) for tile in candidates])
        selected_index = int(
            np.lexsort((candidates, costs, -vote_values))[0]
        )
        selected = int(candidates[selected_index])
        grid[row, column] = selected
        unused[selected] = False
        add_frontier(row, column)
    return validate_permutation(grid.ravel(), name="genetic_child_position_to_slot")


def segment_preserving_genetic_solver(
    initial_layouts: list[np.ndarray],
    compatibility: CompatibilityMatrices,
    *,
    seed: int,
    population_size: int = 24,
    generations: int = 20,
    elite_size: int = 6,
    mutation_probability: float = 0.7,
) -> np.ndarray:
    """Approximate full-layout search with segment-preserving crossover."""
    if not initial_layouts:
        raise ValueError("at least one initial layout is required")
    if (
        population_size < 2
        or generations < 0
        or not 1 <= elite_size < population_size
        or not 0.0 <= mutation_probability <= 1.0
    ):
        raise ValueError("invalid genetic solver parameters")
    rng = np.random.default_rng(seed)
    ranked = CompatibilityMatrices(
        name=f"{compatibility.name}_genetic_ranked",
        right=rank_normalize(compatibility.right),
        down=rank_normalize(compatibility.down),
    )
    unary = placement_unary(compatibility)

    def objective(order: np.ndarray) -> float:
        return _objective_with_unary(order, ranked, unary, 0.05)

    seeds = [
        validate_permutation(layout, name="genetic_seed").copy()
        for layout in initial_layouts
    ]
    population = seeds[:population_size]
    block_shapes = ((1, 2), (2, 1), (2, 2), (3, 3), (4, 4), (1, GRID), (GRID, 1))
    while len(population) < population_size:
        candidate = seeds[int(rng.integers(len(seeds)))].copy()
        for _ in range(1 + int(rng.integers(3))):
            height, width = block_shapes[int(rng.integers(len(block_shapes)))]
            first = (
                int(rng.integers(GRID - height + 1)),
                int(rng.integers(GRID - width + 1)),
            )
            second = (
                int(rng.integers(GRID - height + 1)),
                int(rng.integers(GRID - width + 1)),
            )
            mutated = _swap_blocks(candidate, first, second, (height, width))
            if mutated is not None:
                candidate = mutated
        population.append(candidate)

    for _ in range(generations):
        population.sort(key=lambda order: (objective(order), tuple(order.tolist())))
        next_population = [order.copy() for order in population[:elite_size]]
        parent_pool = population[: max(elite_size * 2, population_size // 2)]
        while len(next_population) < population_size:
            first_index, second_index = rng.choice(
                len(parent_pool), size=2, replace=False
            ).tolist()
            child = _kernel_growing_crossover(
                parent_pool[first_index],
                parent_pool[second_index],
                compatibility,
                rng=rng,
            )
            if rng.random() < mutation_probability:
                height, width = block_shapes[int(rng.integers(len(block_shapes)))]
                first = (
                    int(rng.integers(GRID - height + 1)),
                    int(rng.integers(GRID - width + 1)),
                )
                second = (
                    int(rng.integers(GRID - height + 1)),
                    int(rng.integers(GRID - width + 1)),
                )
                mutated = _swap_blocks(child, first, second, (height, width))
                if mutated is not None:
                    child = mutated
            next_population.append(child)
        population = next_population
    population.sort(key=lambda order: (objective(order), tuple(order.tolist())))
    return validate_permutation(population[0], name="genetic_position_to_slot")


def _sinkhorn(values: np.ndarray, iterations: int) -> np.ndarray:
    result = np.maximum(np.asarray(values, dtype=np.float64), 1e-30)
    for _ in range(iterations):
        result /= np.maximum(result.sum(axis=1, keepdims=True), 1e-30)
        result /= np.maximum(result.sum(axis=0, keepdims=True), 1e-30)
    return result


def relaxation_labeling_solver(
    compatibility: CompatibilityMatrices,
    *,
    initial: np.ndarray | None = None,
    iterations: int = 6,
    sinkhorn_iterations: int = 12,
    temperature: float = 0.20,
    inertia: float = 0.15,
    boundary_weight: float = 0.2,
) -> np.ndarray:
    """G3/G4 dense bistochastic relaxation followed by Hungarian projection."""
    if iterations <= 0 or sinkhorn_iterations <= 0 or temperature <= 0:
        raise ValueError("relaxation iteration counts and temperature must be positive")
    if not 0.0 <= inertia <= 1.0:
        raise ValueError("inertia must be in [0, 1]")
    seed_layout = (
        greedy_row_major(compatibility)
        if initial is None
        else validate_permutation(initial, name="initial_position_to_slot")
    )
    assignment = np.full((TILE_COUNT, TILE_COUNT), 0.10 / TILE_COUNT, dtype=np.float64)
    assignment[np.arange(TILE_COUNT), seed_layout] += 0.90
    right = rank_normalize(compatibility.right).astype(np.float64)
    down = rank_normalize(compatibility.down).astype(np.float64)
    np.fill_diagonal(right, 2.0)
    np.fill_diagonal(down, 2.0)
    unary = placement_unary(compatibility).astype(np.float64)
    positions = np.arange(TILE_COUNT, dtype=np.int32)
    right_positions = positions[positions % GRID < GRID - 1]
    down_positions = positions[positions < TILE_COUNT - GRID]
    best = seed_layout.copy()
    best_objective = _objective_with_unary(
        best, compatibility, placement_unary(compatibility), boundary_weight
    )
    for iteration in range(iterations):
        gradient = boundary_weight * unary
        gradient[right_positions] += assignment[right_positions + 1] @ right.T
        gradient[right_positions + 1] += assignment[right_positions] @ right
        gradient[down_positions] += assignment[down_positions + GRID] @ down.T
        gradient[down_positions + GRID] += assignment[down_positions] @ down
        annealed_temperature = temperature * (1.0 - 0.6 * iteration / max(iterations - 1, 1))
        logits = -gradient / annealed_temperature
        if inertia > 0:
            logits += inertia * np.log(np.maximum(assignment, 1e-30))
        logits -= logits.max(axis=1, keepdims=True)
        assignment = _sinkhorn(np.exp(logits), sinkhorn_iterations)
        rows, columns = linear_sum_assignment(-assignment)
        candidate = np.empty(TILE_COUNT, dtype=np.int32)
        candidate[rows] = columns
        objective = _objective_with_unary(
            candidate, compatibility, placement_unary(compatibility), boundary_weight
        )
        if objective < best_objective:
            best = candidate
            best_objective = objective
    return validate_permutation(best, name="relaxation_position_to_slot")


def _four_side_linear_costs(
    order: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    boundary_weight: float,
    inertia: float,
) -> np.ndarray:
    """Linearise the full four-neighbour energy around a hard layout."""
    grid = order.reshape(GRID, GRID)
    right = rank_normalize(compatibility.right)
    down = rank_normalize(compatibility.down)
    # A candidate can currently occupy a neighbouring cell and move away in
    # the simultaneous assignment.  Treat that temporary self-comparison as a
    # poor finite match instead of poisoning the Hungarian matrix with inf.
    np.fill_diagonal(right, 2.0)
    np.fill_diagonal(down, 2.0)
    unary = placement_unary(compatibility)
    costs = boundary_weight * unary.astype(np.float64)
    counts = np.zeros(TILE_COUNT, dtype=np.float64)
    for row in range(GRID):
        for column in range(GRID):
            position = row * GRID + column
            if column > 0:
                costs[position] += right[int(grid[row, column - 1]), :]
                counts[position] += 1.0
            if column + 1 < GRID:
                costs[position] += right[:, int(grid[row, column + 1])]
                counts[position] += 1.0
            if row > 0:
                costs[position] += down[int(grid[row - 1, column]), :]
                counts[position] += 1.0
            if row + 1 < GRID:
                costs[position] += down[:, int(grid[row + 1, column])]
                counts[position] += 1.0
    costs /= np.maximum(counts[:, None], 1.0)
    if inertia > 0:
        costs[np.arange(TILE_COUNT), order] -= inertia
    costs += 1e-12 * (
        np.arange(TILE_COUNT, dtype=np.float64)[:, None] * TILE_COUNT
        + np.arange(TILE_COUNT, dtype=np.float64)[None, :]
    )
    return costs


def _assignment_cycles(current: np.ndarray, candidate: np.ndarray) -> list[np.ndarray]:
    """Return disjoint old-position cycles that realise candidate."""
    current_position = np.empty(TILE_COUNT, dtype=np.int32)
    current_position[current] = np.arange(TILE_COUNT, dtype=np.int32)
    predecessor = current_position[candidate]
    visited = np.zeros(TILE_COUNT, dtype=bool)
    cycles: list[np.ndarray] = []
    for start in range(TILE_COUNT):
        if visited[start] or int(predecessor[start]) == start:
            visited[start] = True
            continue
        cycle = []
        position = start
        while not visited[position]:
            visited[position] = True
            cycle.append(position)
            position = int(predecessor[position])
        if len(cycle) >= 2:
            cycles.append(np.asarray(cycle, dtype=np.int32))
    return cycles


def four_side_hungarian_refine(
    position_to_slot: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    iterations: int = 8,
    boundary_weight: float = 0.05,
    inertia: float = 0.05,
) -> np.ndarray:
    """Iterative global assignment using all four physical tile contacts.

    Each iteration builds a position-by-tile matrix from the current left,
    right, up and down neighbours, solves the resulting bijection exactly, and
    accepts only moves that improve the original pairwise objective.  If the
    simultaneous assignment overshoots, its disjoint permutation cycles are
    tried independently.  This makes the method a safe large-neighbourhood
    refinement rather than the previous one-shot unresolved-cell fill.
    """
    if iterations < 0 or boundary_weight < 0 or inertia < 0:
        raise ValueError("invalid four-side refinement parameters")
    order = validate_permutation(position_to_slot, name="position_to_slot").copy()
    ranked = CompatibilityMatrices(
        name=f"{compatibility.name}_four_side_ranked",
        right=rank_normalize(compatibility.right),
        down=rank_normalize(compatibility.down),
    )
    unary = placement_unary(compatibility)
    current = _objective_with_unary(order, ranked, unary, boundary_weight)
    tolerance = 1e-9
    for _ in range(iterations):
        costs = _four_side_linear_costs(
            order,
            compatibility,
            boundary_weight=boundary_weight,
            inertia=inertia,
        )
        positions, tiles = linear_sum_assignment(costs)
        candidate = np.empty(TILE_COUNT, dtype=np.int32)
        candidate[positions] = tiles
        candidate_objective = _objective_with_unary(
            candidate, ranked, unary, boundary_weight
        )
        if candidate_objective < current - tolerance:
            order = candidate
            current = candidate_objective
            continue

        improved = False
        cycles = _assignment_cycles(order, candidate)
        cycles.sort(
            key=lambda cycle: (
                float(costs[cycle, candidate[cycle]].sum()),
                -len(cycle),
                tuple(cycle.tolist()),
            )
        )
        for cycle in cycles:
            trial = order.copy()
            trial[cycle] = candidate[cycle]
            objective = _objective_with_unary(
                trial, ranked, unary, boundary_weight
            )
            if objective < current - tolerance:
                order = trial
                current = objective
                improved = True
        if not improved:
            break
    return validate_permutation(order, name="four_side_refined_position_to_slot")


def large_neighborhood_reassign(
    position_to_slot: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    seed: int,
    iterations: int = 40,
    subset_size: int = 128,
    weak_fraction: float = 0.5,
    boundary_weight: float = 0.05,
) -> np.ndarray:
    """Destroy/reassign weak cells with a conditional four-side Hungarian step."""
    if (
        iterations < 0
        or not 2 <= subset_size <= TILE_COUNT
        or not 0.0 <= weak_fraction <= 1.0
        or boundary_weight < 0
    ):
        raise ValueError("invalid large-neighbourhood parameters")
    rng = np.random.default_rng(seed)
    order = validate_permutation(position_to_slot, name="position_to_slot").copy()
    ranked = CompatibilityMatrices(
        name=f"{compatibility.name}_lns_ranked",
        right=rank_normalize(compatibility.right),
        down=rank_normalize(compatibility.down),
    )
    right = ranked.right.copy()
    down = ranked.down.copy()
    np.fill_diagonal(right, 2.0)
    np.fill_diagonal(down, 2.0)
    unary = placement_unary(compatibility)
    current = _objective_with_unary(order, ranked, unary, boundary_weight)
    best = order.copy()
    best_objective = current
    weak_count = int(round(subset_size * weak_fraction))
    for iteration in range(iterations):
        local = _local_cell_costs(order, ranked, unary, boundary_weight)
        weak_pool_size = min(TILE_COUNT, max(weak_count * 3, weak_count))
        weak_pool = np.argsort(-local, kind="stable")[:weak_pool_size]
        if weak_count:
            chosen_weak = rng.choice(
                weak_pool, size=min(weak_count, len(weak_pool)), replace=False
            )
        else:
            chosen_weak = np.empty(0, dtype=np.int32)
        available = np.setdiff1d(
            np.arange(TILE_COUNT, dtype=np.int32),
            chosen_weak,
            assume_unique=False,
        )
        donor_count = subset_size - len(chosen_weak)
        donors = rng.choice(available, size=donor_count, replace=False)
        selected = np.sort(np.concatenate([chosen_weak, donors])).astype(np.int32)
        selected_mask = np.zeros(TILE_COUNT, dtype=bool)
        selected_mask[selected] = True
        selected_tiles = order[selected]
        grid = order.reshape(GRID, GRID)
        costs = np.zeros((subset_size, subset_size), dtype=np.float64)
        for local_position, position in enumerate(selected.tolist()):
            row, column = divmod(position, GRID)
            contacts = 0
            if column > 0 and not selected_mask[position - 1]:
                costs[local_position] += right[
                    int(grid[row, column - 1]), selected_tiles
                ]
                contacts += 1
            if column + 1 < GRID and not selected_mask[position + 1]:
                costs[local_position] += right[
                    selected_tiles, int(grid[row, column + 1])
                ]
                contacts += 1
            if row > 0 and not selected_mask[position - GRID]:
                costs[local_position] += down[
                    int(grid[row - 1, column]), selected_tiles
                ]
                contacts += 1
            if row + 1 < GRID and not selected_mask[position + GRID]:
                costs[local_position] += down[
                    selected_tiles, int(grid[row + 1, column])
                ]
                contacts += 1
            if contacts:
                costs[local_position] /= contacts
            else:
                costs[local_position] = 0.5
            costs[local_position] += boundary_weight * unary[
                position, selected_tiles
            ]
        rows, columns = linear_sum_assignment(costs)
        candidate = order.copy()
        candidate[selected[rows]] = selected_tiles[columns]
        objective = _objective_with_unary(
            candidate, ranked, unary, boundary_weight
        )
        progress = iteration / max(iterations - 1, 1)
        temperature = 0.05 * (1.0 - progress) + 0.001
        delta = objective - current
        if delta < 0.0 or rng.random() < np.exp(-delta / temperature):
            order = candidate
            current = objective
            if objective < best_objective:
                best = candidate.copy()
                best_objective = objective
    return validate_permutation(best, name="lns_position_to_slot")


def _topk_similarity(
    matrix: np.ndarray, *, top_k: int, temperature: float
) -> csr_matrix:
    ranks = rank_normalize(matrix)
    order = np.argsort(ranks, axis=1, kind="stable")[:, :top_k]
    rows = np.repeat(np.arange(TILE_COUNT, dtype=np.int32), top_k)
    columns = order.ravel().astype(np.int32)
    values = np.exp(-ranks[rows, columns] / temperature).astype(np.float64)
    row_sums = np.bincount(rows, weights=values, minlength=TILE_COUNT)
    values /= np.maximum(row_sums[rows], 1e-12)
    return csr_matrix(
        (values, (rows, columns)), shape=(TILE_COUNT, TILE_COUNT)
    )


def multi_phase_relaxation_solver(
    compatibility: CompatibilityMatrices,
    *,
    initial: np.ndarray,
    top_k: int = 8,
    phases: int = 12,
    iterations_per_phase: int = 4,
    anchor_batch: int = 48,
    temperature: float = 0.20,
    reset_memory: float = 0.25,
) -> np.ndarray:
    """Sparse progressive relaxation with anchoring and phase resets."""
    if (
        not 1 <= top_k < TILE_COUNT
        or phases <= 0
        or iterations_per_phase <= 0
        or anchor_batch <= 0
        or temperature <= 0
        or not 0.0 <= reset_memory <= 1.0
    ):
        raise ValueError("invalid multi-phase relaxation parameters")
    seed_layout = validate_permutation(initial, name="initial_position_to_slot")
    right = _topk_similarity(
        compatibility.right, top_k=top_k, temperature=temperature
    )
    down = _topk_similarity(
        compatibility.down, top_k=top_k, temperature=temperature
    )
    assignment = np.full(
        (TILE_COUNT, TILE_COUNT), 0.10 / TILE_COUNT, dtype=np.float64
    )
    assignment[seed_layout, np.arange(TILE_COUNT)] += 0.90
    anchored_tile = np.zeros(TILE_COUNT, dtype=bool)
    anchored_position = np.zeros(TILE_COUNT, dtype=bool)
    anchors: dict[int, int] = {}
    positions = np.arange(TILE_COUNT, dtype=np.int32)
    right_positions = positions[positions % GRID < GRID - 1]
    down_positions = positions[positions < TILE_COUNT - GRID]
    unary = placement_unary(compatibility).T.astype(np.float64)
    prior = np.exp(-0.05 * unary)

    def enforce_anchors(values: np.ndarray) -> np.ndarray:
        values = np.maximum(values, 1e-30)
        if anchors:
            tiles = np.asarray(sorted(anchors), dtype=np.int32)
            anchor_positions = np.asarray([anchors[int(tile)] for tile in tiles], dtype=np.int32)
            values[tiles, :] = 0.0
            values[:, anchor_positions] = 0.0
            values[tiles, anchor_positions] = 1.0
        for _ in range(6):
            free_rows = ~anchored_tile
            free_columns = ~anchored_position
            if np.any(free_rows):
                row_sum = values[free_rows][:, free_columns].sum(axis=1)
                values[np.ix_(free_rows, free_columns)] /= np.maximum(
                    row_sum[:, None], 1e-30
                )
            if np.any(free_columns):
                column_sum = values[free_rows][:, free_columns].sum(axis=0)
                values[np.ix_(free_rows, free_columns)] /= np.maximum(
                    column_sum[None, :], 1e-30
                )
        return values

    assignment = enforce_anchors(assignment)
    for _ in range(phases):
        if len(anchors) >= TILE_COUNT:
            break
        for _ in range(iterations_per_phase):
            support = 1e-6 * prior.copy()
            support[:, right_positions] += right @ assignment[:, right_positions + 1]
            support[:, right_positions + 1] += right.T @ assignment[:, right_positions]
            support[:, down_positions] += down @ assignment[:, down_positions + GRID]
            support[:, down_positions + GRID] += down.T @ assignment[:, down_positions]
            assignment = enforce_anchors(assignment * np.maximum(support, 1e-12))

        free_tiles = np.flatnonzero(~anchored_tile)
        free_positions = np.flatnonzero(~anchored_position)
        if len(free_tiles) == 0:
            break
        if anchors:
            frontier = set()
            for position in anchors.values():
                row, column = divmod(position, GRID)
                for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    nr, nc = row + dr, column + dc
                    neighbour = nr * GRID + nc
                    if (
                        0 <= nr < GRID
                        and 0 <= nc < GRID
                        and not anchored_position[neighbour]
                    ):
                        frontier.add(neighbour)
            candidate_positions = (
                np.asarray(sorted(frontier), dtype=np.int32)
                if frontier
                else free_positions
            )
        else:
            candidate_positions = free_positions
        values = assignment[np.ix_(free_tiles, candidate_positions)]
        flat_order = np.argsort(-values.ravel(), kind="stable")
        chosen_tiles: set[int] = set()
        chosen_positions: set[int] = set()
        target_count = min(anchor_batch, len(free_tiles))
        for flat_index in flat_order.tolist():
            local_tile, local_position = divmod(flat_index, len(candidate_positions))
            tile = int(free_tiles[local_tile])
            position = int(candidate_positions[local_position])
            if tile in chosen_tiles or position in chosen_positions:
                continue
            anchors[tile] = position
            anchored_tile[tile] = True
            anchored_position[position] = True
            chosen_tiles.add(tile)
            chosen_positions.add(position)
            if len(chosen_tiles) >= target_count:
                break
        # Frontier can be smaller than the requested batch.  Always make
        # progress by anchoring the strongest remaining unrestricted pairs.
        if len(chosen_tiles) < target_count:
            free_tiles = np.flatnonzero(~anchored_tile)
            free_positions = np.flatnonzero(~anchored_position)
            values = assignment[np.ix_(free_tiles, free_positions)]
            rows, columns = linear_sum_assignment(-values)
            ranked_pairs = sorted(
                zip(rows.tolist(), columns.tolist(), strict=True),
                key=lambda pair: -values[pair[0], pair[1]],
            )
            for row_index, column_index in ranked_pairs:
                tile = int(free_tiles[row_index])
                position = int(free_positions[column_index])
                anchors[tile] = position
                anchored_tile[tile] = True
                anchored_position[position] = True
                chosen_tiles.add(tile)
                if len(chosen_tiles) >= target_count:
                    break
        free_tiles = np.flatnonzero(~anchored_tile)
        free_positions = np.flatnonzero(~anchored_position)
        reset = np.zeros_like(assignment)
        if len(free_tiles):
            reset[np.ix_(free_tiles, free_positions)] = 1.0 / len(free_positions)
            reset[np.ix_(free_tiles, free_positions)] = (
                (1.0 - reset_memory)
                * reset[np.ix_(free_tiles, free_positions)]
                + reset_memory
                * assignment[np.ix_(free_tiles, free_positions)]
            )
        for tile, position in anchors.items():
            reset[tile, position] = 1.0
        assignment = enforce_anchors(reset)

    rows, columns = linear_sum_assignment(-assignment)
    result = np.empty(TILE_COUNT, dtype=np.int32)
    result[columns] = rows
    return validate_permutation(result, name="multi_phase_relaxation_position_to_slot")


def _adaptive_reciprocal_similarity(
    costs: np.ndarray,
    *,
    top_k: int,
    power: float,
) -> csr_matrix:
    """Return an official-style sparse, reciprocal directional similarity.

    For a source side, ``alpha`` is the mean of its ``top_k`` dissimilarities
    and a candidate at one-based rank ``r`` receives
    ``max(1 - distance / alpha, 0) ** (r * power)``.  The same seam is also
    calibrated from the destination's opposite side (a column of ``costs``),
    and the two values are averaged.  The union of row- and column-top-k
    entries is retained so useful one-sided evidence is not discarded before
    symmetrisation.
    """
    values = np.asarray(costs, dtype=np.float64)
    if values.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError("directional costs must be a 576x576 matrix")
    if not 1 <= top_k < TILE_COUNT or power <= 0:
        raise ValueError("invalid adaptive similarity parameters")

    def row_calibration(
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        finite = np.isfinite(matrix)
        finite[np.arange(TILE_COUNT), np.arange(TILE_COUNT)] = False
        if np.any(finite.sum(axis=1) < top_k):
            raise ValueError("each directional-cost row needs top_k finite entries")
        safe = np.where(finite, np.maximum(matrix, 0.0), np.inf)
        full_order = np.argsort(safe, axis=1, kind="stable")
        selected_order = full_order[:, :top_k]
        selected = np.take_along_axis(safe, selected_order, axis=1)
        alpha = selected.mean(axis=1)
        ranks = np.empty((TILE_COUNT, TILE_COUNT), dtype=np.int32)
        row_indices = np.arange(TILE_COUNT, dtype=np.int32)[:, None]
        ranks[row_indices, full_order] = np.arange(
            1, TILE_COUNT + 1, dtype=np.int32
        )[None, :]
        return selected_order.astype(np.int32), alpha, ranks

    forward_order, forward_alpha, forward_ranks = row_calibration(values)
    opposite_order, opposite_alpha, opposite_ranks = row_calibration(values.T)

    candidates = np.zeros((TILE_COUNT, TILE_COUNT), dtype=bool)
    row_indices = np.arange(TILE_COUNT, dtype=np.int32)[:, None]
    candidates[row_indices, forward_order] = True
    candidates[opposite_order, row_indices] = True
    rows, columns = np.nonzero(candidates)
    rows = rows.astype(np.int32, copy=False)
    columns = columns.astype(np.int32, copy=False)
    distances = np.maximum(values[rows, columns], 0.0)

    def calibrated(
        distance: np.ndarray,
        alpha: np.ndarray,
        rank: np.ndarray,
    ) -> np.ndarray:
        result = np.zeros(len(distance), dtype=np.float64)
        zero_scale = alpha == 0.0
        result[zero_scale & (distance == 0.0)] = 1.0
        regular = ~zero_scale
        base = np.maximum(1.0 - distance[regular] / alpha[regular], 0.0)
        result[regular] = base ** (rank[regular].astype(np.float64) * power)
        return result

    forward = calibrated(
        distances,
        forward_alpha[rows],
        forward_ranks[rows, columns],
    )
    opposite = calibrated(
        distances,
        opposite_alpha[columns],
        opposite_ranks[columns, rows],
    )
    similarities = 0.5 * (forward + opposite)
    positive = similarities > 0.0
    return csr_matrix(
        (similarities[positive], (rows[positive], columns[positive])),
        shape=(TILE_COUNT, TILE_COUNT),
        dtype=np.float64,
    )


def _directional_best_buddies(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic outgoing and incoming best matches."""
    values = np.asarray(costs, dtype=np.float64).copy()
    if values.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError("directional costs must be a 576x576 matrix")
    values[~np.isfinite(values)] = np.inf
    np.fill_diagonal(values, np.inf)
    finite = np.isfinite(values)
    if np.any(~finite.any(axis=1)) or np.any(~finite.any(axis=0)):
        raise ValueError("each directional-cost row and column needs a finite neighbour")
    return (
        np.argmin(values, axis=1).astype(np.int32),
        np.argmin(values, axis=0).astype(np.int32),
    )


def faithful_multi_phase_relaxation_solver(
    compatibility: CompatibilityMatrices,
    *,
    top_k: int = 17,
    phases: int = TILE_COUNT,
    similarity_power: float = 1.0,
    convergence_threshold: float = 1e-4,
    max_iterations: int = 48,
    anchor_probability: float = 0.70,
) -> np.ndarray:
    """Multi-phase relaxation with one frontier anchor and a full reset.

    Probabilities are ``tile x position`` and start uniformly.  Every
    relaxation update is multiplicative and normalises rows only.  A phase
    stops when the average local consistency (ALC) changes by at most
    ``convergence_threshold`` or reaches ``max_iterations``.  Exactly one
    tile-position pair is anchored per phase.  Once an anchor exists, legal
    positions are restricted to its unoccupied four-neighbour frontier.

    Fewer than ``TILE_COUNT`` phases are supported for bounded ablations; the
    remaining relaxed free block is then projected with Hungarian assignment.
    This implementation deliberately does not translate the anchored block at
    board boundaries, so it remains less complete than the reference solver's
    translation-and-branching procedure.
    """
    if (
        not 1 <= top_k < TILE_COUNT
        or not 1 <= phases <= TILE_COUNT
        or similarity_power <= 0.0
        or convergence_threshold < 0.0
        or max_iterations <= 0
        or not 0.0 <= anchor_probability <= 1.0
    ):
        raise ValueError("invalid faithful multi-phase relaxation parameters")

    right = _adaptive_reciprocal_similarity(
        compatibility.right,
        top_k=top_k,
        power=similarity_power,
    )
    down = _adaptive_reciprocal_similarity(
        compatibility.down,
        top_k=top_k,
        power=similarity_power,
    )
    right_dense = right.toarray()
    down_dense = down.toarray()
    right_out, right_in = _directional_best_buddies(compatibility.right)
    down_out, down_in = _directional_best_buddies(compatibility.down)

    anchored_tile = np.zeros(TILE_COUNT, dtype=bool)
    anchored_position = np.zeros(TILE_COUNT, dtype=bool)
    tile_at_position = np.full(TILE_COUNT, -1, dtype=np.int32)
    position_of_tile = np.full(TILE_COUNT, -1, dtype=np.int32)
    assignment = np.full(
        (TILE_COUNT, TILE_COUNT),
        1.0 / float(TILE_COUNT),
        dtype=np.float64,
    )

    def uniform_reset() -> np.ndarray:
        reset = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float64)
        free_tiles = np.flatnonzero(~anchored_tile)
        free_positions = np.flatnonzero(~anchored_position)
        if len(free_tiles):
            reset[np.ix_(free_tiles, free_positions)] = 1.0 / float(
                len(free_positions)
            )
        fixed_tiles = np.flatnonzero(anchored_tile)
        if len(fixed_tiles):
            reset[fixed_tiles, position_of_tile[fixed_tiles]] = 1.0
        return reset

    def row_normalize(
        values: np.ndarray,
        *,
        fallback: np.ndarray | None = None,
    ) -> np.ndarray:
        result = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
        result[:, anchored_position] = 0.0
        fixed_tiles = np.flatnonzero(anchored_tile)
        if len(fixed_tiles):
            result[fixed_tiles, :] = 0.0
            result[fixed_tiles, position_of_tile[fixed_tiles]] = 1.0
        free_tiles = np.flatnonzero(~anchored_tile)
        free_positions = np.flatnonzero(~anchored_position)
        if len(free_tiles):
            block = result[np.ix_(free_tiles, free_positions)]
            sums = block.sum(axis=1, keepdims=True)
            empty = sums[:, 0] <= 1e-300
            if np.any(empty) and fallback is not None:
                fallback_block = np.maximum(
                    np.asarray(fallback, dtype=np.float64)[
                        np.ix_(free_tiles, free_positions)
                    ],
                    0.0,
                )
                block[empty, :] = fallback_block[empty, :]
                sums = block.sum(axis=1, keepdims=True)
                empty = sums[:, 0] <= 1e-300
            if np.any(empty):
                block[empty, :] = 1.0
                sums = block.sum(axis=1, keepdims=True)
            result[np.ix_(free_tiles, free_positions)] = block / sums
        return result

    positions = np.arange(TILE_COUNT, dtype=np.int32)
    right_positions = positions[positions % GRID < GRID - 1]
    down_positions = positions[positions < TILE_COUNT - GRID]

    def relaxation_support(probabilities: np.ndarray) -> np.ndarray:
        support = np.zeros_like(probabilities)
        support[:, right_positions] += right @ probabilities[:, right_positions + 1]
        support[:, right_positions + 1] += right.T @ probabilities[:, right_positions]
        support[:, down_positions] += down @ probabilities[:, down_positions + GRID]
        support[:, down_positions + GRID] += down.T @ probabilities[:, down_positions]
        return support

    def legal_frontier() -> np.ndarray:
        if not np.any(anchored_position):
            return np.flatnonzero(~anchored_position).astype(np.int32)
        frontier: set[int] = set()
        for position in np.flatnonzero(anchored_position).tolist():
            row, column = divmod(position, GRID)
            for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nr, nc = row + dr, column + dc
                if 0 <= nr < GRID and 0 <= nc < GRID:
                    neighbour = nr * GRID + nc
                    if not anchored_position[neighbour]:
                        frontier.add(neighbour)
        if not frontier:
            raise RuntimeError("anchored component has no legal frontier")
        return np.asarray(sorted(frontier), dtype=np.int32)

    def anchor_rank(position: int, free_tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        row, column = divmod(position, GRID)
        mutual = np.zeros(len(free_tiles), dtype=np.int32)
        contact = np.zeros(len(free_tiles), dtype=np.float64)
        if column > 0 and anchored_position[position - 1]:
            neighbour = int(tile_at_position[position - 1])
            mutual += (right_out[neighbour] == free_tiles) & (
                right_in[free_tiles] == neighbour
            )
            contact += right_dense[neighbour, free_tiles]
        if column + 1 < GRID and anchored_position[position + 1]:
            neighbour = int(tile_at_position[position + 1])
            mutual += (right_out[free_tiles] == neighbour) & (
                right_in[neighbour] == free_tiles
            )
            contact += right_dense[free_tiles, neighbour]
        if row > 0 and anchored_position[position - GRID]:
            neighbour = int(tile_at_position[position - GRID])
            mutual += (down_out[neighbour] == free_tiles) & (
                down_in[free_tiles] == neighbour
            )
            contact += down_dense[neighbour, free_tiles]
        if row + 1 < GRID and anchored_position[position + GRID]:
            neighbour = int(tile_at_position[position + GRID])
            mutual += (down_out[free_tiles] == neighbour) & (
                down_in[neighbour] == free_tiles
            )
            contact += down_dense[free_tiles, neighbour]
        return mutual, contact

    def threshold_anchor(free_tiles: np.ndarray) -> tuple[int, int] | None:
        """Select one legal threshold candidate using official tie-breaks."""
        free_positions = np.flatnonzero(~anchored_position).astype(np.int32)
        candidate_positions = legal_frontier()
        frontier_mask = np.zeros(TILE_COUNT, dtype=bool)
        frontier_mask[candidate_positions] = True
        free_block = assignment[np.ix_(free_tiles, free_positions)]
        maxima = np.argmax(free_block, axis=1)
        row_maxima = free_block[np.arange(len(free_tiles)), maxima]
        row_positions = free_positions[maxima]
        eligible = (row_maxima >= anchor_probability) & frontier_mask[row_positions]
        if not np.any(eligible):
            return None

        best_key: tuple[float, ...] | None = None
        best_pair: tuple[int, int] | None = None
        rank_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for local_index in np.flatnonzero(eligible).tolist():
            tile = int(free_tiles[local_index])
            position = int(row_positions[local_index])
            if position not in rank_cache:
                rank_cache[position] = anchor_rank(position, free_tiles)
            mutual, contact = rank_cache[position]
            key = (
                -float(mutual[local_index]),
                -float(contact[local_index]),
                -float(row_maxima[local_index]),
                float(position),
                float(tile),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (tile, position)
        return best_pair

    def strongest_legal_anchor(free_tiles: np.ndarray) -> tuple[int, int]:
        """Return the global highest-probability legal pair deterministically."""
        candidate_positions = legal_frontier()
        values = assignment[np.ix_(free_tiles, candidate_positions)]
        center = 0.5 * float(GRID - 1)
        best_key: tuple[float, ...] | None = None
        best_pair = (-1, -1)
        for local_tile, tile in enumerate(free_tiles.tolist()):
            for local_position, position in enumerate(candidate_positions.tolist()):
                row, column = divmod(position, GRID)
                key = (
                    -float(values[local_tile, local_position]),
                    abs(row - center) + abs(column - center),
                    float(position),
                    float(tile),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = (int(tile), int(position))
        if best_pair[0] < 0:
            raise RuntimeError("no legal faithful relaxation anchor")
        return best_pair

    for phase_index in range(phases):
        free_tiles = np.flatnonzero(~anchored_tile).astype(np.int32)
        if len(free_tiles) == 0:
            break
        support = relaxation_support(assignment)
        previous_alc = float(np.sum(assignment * support, dtype=np.float64))
        selected_anchor: tuple[int, int] | None = None
        for _ in range(max_iterations):
            updated = row_normalize(assignment * support, fallback=assignment)
            updated_support = relaxation_support(updated)
            alc = float(np.sum(updated * updated_support, dtype=np.float64))
            delta = alc - previous_alc
            numerical_tolerance = 1e-12 * max(
                1.0, abs(previous_alc), abs(alc)
            )
            if delta < -numerical_tolerance:
                raise FloatingPointError(
                    "faithful relaxation ALC decreased beyond numerical tolerance"
                )
            assignment = updated
            support = updated_support
            selected_anchor = threshold_anchor(free_tiles)
            if selected_anchor is not None:
                break
            if delta <= convergence_threshold:
                break
            previous_alc = alc

        if selected_anchor is None:
            selected_anchor = strongest_legal_anchor(free_tiles)
        best_tile, best_position = selected_anchor
        anchored_tile[best_tile] = True
        anchored_position[best_position] = True
        tile_at_position[best_position] = best_tile
        position_of_tile[best_tile] = best_position
        free_tiles_after_anchor = np.flatnonzero(~anchored_tile)
        if phase_index + 1 < phases and len(free_tiles_after_anchor):
            assignment = uniform_reset()
        else:
            # Keep the converged posterior meaningful for a bounded-phase
            # Hungarian projection while conditioning it on the last anchor.
            assignment = row_normalize(assignment, fallback=uniform_reset())

    result = tile_at_position.copy()
    free_tiles = np.flatnonzero(~anchored_tile)
    free_positions = np.flatnonzero(~anchored_position)
    if len(free_tiles):
        rows, columns = linear_sum_assignment(
            -assignment[np.ix_(free_tiles, free_positions)]
        )
        result[free_positions[columns]] = free_tiles[rows]
    return validate_permutation(
        result,
        name="faithful_multi_phase_relaxation_position_to_slot",
    )
