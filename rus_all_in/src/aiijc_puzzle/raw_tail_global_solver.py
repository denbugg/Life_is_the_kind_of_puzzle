"""Target-free discrete tail of the historical composed global solver.

The source implementation at ``ae9d231`` first produced learned right/down
cost matrices and a learned candidate-edge harvest.  Those stages require
several unavailable checkpoints.  Everything after that boundary is a
portable, pixel-free solver, implemented here:

1. order the supplied candidate edges by their raw fused score ``-cost``;
2. greedily grow translation-consistent, collision-free, non-wrapping pieces;
3. place the multi-tile pieces by the historical frame-aware search;
4. assign every unowned tile to every empty cell in one Hungarian seam solve.

No target, recovered permutation, filename, or source-grid coordinate enters
the API.  ``border_unary``, when supplied, must likewise have been inferred
from the current input board.  The function only returns a permutation; it
never alters, replaces, stretches, or renders a fragment.

Two historical details matter when comparing outputs.  With the shipping
``verify_hinge.pt`` verifier enabled, ``infer_composed.py`` did *not* use raw
ordering: verifier logits replaced it.  Also, its ``np.argsort`` quicksort did
not specify a tie rule.  This port makes raw-score ties stable in caller order;
all non-tied decisions are identical to the historical raw tail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

Axis = Literal["right", "down"]


def _validate_grid(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid * grid


def _as_finite_matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    matrix = np.asarray(current, dtype=np.float64)
    if matrix.shape != (count, count):
        raise ValueError(f"{name} must have shape {(count, count)}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


@dataclass(frozen=True)
class RawTailEdge:
    """One candidate directed neighbour relation.

    Candidate membership is deliberately external: the historical learned
    chooser/verifier cannot be reconstructed without its checkpoints.  Tile
    ids identify the current input bag only; they must not encode target cells.
    """

    source: int
    target: int
    axis: Axis


@dataclass(frozen=True)
class RawTailGlobalConfig:
    """Fixed historical placement/fill settings, generalized to any grid."""

    baseline_quantile: float = 0.15
    search_rounds: int = 6
    border_weight: float = 1.0
    random_seed: int = 0
    component_cap: int = 0
    fill_rounds: int = 1

    def validate(self, *, grid: int) -> None:
        count = _validate_grid(grid)
        if not np.isfinite(self.baseline_quantile) or not (
            0.0 <= self.baseline_quantile <= 1.0
        ):
            raise ValueError("baseline_quantile must be finite and in [0, 1]")
        if (
            isinstance(self.search_rounds, bool)
            or not isinstance(self.search_rounds, int)
            or self.search_rounds < 0
        ):
            raise ValueError("search_rounds must be a non-negative integer")
        if not np.isfinite(self.border_weight) or self.border_weight < 0:
            raise ValueError("border_weight must be finite and non-negative")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if (
            isinstance(self.component_cap, bool)
            or not isinstance(self.component_cap, int)
            or self.component_cap < 0
            or self.component_cap == 1
            or self.component_cap > count
        ):
            raise ValueError("component_cap must be zero or an integer in [2, grid**2]")
        if (
            isinstance(self.fill_rounds, bool)
            or not isinstance(self.fill_rounds, int)
            or self.fill_rounds < 1
        ):
            raise ValueError("fill_rounds must be a positive integer")


@dataclass(frozen=True)
class RawTailBuildDecision:
    """Auditable result of attempting one edge in raw-priority order."""

    edge: RawTailEdge
    raw_priority: float
    input_rank: int
    status: str


@dataclass(frozen=True)
class RawTailGlobalDiagnostics:
    """Compact, target-free diagnostics for one solver invocation."""

    grid_size: int
    tile_count: int
    candidate_edges: int
    accepted_edges: int
    rejected_edges: int
    component_count: int
    component_sizes: tuple[int, ...]
    placed_component_count: int
    placed_component_tiles: int
    baseline_cost: float
    strict_permutation: bool
    status_counts: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawTailGlobalResult:
    """Strict tile-at-position permutation plus component/build evidence."""

    layout: np.ndarray
    components: tuple[dict[int, tuple[int, int]], ...]
    decisions: tuple[RawTailBuildDecision, ...]
    diagnostics: RawTailGlobalDiagnostics


class _TranslationBuilder:
    def __init__(self, *, grid: int, cap: int) -> None:
        self.grid = grid
        self.cap = cap
        self.fragment_component: dict[int, int] = {}
        self._components: list[dict[int, tuple[int, int]]] = []

    def _span_ok(self, component: dict[int, tuple[int, int]]) -> bool:
        rows, columns = zip(*component.values(), strict=True)
        return max(rows) - min(rows) < self.grid and max(columns) - min(columns) < self.grid

    def add(self, edge: RawTailEdge) -> str:
        delta = (0, 1) if edge.axis == "right" else (1, 0)
        source_component = self.fragment_component.get(edge.source)
        target_component = self.fragment_component.get(edge.target)

        if source_component is None and target_component is None:
            index = len(self._components)
            self._components.append({edge.source: (0, 0), edge.target: delta})
            self.fragment_component[edge.source] = index
            self.fragment_component[edge.target] = index
            return "accepted_new"

        if source_component is not None and target_component is None:
            row, column = self._components[source_component][edge.source]
            return self._add_fragment(
                source_component,
                edge.target,
                (row + delta[0], column + delta[1]),
            )

        if source_component is None and target_component is not None:
            row, column = self._components[target_component][edge.target]
            return self._add_fragment(
                target_component,
                edge.source,
                (row - delta[0], column - delta[1]),
            )

        if source_component == target_component:
            component = self._components[int(source_component)]
            source_row, source_column = component[edge.source]
            target_row, target_column = component[edge.target]
            if (
                target_row - source_row,
                target_column - source_column,
            ) == delta:
                return "accepted_consistent"
            return "rejected_contradiction"

        source_index = int(source_component)
        target_index = int(target_component)
        source_row, source_column = self._components[source_index][edge.source]
        target_row, target_column = self._components[target_index][edge.target]
        shift = (
            source_row + delta[0] - target_row,
            source_column + delta[1] - target_column,
        )
        return self._merge(source_index, target_index, shift)

    def _add_fragment(
        self,
        component_index: int,
        fragment: int,
        coordinate: tuple[int, int],
    ) -> str:
        component = self._components[component_index]
        if self.cap and len(component) + 1 > self.cap:
            return "rejected_cap"
        if coordinate in component.values():
            return "rejected_collision"
        component[fragment] = coordinate
        if not self._span_ok(component):
            del component[fragment]
            return "rejected_span"
        self.fragment_component[fragment] = component_index
        return "accepted_extend"

    def _merge(
        self,
        source_index: int,
        target_index: int,
        shift: tuple[int, int],
    ) -> str:
        source = self._components[source_index]
        target = self._components[target_index]
        if self.cap and len(source) + len(target) > self.cap:
            return "rejected_cap"
        moved = {
            fragment: (row + shift[0], column + shift[1])
            for fragment, (row, column) in target.items()
        }
        occupied = set(source.values())
        if any(coordinate in occupied for coordinate in moved.values()):
            return "rejected_collision"
        merged = dict(source)
        merged.update(moved)
        if not self._span_ok(merged):
            return "rejected_span"
        self._components[source_index] = merged
        self._components[target_index] = {}
        for fragment in moved:
            self.fragment_component[fragment] = source_index
        return "accepted_merge"

    def components(self) -> tuple[dict[int, tuple[int, int]], ...]:
        return tuple(dict(component) for component in self._components if component)


def _validated_edges(
    edges: Sequence[RawTailEdge],
    *,
    count: int,
) -> tuple[RawTailEdge, ...]:
    result: list[RawTailEdge] = []
    seen: set[tuple[int, int, str]] = set()
    for index, value in enumerate(edges):
        if not isinstance(value, RawTailEdge):
            raise TypeError(f"candidate_edges[{index}] must be a RawTailEdge")
        if value.axis not in {"right", "down"}:
            raise ValueError(f"candidate_edges[{index}].axis must be 'right' or 'down'")
        if isinstance(value.source, bool) or not isinstance(value.source, int):
            raise ValueError(f"candidate_edges[{index}].source must be an integer")
        if isinstance(value.target, bool) or not isinstance(value.target, int):
            raise ValueError(f"candidate_edges[{index}].target must be an integer")
        if not 0 <= value.source < count or not 0 <= value.target < count:
            raise ValueError(f"candidate_edges[{index}] tile ids are outside the input bag")
        if value.source == value.target:
            raise ValueError(f"candidate_edges[{index}] is a self-edge")
        key = (value.source, value.target, value.axis)
        if key in seen:
            raise ValueError(f"candidate_edges[{index}] duplicates an earlier edge")
        seen.add(key)
        result.append(value)
    return tuple(result)


def _raw_priority(edge: RawTailEdge, right: np.ndarray, down: np.ndarray) -> float:
    matrix = right if edge.axis == "right" else down
    return -float(matrix[edge.source, edge.target])


def build_raw_tail_components(
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    grid: int = 24,
    component_cap: int = 0,
) -> tuple[
    tuple[dict[int, tuple[int, int]], ...],
    tuple[RawTailBuildDecision, ...],
]:
    """Build components in raw fused-score order.

    Exact raw-score ties are resolved by the caller's candidate order.  This is
    explicit and stable, unlike the historical default-quicksort tie behavior.
    """

    count = _validate_grid(grid)
    right = _as_finite_matrix(cost_right, count=count, name="cost_right")
    down = _as_finite_matrix(cost_down, count=count, name="cost_down")
    edges = _validated_edges(candidate_edges, count=count)
    if (
        isinstance(component_cap, bool)
        or not isinstance(component_cap, int)
        or component_cap < 0
        or component_cap == 1
        or component_cap > count
    ):
        raise ValueError("component_cap must be zero or an integer in [2, grid**2]")

    ranked = sorted(
        enumerate(edges),
        key=lambda item: (-_raw_priority(item[1], right, down), item[0]),
    )
    builder = _TranslationBuilder(grid=grid, cap=component_cap)
    decisions: list[RawTailBuildDecision] = []
    for input_rank, edge in ranked:
        priority = _raw_priority(edge, right, down)
        decisions.append(
            RawTailBuildDecision(
                edge=edge,
                raw_priority=priority,
                input_rank=input_rank,
                status=builder.add(edge),
            )
        )
    return builder.components(), tuple(decisions)


def _normalised_cells(
    component: dict[int, tuple[int, int]],
) -> tuple[dict[tuple[int, int], int], int, int]:
    rows, columns = zip(*component.values(), strict=True)
    origin_row, origin_column = min(rows), min(columns)
    cells = {
        (row - origin_row, column - origin_column): int(tile)
        for tile, (row, column) in component.items()
    }
    return cells, max(rows) - origin_row + 1, max(columns) - origin_column + 1


def _contact_score(
    cells: dict[tuple[int, int], int],
    origin_row: int,
    origin_column: int,
    board: np.ndarray,
    cost_right: np.ndarray,
    cost_down: np.ndarray,
    baseline: float,
    border_unary: np.ndarray | None,
    border_weight: float,
) -> float:
    grid = board.shape[0]
    total = 0.0
    if border_unary is not None and border_weight:
        for (delta_row, delta_column), tile in cells.items():
            total += border_weight * border_unary[
                tile,
                origin_row + delta_row,
                origin_column + delta_column,
            ]
    for (delta_row, delta_column), tile in cells.items():
        row, column = origin_row + delta_row, origin_column + delta_column
        for neighbour_row, neighbour_column, matrix, forward in (
            (row, column - 1, cost_right, False),
            (row, column + 1, cost_right, True),
            (row - 1, column, cost_down, False),
            (row + 1, column, cost_down, True),
        ):
            if not (0 <= neighbour_row < grid and 0 <= neighbour_column < grid):
                continue
            neighbour = int(board[neighbour_row, neighbour_column])
            if neighbour < 0:
                continue
            cost = matrix[tile, neighbour] if forward else matrix[neighbour, tile]
            total += baseline - float(cost)
    return total


def _place_components(
    components: tuple[dict[int, tuple[int, int]], ...],
    cost_right: np.ndarray,
    cost_down: np.ndarray,
    *,
    grid: int,
    baseline_quantile: float,
    rounds: int,
    seed: int,
    border_unary: np.ndarray | None,
    border_weight: float,
) -> tuple[np.ndarray, int, int, float]:
    rng = np.random.default_rng(seed)
    baseline = 0.5 * (
        float(np.quantile(cost_right, baseline_quantile))
        + float(np.quantile(cost_down, baseline_quantile))
    )
    movable = [component for component in components if len(component) > 1]
    # Python's sort is stable: equal-size components retain builder order.
    movable.sort(key=len, reverse=True)
    shapes = [_normalised_cells(component) for component in movable]
    board = np.full((grid, grid), -1, dtype=np.int64)
    positions: list[tuple[int, int] | None] = [None] * len(movable)

    def put(index: int, row: int, column: int) -> None:
        for (delta_row, delta_column), tile in shapes[index][0].items():
            board[row + delta_row, column + delta_column] = tile
        positions[index] = (row, column)

    def lift(index: int) -> None:
        position = positions[index]
        if position is None:
            return
        row, column = position
        for delta_row, delta_column in shapes[index][0]:
            board[row + delta_row, column + delta_column] = -1
        positions[index] = None

    def best_position(index: int) -> tuple[int, int] | None:
        cells, height, width = shapes[index]
        best: tuple[int, int] | None = None
        best_score = -np.inf
        # Strict `>` plus row-major traversal reproduces the historical tie rule.
        for row in range(grid - height + 1):
            for column in range(grid - width + 1):
                if any(
                    board[row + delta_row, column + delta_column] >= 0
                    for delta_row, delta_column in cells
                ):
                    continue
                score = _contact_score(
                    cells,
                    row,
                    column,
                    board,
                    cost_right,
                    cost_down,
                    baseline,
                    border_unary,
                    border_weight,
                )
                if score > best_score:
                    best, best_score = (row, column), score
        return best

    for index in range(len(movable)):
        position = best_position(index)
        if position is not None:
            put(index, *position)

    for _ in range(rounds):
        moved = False
        for raw_index in rng.permutation(len(movable)):
            index = int(raw_index)
            if positions[index] is None:
                continue
            old = positions[index]
            lift(index)
            position = best_position(index)
            if position is None:
                put(index, *old)
                continue
            put(index, *position)
            moved |= position != old

        # This is historical behavior, despite the name "descent": a feasible
        # pair relocation is committed even when the total objective decreases.
        for _ in range(len(movable) if len(movable) > 1 else 0):
            first, second = (
                int(value) for value in rng.choice(len(movable), 2, replace=False)
            )
            if positions[first] is None or positions[second] is None:
                continue
            old_first, old_second = positions[first], positions[second]
            lift(first)
            lift(second)
            first_position = best_position(first)
            if first_position is None:
                put(first, *old_first)
                put(second, *old_second)
                continue
            put(first, *first_position)
            second_position = best_position(second)
            if second_position is None:
                lift(first)
                put(first, *old_first)
                put(second, *old_second)
                continue
            put(second, *second_position)
            moved |= (first_position, second_position) != (old_first, old_second)
        if not moved:
            break

    placed_count = sum(position is not None for position in positions)
    placed_tiles = int(np.count_nonzero(board >= 0))
    return board, placed_count, placed_tiles, baseline


def _seam_assignment(
    layout: np.ndarray,
    free_cells: np.ndarray,
    unused_tiles: np.ndarray,
    cost_right: np.ndarray,
    cost_down: np.ndarray,
    *,
    grid: int,
    context: np.ndarray | None = None,
) -> np.ndarray:
    reference = layout if context is None else context
    assignment_cost = np.zeros((len(free_cells), len(unused_tiles)), dtype=np.float64)
    for index, cell in enumerate(free_cells):
        row, column = divmod(int(cell), grid)
        for neighbour_row, neighbour_column, matrix, forward in (
            (row, column - 1, cost_right, False),
            (row, column + 1, cost_right, True),
            (row - 1, column, cost_down, False),
            (row + 1, column, cost_down, True),
        ):
            if not (0 <= neighbour_row < grid and 0 <= neighbour_column < grid):
                continue
            neighbour = int(reference[neighbour_row * grid + neighbour_column])
            if neighbour >= 0:
                assignment_cost[index] += (
                    matrix[unused_tiles, neighbour]
                    if forward
                    else matrix[neighbour, unused_tiles]
                )
    rows, columns = linear_sum_assignment(assignment_cost)
    layout[free_cells[rows]] = unused_tiles[columns]
    return layout


def _fill_seams(
    board: np.ndarray,
    cost_right: np.ndarray,
    cost_down: np.ndarray,
    *,
    grid: int,
    seed: int,
    rounds: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    layout = board.reshape(-1).copy()
    free_cells = np.flatnonzero(layout < 0)
    unused_tiles = np.setdiff1d(
        np.arange(grid * grid, dtype=np.int64),
        layout[layout >= 0],
    )
    if len(free_cells) != len(unused_tiles):
        raise RuntimeError("partial board is not a one-to-one placement")
    if not len(free_cells):
        return layout
    # The source deliberately shuffled columns before Hungarian.  Otherwise an
    # all-zero row is resolved by tile id, which leaked canonical validation ids.
    unused_tiles = unused_tiles[rng.permutation(len(unused_tiles))]
    layout = _seam_assignment(
        layout,
        free_cells,
        unused_tiles,
        cost_right,
        cost_down,
        grid=grid,
    )
    for _ in range(rounds - 1):
        layout = _seam_assignment(
            layout,
            free_cells,
            unused_tiles,
            cost_right,
            cost_down,
            grid=grid,
            context=layout.copy(),
        )
    return layout


def solve_raw_tail_global(
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    border_unary: Any | None = None,
    grid: int = 24,
    config: RawTailGlobalConfig | None = None,
) -> RawTailGlobalResult:
    """Run the portable raw-tail assembly and return a strict permutation.

    ``cost_right[a, b]`` is the cost of placing ``b`` immediately right of
    ``a``; ``cost_down[a, b]`` is analogous vertically.  Lower is better.
    ``candidate_edges`` is the target-free harvest to spend on rigid component
    construction.  ``border_unary[tile, row, column]`` is an optional additive
    placement bonus inferred from the input; larger is better.
    """

    if config is None:
        config = RawTailGlobalConfig()
    count = _validate_grid(grid)
    config.validate(grid=grid)
    right = _as_finite_matrix(cost_right, count=count, name="cost_right")
    down = _as_finite_matrix(cost_down, count=count, name="cost_down")
    edges = _validated_edges(candidate_edges, count=count)
    unary: np.ndarray | None = None
    if border_unary is not None:
        unary = np.asarray(border_unary, dtype=np.float64)
        if unary.shape != (count, grid, grid):
            raise ValueError(
                f"border_unary must have shape {(count, grid, grid)}, got {unary.shape}"
            )
        if not np.isfinite(unary).all():
            raise ValueError("border_unary must contain only finite values")
        unary = np.ascontiguousarray(unary)

    components, decisions = build_raw_tail_components(
        right,
        down,
        edges,
        grid=grid,
        component_cap=config.component_cap,
    )
    board, placed_count, placed_tiles, baseline = _place_components(
        components,
        right,
        down,
        grid=grid,
        baseline_quantile=config.baseline_quantile,
        rounds=config.search_rounds,
        seed=config.random_seed,
        border_unary=unary,
        border_weight=config.border_weight,
    )
    layout = _fill_seams(
        board,
        right,
        down,
        grid=grid,
        seed=config.random_seed,
        rounds=config.fill_rounds,
    )
    strict = np.array_equal(np.sort(layout), np.arange(count))
    if not strict:
        raise RuntimeError("raw-tail global solver did not return a strict permutation")
    layout = np.asarray(layout, dtype=np.int32)
    layout.setflags(write=False)

    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.status] = counts.get(decision.status, 0) + 1
    accepted = sum(value for key, value in counts.items() if key.startswith("accepted_"))
    diagnostics = RawTailGlobalDiagnostics(
        grid_size=grid,
        tile_count=count,
        candidate_edges=len(edges),
        accepted_edges=accepted,
        rejected_edges=len(edges) - accepted,
        component_count=len(components),
        component_sizes=tuple(sorted((len(component) for component in components), reverse=True)),
        placed_component_count=placed_count,
        placed_component_tiles=placed_tiles,
        baseline_cost=baseline,
        strict_permutation=strict,
        status_counts=tuple(sorted(counts.items())),
    )
    return RawTailGlobalResult(
        layout=layout,
        components=components,
        decisions=decisions,
        diagnostics=diagnostics,
    )
