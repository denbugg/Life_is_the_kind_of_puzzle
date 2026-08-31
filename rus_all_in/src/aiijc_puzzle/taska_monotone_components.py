"""Legal TASKA component placement with only monotone coordinate relocations.

The frozen historical raw-tail solver first places translation-consistent
components largest-first, then alternates two operations: coordinate-wise
single-component best relocation and unconditional random two-component
relocation.  The latter is committed whenever it is feasible, even when the
placement objective becomes worse.

This experimental module changes exactly that placement dynamic: it preserves
the initial largest-first placement and the coordinate-wise best relocations,
but omits the unconditional two-component loop.  Component construction,
original right/down costs, baseline, Hungarian fill, four-arm all-bond
selection, and protected tail are otherwise unchanged.  The frozen historical
solver is imported read-only and is never patched or modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import (
    RawTailBuildDecision,
    RawTailEdge,
    RawTailGlobalConfig,
    RawTailGlobalDiagnostics,
    RawTailGlobalResult,
    _as_finite_matrix,
    _contact_score,
    _fill_seams,
    _normalised_cells,
    _validate_grid,
    build_raw_tail_components,
)
from aiijc_puzzle.taska_edge_calibrator import (
    PrioritizedRawTailBuildDecision,
    PrioritizedRawTailResult,
    build_prioritized_raw_tail_components,
)
from aiijc_puzzle.taska_layout_portfolio import (
    TaskaMultiPortfolioSelection,
    select_lowest_taska_seam_cost_layout,
)
from aiijc_puzzle.taska_protected_tail_polish import (
    TaskaProtectedTailResult,
    polish_unprotected_taska_tail,
)

MONOTONE_ARM_NAMES = ("raw", "logistic", "focal", "nonlinear")
MONOTONE_TAIL_MAX_SWAPS = 96
MONOTONE_TAIL_MINIMUM_GAIN = 1e-9


@dataclass(frozen=True)
class MonotonePlacementTrace:
    """Target-free evidence for the coordinate-only placement search."""

    moved_components_per_round: tuple[int, ...]
    pair_relocation_attempts: int = 0


@dataclass(frozen=True)
class MonotoneRawTailResult:
    """Raw-order result plus the coordinate-only placement trace."""

    solver: RawTailGlobalResult
    placement: MonotonePlacementTrace


@dataclass(frozen=True)
class MonotonePrioritizedRawTailResult:
    """Externally prioritized result plus the coordinate-only placement trace."""

    solver: PrioritizedRawTailResult
    placement: MonotonePlacementTrace


@dataclass(frozen=True)
class TaskaMonotoneComponentPortfolioResult:
    """Four monotone-placement arms, all-bond selection, and protected tail."""

    layouts: tuple[tuple[str, np.ndarray], ...]
    placement_traces: tuple[tuple[str, MonotonePlacementTrace], ...]
    selection: TaskaMultiPortfolioSelection
    polish: TaskaProtectedTailResult


def _validated_unary(
    value: Any | None,
    *,
    count: int,
    grid: int,
) -> np.ndarray | None:
    if value is None:
        return None
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    unary = np.asarray(current, dtype=np.float64)
    if unary.shape != (count, grid, grid):
        raise ValueError(
            f"border_unary must have shape {(count, grid, grid)}, got {unary.shape}"
        )
    if not np.isfinite(unary).all():
        raise ValueError("border_unary must contain only finite values")
    return np.ascontiguousarray(unary)


def _place_components_coordinate_only(
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
) -> tuple[np.ndarray, int, int, float, MonotonePlacementTrace]:
    """Replay historical placement without its unconditional pair-relocation loop."""

    generator = np.random.default_rng(seed)
    baseline = 0.5 * (
        float(np.quantile(cost_right, baseline_quantile))
        + float(np.quantile(cost_down, baseline_quantile))
    )
    movable = [component for component in components if len(component) > 1]
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
                # Preserve the historical strict comparison and row-major tie rule.
                if score > best_score:
                    best, best_score = (row, column), score
        return best

    # Historical largest-first initialization, unchanged.
    for index in range(len(movable)):
        position = best_position(index)
        if position is not None:
            put(index, *position)

    moved_per_round: list[int] = []
    for _ in range(rounds):
        moved_count = 0
        for raw_index in generator.permutation(len(movable)):
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
            moved_count += int(position != old)
        moved_per_round.append(moved_count)
        if moved_count == 0:
            break

    placed_count = sum(position is not None for position in positions)
    placed_tiles = int(np.count_nonzero(board >= 0))
    return (
        board,
        placed_count,
        placed_tiles,
        baseline,
        MonotonePlacementTrace(tuple(moved_per_round)),
    )


def _diagnostics(
    decisions: Sequence[RawTailBuildDecision | PrioritizedRawTailBuildDecision],
    *,
    grid: int,
    component_sizes: Sequence[int],
    placed_count: int,
    placed_tiles: int,
    baseline: float,
) -> RawTailGlobalDiagnostics:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.status] = counts.get(decision.status, 0) + 1
    accepted = sum(value for key, value in counts.items() if key.startswith("accepted_"))
    return RawTailGlobalDiagnostics(
        grid_size=grid,
        tile_count=grid * grid,
        candidate_edges=len(decisions),
        accepted_edges=accepted,
        rejected_edges=len(decisions) - accepted,
        component_count=len(component_sizes),
        component_sizes=tuple(sorted(component_sizes, reverse=True)),
        placed_component_count=placed_count,
        placed_component_tiles=placed_tiles,
        baseline_cost=baseline,
        strict_permutation=True,
        status_counts=tuple(sorted(counts.items())),
    )


def _assemble(
    components: tuple[dict[int, tuple[int, int]], ...],
    decisions: Sequence[RawTailBuildDecision | PrioritizedRawTailBuildDecision],
    right: np.ndarray,
    down: np.ndarray,
    unary: np.ndarray | None,
    *,
    grid: int,
    config: RawTailGlobalConfig,
) -> tuple[np.ndarray, RawTailGlobalDiagnostics, MonotonePlacementTrace]:
    board, placed_count, placed_tiles, baseline, trace = (
        _place_components_coordinate_only(
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
    )
    layout = _fill_seams(
        board,
        right,
        down,
        grid=grid,
        seed=config.random_seed,
        rounds=config.fill_rounds,
    )
    count = grid * grid
    if not np.array_equal(np.sort(layout), np.arange(count)):
        raise RuntimeError("coordinate-only solver did not return a strict permutation")
    frozen = np.ascontiguousarray(layout, dtype=np.int32)
    frozen.setflags(write=False)
    diagnostics = _diagnostics(
        decisions,
        grid=grid,
        component_sizes=[len(component) for component in components],
        placed_count=placed_count,
        placed_tiles=placed_tiles,
        baseline=baseline,
    )
    return frozen, diagnostics, trace


def solve_monotone_raw_tail_global(
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    border_unary: Any | None = None,
    grid: int = 24,
    config: RawTailGlobalConfig | None = None,
) -> MonotoneRawTailResult:
    """Solve the raw-priority arm using coordinate-only component placement."""

    if config is None:
        config = RawTailGlobalConfig()
    count = _validate_grid(grid)
    config.validate(grid=grid)
    right = _as_finite_matrix(cost_right, count=count, name="cost_right")
    down = _as_finite_matrix(cost_down, count=count, name="cost_down")
    unary = _validated_unary(border_unary, count=count, grid=grid)
    components, decisions = build_raw_tail_components(
        right,
        down,
        candidate_edges,
        grid=grid,
        component_cap=config.component_cap,
    )
    layout, diagnostics, trace = _assemble(
        components,
        decisions,
        right,
        down,
        unary,
        grid=grid,
        config=config,
    )
    return MonotoneRawTailResult(
        solver=RawTailGlobalResult(
            layout=layout,
            components=components,
            decisions=decisions,
            diagnostics=diagnostics,
        ),
        placement=trace,
    )


def solve_monotone_prioritized_raw_tail_global(
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    edge_priorities: Any,
    *,
    border_unary: Any | None = None,
    grid: int = 24,
    config: RawTailGlobalConfig | None = None,
) -> MonotonePrioritizedRawTailResult:
    """Solve one externally prioritized arm with coordinate-only placement."""

    if config is None:
        config = RawTailGlobalConfig()
    count = _validate_grid(grid)
    config.validate(grid=grid)
    right = _as_finite_matrix(cost_right, count=count, name="cost_right")
    down = _as_finite_matrix(cost_down, count=count, name="cost_down")
    unary = _validated_unary(border_unary, count=count, grid=grid)
    components, decisions = build_prioritized_raw_tail_components(
        right,
        down,
        candidate_edges,
        edge_priorities,
        grid=grid,
        component_cap=config.component_cap,
    )
    layout, diagnostics, trace = _assemble(
        components,
        decisions,
        right,
        down,
        unary,
        grid=grid,
        config=config,
    )
    return MonotonePrioritizedRawTailResult(
        solver=PrioritizedRawTailResult(
            layout=layout,
            components=components,
            decisions=decisions,
            diagnostics=diagnostics,
        ),
        placement=trace,
    )


def solve_taska_monotone_component_portfolio(
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    edge_priorities: Mapping[str, Any],
    *,
    grid: int = 24,
    solver_config: RawTailGlobalConfig | None = None,
) -> TaskaMonotoneComponentPortfolioResult:
    """Run the fixed four arms, all-bond selector, and protected tail96."""

    if tuple(edge_priorities) != MONOTONE_ARM_NAMES[1:]:
        raise ValueError(
            "edge_priorities must contain logistic, focal, nonlinear in that order"
        )
    config = RawTailGlobalConfig() if solver_config is None else solver_config
    # The registered experiment is seed-0.  Ignore no caller state implicitly:
    # normalize only through an explicit immutable replacement.
    config = replace(config, random_seed=0)
    raw = solve_monotone_raw_tail_global(
        cost_right,
        cost_down,
        candidate_edges,
        grid=grid,
        config=config,
    )
    prioritized = tuple(
        (
            name,
            solve_monotone_prioritized_raw_tail_global(
                cost_right,
                cost_down,
                candidate_edges,
                edge_priorities[name],
                grid=grid,
                config=config,
            ),
        )
        for name in MONOTONE_ARM_NAMES[1:]
    )
    layouts = {
        "raw": raw.solver.layout,
        **{name: result.solver.layout for name, result in prioritized},
    }
    selection = select_lowest_taska_seam_cost_layout(
        layouts,
        cost_right,
        cost_down,
        grid=grid,
    )
    polish = polish_unprotected_taska_tail(
        selection.layout,
        cost_right,
        cost_down,
        candidate_edges,
        grid=grid,
        max_swaps=MONOTONE_TAIL_MAX_SWAPS,
        minimum_gain=MONOTONE_TAIL_MINIMUM_GAIN,
    )
    frozen_layouts: list[tuple[str, np.ndarray]] = []
    for name, layout in layouts.items():
        frozen = np.ascontiguousarray(layout, dtype=np.int32).copy()
        frozen.setflags(write=False)
        frozen_layouts.append((name, frozen))
    traces = (
        ("raw", raw.placement),
        *((name, result.placement) for name, result in prioritized),
    )
    return TaskaMonotoneComponentPortfolioResult(
        layouts=tuple(frozen_layouts),
        placement_traces=traces,
        selection=selection,
        polish=polish,
    )


__all__ = [
    "MONOTONE_ARM_NAMES",
    "MONOTONE_TAIL_MAX_SWAPS",
    "MONOTONE_TAIL_MINIMUM_GAIN",
    "MonotonePlacementTrace",
    "MonotonePrioritizedRawTailResult",
    "MonotoneRawTailResult",
    "TaskaMonotoneComponentPortfolioResult",
    "solve_monotone_prioritized_raw_tail_global",
    "solve_monotone_raw_tail_global",
    "solve_taska_monotone_component_portfolio",
]
