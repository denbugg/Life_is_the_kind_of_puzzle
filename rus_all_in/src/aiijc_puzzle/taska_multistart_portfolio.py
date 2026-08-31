"""Fixed deterministic multistart portfolio for the legal TASKA solver.

The TASKA component builder is deterministic, but its historical component
placement and Hungarian-tail tie breaking use ``RawTailGlobalConfig``'s
``random_seed``.  This module spends exactly four preregistered seeds on each
of the four already frozen edge-order arms, chooses the strict layout with the
lowest original TASKA cost over all 1,104 board bonds, and finally applies the
unchanged 96-swap protected-tail polish.

No target, source coordinate, filename, or recovered permutation enters the
API.  The only inputs are matcher costs, harvested current-bag edges, and the
three already inferred alternative edge-priority vectors.  Every returned
layout is a strict permutation of the original upright tiles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import (
    RawTailEdge,
    RawTailGlobalConfig,
    solve_raw_tail_global,
)
from aiijc_puzzle.taska_edge_calibrator import solve_prioritized_raw_tail_global
from aiijc_puzzle.taska_layout_portfolio import (
    TaskaMultiPortfolioSelection,
    select_lowest_taska_seam_cost_layout,
)
from aiijc_puzzle.taska_protected_tail_polish import (
    TaskaProtectedTailResult,
    polish_unprotected_taska_tail,
)

TASKA_MULTISTART_SEEDS = (0, 1, 2, 3)
TASKA_MULTISTART_ARMS = ("raw", "logistic", "focal", "nonlinear")
TASKA_PRIORITY_ARMS = TASKA_MULTISTART_ARMS[1:]
TASKA_MULTISTART_TAIL_SWAPS = 96


@dataclass(frozen=True)
class TaskaMultistartPortfolioResult:
    """All 16 strict starts, their target-free selection, and tail polish."""

    layouts: tuple[tuple[str, np.ndarray], ...]
    selection: TaskaMultiPortfolioSelection
    polish: TaskaProtectedTailResult


def _validated_priorities(
    values: Mapping[str, Any],
    *,
    edge_count: int,
) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping) or tuple(values) != TASKA_PRIORITY_ARMS:
        raise ValueError(
            "edge_priorities must contain logistic, focal, nonlinear in that order"
        )
    result: dict[str, np.ndarray] = {}
    for name in TASKA_PRIORITY_ARMS:
        vector = np.asarray(values[name], dtype=np.float64)
        if vector.shape != (edge_count,):
            raise ValueError(
                f"edge_priorities[{name!r}] must have shape {(edge_count,)}, "
                f"got {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"edge_priorities[{name!r}] must contain only finite values")
        result[name] = np.ascontiguousarray(vector)
    return result


def solve_taska_multistart_portfolio(
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    edge_priorities: Mapping[str, Any],
    *,
    grid: int = 24,
    solver_config: RawTailGlobalConfig | None = None,
    minimum_gain: float = 1e-9,
) -> TaskaMultistartPortfolioResult:
    """Solve the fixed 4-arm x 4-seed portfolio and protected 96-swap tail.

    ``solver_config.random_seed`` is intentionally ignored: the only seed set
    evaluated here is the module constant ``(0, 1, 2, 3)``.  Every other
    solver field is retained verbatim.
    """

    edges = tuple(candidate_edges)
    priorities = _validated_priorities(edge_priorities, edge_count=len(edges))
    base = solver_config or RawTailGlobalConfig(
        baseline_quantile=0.15,
        search_rounds=6,
        border_weight=0.0,
        random_seed=0,
        component_cap=0,
        fill_rounds=1,
    )
    base.validate(grid=grid)

    layouts: dict[str, np.ndarray] = {}
    # Seed-major order preserves the established four-arm seed-0 tie order,
    # then adds the three preregistered alternative starts without tuning.
    for seed in TASKA_MULTISTART_SEEDS:
        config = replace(base, random_seed=seed)
        raw = solve_raw_tail_global(
            cost_right,
            cost_down,
            edges,
            grid=grid,
            config=config,
        )
        layouts[f"raw_seed{seed}"] = raw.layout
        for arm in TASKA_PRIORITY_ARMS:
            solved = solve_prioritized_raw_tail_global(
                cost_right,
                cost_down,
                edges,
                priorities[arm],
                grid=grid,
                config=config,
            )
            layouts[f"{arm}_seed{seed}"] = solved.layout

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
        edges,
        grid=grid,
        max_swaps=TASKA_MULTISTART_TAIL_SWAPS,
        minimum_gain=minimum_gain,
    )
    frozen_layouts: list[tuple[str, np.ndarray]] = []
    for name, layout in layouts.items():
        frozen = np.asarray(layout, dtype=np.int32).copy()
        frozen.setflags(write=False)
        frozen_layouts.append((name, frozen))
    return TaskaMultistartPortfolioResult(
        layouts=tuple(frozen_layouts),
        selection=selection,
        polish=polish,
    )


__all__ = [
    "TASKA_MULTISTART_ARMS",
    "TASKA_MULTISTART_SEEDS",
    "TASKA_MULTISTART_TAIL_SWAPS",
    "TASKA_PRIORITY_ARMS",
    "TaskaMultistartPortfolioResult",
    "solve_taska_multistart_portfolio",
]
