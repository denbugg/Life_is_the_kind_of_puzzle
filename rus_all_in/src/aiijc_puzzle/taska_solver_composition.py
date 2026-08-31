"""Small legal compositions of independently evaluated TASKA primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_layout_portfolio import (
    TaskaMultiPortfolioSelection,
    TaskaPortfolioSelection,
    select_lower_taska_seam_cost_layout,
    select_lowest_taska_seam_cost_layout,
)
from aiijc_puzzle.taska_protected_tail_polish import (
    TaskaProtectedTailResult,
    polish_unprotected_taska_tail,
)


@dataclass(frozen=True)
class TaskaPortfolioTailResult:
    """Selected layout followed by its protected-tail seam polish."""

    selection: TaskaPortfolioSelection
    polish: TaskaProtectedTailResult


@dataclass(frozen=True)
class TaskaMultiPortfolioTailResult:
    """Named multi-layout selection followed by protected-tail polish."""

    selection: TaskaMultiPortfolioSelection
    polish: TaskaProtectedTailResult


def select_then_polish_taska_layout(
    raw_layout: Any,
    calibrated_layout: Any,
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    grid: int = 24,
    max_swaps: int = 24,
    minimum_gain: float = 1e-9,
) -> TaskaPortfolioTailResult:
    """Choose by all-bond seam cost, then polish only the chosen free tail."""

    selection = select_lower_taska_seam_cost_layout(
        raw_layout,
        calibrated_layout,
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
        max_swaps=max_swaps,
        minimum_gain=minimum_gain,
    )
    return TaskaPortfolioTailResult(selection=selection, polish=polish)


def select_then_polish_taska_layouts(
    layouts: Mapping[str, Any],
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    grid: int = 24,
    max_swaps: int = 24,
    minimum_gain: float = 1e-9,
) -> TaskaMultiPortfolioTailResult:
    """Choose among named layouts by seam cost, then polish the chosen tail."""

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
        max_swaps=max_swaps,
        minimum_gain=minimum_gain,
    )
    return TaskaMultiPortfolioTailResult(selection=selection, polish=polish)


__all__ = [
    "TaskaMultiPortfolioTailResult",
    "TaskaPortfolioTailResult",
    "select_then_polish_taska_layout",
    "select_then_polish_taska_layouts",
]
