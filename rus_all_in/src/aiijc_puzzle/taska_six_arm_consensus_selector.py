"""Rejected whole-layout consensus selector for TASKA research.

The selector is intentionally small and target blind: each strict layout is
represented by its 1,104 directed right/down adjacencies, and the selected arm
maximises total overlap with the other arms.  A supplied fallback layout wins
an exact score tie when it is one of the tied arms.

This primitive is retained only to make the negative experiment reproducible;
it is not part of the production solver.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Axis = Literal["right", "down"]
DirectedAdjacency = tuple[Axis, int, int]


def _strict_layout(value: Any, *, grid: int, name: str) -> np.ndarray:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    count = grid * grid
    layout = np.ascontiguousarray(value, dtype=np.int32)
    if layout.shape != (count,) or not np.array_equal(np.sort(layout), np.arange(count)):
        raise ValueError(f"{name} must be a strict {count}-tile permutation")
    return layout


def directed_adjacencies(value: Any, *, grid: int = 24) -> frozenset[DirectedAdjacency]:
    """Return all directed horizontal and vertical relations in one layout."""

    board = _strict_layout(value, grid=grid, name="layout").reshape(grid, grid)
    edges: set[DirectedAdjacency] = {
        *(
            ("right", int(board[row, column]), int(board[row, column + 1]))
            for row in range(grid)
            for column in range(grid - 1)
        ),
        *(
            ("down", int(board[row, column]), int(board[row + 1, column]))
            for row in range(grid - 1)
            for column in range(grid)
        ),
    }
    expected = 2 * grid * (grid - 1)
    if len(edges) != expected:
        raise RuntimeError("layout adjacency count changed")
    return frozenset(edges)


@dataclass(frozen=True)
class SixArmConsensusSelection:
    """One strict selected layout plus integer overlap scores."""

    layout: np.ndarray
    choice: str
    scores: tuple[tuple[str, int], ...]


def select_adjacency_consensus_layout(
    layouts: Mapping[str, Any],
    fallback_layout: Any,
    *,
    grid: int = 24,
) -> SixArmConsensusSelection:
    """Select the arm whose realised adjacencies agree most with its peers."""

    if not isinstance(layouts, Mapping) or not layouts:
        raise ValueError("layouts must be a non-empty mapping")
    if not all(isinstance(name, str) and name for name in layouts):
        raise ValueError("layout names must be non-empty strings")
    validated = {
        name: _strict_layout(layout, grid=grid, name=f"layouts[{name!r}]")
        for name, layout in layouts.items()
    }
    fallback = _strict_layout(fallback_layout, grid=grid, name="fallback_layout")
    edge_sets = {
        name: directed_adjacencies(layout, grid=grid)
        for name, layout in validated.items()
    }
    frequencies = Counter(edge for edges in edge_sets.values() for edge in edges)
    scores = tuple(
        (name, sum(frequencies[edge] - 1 for edge in edge_sets[name]))
        for name in validated
    )
    maximum = max(score for _, score in scores)
    tied = tuple(name for name, score in scores if score == maximum)
    fallback_names = tuple(
        name for name in tied if np.array_equal(validated[name], fallback)
    )
    choice = fallback_names[0] if fallback_names else tied[0]
    selected = validated[choice].copy()
    selected.setflags(write=False)
    return SixArmConsensusSelection(layout=selected, choice=choice, scores=scores)


__all__ = [
    "DirectedAdjacency",
    "SixArmConsensusSelection",
    "directed_adjacencies",
    "select_adjacency_consensus_layout",
]
