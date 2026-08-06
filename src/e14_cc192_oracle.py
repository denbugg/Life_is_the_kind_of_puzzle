"""Pure structural and solver contract for the fixed E14 CC192 arm.

The module deliberately exposes no edge-budget, margin, repair, orientation,
or search controls.  Structural measurements use the exact production
``solve_buddies._candidate_edges`` and ``build_buddies_components`` functions
with ``max_edges=192``, ``min_margin=0``.  The solve uses the same component
path with zero repair passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from solve_buddies import (
    _candidate_edges,
    build_buddies_components,
    solve_buddies_from_scores,
)


GRID = 24
NUM_TILES = GRID * GRID
MAX_EDGES = 192
MIN_MARGIN = 0.0
REPAIR_PASSES = 0


class CC192ContractError(ValueError):
    """An input or solver output violates the fixed E14 contract."""


@dataclass(frozen=True)
class CC192Structure:
    """Label-aware calibration diagnostics for the exact CC192 seed graph."""

    selected_edge_count: int
    true_edge_count: int
    selected_edge_precision: float
    component_count: int
    covered_tiles: int
    component_coverage: float
    largest_component: int
    component_sizes: tuple[int, ...]


def _dense_matrix(value: np.ndarray, *, label: str) -> np.ndarray:
    matrix = np.ascontiguousarray(value, dtype=np.float32)
    expected = (NUM_TILES, NUM_TILES)
    if matrix.shape != expected:
        raise CC192ContractError(f"{label} must have shape {expected}")
    if not np.isfinite(matrix).all() or bool((matrix < 0.0).any()):
        raise CC192ContractError(f"{label} must be finite and nonnegative")
    if bool((np.diag(matrix) != 0.0).any()):
        raise CC192ContractError(f"{label} diagonal must be exactly zero")
    return matrix


def _permutation(value: np.ndarray) -> np.ndarray:
    permutation = np.asarray(value)
    if permutation.shape != (NUM_TILES,) or permutation.dtype.kind not in "iu":
        raise CC192ContractError(
            "permutation must be an integer input_tile->clean_cell vector"
        )
    permutation = permutation.astype(np.int64, copy=False)
    if not np.array_equal(
        np.sort(permutation), np.arange(NUM_TILES, dtype=np.int64)
    ):
        raise CC192ContractError("permutation is not a bijection")
    return np.ascontiguousarray(permutation)


def _board(value: np.ndarray) -> np.ndarray:
    board = np.asarray(value)
    if board.shape != (NUM_TILES,) or board.dtype.kind not in "iu":
        raise CC192ContractError("solver board must be an integer vector of length 576")
    board = board.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(board), np.arange(NUM_TILES, dtype=np.int64)):
        raise CC192ContractError("solver board is not a strict tile permutation")
    return np.ascontiguousarray(board)


def _edge_is_true(
    permutation: np.ndarray, a: int, b: int, dy: int, dx: int
) -> bool:
    ay, ax = divmod(int(permutation[a]), GRID)
    by, bx = divmod(int(permutation[b]), GRID)
    return (by - ay, bx - ax) == (dy, dx)


def measure_cc192_structure(
    right: np.ndarray, down: np.ndarray, permutation: np.ndarray
) -> CC192Structure:
    """Measure exact selected-edge precision and component tile coverage."""

    right_matrix = _dense_matrix(right, label="right")
    down_matrix = _dense_matrix(down, label="down")
    truth = _permutation(permutation)

    edges = _candidate_edges(
        right_matrix,
        down_matrix,
        max_edges=MAX_EDGES,
        min_margin=MIN_MARGIN,
    )
    true_edges = sum(
        int(_edge_is_true(truth, a, b, dy, dx))
        for _score, _margin, a, b, dy, dx in edges
    )
    precision = float(true_edges / len(edges)) if edges else 0.0

    components = build_buddies_components(
        right_matrix,
        down_matrix,
        max_edges=MAX_EDGES,
        min_margin=MIN_MARGIN,
    )
    component_tiles = [set(map(int, component.keys())) for component in components]
    covered = set().union(*component_tiles) if component_tiles else set()
    if sum(map(len, component_tiles)) != len(covered):
        raise CC192ContractError("buddies components overlap in tile identity")
    sizes = tuple(sorted((len(component) for component in component_tiles), reverse=True))
    return CC192Structure(
        selected_edge_count=int(len(edges)),
        true_edge_count=int(true_edges),
        selected_edge_precision=precision,
        component_count=int(len(components)),
        covered_tiles=int(len(covered)),
        component_coverage=float(len(covered) / NUM_TILES),
        largest_component=int(sizes[0] if sizes else 0),
        component_sizes=sizes,
    )


def solve_cc192(
    right: np.ndarray,
    down: np.ndarray,
    *,
    solver: Callable[..., tuple[np.ndarray, float]] = solve_buddies_from_scores,
) -> tuple[np.ndarray, float]:
    """Run exactly buddies(192, margin=0, repair=0) and validate its output."""

    right_matrix = _dense_matrix(right, label="right")
    down_matrix = _dense_matrix(down, label="down")
    board, objective = solver(
        right_matrix,
        down_matrix,
        max_edges=MAX_EDGES,
        min_margin=MIN_MARGIN,
        repair_passes=REPAIR_PASSES,
    )
    board_array = _board(np.asarray(board))
    objective_value = float(objective)
    if not np.isfinite(objective_value):
        raise CC192ContractError("solver objective is non-finite")
    return board_array, objective_value
