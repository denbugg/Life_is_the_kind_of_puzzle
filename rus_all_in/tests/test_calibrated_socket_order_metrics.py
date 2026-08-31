from __future__ import annotations

import numpy as np

from aiijc_puzzle.calibrated_socket_order import (
    build_component_trace,
    calibrated_priority_matrices,
    edge_set_overlap,
    exact_component_metrics,
)
from aiijc_puzzle.socket_confidence_calibration import extract_hard_edge_features


def _perfect_assignment(layout: np.ndarray, *, grid: int, axis: str) -> np.ndarray:
    count = grid * grid
    board = layout.reshape(grid, grid)
    value = np.full((count + 1, count + 1), -20.0, dtype=np.float64)
    value[count, count] = -1e4
    if axis == "right":
        for row in range(grid):
            value[count, board[row, 0]] = 0.0
            value[board[row, -1], count] = 0.0
            for column in range(grid - 1):
                value[board[row, column], board[row, column + 1]] = 0.0
    else:
        for column in range(grid):
            value[count, board[0, column]] = 0.0
            value[board[-1, column], count] = 0.0
            for row in range(grid - 1):
                value[board[row, column], board[row + 1, column]] = 0.0
    return value


def test_priority_matrix_round_trip_and_component_overlap() -> None:
    grid = 3
    count = grid * grid
    generator = np.random.default_rng(51)
    right = generator.normal(size=(count + 1, count + 1))
    down = generator.normal(size=(count + 1, count + 1))
    right_raw = generator.normal(size=(count, count))
    down_raw = generator.normal(size=(count, count))
    np.fill_diagonal(right[:count, :count], -1e4)
    np.fill_diagonal(down[:count, :count], -1e4)
    np.fill_diagonal(right_raw, -1e4)
    np.fill_diagonal(down_raw, -1e4)
    right[-1, -1] = down[-1, -1] = -1e4
    features = extract_hard_edge_features(
        right_log_assignment=right,
        down_log_assignment=down,
        right_raw=right_raw,
        down_raw=down_raw,
        grid=grid,
    )
    probability = np.linspace(0.01, 0.99, len(features.values))
    priorities = calibrated_priority_matrices(features, probability, grid=grid)
    for source, target, axis, expected in zip(
        features.source,
        features.target,
        features.axis,
        probability,
        strict=True,
    ):
        name = "down" if axis else "right"
        assert priorities[name][source, target] == expected

    control = build_component_trace(
        right,
        down,
        grid=grid,
        edge_budget_per_axis=3,
    )
    calibrated = build_component_trace(
        right,
        down,
        grid=grid,
        edge_budget_per_axis=3,
        component_edge_priority=priorities,
    )
    overlap = edge_set_overlap(control, calibrated)
    assert overlap["edge_count_each"] == 6
    assert 0 <= overlap["membership_intersection"] <= 6
    assert sum(calibrated.status_counts.values()) == 6


def test_oracle_component_metrics_are_exact() -> None:
    grid = 3
    reference = np.random.default_rng(61).permutation(grid * grid)
    right = _perfect_assignment(reference, grid=grid, axis="right")
    down = _perfect_assignment(reference, grid=grid, axis="down")
    trace = build_component_trace(
        right,
        down,
        grid=grid,
        edge_budget_per_axis=grid * (grid - 1),
    )
    metrics = exact_component_metrics(trace, reference, grid=grid)
    assert metrics["selected_edge_precision"] == 1.0
    assert metrics["false_added_bridges"] == 0
    assert metrics["tile_weighted_translation_purity"] == 1.0
    assert metrics["pairwise_relative_accuracy"] == 1.0
    assert metrics["fully_exact_component_tiles"] == grid * grid
    assert metrics["largest_component"] == grid * grid
