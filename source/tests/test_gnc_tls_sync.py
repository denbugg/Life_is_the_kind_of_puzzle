from __future__ import annotations

from numbers import Real

import numpy as np
import pytest

from puzzle_assembly.gnc_tls_sync import (
    GncTlsConfig,
    _gnc_tls_weights,
    solve_gnc_tls,
)


def _candidate_graph(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unit-offset groups with 25% missing truths but an explicit spanning tree."""

    source: list[int] = []
    destination: list[int] = []
    offsets: list[tuple[float, float]] = []
    confidence: list[float] = []
    group_index = 0
    # All horizontal edges plus first-column vertical edges form a connected
    # spanning tree.  Missing truths are therefore drawn only from the remaining
    # (size-1)^2 non-tree vertical adjacencies.  For 4x4 this drops six of nine;
    # for 3x3 it drops three of four: exactly 25% of all right/down truths.
    non_tree_vertical_count = (size - 1) ** 2
    missing_truth_count = size * (size - 1) // 2
    kept_non_tree_count = non_tree_vertical_count - missing_truth_count
    kept_non_tree_indices = set(
        np.rint(
            np.linspace(0, non_tree_vertical_count - 1, kept_non_tree_count)
        )
        .astype(np.int32)
        .tolist()
    )
    for row in range(size):
        for column in range(size):
            query = row * size + column
            relations: list[tuple[int, tuple[float, float], bool]] = []
            if column + 1 < size:
                # Every horizontal truth is a protected spanning-tree edge.
                relations.append((query + 1, (1.0, 0.0), True))
            if row + 1 < size:
                if column == 0:
                    # These vertical truths connect all protected horizontal rows.
                    truth_present = True
                else:
                    non_tree_index = row * (size - 1) + (column - 1)
                    truth_present = non_tree_index in kept_non_tree_indices
                relations.append((query + size, (0.0, 1.0), truth_present))
            for truth, offset, truth_present in relations:
                forbidden = {query, truth}
                false_candidates = [
                    tile
                    for tile in range(size * size)
                    if tile not in forbidden
                ]
                first_false = false_candidates[(3 * group_index + 1) % len(false_candidates)]
                second_false = false_candidates[(5 * group_index + 2) % len(false_candidates)]
                if second_false == first_false:
                    second_false = false_candidates[
                        (false_candidates.index(first_false) + 1) % len(false_candidates)
                    ]
                if truth_present:
                    candidates = [(truth, 0.70), (first_false, 0.20), (second_false, 0.10)]
                else:
                    candidates = [(first_false, 0.55), (second_false, 0.45)]
                for candidate, value in candidates:
                    source.append(query)
                    destination.append(candidate)
                    offsets.append(offset)
                    confidence.append(value)
                group_index += 1
    return (
        np.asarray(source, dtype=np.int32),
        np.asarray(destination, dtype=np.int32),
        np.asarray(offsets, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
    )


def _block_scrambled_grid(size: int) -> np.ndarray:
    grid = np.arange(size * size, dtype=np.int32).reshape(size, size)
    if size == 4:
        return np.block(
            [
                [grid[2:, 2:], grid[2:, :2]],
                [grid[:2, 2:], grid[:2, :2]],
            ]
        )
    # Three vertical one-column blocks in a non-cyclic order.
    return np.concatenate([grid[:, 2:], grid[:, :1], grid[:, 1:2]], axis=1)


def _config(size: int) -> GncTlsConfig:
    return GncTlsConfig(
        grid_size=size,
        gnc_stages=10,
        irls_iterations=6,
        gnc_mu_initial=0.02,
        gnc_mu_final=100.0,
        robust_cutoff=0.75,
        initial_anchor_weight=1e-4,
        current_anchor_weight=0.0,
        max_candidates_per_tile=size * size,
        max_candidate_radius=float(size),
        restarts=2,
        start_perturbation=0.02,
    )


def _perfect_unit_graph(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source: list[int] = []
    destination: list[int] = []
    offsets: list[tuple[float, float]] = []
    for row in range(size):
        for column in range(size):
            query = row * size + column
            if column + 1 < size:
                source.append(query)
                destination.append(query + 1)
                offsets.append((1.0, 0.0))
            if row + 1 < size:
                source.append(query)
                destination.append(query + size)
                offsets.append((0.0, 1.0))
    return (
        np.asarray(source, dtype=np.int32),
        np.asarray(destination, dtype=np.int32),
        np.asarray(offsets, dtype=np.float64),
        np.ones(len(source), dtype=np.float64),
    )


def _assert_numeric_finite(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_numeric_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_numeric_finite(child)
    elif isinstance(value, Real) and not isinstance(value, bool):
        assert np.isfinite(value)


def test_recovers_exact_grid_with_missing_truths_false_candidates_and_block_scramble() -> None:
    size = 4
    source, destination, offsets, confidence = _candidate_graph(size)
    result = solve_gnc_tls(
        source,
        destination,
        offsets,
        confidence,
        _block_scrambled_grid(size),
        _config(size),
        seed=17,
    )
    diagnostics = result.diagnostics
    convex = diagnostics["convex_initialization"]
    print(
        {
            "gnc_exact_recovery": {
                "initial_input_edge_score": diagnostics["initial_input_edge_score"],
                "best_input_edge_score": diagnostics["best_input_edge_score"],
                "best_restart": diagnostics["best_restart"],
                "best_stage": diagnostics["best_stage"],
                "convex_mean_residual_squared": convex["mean_residual_squared"],
                "convex_max_residual_squared": convex["maximum_residual_squared"],
                "convex_weighted_mean_residual_squared": convex[
                    "confidence_weighted_mean_residual_squared"
                ],
                "used_mu_initial": diagnostics["used_mu_initial"],
                "stages": [
                    {
                        "restart": stage["restart"],
                        "stage": stage["stage"],
                        "mu": stage["mu"],
                        "projected_edge_score": stage["projected_edge_score"],
                        "consistent_fraction": stage[
                            "consistent_confidence_fraction"
                        ],
                        "mean_robust_weight": stage["mean_robust_weight"],
                        "projection_distance": stage[
                            "projection_squared_distance"
                        ],
                    }
                    for stage in diagnostics["stages"]
                ],
            }
        }
    )
    expected = np.arange(size * size, dtype=np.int32).reshape(size, size)
    np.testing.assert_array_equal(result.grid, expected)
    np.testing.assert_array_equal(result.tile_to_cell, np.arange(size * size))


def test_sign_convention_is_destination_minus_source_in_column_row_order() -> None:
    size = 3
    source, destination, offsets, confidence = _perfect_unit_graph(size)
    result = solve_gnc_tls(
        source,
        destination,
        offsets,
        confidence,
        _block_scrambled_grid(size),
        _config(size),
        seed=5,
    )
    np.testing.assert_array_equal(
        result.grid, np.arange(size * size, dtype=np.int32).reshape(size, size)
    )
    positions = result.continuous_positions
    residual = positions[destination] - positions[source] - offsets
    assert float(np.mean(np.linalg.norm(residual, axis=1))) < 0.1


def test_piecewise_tls_weights_can_restore_a_true_edge_weight() -> None:
    cutoff = 0.75
    mu = 0.2
    initially_inconsistent = _gnc_tls_weights(np.asarray([1.0]), cutoff, mu)[0]
    later_consistent = _gnc_tls_weights(np.asarray([0.0]), cutoff, mu)[0]
    assert 0.0 <= initially_inconsistent < later_consistent
    assert later_consistent == 1.0


def test_duplicate_edges_do_not_change_solution_or_normalized_score() -> None:
    size = 3
    source, destination, offsets, confidence = _candidate_graph(size)
    config = _config(size)
    initial = _block_scrambled_grid(size)
    baseline = solve_gnc_tls(
        source, destination, offsets, confidence, initial, config, seed=91
    )
    duplicate_indices = np.arange(0, len(source), 3)
    duplicated = solve_gnc_tls(
        np.concatenate([source, source[duplicate_indices]]),
        np.concatenate([destination, destination[duplicate_indices]]),
        np.concatenate([offsets, offsets[duplicate_indices]], axis=0),
        np.concatenate([confidence, confidence[duplicate_indices] * 0.5]),
        initial,
        config,
        seed=91,
    )
    np.testing.assert_array_equal(baseline.tile_to_cell, duplicated.tile_to_cell)
    assert baseline.diagnostics["best_input_edge_score"] == pytest.approx(
        duplicated.diagnostics["best_input_edge_score"]
    )
    assert duplicated.diagnostics["raw_edge_count"] > baseline.diagnostics["raw_edge_count"]
    assert duplicated.diagnostics["deduplicated_edge_count"] == baseline.diagnostics[
        "deduplicated_edge_count"
    ]


def test_projection_is_permutation_deterministic_and_never_uses_forbidden_costs() -> None:
    size = 3
    source, destination, offsets, confidence = _candidate_graph(size)
    config = GncTlsConfig(
        **{
            **_config(size).__dict__,
            "max_candidates_per_tile": 1,
            "max_candidate_radius": 0.01,
        }
    )
    initial = _block_scrambled_grid(size)
    first = solve_gnc_tls(source, destination, offsets, confidence, initial, config, seed=123)
    second = solve_gnc_tls(source, destination, offsets, confidence, initial, config, seed=123)
    np.testing.assert_array_equal(first.tile_to_cell, second.tile_to_cell)
    np.testing.assert_array_equal(first.grid, second.grid)
    np.testing.assert_array_equal(np.sort(first.tile_to_cell), np.arange(size * size))
    np.testing.assert_array_equal(np.sort(first.grid.ravel()), np.arange(size * size))
    assert all(
        record["outside_candidate_assignments"] == 0
        for record in first.diagnostics["stages"]
    )


def test_initial_grid_is_retained_when_no_projection_improves_input_score() -> None:
    size = 3
    source, destination, offsets, confidence = _perfect_unit_graph(size)
    initial = np.arange(size * size, dtype=np.int32).reshape(size, size)
    result = solve_gnc_tls(
        source, destination, offsets, confidence, initial, _config(size), seed=7
    )
    np.testing.assert_array_equal(result.grid, initial)
    assert result.diagnostics["initial_grid_selected"] is True
    assert result.diagnostics["best_restart"] == -1


@pytest.mark.parametrize(
    "mutation",
    [
        "offset_shape",
        "nan_confidence",
        "confidence_range",
        "edge_range",
        "self_edge",
        "duplicate_grid_tile",
    ],
)
def test_validation_errors(mutation: str) -> None:
    size = 3
    source, destination, offsets, confidence = _perfect_unit_graph(size)
    grid = np.arange(size * size, dtype=np.int32).reshape(size, size)
    if mutation == "offset_shape":
        offsets = offsets[:-1]
    elif mutation == "nan_confidence":
        confidence[0] = np.nan
    elif mutation == "confidence_range":
        confidence[0] = 1.1
    elif mutation == "edge_range":
        source[0] = size * size
    elif mutation == "self_edge":
        destination[0] = source[0]
    elif mutation == "duplicate_grid_tile":
        grid[0, 0] = grid[0, 1]
    with pytest.raises((TypeError, ValueError)):
        solve_gnc_tls(
            source, destination, offsets, confidence, grid, _config(size), seed=0
        )


def test_diagnostics_are_finite_and_record_bidirectional_weight_updates() -> None:
    size = 3
    source, destination, offsets, confidence = _candidate_graph(size)
    result = solve_gnc_tls(
        source,
        destination,
        offsets,
        confidence,
        _block_scrambled_grid(size),
        _config(size),
        seed=31,
    )
    _assert_numeric_finite(result.diagnostics)
    assert result.continuous_positions.shape == (size * size, 2)
    assert np.all(np.isfinite(result.continuous_positions))
    mu_schedule = result.diagnostics["mu_schedule"]
    assert all(
        first < second
        for first, second in zip(mu_schedule[:-1], mu_schedule[1:], strict=True)
    )
    assert mu_schedule[0] <= result.diagnostics["derived_mu_initial"] + 1e-15
    assert mu_schedule[-1] == pytest.approx(100.0)
    convex = result.diagnostics["convex_initialization"]
    assert convex["all_robust_weights_one"] is True
    assert len(convex["positions_sha256"]) == 64
    assert len(convex["residual_squared_sha256"]) == 64
    np.testing.assert_allclose(convex["centroid_column_row"], [1.0, 1.0], atol=1e-12)
    assert result.diagnostics["maximum_initial_residual_squared"] == pytest.approx(
        convex["maximum_residual_squared"]
    )
    assert sum(record["weight_decreases"] for record in result.diagnostics["stages"]) > 0
    assert sum(record["weight_increases"] for record in result.diagnostics["stages"]) > 0
