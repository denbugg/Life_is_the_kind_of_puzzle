from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from puzzle_assembly.longsync_translation import (
    enumerate_simple_length3_paths,
    longsync4_translation,
)


def _canonical_graph_from_positions(
    positions: np.ndarray, pairs: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(sorted(pairs), dtype=np.int64)
    displacement = positions[edges[:, 1]] - positions[edges[:, 0]]
    return edges, displacement.astype(np.float64)


def _grid_graph(size: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(
        [(column, row) for row in range(size) for column in range(size)],
        dtype=np.float64,
    )
    pairs: list[tuple[int, int]] = []
    for row in range(size):
        for column in range(size):
            node = row * size + column
            if column + 1 < size:
                pairs.append((node, node + 1))
            if row + 1 < size:
                pairs.append((node, node + size))
    return _canonical_graph_from_positions(positions, pairs)


def _brute_reference(
    n_nodes: int,
    edges: np.ndarray,
    displacements: np.ndarray,
    iterations: int,
) -> tuple[tuple[tuple[tuple[int, int, int, int], ...], ...], np.ndarray, np.ndarray]:
    lookup = {tuple(edge): index for index, edge in enumerate(edges.tolist())}

    def directed(source: int, destination: int) -> tuple[int, np.ndarray] | None:
        pair = (min(source, destination), max(source, destination))
        if pair not in lookup:
            return None
        index = lookup[pair]
        sign = 1.0 if source < destination else -1.0
        return index, sign * displacements[index]

    all_paths: list[tuple[tuple[int, int, int, int], ...]] = []
    distances: list[np.ndarray] = []
    path_edges: list[np.ndarray] = []
    for edge_index, (first, last) in enumerate(edges.tolist()):
        paths: list[tuple[int, int, int, int]] = []
        values: list[float] = []
        indices: list[list[int]] = []
        for middle_first, middle_second in permutations(
            [node for node in range(n_nodes) if node not in {first, last}], 2
        ):
            steps = [
                directed(first, middle_first),
                directed(middle_first, middle_second),
                directed(middle_second, last),
            ]
            if any(step is None for step in steps):
                continue
            concrete = [step for step in steps if step is not None]
            paths.append((first, middle_first, middle_second, last))
            indices.append([step[0] for step in concrete])
            path_sum = sum((step[1] for step in concrete), np.zeros(2))
            values.append(float(np.linalg.norm(path_sum - displacements[edge_index])))
        all_paths.append(tuple(paths))
        distances.append(np.asarray(values, dtype=np.float64))
        path_edges.append(np.asarray(indices, dtype=np.int64).reshape(-1, 3))

    history = np.zeros((iterations, len(edges)), dtype=np.float64)
    weights = np.ones(len(edges), dtype=np.float64)
    for iteration in range(iterations):
        for edge_index in range(len(edges)):
            if len(distances[edge_index]) == 0:
                continue
            cycle_weights = np.prod(weights[path_edges[edge_index]], axis=1)
            history[iteration, edge_index] = np.sqrt(
                np.sum(cycle_weights * distances[edge_index] ** 2)
                / np.sum(cycle_weights)
            )
        beta = min(2.0**iteration, 20.0)
        weights = np.exp(-beta * history[iteration])
    return tuple(all_paths), history, weights


def test_clean_4x4_grid_has_exact_zero_corruption() -> None:
    edges, displacement = _grid_graph(4)
    result = longsync4_translation(16, edges, displacement)

    assert np.all(result.supported)
    assert np.all(result.support_counts >= 1)
    np.testing.assert_array_equal(result.corruption, 0.0)
    np.testing.assert_array_equal(result.weights, 1.0)
    np.testing.assert_array_equal(result.corruption_history, 0.0)
    np.testing.assert_array_equal(
        result.beta_history, [1.0, 2.0, 4.0, 8.0, 16.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    )


def test_one_corrupted_grid_edge_ranks_worse_than_clean_edges() -> None:
    edges, displacement = _grid_graph(4)
    corrupted_index = int(np.flatnonzero(np.all(edges == [5, 6], axis=1))[0])
    displacement[corrupted_index] += np.asarray([0.0, 2.0])

    result = longsync4_translation(16, edges, displacement)

    assert result.corruption[corrupted_index] > 0.0
    assert result.corruption[corrupted_index] > np.median(
        np.delete(result.corruption, corrupted_index)
    )
    assert result.weights[corrupted_index] < np.median(
        np.delete(result.weights, corrupted_index)
    )


def test_matches_independent_bruteforce_reference_on_small_sparse_graph() -> None:
    positions = np.asarray(
        [[0.0, 0.0], [1.0, 0.2], [1.1, 1.0], [-0.1, 0.8], [2.0, 0.5]]
    )
    pairs = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    edges, displacement = _canonical_graph_from_positions(positions, pairs)
    displacement[2] += [0.3, -0.15]
    expected_paths, expected_history, expected_weights = _brute_reference(
        5, edges, displacement, iterations=6
    )

    result = longsync4_translation(5, edges, displacement, iterations=6)

    assert result.alternate_paths == expected_paths
    np.testing.assert_array_equal(
        result.support_counts, [len(paths) for paths in expected_paths]
    )
    np.testing.assert_allclose(result.corruption_history, expected_history, atol=1e-14)
    np.testing.assert_allclose(result.weights, expected_weights, atol=1e-14)


def test_paths_are_exact_simple_length3_alternates() -> None:
    edges = np.asarray(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int64
    )
    paths = enumerate_simple_length3_paths(4, edges)

    assert all(len(edge_paths) == 2 for edge_paths in paths)
    edge_set = {tuple(edge) for edge in edges.tolist()}
    for target, edge_paths in zip(edges.tolist(), paths, strict=True):
        for path in edge_paths:
            assert path[0] == target[0] and path[-1] == target[1]
            assert len(set(path)) == 4
            assert all(
                (min(first, second), max(first, second)) in edge_set
                for first, second in zip(path[:-1], path[1:])
            )


def test_node_renaming_is_permutation_equivariant() -> None:
    positions = np.asarray(
        [[0.0, 0.0], [1.0, 0.1], [1.2, 1.1], [0.0, 1.0], [2.0, 0.4]]
    )
    pairs = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    edges, displacement = _canonical_graph_from_positions(positions, pairs)
    displacement[1] += [0.15, 0.35]
    original = longsync4_translation(5, edges, displacement, iterations=7)

    rename = np.asarray([3, 0, 4, 1, 2], dtype=np.int64)
    renamed_records: list[tuple[tuple[int, int], np.ndarray, tuple[int, int]]] = []
    for edge, vector in zip(edges.tolist(), displacement, strict=True):
        renamed_first, renamed_second = int(rename[edge[0]]), int(rename[edge[1]])
        if renamed_first < renamed_second:
            canonical = (renamed_first, renamed_second)
            canonical_vector = vector
        else:
            canonical = (renamed_second, renamed_first)
            canonical_vector = -vector
        renamed_records.append((canonical, canonical_vector, tuple(edge)))
    renamed_records.sort(key=lambda item: item[0])
    renamed_edges = np.asarray([item[0] for item in renamed_records], dtype=np.int64)
    renamed_displacement = np.asarray([item[1] for item in renamed_records])
    renamed = longsync4_translation(5, renamed_edges, renamed_displacement, iterations=7)

    renamed_by_original = {
        original_pair: (renamed.corruption[index], renamed.support_counts[index])
        for index, (_, _, original_pair) in enumerate(renamed_records)
    }
    for index, edge in enumerate(edges.tolist()):
        corruption, support = renamed_by_original[tuple(edge)]
        assert corruption == pytest.approx(original.corruption[index], abs=1e-14)
        assert support == original.support_counts[index]


def test_global_translation_of_node_coordinates_is_irrelevant() -> None:
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edges, displacement = _canonical_graph_from_positions(positions, pairs)
    shifted_edges, shifted_displacement = _canonical_graph_from_positions(
        positions + np.asarray([17.0, -23.0]), pairs
    )

    first = longsync4_translation(4, edges, displacement)
    shifted = longsync4_translation(4, shifted_edges, shifted_displacement)

    np.testing.assert_array_equal(first.corruption, shifted.corruption)
    np.testing.assert_array_equal(first.weights, shifted.weights)
    np.testing.assert_array_equal(first.support_counts, shifted.support_counts)


def test_edges_without_cycles_are_neutral_and_explicitly_unsupported() -> None:
    edges = np.asarray([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
    displacement = np.asarray([(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)])

    result = longsync4_translation(4, edges, displacement)

    np.testing.assert_array_equal(result.support_counts, 0)
    np.testing.assert_array_equal(result.supported, False)
    np.testing.assert_array_equal(result.corruption, 0.0)
    np.testing.assert_array_equal(result.weights, 1.0)
    assert result.alternate_paths == ((), (), ())
    assert np.all(np.isfinite(result.corruption_history))


def test_repeat_is_bitwise_deterministic_and_finite() -> None:
    edges, displacement = _grid_graph(4)
    displacement[3] += [0.25, -0.5]

    first = longsync4_translation(16, edges[::-1], displacement[::-1])
    second = longsync4_translation(16, edges[::-1], displacement[::-1])

    np.testing.assert_array_equal(first.corruption_history, second.corruption_history)
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.alternate_paths == second.alternate_paths
    assert np.all(np.isfinite(first.corruption_history))
    assert np.all(np.isfinite(first.weights))


@pytest.mark.parametrize(
    ("n_nodes", "edges", "displacement", "error", "match"),
    [
        (0, np.empty((0, 2), dtype=int), np.empty((0, 2)), ValueError, "positive"),
        (True, np.empty((0, 2), dtype=int), np.empty((0, 2)), TypeError, "positive"),
        (3, np.asarray([0, 1]), np.asarray([[1.0, 0.0]]), ValueError, "shape"),
        (3, np.asarray([[0.0, 1.0]]), np.asarray([[1.0, 0.0]]), TypeError, "integer"),
        (3, np.asarray([[1, 1]]), np.asarray([[1.0, 0.0]]), ValueError, "canonical"),
        # A reverse row with an apparently inverse displacement is still invalid:
        # the public contract stores only canonical i<j rows directed i -> j.
        (3, np.asarray([[1, 0]]), np.asarray([[-1.0, 0.0]]), ValueError, "canonical"),
        (3, np.asarray([[0, 3]]), np.asarray([[1.0, 0.0]]), ValueError, "outside"),
        (
            3,
            np.asarray([[0, 1], [0, 1]]),
            np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            ValueError,
            "duplicate",
        ),
        (3, np.asarray([[0, 1]]), np.asarray([1.0, 0.0]), ValueError, "shape"),
        (3, np.asarray([[0, 1]]), np.asarray([[np.nan, 0.0]]), ValueError, "finite"),
    ],
)
def test_input_validation(
    n_nodes: int,
    edges: np.ndarray,
    displacement: np.ndarray,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        longsync4_translation(n_nodes, edges, displacement)


@pytest.mark.parametrize("iterations", [0, -1, True, 2.5])
def test_iteration_validation(iterations: object) -> None:
    edges = np.asarray([[0, 1]], dtype=np.int64)
    displacement = np.asarray([[1.0, 0.0]])
    error = TypeError if isinstance(iterations, (bool, float)) else ValueError
    with pytest.raises(error, match="positive integer"):
        longsync4_translation(2, edges, displacement, iterations=iterations)  # type: ignore[arg-type]


def test_large_iteration_count_keeps_beta_schedule_finite() -> None:
    edges = np.asarray([[0, 1]], dtype=np.int64)
    displacement = np.asarray([[1.0, 0.0]])

    result = longsync4_translation(2, edges, displacement, iterations=2048)

    assert result.beta_history.shape == (2048,)
    assert result.beta_history[-1] == 20.0
    assert np.all(np.isfinite(result.beta_history))
