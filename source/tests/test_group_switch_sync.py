from __future__ import annotations

from numbers import Real

import numpy as np
import pytest

from puzzle_assembly.group_switch_sync import GroupSwitchConfig, solve_group_switch


def _candidate_graph(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source: list[int] = []
    destination: list[int] = []
    offsets: list[tuple[float, float]] = []
    confidence: list[float] = []
    non_tree_count = (size - 1) ** 2
    missing_count = size * (size - 1) // 2
    kept_count = non_tree_count - missing_count
    kept = set(
        np.rint(np.linspace(0, non_tree_count - 1, kept_count))
        .astype(np.int32)
        .tolist()
    )
    group = 0
    for row in range(size):
        for column in range(size):
            query = row * size + column
            relations: list[tuple[int, tuple[float, float], bool]] = []
            if column + 1 < size:
                relations.append((query + 1, (1.0, 0.0), True))
            if row + 1 < size:
                present = column == 0 or row * (size - 1) + column - 1 in kept
                relations.append((query + size, (0.0, 1.0), present))
            for truth, offset, present in relations:
                alternatives = [
                    tile for tile in range(size * size) if tile not in {query, truth}
                ]
                first = alternatives[(3 * group + 1) % len(alternatives)]
                second = alternatives[(5 * group + 2) % len(alternatives)]
                if first == second:
                    second = alternatives[(alternatives.index(first) + 1) % len(alternatives)]
                candidates = (
                    [(truth, 0.75), (first, 0.15), (second, 0.10)]
                    if present
                    else [(first, 0.55), (second, 0.45)]
                )
                for candidate, value in candidates:
                    source.append(query)
                    destination.append(candidate)
                    offsets.append(offset)
                    confidence.append(value)
                group += 1
    return (
        np.asarray(source, dtype=np.int32),
        np.asarray(destination, dtype=np.int32),
        np.asarray(offsets, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
    )


def _perfect_graph(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source: list[int] = []
    destination: list[int] = []
    offsets: list[tuple[float, float]] = []
    for row in range(size):
        for column in range(size):
            tile = row * size + column
            if column + 1 < size:
                source.append(tile)
                destination.append(tile + 1)
                offsets.append((1.0, 0.0))
            if row + 1 < size:
                source.append(tile)
                destination.append(tile + size)
                offsets.append((0.0, 1.0))
    return (
        np.asarray(source, dtype=np.int32),
        np.asarray(destination, dtype=np.int32),
        np.asarray(offsets, dtype=np.float64),
        np.ones(len(source), dtype=np.float64),
    )


def _rank2_truth_graph(
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output_source: list[int] = []
    output_destination: list[int] = []
    output_offsets: list[tuple[float, float]] = []
    output_confidence: list[float] = []
    relations: list[tuple[int, int, tuple[float, float]]] = []
    # Horizontal rows plus the first vertical column are exactly a spanning
    # tree.  There are no redundant perfect edges that can recover the layout
    # while ignoring one of these deliberately rank-two truths.
    for row in range(size):
        for column in range(size - 1):
            query = row * size + column
            relations.append((query, query + 1, (1.0, 0.0)))
    for row in range(size - 1):
        query = row * size
        relations.append((query, query + size, (0.0, 1.0)))

    hubs = (0, size * size - 1)
    for group, (query, neighbour, offset) in enumerate(relations):
        # False rank-one candidates repeatedly demand that one of only two hub
        # tiles occupy incompatible relative locations.  They are individually
        # more likely, but cannot form an injective globally consistent layout.
        false = hubs[group % len(hubs)]
        if false in {query, neighbour}:
            false = hubs[(group + 1) % len(hubs)]
        output_source.extend([query, query])
        output_destination.extend([false, neighbour])
        output_offsets.extend([offset, offset])
        output_confidence.extend([0.55, 0.45])
    return (
        np.asarray(output_source, dtype=np.int32),
        np.asarray(output_destination, dtype=np.int32),
        np.asarray(output_offsets, dtype=np.float64),
        np.asarray(output_confidence, dtype=np.float64),
    )


def _four_side_reciprocal_graph(
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = size * size
    source: list[int] = []
    destination: list[int] = []
    offsets: list[tuple[float, float]] = []
    confidence: list[float] = []
    directions = (
        ("R", 1, 0, size + 2, 2 * size + 2),
        ("L", -1, 0, 2 * size + 2, size + 2),
        ("D", 0, 1, size + 1, 2 * size + 3),
        ("U", 0, -1, 2 * size + 3, size + 1),
    )
    directional_groups: dict[
        str, list[tuple[int, int, tuple[float, float], int, int]]
    ] = {name: [] for name, *_ in directions}
    for name, dx, dy, false_shift, second_false_shift in directions:
        for row in range(size):
            for column in range(size):
                target_row = row + dy
                target_column = column + dx
                if not (0 <= target_row < size and 0 <= target_column < size):
                    continue
                query = row * size + column
                truth = target_row * size + target_column
                false = (query + false_shift) % count
                second_false = (query + second_false_shift) % count
                assert false not in {query, truth}
                assert second_false not in {query, truth, false}
                directional_groups[name].append(
                    (query, truth, (float(dx), float(dy)), false, second_false)
                )

    tree_pairs = {
        tuple(sorted((row * size + column, row * size + column + 1)))
        for row in range(size)
        for column in range(size - 1)
    }
    tree_pairs.update(
        tuple(sorted((row * size, (row + 1) * size)))
        for row in range(size - 1)
    )
    nonessential = [
        (name, index)
        for name, groups in directional_groups.items()
        for index, (query, truth, *_rest) in enumerate(groups)
        if tuple(sorted((query, truth))) not in tree_pairs
    ]
    total_groups = sum(len(groups) for groups in directional_groups.values())
    missing_count = int(round(0.20 * total_groups))
    missing = set(
        nonessential[index]
        for index in np.rint(
            np.linspace(0, len(nonessential) - 1, missing_count)
        ).astype(np.int32)
    )
    assert len(missing) == missing_count

    # R/L/D/U are generated independently.  Every retained truth is rank two
    # behind a direction-specific asymmetric false candidate.  A deterministic
    # 20% of nonessential groups have no truth at all and must be rejected/null.
    for name, *_ in directions:
        for index, (query, truth, offset, false, second_false) in enumerate(
            directional_groups[name]
        ):
            source.extend([query, query])
            offsets.extend([offset, offset])
            if (name, index) in missing:
                destination.extend([false, second_false])
                confidence.extend([0.65, 0.35])
            else:
                destination.extend([false, truth])
                confidence.extend([0.55, 0.45])
    return (
        np.asarray(source, dtype=np.int32),
        np.asarray(destination, dtype=np.int32),
        np.asarray(offsets, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
    )


def _candidate_groups(
    source: np.ndarray,
    destination: np.ndarray,
    offsets: np.ndarray,
    confidence: np.ndarray,
) -> dict[tuple[int, float, float], list[tuple[int, float]]]:
    groups: dict[tuple[int, float, float], list[tuple[int, float]]] = {}
    for query, candidate, offset, value in zip(
        source.tolist(),
        destination.tolist(),
        offsets.tolist(),
        confidence.tolist(),
        strict=True,
    ):
        groups.setdefault((query, float(offset[0]), float(offset[1])), []).append(
            (candidate, value)
        )
    return groups


def _truth_destination(
    query: int, dx: float, dy: float, size: int
) -> int | None:
    row, column = divmod(query, size)
    target_row = row + int(dy)
    target_column = column + int(dx)
    if not (0 <= target_row < size and 0 <= target_column < size):
        return None
    return target_row * size + target_column


def _connected(size: int, edges: list[tuple[int, int]]) -> bool:
    adjacency = [set() for _ in range(size * size)]
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    reached = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        unseen = adjacency[current] - reached
        reached.update(unseen)
        frontier.extend(unseen)
    return len(reached) == size * size


def _block_scramble(size: int) -> np.ndarray:
    grid = np.arange(size * size, dtype=np.int32).reshape(size, size)
    return np.block(
        [[grid[2:, 2:], grid[2:, :2]], [grid[:2, 2:], grid[:2, :2]]]
    )


def _config(size: int) -> GroupSwitchConfig:
    return GroupSwitchConfig(
        grid_size=size,
        stages=9,
        iterations_per_stage=5,
        temperature_initial=4.0,
        temperature_final=0.03,
        null_prior=0.10,
        null_cost=0.75,
        initial_anchor_weight=1e-4,
        max_candidates_per_tile=size * size,
        max_candidate_radius=float(size),
        restarts=3,
    )


def _finite(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite(child)
    elif isinstance(value, Real) and not isinstance(value, bool):
        assert np.isfinite(value)


def test_exact_recovery_with_connected_missing_truth_candidate_graph() -> None:
    size = 4
    source, destination, offsets, confidence = _candidate_graph(size)
    result = solve_group_switch(
        source,
        destination,
        offsets,
        confidence,
        _block_scramble(size),
        _config(size),
        seed=19,
    )
    np.testing.assert_array_equal(
        result.grid, np.arange(size * size, dtype=np.int32).reshape(size, size)
    )
    np.testing.assert_array_equal(result.tile_to_cell, np.arange(size * size))


def test_all_null_groups_retain_a_valid_initial_permutation() -> None:
    size = 4
    source, destination, offsets, confidence = _candidate_graph(size)
    # Explicit impossible outward hypotheses on right/bottom boundaries must
    # also resolve to null rather than moving the permutation.
    source = np.concatenate([source, np.asarray([3, 12], dtype=np.int32)])
    destination = np.concatenate(
        [destination, np.asarray([0, 0], dtype=np.int32)]
    )
    offsets = np.concatenate(
        [offsets, np.asarray([(1.0, 0.0), (0.0, 1.0)], dtype=np.float64)],
        axis=0,
    )
    confidence = np.concatenate([confidence, np.zeros(2, dtype=np.float64)])
    confidence[:] = 0.0
    initial = _block_scramble(size)
    result = solve_group_switch(
        source, destination, offsets, confidence, initial, _config(size), seed=2
    )
    np.testing.assert_array_equal(result.grid, initial)
    np.testing.assert_array_equal(np.sort(result.tile_to_cell), np.arange(size * size))
    assert result.diagnostics["zero_prior_group_count"] == result.diagnostics["group_count"]
    assert all(
        restart["initial_candidate_switches"] == 0
        and restart["initial_null_switches"] == result.diagnostics["group_count"]
        and restart["initial_direction_destination_collisions"] == 0
        and restart["initial_forbidden_assignments"] == 0
        for restart in result.diagnostics["restarts"]
    )
    assert all(
        stage["candidate_mass_max"] == 0.0
        and stage["null_mass_min"] == 1.0
        and stage["posterior_active_edges"] == 0
        for stage in result.diagnostics["stages"]
    )


def test_null_prior_one_forces_all_null_with_positive_candidate_confidence() -> None:
    size = 4
    source, destination, offsets, confidence = _candidate_graph(size)
    initial = _block_scramble(size)
    base = _config(size)
    config = GroupSwitchConfig(**{**base.__dict__, "null_prior": 1.0})
    result = solve_group_switch(
        source, destination, offsets, confidence, initial, config, seed=3
    )
    np.testing.assert_array_equal(result.grid, initial)
    assert all(
        restart["initial_candidate_switches"] == 0
        and restart["initial_null_switches"] == result.diagnostics["group_count"]
        and restart["initial_direction_destination_collisions"] == 0
        and restart["initial_forbidden_assignments"] == 0
        for restart in result.diagnostics["restarts"]
    )
    assert all(
        stage["candidate_mass_max"] == 0.0
        and stage["null_mass_min"] == 1.0
        and stage["posterior_active_edges"] == 0
        for stage in result.diagnostics["stages"]
    )


def test_group_candidate_posterior_mass_never_exceeds_one() -> None:
    size = 4
    source, destination, offsets, confidence = _candidate_graph(size)
    result = solve_group_switch(
        source, destination, offsets, confidence, _block_scramble(size), _config(size), seed=7
    )
    assert all(stage["candidate_mass_max"] <= 1.0 + 1e-12 for stage in result.diagnostics["stages"])
    assert all(
        abs(stage["candidate_mass_mean"] + stage["null_mass_mean"] - 1.0) <= 1e-12
        for stage in result.diagnostics["stages"]
    )
    assert result.diagnostics["categorical_prior_normalization_max_error"] <= 1e-12


def test_duplicate_candidates_are_invariant() -> None:
    size = 4
    source, destination, offsets, confidence = _candidate_graph(size)
    initial = _block_scramble(size)
    config = _config(size)
    baseline = solve_group_switch(
        source, destination, offsets, confidence, initial, config, seed=11
    )
    chosen = np.arange(0, len(source), 4)
    duplicate = solve_group_switch(
        np.concatenate([source, source[chosen]]),
        np.concatenate([destination, destination[chosen]]),
        np.concatenate([offsets, offsets[chosen]], axis=0),
        np.concatenate([confidence, confidence[chosen] * 0.5]),
        initial,
        config,
        seed=11,
    )
    np.testing.assert_array_equal(baseline.tile_to_cell, duplicate.tile_to_cell)
    assert baseline.diagnostics["best_group_objective"] == pytest.approx(
        duplicate.diagnostics["best_group_objective"]
    )
    assert duplicate.diagnostics["deduplicated_edge_count"] == baseline.diagnostics[
        "deduplicated_edge_count"
    ]


def test_seed_is_deterministic_and_hungarian_is_strict() -> None:
    size = 4
    source, destination, offsets, confidence = _candidate_graph(size)
    initial = _block_scramble(size)
    base = _config(size)
    config = GroupSwitchConfig(
        **{
            **base.__dict__,
            "max_candidates_per_tile": 1,
            "max_candidate_radius": 0.01,
        }
    )
    first = solve_group_switch(
        source, destination, offsets, confidence, initial, config, seed=101
    )
    second = solve_group_switch(
        source, destination, offsets, confidence, initial, config, seed=101
    )
    np.testing.assert_array_equal(first.tile_to_cell, second.tile_to_cell)
    np.testing.assert_allclose(first.continuous_positions, second.continuous_positions)
    np.testing.assert_array_equal(np.sort(first.grid.ravel()), np.arange(size * size))
    assert all(stage["outside_candidate_assignments"] == 0 for stage in first.diagnostics["stages"])
    assert first.diagnostics["initial_switch_method"] == (
        "per_direction_one_to_one_assignment"
    )
    assert all(
        restart["initial_direction_destination_collisions"] == 0
        and restart["initial_forbidden_assignments"] == 0
        for restart in first.diagnostics["restarts"]
    )
    assert first.diagnostics["restart_unique_initial_switches"] >= 2
    assert first.diagnostics["posterior_normalization_max_error"] <= 1e-12
    _finite(first.diagnostics)


def test_offset_sign_is_destination_minus_source_column_row() -> None:
    size = 4
    source, destination, offsets, confidence = _perfect_graph(size)
    result = solve_group_switch(
        source, destination, offsets, confidence, _block_scramble(size), _config(size), seed=5
    )
    np.testing.assert_array_equal(
        result.grid, np.arange(size * size, dtype=np.int32).reshape(size, size)
    )
    assert result.diagnostics["initial_switch_method"] == (
        "per_direction_one_to_one_assignment"
    )
    assert all(
        restart["initial_direction_destination_collisions"] == 0
        and restart["initial_forbidden_assignments"] == 0
        for restart in result.diagnostics["restarts"]
    )
    residual = (
        result.continuous_positions[destination]
        - result.continuous_positions[source]
        - offsets
    )
    assert float(np.mean(np.linalg.norm(residual, axis=1))) < 0.1


def test_geometry_can_recover_truth_that_is_rank_two_by_prior() -> None:
    size = 4
    source, destination, offsets, confidence = _rank2_truth_graph(size)
    groups = _candidate_groups(source, destination, offsets, confidence)
    truth_edges: list[tuple[int, int]] = []
    false_constraints: dict[tuple[int, float, float], set[int]] = {}
    assert len(groups) == size * size - 1
    for (query, dx, dy), candidates in groups.items():
        truth = _truth_destination(query, dx, dy, size)
        assert truth is not None
        by_destination = dict(candidates)
        assert by_destination[truth] == pytest.approx(0.45)
        false, false_confidence = max(candidates, key=lambda item: item[1])
        assert false != truth
        assert false_confidence == pytest.approx(0.55)
        truth_edges.append((query, truth))
        false_constraints.setdefault((false, dx, dy), set()).add(query)
    assert _connected(size, truth_edges)
    # A tree has no redundant truth: deleting any rank-two edge disconnects it.
    assert all(
        not _connected(size, truth_edges[:index] + truth_edges[index + 1 :])
        for index in range(len(truth_edges))
    )
    # Reused false hubs demand identical relative positions from distinct tiles.
    assert sum(len(queries) >= 2 for queries in false_constraints.values()) >= 2
    result = solve_group_switch(
        source,
        destination,
        offsets,
        confidence,
        _block_scramble(size),
        _config(size),
        seed=37,
    )
    np.testing.assert_array_equal(
        result.grid, np.arange(size * size, dtype=np.int32).reshape(size, size)
    )
    assert result.diagnostics["initial_switch_method"] == (
        "per_direction_one_to_one_assignment"
    )
    assert all(
        restart["initial_direction_destination_collisions"] == 0
        and restart["initial_forbidden_assignments"] == 0
        for restart in result.diagnostics["restarts"]
    )


def test_four_side_reciprocal_constraints_preserve_sign_and_exact_layout() -> None:
    size = 4
    source, destination, offsets, confidence = _four_side_reciprocal_graph(size)
    groups = _candidate_groups(source, destination, offsets, confidence)
    expected_group_count = 4 * size * (size - 1)
    assert len(groups) == expected_group_count
    assert {
        (dx, dy) for _query, dx, dy in groups
    } == {(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)}

    tree_pairs = {
        tuple(sorted((row * size + column, row * size + column + 1)))
        for row in range(size)
        for column in range(size - 1)
    }
    tree_pairs.update(
        tuple(sorted((row * size, (row + 1) * size)))
        for row in range(size - 1)
    )
    truth_present: set[tuple[int, float, float]] = set()
    truth_missing: set[tuple[int, float, float]] = set()
    rank_two_truth = 0
    false_by_group: dict[tuple[int, float, float], int] = {}
    for key, candidates in groups.items():
        query, dx, dy = key
        truth = _truth_destination(query, dx, dy, size)
        assert truth is not None
        by_destination = dict(candidates)
        false_by_group[key] = max(candidates, key=lambda item: item[1])[0]
        if truth in by_destination:
            truth_present.add(key)
            assert by_destination[truth] == pytest.approx(0.45)
            assert max(candidates, key=lambda item: item[1])[1] == pytest.approx(0.55)
            rank_two_truth += 1
        else:
            truth_missing.add(key)
            assert tuple(sorted((query, truth))) not in tree_pairs
    missing_fraction = len(truth_missing) / expected_group_count
    assert 0.15 <= missing_fraction <= 0.25
    assert rank_two_truth == len(truth_present)

    # The deliberately preserved reciprocal spanning tree is connected and
    # identifiable even though every one of its truths is only rank two.
    reciprocal_tree_edges: list[tuple[int, int]] = []
    for first, second in tree_pairs:
        first_row, first_column = divmod(first, size)
        second_row, second_column = divmod(second, size)
        dx = float(second_column - first_column)
        dy = float(second_row - first_row)
        assert (first, dx, dy) in truth_present
        assert (second, -dx, -dy) in truth_present
        reciprocal_tree_edges.append((first, second))
    assert _connected(size, reciprocal_tree_edges)

    asymmetric = []
    for query, dx, dy in truth_present:
        truth = _truth_destination(query, dx, dy, size)
        assert truth is not None
        reverse = (truth, -dx, -dy)
        if reverse in truth_present:
            asymmetric.append(false_by_group[(query, dx, dy)] != false_by_group[reverse])
    assert np.mean(asymmetric) >= 0.90

    result = solve_group_switch(
        source,
        destination,
        offsets,
        confidence,
        _block_scramble(size),
        _config(size),
        seed=41,
    )
    np.testing.assert_array_equal(
        result.grid, np.arange(size * size, dtype=np.int32).reshape(size, size)
    )
    positions = np.column_stack(
        [result.tile_to_cell % size, result.tile_to_cell // size]
    )
    reciprocal_consistency = []
    for query, dx, dy in truth_present:
        truth = _truth_destination(query, dx, dy, size)
        assert truth is not None
        if (truth, -dx, -dy) not in truth_present:
            continue
        forward = positions[truth] - positions[query]
        reverse = positions[query] - positions[truth]
        reciprocal_consistency.append(
            np.array_equal(forward, np.asarray([dx, dy]))
            and np.array_equal(reverse, np.asarray([-dx, -dy]))
        )
    assert np.mean(reciprocal_consistency) >= 0.90
