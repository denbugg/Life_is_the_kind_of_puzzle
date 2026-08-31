from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import aiijc_puzzle.union_fragment_synchronizer as synchronizer_module
from aiijc_puzzle.union_fragment_synchronizer import (
    RigidFragment,
    UnionDisplacementFactor,
    UnionFragmentSynchronizerConfig,
    audit_rigid_fragment_layout,
    build_reversible_displacement_factors,
    decode_union_fragment_layout,
    freeze_union_candidate_snapshot,
    solve_rigid_exact_cover,
    synchronise_fragment_origins,
)


def _perfect_assignment(layout: np.ndarray, *, grid: int, axis: str) -> np.ndarray:
    count = grid * grid
    board = np.asarray(layout).reshape(grid, grid)
    value = np.full((count + 1, count + 1), -20.0, dtype=np.float64)
    value[count, count] = -1e4
    if axis == "right":
        for row in range(grid):
            value[count, board[row, 0]] = 0.0
            value[board[row, -1], count] = 0.0
            for column in range(grid - 1):
                value[board[row, column], board[row, column + 1]] = 0.0
    elif axis == "down":
        for column in range(grid):
            value[count, board[0, column]] = 0.0
            value[board[-1, column], count] = 0.0
            for row in range(grid - 1):
                value[board[row, column], board[row + 1, column]] = 0.0
    else:
        raise ValueError(axis)
    return value


def _dense_candidate_snapshot(layout: np.ndarray, *, grid: int):
    count = grid * grid
    board = np.asarray(layout).reshape(grid, grid)
    true_edges: dict[int, set[tuple[int, int]]] = {0: set(), 1: set()}
    for row in range(grid):
        for column in range(grid - 1):
            true_edges[0].add((int(board[row, column]), int(board[row, column + 1])))
    for row in range(grid - 1):
        for column in range(grid):
            true_edges[1].add((int(board[row, column]), int(board[row + 1, column])))
    axes: list[int] = []
    sources: list[int] = []
    targets: list[int] = []
    scores: list[float] = []
    for axis in (0, 1):
        for source in range(count):
            for target in range(count):
                if source == target:
                    continue
                axes.append(axis)
                sources.append(source)
                targets.append(target)
                scores.append(8.0 if (source, target) in true_edges[axis] else -8.0)
    return freeze_union_candidate_snapshot(
        axes,
        sources,
        targets,
        scores,
        grid=grid,
    )


def _singleton_fragments(*, grid: int) -> tuple[RigidFragment, ...]:
    return tuple(RigidFragment((tile,), (0,), (0,)) for tile in range(grid * grid))


def _single_hypothesis_factor(
    first: int,
    second: int,
    shift: tuple[int, int],
    *,
    reliability: float,
) -> UnionDisplacementFactor:
    return UnionDisplacementFactor(
        first_component=first,
        second_component=second,
        row_shifts=np.asarray([shift[0]]),
        column_shifts=np.asarray([shift[1]]),
        probabilities=np.asarray([1.0]),
        reliability=reliability,
        total_mass=1.0,
    )


def test_candidate_snapshot_is_an_immutable_cpu_copy() -> None:
    scores = np.asarray([1.0, 2.0], dtype=np.float32)
    snapshot = freeze_union_candidate_snapshot(
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.asarray([1, 0]),
        scores,
        grid=2,
    )
    scores[:] = -100.0
    assert snapshot.scores.tolist() == [1.0, 2.0]
    assert len(snapshot.sha256) == 64
    with pytest.raises(ValueError, match="read-only"):
        snapshot.scores[0] = 4.0


def test_full_score_outgoing_incoming_mass_aggregates_duplicate_displacements() -> None:
    # Every candidate has p_out=p_in=1/2, hence mass 1/2.  Parallel
    # boundary contacts aggregate into two equally likely component shifts.
    fragments = (
        RigidFragment((0, 1), (0, 1), (0, 0)),
        RigidFragment((2, 3), (0, 1), (0, 0)),
    )
    snapshot = freeze_union_candidate_snapshot(
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [2, 3, 2, 3],
        [0.0, 0.0, 0.0, 0.0],
        grid=2,
    )
    factors = build_reversible_displacement_factors(snapshot, fragments)
    assert len(factors) == 1
    factor = factors[0]
    assert factor.total_mass == pytest.approx(2.0)
    hypotheses = {
        (int(row), int(column)): float(probability)
        for row, column, probability in zip(
            factor.row_shifts,
            factor.column_shifts,
            factor.probabilities,
            strict=True,
        )
    }
    assert hypotheses == pytest.approx({(0, 1): 0.5, (1, 1): 0.5})
    assert factor.reliability == pytest.approx(0.0, abs=1e-12)


def test_reversing_component_order_negates_the_canonical_shift() -> None:
    fragments = _singleton_fragments(grid=3)
    snapshot = freeze_union_candidate_snapshot([0], [1], [0], [0.0], grid=3)
    factor = build_reversible_displacement_factors(snapshot, fragments)[0]
    assert (factor.first_component, factor.second_component) == (0, 1)
    assert factor.row_shifts.tolist() == [0]
    assert factor.column_shifts.tolist() == [2]


def test_consensus_can_reject_a_wrong_maximum_forest_edge() -> None:
    grid = 3
    fragments = _singleton_fragments(grid=grid)
    # The strongest 0-1 edge is wrong and therefore enters the maximum
    # spanning forest.  Two independent, slightly weaker paths agree on the
    # correct shift and move component 1 during reversible coordinate ascent.
    factors = (
        _single_hypothesis_factor(0, 1, (0, 2), reliability=0.95),
        _single_hypothesis_factor(0, 2, (0, 2), reliability=0.90),
        _single_hypothesis_factor(1, 2, (0, 1), reliability=0.90),
        _single_hypothesis_factor(0, 3, (1, 0), reliability=0.90),
        _single_hypothesis_factor(1, 3, (1, 2), reliability=0.90),
    )
    result = synchronise_fragment_origins(
        fragments,
        factors,
        np.arange(grid * grid),
        grid=grid,
        max_passes=8,
    )
    assert 0 in result.forest_factor_indices
    first_row, first_column = divmod(int(result.origins[0]), grid)
    second_row, second_column = divmod(int(result.origins[1]), grid)
    assert ((second_row - first_row) % grid, (second_column - first_column) % grid) == (
        0,
        1,
    )
    assert result.final_objective >= result.initial_objective


def test_exact_cover_preserves_whole_fragments_and_every_slot() -> None:
    grid = 2
    fragments = (
        RigidFragment((0, 1), (0, 0), (0, 1)),
        RigidFragment((2, 3), (0, 0), (0, 1)),
    )
    unaries = np.full((2, grid * grid), -10.0)
    unaries[0, 0] = 0.0
    unaries[1, 2] = 0.0
    result = solve_rigid_exact_cover(
        fragments,
        unaries,
        np.arange(grid * grid),
        grid=grid,
    )
    assert not result.used_fallback
    assert result.milp_status == 0
    assert np.array_equal(result.layout, np.arange(grid * grid))
    assert result.audit.strict_permutation
    assert result.audit.rigidity_preserved
    assert result.audit.preserved_tiles == grid * grid


def test_all_singleton_exact_cover_equals_hungarian() -> None:
    grid = 2
    fragments = _singleton_fragments(grid=grid)
    unaries = np.asarray(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )
    result = solve_rigid_exact_cover(
        fragments,
        unaries,
        np.asarray([3, 2, 1, 0]),
        grid=grid,
    )
    assert not result.used_fallback
    assert result.milp_message == "all-singleton Hungarian exact cover"
    assert np.array_equal(result.layout, np.arange(grid * grid))


def test_inconclusive_milp_fails_closed_to_exact_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = 2
    fragments = (
        RigidFragment((0, 1), (0, 0), (0, 1)),
        RigidFragment((2, 3), (0, 0), (0, 1)),
    )
    fallback = np.asarray([2, 3, 0, 1], dtype=np.int32)
    unaries = np.zeros((2, grid * grid))
    unaries[0, 0] = 1.0
    monkeypatch.setattr(
        synchronizer_module,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(
            status=1,
            message="Time limit reached",
            x=None,
            mip_gap=None,
        ),
    )
    result = solve_rigid_exact_cover(
        fragments,
        unaries,
        fallback,
        grid=grid,
    )
    assert result.used_fallback
    assert result.fallback_reason == "milp-nonoptimal-status-1"
    assert np.array_equal(result.layout, fallback)


def test_audit_detects_a_strict_layout_that_breaks_a_rigid_fragment() -> None:
    fragments = (
        RigidFragment((0, 1), (0, 0), (0, 1)),
        RigidFragment((2,), (0,), (0,)),
        RigidFragment((3,), (0,), (0,)),
    )
    audit = audit_rigid_fragment_layout(np.asarray([0, 2, 1, 3]), fragments, grid=2)
    assert audit.strict_permutation
    assert not audit.rigidity_preserved
    assert audit.preserved_components == 2


def test_end_to_end_perfect_board_is_strict_rigid_and_deterministic() -> None:
    grid = 3
    reference = np.random.default_rng(73).permutation(grid * grid)
    right = _perfect_assignment(reference, grid=grid, axis="right")
    down = _perfect_assignment(reference, grid=grid, axis="down")
    snapshot = _dense_candidate_snapshot(reference, grid=grid)
    config = UnionFragmentSynchronizerConfig(
        hard_edge_budget_per_axis=1,
        synchronization_passes=8,
        milp_time_limit_seconds=5.0,
        milp_relative_gap=0.0,
        cyclic_border_weight=5.0,
    )
    first = decode_union_fragment_layout(
        right,
        down,
        snapshot,
        reference,
        config=config,
    )
    second = decode_union_fragment_layout(
        right,
        down,
        snapshot,
        reference,
        config=config,
    )
    assert not first.used_fallback
    assert np.array_equal(first.layout, reference)
    assert np.array_equal(first.layout, second.layout)
    assert first.audit.strict_permutation
    assert first.audit.rigidity_preserved
    assert first.diagnostics.factor_count > 0
    assert first.diagnostics.milp_status == 0
    assert first.report()["layout_sha256"] == second.report()["layout_sha256"]
