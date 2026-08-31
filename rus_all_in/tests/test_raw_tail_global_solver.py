from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import (
    RawTailEdge,
    RawTailGlobalConfig,
    build_raw_tail_components,
    solve_raw_tail_global,
)


def _costs(grid: int) -> tuple[np.ndarray, np.ndarray]:
    count = grid * grid
    return (
        np.full((count, count), 10.0, dtype=np.float64),
        np.full((count, count), 10.0, dtype=np.float64),
    )


def _square_edges() -> tuple[RawTailEdge, ...]:
    return (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(2, 3, "right"),
        RawTailEdge(0, 2, "down"),
        RawTailEdge(1, 3, "down"),
    )


def test_complete_rigid_square_returns_the_only_legal_layout() -> None:
    right, down = _costs(2)
    right[0, 1] = right[2, 3] = 0.0
    down[0, 2] = down[1, 3] = 0.0

    result = solve_raw_tail_global(right, down, _square_edges(), grid=2)

    assert np.array_equal(result.layout, np.arange(4))
    assert not result.layout.flags.writeable
    assert result.diagnostics.strict_permutation
    assert result.diagnostics.placed_component_tiles == 4
    assert result.diagnostics.component_sizes == (4,)


def test_raw_cost_priority_decides_which_colliding_edge_survives() -> None:
    right, down = _costs(2)
    right[0, 2] = -2.0
    right[0, 1] = -1.0
    candidates = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(0, 2, "right"),
    )

    components, decisions = build_raw_tail_components(
        right,
        down,
        candidates,
        grid=2,
    )

    assert tuple(decision.edge.target for decision in decisions) == (2, 1)
    assert tuple(decision.status for decision in decisions) == (
        "accepted_new",
        "rejected_collision",
    )
    assert components == ({0: (0, 0), 2: (0, 1)},)


def test_equal_raw_scores_use_explicit_stable_input_order() -> None:
    right, down = _costs(2)
    candidates = (
        RawTailEdge(0, 2, "right"),
        RawTailEdge(0, 1, "right"),
    )

    components, decisions = build_raw_tail_components(
        right,
        down,
        candidates,
        grid=2,
    )

    assert tuple(decision.input_rank for decision in decisions) == (0, 1)
    assert components == ({0: (0, 0), 2: (0, 1)},)


def test_empty_harvest_fill_is_seeded_strict_and_not_canonical_identity() -> None:
    right = np.zeros((4, 4), dtype=np.float64)
    down = np.zeros((4, 4), dtype=np.float64)
    config = RawTailGlobalConfig(random_seed=0)

    first = solve_raw_tail_global(right, down, (), grid=2, config=config)
    second = solve_raw_tail_global(right, down, (), grid=2, config=config)

    assert np.array_equal(first.layout, second.layout)
    assert np.array_equal(np.sort(first.layout), np.arange(4))
    assert not np.array_equal(first.layout, np.arange(4))


def test_full_component_is_equivariant_to_an_input_bag_relabeling() -> None:
    right, down = _costs(2)
    right[0, 1] = right[2, 3] = 0.0
    down[0, 2] = down[1, 3] = 0.0
    relabel = np.asarray([2, 0, 3, 1])
    inverse = np.argsort(relabel)
    relabeled_right = right[np.ix_(inverse, inverse)]
    relabeled_down = down[np.ix_(inverse, inverse)]
    relabeled_edges = tuple(
        RawTailEdge(int(relabel[edge.source]), int(relabel[edge.target]), edge.axis)
        for edge in _square_edges()
    )

    original = solve_raw_tail_global(right, down, _square_edges(), grid=2)
    relabeled = solve_raw_tail_global(
        relabeled_right,
        relabeled_down,
        relabeled_edges,
        grid=2,
    )

    assert np.array_equal(relabeled.layout, relabel[original.layout])


def test_rejects_target_shaped_shortcuts_and_invalid_edges() -> None:
    right, down = _costs(2)
    with pytest.raises(ValueError, match="border_unary"):
        solve_raw_tail_global(
            right,
            down,
            (),
            border_unary=np.zeros((4, 4)),
            grid=2,
        )
    with pytest.raises(ValueError, match="self-edge"):
        solve_raw_tail_global(right, down, (RawTailEdge(0, 0, "right"),), grid=2)
