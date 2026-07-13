from __future__ import annotations

import pytest

from puzzle_assembly.growing_consensus import (
    DirectedConsensusEdge,
    SquareWitness,
    discover_order2_consensus,
)


def _rows(tile_count: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    result = [[] for _ in range(tile_count)]
    for first, second in edges:
        result[first].append(second)
    return result


def _proposal_map(result):
    return {proposal.edge: proposal for proposal in result.proposals}


def test_two_incomplete_squares_propose_missing_top_edge() -> None:
    # Two different squares agree that tile 1 belongs right of tile 0.  The
    # proposed edge is intentionally absent from the input candidate graph.
    right = _rows(6, [(2, 3), (4, 5)])
    down = _rows(6, [(0, 2), (0, 4), (1, 3), (1, 5)])

    result = discover_order2_consensus(
        right, down, tile_count=6, min_support=2
    )
    edge = DirectedConsensusEdge(0, 1, 1, 0)

    assert edge in _proposal_map(result)
    assert _proposal_map(result)[edge].support == 2
    assert result.complete_loops == ()


def test_one_incomplete_square_is_not_enough() -> None:
    right = _rows(4, [(2, 3)])
    down = _rows(4, [(0, 2), (1, 3)])

    result = discover_order2_consensus(
        right, down, tile_count=4, min_support=2
    )

    assert result.proposals == ()


def test_complete_square_is_reported_but_not_reproposed() -> None:
    right = _rows(4, [(0, 1), (2, 3)])
    down = _rows(4, [(0, 2), (1, 3)])

    result = discover_order2_consensus(
        right, down, tile_count=4, min_support=2
    )

    assert len(result.complete_loops) == 1
    assert result.complete_loops[0] == SquareWitness(0, 1, 2, 3)
    assert result.proposals == ()


def test_two_incomplete_squares_propose_missing_left_edge() -> None:
    # The two right-hand paths independently imply that tile 2 belongs below 0.
    right = _rows(6, [(0, 1), (0, 4), (2, 3), (2, 5)])
    down = _rows(6, [(1, 3), (4, 5)])

    result = discover_order2_consensus(
        right, down, tile_count=6, min_support=2
    )
    edge = DirectedConsensusEdge(0, 2, 0, 1)

    assert edge in _proposal_map(result)
    assert _proposal_map(result)[edge].support == 2


def test_witnesses_and_output_order_are_deterministic() -> None:
    right = _rows(8, [(2, 3), (4, 5), (6, 7)])
    down = _rows(
        8,
        [(0, 2), (0, 4), (0, 6), (1, 3), (1, 5), (1, 7)],
    )

    first = discover_order2_consensus(
        right, down, tile_count=8, min_support=2
    )
    second = discover_order2_consensus(
        [list(reversed(row)) for row in right],
        [list(reversed(row)) for row in down],
        tile_count=8,
        min_support=2,
    )

    assert first == second
    assert first.proposals[0].support == 3


@pytest.mark.parametrize(
    ("right", "down", "tile_count", "message"),
    [
        ([[]], [[], []], 1, "must contain 1 rows"),
        ([[0]], [[]], 1, "contains a self edge"),
        ([[1]], [[]], 1, "contains an out-of-range tile"),
    ],
)
def test_candidate_graph_validation(right, down, tile_count, message) -> None:
    with pytest.raises(ValueError, match=message):
        discover_order2_consensus(
            right, down, tile_count=tile_count, min_support=2
        )


def test_min_support_must_require_consensus() -> None:
    with pytest.raises(ValueError, match="at least two"):
        discover_order2_consensus([[], []], [[], []], tile_count=2, min_support=1)


def test_repeated_tile_paths_are_ignored_before_edge_construction() -> None:
    # The path 0-right-1-down-2 and 0-down-2 repeats tile 2 at both bottom
    # positions.  It is not a square and must be ignored rather than raising.
    right = _rows(3, [(0, 1)])
    down = _rows(3, [(0, 2), (1, 2)])

    result = discover_order2_consensus(
        right, down, tile_count=3, min_support=2
    )

    assert result.complete_loops == ()
    assert result.proposals == ()
