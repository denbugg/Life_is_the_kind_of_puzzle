from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.direct_residual_union_priority import (
    build_direct_rank_delta_union_priority,
    build_direct_residual_union_priority,
)
from aiijc_puzzle.socket_decoder import hard_partial_axis_matching


def _assignment(grid: int, seed: int) -> np.ndarray:
    count = grid * grid
    value = np.random.default_rng(seed).normal(size=(count + 1, count + 1))
    np.fill_diagonal(value[:count, :count], -1e4)
    value[count, count] = -1e4
    return value


def _unused_identity(
    matching_identities: set[tuple[int, int]],
    *,
    count: int,
) -> tuple[int, int]:
    for source in range(count):
        for target in range(count):
            if source != target and (source, target) not in matching_identities:
                return source, target
    raise AssertionError("fixture has no unused directed identity")


def test_transfer_adds_only_identity_matched_residuals_to_union_confidence() -> None:
    grid = 3
    count = grid * grid
    right_assignment = _assignment(grid, 101)
    down_assignment = _assignment(grid, 102)
    right = hard_partial_axis_matching(right_assignment, grid=grid, axis="right")
    down = hard_partial_axis_matching(down_assignment, grid=grid, axis="down")
    right_edge = right.edges[0]
    down_edge = down.edges[-1]
    unused_source, unused_target = _unused_identity(
        {(edge.source, edge.target) for edge in right.edges},
        count=count,
    )

    kwargs = {
        "direct_source": np.asarray(
            [right_edge.source, down_edge.source, unused_source], dtype=np.int32
        ),
        "direct_target": np.asarray(
            [right_edge.target, down_edge.target, unused_target], dtype=np.int32
        ),
        "direct_axis": np.asarray([0, 1, 0], dtype=np.int8),
        "direct_raw_scores": np.asarray([10.0, -2.0, 0.25], dtype=np.float32),
        "direct_learned_scores": np.asarray([12.25, -3.5, 9.0], dtype=np.float32),
        "grid": grid,
    }
    first = build_direct_residual_union_priority(
        right_assignment,
        down_assignment,
        **kwargs,
    )
    second = build_direct_residual_union_priority(
        right_assignment,
        down_assignment,
        **kwargs,
    )

    priorities = first.component_edge_priority
    assert set(priorities) == {"right", "down"}
    for axis in ("right", "down"):
        assert priorities[axis].shape == (count, count)
        assert priorities[axis].dtype == np.float64
        assert priorities[axis].flags.c_contiguous
        np.testing.assert_array_equal(
            priorities[axis], second.component_edge_priority[axis]
        )

    right_by_identity = {(edge.source, edge.target): edge for edge in right.edges}
    down_by_identity = {(edge.source, edge.target): edge for edge in down.edges}
    for source in range(count):
        for target in range(count):
            expected_right = 0.0
            expected_down = 0.0
            if (source, target) in right_by_identity:
                expected_right = right_by_identity[source, target].confidence
            if (source, target) == (right_edge.source, right_edge.target):
                expected_right += 2.25
            if (source, target) in down_by_identity:
                expected_down = down_by_identity[source, target].confidence
            if (source, target) == (down_edge.source, down_edge.target):
                expected_down -= 1.5
            assert priorities["right"][source, target] == pytest.approx(expected_right)
            assert priorities["down"][source, target] == pytest.approx(expected_down)

    diagnostics = first.diagnostics
    assert diagnostics.union_edges_per_axis == {"right": 6, "down": 6}
    assert diagnostics.direct_edges_per_axis == {"right": 2, "down": 1}
    assert diagnostics.matched_edges_per_axis == {"right": 1, "down": 1}
    assert diagnostics.unmatched_union_edges_per_axis == {"right": 5, "down": 5}
    assert diagnostics.unused_direct_edges_per_axis == {"right": 1, "down": 0}
    assert diagnostics.matched_edge_count == 2
    assert diagnostics.residual_min == pytest.approx(-1.5)
    assert diagnostics.residual_max == pytest.approx(2.25)
    assert diagnostics.residual_mean == pytest.approx(0.375)
    assert first.report() == second.report()
    assert first.report()["schema"] == "aiijc-direct-residual-union-priority-v1"


def test_empty_direct_supply_preserves_union_confidence() -> None:
    grid = 2
    right_assignment = _assignment(grid, 201)
    down_assignment = _assignment(grid, 202)
    result = build_direct_residual_union_priority(
        right_assignment,
        down_assignment,
        direct_source=[],
        direct_target=[],
        direct_axis=[],
        direct_raw_scores=[],
        direct_learned_scores=[],
        grid=grid,
    )
    for axis, assignment in (("right", right_assignment), ("down", down_assignment)):
        matching = hard_partial_axis_matching(assignment, grid=grid, axis=axis)
        for edge in matching.edges:
            assert result.component_edge_priority[axis][
                edge.source, edge.target
            ] == pytest.approx(edge.confidence)
    assert result.diagnostics.matched_edges_per_axis == {"right": 0, "down": 0}
    assert result.diagnostics.residual_min is None
    assert result.diagnostics.residual_max is None
    assert result.diagnostics.residual_mean is None


def test_duplicate_direct_identity_is_rejected() -> None:
    grid = 2
    assignment = _assignment(grid, 301)
    with pytest.raises(ValueError, match="duplicate"):
        build_direct_residual_union_priority(
            assignment,
            assignment,
            direct_source=[0, 0],
            direct_target=[1, 1],
            direct_axis=[0, 0],
            direct_raw_scores=[0.0, 1.0],
            direct_learned_scores=[1.0, 2.0],
            grid=grid,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"direct_target": [1, 2]}, "identical shapes"),
        ({"direct_source": [0.5]}, "integers"),
        ({"direct_source": [4]}, r"\[0, 4\)"),
        ({"direct_target": [0]}, "self-edges"),
        ({"direct_axis": [2]}, r"0 \(right\) or 1 \(down\)"),
        ({"direct_raw_scores": [np.nan]}, "finite numeric"),
        ({"direct_learned_scores": [[1.0]]}, "one-dimensional"),
    ],
)
def test_malformed_direct_evidence_is_rejected(
    override: dict[str, object],
    message: str,
) -> None:
    grid = 2
    assignment = _assignment(grid, 401)
    kwargs: dict[str, object] = {
        "direct_source": [0],
        "direct_target": [1],
        "direct_axis": [0],
        "direct_raw_scores": [0.0],
        "direct_learned_scores": [1.0],
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        build_direct_residual_union_priority(
            assignment,
            assignment,
            grid=grid,
            **kwargs,
        )


@pytest.mark.parametrize("grid", [True, 1, 2.5])
def test_invalid_grid_is_rejected_before_matching(grid: object) -> None:
    assignment = _assignment(2, 501)
    with pytest.raises(ValueError, match="grid"):
        build_direct_residual_union_priority(
            assignment,
            assignment,
            direct_source=[],
            direct_target=[],
            direct_axis=[],
            direct_raw_scores=[],
            direct_learned_scores=[],
            grid=grid,  # type: ignore[arg-type]
        )


def test_assignment_shape_is_validated_against_grid() -> None:
    assignment = _assignment(2, 601)
    with pytest.raises(ValueError, match="square matrix"):
        build_direct_residual_union_priority(
            assignment[:-1],
            assignment,
            direct_source=[],
            direct_target=[],
            direct_axis=[],
            direct_raw_scores=[],
            direct_learned_scores=[],
            grid=2,
        )


def test_rank_delta_transfer_ignores_common_score_offset_exactly() -> None:
    grid = 3
    count = grid * grid
    right_assignment = _assignment(grid, 701)
    down_assignment = _assignment(grid, 702)
    matchings = (
        hard_partial_axis_matching(right_assignment, grid=grid, axis="right"),
        hard_partial_axis_matching(down_assignment, grid=grid, axis="down"),
    )
    sources = np.asarray(
        [edge.source for matching in matchings for edge in matching.edges],
        dtype=np.int32,
    )
    targets = np.asarray(
        [edge.target for matching in matchings for edge in matching.edges],
        dtype=np.int32,
    )
    axes = np.repeat(np.asarray([0, 1], dtype=np.int8), len(matchings[0].edges))
    raw = np.random.default_rng(703).normal(size=len(sources))
    result = build_direct_rank_delta_union_priority(
        right_assignment,
        down_assignment,
        direct_source=sources,
        direct_target=targets,
        direct_axis=axes,
        direct_raw_scores=raw,
        direct_learned_scores=raw + 100.0,
        grid=grid,
    )

    for axis, matching in zip(("right", "down"), matchings, strict=True):
        matrix = result.component_edge_priority[axis]
        expected = np.zeros((count, count), dtype=np.float64)
        for edge in matching.edges:
            expected[edge.source, edge.target] = edge.confidence
        np.testing.assert_array_equal(matrix, expected)
    assert result.diagnostics.changed_rank_positions_per_axis == {
        "right": 0,
        "down": 0,
    }
    assert result.diagnostics.confidence_multiset_preserved_per_axis == {
        "right": True,
        "down": True,
    }


def test_rank_delta_transfer_moves_edges_but_reuses_union_confidence_multiset() -> None:
    grid = 2
    right_assignment = _assignment(grid, 801)
    down_assignment = _assignment(grid, 802)
    right = hard_partial_axis_matching(right_assignment, grid=grid, axis="right")
    down = hard_partial_axis_matching(down_assignment, grid=grid, axis="down")
    high, low = right.edges
    kwargs = {
        "direct_source": np.asarray([high.source, low.source], dtype=np.int32),
        "direct_target": np.asarray([high.target, low.target], dtype=np.int32),
        "direct_axis": np.asarray([0, 0], dtype=np.int8),
        "direct_raw_scores": np.asarray([2.0, 1.0]),
        "direct_learned_scores": np.asarray([1.0, 2.0]),
        "grid": grid,
    }
    result = build_direct_rank_delta_union_priority(
        right_assignment,
        down_assignment,
        **kwargs,
    )
    reversed_input = build_direct_rank_delta_union_priority(
        right_assignment,
        down_assignment,
        direct_source=kwargs["direct_source"][::-1],
        direct_target=kwargs["direct_target"][::-1],
        direct_axis=kwargs["direct_axis"][::-1],
        direct_raw_scores=kwargs["direct_raw_scores"][::-1],
        direct_learned_scores=kwargs["direct_learned_scores"][::-1],
        grid=grid,
    )

    right_priority = result.component_edge_priority["right"]
    assert right_priority[high.source, high.target] == pytest.approx(low.confidence)
    assert right_priority[low.source, low.target] == pytest.approx(high.confidence)
    np.testing.assert_array_equal(
        right_priority,
        reversed_input.component_edge_priority["right"],
    )
    down_priority = result.component_edge_priority["down"]
    for edge in down.edges:
        assert down_priority[edge.source, edge.target] == pytest.approx(edge.confidence)
    assert result.diagnostics.matched_edges_per_axis == {"right": 2, "down": 0}
    assert result.diagnostics.changed_rank_positions_per_axis == {
        "right": 2,
        "down": 0,
    }
    assert result.diagnostics.rank_delta_min_per_axis == {
        "right": pytest.approx(-1.0),
        "down": None,
    }
    assert result.diagnostics.rank_delta_max_per_axis == {
        "right": pytest.approx(1.0),
        "down": None,
    }
    assert result.report()["schema"] == "aiijc-direct-rank-delta-union-priority-v1"
