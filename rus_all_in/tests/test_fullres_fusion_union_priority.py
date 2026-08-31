from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.component_relation_reranker import (
    ComponentRelationCandidate,
    RelationContact,
)
from aiijc_puzzle.fullres_fusion_union_priority import (
    FusionUnionPriorityConfig,
    build_fullres_fusion_union_priority,
)
from aiijc_puzzle.socket_decoder import hard_partial_axis_matching


def _assignment(grid: int, seed: int) -> np.ndarray:
    count = grid * grid
    value = np.random.default_rng(seed).normal(size=(count + 1, count + 1))
    np.fill_diagonal(value[:count, :count], -1e4)
    value[count, count] = -1e4
    return value


def _candidate(
    query: int,
    direction: str,
    source: int,
    target: int,
    *,
    target_component: int | None = None,
    offset: int = 0,
) -> ComponentRelationCandidate:
    return ComponentRelationCandidate(
        source_component=query,
        target_component=(query + 100 if target_component is None else target_component),
        direction=direction,
        target_row_offset=offset,
        target_column_offset=offset,
        contacts=(RelationContact(source, target, ()),),
        proposal_count=1,
        baseline_score=0.0,
    )


def _baseline_by_identity(
    assignment: np.ndarray,
    *,
    grid: int,
    axis: str,
) -> dict[tuple[int, int], float]:
    matching = hard_partial_axis_matching(assignment, grid=grid, axis=axis)
    return {(edge.source, edge.target): edge.confidence for edge in matching.edges}


def test_only_existing_union_hard_edges_receive_scale_normalised_boost() -> None:
    grid = 3
    count = grid * grid
    right_assignment = _assignment(grid, 101)
    down_assignment = _assignment(grid, 102)
    right = hard_partial_axis_matching(right_assignment, grid=grid, axis="right")
    down = hard_partial_axis_matching(down_assignment, grid=grid, axis="down")
    right_edge = right.edges[0]
    down_edge = down.edges[-1]
    candidate = (
        _candidate(0, "right", right_edge.source, right_edge.target),
        _candidate(1, "down", down_edge.source, down_edge.target),
    )
    result = build_fullres_fusion_union_priority(
        right_assignment,
        down_assignment,
        candidate,
        fusion_scores=[2.0, 1.0],
        confidence_logits=[0.0, 0.0],
        grid=grid,
    )

    right_scale = float(np.std([edge.confidence for edge in right.edges]))
    down_scale = float(np.std([edge.confidence for edge in down.edges]))
    assert result.component_edge_priority["right"][
        right_edge.source, right_edge.target
    ] == pytest.approx(right_edge.confidence + 0.5 * right_scale)
    assert result.component_edge_priority["down"][
        down_edge.source, down_edge.target
    ] == pytest.approx(down_edge.confidence + 0.5 * down_scale)

    right_baseline = _baseline_by_identity(right_assignment, grid=grid, axis="right")
    down_baseline = _baseline_by_identity(down_assignment, grid=grid, axis="down")
    for source in range(count):
        for target in range(count):
            if (source, target) not in right_baseline:
                assert result.component_edge_priority["right"][source, target] == 0.0
            if (source, target) not in down_baseline:
                assert result.component_edge_priority["down"][source, target] == 0.0
    assert result.diagnostics.supported_hard_edges_per_axis == {
        "right": 1,
        "down": 1,
    }
    assert result.report()["legality"]["new_hard_edges_introduced"] is False


def test_query_confidence_cap_and_within_query_rank_are_both_applied() -> None:
    grid = 3
    right_assignment = _assignment(grid, 201)
    down_assignment = _assignment(grid, 202)
    right = hard_partial_axis_matching(right_assignment, grid=grid, axis="right")
    first, second, ignored = right.edges[:3]
    candidates = (
        _candidate(0, "right", first.source, first.target, offset=1),
        _candidate(0, "right", second.source, second.target, offset=2),
        _candidate(1, "right", ignored.source, ignored.target, offset=3),
    )
    result = build_fullres_fusion_union_priority(
        right_assignment,
        down_assignment,
        candidates,
        fusion_scores=[4.0, 3.0, 100.0],
        confidence_logits=[2.0, 0.0, -2.0],
        grid=grid,
        config=FusionUnionPriorityConfig(query_cap=1, candidate_rank_cap=2),
    )
    scale = float(np.std([edge.confidence for edge in right.edges]))
    sigmoid_two = 1.0 / (1.0 + np.exp(-2.0))
    assert result.component_edge_priority["right"][
        first.source, first.target
    ] == pytest.approx(first.confidence + scale * sigmoid_two)
    assert result.component_edge_priority["right"][
        second.source, second.target
    ] == pytest.approx(second.confidence + scale * 0.25)
    assert result.component_edge_priority["right"][
        ignored.source, ignored.target
    ] == pytest.approx(ignored.confidence)
    assert result.diagnostics.selected_query_count == 1
    assert result.diagnostics.considered_candidate_count == 2


def test_left_and_up_contacts_canonicalise_to_right_and_down_hard_edges() -> None:
    grid = 3
    right_assignment = _assignment(grid, 301)
    down_assignment = _assignment(grid, 302)
    right = hard_partial_axis_matching(right_assignment, grid=grid, axis="right")
    down = hard_partial_axis_matching(down_assignment, grid=grid, axis="down")
    horizontal = right.edges[0]
    vertical = down.edges[0]
    candidates = (
        _candidate(0, "left", horizontal.target, horizontal.source),
        _candidate(1, "up", vertical.target, vertical.source),
    )
    result = build_fullres_fusion_union_priority(
        right_assignment,
        down_assignment,
        candidates,
        fusion_scores=[1.0, 1.0],
        confidence_logits=[0.0, 0.0],
        grid=grid,
    )
    assert result.component_edge_priority["right"][
        horizontal.source, horizontal.target
    ] > horizontal.confidence
    assert result.component_edge_priority["down"][
        vertical.source, vertical.target
    ] > vertical.confidence


def test_reversible_consensus_uses_bounded_noisy_or() -> None:
    grid = 3
    right_assignment = _assignment(grid, 401)
    down_assignment = _assignment(grid, 402)
    edge = hard_partial_axis_matching(
        right_assignment,
        grid=grid,
        axis="right",
    ).edges[0]
    candidates = (
        _candidate(0, "right", edge.source, edge.target),
        _candidate(1, "left", edge.target, edge.source),
    )
    result = build_fullres_fusion_union_priority(
        right_assignment,
        down_assignment,
        candidates,
        fusion_scores=[1.0, 1.0],
        confidence_logits=[0.0, 0.0],
        grid=grid,
    )
    scale = result.diagnostics.hard_confidence_scale_per_axis["right"]
    assert result.component_edge_priority["right"][
        edge.source, edge.target
    ] == pytest.approx(edge.confidence + 0.75 * scale)
    assert result.diagnostics.maximum_support_count == 2
    assert result.diagnostics.confidence_signal_max == pytest.approx(0.75)


def test_relation_input_permutation_is_bitwise_invariant() -> None:
    grid = 3
    right_assignment = _assignment(grid, 501)
    down_assignment = _assignment(grid, 502)
    edges = hard_partial_axis_matching(
        right_assignment,
        grid=grid,
        axis="right",
    ).edges[:3]
    candidates = tuple(
        _candidate(index, "right", edge.source, edge.target, offset=index)
        for index, edge in enumerate(edges)
    )
    scores = np.asarray([3.0, 2.0, 1.0])
    logits = np.asarray([-1.0, 2.0, 0.5])
    first = build_fullres_fusion_union_priority(
        right_assignment,
        down_assignment,
        candidates,
        scores,
        logits,
        grid=grid,
    )
    order = np.asarray([2, 0, 1])
    second = build_fullres_fusion_union_priority(
        right_assignment,
        down_assignment,
        tuple(candidates[index] for index in order),
        scores[order],
        logits[order],
        grid=grid,
    )
    assert first.report() == second.report()
    np.testing.assert_array_equal(
        first.component_edge_priority["right"],
        second.component_edge_priority["right"],
    )
    np.testing.assert_array_equal(
        first.component_edge_priority["down"],
        second.component_edge_priority["down"],
    )


def test_no_hard_edge_overlap_preserves_baseline_priority_exactly() -> None:
    grid = 2
    count = grid * grid
    right_assignment = _assignment(grid, 601)
    down_assignment = _assignment(grid, 602)
    right_identities = {
        (edge.source, edge.target)
        for edge in hard_partial_axis_matching(
            right_assignment,
            grid=grid,
            axis="right",
        ).edges
    }
    unused = next(
        (source, target)
        for source in range(count)
        for target in range(count)
        if source != target and (source, target) not in right_identities
    )
    result = build_fullres_fusion_union_priority(
        right_assignment,
        down_assignment,
        (_candidate(0, "right", *unused),),
        fusion_scores=[1.0],
        confidence_logits=[10.0],
        grid=grid,
    )
    for axis, assignment in (("right", right_assignment), ("down", down_assignment)):
        matching = hard_partial_axis_matching(assignment, grid=grid, axis=axis)
        for edge in matching.edges:
            assert result.component_edge_priority[axis][
                edge.source, edge.target
            ] == edge.confidence
    assert result.diagnostics.supported_hard_edges_per_axis == {
        "right": 0,
        "down": 0,
    }
    assert result.diagnostics.priority_boost_mean is None


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"fusion_scores": []}, "finite shape"),
        ({"confidence_logits": [np.nan]}, "finite shape"),
        ({"candidates": []}, "non-empty tuple"),
        ({"grid": True}, "grid must be an integer"),
        (
            {"config": FusionUnionPriorityConfig(query_cap=0)},
            "caps must be positive",
        ),
        (
            {"config": FusionUnionPriorityConfig(boost_scale=-1.0)},
            "boost_scale",
        ),
    ],
)
def test_malformed_inputs_fail_closed(
    override: dict[str, object],
    message: str,
) -> None:
    grid = 2
    assignment = _assignment(grid, 701)
    kwargs: dict[str, object] = {
        "right_log_assignment": assignment,
        "down_log_assignment": assignment,
        "candidates": (_candidate(0, "right", 0, 1),),
        "fusion_scores": [0.0],
        "confidence_logits": [0.0],
        "grid": grid,
    }
    kwargs.update(override)
    with pytest.raises((TypeError, ValueError), match=message):
        build_fullres_fusion_union_priority(**kwargs)  # type: ignore[arg-type]


def test_duplicate_relation_key_and_bad_contact_fail_closed() -> None:
    grid = 2
    assignment = _assignment(grid, 801)
    candidate = _candidate(0, "right", 0, 1)
    with pytest.raises(ValueError, match="duplicate relation_key"):
        build_fullres_fusion_union_priority(
            assignment,
            assignment,
            (candidate, candidate),
            [1.0, 0.0],
            [1.0, 0.0],
            grid=grid,
        )
    invalid = _candidate(0, "right", 0, grid * grid)
    with pytest.raises(ValueError, match="outside the board"):
        build_fullres_fusion_union_priority(
            assignment,
            assignment,
            (invalid,),
            [1.0],
            [1.0],
            grid=grid,
        )
