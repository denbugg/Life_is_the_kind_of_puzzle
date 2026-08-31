from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from aiijc_puzzle.component_relation_reranker import (
    ComponentRelationCandidate,
    RelationContact,
)
from aiijc_puzzle.fullres_fusion_snapshot import (
    build_fullres_fusion_snapshot,
    fusion_candidates_to_union_snapshot,
)


def _contact(source: int, target: int) -> RelationContact:
    return RelationContact(source_tile=source, target_tile=target, features=())


def _candidate(
    direction: str,
    *contacts: tuple[int, int],
    source_component: int = 0,
    target_component: int = 1,
) -> ComponentRelationCandidate:
    offsets = {
        "right": (0, 1),
        "down": (1, 0),
        "left": (0, -1),
        "up": (-1, 0),
    }
    row_offset, column_offset = offsets.get(direction, (0, 0))
    return ComponentRelationCandidate(
        source_component=source_component,
        target_component=target_component,
        direction=direction,
        target_row_offset=row_offset,
        target_column_offset=column_offset,
        contacts=tuple(_contact(source, target) for source, target in contacts),
        proposal_count=1,
        baseline_score=0.0,
    )


def test_all_directions_are_canonicalised_to_right_and_down_edges() -> None:
    candidates = (
        _candidate("right", (0, 1)),
        _candidate("left", (2, 3)),
        _candidate("down", (4, 5)),
        _candidate("up", (6, 7)),
    )
    snapshot, diagnostics = build_fullres_fusion_snapshot(
        candidates,
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        grid=3,
    )

    np.testing.assert_array_equal(snapshot.axis, [0, 0, 1, 1])
    np.testing.assert_array_equal(snapshot.source, [0, 3, 4, 7])
    np.testing.assert_array_equal(snapshot.target, [1, 2, 5, 6])
    np.testing.assert_allclose(snapshot.scores, [0.1, 0.2, 0.3, 0.4], rtol=0, atol=1e-7)
    assert diagnostics.relation_count == 4
    assert diagnostics.contact_count == 4
    assert diagnostics.unique_edge_count == 4
    assert diagnostics.duplicate_contact_count == 0
    assert diagnostics.direction_relation_counts == (1, 1, 1, 1)
    assert diagnostics.direction_contact_counts == (1, 1, 1, 1)


def test_relation_and_contact_permutations_produce_identical_snapshot() -> None:
    first_candidates = (
        _candidate("right", (2, 3), (0, 1), source_component=2),
        _candidate("left", (1, 0), source_component=1),
        _candidate("down", (4, 5), source_component=4),
    )
    first_scores = np.asarray([-1.25, 0.75, 2.0])
    second_candidates = (
        first_candidates[2],
        replace(first_candidates[0], contacts=tuple(reversed(first_candidates[0].contacts))),
        first_candidates[1],
    )
    second_scores = np.asarray([2.0, -1.25, 0.75])

    first, first_diagnostics = build_fullres_fusion_snapshot(
        first_candidates,
        first_scores,
        grid=3,
    )
    second, second_diagnostics = build_fullres_fusion_snapshot(
        second_candidates,
        second_scores,
        grid=3,
    )

    assert first.sha256 == second.sha256
    np.testing.assert_array_equal(first.axis, second.axis)
    np.testing.assert_array_equal(first.source, second.source)
    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.scores, second.scores)
    assert first_diagnostics == second_diagnostics


def test_duplicate_canonical_edges_use_deterministic_logsumexp() -> None:
    candidates = (
        _candidate("right", (0, 1), (0, 1)),
        _candidate("left", (1, 0)),
        _candidate("right", (0, 1)),
    )
    snapshot, diagnostics = build_fullres_fusion_snapshot(
        candidates,
        np.asarray([0.0, math.log(3.0), math.log(4.0)]),
        grid=2,
    )

    assert snapshot.count == 1
    assert (int(snapshot.axis[0]), int(snapshot.source[0]), int(snapshot.target[0])) == (0, 0, 1)
    # The first relation contributes twice: 1 + 1 + 3 + 4 = 9 mass.
    assert float(snapshot.scores[0]) == pytest.approx(math.log(9.0))
    assert diagnostics.relation_count == 3
    assert diagnostics.contact_count == 4
    assert diagnostics.unique_edge_count == 1
    assert diagnostics.duplicate_contact_count == 3
    assert diagnostics.direction_relation_counts == (2, 0, 1, 0)
    assert diagnostics.direction_contact_counts == (3, 0, 1, 0)

    wrapper_snapshot = fusion_candidates_to_union_snapshot(
        candidates,
        np.asarray([0.0, math.log(3.0), math.log(4.0)]),
        grid=2,
    )
    assert wrapper_snapshot.sha256 == snapshot.sha256


@pytest.mark.parametrize(
    ("candidates", "scores", "grid", "message"),
    [
        ((), np.empty(0), 2, "must not be empty"),
        ((_candidate("right", (0, 1)),), np.asarray([0.0, 1.0]), 2, "shape"),
        ((_candidate("right", (0, 1)),), np.asarray([np.nan]), 2, "finite"),
        ((_candidate("diagonal", (0, 1)),), np.asarray([0.0]), 2, "unknown direction"),
        ((_candidate("right"),), np.asarray([0.0]), 2, "at least one contact"),
        ((_candidate("right", (1, 1)),), np.asarray([0.0]), 2, "self"),
        ((_candidate("right", (0, 4)),), np.asarray([0.0]), 2, "outside"),
        ((_candidate("right", (0, 1)),), np.asarray([0.0]), 1, "grid"),
    ],
)
def test_malformed_inputs_are_rejected(
    candidates: tuple[ComponentRelationCandidate, ...],
    scores: np.ndarray,
    grid: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_fullres_fusion_snapshot(candidates, scores, grid=grid)


def test_non_candidate_and_non_contact_entries_are_rejected() -> None:
    with pytest.raises(TypeError, match="ComponentRelationCandidate"):
        build_fullres_fusion_snapshot((object(),), np.asarray([0.0]), grid=2)  # type: ignore[arg-type]

    malformed = replace(_candidate("right", (0, 1)), contacts=(object(),))
    with pytest.raises(TypeError, match="RelationContact"):
        build_fullres_fusion_snapshot((malformed,), np.asarray([0.0]), grid=2)
