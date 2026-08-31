from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.taska_relation_ranked_union import (
    rank_relation_edge_union,
    solve_relation_ranked_union,
)
from aiijc_puzzle.taska_relation_truth_selector import (
    FEATURE_NAMES,
    RelationFeatureBoard,
    realised_edges,
)


class FeatureProbabilityModel:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.asarray(features[:, 0], dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def _board() -> RelationFeatureBoard:
    layouts = (
        np.asarray([0, 1, 2, 3]),
        np.asarray([0, 1, 3, 2]),
        np.asarray([1, 0, 2, 3]),
        np.asarray([2, 3, 0, 1]),
        np.asarray([3, 2, 1, 0]),
        np.asarray([0, 2, 1, 3]),
    )
    edges = tuple(realised_edges(layout, grid=2) for layout in layouts)
    features = np.zeros((6, 4, len(FEATURE_NAMES)), dtype=np.float64)
    features[:, :, 0] = np.linspace(0.05, 0.95, 24).reshape(6, 4)
    # The first edge is shared by arms 0 and 1.  An exact max tie must retain
    # arm 0; a strictly larger value must instead select arm 1.
    assert edges[0][0] == edges[1][0]
    features[0, 0, 0] = 0.77
    features[1, 0, 0] = 0.77
    return RelationFeatureBoard(
        layouts=layouts,
        edges=edges,
        features=features,
        control_choice="raw",
        grid_size=2,
    )


def test_union_uses_all_unique_edges_max_probability_and_stable_ties() -> None:
    board = _board()
    union = rank_relation_edge_union(board, FeatureProbabilityModel())
    all_edges = {edge for arm_edges in board.edges for edge in arm_edges}
    assert set(union.edges) == all_edges
    assert union.occurrence_count == 24
    assert union.duplicate_occurrence_count == 24 - len(all_edges)
    shared_index = union.edges.index(board.edges[0][0])
    assert union.probabilities[shared_index] == pytest.approx(0.77)
    assert union.winning_arm_indices[shared_index] == 0
    assert np.all(union.probabilities[:-1] >= union.probabilities[1:])
    assert not union.probabilities.flags.writeable


def test_strictly_larger_duplicate_occurrence_wins() -> None:
    board = _board()
    features = board.features.copy()
    features[1, 0, 0] = 0.78
    changed = RelationFeatureBoard(
        layouts=board.layouts,
        edges=board.edges,
        features=features,
        control_choice=board.control_choice,
        grid_size=2,
    )
    union = rank_relation_edge_union(changed, FeatureProbabilityModel())
    shared_index = union.edges.index(board.edges[0][0])
    assert union.probabilities[shared_index] == pytest.approx(0.78)
    assert union.winning_arm_indices[shared_index] == 1


def test_all_edge_union_returns_read_only_strict_permutation() -> None:
    board = _board()
    count = 4
    right = np.full((count, count), 5.0, dtype=np.float64)
    down = np.full((count, count), 5.0, dtype=np.float64)
    np.fill_diagonal(right, 0.0)
    np.fill_diagonal(down, 0.0)
    result = solve_relation_ranked_union(
        board,
        FeatureProbabilityModel(),
        right,
        down,
    )
    assert np.array_equal(np.sort(result.layout), np.arange(count))
    assert not result.layout.flags.writeable
    assert result.solver_diagnostics.candidate_edges == len(result.union.edges)
    assert result.diagnostics()["all_unique_edges_used"] is True
    assert result.diagnostics()["threshold"] is None
    assert result.diagnostics()["top_k"] is None
