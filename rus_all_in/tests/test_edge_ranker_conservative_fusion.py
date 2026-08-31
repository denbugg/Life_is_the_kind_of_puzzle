from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aiijc_puzzle.edge_ranker import EdgeBoard, EdgeRow
from aiijc_puzzle.edge_ranker_conservative_fusion import (
    FusionArm,
    apply_conservative_fusion,
    learned_mutual_proposals,
)


def _board() -> EdgeBoard:
    baseline = np.asarray(
        [
            [-10.0, 5.0, 1.0, 1.0],
            [4.0, -10.0, 5.0, 1.0],
            [4.0, 5.0, -10.0, 6.0],
            [4.0, 5.0, 6.0, -10.0],
        ],
        dtype=np.float32,
    )
    rows = []
    for direction in (0, 1):
        for source in range(4):
            candidates = np.asarray([target for target in range(4) if target != source])
            features = np.zeros((3, 12), dtype=np.float32)
            features[:, 1::3] = np.asarray([[0.0], [0.2], [0.4]])
            features[:, 2::3] = 1.0
            rows.append(
                EdgeRow(
                    anchor=source,
                    candidates=candidates,
                    features=features,
                    baseline_scores=baseline[source, candidates],
                    direction=direction,
                )
            )
    return EdgeBoard(
        filename="synthetic.png",
        tiles=np.zeros((4, 20, 20, 3), dtype=np.uint8),
        rows=tuple(rows),
        right_baseline=baseline,
        down_baseline=baseline,
        views=("raw", "tile_z", "bilateral", "gray"),
        candidate_k=16,
    )


def test_arm_validation() -> None:
    with pytest.raises(ValueError):
        FusionArm("", 4, 2, 0.0)
    with pytest.raises(ValueError):
        FusionArm("x", 97, 2, 0.0)
    with pytest.raises(ValueError):
        FusionArm("x", 4, 5, 0.0)


def test_fusion_preserves_existing_mutual_edges_and_promotes_disjoint_proposal() -> None:
    board = _board()
    # Existing mutual edges occupy sources 0/2/3 and targets 1/3/2.  Create a
    # learned mutual edge 1 -> 0 using the only disjoint source/target pair.
    learned = board.right_baseline.copy()
    learned[1, 0] = 8.0
    proposals = learned_mutual_proposals(board, learned, board.down_baseline)
    assert any((item.source, item.target) == (1, 0) for item in proposals)
    right, _, diagnostics = apply_conservative_fusion(
        board,
        learned,
        board.down_baseline,
        FusionArm("test", 1, 0, 0.0),
    )
    assert np.argmax(right[0]) == 1
    assert np.argmax(right[1]) == 0
    assert diagnostics["selected_count"] == 1


def test_nonpositive_delta_is_not_proposed() -> None:
    board = _board()
    learned = board.right_baseline.copy()
    # A learned mutual ordering achieved only by lowering competitors must not
    # introduce an edge: the promoted pair itself has no positive model signal.
    learned[2] = np.asarray([1.5, 1.0, -10.0, 0.5], dtype=np.float32)
    proposals = learned_mutual_proposals(board, learned, board.down_baseline)
    assert all((item.direction, item.source, item.target) != (0, 2, 0) for item in proposals)


def test_zero_cap_is_exact_baseline_identity() -> None:
    board = _board()
    learned = replace(board, right_baseline=board.right_baseline + 1.0).right_baseline
    right, down, diagnostics = apply_conservative_fusion(
        board,
        learned,
        learned,
        FusionArm("identity", 0, 0, 0.0),
    )
    np.testing.assert_array_equal(right, board.right_baseline)
    np.testing.assert_array_equal(down, board.down_baseline)
    assert diagnostics["selected_count"] == 0
