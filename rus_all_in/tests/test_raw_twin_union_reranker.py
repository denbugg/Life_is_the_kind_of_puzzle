from __future__ import annotations

import numpy as np
import pytest
import torch

from aiijc_puzzle.fullres_twin_side_matcher import FullResolutionTwinSideMatcher
from aiijc_puzzle.raw_twin_union_reranker import (
    FEATURE_NAMES,
    RawTwinUnionBoard,
    RawTwinUnionReranker,
    bidirectional_union_loss,
    candidate_score_matrices,
    prepare_raw_twin_union_board,
    restricted_partial_ot,
    union_edge_labels,
)
from aiijc_puzzle.socket_decoder import hard_partial_axis_matching
from aiijc_puzzle.socket_matcher import SocketOutput, partial_log_optimal_transport


def _fake_frozen_outputs(grid: int = 4) -> tuple[torch.Tensor, SocketOutput, object]:
    torch.manual_seed(3)
    count = grid * grid
    raw_right = torch.randn(1, count, count)
    raw_down = torch.randn(1, count, count)
    diagonal = torch.eye(count, dtype=torch.bool).unsqueeze(0)
    raw_right = raw_right.masked_fill(diagonal, -1e4)
    raw_down = raw_down.masked_fill(diagonal, -1e4)
    border = torch.zeros(1, count)
    socket = SocketOutput(
        right_raw=raw_right,
        down_raw=raw_down,
        right_log_assignment=partial_log_optimal_transport(
            raw_right,
            border,
            unmatched=grid,
        ),
        down_log_assignment=partial_log_optimal_transport(
            raw_down,
            border,
            unmatched=grid,
        ),
        right_out_border_logits=border,
        left_in_border_logits=border,
        bottom_out_border_logits=border,
        top_in_border_logits=border,
    )
    tokens = torch.randn(1, count, 64)
    twin = FullResolutionTwinSideMatcher(
        dimension=48,
        field_blocks=1,
        sequence_blocks=1,
    )(torch.rand(1, count, 3, 20, 20))
    return tokens, socket, twin


def _small_board() -> RawTwinUnionBoard:
    grid = 4
    count = grid * grid
    source = []
    target = []
    axis = []
    rows = []
    for direction in range(2):
        direction_rows = []
        for tile in range(count):
            candidates = np.asarray([item for item in range(count) if item != tile])
            direction_rows.append(candidates)
            source.extend([tile] * len(candidates))
            target.extend(candidates.tolist())
            axis.extend([direction] * len(candidates))
        rows.append(tuple(direction_rows))
    edge_count = len(source)
    return RawTwinUnionBoard(
        values=torch.randn(edge_count, len(FEATURE_NAMES)),
        raw_scores=torch.randn(edge_count),
        axis=torch.tensor(axis, dtype=torch.long),
        source=torch.tensor(source, dtype=torch.long),
        target=torch.tensor(target, dtype=torch.long),
        rows=(rows[0], rows[1]),
        grid=grid,
    )


def test_feature_builder_preserves_union_and_frozen_token_contract() -> None:
    tokens, socket, twin = _fake_frozen_outputs()
    board = prepare_raw_twin_union_board(tokens, socket, twin, grid=4, topk=5)
    assert board.values.shape[1] == len(FEATURE_NAMES) == 280
    assert len(board.values) == sum(len(row) for axis in board.rows for row in axis)
    assert all(5 <= len(row) <= 11 for axis in board.rows for row in axis)
    assert torch.isfinite(board.values).all()
    assert torch.isfinite(board.raw_scores).all()
    for axis, assignment in enumerate((socket.right_log_assignment, socket.down_log_assignment)):
        matching = hard_partial_axis_matching(
            assignment,
            grid=4,
            axis="right" if axis == 0 else "down",
        )
        assert all(edge.target in board.rows[axis][edge.source] for edge in matching.edges)


def test_zero_initialisation_exactly_preserves_raw_union_scores() -> None:
    board = _small_board()
    model = RawTwinUnionReranker()
    output = model(board)
    assert torch.equal(output.scores, board.raw_scores)
    assert torch.count_nonzero(output.residual).item() == 0


def test_model_is_equivariant_to_candidate_edge_order() -> None:
    torch.manual_seed(8)
    board = _small_board()
    model = RawTwinUnionReranker().eval()
    permutation = torch.randperm(len(board.values))
    permuted = RawTwinUnionBoard(
        values=board.values[permutation],
        raw_scores=board.raw_scores[permutation],
        axis=board.axis[permutation],
        source=board.source[permutation],
        target=board.target[permutation],
        rows=board.rows,
        grid=board.grid,
    )
    with torch.no_grad():
        reference = model(board).scores
        observed = model(permuted).scores
    torch.testing.assert_close(observed, reference[permutation])


def test_model_is_equivariant_to_arbitrary_tile_identity_relabelling() -> None:
    torch.manual_seed(9)
    board = _small_board()
    model = RawTwinUnionReranker().eval()
    relabel = torch.randperm(16)
    renamed = RawTwinUnionBoard(
        values=board.values,
        raw_scores=board.raw_scores,
        axis=board.axis,
        source=relabel[board.source],
        target=relabel[board.target],
        rows=board.rows,
        grid=board.grid,
    )
    with torch.no_grad():
        torch.testing.assert_close(model(renamed).scores, model(board).scores)


def test_bidirectional_loss_has_finite_gradient_and_both_group_families() -> None:
    torch.manual_seed(10)
    board = _small_board()
    model = RawTwinUnionReranker()
    layout = torch.randperm(16)
    labels = union_edge_labels(board, layout)
    output = model(board)
    loss, diagnostics = bidirectional_union_loss(output, board, labels)
    assert torch.isfinite(loss)
    assert diagnostics["row_supervised_groups"] == 24
    assert diagnostics["column_supervised_groups"] == 24
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_outside_union_is_forbidden_before_partial_ot() -> None:
    tokens, socket, twin = _fake_frozen_outputs()
    board = prepare_raw_twin_union_board(tokens, socket, twin, grid=4, topk=3)
    right, down = candidate_score_matrices(board, board.raw_scores)
    matrices = (right[0], down[0])
    for direction, matrix in enumerate(matrices):
        for source, candidates in enumerate(board.rows[direction]):
            outside = np.setdiff1d(np.arange(16), candidates)
            assert torch.all(matrix[source, torch.from_numpy(outside)] == -1e4)
    right_ot, down_ot = restricted_partial_ot(board, board.raw_scores, socket)
    assert right_ot.shape == down_ot.shape == (1, 17, 17)
    assert torch.isfinite(right_ot).all() and torch.isfinite(down_ot).all()
    for axis, assignment in enumerate((right_ot, down_ot)):
        name = "right" if axis == 0 else "down"
        matching = hard_partial_axis_matching(
            assignment,
            grid=4,
            axis=name,
        )
        assert len(matching.edges) == 12
        assert all(edge.target in board.rows[axis][edge.source] for edge in matching.edges)
        full_raw = hard_partial_axis_matching(
            socket.right_log_assignment if axis == 0 else socket.down_log_assignment,
            grid=4,
            axis=name,
        )
        assert {(edge.source, edge.target) for edge in matching.edges} == {
            (edge.source, edge.target) for edge in full_raw.edges
        }


def test_strict_layout_labels_reject_duplicates() -> None:
    board = _small_board()
    with pytest.raises(ValueError, match="strict grid permutation"):
        union_edge_labels(board, torch.zeros(16, dtype=torch.long))
