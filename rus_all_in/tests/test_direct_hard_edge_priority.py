from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.direct_hard_edge_priority import (
    DirectHardEdgePriority,
    fixed_budget_metrics,
    hard_edge_listwise_loss,
    learned_priority_matrices,
    prepare_direct_hard_edge_board,
    transfer_cyclic_origin_by_baseline_overlap,
)
from aiijc_puzzle.socket_confidence_calibration import extract_hard_edge_features
from aiijc_puzzle.socket_matcher import SocketOutput


def _socket_case(grid: int = 3) -> tuple[torch.Tensor, object, SocketOutput]:
    count = grid * grid
    generator = np.random.default_rng(17)
    raw_right = generator.normal(size=(count, count)).astype(np.float32)
    raw_down = generator.normal(size=(count, count)).astype(np.float32)
    np.fill_diagonal(raw_right, -20.0)
    np.fill_diagonal(raw_down, -20.0)
    right = np.full((count + 1, count + 1), -4.0, dtype=np.float32)
    down = np.full((count + 1, count + 1), -4.0, dtype=np.float32)
    right[:count, :count] = raw_right
    down[:count, :count] = raw_down
    right[:count, count] = generator.normal(size=count)
    right[count, :count] = generator.normal(size=count)
    down[:count, count] = generator.normal(size=count)
    down[count, :count] = generator.normal(size=count)
    right[count, count] = -np.inf
    down[count, count] = -np.inf
    border = [
        torch.from_numpy(generator.normal(size=count).astype(np.float32))[None]
        for _ in range(4)
    ]
    output = SocketOutput(
        right_raw=torch.from_numpy(raw_right)[None],
        down_raw=torch.from_numpy(raw_down)[None],
        right_log_assignment=torch.from_numpy(right)[None],
        down_log_assignment=torch.from_numpy(down)[None],
        right_out_border_logits=border[0],
        left_in_border_logits=border[1],
        bottom_out_border_logits=border[2],
        top_in_border_logits=border[3],
    )
    features = extract_hard_edge_features(
        right_log_assignment=output.right_log_assignment[0],
        down_log_assignment=output.down_log_assignment[0],
        right_raw=output.right_raw[0],
        down_raw=output.down_raw[0],
        grid=grid,
    )
    tokens = torch.from_numpy(generator.normal(size=(count, 8)).astype(np.float32))
    return tokens, features, output


def test_direct_features_and_zero_init_preserve_raw_order() -> None:
    tokens, features, output = _socket_case()
    board = prepare_direct_hard_edge_board(
        tokens,
        features,
        output,
        grid=3,
        provisional_edge_budget_per_axis=2,
    )
    assert board.values.shape == (12, 20 + 4 * 8 + 2 + 18)
    model = DirectHardEdgePriority(board.values.shape[1], hidden_dimension=16)
    score = model(board.values, board.raw_priority, board.axis)
    torch.testing.assert_close(score, board.raw_priority)


def test_board_context_model_is_edge_permutation_equivariant() -> None:
    tokens, features, output = _socket_case()
    board = prepare_direct_hard_edge_board(
        tokens,
        features,
        output,
        grid=3,
        provisional_edge_budget_per_axis=2,
    )
    model = DirectHardEdgePriority(board.values.shape[1], hidden_dimension=16)
    with torch.no_grad():
        model.residual_head[-1].weight.normal_(std=0.1)
    order = torch.tensor([7, 0, 11, 3, 4, 8, 2, 10, 1, 9, 6, 5])
    expected = model(board.values, board.raw_priority, board.axis)
    observed = model(
        board.values[order],
        board.raw_priority[order],
        board.axis[order],
    )
    torch.testing.assert_close(observed, expected[order])


def test_listwise_loss_and_fixed_budget_metrics() -> None:
    scores = torch.tensor([4.0, 3.0, -1.0, -2.0, 2.0, 1.0, 0.0, -3.0])
    labels = torch.tensor([True, True, False, False, True, False, True, False])
    axis = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    loss, diagnostics = hard_edge_listwise_loss(scores, labels, axis)
    assert 0.0 < float(loss) < 1.0
    assert diagnostics["positive_edges"] == 4
    metrics = fixed_budget_metrics(
        scores.numpy(),
        labels.numpy(),
        axis.numpy(),
        edge_budget_per_axis=2,
    )
    assert metrics["correct_selected_edges"] == 3
    assert metrics["selected_edge_precision"] == 0.75


def test_learned_priorities_populate_only_frozen_hard_edges() -> None:
    tokens, features, output = _socket_case()
    board = prepare_direct_hard_edge_board(
        tokens,
        features,
        output,
        grid=3,
        provisional_edge_budget_per_axis=2,
    )
    priorities = learned_priority_matrices(board, board.raw_priority, grid=3)
    assert priorities["right"].shape == (9, 9)
    assert priorities["down"].shape == (9, 9)
    assert np.count_nonzero(priorities["right"]) == 6
    assert np.count_nonzero(priorities["down"]) == 6


def test_baseline_overlap_transfer_recovers_only_a_global_roll() -> None:
    baseline = np.arange(16, dtype=np.int32)
    learned = np.roll(baseline.reshape(4, 4), shift=(1, 2), axis=(0, 1)).reshape(-1)
    result = transfer_cyclic_origin_by_baseline_overlap(learned, baseline, grid=4)
    assert result.overlap_count == 16
    assert (result.row_roll, result.column_roll) == (3, 2)
    assert np.array_equal(result.layout, baseline)
