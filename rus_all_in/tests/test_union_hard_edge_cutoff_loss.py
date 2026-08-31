from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from aiijc_puzzle.union_hard_edge_cutoff_loss import (
    CUTOFF_EXCHANGE_LOSS_SCHEMA,
    CUTOFF_EXCHANGE_RESIDUAL_WEIGHT,
    union_hard_edge_cutoff_exchange_loss,
    union_hard_edge_cutoff_membership,
)
from aiijc_puzzle.union_hard_edge_priority import (
    FEATURE_NAMES,
    UnionHardEdgeBoard,
    UnionHardEdgeOutput,
    UnionHardEdgePriority,
)


def _board(*, edge_budget_per_axis: int = 2) -> UnionHardEdgeBoard:
    return UnionHardEdgeBoard(
        values=torch.linspace(-1.0, 1.0, 12 * len(FEATURE_NAMES)).reshape(
            12, len(FEATURE_NAMES)
        ),
        base_priority=torch.tensor(
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        ),
        priority_scale=torch.ones(12),
        axis=torch.tensor([0] * 6 + [1] * 6, dtype=torch.long),
        source=np.asarray([5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5], dtype=np.int32),
        target=np.asarray([6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
        grid=3,
        edge_budget_per_axis=edge_budget_per_axis,
        direct_matches_per_axis=(0, 0),
        fullres_supported_per_axis=(0, 0),
    )


def _output(scores: torch.Tensor, *, normalised: torch.Tensor | None = None) -> UnionHardEdgeOutput:
    if normalised is None:
        normalised = torch.zeros_like(scores)
    return UnionHardEdgeOutput(
        scores=scores,
        residual=torch.zeros_like(scores),
        normalised_residual=normalised,
    )


def _labels() -> torch.Tensor:
    # Per axis: current top-2 contains one false edge (index 1/8), while one
    # true edge immediately below the cutoff (index 2/7) is missed.
    return torch.tensor(
        [True, False, True, False, False, False, True, True, False, False, False, False]
    )


def test_loss_is_exact_current_cutoff_exchange_objective() -> None:
    board = _board()
    scores = torch.tensor(
        [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 10.0, 8.0, 9.0, 7.0, 6.0, 5.0]
    )
    loss, diagnostics = union_hard_edge_cutoff_exchange_loss(
        _output(scores),
        board,
        _labels(),
    )
    expected = F.softplus(torch.tensor(1.0))
    torch.testing.assert_close(loss, expected)
    assert diagnostics == {
        "schema": CUTOFF_EXCHANGE_LOSS_SCHEMA,
        "loss": pytest.approx(float(expected)),
        "cutoff_exchange_loss": pytest.approx(float(expected)),
        "normalised_residual_l2": 0.0,
        "residual_weight": CUTOFF_EXCHANGE_RESIDUAL_WEIGHT,
        "edge_budget_per_axis": 2,
        "selected_edges": 4,
        "correct_selected_edges": 2,
        "false_selected_edges": 2,
        "missed_true_edges": 2,
        "exchange_pairs": 2,
        "active_axes": 2,
    }


def test_gradient_lowers_false_selected_and_raises_missed_true_scores() -> None:
    board = _board()
    scores = torch.tensor(
        [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 10.0, 8.0, 9.0, 7.0, 6.0, 5.0],
        requires_grad=True,
    )
    loss, _ = union_hard_edge_cutoff_exchange_loss(_output(scores), board, _labels())
    loss.backward()
    assert scores.grad is not None
    assert scores.grad[1] > 0 and scores.grad[8] > 0
    assert scores.grad[2] < 0 and scores.grad[7] < 0
    untouched = torch.tensor([0, 3, 4, 5, 6, 9, 10, 11])
    torch.testing.assert_close(scores.grad[untouched], torch.zeros(len(untouched)))


def test_exchange_term_is_relative_not_a_frozen_absolute_threshold_hinge() -> None:
    board = _board()
    scores = torch.tensor(
        [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 10.0, 8.0, 9.0, 7.0, 6.0, 5.0]
    )
    original, original_diagnostics = union_hard_edge_cutoff_exchange_loss(
        _output(scores), board, _labels()
    )
    shifted, shifted_diagnostics = union_hard_edge_cutoff_exchange_loss(
        _output(scores + 123.0), board, _labels()
    )
    torch.testing.assert_close(shifted, original)
    assert shifted_diagnostics["cutoff_exchange_loss"] == pytest.approx(
        original_diagnostics["cutoff_exchange_loss"]
    )


def test_membership_uses_decoder_tie_break_at_cutoff() -> None:
    board = _board(edge_budget_per_axis=1)
    # On axis 0, indices 0..2 tie on learned score and base score.  Immutable
    # identity breaks the tie: source 3 (index 2) precedes sources 4 and 5.
    scores = torch.tensor([5.0, 5.0, 5.0, 0.0, 0.0, 0.0] + [6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    board = replace(
        board,
        base_priority=torch.tensor(
            [3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        ),
    )
    first = union_hard_edge_cutoff_membership(_output(scores), board)
    second = union_hard_edge_cutoff_membership(_output(scores.clone()), board)
    torch.testing.assert_close(first, second)
    assert torch.nonzero(first, as_tuple=False).flatten().tolist() == [2, 6]


def test_existing_priority_checkpoint_state_can_continue_without_architecture_change() -> None:
    torch.manual_seed(19)
    board = _board()
    frozen = UnionHardEdgePriority(hidden_dimension=8)
    with torch.no_grad():
        frozen.residual_head[-1].weight.normal_(std=0.05)
        frozen.residual_head[-1].bias.fill_(0.01)
    checkpoint = {name: value.detach().clone() for name, value in frozen.state_dict().items()}

    continued = UnionHardEdgePriority(hidden_dimension=8)
    continued.load_state_dict(checkpoint, strict=True)
    expected = frozen(board)
    observed = continued(board)
    torch.testing.assert_close(observed.scores, expected.scores)

    loss, diagnostics = union_hard_edge_cutoff_exchange_loss(observed, board, _labels())
    loss.backward()
    assert diagnostics["edge_budget_per_axis"] == 2
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in continued.parameters()
    )


def test_already_optimal_axis_contributes_zero_but_residual_penalty_remains() -> None:
    board = _board()
    scores = torch.tensor(
        [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0]
    )
    labels = torch.tensor(
        [True, True, False, False, False, False, True, False, True, False, False, False]
    )
    normalised = torch.full_like(scores, 2.0)
    loss, diagnostics = union_hard_edge_cutoff_exchange_loss(
        _output(scores, normalised=normalised),
        board,
        labels,
    )
    # Axis 0 is already optimal and is still included with zero weight; axis 1
    # has one softplus(1) exchange.  The fixed residual L2 is 4.
    expected = 0.5 * F.softplus(torch.tensor(1.0)) + 4 * CUTOFF_EXCHANGE_RESIDUAL_WEIGHT
    torch.testing.assert_close(loss, expected)
    assert diagnostics["active_axes"] == 1
    assert diagnostics["false_selected_edges"] == 1
    assert diagnostics["missed_true_edges"] == 1


def test_malformed_labels_and_nonfinite_outputs_fail_closed() -> None:
    board = _board()
    scores = board.base_priority.clone()
    with pytest.raises(ValueError, match="boolean vector"):
        union_hard_edge_cutoff_exchange_loss(_output(scores), board, _labels().float())
    invalid = scores.clone()
    invalid[0] = float("nan")
    with pytest.raises(ValueError, match="finite floating-point"):
        union_hard_edge_cutoff_membership(_output(invalid), board)

    no_axis_positive = _labels()
    no_axis_positive[:6] = False
    with pytest.raises(ValueError, match="both true and false"):
        union_hard_edge_cutoff_exchange_loss(_output(scores), board, no_axis_positive)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_membership_moves_to_cpu_before_float64_tie_sort_on_mps() -> None:
    device = torch.device("mps")
    cpu_board = _board()
    board = replace(
        cpu_board,
        values=cpu_board.values.to(device),
        base_priority=cpu_board.base_priority.to(device),
        priority_scale=cpu_board.priority_scale.to(device),
        axis=cpu_board.axis.to(device),
    )
    scores = torch.tensor(
        [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 10.0, 8.0, 9.0, 7.0, 6.0, 5.0],
        device=device,
        requires_grad=True,
    )
    loss, diagnostics = union_hard_edge_cutoff_exchange_loss(
        _output(scores),
        board,
        _labels().to(device),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert diagnostics["selected_edges"] == 4
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
