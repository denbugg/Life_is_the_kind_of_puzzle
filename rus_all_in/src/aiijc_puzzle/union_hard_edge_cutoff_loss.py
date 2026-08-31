"""Cutoff-aware training loss for the frozen Union hard-edge priority head.

The production decoder consumes only the strongest fixed number of projected
edges on each axis (144 for the 24x24 board).  The original all-pairs loss also
spends gradient on correctly ordered edges far away from that decision.  This
module instead trains only the exchanges that can repair the *current* cutoff:
every false selected edge is ranked below every missed true edge on the same
axis.

This is intentionally not a hinge against a frozen numeric K-th threshold.
Both endpoints come from the current score vector, so a common score shift
cannot change the exchange term or drag the whole distribution toward an old
absolute cutoff.  The inherited normalised-residual L2 is retained as the
explicit anchor to the frozen Union priorities.

Feature construction and model inference remain target-free.  Exact labels
enter only this training-time loss, after a :class:`UnionHardEdgeBoard` has
already frozen the dirty-visible features and immutable edge identities.  The
model architecture and state dictionary are unchanged, so an existing frozen
``UnionHardEdgePriority`` checkpoint can be continued directly with this loss.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from aiijc_puzzle.union_hard_edge_priority import (
    UnionHardEdgeBoard,
    UnionHardEdgeOutput,
    validate_union_hard_edge_board,
)

# Fixed continuation contract.  This deliberately is not exposed as a loss
# argument: the experiment is one predeclared objective, not a weight sweep.
CUTOFF_EXCHANGE_RESIDUAL_WEIGHT = 1e-3
CUTOFF_EXCHANGE_LOSS_SCHEMA = "union-hard-edge-cutoff-exchange-v1"


def _validate_output(output: UnionHardEdgeOutput, board: UnionHardEdgeBoard) -> None:
    if not isinstance(output, UnionHardEdgeOutput):
        raise TypeError("output must be UnionHardEdgeOutput")
    validate_union_hard_edge_board(board)
    edge_count = len(board.values)
    vectors = {
        "scores": output.scores,
        "residual": output.residual,
        "normalised_residual": output.normalised_residual,
    }
    for name, value in vectors.items():
        if value.shape != (edge_count,):
            raise ValueError(f"output {name} must have shape {(edge_count,)}")
        if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"output {name} must contain finite floating-point values")
        if value.device != board.values.device:
            raise ValueError(f"output {name} must be on the board device")


def _decoder_axis_order(
    output: UnionHardEdgeOutput,
    board: UnionHardEdgeBoard,
    *,
    axis_index: int,
) -> np.ndarray:
    """Return the decoder-equivalent deterministic order for one axis.

    ``prioritise_component_edges`` breaks learned-priority ties with the
    original hard-edge confidence and then the immutable source/target
    identity.  Explicitly reproducing all of those keys makes the top-K set
    deterministic even at zero-init, where learned scores equal base scores.
    """

    axes = board.axis.detach().cpu().numpy()
    indices = np.flatnonzero(axes == axis_index)
    # MPS cannot materialise float64 tensors.  Move the frozen selection keys
    # to CPU first, then widen for deterministic NumPy lexicographic sorting.
    scores = output.scores.detach().cpu().double().numpy()
    base = board.base_priority.detach().cpu().double().numpy()
    order = np.lexsort(
        (
            board.target[indices],
            board.source[indices],
            -base[indices],
            -scores[indices],
        )
    )
    return np.ascontiguousarray(indices[order], dtype=np.int64)


def union_hard_edge_cutoff_membership(
    output: UnionHardEdgeOutput,
    board: UnionHardEdgeBoard,
) -> torch.Tensor:
    """Select the current decoder-budget membership on both board axes.

    The returned boolean vector is detached by construction: top-K membership
    is a discrete decision, while gradients flow through the selected score
    pairs in :func:`union_hard_edge_cutoff_exchange_loss`.
    """

    _validate_output(output, board)
    selected = np.zeros(len(board.values), dtype=bool)
    for axis_index in (0, 1):
        order = _decoder_axis_order(output, board, axis_index=axis_index)
        if len(order) < board.edge_budget_per_axis:
            raise ValueError("edge budget exceeds one axis hard-edge supply")
        selected[order[: board.edge_budget_per_axis]] = True
    return torch.from_numpy(selected).to(device=output.scores.device)


def union_hard_edge_cutoff_exchange_loss(
    output: UnionHardEdgeOutput,
    board: UnionHardEdgeBoard,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Rank false selected edges below missed true edges at the current cutoff.

    For each axis, the loss is the mean ``softplus(false - missed_true)`` over
    the complete cross product of current false selections and missed true
    positives.  This relative term is invariant to a common score shift; it is
    not an absolute K-th-threshold hinge.  Axes are weighted equally, including
    an already-correct axis whose exchange term is zero.  A fixed inherited L2
    penalty keeps bounded residuals close to the frozen Union priority scale.
    """

    _validate_output(output, board)
    if labels.shape != (len(board.values),) or labels.dtype != torch.bool:
        raise ValueError("labels must be a boolean vector aligned with board values")
    if labels.device != output.scores.device:
        raise ValueError("labels must be on the output device")

    selected = union_hard_edge_cutoff_membership(output, board)
    axis_losses: list[torch.Tensor] = []
    false_selected_total = 0
    missed_true_total = 0
    correct_selected_total = 0
    exchange_pair_total = 0
    active_axes = 0
    for axis_index in (0, 1):
        on_axis = board.axis == axis_index
        positive_count = int(torch.count_nonzero(labels & on_axis).item())
        negative_count = int(torch.count_nonzero(~labels & on_axis).item())
        if not positive_count or not negative_count:
            raise ValueError("each axis requires both true and false hard-edge labels")

        false_selected = selected & on_axis & ~labels
        missed_true = ~selected & on_axis & labels
        false_scores = output.scores[false_selected]
        missed_scores = output.scores[missed_true]
        false_count = len(false_scores)
        missed_count = len(missed_scores)
        false_selected_total += false_count
        missed_true_total += missed_count
        correct_selected_total += int(torch.count_nonzero(selected & on_axis & labels).item())
        exchange_pair_total += false_count * missed_count
        if false_count and missed_count:
            axis_losses.append(
                F.softplus(false_scores.unsqueeze(1) - missed_scores.unsqueeze(0)).mean()
            )
            active_axes += 1
        else:
            # Retain a differentiable scalar even when an axis has no bad
            # cutoff exchange left to repair.
            axis_losses.append(output.scores.sum() * 0.0)

    exchange = torch.stack(axis_losses).mean()
    residual_l2 = output.normalised_residual.square().mean()
    loss = exchange + CUTOFF_EXCHANGE_RESIDUAL_WEIGHT * residual_l2
    return loss, {
        "schema": CUTOFF_EXCHANGE_LOSS_SCHEMA,
        "loss": float(loss.detach()),
        "cutoff_exchange_loss": float(exchange.detach()),
        "normalised_residual_l2": float(residual_l2.detach()),
        "residual_weight": CUTOFF_EXCHANGE_RESIDUAL_WEIGHT,
        "edge_budget_per_axis": board.edge_budget_per_axis,
        "selected_edges": int(selected.sum().item()),
        "correct_selected_edges": correct_selected_total,
        "false_selected_edges": false_selected_total,
        "missed_true_edges": missed_true_total,
        "exchange_pairs": exchange_pair_total,
        "active_axes": active_axes,
    }


__all__ = [
    "CUTOFF_EXCHANGE_LOSS_SCHEMA",
    "CUTOFF_EXCHANGE_RESIDUAL_WEIGHT",
    "union_hard_edge_cutoff_exchange_loss",
    "union_hard_edge_cutoff_membership",
]
