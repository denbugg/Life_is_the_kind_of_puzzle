"""State-dict-neutral transpose views for absolute coordinate prediction.

The competition output must contain the original upright tiles.  The helpers in
this module transpose pixels only while asking the coordinate model for a
second, axis-swapped view.  Predicted coordinates are mapped back to the
original frame before decoding, so no transformed tile can reach the final
layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch.nn import functional as F

from aiijc_puzzle.absolute_coordinate_sorter import AbsoluteCoordinateOutput


class _CoordinateModel(Protocol):
    """Structural type used by the parameter-free TTA helper."""

    grid: int

    def __call__(self, tiles: torch.Tensor) -> AbsoluteCoordinateOutput: ...


@dataclass(frozen=True)
class TransposeCoordinateViews:
    """Original-frame coordinate logits from one upright and one transpose view."""

    original: AbsoluteCoordinateOutput
    transposed: AbsoluteCoordinateOutput
    mapped_transposed_row_logits: torch.Tensor
    mapped_transposed_column_logits: torch.Tensor


@dataclass(frozen=True)
class FusedCoordinateLogits:
    """Parameter-free original-frame row, column, and additive slot logits."""

    row_logits: torch.Tensor
    column_logits: torch.Tensor
    slot_logits: torch.Tensor


def transpose_tile_view(tiles: torch.Tensor) -> torch.Tensor:
    """Transpose every square tile without changing its shuffled input index."""

    if tiles.ndim != 5 or tiles.shape[2] != 3 or tiles.shape[3] != tiles.shape[4]:
        raise ValueError(
            "tiles must have shape B x N x 3 x H x H, got "
            f"{tuple(tiles.shape)}"
        )
    return tiles.transpose(-2, -1).contiguous()


def transpose_positions(positions: torch.Tensor, *, grid: int) -> torch.Tensor:
    """Map row-major tile positions into the transposed board frame."""

    if grid < 2:
        raise ValueError("grid must be at least 2")
    count = grid * grid
    if positions.ndim != 2 or positions.shape[1] != count:
        raise ValueError(
            f"positions must have shape B x {count}, got {tuple(positions.shape)}"
        )
    if positions.dtype == torch.bool or positions.is_floating_point():
        raise ValueError("positions must have an integer dtype")
    expected = torch.arange(count, device=positions.device).expand(
        positions.shape[0], -1
    )
    if not torch.equal(positions.sort(dim=1).values, expected):
        raise ValueError("every positions row must be a complete permutation")
    return (positions % grid) * grid + positions // grid


def map_transposed_axis_logits(
    row_logits: torch.Tensor,
    column_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map coordinate logits from a transposed board to the original frame.

    A transposed-view column is an original-view row, while a transposed-view
    row is an original-view column.  Pure transposition introduces no index
    reversal.
    """

    if row_logits.ndim != 3 or column_logits.shape != row_logits.shape:
        raise ValueError("row_logits and column_logits must share shape B x N x grid")
    return column_logits, row_logits


def collect_transpose_coordinate_views(
    model: _CoordinateModel,
    tiles: torch.Tensor,
) -> TransposeCoordinateViews:
    """Run the unchanged coordinate model on upright and transposed pixel views."""

    original = model(tiles)
    transposed = model(transpose_tile_view(tiles))
    mapped_row, mapped_column = map_transposed_axis_logits(
        transposed.row_logits,
        transposed.column_logits,
    )
    return TransposeCoordinateViews(
        original=original,
        transposed=transposed,
        mapped_transposed_row_logits=mapped_row,
        mapped_transposed_column_logits=mapped_column,
    )


def _axis_log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1)


def fuse_transpose_coordinate_views(
    views: TransposeCoordinateViews,
    *,
    grid: int,
    mode: Literal["symmetric", "row-teacher"],
) -> FusedCoordinateLogits:
    """Fuse mapped views after removing arbitrary per-head logit scales.

    ``symmetric`` averages both corresponding axis distributions.
    ``row-teacher`` preserves the established upright row head and averages the
    weak upright column with the strong row head evaluated in the transposed
    frame.  Both modes are parameter-free and keep the original tile order.
    """

    original = views.original
    original_row = _axis_log_probabilities(original.row_logits)
    original_column = _axis_log_probabilities(original.column_logits)
    mapped_row = _axis_log_probabilities(views.mapped_transposed_row_logits)
    mapped_column = _axis_log_probabilities(views.mapped_transposed_column_logits)
    if original_row.shape[-1] != grid:
        raise ValueError(
            f"coordinate logits have grid {original_row.shape[-1]}, expected {grid}"
        )
    if mode == "symmetric":
        row_logits = 0.5 * (original_row + mapped_row)
        column_logits = 0.5 * (original_column + mapped_column)
    elif mode == "row-teacher":
        row_logits = original_row
        column_logits = 0.5 * (original_column + mapped_column)
    else:
        raise ValueError(f"unsupported transpose fusion mode: {mode!r}")
    cells = torch.arange(grid * grid, device=row_logits.device)
    slot_logits = row_logits[:, :, cells // grid] + column_logits[:, :, cells % grid]
    return FusedCoordinateLogits(
        row_logits=row_logits,
        column_logits=column_logits,
        slot_logits=slot_logits,
    )


def symmetric_axis_consistency_loss(
    views: TransposeCoordinateViews,
) -> torch.Tensor:
    """Mean symmetric KL after mapping transpose predictions to original axes."""

    pairs = (
        (views.original.row_logits, views.mapped_transposed_row_logits),
        (views.original.column_logits, views.mapped_transposed_column_logits),
    )
    losses: list[torch.Tensor] = []
    for first, second in pairs:
        first_log = F.log_softmax(first, dim=-1)
        second_log = F.log_softmax(second, dim=-1)
        first_probability = first_log.exp()
        second_probability = second_log.exp()
        forward = (first_probability * (first_log - second_log)).sum(dim=-1)
        backward = (second_probability * (second_log - first_log)).sum(dim=-1)
        losses.append(0.5 * (forward + backward).mean())
    return torch.stack(losses).mean()
