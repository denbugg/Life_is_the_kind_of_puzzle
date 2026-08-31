"""Dedicated full-resolution frame-side classifier and strict cyclic placer.

The model classifies whether each original upright tile belongs to the
top/bottom/left/right canvas frame.  It is permutation equivariant over tiles,
contains no absolute slot or tile-identity embedding, and keeps every learned
spatial feature at 20x20 before extracting oriented boundary sequences.

The placer consumes exactly 24 predicted tiles per side.  It only rolls an
already strict layout and chooses lexicographically by integer frame-set hits,
then by the frozen Socket cut objective.  This makes inference scale-free and
prevents a post-hoc blend-weight sweep.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aiijc_puzzle.socket_decoder import socket_layout_objective

SIDES = ("top", "bottom", "left", "right")


@dataclass(frozen=True)
class FrameSideConfig:
    """No-downsample frame classifier architecture."""

    context_dimension: int = 64
    width: int = 32
    blocks: int = 5
    boundary_width: int = 5
    sequence_dilations: tuple[int, ...] = (1, 2, 4)
    use_restored_view: bool = True

    def validate(self) -> None:
        if self.context_dimension < 1 or self.width < 8 or self.blocks < 1:
            raise ValueError("context_dimension/width/blocks must be positive")
        if not 1 <= self.boundary_width <= 10:
            raise ValueError("boundary_width must lie in [1,10]")
        if not self.sequence_dilations or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.sequence_dilations
        ):
            raise ValueError("sequence dilations must be positive integers")


@dataclass(frozen=True)
class FrameCyclicDiagnostics:
    grid_size: int
    candidates_evaluated: int
    selected_row_roll: int
    selected_column_roll: int
    selected_frame_hits: int
    maximum_frame_hits: int
    frame_tie_count: int
    cut_objective: float
    changed: bool
    strict_permutation: bool
    runtime_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameCyclicResult:
    layout: np.ndarray
    diagnostics: FrameCyclicDiagnostics

    def report(self, *, include_layout: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "placer": "frame-top24-lexicographic-cut-cyclic-v1",
            "layout_sha256": hashlib.sha256(
                np.asarray(self.layout, dtype="<i4").tobytes()
            ).hexdigest(),
            "diagnostics": self.diagnostics.as_dict(),
        }
        if include_layout:
            payload["tile_at_position"] = self.layout.tolist()
        return payload


class _FullResolutionBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        groups = math.gcd(width, max(1, width // 8))
        self.normalise = nn.GroupNorm(groups, width)
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.expand = nn.Conv2d(width, width * 2, 1)
        self.project = nn.Conv2d(width, width, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(F.gelu(self.normalise(value)))
        first, second = self.expand(residual).chunk(2, dim=1)
        return value + self.project(first * torch.sigmoid(second))


class _SequenceBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        groups = math.gcd(width, max(1, width // 8))
        self.normalise = nn.GroupNorm(groups, width)
        self.convolution = nn.Conv1d(
            width,
            width,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="reflect",
        )
        self.project = nn.Conv1d(width, width, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.project(F.gelu(self.convolution(self.normalise(value))))


def _normalised_pixels(value: torch.Tensor) -> torch.Tensor:
    mean = value.mean(dim=(-2, -1), keepdim=True)
    scale = value.std(dim=(-2, -1), keepdim=True).clamp_min(1.0 / 255.0)
    return ((value - mean) / scale).clamp(-5.0, 5.0)


def _standardise_board(value: torch.Tensor) -> torch.Tensor:
    mean = value.mean(dim=1, keepdim=True)
    scale = value.std(dim=1, keepdim=True).clamp_min(1e-4)
    return ((value - mean) / scale).clamp(-6.0, 6.0)


class FrameSideClassifier(nn.Module):
    """Permutation-equivariant four-side classifier over full-resolution tiles."""

    def __init__(self, config: FrameSideConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        input_channels = 12 if config.use_restored_view else 6
        self.stem = nn.Conv2d(input_channels, config.width, 3, padding=1)
        self.blocks = nn.Sequential(
            *[_FullResolutionBlock(config.width) for _ in range(config.blocks)]
        )
        self.sequence = nn.Sequential(
            *[
                _SequenceBlock(config.width, dilation)
                for dilation in config.sequence_dilations
            ]
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(config.context_dimension),
            nn.Linear(config.context_dimension, config.width),
            nn.GELU(),
        )
        self.side_embedding = nn.Embedding(len(SIDES), 8)
        # sequence mean/max, tile mean/max, local context, board mean/max,
        # all four Socket border logits, and a side embedding.
        head_dimension = config.width * 7 + 4 + 8
        self.head = nn.Sequential(
            nn.LayerNorm(head_dimension),
            nn.Linear(head_dimension, config.width * 2),
            nn.GELU(),
            nn.Linear(config.width * 2, 1),
        )

    def full_resolution_field(
        self,
        raw_tiles: torch.Tensor,
        restored_tiles: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return B*N x C x 20 x 20 without any spatial resampling."""

        if raw_tiles.ndim != 5 or raw_tiles.shape[2:] != (3, 20, 20):
            raise ValueError("raw_tiles must have shape B x N x 3 x 20 x 20")
        raw = raw_tiles.float()
        if raw.min() < 0 or raw.max() > 1:
            raise ValueError("raw_tiles must lie in [0,1]")
        inputs = [raw, _normalised_pixels(raw)]
        if self.config.use_restored_view:
            if restored_tiles is None or restored_tiles.shape != raw_tiles.shape:
                raise ValueError("restored view is required with the same tile shape")
            restored = restored_tiles.float()
            if restored.min() < 0 or restored.max() > 1:
                raise ValueError("restored_tiles must lie in [0,1]")
            inputs.extend((restored, _normalised_pixels(restored)))
        elif restored_tiles is not None:
            raise ValueError("restored view was supplied to a raw-only model")
        batch, count = raw.shape[:2]
        value = torch.cat(inputs, dim=2).reshape(batch * count, -1, 20, 20)
        return self.blocks(self.stem(value))

    def _oriented_sequences(self, field: torch.Tensor) -> torch.Tensor:
        band = self.config.boundary_width
        top = field[..., :band, :]
        bottom = field[..., -band:, :].flip(-2)
        left = field[..., :, :band].transpose(-2, -1)
        right = field[..., :, -band:].flip(-1).transpose(-2, -1)
        return torch.stack((top, bottom, left, right), dim=1).mean(dim=-2)

    def forward(
        self,
        raw_tiles: torch.Tensor,
        restored_tiles: torch.Tensor | None,
        socket_context: torch.Tensor,
        socket_border_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return B x N x 4 logits in ``SIDES`` order."""

        batch, count = raw_tiles.shape[:2]
        if socket_context.shape != (batch, count, self.config.context_dimension):
            raise ValueError("socket_context shape does not match the model contract")
        if socket_border_logits.shape != (batch, count, len(SIDES)):
            raise ValueError("socket_border_logits must have shape B x N x 4")
        field = self.full_resolution_field(raw_tiles, restored_tiles)
        width = self.config.width
        sequences = self._oriented_sequences(field).reshape(
            batch * count * len(SIDES), width, 20
        )
        sequences = self.sequence(sequences)
        sequence_summary = torch.cat(
            (sequences.mean(dim=-1), sequences.amax(dim=-1)), dim=-1
        ).reshape(batch, count, len(SIDES), width * 2)
        tile_summary = torch.cat(
            (field.mean(dim=(-2, -1)), field.amax(dim=(-2, -1))), dim=-1
        ).reshape(batch, count, width * 2)
        local_context = self.context_projection(socket_context)
        board_source = torch.cat((tile_summary[..., :width], local_context), dim=-1)
        board_summary = torch.cat(
            (board_source.mean(dim=1), board_source.amax(dim=1)), dim=-1
        )
        # Reduce the 4*width board summary to the declared 2*width by pairing
        # visual/context statistics, preserving a small fixed architecture.
        board_summary = board_summary.reshape(batch, 2, 2, width).mean(dim=2).reshape(
            batch, width * 2
        )
        border = _standardise_board(socket_border_logits)
        side_ids = torch.arange(len(SIDES), device=raw_tiles.device)
        side_embedding = self.side_embedding(side_ids)
        pieces = (
            sequence_summary,
            tile_summary[:, :, None, :].expand(-1, -1, len(SIDES), -1),
            local_context[:, :, None, :].expand(-1, -1, len(SIDES), -1),
            board_summary[:, None, None, :].expand(-1, count, len(SIDES), -1),
            border[:, :, None, :].expand(-1, -1, len(SIDES), -1),
            side_embedding[None, None].expand(batch, count, -1, -1),
        )
        return self.head(torch.cat(pieces, dim=-1))[..., 0]


def frame_side_targets(tile_to_position: Any, *, grid: int) -> torch.Tensor:
    """Return exact B x N x 4 border membership with cardinality ``grid``."""

    value = torch.as_tensor(tile_to_position, dtype=torch.long)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    count = grid * grid
    if value.ndim != 2 or value.shape[1] != count:
        raise ValueError("tile_to_position must have shape B x grid**2")
    expected = torch.arange(count, device=value.device).expand_as(value)
    if not torch.equal(torch.sort(value, dim=1).values, expected):
        raise ValueError("tile_to_position rows must be strict permutations")
    row = torch.div(value, grid, rounding_mode="floor")
    column = value.remainder(grid)
    targets = torch.stack(
        (row == 0, row == grid - 1, column == 0, column == grid - 1), dim=-1
    )
    if not torch.all(targets.sum(dim=1) == grid):
        raise RuntimeError("frame targets violate exact cardinality")
    return targets


def frame_side_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    consistency_logits: tuple[torch.Tensor, torch.Tensor] | None = None,
    consistency_weight: float = 0.10,
    bce_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """All-positive listwise loss, balanced BCE, and paired-view consistency."""

    if logits.ndim != 3 or logits.shape[-1] != len(SIDES):
        raise ValueError("logits must have shape B x N x 4")
    labels = targets.to(device=logits.device, dtype=torch.bool)
    if labels.shape != logits.shape:
        raise ValueError("targets must align with logits")
    positive_count = labels.sum(dim=1)
    if not torch.all(positive_count == positive_count[:, :1]):
        raise ValueError("every side must have equal exact cardinality")
    listwise = (
        torch.logsumexp(logits, dim=1)
        - (logits * labels).sum(dim=1) / positive_count
    ).mean()
    positive_weight = (logits.shape[1] - positive_count.float()) / positive_count.float()
    weights = torch.where(labels, positive_weight[:, None, :], 1.0)
    bce = (
        F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
        * weights
    ).mean()
    consistency = logits.new_zeros(())
    if consistency_logits is not None:
        first, second = consistency_logits
        if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != len(SIDES):
            raise ValueError("aligned consistency logits must have equal B x N x 4 shape")

        def standardise(value: torch.Tensor) -> torch.Tensor:
            return (value - value.mean(dim=1, keepdim=True)) / value.std(
                dim=1, keepdim=True
            ).clamp_min(1e-4)

        consistency = F.smooth_l1_loss(standardise(first), standardise(second))
    loss = listwise + bce_weight * bce + consistency_weight * consistency
    return loss, {
        "listwise": float(listwise.detach()),
        "balanced_bce": float(bce.detach()),
        "paired_consistency": float(consistency.detach()),
        "loss": float(loss.detach()),
    }


def top_frame_sets(logits: Any, *, grid: int) -> np.ndarray:
    """Return 4 x grid stable top-tile indices."""

    value = logits.detach().cpu().numpy() if hasattr(logits, "detach") else logits
    array = np.asarray(value, dtype=np.float64)
    count = grid * grid
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (count, len(SIDES)) or not np.isfinite(array).all():
        raise ValueError("frame logits must have finite shape grid**2 x 4")
    return np.ascontiguousarray(
        np.argsort(-array, axis=0, kind="stable")[:grid].T,
        dtype=np.int32,
    )


def frame_topk_metrics(
    predicted_sets: Any,
    tile_to_position: Any,
    *,
    grid: int,
) -> dict[str, Any]:
    """Per-side and macro precision/recall/F1 for exact-cardinality predictions."""

    predicted = np.asarray(predicted_sets, dtype=np.int64)
    if predicted.shape != (len(SIDES), grid):
        raise ValueError("predicted_sets must have shape 4 x grid")
    if any(len(set(row.tolist())) != grid for row in predicted):
        raise ValueError("each predicted side must contain grid unique tiles")
    positions = np.asarray(tile_to_position, dtype=np.int64)
    count = grid * grid
    if positions.shape != (count,) or not np.array_equal(np.sort(positions), np.arange(count)):
        raise ValueError("tile_to_position must be one strict permutation")
    row, column = divmod(positions, grid)
    truth = (row == 0, row == grid - 1, column == 0, column == grid - 1)
    sides: dict[str, Any] = {}
    f1: list[float] = []
    for index, name in enumerate(SIDES):
        correct = int(np.count_nonzero(truth[index][predicted[index]]))
        score = correct / grid
        sides[name] = {
            "correct": correct,
            "selected": grid,
            "truth": grid,
            "precision": score,
            "recall": score,
            "f1": score,
        }
        f1.append(score)
    return {"sides": sides, "macro_f1": float(np.mean(f1))}


def _assignment(value: Any, *, count: int, name: str) -> np.ndarray:
    item = value.detach().cpu().numpy() if hasattr(value, "detach") else value
    result = np.asarray(item, dtype=np.float64)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.shape != (count + 1, count + 1):
        raise ValueError(f"{name} assignment has invalid shape")
    usable = result.copy()
    usable[count, count] = 0.0
    if not np.isfinite(usable).all():
        raise ValueError(f"{name} assignment contains invalid usable values")
    return np.ascontiguousarray(result)


def select_frame_cyclic_translation(
    layout: Any,
    predicted_sets: Any,
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    grid: int = 24,
) -> FrameCyclicResult:
    """Roll a strict layout by frame-hit count, breaking ties with cut evidence."""

    started = perf_counter()
    count = grid * grid
    initial = np.asarray(layout, dtype=np.int32)
    if initial.shape != (count,) or not np.array_equal(np.sort(initial), np.arange(count)):
        raise ValueError("layout must be a strict tile permutation")
    sets = np.asarray(predicted_sets, dtype=np.int64)
    if sets.shape != (len(SIDES), grid) or any(
        len(set(row.tolist())) != grid for row in sets
    ):
        raise ValueError("predicted frame sets must have shape 4 x grid and unique rows")
    selected = tuple(frozenset(row.tolist()) for row in sets)
    right = _assignment(right_log_assignment, count=count, name="right")
    down = _assignment(down_log_assignment, count=count, name="down")
    zero_unary = np.zeros((count, count), dtype=np.float64)
    board = initial.reshape(grid, grid)
    candidates: list[tuple[int, float, int, int, np.ndarray]] = []
    for row_roll in range(grid):
        for column_roll in range(grid):
            candidate = np.roll(board, shift=(row_roll, column_roll), axis=(0, 1))
            hits = (
                sum(int(tile) in selected[0] for tile in candidate[0])
                + sum(int(tile) in selected[1] for tile in candidate[-1])
                + sum(int(tile) in selected[2] for tile in candidate[:, 0])
                + sum(int(tile) in selected[3] for tile in candidate[:, -1])
            )
            flat = np.ascontiguousarray(candidate.reshape(-1), dtype=np.int32)
            cut = socket_layout_objective(
                flat,
                right[:count, :count],
                down[:count, :count],
                zero_unary,
                grid=grid,
                border_weight=0.0,
            )
            candidates.append((hits, float(cut), row_roll, column_roll, flat))
    maximum_hits = max(item[0] for item in candidates)
    ties = [item for item in candidates if item[0] == maximum_hits]
    best = max(ties, key=lambda item: (item[1], -item[2], -item[3]))
    result = np.ascontiguousarray(best[4], dtype=np.int32)
    if not np.array_equal(np.sort(result), np.arange(count)):
        raise RuntimeError("frame cyclic selection broke the tile permutation")
    return FrameCyclicResult(
        result,
        FrameCyclicDiagnostics(
            grid_size=grid,
            candidates_evaluated=count,
            selected_row_roll=best[2],
            selected_column_roll=best[3],
            selected_frame_hits=best[0],
            maximum_frame_hits=maximum_hits,
            frame_tie_count=len(ties),
            cut_objective=best[1],
            changed=(best[2], best[3]) != (0, 0),
            strict_permutation=True,
            runtime_seconds=perf_counter() - started,
        ),
    )


__all__ = [
    "FrameCyclicDiagnostics",
    "FrameCyclicResult",
    "FrameSideClassifier",
    "FrameSideConfig",
    "SIDES",
    "frame_side_loss",
    "frame_side_targets",
    "frame_topk_metrics",
    "select_frame_cyclic_translation",
    "top_frame_sets",
]
