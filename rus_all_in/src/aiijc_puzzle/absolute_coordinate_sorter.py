"""Permutation-equivariant absolute coordinate head for unordered puzzle tiles.

The existing SocketMatcher learns useful local relations but its component
decoder has no reliable absolute coordinate gauge.  This module keeps the
dirty-visible SocketMatcher backbone and trains a separate board-conditioned
head directly against exact synthetic tile positions.

There is deliberately no embedding of the shuffled input index.  Learned
queries label *output* rows and columns; permuting the input tiles therefore
only permutes the tile dimension of every prediction.  A square Hungarian
projection converts the tile-to-slot logits to a strict tile-at-position
permutation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from aiijc_puzzle.socket_matcher import (
    SocketMatcher,
    SocketOutput,
    partial_log_optimal_transport,
    robust_tile_views,
    socket_score_statistics,
)


@dataclass(frozen=True)
class AbsoluteCoordinateOutput:
    """Coordinate predictions in shuffled-tile order."""

    row_logits: torch.Tensor
    column_logits: torch.Tensor
    slot_logits: torch.Tensor
    slot_log_assignment: torch.Tensor
    socket_output: SocketOutput


@dataclass(frozen=True)
class ComponentTranslationTarget:
    """One exact feasible translation target for a rigid predicted component."""

    tiles: tuple[int, ...]
    relative_rows: tuple[int, ...]
    relative_columns: tuple[int, ...]
    target_row_shift: int
    target_column_shift: int
    feasible_row_shifts: int
    feasible_column_shifts: int


class _ResidualSetBlock(nn.Module):
    """One permutation-equivariant self-attention/MLP update."""

    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=0.05,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Linear(dimension * 4, dimension),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalised = self.input_norm(tokens)
        update, _ = self.attention(
            normalised,
            normalised,
            normalised,
            need_weights=False,
        )
        value = tokens + update
        return value + self.feed_forward(value)


class _BoardConditionedAxisHead(nn.Module):
    """Compare every tile with ordered, board-conditioned output-axis queries."""

    def __init__(self, dimension: int, heads: int, grid: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.grid = grid
        # These queries label output coordinates, never shuffled input indices.
        self.coordinate_queries = nn.Parameter(torch.randn(1, grid, dimension) * 0.02)
        self.query_attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=0.05,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(dimension)
        self.tile_projection = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
        )
        self.query_projection = nn.Linear(dimension, dimension)
        self.coordinate_bias = nn.Parameter(torch.zeros(grid))
        self.log_scale = nn.Parameter(torch.tensor(math.log(5.0)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        queries = self.coordinate_queries.expand(batch, -1, -1)
        query_update, _ = self.query_attention(
            self.query_norm(queries),
            tokens,
            tokens,
            need_weights=False,
        )
        queries = queries + query_update
        tiles = F.normalize(self.tile_projection(tokens), dim=2)
        coordinates = F.normalize(self.query_projection(queries), dim=2)
        scale = self.log_scale.exp().clamp(1.0, 100.0)
        return scale * tiles @ coordinates.transpose(1, 2) + self.coordinate_bias


def square_log_sinkhorn(scores: torch.Tensor, *, iterations: int) -> torch.Tensor:
    """Scale a square batch of logits to doubly stochastic log assignments."""

    if scores.ndim != 3 or scores.shape[1] != scores.shape[2]:
        raise ValueError(f"scores must have shape B x N x N, got {tuple(scores.shape)}")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    value = scores
    for _ in range(iterations):
        value = value - torch.logsumexp(value, dim=2, keepdim=True)
        value = value - torch.logsumexp(value, dim=1, keepdim=True)
    return value


class AbsoluteCoordinateSorter(nn.Module):
    """Socket-backed set model trained to predict literal board coordinates."""

    def __init__(
        self,
        backbone: SocketMatcher,
        *,
        grid: int = 24,
        head_dimension: int = 64,
        heads: int = 4,
        set_layers: int = 2,
        sinkhorn_iterations: int = 8,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if grid < 2:
            raise ValueError("grid must be at least 2")
        if head_dimension % heads:
            raise ValueError("head_dimension must be divisible by heads")
        if set_layers <= 0:
            raise ValueError("set_layers must be positive")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        self.backbone = backbone
        self.grid = grid
        self.head_dimension = head_dimension
        self.heads = heads
        self.set_layers = set_layers
        self.sinkhorn_iterations = sinkhorn_iterations
        self.freeze_backbone = freeze_backbone

        # Context plus four directional socket embeddings and four sets of six
        # board-relative partner-score statistics.
        feature_dimension = 5 * backbone.dimension + 24
        self.input_projection = nn.Sequential(
            nn.LayerNorm(feature_dimension),
            nn.Linear(feature_dimension, head_dimension),
            nn.GELU(),
            nn.Linear(head_dimension, head_dimension),
        )
        self.set_blocks = nn.ModuleList(
            [_ResidualSetBlock(head_dimension, heads) for _ in range(set_layers)]
        )
        self.row_head = _BoardConditionedAxisHead(head_dimension, heads, grid)
        self.column_head = _BoardConditionedAxisHead(head_dimension, heads, grid)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> AbsoluteCoordinateSorter:
        super().train(mode)
        if self.freeze_backbone:
            # A frozen backbone must also have deterministic dropout behaviour.
            self.backbone.eval()
        return self

    def _socket_features(
        self, tiles: torch.Tensor
    ) -> tuple[torch.Tensor, SocketOutput]:
        views = robust_tile_views(tiles)
        context = self.backbone.tile_context(views)
        sides = self.backbone._side_embeddings(views, context)  # noqa: SLF001
        right_source, left_target = self.backbone.horizontal(sides["right"], sides["left"])
        down_source, top_target = self.backbone.vertical(sides["bottom"], sides["top"])
        right_raw = self.backbone._similarity(  # noqa: SLF001
            right_source,
            left_target,
            self.backbone.horizontal_scale,
        )
        down_raw = self.backbone._similarity(  # noqa: SLF001
            down_source,
            top_target,
            self.backbone.vertical_scale,
        )
        right_out_border = self.backbone._border_logits(  # noqa: SLF001
            side="right",
            embedding=right_source,
            raw_scores=right_raw,
            outgoing=True,
            shared_bin=self.backbone.horizontal_bin,
        )
        left_in_border = self.backbone._border_logits(  # noqa: SLF001
            side="left",
            embedding=left_target,
            raw_scores=right_raw,
            outgoing=False,
            shared_bin=self.backbone.horizontal_bin,
        )
        bottom_out_border = self.backbone._border_logits(  # noqa: SLF001
            side="bottom",
            embedding=down_source,
            raw_scores=down_raw,
            outgoing=True,
            shared_bin=self.backbone.vertical_bin,
        )
        top_in_border = self.backbone._border_logits(  # noqa: SLF001
            side="top",
            embedding=top_target,
            raw_scores=down_raw,
            outgoing=False,
            shared_bin=self.backbone.vertical_bin,
        )
        count = tiles.shape[1]
        unmatched = self.grid
        if count != self.grid * self.grid:
            raise ValueError(
                f"AbsoluteCoordinateSorter expects {self.grid**2} tiles, got {count}"
            )
        socket_output = SocketOutput(
            right_raw=right_raw,
            down_raw=down_raw,
            right_log_assignment=partial_log_optimal_transport(
                right_raw,
                right_out_border,
                unmatched=unmatched,
                iterations=self.backbone.sinkhorn_iterations,
                target_bin_score=left_in_border,
            ),
            down_log_assignment=partial_log_optimal_transport(
                down_raw,
                bottom_out_border,
                unmatched=unmatched,
                iterations=self.backbone.sinkhorn_iterations,
                target_bin_score=top_in_border,
            ),
            right_out_border_logits=right_out_border,
            left_in_border_logits=left_in_border,
            bottom_out_border_logits=bottom_out_border,
            top_in_border_logits=top_in_border,
        )
        statistics = torch.cat(
            (
                socket_score_statistics(right_raw, outgoing=True),
                socket_score_statistics(right_raw, outgoing=False),
                socket_score_statistics(down_raw, outgoing=True),
                socket_score_statistics(down_raw, outgoing=False),
            ),
            dim=2,
        )
        features = torch.cat(
            (context, right_source, left_target, down_source, top_target, statistics),
            dim=2,
        )
        return features, socket_output

    def encode_coordinate_tokens(
        self,
        tiles: torch.Tensor,
    ) -> tuple[torch.Tensor, SocketOutput]:
        """Return state-dict-neutral, permutation-equivariant tile tokens.

        This is the public attachment point for downstream component heads.  It
        deliberately exposes the same post-set-block tokens consumed by the
        coordinate heads and never introduces a shuffled-input index signal.
        """

        if tiles.ndim != 5 or tiles.shape[2:] != (3, 20, 20):
            raise ValueError(f"tiles must have shape B x N x 3 x 20 x 20, got {tiles.shape}")
        if self.freeze_backbone:
            with torch.no_grad():
                features, socket_output = self._socket_features(tiles)
        else:
            features, socket_output = self._socket_features(tiles)
        tokens = self.input_projection(features)
        for block in self.set_blocks:
            tokens = block(tokens)
        return tokens, socket_output

    def forward(self, tiles: torch.Tensor) -> AbsoluteCoordinateOutput:
        tokens, socket_output = self.encode_coordinate_tokens(tiles)
        row_logits = self.row_head(tokens)
        column_logits = self.column_head(tokens)
        cells = torch.arange(self.grid * self.grid, device=tiles.device)
        rows = cells // self.grid
        columns = cells % self.grid
        slot_logits = row_logits[:, :, rows] + column_logits[:, :, columns]
        return AbsoluteCoordinateOutput(
            row_logits=row_logits,
            column_logits=column_logits,
            slot_logits=slot_logits,
            slot_log_assignment=square_log_sinkhorn(
                slot_logits,
                iterations=self.sinkhorn_iterations,
            ),
            socket_output=socket_output,
        )


def coordinate_sorting_loss(
    output: AbsoluteCoordinateOutput,
    input_tile_to_position: torch.Tensor,
    *,
    grid: int,
    assignment_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise literal row, column and one-to-one slot on exact permutations."""

    target = input_tile_to_position.long()
    count = grid * grid
    if target.ndim != 2 or target.shape != output.row_logits.shape[:2]:
        raise ValueError(
            "input_tile_to_position must match the output tile dimensions, got "
            f"{tuple(target.shape)} and {tuple(output.row_logits.shape[:2])}"
        )
    expected = torch.arange(count, device=target.device).expand(target.shape[0], -1)
    if not torch.equal(target.sort(dim=1).values, expected):
        raise ValueError("each input_tile_to_position row must be a complete permutation")
    if not math.isfinite(assignment_weight) or assignment_weight < 0:
        raise ValueError("assignment_weight must be finite and non-negative")
    rows = target // grid
    columns = target % grid
    row_nll = F.cross_entropy(output.row_logits.reshape(-1, grid), rows.reshape(-1))
    column_nll = F.cross_entropy(
        output.column_logits.reshape(-1, grid),
        columns.reshape(-1),
    )
    batch_index = torch.arange(target.shape[0], device=target.device)[:, None]
    tile_index = torch.arange(count, device=target.device)[None, :]
    assignment_nll = -output.slot_log_assignment[batch_index, tile_index, target].mean()
    loss = row_nll + column_nll + assignment_weight * assignment_nll
    with torch.no_grad():
        row_accuracy = (output.row_logits.argmax(2) == rows).float().mean()
        column_accuracy = (output.column_logits.argmax(2) == columns).float().mean()
        slot_accuracy = (output.slot_logits.argmax(2) == target).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "row_nll": float(row_nll.detach()),
        "column_nll": float(column_nll.detach()),
        "assignment_nll": float(assignment_nll.detach()),
        "row_argmax_accuracy": float(row_accuracy),
        "column_argmax_accuracy": float(column_accuracy),
        "slot_argmax_accuracy": float(slot_accuracy),
    }


def truth_consistent_component_targets(
    components: Sequence[dict[int, tuple[int, int]]],
    input_tile_to_position: torch.Tensor | np.ndarray,
    *,
    grid: int,
    minimum_size: int = 2,
) -> tuple[ComponentTranslationTarget, ...]:
    """Keep predicted components that have one exact translation under truth.

    Components are produced solely from frozen dirty-visible socket evidence.
    Exact synthetic labels are used only to decide whether a component's
    relative geometry supports a unique supervised translation; inconsistent
    false-bridge components are skipped rather than assigned a noisy target.
    """

    if grid < 2:
        raise ValueError("grid must be at least 2")
    if not 2 <= minimum_size <= grid * grid:
        raise ValueError("minimum_size must be in [2, grid**2]")
    value: Any = input_tile_to_position
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    positions = np.asarray(value, dtype=np.int64)
    if positions.ndim == 2 and positions.shape[0] == 1:
        positions = positions[0]
    count = grid * grid
    if positions.shape != (count,) or not np.array_equal(np.sort(positions), np.arange(count)):
        raise ValueError("input_tile_to_position must be one exact permutation")

    targets: list[ComponentTranslationTarget] = []
    for component in components:
        if len(component) < minimum_size:
            continue
        tiles = tuple(sorted(component))
        relative = np.asarray([component[tile] for tile in tiles], dtype=np.int64)
        if relative.ndim != 2 or relative.shape[1] != 2 or np.any(relative < 0):
            raise ValueError("component coordinates must be non-negative row/column pairs")
        if len({tuple(coordinate) for coordinate in relative.tolist()}) != len(tiles):
            raise ValueError("component coordinates must be collision-free")
        true_rows = positions[np.asarray(tiles)] // grid
        true_columns = positions[np.asarray(tiles)] % grid
        row_shifts = true_rows - relative[:, 0]
        column_shifts = true_columns - relative[:, 1]
        if np.any(row_shifts != row_shifts[0]) or np.any(column_shifts != column_shifts[0]):
            continue
        height = int(relative[:, 0].max()) + 1
        width = int(relative[:, 1].max()) + 1
        feasible_rows = grid - height + 1
        feasible_columns = grid - width + 1
        target_row = int(row_shifts[0])
        target_column = int(column_shifts[0])
        if not 0 <= target_row < feasible_rows or not 0 <= target_column < feasible_columns:
            raise RuntimeError("truth-consistent component target is not a feasible grid shift")
        targets.append(
            ComponentTranslationTarget(
                tiles=tiles,
                relative_rows=tuple(int(value) for value in relative[:, 0]),
                relative_columns=tuple(int(value) for value in relative[:, 1]),
                target_row_shift=target_row,
                target_column_shift=target_column,
                feasible_row_shifts=feasible_rows,
                feasible_column_shifts=feasible_columns,
            )
        )
    return tuple(targets)


def component_translation_loss(
    slot_logits: torch.Tensor,
    targets: Sequence[ComponentTranslationTarget],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Classify the exact feasible 2-D shift of each retained component."""

    if slot_logits.ndim != 3 or slot_logits.shape[0] != 1:
        raise ValueError("slot_logits must have singleton-batch shape 1 x N x N")
    count = slot_logits.shape[1]
    grid = round(math.sqrt(count))
    if grid * grid != count or slot_logits.shape[2] != count:
        raise ValueError("slot_logits tile/slot dimensions must be one square grid")
    losses: list[torch.Tensor] = []
    uniform_nlls: list[float] = []
    top1_correct = 0
    chance_top1 = 0.0
    supervised_tiles = 0
    maximum_size = 0
    for target in targets:
        tiles = torch.tensor(target.tiles, device=slot_logits.device, dtype=torch.long)
        relative_rows = torch.tensor(
            target.relative_rows,
            device=slot_logits.device,
            dtype=torch.long,
        )
        relative_columns = torch.tensor(
            target.relative_columns,
            device=slot_logits.device,
            dtype=torch.long,
        )
        row_shifts = torch.arange(target.feasible_row_shifts, device=slot_logits.device)
        column_shifts = torch.arange(
            target.feasible_column_shifts,
            device=slot_logits.device,
        )
        shift_rows, shift_columns = torch.meshgrid(
            row_shifts,
            column_shifts,
            indexing="ij",
        )
        positions = (
            (relative_rows[:, None] + shift_rows.reshape(1, -1)) * grid
            + relative_columns[:, None]
            + shift_columns.reshape(1, -1)
        )
        scores = slot_logits[0, tiles[:, None], positions].sum(dim=0)
        target_index = (
            target.target_row_shift * target.feasible_column_shifts
            + target.target_column_shift
        )
        losses.append(
            F.cross_entropy(
                scores.unsqueeze(0),
                torch.tensor([target_index], device=slot_logits.device),
            )
        )
        class_count = int(scores.numel())
        uniform_nlls.append(math.log(class_count))
        top1_correct += int(int(scores.detach().argmax()) == target_index)
        chance_top1 += 1.0 / class_count
        supervised_tiles += len(target.tiles)
        maximum_size = max(maximum_size, len(target.tiles))
    # Preserve a valid differentiable zero for rare boards with no exact
    # non-singleton predicted component.
    loss = torch.stack(losses).mean() if losses else slot_logits.sum() * 0.0
    uniform_nll = float(np.mean(uniform_nlls)) if uniform_nlls else 0.0
    observed_nll = float(loss.detach())
    return loss, {
        "component_translation_nll": observed_nll,
        "component_translation_uniform_nll": uniform_nll,
        "component_translation_nll_minus_uniform": observed_nll - uniform_nll,
        "component_translation_nll_ratio_to_uniform": (
            observed_nll / uniform_nll if uniform_nll > 0 else 0.0
        ),
        "component_translation_shift_top1_accuracy": (
            float(top1_correct / len(targets)) if targets else 0.0
        ),
        "component_translation_shift_chance_accuracy": (
            float(chance_top1 / len(targets)) if targets else 0.0
        ),
        "supervised_component_count": float(len(targets)),
        "supervised_component_tiles": float(supervised_tiles),
        "maximum_supervised_component_size": float(maximum_size),
        "mean_supervised_component_size": (
            float(supervised_tiles / len(targets)) if targets else 0.0
        ),
    }


def train_consistent_component_unary(slot_logits: np.ndarray) -> np.ndarray:
    """Normalise slot logits without changing any component-shift argmax.

    Component translation CE is trained on sums of raw tile-to-slot logits.
    Subtracting one constant per tile adds the same constant to every feasible
    shift of a component, and dividing the whole board by one positive scale
    preserves their ordering.  In contrast, scaling every tile independently
    changes the trained component energy.
    """

    value = np.asarray(slot_logits, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"slot_logits must be one square matrix, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("slot_logits contains non-finite values")
    centred = value - value.mean(axis=1, keepdims=True)
    scale = max(float(centred.std()), 1e-6)
    return centred / scale


def decode_coordinate_logits(slot_logits: torch.Tensor | np.ndarray) -> np.ndarray:
    """Project one tile-to-slot score matrix to strict tile-at-position order."""

    value: Any = slot_logits
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    scores = np.asarray(value, dtype=np.float64)
    if scores.ndim == 3 and scores.shape[0] == 1:
        scores = scores[0]
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"slot_logits must be a square matrix, got {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("slot_logits contains non-finite values")
    tiles, slots = linear_sum_assignment(-scores)
    layout = np.empty(len(tiles), dtype=np.int32)
    layout[slots] = tiles
    if not np.array_equal(np.sort(layout), np.arange(len(layout))):
        raise RuntimeError("Hungarian coordinate decoder did not return a strict permutation")
    return layout
