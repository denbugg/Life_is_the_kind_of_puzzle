"""Explicit component-to-shift head for frozen Socket decoder components.

Unlike the earlier component loss, this module exposes component membership and
relative coordinates to the model itself.  It consumes one board of
permutation-equivariant tile tokens plus the exact components constructed from
dirty-visible Socket assignments, predicts a feasible row/column translation
for every component, and converts those predictions to the existing
``tile x slot`` decoder-unary contract.

Exact shuffle labels are accepted only by the target/loss helpers.  Inference
and unary conversion are target blind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ComponentDescriptor:
    """One frozen component in shuffled-tile identity space."""

    tiles: tuple[int, ...]
    relative_rows: tuple[int, ...]
    relative_columns: tuple[int, ...]
    confidence: float

    @property
    def size(self) -> int:
        return len(self.tiles)

    @property
    def height(self) -> int:
        return max(self.relative_rows) + 1

    @property
    def width(self) -> int:
        return max(self.relative_columns) + 1


@dataclass(frozen=True)
class ComponentShiftTarget:
    """Dominant feasible exact-translation mode of a predicted component."""

    component_index: int
    target_row_shift: int
    target_column_shift: int
    support: int
    purity: float
    size: int


@dataclass(frozen=True)
class ComponentShiftOutput:
    """Padded feasible-axis logits in component order."""

    row_logits: torch.Tensor
    column_logits: torch.Tensor
    feasible_row_shifts: tuple[int, ...]
    feasible_column_shifts: tuple[int, ...]


def _normalise_component(
    component: dict[int, tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    if not component:
        raise ValueError("components must not be empty")
    minimum_row = min(row for row, _ in component.values())
    minimum_column = min(column for _, column in component.values())
    return {
        tile: (row - minimum_row, column - minimum_column)
        for tile, (row, column) in component.items()
    }


def _validate_descriptors(
    components: tuple[ComponentDescriptor, ...],
    *,
    grid: int,
    require_partition: bool,
) -> None:
    if grid < 2:
        raise ValueError("grid must be at least two")
    count = grid * grid
    if not components:
        raise ValueError("components must be non-empty")
    observed: list[int] = []
    for component in components:
        size = component.size
        if size == 0 or not (
            len(component.relative_rows) == len(component.relative_columns) == size
        ):
            raise ValueError("component tiles and relative coordinates must align")
        if len(set(component.tiles)) != size:
            raise ValueError("component tile identities must be unique")
        if any(not 0 <= tile < count for tile in component.tiles):
            raise ValueError("component contains an out-of-range tile identity")
        coordinates = tuple(zip(component.relative_rows, component.relative_columns, strict=True))
        if any(row < 0 or column < 0 for row, column in coordinates):
            raise ValueError("component coordinates must be non-negative")
        if len(set(coordinates)) != size:
            raise ValueError("component coordinates must be collision-free")
        if component.height > grid or component.width > grid:
            raise ValueError("component shape exceeds the board")
        if not math.isfinite(component.confidence):
            raise ValueError("component confidence must be finite")
        observed.extend(component.tiles)
    if len(set(observed)) != len(observed):
        raise ValueError("components overlap in tile identity")
    if require_partition and sorted(observed) != list(range(count)):
        raise ValueError("components must partition every board tile exactly once")


def component_descriptors_from_decoder(
    component_build: Any,
    *,
    grid: int,
) -> tuple[ComponentDescriptor, ...]:
    """Convert ``rebuild_decoder_components`` output to model descriptors.

    Confidence is the mean dirty-visible confidence of accepted constraints
    whose endpoints lie in that final component.  Singletons have confidence
    zero.  No layout label is accepted or inspected.
    """

    raw_components = getattr(component_build, "components", None)
    constraints = getattr(component_build, "constraints", None)
    if not isinstance(raw_components, tuple) or not isinstance(constraints, tuple):
        raise ValueError("component_build does not expose decoder components/constraints")
    descriptors: list[ComponentDescriptor] = []
    for raw_component in raw_components:
        if not isinstance(raw_component, dict):
            raise ValueError("decoder component must be a tile-to-coordinate mapping")
        component = _normalise_component(raw_component)
        tiles = tuple(sorted(component))
        tile_set = set(tiles)
        accepted_confidence = [
            float(constraint.edge.confidence)
            for constraint in constraints
            if constraint.status in {"added", "consistent"}
            and constraint.edge.source in tile_set
            and constraint.edge.target in tile_set
        ]
        descriptors.append(
            ComponentDescriptor(
                tiles=tiles,
                relative_rows=tuple(component[tile][0] for tile in tiles),
                relative_columns=tuple(component[tile][1] for tile in tiles),
                confidence=(
                    float(np.mean(accepted_confidence)) if accepted_confidence else 0.0
                ),
            )
        )
    result = tuple(descriptors)
    _validate_descriptors(result, grid=grid, require_partition=True)
    return result


class ComponentShiftHead(nn.Module):
    """Permutation-invariant member aggregation and feasible-shift classifier."""

    def __init__(
        self,
        tile_dimension: int,
        *,
        grid: int = 24,
        hidden_dimension: int = 64,
    ) -> None:
        super().__init__()
        if tile_dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("tile_dimension and hidden_dimension must be positive")
        if grid < 2:
            raise ValueError("grid must be at least two")
        self.tile_dimension = tile_dimension
        self.grid = grid
        self.hidden_dimension = hidden_dimension
        self.tile_projection = nn.Sequential(
            nn.LayerNorm(tile_dimension),
            nn.Linear(tile_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.relative_projection = nn.Sequential(
            nn.Linear(4, hidden_dimension),
            nn.GELU(),
        )
        self.member_update = nn.Sequential(
            nn.Linear(2 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        self.structure_projection = nn.Sequential(
            nn.Linear(7, hidden_dimension),
            nn.GELU(),
        )
        self.component_fusion = nn.Sequential(
            nn.LayerNorm(4 * hidden_dimension),
            nn.Linear(4 * hidden_dimension, 2 * hidden_dimension),
            nn.GELU(),
            nn.Linear(2 * hidden_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.row_head = nn.Linear(hidden_dimension, grid)
        self.column_head = nn.Linear(hidden_dimension, grid)

    def forward(
        self,
        tile_tokens: torch.Tensor,
        components: tuple[ComponentDescriptor, ...],
    ) -> ComponentShiftOutput:
        if tile_tokens.ndim == 3 and tile_tokens.shape[0] == 1:
            tile_tokens = tile_tokens[0]
        expected = (self.grid * self.grid, self.tile_dimension)
        if tile_tokens.ndim != 2 or tuple(tile_tokens.shape) != expected:
            raise ValueError(f"tile_tokens must have shape {expected}, got {tile_tokens.shape}")
        if not torch.isfinite(tile_tokens).all():
            raise ValueError("tile_tokens must be finite")
        _validate_descriptors(components, grid=self.grid, require_partition=True)

        projected_tiles = self.tile_projection(tile_tokens)
        board_token = projected_tiles.mean(dim=0)
        component_tokens: list[torch.Tensor] = []
        feasible_rows: list[int] = []
        feasible_columns: list[int] = []
        for component in components:
            tiles = torch.tensor(component.tiles, device=tile_tokens.device, dtype=torch.long)
            rows = torch.tensor(
                component.relative_rows,
                device=tile_tokens.device,
                dtype=tile_tokens.dtype,
            )
            columns = torch.tensor(
                component.relative_columns,
                device=tile_tokens.device,
                dtype=tile_tokens.dtype,
            )
            normaliser = float(max(self.grid - 1, 1))
            height_normaliser = float(max(component.height - 1, 1))
            width_normaliser = float(max(component.width - 1, 1))
            relative = torch.stack(
                (
                    rows / normaliser,
                    columns / normaliser,
                    rows / height_normaliser - 0.5,
                    columns / width_normaliser - 0.5,
                ),
                dim=1,
            )
            member = self.member_update(
                torch.cat(
                    (projected_tiles[tiles], self.relative_projection(relative)),
                    dim=1,
                )
            )
            mean_member = member.mean(dim=0)
            max_member = member.max(dim=0).values
            size = float(component.size)
            area = float(component.height * component.width)
            structure = tile_tokens.new_tensor(
                (
                    size / (self.grid * self.grid),
                    math.log1p(size) / math.log1p(self.grid * self.grid),
                    component.height / self.grid,
                    component.width / self.grid,
                    size / area,
                    math.tanh(component.confidence / 5.0),
                    float(component.size == 1),
                )
            )
            component_tokens.append(
                self.component_fusion(
                    torch.cat(
                        (
                            mean_member,
                            max_member,
                            board_token,
                            self.structure_projection(structure),
                        )
                    )
                )
            )
            feasible_rows.append(self.grid - component.height + 1)
            feasible_columns.append(self.grid - component.width + 1)

        tokens = torch.stack(component_tokens)
        row_logits = self.row_head(tokens)
        column_logits = self.column_head(tokens)
        coordinates = torch.arange(self.grid, device=tile_tokens.device)[None, :]
        row_limit = torch.tensor(feasible_rows, device=tile_tokens.device)[:, None]
        column_limit = torch.tensor(feasible_columns, device=tile_tokens.device)[:, None]
        row_logits = row_logits.masked_fill(coordinates >= row_limit, -1e4)
        column_logits = column_logits.masked_fill(coordinates >= column_limit, -1e4)
        return ComponentShiftOutput(
            row_logits=row_logits,
            column_logits=column_logits,
            feasible_row_shifts=tuple(feasible_rows),
            feasible_column_shifts=tuple(feasible_columns),
        )


def dominant_component_shift_targets(
    components: tuple[ComponentDescriptor, ...],
    input_tile_to_position: torch.Tensor | np.ndarray,
    *,
    grid: int,
) -> tuple[ComponentShiftTarget, ...]:
    """Choose the feasible shift with maximum exact-tile support per component.

    False-bridge components remain in the target set.  Their dominant mode has
    purity below one and is down-weighted, never silently filtered out.
    """

    _validate_descriptors(components, grid=grid, require_partition=True)
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

    targets: list[ComponentShiftTarget] = []
    for component_index, component in enumerate(components):
        feasible_rows = grid - component.height + 1
        feasible_columns = grid - component.width + 1
        support = np.zeros((feasible_rows, feasible_columns), dtype=np.int32)
        for tile, relative_row, relative_column in zip(
            component.tiles,
            component.relative_rows,
            component.relative_columns,
            strict=True,
        ):
            true_row, true_column = divmod(int(positions[tile]), grid)
            row_shift = true_row - relative_row
            column_shift = true_column - relative_column
            if 0 <= row_shift < feasible_rows and 0 <= column_shift < feasible_columns:
                support[row_shift, column_shift] += 1
        # np.argmax supplies the fixed row-major tie break, including the rare
        # zero-support case.  Such a component still receives a positive floor
        # weight in component_shift_loss.
        flat_target = int(np.argmax(support))
        row_shift, column_shift = divmod(flat_target, feasible_columns)
        best_support = int(support[row_shift, column_shift])
        targets.append(
            ComponentShiftTarget(
                component_index=component_index,
                target_row_shift=row_shift,
                target_column_shift=column_shift,
                support=best_support,
                purity=best_support / component.size,
                size=component.size,
            )
        )
    return tuple(targets)


def component_shift_loss(
    output: ComponentShiftOutput,
    targets: tuple[ComponentShiftTarget, ...],
    *,
    impurity_weight_floor: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Size/purity-weighted feasible row+column cross entropy."""

    component_count = output.row_logits.shape[0]
    if output.row_logits.ndim != 2 or output.column_logits.shape != output.row_logits.shape:
        raise ValueError("row/column logits must have identical C x grid shape")
    if len(targets) != component_count or tuple(
        target.component_index for target in targets
    ) != tuple(range(component_count)):
        raise ValueError("targets must align one-to-one with output components")
    if not math.isfinite(impurity_weight_floor) or not 0 < impurity_weight_floor <= 1:
        raise ValueError("impurity_weight_floor must be in (0, 1]")
    device = output.row_logits.device
    row_target = torch.tensor(
        [target.target_row_shift for target in targets],
        device=device,
        dtype=torch.long,
    )
    column_target = torch.tensor(
        [target.target_column_shift for target in targets],
        device=device,
        dtype=torch.long,
    )
    row_loss = F.cross_entropy(output.row_logits, row_target, reduction="none")
    column_loss = F.cross_entropy(output.column_logits, column_target, reduction="none")
    weights = output.row_logits.new_tensor(
        [
            target.size
            * (impurity_weight_floor + (1.0 - impurity_weight_floor) * target.purity)
            for target in targets
        ]
    )
    loss = ((row_loss + column_loss) * weights).sum() / weights.sum().clamp_min(1e-8)
    with torch.no_grad():
        exact = (output.row_logits.argmax(1) == row_target) & (
            output.column_logits.argmax(1) == column_target
        )
        total_tiles = sum(target.size for target in targets)
        pure_tiles = sum(target.size for target in targets if target.purity == 1.0)
    return loss, {
        "component_shift_nll": float(loss.detach()),
        "component_count": float(component_count),
        "component_tiles": float(total_tiles),
        "pure_component_tile_fraction": pure_tiles / total_tiles,
        "mean_target_purity": float(np.mean([target.purity for target in targets])),
        "zero_support_components": float(sum(target.support == 0 for target in targets)),
        "component_shift_argmax_accuracy": float(exact.float().mean()),
        "mean_training_weight": float(weights.mean()),
    }


def component_shift_unary(
    output: ComponentShiftOutput,
    components: tuple[ComponentDescriptor, ...],
    *,
    grid: int,
    invalid_margin: float = 2.0,
) -> torch.Tensor:
    """Convert component shift distributions to decoder ``tile x slot`` unary.

    For every feasible rigid placement, summing the returned unary over all
    component members equals its factorised row+column log probability.  This
    is exactly the contract consumed by ``decode_socket_assignments``.
    """

    _validate_descriptors(components, grid=grid, require_partition=True)
    if output.row_logits.shape != (len(components), grid) or (
        output.column_logits.shape != output.row_logits.shape
    ):
        raise ValueError("component output shape disagrees with descriptors/grid")
    if not math.isfinite(invalid_margin) or invalid_margin <= 0:
        raise ValueError("invalid_margin must be finite and positive")
    count = grid * grid
    unary = output.row_logits.new_empty((count, count))
    for component_index, component in enumerate(components):
        feasible_rows = output.feasible_row_shifts[component_index]
        feasible_columns = output.feasible_column_shifts[component_index]
        if feasible_rows != grid - component.height + 1 or (
            feasible_columns != grid - component.width + 1
        ):
            raise ValueError("component output feasible-shift metadata is inconsistent")
        row_log_probability = F.log_softmax(
            output.row_logits[component_index, :feasible_rows], dim=0
        )
        column_log_probability = F.log_softmax(
            output.column_logits[component_index, :feasible_columns], dim=0
        )
        shift_score = row_log_probability[:, None] + column_log_probability[None, :]
        floor = (shift_score.min() - invalid_margin) / component.size
        tiles = torch.tensor(component.tiles, device=unary.device, dtype=torch.long)
        unary[tiles] = floor
        for tile, relative_row, relative_column in zip(
            component.tiles,
            component.relative_rows,
            component.relative_columns,
            strict=True,
        ):
            for row_shift in range(feasible_rows):
                for column_shift in range(feasible_columns):
                    slot = (relative_row + row_shift) * grid + relative_column + column_shift
                    unary[tile, slot] = shift_score[row_shift, column_shift] / component.size
    if not torch.isfinite(unary).all():
        raise RuntimeError("component shift unary contains non-finite entries")
    return unary


__all__ = [
    "ComponentDescriptor",
    "ComponentShiftHead",
    "ComponentShiftOutput",
    "ComponentShiftTarget",
    "component_descriptors_from_decoder",
    "component_shift_loss",
    "component_shift_unary",
    "dominant_component_shift_targets",
]
