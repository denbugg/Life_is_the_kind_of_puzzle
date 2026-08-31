"""Board-conditioned native-pixel component anchoring.

This module deliberately does not vote for one shared global roll.  A model
scores nontrivial frozen decoder components independently for internal purity
and an absolute feasible 2-D translation.  At most one component may be
anchored; every other original upright tile is packed collision-free while
minimising displacement from the frozen decoder layout.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from aiijc_puzzle.component_anchor_diagnostic import ComponentConstraint


@dataclass(frozen=True)
class ComponentAbsoluteConfig:
    """Small native-pixel and component-set model configuration."""

    grid: int = 24
    pixel_width: int = 24
    pixel_blocks: int = 3
    lattice_blocks: int = 3
    model_dimension: int = 64
    set_layers: int = 2
    set_heads: int = 4
    geometry_dimension: int = 12

    def validate(self) -> None:
        if self.grid < 2 or self.pixel_width < 8:
            raise ValueError("grid and pixel width are too small")
        if self.pixel_blocks < 1 or self.lattice_blocks < 1 or self.set_layers < 1:
            raise ValueError("block and set-layer counts must be positive")
        if self.model_dimension % self.set_heads:
            raise ValueError("model dimension must be divisible by set heads")
        if self.geometry_dimension != 12:
            raise ValueError("v1 geometry contract has exactly 12 features")


@dataclass(frozen=True)
class AnchorPackingDiagnostics:
    anchor_component_index: int
    anchor_size: int
    anchor_row_offset: int
    anchor_column_offset: int
    baseline_preserved_components: int
    repacked_components: int
    deferred_tiles: int
    total_tile_l1_displacement: int
    strict_permutation: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise_component(
    component: Mapping[int, tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    if not component:
        raise ValueError("component must not be empty")
    minimum_row = min(row for row, _ in component.values())
    minimum_column = min(column for _, column in component.values())
    result = {
        int(tile): (int(row - minimum_row), int(column - minimum_column))
        for tile, (row, column) in component.items()
    }
    if len(result) != len(component):
        raise ValueError("component has duplicate tile identifiers")
    if len(set(result.values())) != len(result):
        raise ValueError("component has colliding relative coordinates")
    return result


def _component_shape(component: Mapping[int, tuple[int, int]]) -> tuple[int, int]:
    return (
        max(row for row, _ in component.values()) + 1,
        max(column for _, column in component.values()) + 1,
    )


def validate_component_partition(
    components: Sequence[Mapping[int, tuple[int, int]]],
    *,
    grid: int,
) -> tuple[dict[int, tuple[int, int]], ...]:
    """Validate and normalise one exact partition of all ``grid**2`` tiles."""

    normalised = tuple(_normalise_component(component) for component in components)
    tiles = [tile for component in normalised for tile in component]
    count = grid * grid
    if sorted(tiles) != list(range(count)):
        raise ValueError("components must partition every original tile exactly once")
    return normalised


def render_native_component_mosaic(
    raw_tiles: torch.Tensor,
    component: Mapping[int, tuple[int, int]],
) -> torch.Tensor:
    """Render raw+tile-normalised pixels and mask without resizing any tile."""

    if raw_tiles.ndim != 4 or raw_tiles.shape[1:] != (3, 20, 20):
        raise ValueError("raw_tiles must have shape N x 3 x 20 x 20")
    if raw_tiles.min() < 0 or raw_tiles.max() > 1:
        raise ValueError("raw_tiles must lie in [0,1]")
    component = _normalise_component(component)
    height, width = _component_shape(component)
    result = raw_tiles.new_zeros((7, height * 20, width * 20))
    for tile, (row, column) in component.items():
        pixels = raw_tiles[tile]
        mean = pixels.mean(dim=(-2, -1), keepdim=True)
        scale = pixels.std(dim=(-2, -1), keepdim=True).clamp_min(1.0 / 255.0)
        normalised = ((pixels - mean) / scale).clamp(-5.0, 5.0)
        row_slice = slice(row * 20, (row + 1) * 20)
        column_slice = slice(column * 20, (column + 1) * 20)
        result[:3, row_slice, column_slice] = pixels
        result[3:6, row_slice, column_slice] = normalised
        result[6, row_slice, column_slice] = 1.0
    return result


class _PixelBlock(nn.Module):
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


class _MaskedLatticeBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        groups = math.gcd(width, max(1, width // 8))
        self.normalise = nn.GroupNorm(groups, width)
        self.convolution = nn.Conv2d(width, width, 3, padding=1)
        self.project = nn.Conv2d(width, width, 1)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        update = self.project(F.gelu(self.convolution(self.normalise(value))))
        return (value + update) * mask


class ComponentAbsolutePlacerModel(nn.Module):
    """Permutation-equivariant component purity and joint offset classifier."""

    def __init__(self, config: ComponentAbsoluteConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.pixel_width
        self.pixel_stem = nn.Conv2d(6, width, 3, padding=1)
        self.pixel_blocks = nn.Sequential(
            *[_PixelBlock(width) for _ in range(config.pixel_blocks)]
        )
        self.tile_projection = nn.Sequential(
            nn.LayerNorm(width * 6),
            nn.Linear(width * 6, width),
            nn.GELU(),
        )
        self.lattice_blocks = nn.ModuleList(
            [_MaskedLatticeBlock(width) for _ in range(config.lattice_blocks)]
        )
        self.component_projection = nn.Sequential(
            nn.LayerNorm(width * 4 + config.geometry_dimension),
            nn.Linear(width * 4 + config.geometry_dimension, config.model_dimension),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dimension,
            nhead=config.set_heads,
            dim_feedforward=config.model_dimension * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.board_context = nn.TransformerEncoder(layer, num_layers=config.set_layers)
        self.purity_head = nn.Linear(config.model_dimension, 1)
        self.offset_head = nn.Linear(
            config.model_dimension,
            config.grid * config.grid,
        )

    def _native_tile_tokens(
        self,
        mosaics: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, list[tuple[int, int, torch.Tensor]]]:
        patches: list[torch.Tensor] = []
        contracts: list[tuple[int, int, torch.Tensor]] = []
        for mosaic in mosaics:
            if mosaic.ndim != 3 or mosaic.shape[0] != 7:
                raise ValueError("component mosaics must have shape 7 x 20H x 20W")
            if mosaic.shape[1] % 20 or mosaic.shape[2] % 20:
                raise ValueError("component mosaic dimensions must be multiples of 20")
            height, width = mosaic.shape[1] // 20, mosaic.shape[2] // 20
            cells = F.unfold(mosaic[:6].unsqueeze(0), kernel_size=20, stride=20)
            cells = cells.transpose(1, 2).reshape(height * width, 6, 20, 20)
            mask_cells = F.unfold(
                mosaic[6:].unsqueeze(0), kernel_size=20, stride=20
            ).mean(dim=1)[0]
            occupied = mask_cells > 0.5
            if not occupied.any():
                raise ValueError("component mosaic has no occupied native tile")
            patches.append(cells[occupied])
            contracts.append((height, width, occupied))
        pixels = torch.cat(patches, dim=0)
        field = self.pixel_blocks(self.pixel_stem(pixels))
        summaries = (
            field.mean(dim=(-2, -1)),
            field.amax(dim=(-2, -1)),
            field[..., :4, :].mean(dim=(-2, -1)),
            field[..., -4:, :].mean(dim=(-2, -1)),
            field[..., :, :4].mean(dim=(-2, -1)),
            field[..., :, -4:].mean(dim=(-2, -1)),
        )
        return self.tile_projection(torch.cat(summaries, dim=-1)), contracts

    def forward(
        self,
        mosaics: Sequence[torch.Tensor],
        geometry_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return component purity logits and masked joint 24x24 offset logits."""

        count = len(mosaics)
        if count < 1 or geometry_features.shape != (
            count,
            self.config.geometry_dimension,
        ):
            raise ValueError("geometry features do not match component mosaics")
        tile_tokens, contracts = self._native_tile_tokens(mosaics)
        width = self.config.pixel_width
        maximum_height = max(item[0] for item in contracts)
        maximum_width = max(item[1] for item in contracts)
        lattice = tile_tokens.new_zeros((count, width, maximum_height, maximum_width))
        mask = tile_tokens.new_zeros((count, 1, maximum_height, maximum_width))
        offset = 0
        raw_mean: list[torch.Tensor] = []
        raw_max: list[torch.Tensor] = []
        for index, (height, width_cells, occupied) in enumerate(contracts):
            amount = int(occupied.sum())
            tokens = tile_tokens[offset : offset + amount]
            offset += amount
            occupied_grid = occupied.reshape(height, width_cells)
            row_indices, column_indices = torch.nonzero(
                occupied_grid,
                as_tuple=True,
            )
            lattice[index, :, row_indices, column_indices] = tokens.T
            mask[index, :, row_indices, column_indices] = 1.0
            raw_mean.append(tokens.mean(dim=0))
            raw_max.append(tokens.amax(dim=0))
        for block in self.lattice_blocks:
            lattice = block(lattice, mask)
        denominator = mask.sum(dim=(-2, -1)).clamp_min(1.0)
        lattice_mean = (lattice * mask).sum(dim=(-2, -1)) / denominator
        lattice_max = lattice.masked_fill(mask == 0, -torch.inf).amax(dim=(-2, -1))
        component = self.component_projection(
            torch.cat(
                (
                    torch.stack(raw_mean),
                    torch.stack(raw_max),
                    lattice_mean,
                    lattice_max,
                    geometry_features,
                ),
                dim=-1,
            )
        )
        contextual = self.board_context(component.unsqueeze(0))[0]
        purity = self.purity_head(contextual)[:, 0]
        offsets = self.offset_head(contextual)
        heights = geometry_features[:, 2].mul(self.config.grid).round().long()
        widths = geometry_features[:, 3].mul(self.config.grid).round().long()
        feasible = feasible_offset_mask(
            heights,
            widths,
            grid=self.config.grid,
        )
        return purity, offsets.masked_fill(~feasible, -torch.inf)


def feasible_offset_mask(
    heights: Any,
    widths: Any,
    *,
    grid: int,
) -> torch.Tensor:
    height = torch.as_tensor(heights, dtype=torch.long)
    width = torch.as_tensor(widths, dtype=torch.long, device=height.device)
    if height.ndim != 1 or width.shape != height.shape:
        raise ValueError("heights and widths must be aligned vectors")
    if ((height < 1) | (height > grid) | (width < 1) | (width > grid)).any():
        raise ValueError("component shapes must fit the board")
    rows = torch.arange(grid, device=height.device)[None, :, None]
    columns = torch.arange(grid, device=height.device)[None, None, :]
    return ((rows + height[:, None, None] <= grid) & (
        columns + width[:, None, None] <= grid
    )).reshape(len(height), grid * grid)


def component_absolute_targets(
    components: Sequence[Mapping[int, tuple[int, int]]],
    tile_to_position: Any,
    *,
    grid: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact-purity labels, pure offsets, and dominant support fractions."""

    positions = np.asarray(tile_to_position, dtype=np.int64)
    count = grid * grid
    if positions.shape != (count,) or not np.array_equal(np.sort(positions), np.arange(count)):
        raise ValueError("tile_to_position must be one strict permutation")
    true_row, true_column = divmod(positions, grid)
    purity: list[bool] = []
    offset: list[int] = []
    support_fraction: list[float] = []
    for raw_component in components:
        component = _normalise_component(raw_component)
        shifts = Counter(
            (
                int(true_row[tile]) - row,
                int(true_column[tile]) - column,
            )
            for tile, (row, column) in component.items()
        )
        shift, support = min(shifts.items(), key=lambda item: (-item[1], item[0]))
        height, width = _component_shape(component)
        is_feasible = (
            0 <= shift[0] <= grid - height and 0 <= shift[1] <= grid - width
        )
        exact = support == len(component) and is_feasible
        purity.append(exact)
        offset.append(shift[0] * grid + shift[1] if exact else -1)
        support_fraction.append(support / len(component))
    return (
        torch.tensor(purity, dtype=torch.bool),
        torch.tensor(offset, dtype=torch.long),
        torch.tensor(support_fraction, dtype=torch.float32),
    )


def align_components_across_corruptions(
    first_components: Sequence[Mapping[int, tuple[int, int]]],
    first_tile_to_position: Any,
    second_components: Sequence[Mapping[int, tuple[int, int]]],
    second_tile_to_position: Any,
    *,
    grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align train-only component views by true member positions and geometry.

    Input tile identifiers change across synthetic corruption draws.  Using
    exact source positions here is therefore intentional and restricted to the
    supervised fit split; the returned alignment is never an inference input.
    """

    count = grid * grid

    def signatures(
        components: Sequence[Mapping[int, tuple[int, int]]],
        tile_to_position: Any,
    ) -> dict[tuple[tuple[int, int, int], ...], int]:
        positions = np.asarray(tile_to_position, dtype=np.int64)
        if positions.shape != (count,) or not np.array_equal(
            np.sort(positions), np.arange(count)
        ):
            raise ValueError("tile_to_position must be one strict permutation")
        result: dict[tuple[tuple[int, int, int], ...], int] = {}
        for index, raw_component in enumerate(components):
            component = _normalise_component(raw_component)
            signature = tuple(
                sorted(
                    (int(positions[tile]), int(row), int(column))
                    for tile, (row, column) in component.items()
                )
            )
            if signature in result:
                raise ValueError("component geometry signature is not unique")
            result[signature] = index
        return result

    first = signatures(first_components, first_tile_to_position)
    second = signatures(second_components, second_tile_to_position)
    shared = sorted(first.keys() & second.keys())
    return (
        np.asarray([first[item] for item in shared], dtype=np.int64),
        np.asarray([second[item] for item in shared], dtype=np.int64),
    )


def paired_component_consistency_loss(
    first_purity_logits: torch.Tensor,
    first_offset_logits: torch.Tensor,
    second_purity_logits: torch.Tensor,
    second_offset_logits: torch.Tensor,
    first_indices: Any,
    second_indices: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric paired-draw consistency on exactly aligned components."""

    first_index = torch.as_tensor(
        first_indices,
        dtype=torch.long,
        device=first_purity_logits.device,
    )
    second_index = torch.as_tensor(
        second_indices,
        dtype=torch.long,
        device=second_purity_logits.device,
    )
    if first_index.ndim != 1 or second_index.shape != first_index.shape:
        raise ValueError("paired component index vectors must align")
    if first_offset_logits.shape[0] != len(first_purity_logits) or (
        second_offset_logits.shape[0] != len(second_purity_logits)
    ):
        raise ValueError("paired logits must align within each corruption")
    if not len(first_index):
        zero = first_purity_logits.sum() * 0.0
        return zero, {"aligned_components": 0.0, "paired_consistency": 0.0}
    first_purity = torch.sigmoid(first_purity_logits[first_index])
    second_purity = torch.sigmoid(second_purity_logits[second_index])
    purity_loss = F.smooth_l1_loss(first_purity, second_purity)
    first_offset = torch.softmax(first_offset_logits[first_index], dim=-1)
    second_offset = torch.softmax(second_offset_logits[second_index], dim=-1)
    offset_loss = ((first_offset - second_offset) ** 2).sum(dim=-1).mean()
    loss = purity_loss + offset_loss
    return loss, {
        "aligned_components": float(len(first_index)),
        "paired_consistency": float(loss.detach()),
    }


def component_absolute_loss(
    purity_logits: torch.Tensor,
    offset_logits: torch.Tensor,
    purity_targets: torch.Tensor,
    offset_targets: torch.Tensor,
    *,
    component_sizes: torch.Tensor | None = None,
    offset_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Balanced purity BCE plus size-weighted offset CE on exact components only."""

    if purity_logits.ndim != 1 or offset_logits.shape[0] != len(purity_logits):
        raise ValueError("purity and offset logits must align by component")
    labels = purity_targets.to(device=purity_logits.device, dtype=torch.bool)
    offsets = offset_targets.to(device=purity_logits.device, dtype=torch.long)
    if labels.shape != purity_logits.shape or offsets.shape != labels.shape:
        raise ValueError("component targets must align with logits")
    positives = labels.sum().clamp_min(1)
    negatives = (~labels).sum().clamp_min(1)
    positive_weight = negatives.float() / positives.float()
    weights = torch.where(labels, positive_weight, 1.0)
    purity_loss = (
        F.binary_cross_entropy_with_logits(
            purity_logits,
            labels.float(),
            reduction="none",
        )
        * weights
    ).mean()
    offset_loss = purity_logits.new_zeros(())
    if labels.any():
        raw = F.cross_entropy(offset_logits[labels], offsets[labels], reduction="none")
        if component_sizes is not None:
            sizes = component_sizes.to(device=raw.device, dtype=raw.dtype)[labels]
            raw = raw * sizes / sizes.mean().clamp_min(1.0)
        offset_loss = raw.mean()
    loss = purity_loss + float(offset_weight) * offset_loss
    return loss, {
        "purity_bce": float(purity_loss.detach()),
        "offset_ce": float(offset_loss.detach()),
        "loss": float(loss.detach()),
        "pure_fraction": float(labels.float().mean()),
    }


def component_geometry_features(
    components: Sequence[Mapping[int, tuple[int, int]]],
    constraints: Sequence[ComponentConstraint],
    baseline_layout: Any,
    *,
    grid: int,
) -> np.ndarray:
    """Build the fixed 12-D dirty-visible component geometry contract."""

    count = grid * grid
    baseline = np.asarray(baseline_layout, dtype=np.int64)
    if baseline.shape != (count,) or not np.array_equal(np.sort(baseline), np.arange(count)):
        raise ValueError("baseline layout must be a strict permutation")
    baseline_position = np.empty((count, 2), dtype=np.int32)
    baseline_position[baseline, 0], baseline_position[baseline, 1] = divmod(
        np.arange(count), grid
    )
    rows: list[list[float]] = []
    confidence_columns: list[tuple[float, float, float]] = []
    for raw_component in components:
        component = _normalise_component(raw_component)
        tiles = set(component)
        internal = [
            item.edge.confidence
            for item in constraints
            if item.status in {"added", "consistent"}
            and item.edge.source in tiles
            and item.edge.target in tiles
        ]
        confidence_columns.append(
            (
                float(np.mean(internal)) if internal else 0.0,
                float(np.min(internal)) if internal else 0.0,
                float(np.max(internal)) if internal else 0.0,
            )
        )
    confidence = np.asarray(confidence_columns, dtype=np.float64)
    confidence = (confidence - confidence.mean(axis=0, keepdims=True)) / np.maximum(
        confidence.std(axis=0, keepdims=True), 1e-6
    )
    for index, raw_component in enumerate(components):
        component = _normalise_component(raw_component)
        height, width = _component_shape(component)
        shifts = Counter(
            (
                int(baseline_position[tile, 0]) - row,
                int(baseline_position[tile, 1]) - column,
            )
            for tile, (row, column) in component.items()
        )
        shift, support = min(shifts.items(), key=lambda item: (-item[1], item[0]))
        rows.append(
            [
                len(component) / count,
                math.log1p(len(component)) / math.log1p(count),
                height / grid,
                width / grid,
                len(component) / (height * width),
                confidence[index, 0],
                confidence[index, 1],
                confidence[index, 2],
                shift[0] / max(1, grid - 1),
                shift[1] / max(1, grid - 1),
                support / len(component),
                (height * width) / count,
            ]
        )
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (len(components), 12) or not np.isfinite(result).all():
        raise RuntimeError("component geometry feature contract failed")
    return result


def average_precision(labels: Any, scores: Any) -> float:
    """Stable binary average precision without a sklearn dependency."""

    truth = np.asarray(labels, dtype=bool)
    value = np.asarray(scores, dtype=np.float64)
    if truth.ndim != 1 or value.shape != truth.shape or not np.isfinite(value).all():
        raise ValueError("labels and scores must be aligned finite vectors")
    positives = int(truth.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-value, kind="stable")
    ranked = truth[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def anchor_confidence(
    purity_logits: torch.Tensor,
    offset_logits: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return calibrated product score and predicted joint offset per component."""

    if purity_logits.ndim != 1 or offset_logits.shape[0] != len(purity_logits):
        raise ValueError("purity and offset logits must align")
    purity = torch.sigmoid(purity_logits)
    probability = torch.softmax(offset_logits, dim=-1)
    offset_probability, offset = probability.max(dim=-1)
    return (
        (purity * offset_probability).detach().cpu().numpy().astype(np.float64),
        offset.detach().cpu().numpy().astype(np.int32),
    )


def place_one_component_anchor(
    components: Sequence[Mapping[int, tuple[int, int]]],
    baseline_layout: Any,
    *,
    anchor_component_index: int,
    anchor_offset: int,
    grid: int,
) -> tuple[np.ndarray, AnchorPackingDiagnostics]:
    """Anchor one component and conservatively repack all remaining tiles."""

    normalised = validate_component_partition(components, grid=grid)
    count = grid * grid
    baseline = np.asarray(baseline_layout, dtype=np.int32)
    if baseline.shape != (count,) or not np.array_equal(np.sort(baseline), np.arange(count)):
        raise ValueError("baseline layout must be a strict permutation")
    if not 0 <= anchor_component_index < len(normalised):
        raise ValueError("anchor component index is out of range")
    if not 0 <= anchor_offset < count:
        raise ValueError("anchor offset is out of range")
    anchor = normalised[anchor_component_index]
    anchor_height, anchor_width = _component_shape(anchor)
    anchor_row, anchor_column = divmod(anchor_offset, grid)
    if anchor_row + anchor_height > grid or anchor_column + anchor_width > grid:
        raise ValueError("anchor offset is infeasible for the component shape")
    baseline_position = np.empty((count, 2), dtype=np.int32)
    baseline_position[baseline, 0], baseline_position[baseline, 1] = divmod(
        np.arange(count), grid
    )
    board = np.full((grid, grid), -1, dtype=np.int32)

    def coordinates(
        component: Mapping[int, tuple[int, int]], row_shift: int, column_shift: int
    ) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (tile, row + row_shift, column + column_shift)
            for tile, (row, column) in component.items()
        )

    for tile, row, column in coordinates(anchor, anchor_row, anchor_column):
        board[row, column] = tile
    preserved = 0
    repack_queue: list[tuple[int, Mapping[int, tuple[int, int]]]] = []
    for index, component in enumerate(normalised):
        if index == anchor_component_index:
            continue
        shifts = Counter(
            (
                int(baseline_position[tile, 0]) - row,
                int(baseline_position[tile, 1]) - column,
            )
            for tile, (row, column) in component.items()
        )
        shift, support = min(shifts.items(), key=lambda item: (-item[1], item[0]))
        height, width = _component_shape(component)
        baseline_rigid = support == len(component) and (
            0 <= shift[0] <= grid - height and 0 <= shift[1] <= grid - width
        )
        locations = coordinates(component, *shift) if baseline_rigid else ()
        if locations and all(board[row, column] < 0 for _, row, column in locations):
            for tile, row, column in locations:
                board[row, column] = tile
            preserved += 1
        else:
            repack_queue.append((index, component))
    repacked = 0
    deferred: list[int] = []
    for _, component in sorted(
        repack_queue,
        key=lambda item: (-len(item[1]), min(item[1])),
    ):
        height, width = _component_shape(component)
        best: tuple[int, int, int] | None = None
        for row_shift in range(grid - height + 1):
            for column_shift in range(grid - width + 1):
                locations = coordinates(component, row_shift, column_shift)
                if any(board[row, column] >= 0 for _, row, column in locations):
                    continue
                displacement = sum(
                    abs(int(baseline_position[tile, 0]) - row)
                    + abs(int(baseline_position[tile, 1]) - column)
                    for tile, row, column in locations
                )
                candidate = (displacement, row_shift, column_shift)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            deferred.extend(component)
            continue
        for tile, row, column in coordinates(component, best[1], best[2]):
            board[row, column] = tile
        repacked += 1
    empty = np.argwhere(board < 0)
    if len(empty) != len(deferred):
        raise RuntimeError("anchor packing lost or duplicated tiles")
    if deferred:
        cost = np.empty((len(deferred), len(empty)), dtype=np.float64)
        for tile_index, tile in enumerate(deferred):
            cost[tile_index] = np.abs(empty - baseline_position[tile]).sum(axis=1)
        tile_rows, slot_columns = linear_sum_assignment(cost)
        for tile_row, slot_column in zip(tile_rows, slot_columns, strict=True):
            row, column = empty[slot_column]
            board[row, column] = deferred[tile_row]
    result = np.ascontiguousarray(board.reshape(-1), dtype=np.int32)
    if not np.array_equal(np.sort(result), np.arange(count)):
        raise RuntimeError("anchor packing output is not a strict permutation")
    result_position = np.empty((count, 2), dtype=np.int32)
    result_position[result, 0], result_position[result, 1] = divmod(np.arange(count), grid)
    displacement = int(np.abs(result_position - baseline_position).sum())
    return result, AnchorPackingDiagnostics(
        anchor_component_index=anchor_component_index,
        anchor_size=len(anchor),
        anchor_row_offset=anchor_row,
        anchor_column_offset=anchor_column,
        baseline_preserved_components=preserved,
        repacked_components=repacked,
        deferred_tiles=len(deferred),
        total_tile_l1_displacement=displacement,
        strict_permutation=True,
    )


__all__ = [
    "AnchorPackingDiagnostics",
    "ComponentAbsoluteConfig",
    "ComponentAbsolutePlacerModel",
    "anchor_confidence",
    "average_precision",
    "align_components_across_corruptions",
    "component_absolute_loss",
    "component_absolute_targets",
    "component_geometry_features",
    "feasible_offset_mask",
    "paired_component_consistency_loss",
    "place_one_component_anchor",
    "render_native_component_mosaic",
    "validate_component_partition",
]
