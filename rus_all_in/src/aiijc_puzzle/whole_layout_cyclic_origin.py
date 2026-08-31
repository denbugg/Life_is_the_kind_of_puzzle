"""Learn one global cyclic origin from an already assembled dirty layout.

The model in this module never assigns individual tiles to absolute slots.  It
receives target-free per-tile evidence gathered into the coordinates of one
strict decoder layout and returns one score for each whole-board cyclic roll.
Circular convolutions and the absence of position embeddings make the scorer
equivariant to rolling its input grid.  Selection can therefore only apply a
``numpy.roll`` to the existing permutation of original upright tiles.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from aiijc_puzzle.component_anchor_diagnostic import DecoderComponentBuild
from aiijc_puzzle.socket_matcher import SocketOutput

RAW_FEATURE_NAMES = (
    *(f"rgb_mean_{channel}" for channel in "rgb"),
    *(f"rgb_std_{channel}" for channel in "rgb"),
    *(f"top_mean_{channel}" for channel in "rgb"),
    *(f"bottom_mean_{channel}" for channel in "rgb"),
    *(f"left_mean_{channel}" for channel in "rgb"),
    *(f"right_mean_{channel}" for channel in "rgb"),
    *(f"abs_dx_{channel}" for channel in "rgb"),
    *(f"abs_dy_{channel}" for channel in "rgb"),
    "luma_std",
)
SOCKET_FEATURE_NAMES = (
    "right_out_border",
    "left_in_border",
    "bottom_out_border",
    "top_in_border",
    "right_out_real_max",
    "left_in_real_max",
    "down_out_real_max",
    "top_in_real_max",
    "right_out_real_vs_bin",
    "left_in_real_vs_bin",
    "down_out_real_vs_bin",
    "top_in_real_vs_bin",
)
COMPONENT_FEATURE_NAMES = (
    "component_log_size",
    "component_height",
    "component_width",
    "component_density",
    "component_relative_row",
    "component_relative_column",
    "component_member_boundary",
    "component_edge_confidence",
)


@dataclass(frozen=True)
class WholeLayoutOriginConfig:
    """Small shift-equivariant CNN architecture."""

    input_channels: int
    width: int = 32
    dilations: tuple[int, ...] = (1, 2, 4, 8)

    def validate(self) -> None:
        if self.input_channels < 1:
            raise ValueError("input_channels must be positive")
        if self.width < 8:
            raise ValueError("width must be at least eight")
        if not self.dilations or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.dilations
        ):
            raise ValueError("dilations must be positive integers")


class _CircularResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        groups = math.gcd(width, max(1, width // 8))
        self.network = nn.Sequential(
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(
                width,
                width,
                3,
                padding=dilation,
                dilation=dilation,
                padding_mode="circular",
            ),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


class WholeLayoutCyclicOriginCNN(nn.Module):
    """Score all whole-board origins without an absolute position embedding."""

    def __init__(self, config: WholeLayoutOriginConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.stem = nn.Conv2d(config.input_channels, config.width, 1)
        self.blocks = nn.Sequential(
            *[
                _CircularResidualBlock(config.width, dilation)
                for dilation in config.dilations
            ]
        )
        groups = math.gcd(config.width, max(1, config.width // 8))
        self.head = nn.Sequential(
            nn.GroupNorm(groups, config.width),
            nn.GELU(),
            nn.Conv2d(config.width, 1, 1),
        )

    def anchor_logits(self, feature_grid: torch.Tensor) -> torch.Tensor:
        """Return a score for choosing each current grid cell as top-left."""

        if feature_grid.ndim != 4:
            raise ValueError("feature_grid must have shape B x C x G x G")
        if feature_grid.shape[1] != self.config.input_channels:
            raise ValueError("feature_grid channel count does not match the model")
        if feature_grid.shape[2] != feature_grid.shape[3] or feature_grid.shape[2] < 2:
            raise ValueError("feature_grid must have a square spatial lattice")
        return self.head(self.blocks(self.stem(feature_grid)))[:, 0]

    def forward(self, feature_grid: torch.Tensor) -> torch.Tensor:
        """Return logits indexed by the ``numpy.roll`` row/column shift."""

        return anchor_to_roll_logits(self.anchor_logits(feature_grid))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    result = value
    if hasattr(result, "detach"):
        result = result.detach()
    if hasattr(result, "cpu"):
        result = result.cpu()
    if hasattr(result, "numpy"):
        result = result.numpy()
    array = np.asarray(result, dtype=np.float32)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def raw_tile_features(tiles: Any) -> np.ndarray:
    """Return compact dirty-visible colour, boundary and gradient features."""

    value = np.asarray(tiles)
    if value.ndim != 4 or value.shape[1:] != (20, 20, 3):
        raise ValueError("tiles must have shape N x 20 x 20 x 3")
    if value.dtype == np.uint8:
        image = value.astype(np.float32) / 255.0
    else:
        image = value.astype(np.float32)
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("floating tiles must be finite and in [0, 1]")
    mean = image.mean(axis=(1, 2))
    standard_deviation = image.std(axis=(1, 2))
    top = image[:, 0].mean(axis=1)
    bottom = image[:, -1].mean(axis=1)
    left = image[:, :, 0].mean(axis=1)
    right = image[:, :, -1].mean(axis=1)
    dx = np.abs(np.diff(image, axis=2)).mean(axis=(1, 2))
    dy = np.abs(np.diff(image, axis=1)).mean(axis=(1, 2))
    luma = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    luma_std = luma.std(axis=(1, 2))[:, None]
    result = np.concatenate(
        (mean, standard_deviation, top, bottom, left, right, dx, dy, luma_std),
        axis=1,
    )
    if result.shape[1] != len(RAW_FEATURE_NAMES):
        raise RuntimeError("raw feature schema changed")
    return np.ascontiguousarray(result, dtype=np.float32)


def socket_tile_features(output: SocketOutput, *, grid: int) -> np.ndarray:
    """Return frozen border and real-vs-dustbin confidence per input tile."""

    count = grid * grid
    right = _as_numpy(output.right_log_assignment, name="right_log_assignment")
    down = _as_numpy(output.down_log_assignment, name="down_log_assignment")
    expected = (count + 1, count + 1)
    if right.shape != expected or down.shape != expected:
        raise ValueError(f"Socket assignments must have shape {expected}")
    borders = [
        _as_numpy(output.right_out_border_logits, name="right_out_border_logits"),
        _as_numpy(output.left_in_border_logits, name="left_in_border_logits"),
        _as_numpy(output.bottom_out_border_logits, name="bottom_out_border_logits"),
        _as_numpy(output.top_in_border_logits, name="top_in_border_logits"),
    ]
    if any(value.shape != (count,) for value in borders):
        raise ValueError("Socket border logits must contain one value per tile")
    right_out_max = right[:count, :count].max(axis=1)
    left_in_max = right[:count, :count].max(axis=0)
    down_out_max = down[:count, :count].max(axis=1)
    top_in_max = down[:count, :count].max(axis=0)
    real_maxima = (right_out_max, left_in_max, down_out_max, top_in_max)
    bins = (
        right[:count, count],
        right[count, :count],
        down[:count, count],
        down[count, :count],
    )
    margins = tuple(a - b for a, b in zip(real_maxima, bins, strict=True))
    result = np.stack((*borders, *real_maxima, *margins), axis=1)
    if result.shape != (count, len(SOCKET_FEATURE_NAMES)):
        raise RuntimeError("Socket feature schema changed")
    return np.ascontiguousarray(result, dtype=np.float32)


def component_tile_features(
    build: DecoderComponentBuild,
    *,
    grid: int,
) -> np.ndarray:
    """Describe target-blind decoder components for every member tile."""

    count = grid * grid
    result = np.zeros((count, len(COMPONENT_FEATURE_NAMES)), dtype=np.float32)
    observed: set[int] = set()
    for component in build.components:
        tiles = set(component)
        if not tiles or observed & tiles:
            raise ValueError("decoder components must be non-empty and disjoint")
        observed.update(tiles)
        coordinates = np.asarray([component[tile] for tile in sorted(tiles)], dtype=np.int32)
        height = int(coordinates[:, 0].max()) + 1
        width = int(coordinates[:, 1].max()) + 1
        size = len(tiles)
        accepted_confidence = [
            constraint.edge.confidence
            for constraint in build.constraints
            if constraint.status in {"added", "consistent"}
            and constraint.edge.source in tiles
            and constraint.edge.target in tiles
        ]
        confidence = float(np.mean(accepted_confidence)) if accepted_confidence else 0.0
        for tile, (row, column) in component.items():
            result[tile] = (
                math.log1p(size) / math.log1p(count),
                height / grid,
                width / grid,
                size / (height * width),
                row / max(1, height - 1),
                column / max(1, width - 1),
                float(row in {0, height - 1} or column in {0, width - 1}),
                confidence,
            )
    if observed != set(range(count)):
        raise ValueError("decoder components must partition every tile")
    return result


def combine_tile_features(
    tiles: Any,
    context_tokens: Any,
    socket_output: SocketOutput,
    component_build: DecoderComponentBuild,
    *,
    grid: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build and board-normalise the frozen per-tile feature matrix."""

    count = grid * grid
    context = _as_numpy(context_tokens, name="context_tokens")
    if context.ndim == 3 and context.shape[0] == 1:
        context = context[0]
    if context.ndim != 2 or context.shape[0] != count:
        raise ValueError("context_tokens must have shape N x D or 1 x N x D")
    raw = raw_tile_features(tiles)
    if len(raw) != count:
        raise ValueError("tile count does not match grid")
    values = np.concatenate(
        (
            raw,
            context,
            socket_tile_features(socket_output, grid=grid),
            component_tile_features(component_build, grid=grid),
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    centre = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    normalised = np.clip((values - centre) / np.maximum(scale, 1e-4), -5.0, 5.0)
    names = (
        *RAW_FEATURE_NAMES,
        *(f"d64_context_{index}" for index in range(context.shape[1])),
        *SOCKET_FEATURE_NAMES,
        *COMPONENT_FEATURE_NAMES,
    )
    if normalised.shape != (count, len(names)):
        raise RuntimeError("combined feature schema changed")
    return np.ascontiguousarray(normalised, dtype=np.float32), tuple(names)


def _strict_layout(layout: Any, *, grid: int) -> np.ndarray:
    count = grid * grid
    value = np.asarray(layout, dtype=np.int32)
    if value.shape != (count,) or not np.array_equal(np.sort(value), np.arange(count)):
        raise ValueError("layout must be a strict tile permutation")
    return np.ascontiguousarray(value)


def assemble_feature_grid(tile_features: Any, layout: Any, *, grid: int) -> np.ndarray:
    """Gather per-tile features into predicted tile-at-position coordinates."""

    features = np.asarray(tile_features, dtype=np.float32)
    count = grid * grid
    if features.ndim != 2 or features.shape[0] != count:
        raise ValueError("tile_features must have shape N x C")
    permutation = _strict_layout(layout, grid=grid)
    return np.ascontiguousarray(
        features[permutation].reshape(grid, grid, -1).transpose(2, 0, 1)
    )


def anchor_to_roll_logits(anchor_logits: torch.Tensor) -> torch.Tensor:
    """Map a top-left anchor map to logits indexed by positive roll shift."""

    if anchor_logits.ndim != 3 or anchor_logits.shape[1] != anchor_logits.shape[2]:
        raise ValueError("anchor_logits must have shape B x G x G")
    grid = anchor_logits.shape[1]
    negative = torch.remainder(-torch.arange(grid, device=anchor_logits.device), grid)
    return anchor_logits.index_select(1, negative).index_select(2, negative)


def cyclic_exact_counts(layout: Any, reference: Any, *, grid: int) -> np.ndarray:
    """Count exact tiles for every whole-layout cyclic roll."""

    predicted = _strict_layout(layout, grid=grid).reshape(grid, grid)
    target = _strict_layout(reference, grid=grid).reshape(grid, grid)
    counts = np.empty((grid, grid), dtype=np.int32)
    for row_roll in range(grid):
        for column_roll in range(grid):
            counts[row_roll, column_roll] = int(
                np.sum(
                    np.roll(
                        predicted,
                        shift=(row_roll, column_roll),
                        axis=(0, 1),
                    )
                    == target
                )
            )
    return counts


def best_roll_nll(
    roll_logits: torch.Tensor,
    exact_counts: torch.Tensor,
) -> torch.Tensor:
    """Negative log probability assigned to any maximum-exact roll."""

    if roll_logits.ndim != 3 or exact_counts.shape != roll_logits.shape:
        raise ValueError("roll_logits and exact_counts must share B x G x G shape")
    flat_logits = roll_logits.flatten(1)
    flat_counts = exact_counts.flatten(1)
    best = flat_counts == flat_counts.max(dim=1, keepdim=True).values
    masked = flat_logits.masked_fill(~best, -torch.inf)
    return (torch.logsumexp(flat_logits, dim=1) - torch.logsumexp(masked, dim=1)).mean()


@dataclass(frozen=True)
class LearnedCyclicOriginResult:
    layout: np.ndarray
    selected_roll: tuple[int, int]
    changed: bool
    strict_permutation: bool

    def report(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("layout")
        return payload


def select_learned_cyclic_origin(
    layout: Any,
    roll_logits: Any,
    *,
    grid: int,
) -> LearnedCyclicOriginResult:
    """Apply the learned top-scoring roll to the strict original permutation."""

    initial = _strict_layout(layout, grid=grid)
    scores = _as_numpy(roll_logits, name="roll_logits")
    if scores.shape != (grid, grid):
        raise ValueError(f"roll_logits must have shape {(grid, grid)}")
    flat_index = int(np.argmax(scores))
    row_roll, column_roll = divmod(flat_index, grid)
    result = np.roll(
        initial.reshape(grid, grid),
        shift=(row_roll, column_roll),
        axis=(0, 1),
    ).reshape(-1)
    result = _strict_layout(result, grid=grid)
    return LearnedCyclicOriginResult(
        layout=result,
        selected_roll=(row_roll, column_roll),
        changed=(row_roll, column_roll) != (0, 0),
        strict_permutation=True,
    )


def topk_hits_best_rolls(
    roll_logits: Any,
    exact_counts: Any,
    *,
    caps: Sequence[int] = (1, 5),
) -> dict[int, bool]:
    """Report whether each deterministic top-k contains a maximum-exact roll."""

    scores = np.asarray(roll_logits, dtype=np.float64)
    counts = np.asarray(exact_counts)
    if scores.ndim != 2 or counts.shape != scores.shape or scores.shape[0] != scores.shape[1]:
        raise ValueError("roll_logits and exact_counts must share a square shape")
    flat_scores = scores.reshape(-1)
    flat_counts = counts.reshape(-1)
    order = np.lexsort((np.arange(len(flat_scores)), -flat_scores))
    best = flat_counts == flat_counts.max()
    result: dict[int, bool] = {}
    for cap in caps:
        if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= len(order):
            raise ValueError("top-k caps must be valid positive integers")
        result[cap] = bool(np.any(best[order[:cap]]))
    return result


def uniform_best_roll_nll(exact_counts: Any) -> float:
    counts = np.asarray(exact_counts)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("exact_counts must be square")
    best_count = int(np.sum(counts == counts.max()))
    return math.log(counts.size / best_count)


def learned_best_roll_nll(roll_logits: Any, exact_counts: Any) -> float:
    scores = torch.as_tensor(np.asarray(roll_logits), dtype=torch.float64).unsqueeze(0)
    counts = torch.as_tensor(np.asarray(exact_counts)).unsqueeze(0)
    return float(best_roll_nll(scores, counts).item())
