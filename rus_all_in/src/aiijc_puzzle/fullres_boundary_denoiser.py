"""Full-resolution tile denoiser used only as an auxiliary matcher view.

The challenge fragments are only 20x20 pixels.  This module therefore avoids
pooling, strided convolutions and any other spatial resampling: every learned
feature map remains 20x20.  The network predicts a bounded residual around one
upright dirty tile and is shared independently across tiles.

The public inference helper deliberately returns an ``N x 20 x 20 x 3`` tile
view, not an assembled canvas or a layout.  A sorter may use that view to build
descriptors or frozen SocketMatcher scores, but the legal output remains a
strict permutation of the original raw tiles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

TILE_SIZE = 20


@dataclass(frozen=True)
class FullResolutionDenoiserConfig:
    """Architecture and residual bound for the matcher-only denoiser."""

    width: int = 32
    blocks: int = 8
    residual_limit: float = 64.0 / 255.0

    def validate(self) -> None:
        if isinstance(self.width, bool) or self.width < 8 or self.width % 2:
            raise ValueError("width must be an even integer >= 8")
        if isinstance(self.blocks, bool) or self.blocks < 1:
            raise ValueError("blocks must be a positive integer")
        if not 0.0 < self.residual_limit <= 1.0:
            raise ValueError("residual_limit must be in (0, 1]")


class SimpleGate(nn.Module):
    """Parameter-free NAF multiplicative gate."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return first * second


class FullResolutionNAFBlock(nn.Module):
    """NAF-style residual block that preserves height and width exactly."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded = channels * 2
        self.norm = nn.GroupNorm(1, channels)
        self.in_project = nn.Conv2d(channels, expanded, 1)
        self.depthwise = nn.Conv2d(
            expanded,
            expanded,
            3,
            padding=1,
            groups=expanded,
        )
        self.gate = SimpleGate()
        self.out_project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        branch = self.depthwise(self.in_project(self.norm(value)))
        branch = self.gate(branch)
        return value + self.scale * self.out_project(branch)


def per_tile_standardise(value: torch.Tensor) -> torch.Tensor:
    """Expose shape while bounding sensitivity to per-tile gain and bias.

    Standardisation is per tile and channel.  The original RGB tensor is still
    supplied alongside this view, so the residual predictor retains absolute
    colour information rather than trying to reconstruct it from normalised
    pixels alone.
    """

    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("value must have shape (batch, 3, height, width)")
    mean = value.mean(dim=(-2, -1), keepdim=True)
    variance = (value - mean).square().mean(dim=(-2, -1), keepdim=True)
    normalised = (value - mean) * torch.rsqrt(variance + 1e-4)
    return normalised.clamp(-3.0, 3.0) / 3.0


class FullResolutionBoundaryDenoiser(nn.Module):
    """Zero-initialised bounded residual model with no spatial downsampling."""

    def __init__(self, config: FullResolutionDenoiserConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = FullResolutionDenoiserConfig()
        config.validate()
        self.config = config
        self.intro = nn.Conv2d(6, config.width, 3, padding=1)
        self.body = nn.Sequential(
            *(FullResolutionNAFBlock(config.width) for _ in range(config.blocks))
        )
        self.ending = nn.Conv2d(config.width, 3, 3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, dirty: torch.Tensor) -> torch.Tensor:
        if dirty.ndim != 4 or dirty.shape[1:] != (3, TILE_SIZE, TILE_SIZE):
            raise ValueError("dirty must have shape (batch, 3, 20, 20)")
        if not torch.is_floating_point(dirty):
            raise ValueError("dirty must be floating point in [0, 1]")
        features = self.body(self.intro(torch.cat((dirty, per_tile_standardise(dirty)), 1)))
        residual = self.config.residual_limit * torch.tanh(self.ending(features))
        return torch.clamp(dirty + residual, 0.0, 1.0)


def model_config_dict(model: FullResolutionBoundaryDenoiser) -> dict[str, Any]:
    """Return JSON-compatible architecture metadata."""

    return asdict(model.config)


def boundary_mask(
    *,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Return a ``1x1x20x20`` perimeter mask of the requested width."""

    if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 10:
        raise ValueError("width must be an integer in [1, 10]")
    mask = torch.zeros((1, 1, TILE_SIZE, TILE_SIZE), device=device, dtype=torch.bool)
    mask[..., :width, :] = True
    mask[..., -width:, :] = True
    mask[..., :, :width] = True
    mask[..., :, -width:] = True
    return mask.to(dtype=dtype)


def _charbonnier(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.sqrt(value.square() + epsilon**2) - epsilon


def _standardise_masked(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(dim=(-2, -1), keepdim=True)
    mean = (value * mask).sum(dim=(-2, -1), keepdim=True) / count
    variance = ((value - mean).square() * mask).sum(dim=(-2, -1), keepdim=True) / count
    return ((value - mean) * torch.rsqrt(variance + 1e-4)).clamp(-3.0, 3.0) / 3.0


def boundary_denoising_loss(
    prediction: torch.Tensor,
    clean: torch.Tensor,
    dirty: torch.Tensor,
    *,
    border_width: int = 6,
    gradient_weight: float = 0.35,
    shape_weight: float = 0.15,
    identity_weight: float = 0.02,
    epsilon: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Boundary-focused clean supervision with a small identity prior.

    The primary term restores clean border-strip RGB.  Finite-difference and
    normalised-shape terms preserve boundary phase despite strong per-tile
    brightness/contrast corruption.  The final weak full-tile penalty prevents
    unsupported interior changes; it does not turn this into a generic canvas
    restoration objective.
    """

    if prediction.shape != clean.shape or prediction.shape != dirty.shape:
        raise ValueError("prediction, clean and dirty must have equal shapes")
    if prediction.ndim != 4 or prediction.shape[1:] != (3, TILE_SIZE, TILE_SIZE):
        raise ValueError("inputs must have shape (batch, 3, 20, 20)")
    weights = (gradient_weight, shape_weight, identity_weight)
    if any(not np.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("loss weights must be finite and non-negative")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    mask = boundary_mask(
        width=border_width,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    denominator = mask.sum() * prediction.shape[0] * prediction.shape[1]
    border = (_charbonnier(prediction - clean, epsilon) * mask).sum() / denominator

    pred_shape = _standardise_masked(prediction, mask)
    clean_shape = _standardise_masked(clean, mask)
    shape = (_charbonnier(pred_shape - clean_shape, epsilon) * mask).sum() / denominator

    horizontal_mask = torch.maximum(mask[..., :, 1:], mask[..., :, :-1])
    vertical_mask = torch.maximum(mask[..., 1:, :], mask[..., :-1, :])
    pred_horizontal = prediction[..., :, 1:] - prediction[..., :, :-1]
    clean_horizontal = clean[..., :, 1:] - clean[..., :, :-1]
    pred_vertical = prediction[..., 1:, :] - prediction[..., :-1, :]
    clean_vertical = clean[..., 1:, :] - clean[..., :-1, :]
    horizontal_denominator = (
        horizontal_mask.sum() * prediction.shape[0] * prediction.shape[1]
    )
    vertical_denominator = vertical_mask.sum() * prediction.shape[0] * prediction.shape[1]
    gradient = 0.5 * (
        (_charbonnier(pred_horizontal - clean_horizontal, epsilon) * horizontal_mask).sum()
        / horizontal_denominator
        + (_charbonnier(pred_vertical - clean_vertical, epsilon) * vertical_mask).sum()
        / vertical_denominator
    )
    identity = _charbonnier(prediction - dirty, epsilon).mean()
    total = border + gradient_weight * gradient + shape_weight * shape + identity_weight * identity
    return total, {
        "border": border,
        "gradient": gradient,
        "shape": shape,
        "identity": identity,
        "total": total,
    }


@torch.inference_mode()
def restore_matcher_view(
    model: FullResolutionBoundaryDenoiser,
    raw_tiles: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 576,
) -> np.ndarray:
    """Return a denoised tile view for matching; never use it as output pixels."""

    tiles = np.asarray(raw_tiles)
    if tiles.ndim != 4 or tiles.shape[1:] != (20, 20, 3) or tiles.dtype != np.uint8:
        raise ValueError("raw_tiles must be uint8 N x 20 x 20 x 3")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    restored: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(tiles), batch_size):
        tensor = (
            torch.from_numpy(np.ascontiguousarray(tiles[start : start + batch_size]))
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=torch.float32)
            .div_(255.0)
        )
        prediction = model(tensor)
        restored.append(
            prediction.mul(255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
    return np.ascontiguousarray(np.concatenate(restored, axis=0))


__all__ = [
    "FullResolutionBoundaryDenoiser",
    "FullResolutionDenoiserConfig",
    "FullResolutionNAFBlock",
    "boundary_denoising_loss",
    "boundary_mask",
    "model_config_dict",
    "per_tile_standardise",
    "restore_matcher_view",
]
