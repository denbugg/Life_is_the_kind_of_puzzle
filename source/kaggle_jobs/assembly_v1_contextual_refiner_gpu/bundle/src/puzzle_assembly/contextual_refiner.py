"""Conservative context features and a bounded post-assembly residual model.

This module deliberately cannot rearrange tiles and never accepts a clean
target.  It is intended for a separately gated experiment after a layout and
the analytic seam-graph harmonizer have both been frozen.

The user's local-averaging intuition is exposed as an edge-aware 5x5 tile-grid
prior.  The neural model does *not* synthesize a replacement image: it predicts
only a small, low-resolution additive field plus an even smaller seam-local
field.  Zero initialization makes an untrained checkpoint exactly the identity.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from puzzle_denoise_v2.model import LayerNorm2d, SimpleGate
from puzzle_denoise_v2.tiles import GRID, IMAGE_SIZE, TILE


CONTEXT_FEATURE_CHANNELS = 13


def _validate_image(image: torch.Tensor, *, name: str) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3 or tuple(image.shape[-2:]) != (
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise ValueError(f"{name} must be Bx3x{IMAGE_SIZE}x{IMAGE_SIZE}")
    if not torch.is_floating_point(image) or not torch.isfinite(image).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    if bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError(f"{name} must lie in [0, 1]")
    return image


def _validate_confidence_grid(
    confidence: torch.Tensor,
    *,
    batch: int,
    name: str,
) -> torch.Tensor:
    if confidence.shape != (batch, 1, GRID, GRID):
        raise ValueError(f"{name} must be Bx1x{GRID}x{GRID}")
    if not torch.is_floating_point(confidence) or not torch.isfinite(confidence).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    if bool((confidence < 0).any()) or bool((confidence > 1).any()):
        raise ValueError(f"{name} must lie in [0, 1]")
    return confidence


def tile_mean_grid(image: torch.Tensor, *, centre_crop: int = 0) -> torch.Tensor:
    """Return exact Bx3x24x24 tile means without mixing tile boundaries."""

    image = _validate_image(image, name="image")
    if centre_crop < 0 or 2 * centre_crop >= TILE:
        raise ValueError("centre_crop must leave at least one pixel per tile")
    batch, channels = image.shape[:2]
    tiles = image.reshape(batch, channels, GRID, TILE, GRID, TILE)
    if centre_crop:
        tiles = tiles[
            :,
            :,
            :,
            centre_crop : TILE - centre_crop,
            :,
            centre_crop : TILE - centre_crop,
        ]
    return tiles.mean(dim=(3, 5))


def broadcast_tile_grid(grid: torch.Tensor) -> torch.Tensor:
    """Broadcast a BxCx24x24 field piecewise-constantly to 480x480."""

    if grid.ndim != 4 or tuple(grid.shape[-2:]) != (GRID, GRID):
        raise ValueError(f"grid must be BxCx{GRID}x{GRID}")
    if not torch.is_floating_point(grid) or not torch.isfinite(grid).all():
        raise ValueError("grid must contain finite floating-point values")
    return grid.repeat_interleave(TILE, dim=2).repeat_interleave(TILE, dim=3)


def bilateral_tile_consensus_residual(
    image: torch.Tensor,
    *,
    radius: int = 2,
    sigma_spatial: float = 1.5,
    sigma_colour: float = 30.0 / 255.0,
    centre_crop: int = 4,
) -> torch.Tensor:
    """Return edge-aware local consensus minus each tile's observed mean.

    ``radius=2`` is the deployable 5x5-tile interpretation of local averaging.
    Unlike the non-deployable repeated-observation LLN ceiling, this function
    has one observation and therefore uses colour similarity to avoid averaging
    across semantic edges.
    """

    image = _validate_image(image, name="image")
    if radius <= 0 or radius >= GRID:
        raise ValueError("radius must be in [1, 23]")
    if sigma_spatial <= 0 or sigma_colour <= 0:
        raise ValueError("bilateral sigmas must be positive")
    means = tile_mean_grid(image)
    guide = tile_mean_grid(image, centre_crop=centre_crop)
    pad = (radius, radius, radius, radius)
    means_padded = F.pad(means, pad, mode="replicate")
    guide_padded = F.pad(guide, pad, mode="replicate")
    numerator = torch.zeros_like(means)
    denominator = torch.zeros_like(means[:, :1])
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            row = radius + dy
            column = radius + dx
            shifted_mean = means_padded[:, :, row : row + GRID, column : column + GRID]
            shifted_guide = guide_padded[:, :, row : row + GRID, column : column + GRID]
            colour_sq = (guide - shifted_guide).square().mean(dim=1, keepdim=True)
            spatial_sq = float(dx * dx + dy * dy)
            weight = torch.exp(
                -0.5 * spatial_sq / float(sigma_spatial**2)
                -0.5 * colour_sq / float(sigma_colour**2)
            )
            numerator = numerator + weight * shifted_mean
            denominator = denominator + weight
    consensus = numerator / denominator.clamp_min(torch.finfo(image.dtype).eps)
    return consensus - means


def internal_seam_mask(
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    band: int = 2,
) -> torch.Tensor:
    """Return a 1x1x480x480 mask around internal predicted-layout seams."""

    if band <= 0 or band > TILE // 2:
        raise ValueError("band must be in [1, 10]")
    line = torch.zeros(IMAGE_SIZE, device=device, dtype=dtype)
    for boundary in range(TILE, IMAGE_SIZE, TILE):
        line[boundary - band : boundary + band] = 1
    return torch.maximum(line[:, None], line[None, :])[None, None]


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    """Return target-blind RGB finite-difference magnitude as Bx1xHxW."""

    image = _validate_image(image, name="image")
    dx = F.pad(image[:, :, :, 1:] - image[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(image[:, :, 1:, :] - image[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt((dx.square() + dy.square()).mean(dim=1, keepdim=True) + 1e-12)


def build_context_features(
    harmonized_image: torch.Tensor,
    preanalytic_image: torch.Tensor,
    seam_confidence_grid: torch.Tensor,
    layout_confidence_grid: torch.Tensor,
    *,
    radius: int = 2,
    seam_band: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build target-blind channels, correction gate, and seam mask.

    Confidence fields are explicit inputs because correct-layout ceilings use
    all ones while a frozen real layout must use its already-computed solver
    confidences.  This prevents silently pretending that uncertain placements
    are trustworthy.
    """

    harmonized_image = _validate_image(harmonized_image, name="harmonized_image")
    preanalytic_image = _validate_image(preanalytic_image, name="preanalytic_image")
    if preanalytic_image.shape != harmonized_image.shape:
        raise ValueError("harmonized_image and preanalytic_image must have equal shapes")
    batch = harmonized_image.shape[0]
    seam_confidence_grid = _validate_confidence_grid(
        seam_confidence_grid, batch=batch, name="seam_confidence_grid"
    )
    layout_confidence_grid = _validate_confidence_grid(
        layout_confidence_grid, batch=batch, name="layout_confidence_grid"
    )

    consensus_grid = bilateral_tile_consensus_residual(
        harmonized_image,
        radius=radius,
    )
    consensus = broadcast_tile_grid(consensus_grid)
    seam = internal_seam_mask(
        device=harmonized_image.device,
        dtype=harmonized_image.dtype,
        band=seam_band,
    ).expand(batch, -1, -1, -1)
    edge = (gradient_magnitude(harmonized_image) / 0.25).clamp(0.0, 1.0)
    seam_confidence = broadcast_tile_grid(seam_confidence_grid)
    layout_confidence = broadcast_tile_grid(layout_confidence_grid)
    correction_gate = seam_confidence * layout_confidence
    features = torch.cat(
        (
            harmonized_image.mul(2.0).sub(1.0),
            (harmonized_image - preanalytic_image).mul(4.0).clamp(-1.0, 1.0),
            consensus.mul(4.0).clamp(-1.0, 1.0),
            seam,
            edge,
            seam_confidence,
            layout_confidence,
        ),
        dim=1,
    )
    if features.shape[1] != CONTEXT_FEATURE_CHANNELS:
        raise AssertionError("context channel contract changed")
    return features, correction_gate, seam


class ContextNAFBlock(nn.Module):
    """Small NAF-style block for quarter-resolution context features."""

    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        expanded = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.in_conv = nn.Conv2d(channels, expanded, 1)
        self.depthwise = nn.Conv2d(
            expanded,
            expanded,
            3,
            padding=dilation,
            dilation=dilation,
            groups=expanded,
            padding_mode="reflect",
        )
        self.gate1 = SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1)
        )
        self.out_conv = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, expanded, 1)
        self.gate2 = SimpleGate()
        self.ffn_out = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gate1(self.depthwise(self.in_conv(self.norm1(x))))
        y = y * self.channel_attention(y)
        x = x + self.out_conv(y) * self.beta
        y = self.ffn_out(self.gate2(self.ffn_in(self.norm2(x))))
        return x + y * self.gamma


class ContextualResidualNAF(nn.Module):
    """Zero-init bounded residual refiner that cannot alter tile geometry.

    The heads operate at quarter resolution.  Their bilinearly upsampled
    additive fields cannot directly copy or average high-frequency image
    texture.  The total per-channel change is bounded by the sum of the base
    and seam limits, and an external confidence gate can force exact identity.
    """

    def __init__(
        self,
        *,
        width: int = 32,
        blocks: int = 8,
        downsample: int = 4,
        base_limit_rgb: float = 6.0 / 255.0,
        seam_limit_rgb: float = 2.0 / 255.0,
    ) -> None:
        super().__init__()
        if width <= 0 or blocks <= 0:
            raise ValueError("width and blocks must be positive")
        if downsample not in {2, 4, 8}:
            raise ValueError("downsample must be one of 2, 4, or 8")
        if not 0 < base_limit_rgb <= 16.0 / 255.0:
            raise ValueError("base_limit_rgb must be in (0, 16/255]")
        if not 0 < seam_limit_rgb <= 8.0 / 255.0:
            raise ValueError("seam_limit_rgb must be in (0, 8/255]")
        self.downsample = downsample
        self.base_limit_rgb = float(base_limit_rgb)
        self.seam_limit_rgb = float(seam_limit_rgb)
        kernel = 2 * downsample - 1
        self.stem = nn.Conv2d(
            CONTEXT_FEATURE_CHANNELS,
            width,
            kernel,
            stride=downsample,
            padding=kernel // 2,
            padding_mode="reflect",
        )
        dilations = (1, 2, 3, 1)
        self.body = nn.Sequential(
            *[
                ContextNAFBlock(width, dilation=dilations[index % len(dilations)])
                for index in range(blocks)
            ]
        )
        self.base_tail = nn.Conv2d(width, 3, 3, padding=1, padding_mode="reflect")
        self.seam_tail = nn.Conv2d(width, 3, 3, padding=1, padding_mode="reflect")
        for tail in (self.base_tail, self.seam_tail):
            nn.init.zeros_(tail.weight)
            nn.init.zeros_(tail.bias)

    def forward(
        self,
        harmonized_image: torch.Tensor,
        features: torch.Tensor,
        correction_gate: torch.Tensor,
        seam_mask: torch.Tensor,
    ) -> torch.Tensor:
        harmonized_image = _validate_image(
            harmonized_image, name="harmonized_image"
        )
        batch, _, height, width = harmonized_image.shape
        if features.shape != (batch, CONTEXT_FEATURE_CHANNELS, height, width):
            raise ValueError("features do not match the contextual channel contract")
        if correction_gate.shape != (batch, 1, height, width):
            raise ValueError("correction_gate must be Bx1x480x480")
        if seam_mask.shape != (batch, 1, height, width):
            raise ValueError("seam_mask must be Bx1x480x480")
        for value, name in (
            (features, "features"),
            (correction_gate, "correction_gate"),
            (seam_mask, "seam_mask"),
        ):
            if not torch.is_floating_point(value) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite floating-point values")
        if bool((correction_gate < 0).any()) or bool((correction_gate > 1).any()):
            raise ValueError("correction_gate must lie in [0, 1]")
        if bool((seam_mask < 0).any()) or bool((seam_mask > 1).any()):
            raise ValueError("seam_mask must lie in [0, 1]")

        context = self.body(self.stem(features))
        base = F.interpolate(
            self.base_tail(context),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).tanh() * self.base_limit_rgb
        seam = F.interpolate(
            self.seam_tail(context),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).tanh() * self.seam_limit_rgb
        residual = correction_gate * (base + seam_mask * seam)
        return (harmonized_image + residual).clamp(0.0, 1.0)


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def maximum_rgb_change(model: ContextualResidualNAF) -> float:
    """Return the hard per-channel correction bound in normalized RGB."""

    return math.fsum((model.base_limit_rgb, model.seam_limit_rgb))


__all__ = [
    "CONTEXT_FEATURE_CHANNELS",
    "ContextualResidualNAF",
    "bilateral_tile_consensus_residual",
    "broadcast_tile_grid",
    "build_context_features",
    "gradient_magnitude",
    "internal_seam_mask",
    "maximum_rgb_change",
    "model_parameter_count",
    "tile_mean_grid",
]
