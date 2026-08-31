"""Legal, board-local residual restoration after the frozen h20 tail.

The model in this module never changes the inferred tile permutation and never
uses neighbouring tiles.  It receives the 20x20 harmonized tile immediately
before NLM and the same 20x20 tile immediately after full-canvas NLM h20.  A
small shared NAF-style network predicts a bounded residual around the h20 tile.

Clean targets are used only by the training runner.  Inference functions here
accept one dirty board and a frozen checkpoint; there is no target, atlas,
reference image, or cross-board state in their API.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.frozen_final_evaluator import _validate_method_configs
from aiijc_puzzle.legacy_upgrade import directional_scores, solve_buddies
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.postassembly_harmonizer import (
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.protocol import assemble_tiles, split_tiles


@dataclass(frozen=True)
class AfterH20ModelConfig:
    """Frozen architecture for the tile-local residual model."""

    width: int = 32
    blocks: int = 8
    residual_limit: float = 64.0 / 255.0

    def validate(self) -> None:
        if self.width < 8 or self.width % 2:
            raise ValueError("width must be an even integer >= 8")
        if self.blocks < 1:
            raise ValueError("blocks must be positive")
        if not 0.0 < self.residual_limit <= 1.0:
            raise ValueError("residual_limit must be in (0, 1]")


class SimpleGate(nn.Module):
    """Multiplicative gate used by NAFNet blocks."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return first * second


class NAFResidualBlock(nn.Module):
    """MPS-efficient NAF-style block without spatial downsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded = channels * 2
        self.norm = nn.GroupNorm(1, channels)
        self.in_project = nn.Conv2d(channels, expanded, 1)
        self.depthwise = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.gate = SimpleGate()
        self.out_project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        branch = self.depthwise(self.in_project(self.norm(value)))
        branch = self.gate(branch)
        return value + self.scale * self.out_project(branch)


class AfterH20TileRestorer(nn.Module):
    """Predict a bounded clean-image residual around a post-NLM h20 tile."""

    def __init__(self, config: AfterH20ModelConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = AfterH20ModelConfig()
        config.validate()
        self.config = config
        self.intro = nn.Conv2d(6, config.width, 3, padding=1)
        self.body = nn.Sequential(*(NAFResidualBlock(config.width) for _ in range(config.blocks)))
        self.ending = nn.Conv2d(config.width, 3, 3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, pre_h20: torch.Tensor, h20: torch.Tensor) -> torch.Tensor:
        if pre_h20.shape != h20.shape or pre_h20.ndim != 4 or pre_h20.shape[1] != 3:
            raise ValueError("pre_h20 and h20 must both have shape (batch, 3, H, W)")
        features = self.body(self.intro(torch.cat((pre_h20, h20), dim=1)))
        residual = self.config.residual_limit * torch.tanh(self.ending(features))
        return torch.clamp(h20 + residual, 0.0, 1.0)


def model_config_dict(model: AfterH20TileRestorer) -> dict[str, Any]:
    """Return JSON-compatible architecture metadata."""

    return asdict(model.config)


def harmonize_ordered_tiles(ordered_tiles: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the exact frozen RGB-offset and luminance-gain stages."""

    rgb_config, luma_config, _ = _validate_method_configs()
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered_tiles, rgb_config)
    rgb_tiles = apply_rgb_offsets(ordered_tiles, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(rgb_tiles, luma_config)
    corrected = apply_luminance_gains(rgb_tiles, gains)
    return corrected, {
        "rgb_seam_offsets": rgb_diagnostics,
        "bounded_luminance_gains": luma_diagnostics,
    }


def infer_frozen_inputs(dirty_image: np.ndarray, *, include_h28: bool = True) -> dict[str, Any]:
    """Infer layout and the pre/h20/h28 tensors from one dirty board only."""

    dirty_tiles = split_tiles(dirty_image)
    right, down = directional_scores(dirty_tiles, views=("bilateral",))["bilateral"]
    solved = solve_buddies(right, down, max_edges=96)
    layout = np.asarray(solved.layout, dtype=np.int32)
    raw = assemble_tiles(dirty_tiles[layout])
    audit = audit_raw_permutation(
        dirty_image,
        raw,
        layout,
        restoration_applied_after_audit=True,
    )
    if not audit.passed:
        raise RuntimeError(f"strict raw permutation audit failed: {audit.as_dict()}")
    pre_h20_tiles, harmonizer = harmonize_ordered_tiles(dirty_tiles[layout])
    pre_h20 = assemble_tiles(pre_h20_tiles)
    h20 = apply_nlm_color(pre_h20, h=20).image
    result = {
        "layout": layout,
        "raw": raw,
        "pre_h20": pre_h20,
        "h20": h20,
        "audit": audit.as_dict(),
        "harmonizer": harmonizer,
        "layout_objective": float(solved.objective),
    }
    if include_h28:
        result["h28"] = apply_nlm_color(pre_h20, h=28).image
    return result


def paired_clean_tiles(
    dirty_image: np.ndarray,
    clean_image: np.ndarray,
    predicted_layout: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover train-only clean identities in the predicted dirty layout.

    Returns clean tiles aligned position-for-position with ``predicted_layout``
    and the target-supervised confidence margin for each returned tile.
    """

    dirty_tiles = split_tiles(dirty_image)
    clean_tiles = split_tiles(clean_image)
    recovered = recover_layout(dirty_tiles, clean_tiles)
    clean_position_for_dirty = recovered.position_of_dirty
    target_positions = clean_position_for_dirty[np.asarray(predicted_layout, dtype=np.int64)]
    return (
        np.ascontiguousarray(clean_tiles[target_positions]),
        np.ascontiguousarray(recovered.margin_at_position[target_positions]),
        {
            "target_position_for_dirty_sha256": hashlib.sha256(
                clean_position_for_dirty.astype("<i4").tobytes()
            ).hexdigest(),
            "margin_quantiles": {
                str(q): float(np.quantile(recovered.margin_at_position, q))
                for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            },
        },
    )


@torch.inference_mode()
def restore_tiles(
    model: AfterH20TileRestorer,
    pre_h20_image: np.ndarray,
    h20_image: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 576,
) -> np.ndarray:
    """Run the shared model independently on upright 20x20 tiles."""

    pre_tiles = split_tiles(pre_h20_image)
    h20_tiles = split_tiles(h20_image)
    output: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(pre_tiles), batch_size):
        stop = min(start + batch_size, len(pre_tiles))
        pre = torch.from_numpy(pre_tiles[start:stop].copy()).permute(0, 3, 1, 2)
        post = torch.from_numpy(h20_tiles[start:stop].copy()).permute(0, 3, 1, 2)
        prediction = model(
            pre.to(device=device, dtype=torch.float32) / 255.0,
            post.to(device=device, dtype=torch.float32) / 255.0,
        )
        array = (
            prediction.mul(255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        output.append(array)
    return assemble_tiles(np.concatenate(output, axis=0))


def blend_around_h20(h20: np.ndarray, restored: np.ndarray, alpha: float) -> np.ndarray:
    """Convex uint8 blend whose alpha=0 endpoint is byte-exact h20."""

    if h20.shape != restored.shape or h20.dtype != np.uint8 or restored.dtype != np.uint8:
        raise ValueError("h20 and restored must be equal-shape uint8 images")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 0.0:
        return h20.copy()
    return np.clip(
        np.rint((1.0 - alpha) * h20.astype(np.float32) + alpha * restored.astype(np.float32)),
        0,
        255,
    ).astype(np.uint8)


__all__ = [
    "AfterH20ModelConfig",
    "AfterH20TileRestorer",
    "blend_around_h20",
    "harmonize_ordered_tiles",
    "infer_frozen_inputs",
    "model_config_dict",
    "paired_clean_tiles",
    "restore_tiles",
]
