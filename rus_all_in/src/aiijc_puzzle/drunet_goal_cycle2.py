"""Bounded legal DRUNet restoration candidates for goal cycle 2.

Every neural call operates on an exact roster of 576 upright 20x20 tiles.
The shared layout and harmonization are supplied by the caller.  This module
only compares two new fixed restorers with the frozen DRUNet40/t40 reference:

* tilewise DRUNet50 followed by independent h20/h28/h50 canvases and a t60
  h28-safe/h50-flat blend;
* the DRUNet50 h28 canvas followed by tilewise DRUNet30 and an exact half
  blend with that same DRUNet50 h28 canvas.

The optional composition is a fixed half blend of those two candidates.  It is
rendered target-blind and may only be selected by the preregistered train rule.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import torch
from torch import nn

from aiijc_puzzle.edge_protected_nlm import colored_nlm, protected_masks
from aiijc_puzzle.nlm_luma_chroma import structure_diagnostics
from aiijc_puzzle.pretrained_drunet_protected_stack import (
    ARM_COMBINED as FROZEN_STACK_ARM,
)
from aiijc_puzzle.pretrained_drunet_protected_stack import render_combined_arms
from aiijc_puzzle.pretrained_tile_denoiser import (
    blend_uint8_fraction,
    render_drunet_tiles,
)
from aiijc_puzzle.protocol import IMAGE_SIZE, TILE_COUNT, TILE_SIZE, assemble_tiles
from aiijc_puzzle.protocol import split_tiles as split_tiles

REFERENCE_CURRENT_D = "A_current_drunet40_protected_h28_h40_t40"
CANDIDATE_SIGMA50 = "B_drunet50_protected_h28_h50_t60"
CANDIDATE_POST_H28 = "C_drunet50_h28_tilewise_drunet30_alpha_0_5"
CANDIDATE_COMBINATION = "D_half_B_half_C"
ARM_NAMES = (
    REFERENCE_CURRENT_D,
    CANDIDATE_SIGMA50,
    CANDIDATE_POST_H28,
    CANDIDATE_COMBINATION,
)

DIRECT_SIGMA = 50.0
POST_H28_SIGMA = 30.0
SOBEL_THRESHOLD = 60.0
MODEL_BATCH_SIZE = 144
HALF = Fraction(1, 2)


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 tiles {expected}, got {value.dtype} {value.shape}")
    return np.ascontiguousarray(value)


def _validate_board(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB board {expected}, got {value.dtype} {value.shape}")
    return np.ascontiguousarray(value)


def blend_h28_safe_h50_flat_t60(
    mask_source_h20: np.ndarray,
    safe_h28: np.ndarray,
    flat_h50: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend independent h28/h50 images using the fixed t60 protected mask."""

    mask_source = _validate_board(mask_source_h20)
    safe = _validate_board(safe_h28)
    flat = _validate_board(flat_h50)
    binary, soft, protected_fraction = protected_masks(
        mask_source,
        sobel_threshold=SOBEL_THRESHOLD,
    )
    mixed = np.rint(
        soft[..., None] * safe.astype(np.float32)
        + (1.0 - soft[..., None]) * flat.astype(np.float32)
    )
    output = np.ascontiguousarray(mixed.clip(0, 255).astype(np.uint8))
    return output, {
        "mask_source": "DRUNet50 canvas then one independent colored NLM h20",
        "safe_source": "same DRUNet50 canvas then one independent colored NLM h28",
        "flat_source": "same DRUNet50 canvas then one independent colored NLM h50",
        "sobel_threshold": SOBEL_THRESHOLD,
        "binary_dilated_protected_fraction": float(protected_fraction),
        "mean_soft_safe_weight": float(soft.mean()),
        "minimum_soft_safe_weight": float(soft.min()),
        "maximum_soft_safe_weight": float(soft.max()),
        "binary_mask_sha256": _array_digest(binary),
        "soft_mask_sha256": _array_digest(soft),
    }


def _array_digest(value: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{','.join(map(str, array.shape))}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def tile_flatness_counts(image: np.ndarray) -> dict[str, int]:
    """Count spatially constant RGB tiles and legacy near-flat std<2 tiles."""

    tiles = split_tiles(_validate_board(image))
    spatial_range = tiles.max(axis=(1, 2)) - tiles.min(axis=(1, 2))
    exact_constant = np.all(spatial_range == 0, axis=1)
    standard_deviation = tiles.astype(np.float64).std(axis=(1, 2, 3))
    return {
        "exact_spatially_constant_rgb_tiles": int(exact_constant.sum()),
        "near_flat_tiles_global_std_lt_2": int(np.sum(standard_deviation < 2.0)),
        "near_flat_tiles_global_std_lt_4": int(np.sum(standard_deviation < 4.0)),
    }


def render_goal_cycle2_arms(
    model: nn.Module,
    harmonized_tiles: np.ndarray,
    *,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render the fixed reference, two candidates, and one fixed composition."""

    tiles = _validate_tiles(harmonized_tiles)

    frozen_predictions, frozen_diagnostics = render_combined_arms(
        model,
        tiles,
        device=device,
    )
    current_d = frozen_predictions[FROZEN_STACK_ARM]

    direct_tiles, direct_diagnostics = render_drunet_tiles(
        model,
        tiles,
        sigma_255=DIRECT_SIGMA,
        device=device,
        batch_size=MODEL_BATCH_SIZE,
    )
    direct_canvas = assemble_tiles(direct_tiles)
    direct_h20 = colored_nlm(direct_canvas, 20)
    direct_h28 = colored_nlm(direct_canvas, 28)
    direct_h50 = colored_nlm(direct_canvas, 50)
    sigma50, sigma50_mask = blend_h28_safe_h50_flat_t60(
        direct_h20,
        direct_h28,
        direct_h50,
    )

    post_tiles, post_diagnostics = render_drunet_tiles(
        model,
        split_tiles(direct_h28),
        sigma_255=POST_H28_SIGMA,
        device=device,
        batch_size=MODEL_BATCH_SIZE,
    )
    post_canvas = assemble_tiles(post_tiles)
    post_h28 = blend_uint8_fraction(direct_h28, post_canvas, HALF)
    combination = blend_uint8_fraction(sigma50, post_h28, HALF)

    predictions = {
        REFERENCE_CURRENT_D: current_d,
        CANDIDATE_SIGMA50: sigma50,
        CANDIDATE_POST_H28: post_h28,
        CANDIDATE_COMBINATION: combination,
    }
    if tuple(predictions) != ARM_NAMES:
        raise RuntimeError("goal-cycle-2 arm order drifted")
    if any(
        value.shape != current_d.shape or value.dtype != np.uint8 for value in predictions.values()
    ):
        raise RuntimeError("goal-cycle-2 renderer changed output geometry or dtype")

    return predictions, {
        "frozen_reference": frozen_diagnostics,
        "drunet50": direct_diagnostics.as_dict(),
        "drunet50_mask": sigma50_mask,
        "post_h28_drunet30": post_diagnostics.as_dict(),
        "fixed_composition": "exact half-up uint8 blend of B and C",
        "structure": {name: structure_diagnostics(image) for name, image in predictions.items()},
        "tile_flatness": {name: tile_flatness_counts(image) for name, image in predictions.items()},
    }


__all__ = [
    "ARM_NAMES",
    "CANDIDATE_COMBINATION",
    "CANDIDATE_POST_H28",
    "CANDIDATE_SIGMA50",
    "DIRECT_SIGMA",
    "MODEL_BATCH_SIZE",
    "POST_H28_SIGMA",
    "REFERENCE_CURRENT_D",
    "SOBEL_THRESHOLD",
    "blend_h28_safe_h50_flat_t60",
    "render_goal_cycle2_arms",
    "tile_flatness_counts",
]
