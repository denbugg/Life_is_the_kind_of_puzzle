"""Fixed B-only DRUNet50 h28-safe/h50-flat protected restoration.

This is the efficient broad-measurement renderer justified by the formal
cycle-2 train reproduction.  It has no arm selection: one independently
tilewise DRUNet50 canvas is assembled, three NLM images are computed once and
independently, and the fixed t60 mask blends h28-protected with h50-flat pixels.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn

from aiijc_puzzle.drunet_goal_cycle2 import (
    DIRECT_SIGMA,
    MODEL_BATCH_SIZE,
    blend_h28_safe_h50_flat_t60,
    tile_flatness_counts,
)
from aiijc_puzzle.edge_protected_nlm import colored_nlm
from aiijc_puzzle.nlm_luma_chroma import structure_diagnostics
from aiijc_puzzle.pretrained_tile_denoiser import render_drunet_tiles
from aiijc_puzzle.protocol import TILE_COUNT, TILE_SIZE, assemble_tiles

REFERENCE_DRUNET50_H28 = "R_drunet50_h28_safety_reference"
CANDIDATE_DRUNET50_PROTECTED = "B_drunet50_protected_h28_h50_t60"


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 tiles {expected}, got {value.dtype} {value.shape}")
    return np.ascontiguousarray(value)


def image_digest(image: np.ndarray) -> str:
    value = np.ascontiguousarray(image)
    return hashlib.sha256(value.tobytes()).hexdigest()


def render_sigma50_protected(
    model: nn.Module,
    harmonized_tiles: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the fixed h28 safety reference and fixed protected candidate."""

    tiles = _validate_tiles(harmonized_tiles)
    restored_tiles, drunet = render_drunet_tiles(
        model,
        tiles,
        sigma_255=DIRECT_SIGMA,
        device=device,
        batch_size=MODEL_BATCH_SIZE,
    )
    canvas = assemble_tiles(restored_tiles)
    h20 = colored_nlm(canvas, 20)
    h28 = colored_nlm(canvas, 28)
    h50 = colored_nlm(canvas, 50)
    candidate, mask = blend_h28_safe_h50_flat_t60(h20, h28, h50)
    if np.array_equal(candidate, h28):
        raise RuntimeError("fixed protected candidate unexpectedly equals h28 reference")
    return (
        h28,
        candidate,
        {
            "drunet": drunet.as_dict(),
            "neural_intermediate_pixel_sha256": {
                "drunet50_canvas": image_digest(canvas),
                "drunet50_then_h20_mask_source": image_digest(h20),
                "drunet50_then_h28_reference_and_safe": image_digest(h28),
                "drunet50_then_h50_flat": image_digest(h50),
            },
            "mask": mask,
            "structure": {
                REFERENCE_DRUNET50_H28: structure_diagnostics(h28),
                CANDIDATE_DRUNET50_PROTECTED: structure_diagnostics(candidate),
            },
            "tile_flatness": {
                REFERENCE_DRUNET50_H28: tile_flatness_counts(h28),
                CANDIDATE_DRUNET50_PROTECTED: tile_flatness_counts(candidate),
            },
        },
    )


__all__ = [
    "CANDIDATE_DRUNET50_PROTECTED",
    "REFERENCE_DRUNET50_H28",
    "image_digest",
    "render_sigma50_protected",
]
