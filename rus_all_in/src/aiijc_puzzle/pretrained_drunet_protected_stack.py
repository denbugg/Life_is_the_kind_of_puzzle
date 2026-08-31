"""Fixed legal DRUNet40 plus v1 edge-protected NLM composition.

The neural model restores each upright tile independently.  After exact
reassembly, three NLM images are computed independently from that one DRUNet
canvas.  The established v1 threshold-40 mask is derived from DRUNet+h20 and
uses DRUNet+h28 around protected content/grid edges and DRUNet+h40 elsewhere.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn

from aiijc_puzzle.edge_protected_nlm import colored_nlm, protected_masks, validate_rgb
from aiijc_puzzle.nlm_luma_chroma import structure_diagnostics
from aiijc_puzzle.pretrained_tile_denoiser import render_drunet_tiles
from aiijc_puzzle.protocol import TILE_COUNT, TILE_SIZE, assemble_tiles

ARM_ORIGINAL_H20 = "A_original_nlm_h20"
ARM_ORIGINAL_H28 = "B_original_nlm_h28"
ARM_DRUNET_H28 = "C_drunet_sigma40_then_nlm_h28"
ARM_COMBINED = "D_drunet_sigma40_protected_h28_h40_t40"
ARM_NAMES = (ARM_ORIGINAL_H20, ARM_ORIGINAL_H28, ARM_DRUNET_H28, ARM_COMBINED)
SOBEL_THRESHOLD = 40.0
DRUNET_SIGMA = 40.0
MODEL_BATCH_SIZE = 144


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(validate_rgb(image).tobytes()).hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{','.join(map(str, array.shape))}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 tiles {expected}, got {value.dtype} {value.shape}")
    return np.ascontiguousarray(value)


def blend_v1_t40_from_h20_mask(
    mask_source_h20: np.ndarray,
    safe_h28: np.ndarray,
    flat_h40: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the exact established t40 mask source with different safe/flat images."""

    mask_source = validate_rgb(mask_source_h20)
    safe = validate_rgb(safe_h28)
    aggressive = validate_rgb(flat_h40)
    binary, soft, protected_fraction = protected_masks(
        mask_source,
        sobel_threshold=SOBEL_THRESHOLD,
    )
    mixed = np.rint(
        soft[..., None] * safe.astype(np.float32)
        + (1.0 - soft[..., None]) * aggressive.astype(np.float32)
    )
    output = np.ascontiguousarray(mixed.clip(0, 255).astype(np.uint8))
    return output, {
        "mask_source": "independent DRUNet sigma40 canvas then one colored NLM h20",
        "safe_source": "same DRUNet canvas then independent one colored NLM h28",
        "flat_source": "same DRUNet canvas then independent one colored NLM h40",
        "sobel_threshold": SOBEL_THRESHOLD,
        "binary_dilated_protected_fraction": protected_fraction,
        "mean_soft_safe_weight": float(soft.mean()),
        "minimum_soft_safe_weight": float(soft.min()),
        "maximum_soft_safe_weight": float(soft.max()),
        "binary_mask_sha256": array_digest(binary),
        "soft_mask_sha256": array_digest(soft),
    }


def render_combined_arms(
    model: nn.Module,
    harmonized_tiles: np.ndarray,
    *,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render exactly the four fixed controls/candidates from one strict layout."""

    tiles = _validate_tiles(harmonized_tiles)
    original_canvas = assemble_tiles(tiles)
    original_h20 = colored_nlm(original_canvas, 20)
    original_h28 = colored_nlm(original_canvas, 28)
    restored_tiles, neural_diagnostics = render_drunet_tiles(
        model,
        tiles,
        sigma_255=DRUNET_SIGMA,
        device=device,
        batch_size=MODEL_BATCH_SIZE,
    )
    neural_canvas = assemble_tiles(restored_tiles)
    neural_h20 = colored_nlm(neural_canvas, 20)
    neural_h28 = colored_nlm(neural_canvas, 28)
    neural_h40 = colored_nlm(neural_canvas, 40)
    combined, mask_diagnostics = blend_v1_t40_from_h20_mask(
        neural_h20,
        neural_h28,
        neural_h40,
    )
    predictions = {
        ARM_ORIGINAL_H20: original_h20,
        ARM_ORIGINAL_H28: original_h28,
        ARM_DRUNET_H28: neural_h28,
        ARM_COMBINED: combined,
    }
    return predictions, {
        "drunet": neural_diagnostics.as_dict(),
        "neural_intermediate_pixel_sha256": {
            "drunet_canvas": image_digest(neural_canvas),
            "drunet_then_h20_mask_source": image_digest(neural_h20),
            "drunet_then_h28_safe": image_digest(neural_h28),
            "drunet_then_h40_flat": image_digest(neural_h40),
        },
        "mask": mask_diagnostics,
        "structure": {
            name: structure_diagnostics(image) for name, image in predictions.items()
        },
    }


__all__ = [
    "ARM_COMBINED",
    "ARM_DRUNET_H28",
    "ARM_NAMES",
    "ARM_ORIGINAL_H20",
    "ARM_ORIGINAL_H28",
    "DRUNET_SIGMA",
    "MODEL_BATCH_SIZE",
    "SOBEL_THRESHOLD",
    "array_digest",
    "blend_v1_t40_from_h20_mask",
    "image_digest",
    "render_combined_arms",
]
