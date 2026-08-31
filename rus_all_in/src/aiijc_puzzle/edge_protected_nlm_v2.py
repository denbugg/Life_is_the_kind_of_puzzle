"""Frozen h28-safe / h40-flat NLM blend using the v1 h20-derived t40 mask."""

from __future__ import annotations

import numpy as np

from aiijc_puzzle.edge_protected_nlm import protected_masks, validate_rgb

SOBEL_THRESHOLD = 40.0


def blend_h28safe_h40flat(
    mask_source_h20: np.ndarray,
    safe_h28: np.ndarray,
    aggressive_h40: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Apply the only preregistered v2 blend without routing or target access."""

    mask_source = validate_rgb(mask_source_h20)
    safe = validate_rgb(safe_h28)
    aggressive = validate_rgb(aggressive_h40)
    dilated, soft, protected_fraction = protected_masks(
        mask_source,
        sobel_threshold=SOBEL_THRESHOLD,
    )
    mixed = np.rint(
        soft[..., None] * safe.astype(np.float32)
        + (1.0 - soft[..., None]) * aggressive.astype(np.float32)
    )
    output = np.ascontiguousarray(mixed.clip(0, 255).astype(np.uint8))
    return (
        output,
        np.ascontiguousarray(dilated),
        np.ascontiguousarray(soft),
        {
            "binary_dilated_protected_fraction": protected_fraction,
            "mean_soft_h28_weight": float(soft.mean()),
            "minimum_soft_h28_weight": float(soft.min()),
            "maximum_soft_h28_weight": float(soft.max()),
        },
    )


__all__ = ["SOBEL_THRESHOLD", "blend_h28safe_h40flat"]
