"""Bounded one-to-one DualNAF tile residuals for a frozen puzzle layout.

The module deliberately has no data-discovery or target-loading code.  It only
combines an original upright tile with the frozen renderer output for that same
tile, then applies the already established target-blind postassembly tail.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.postassembly_harmonizer import (
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_SEED,
    IMAGE_SIZE,
    TILE_COUNT,
    TILE_SIZE,
    assemble_tiles,
)

ARM_ALPHAS: Mapping[str, float] = {
    "baseline_alpha_0": 0.0,
    "dualnaf_residual_alpha_0_125": 0.125,
    "dualnaf_residual_alpha_0_25": 0.25,
    "dualnaf_residual_alpha_0_375": 0.375,
    "dualnaf_residual_alpha_0_5": 0.5,
}
CONTROL_ARM = "baseline_alpha_0"
CANDIDATE_ARMS = tuple(name for name in ARM_ALPHAS if name != CONTROL_ARM)
TAIL_NLM_H = 20
BOOTSTRAP_REPLICATES = 20_000


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 tiles with shape {expected}, got {value.dtype} {value.shape}"
        )
    return np.ascontiguousarray(value)


def image_digest(image: np.ndarray) -> str:
    """Hash one strict prediction's uncompressed RGB pixels."""

    value = np.asarray(image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB image with shape {expected}")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def blend_same_index_tiles(
    original_tiles: np.ndarray,
    rendered_tiles: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return the frozen convex blend without changing tile identity or geometry."""

    original = _validate_tiles(original_tiles)
    rendered = _validate_tiles(rendered_tiles)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 0.5:
        raise ValueError("alpha must be finite and in the preregistered [0, 0.5] range")
    if alpha == 0:
        return original.copy()
    mixed = np.rint(
        original.astype(np.float32) * (1.0 - alpha)
        + rendered.astype(np.float32) * alpha
    )
    return np.ascontiguousarray(mixed.clip(0, 255).astype(np.uint8))


def apply_frozen_tail(ordered_tiles: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply RGB offsets, bounded luma, and exactly one colored NLM h20 pass."""

    ordered = _validate_tiles(ordered_tiles)
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered, SeamGraphConfig())
    rgb_tiles = apply_rgb_offsets(ordered, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(rgb_tiles, LuminanceGainConfig())
    harmonized = assemble_tiles(apply_luminance_gains(rgb_tiles, gains))
    final = apply_nlm_color(harmonized, h=TAIL_NLM_H).image
    return final, {
        "rgb_seam_offsets": rgb_diagnostics,
        "bounded_luminance_gains": luma_diagnostics,
    }


def render_arm_roster(
    ordered_original: np.ndarray,
    ordered_rendered: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render exactly the five preregistered arms from a shared strict layout."""

    original = _validate_tiles(ordered_original)
    rendered = _validate_tiles(ordered_rendered)
    predictions: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for name, alpha in ARM_ALPHAS.items():
        blended = blend_same_index_tiles(original, rendered, alpha)
        prediction, tail_diagnostics = apply_frozen_tail(blended)
        delta = np.abs(blended.astype(np.int16) - original.astype(np.int16))
        predictions[name] = prediction
        diagnostics[name] = {
            "alpha": alpha,
            "blend_mean_abs_change": float(delta.mean()),
            "blend_q99_abs_change": float(np.quantile(delta, 0.99)),
            "blend_maximum_abs_change": int(delta.max()),
            "tail": tail_diagnostics,
        }
    if tuple(predictions) != tuple(ARM_ALPHAS):
        raise RuntimeError("preregistered arm order drifted")
    return predictions, diagnostics


def image_structure_diagnostics(image: np.ndarray) -> dict[str, float]:
    """Return target-free edge energy and clipping diagnostics for manual gating."""

    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError("structure diagnostics require one strict uint8 RGB prediction")
    gray = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY).astype(np.float32)
    horizontal = np.abs(np.diff(gray, axis=1)).mean()
    vertical = np.abs(np.diff(gray, axis=0)).mean()
    return {
        "gradient_energy": float((horizontal + vertical) / 2.0),
        "clipped_fraction": float(np.mean((value == 0) | (value == 255))),
    }


def paired_bootstrap_ci(values: Sequence[float]) -> tuple[float, float]:
    """Compute the frozen 20k paired bootstrap interval for mean differences."""

    difference = np.asarray(values, dtype=np.float64)
    if difference.ndim != 1 or not len(difference) or not np.isfinite(difference).all():
        raise ValueError("paired differences must be a non-empty finite vector")
    rng = np.random.default_rng(EXPERIMENT_SUBSET_SEED)
    samples: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        count = min(remaining, 4096)
        indices = rng.integers(0, len(difference), size=(count, len(difference)))
        samples.append(difference[indices].mean(axis=1))
        remaining -= count
    lower, upper = np.quantile(np.concatenate(samples), (0.025, 0.975))
    return float(lower), float(upper)


def choose_winner(mean_ssim: Mapping[str, float]) -> str:
    """Choose highest candidate mean SSIM, breaking exact ties toward lower alpha."""

    if set(mean_ssim) != set(ARM_ALPHAS):
        raise ValueError("mean SSIM roster differs from preregistration")
    return max(CANDIDATE_ARMS, key=lambda name: (float(mean_ssim[name]), -ARM_ALPHAS[name]))
