"""Vectorized fixed-orientation classical seam scores on sparse candidates.

Every score is label-free and is evaluated only on the supplied candidate
graph.  Direction is the target tile relative to the source tile: U/D/L/R.

Depth definition
----------------
For depth ``d`` in ``{0, 1, 2}``, both compared traces are moved ``d`` pixels
inward from their proposed physical seam.  For example, RIGHT compares source
column ``W-1-d`` with target column ``d``; UP compares source row ``d`` with
target row ``H-1-d``.  Tiles are never rotated or reflected.

Families
--------
``rgb_ssd``
    Mean squared difference after the same scalar per-tile exposure
    normalization used by the neural seam ranker.
``lab_ssd``
    Mean squared difference in CIE Lab with fixed channel scales
    ``(100, 128, 128)``.
``mgc``
    Symmetric linear gradient-continuation residual.  Residual RGB vectors are
    weighted by a scene-only covariance of internal border gradients, yielding
    a regularized Mahalanobis Gradient Compatibility distance.

All returned values are compatibility scores (negative distance; larger is
better), shaped ``(variants, 4, N, K)``.  Invalid base candidate slots remain
``-inf``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from skimage.color import rgb2lab


UP: Final[int] = 0
DOWN: Final[int] = 1
LEFT: Final[int] = 2
RIGHT: Final[int] = 3
NUM_DIRECTIONS: Final[int] = 4
DEPTHS: Final[tuple[int, ...]] = (0, 1, 2)

# This is a scientific prior, not a metric-driven order.  One-pixel inset is
# tried first to suppress JPEG/border artifacts while retaining local content;
# MGC precedes perceptual Lab and plain RGB.  Depth two follows, then the raw
# outer boundary.  The evaluator selects the first qualifying entry only.
PREDECLARED_VARIANT_ORDER: Final[tuple[str, ...]] = (
    "mgc_d1",
    "lab_ssd_d1",
    "rgb_ssd_d1",
    "mgc_d2",
    "lab_ssd_d2",
    "rgb_ssd_d2",
    "mgc_d0",
    "lab_ssd_d0",
    "rgb_ssd_d0",
)
VARIANT_NAMES: Final[tuple[str, ...]] = PREDECLARED_VARIANT_ORDER


@dataclass(frozen=True)
class BorderLines:
    """Source/target boundary and one-pixel-inner traces for one direction."""

    source_boundary: np.ndarray
    source_inner: np.ndarray
    target_boundary: np.ndarray
    target_inner: np.ndarray


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    values = np.asarray(tiles)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"tiles must have shape (N,H,W,3), got {values.shape}")
    if values.shape[1] != values.shape[2]:
        raise ValueError("fixed puzzle tiles must be square")
    if values.shape[1] < max(DEPTHS) + 2:
        raise ValueError("tiles are too small for depth-2 gradient traces")
    if np.issubdtype(values.dtype, np.integer):
        if np.any(values < 0) or np.any(values > 255):
            raise ValueError("integer tiles must lie in [0,255]")
        return values.astype(np.float32) / 255.0
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("tiles must be uint8 or floating point")
    if not np.isfinite(values).all():
        raise ValueError("tiles must be finite")
    work = values.astype(np.float32, copy=False)
    if float(work.min()) < 0.0 or float(work.max()) > 1.0:
        raise ValueError("floating tiles must lie in [0,1]")
    return work


def _validate_candidates(
    candidates: np.ndarray,
    valid: np.ndarray,
    *,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_ids = np.asarray(candidates)
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != count:
        raise ValueError(f"candidates must have shape ({count},K)")
    if not np.issubdtype(candidate_ids.dtype, np.integer):
        raise TypeError("candidate ids must be integers")
    if np.any(candidate_ids < 0) or np.any(candidate_ids >= count):
        raise ValueError(f"candidate ids must lie in [0,{count})")
    mask = np.asarray(valid, dtype=bool)
    if mask.shape == candidate_ids.shape:
        mask = np.broadcast_to(mask[None], (NUM_DIRECTIONS, *mask.shape)).copy()
    if mask.shape != (NUM_DIRECTIONS, *candidate_ids.shape):
        raise ValueError("valid must have shape (N,K) or (4,N,K)")
    if not mask.any(axis=-1).all():
        raise ValueError("every directional row needs at least one valid candidate")
    return candidate_ids.astype(np.int64, copy=False), mask


def exposure_normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Normalize each tile by one RGB-wide mean and RMS, preserving chroma."""
    values = np.asarray(rgb, dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("rgb must have shape (N,H,W,3)")
    mean = values.mean(axis=(1, 2, 3), keepdims=True)
    rms = np.sqrt(np.square(values - mean).mean(axis=(1, 2, 3), keepdims=True) + 1.0e-5)
    return np.clip((values - mean) / rms, -5.0, 5.0).astype(np.float32)


def scaled_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] to Lab and apply fixed perceptual channel scales."""
    lab = rgb2lab(np.asarray(rgb, dtype=np.float32), channel_axis=-1).astype(np.float32)
    return lab / np.asarray((100.0, 128.0, 128.0), dtype=np.float32)


def directional_border_lines(
    features: np.ndarray,
    direction: int,
    depth: int,
) -> BorderLines:
    """Extract unrotated boundary/inner traces for every tile.

    The source traces face outward toward the proposed target.  Target traces
    face outward in the inverse direction, so ``boundary - inner`` is an
    outward gradient on both sides.
    """
    values = np.asarray(features)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("features must have shape (N,H,W,3)")
    if direction not in (UP, DOWN, LEFT, RIGHT):
        raise ValueError("direction must be U/D/L/R")
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}")
    height, width = values.shape[1:3]
    if min(height, width) <= depth + 1:
        raise ValueError("feature map is too small for requested depth")

    if direction == RIGHT:
        return BorderLines(
            values[:, :, width - 1 - depth, :],
            values[:, :, width - 2 - depth, :],
            values[:, :, depth, :],
            values[:, :, depth + 1, :],
        )
    if direction == LEFT:
        return BorderLines(
            values[:, :, depth, :],
            values[:, :, depth + 1, :],
            values[:, :, width - 1 - depth, :],
            values[:, :, width - 2 - depth, :],
        )
    if direction == DOWN:
        return BorderLines(
            values[:, height - 1 - depth, :, :],
            values[:, height - 2 - depth, :, :],
            values[:, depth, :, :],
            values[:, depth + 1, :, :],
        )
    return BorderLines(
        values[:, depth, :, :],
        values[:, depth + 1, :, :],
        values[:, height - 1 - depth, :, :],
        values[:, height - 2 - depth, :, :],
    )


def _ssd_candidate_scores(lines: BorderLines, candidates: np.ndarray) -> np.ndarray:
    source = lines.source_boundary[:, None, :, :]
    target = lines.target_boundary[candidates]
    return -np.square(source - target).mean(axis=(2, 3), dtype=np.float64).astype(np.float32)


def _gradient_precision(lines: BorderLines, *, ridge: float) -> np.ndarray:
    gradients = np.concatenate(
        (
            (lines.source_boundary - lines.source_inner).reshape(-1, 3),
            (lines.target_boundary - lines.target_inner).reshape(-1, 3),
        ),
        axis=0,
    ).astype(np.float64)
    centered = gradients - gradients.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered))
    trace_scale = max(float(np.trace(covariance)) / 3.0, 1.0e-6)
    covariance += np.eye(3, dtype=np.float64) * (ridge * trace_scale + 1.0e-6)
    return np.linalg.inv(covariance)


def _mgc_candidate_scores(
    lines: BorderLines,
    candidates: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    precision = _gradient_precision(lines, ridge=ridge)
    source_boundary = lines.source_boundary[:, None, :, :].astype(np.float64)
    source_inner = lines.source_inner[:, None, :, :].astype(np.float64)
    target_boundary = lines.target_boundary[candidates].astype(np.float64)
    target_inner = lines.target_inner[candidates].astype(np.float64)

    predicted_target = 2.0 * source_boundary - source_inner
    predicted_source = 2.0 * target_boundary - target_inner
    forward = target_boundary - predicted_target
    reverse = source_boundary - predicted_source
    forward_distance = np.einsum(
        "...c,cd,...d->...",
        forward,
        precision,
        forward,
        optimize=True,
    ).mean(axis=-1)
    reverse_distance = np.einsum(
        "...c,cd,...d->...",
        reverse,
        precision,
        reverse,
        optimize=True,
    ).mean(axis=-1)
    return (-0.5 * (forward_distance + reverse_distance)).astype(np.float32)


def compute_classical_candidate_scores(
    tiles: np.ndarray,
    candidates: np.ndarray,
    valid: np.ndarray,
    *,
    mgc_ridge: float = 0.05,
) -> np.ndarray:
    """Compute all nine label-free classical candidate score tensors."""
    if not np.isfinite(mgc_ridge) or mgc_ridge <= 0.0:
        raise ValueError("mgc_ridge must be finite and positive")
    rgb = _validate_tiles(tiles)
    candidate_ids, mask = _validate_candidates(candidates, valid, count=rgb.shape[0])
    normalized_rgb = exposure_normalize_rgb(rgb)
    lab = scaled_lab(rgb)
    by_name: dict[str, np.ndarray] = {
        name: np.full((NUM_DIRECTIONS, *candidate_ids.shape), -np.inf, dtype=np.float32)
        for name in VARIANT_NAMES
    }

    for depth in DEPTHS:
        for direction in range(NUM_DIRECTIONS):
            rgb_lines = directional_border_lines(normalized_rgb, direction, depth)
            lab_lines = directional_border_lines(lab, direction, depth)
            rgb_score = _ssd_candidate_scores(rgb_lines, candidate_ids)
            lab_score = _ssd_candidate_scores(lab_lines, candidate_ids)
            mgc_score = _mgc_candidate_scores(
                rgb_lines,
                candidate_ids,
                ridge=mgc_ridge,
            )
            direction_mask = mask[direction]
            by_name[f"rgb_ssd_d{depth}"][direction, direction_mask] = rgb_score[direction_mask]
            by_name[f"lab_ssd_d{depth}"][direction, direction_mask] = lab_score[direction_mask]
            by_name[f"mgc_d{depth}"][direction, direction_mask] = mgc_score[direction_mask]
    return np.stack([by_name[name] for name in VARIANT_NAMES], axis=0)


def variant_index(name: str) -> int:
    try:
        return VARIANT_NAMES.index(name)
    except ValueError as error:
        raise ValueError(f"unknown classical variant {name!r}") from error


__all__ = (
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "NUM_DIRECTIONS",
    "DEPTHS",
    "PREDECLARED_VARIANT_ORDER",
    "VARIANT_NAMES",
    "BorderLines",
    "exposure_normalize_rgb",
    "scaled_lab",
    "directional_border_lines",
    "compute_classical_candidate_scores",
    "variant_index",
)
