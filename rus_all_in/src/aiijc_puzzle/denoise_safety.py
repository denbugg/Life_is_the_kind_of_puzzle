"""Target-free diagnostics for post-layout image restoration.

The challenge permits restoration after a strict tile permutation, but that
permission must not become a license to replace a board with a constant or a
template.  The helpers in this module measure spatial alignment, tile-level
identity, image diversity and collapse without consulting a clean target.
"""

from __future__ import annotations

import cv2
import numpy as np

from aiijc_puzzle.protocol import contest_ssim, split_tiles


def grayscale(image: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 image as uint8 luminance."""
    return cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)


def entropy_bits(image: np.ndarray) -> float:
    """Shannon entropy of the luminance histogram, in bits."""
    histogram = np.bincount(grayscale(image).reshape(-1), minlength=256).astype(np.float64)
    probability = histogram[histogram > 0] / histogram.sum()
    return float(-(probability * np.log2(probability)).sum())


def gradient_energy(image: np.ndarray) -> float:
    """Mean Sobel magnitude, used as a simple local-detail measure."""
    gray = grayscale(image).astype(np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(dx, dy)))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.std() < 1e-8 or second.std() < 1e-8:
        return float(first.std() < 1e-8 and second.std() < 1e-8 and np.allclose(first, second))
    return float(np.corrcoef(first, second)[0, 1])


def _coarse_tile_descriptors(image: np.ndarray) -> np.ndarray:
    tiles = split_tiles(image)
    pooled = (
        tiles.reshape(len(tiles), 4, 5, 4, 5, 3)
        .astype(np.float32)
        .mean(axis=(2, 4))
        .reshape(len(tiles), -1)
    )
    channel_mean = tiles.astype(np.float32).mean(axis=(1, 2))
    channel_std = tiles.astype(np.float32).std(axis=(1, 2))
    return np.concatenate((pooled, channel_mean, channel_std), axis=1) / 255.0


def coarse_tile_identity_rate(raw: np.ndarray, restored: np.ndarray) -> float:
    """Fraction of output tiles nearest to their own raw-position descriptor."""
    source = _coarse_tile_descriptors(raw)
    output = _coarse_tile_descriptors(restored)
    source_norm = np.square(source).sum(axis=1)
    output_norm = np.square(output).sum(axis=1)
    distance = output_norm[:, None] + source_norm[None] - 2.0 * output @ source.T
    return float(np.mean(np.argmin(distance, axis=1) == np.arange(len(source))))


def tile_texture_correlation(raw: np.ndarray, restored: np.ndarray) -> float:
    """Mean same-position correlation of within-tile high-pass texture."""
    raw_gray = grayscale(raw).astype(np.float32)
    restored_gray = grayscale(restored).astype(np.float32)
    raw_high_image = raw_gray - cv2.GaussianBlur(raw_gray, (0, 0), 1.2)
    restored_high_image = restored_gray - cv2.GaussianBlur(restored_gray, (0, 0), 1.2)

    def split_gray(value: np.ndarray) -> np.ndarray:
        grid = value.shape[0] // 20
        return value.reshape(grid, 20, grid, 20).transpose(0, 2, 1, 3).reshape(-1, 400)

    raw_high = split_gray(raw_high_image)
    restored_high = split_gray(restored_high_image)
    raw_high -= raw_high.mean(axis=1, keepdims=True)
    restored_high -= restored_high.mean(axis=1, keepdims=True)
    numerator = np.sum(raw_high * restored_high, axis=1)
    denominator = np.linalg.norm(raw_high, axis=1) * np.linalg.norm(restored_high, axis=1)
    correlation = numerator / np.maximum(denominator, 1e-6)
    return float(np.mean(correlation))


def _grid_masks(side: int, tile_size: int = 20) -> tuple[np.ndarray, np.ndarray]:
    coordinate = np.arange(side)
    distance = np.minimum(coordinate % tile_size, (-coordinate) % tile_size)
    seam_axis = distance <= 1
    seam = seam_axis[:, None] | seam_axis[None, :]
    return seam, ~seam


def restoration_diagnostics(raw: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    """Measure preservation and collapse relative to a frozen raw assembly."""
    raw = np.asarray(raw, dtype=np.uint8)
    restored = np.asarray(restored, dtype=np.uint8)
    if raw.shape != restored.shape or raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError("raw and restored must be equal-shaped RGB images")

    raw_f = raw.astype(np.float32)
    restored_f = restored.astype(np.float32)
    raw_gray = grayscale(raw).astype(np.float32)
    restored_gray = grayscale(restored).astype(np.float32)
    shift, response = cv2.phaseCorrelate(raw_gray, restored_gray)
    raw_tiles = split_tiles(raw_f)
    restored_tiles = split_tiles(restored_f)
    raw_tile_means = raw_tiles.mean(axis=(1, 2))
    restored_tile_means = restored_tiles.mean(axis=(1, 2))
    raw_range = float(np.percentile(raw_f, 99) - np.percentile(raw_f, 1))
    restored_range = float(np.percentile(restored_f, 99) - np.percentile(restored_f, 1))
    raw_gradient = gradient_energy(raw)
    restored_gradient = gradient_energy(restored)
    tile_std = restored_tiles.std(axis=(1, 2, 3))
    absolute_change = np.mean(np.abs(restored_f - raw_f), axis=2)
    seam, interior = _grid_masks(raw.shape[0])
    seam_change = float(absolute_change[seam].mean())
    interior_change = float(absolute_change[interior].mean())

    return {
        "phase_shift_pixels": float(np.hypot(*shift)),
        "phase_response": float(response),
        "raw_structural_ssim": contest_ssim(raw, restored),
        "mean_absolute_change": float(absolute_change.mean()),
        "global_std_ratio": float(restored_f.std() / max(raw_f.std(), 1e-6)),
        "tile_mean_std_ratio": float(restored_tile_means.std() / max(raw_tile_means.std(), 1e-6)),
        "dynamic_range_ratio": restored_range / max(raw_range, 1e-6),
        "gradient_energy_ratio": restored_gradient / max(raw_gradient, 1e-6),
        "entropy_bits": entropy_bits(restored),
        "entropy_delta_bits": entropy_bits(restored) - entropy_bits(raw),
        "near_constant_tile_fraction_std_lt_2": float(np.mean(tile_std < 2.0)),
        "near_constant_tile_fraction_std_lt_4": float(np.mean(tile_std < 4.0)),
        "near_constant_tile_fraction_std_lt_8": float(np.mean(tile_std < 8.0)),
        "tile_mean_correlation": _correlation(raw_tile_means, restored_tile_means),
        "coarse_tile_descriptor_top1": coarse_tile_identity_rate(raw, restored),
        "tile_texture_correlation": tile_texture_correlation(raw, restored),
        "seam_to_interior_change_ratio": seam_change / max(interior_change, 1e-6),
    }


def board_descriptor(image: np.ndarray) -> np.ndarray:
    """Compact absolute-color spatial descriptor for cross-board identity checks."""
    pooled = cv2.resize(np.asarray(image, dtype=np.uint8), (32, 32), interpolation=cv2.INTER_AREA)
    return pooled.astype(np.float32).reshape(-1) / 255.0


def cross_board_diagnostics(
    raw_images: list[np.ndarray], restored_images: list[np.ndarray]
) -> dict[str, float | int]:
    """Detect convergence of distinct inputs toward one generic output."""
    if len(raw_images) != len(restored_images) or len(raw_images) < 2:
        raise ValueError("cross-board diagnostics need equal lists of at least two images")
    raw = np.stack([board_descriptor(image) for image in raw_images])
    restored = np.stack([board_descriptor(image) for image in restored_images])
    distance = (
        np.square(restored).sum(axis=1)[:, None]
        + np.square(raw).sum(axis=1)[None]
        - 2.0 * restored @ raw.T
    )
    own = np.diag(distance)
    masked = distance.copy()
    np.fill_diagonal(masked, np.inf)
    nearest_other = masked.min(axis=1)

    def pairwise_mean(values: np.ndarray) -> float:
        difference = values[:, None] - values[None]
        pairwise = np.sqrt(np.maximum(np.square(difference).sum(axis=2), 0.0))
        return float(pairwise[np.triu_indices(len(values), k=1)].mean())

    raw_pairwise = pairwise_mean(raw)
    restored_pairwise = pairwise_mean(restored)
    margins = (nearest_other - own) / np.maximum(nearest_other, 1e-8)
    return {
        "boards": len(raw_images),
        "own_raw_board_top1_count": int(np.sum(np.argmin(distance, axis=1) == np.arange(len(raw)))),
        "own_raw_board_margin_mean": float(margins.mean()),
        "own_raw_board_margin_min": float(margins.min()),
        "pairwise_board_distance_ratio": restored_pairwise / max(raw_pairwise, 1e-8),
        "cross_board_pixel_variance_ratio": float(
            restored.var(axis=0).mean() / max(raw.var(axis=0).mean(), 1e-8)
        ),
    }
