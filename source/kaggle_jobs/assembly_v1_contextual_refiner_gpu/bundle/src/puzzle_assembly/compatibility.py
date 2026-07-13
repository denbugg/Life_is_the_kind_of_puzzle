"""Chunked directional compatibility scores for 576 denoised tiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from .geometry import TILE, TILE_COUNT


@dataclass(frozen=True)
class CompatibilityMatrices:
    name: str
    right: np.ndarray
    down: np.ndarray

    def __post_init__(self) -> None:
        for direction, matrix in (("right", self.right), ("down", self.down)):
            matrix = np.asarray(matrix)
            if matrix.shape != (TILE_COUNT, TILE_COUNT):
                raise ValueError(f"{self.name}.{direction} has shape {matrix.shape}")
            if matrix.dtype != np.float32:
                object.__setattr__(self, direction, matrix.astype(np.float32))


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    tiles = np.asarray(tiles)
    if tiles.shape != (TILE_COUNT, TILE, TILE, 3):
        raise ValueError(f"expected {(TILE_COUNT, TILE, TILE, 3)}, got {tiles.shape}")
    if tiles.dtype != np.uint8:
        raise TypeError(f"tiles must be uint8, got {tiles.dtype}")
    return tiles


def _pairwise_l1(left: np.ndarray, right: np.ndarray, *, chunk_size: int) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32).reshape(len(left), -1)
    right = np.asarray(right, dtype=np.float32).reshape(len(right), -1)
    if left.shape[1] != right.shape[1]:
        raise ValueError("pairwise feature dimensions differ")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    result = np.empty((len(left), len(right)), dtype=np.float32)
    for start in range(0, len(left), chunk_size):
        block = left[start : start + chunk_size]
        result[start : start + len(block)] = np.mean(
            np.abs(block[:, None, :] - right[None, :, :]), axis=2, dtype=np.float32
        )
    return result


def _pairwise_hamming(left: np.ndarray, right: np.ndarray, *, chunk_size: int) -> np.ndarray:
    left = np.asarray(left, dtype=bool).reshape(len(left), -1)
    right = np.asarray(right, dtype=bool).reshape(len(right), -1)
    if left.shape[1] != right.shape[1]:
        raise ValueError("pairwise feature dimensions differ")
    result = np.empty((len(left), len(right)), dtype=np.float32)
    for start in range(0, len(left), chunk_size):
        block = left[start : start + chunk_size]
        result[start : start + len(block)] = np.mean(
            block[:, None, :] != right[None, :, :], axis=2, dtype=np.float32
        )
    return result


def _pairwise_bounded_affine(
    query: np.ndarray,
    key: np.ndarray,
    *,
    chunk_size: int,
    scale_bounds: tuple[float, float] = (0.7, 1.3),
    offset_bound: float = 30.0 / 255.0,
) -> np.ndarray:
    """Fit key*a+b to each query with documented nuisance bounds."""
    query = np.asarray(query, dtype=np.float32)
    key = np.asarray(key, dtype=np.float32)
    if query.shape[1:] != key.shape[1:] or query.ndim != 3 or query.shape[-1] != 3:
        raise ValueError("affine features must have matching NxLx3 shapes")
    result = np.empty((len(query), len(key)), dtype=np.float32)
    key_expanded = key[None, :, :, :]
    key_mean = key_expanded.mean(axis=2, keepdims=True)
    key_centered = key_expanded - key_mean
    key_var = np.mean(key_centered * key_centered, axis=2, keepdims=True)
    for start in range(0, len(query), chunk_size):
        block = query[start : start + chunk_size, None, :, :]
        block_mean = block.mean(axis=2, keepdims=True)
        covariance = np.mean(
            (block - block_mean) * key_centered, axis=2, keepdims=True
        )
        scale = covariance / np.maximum(key_var, 1e-6)
        scale = np.where(key_var > 1e-6, scale, 1.0)
        scale = np.clip(scale, scale_bounds[0], scale_bounds[1])
        offset = np.clip(block_mean - scale * key_mean, -offset_bound, offset_bound)
        predicted = scale * key_expanded + offset
        result[start : start + len(block)] = np.mean(
            np.abs(block - predicted), axis=(2, 3), dtype=np.float32
        )
    return result


def _pairwise_mahalanobis(
    left: np.ndarray,
    right: np.ndarray,
    inverse_covariance: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    result = np.empty((len(left), len(right)), dtype=np.float32)
    for start in range(0, len(left), chunk_size):
        difference = left[start : start + chunk_size, None, :, :] - right[None, :, :, :]
        squared = np.einsum(
            "...c,cd,...d->...", difference, inverse_covariance, difference, optimize=True
        )
        result[start : start + len(difference)] = np.mean(squared, axis=2, dtype=np.float32)
    return result


def _mask_self(right: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.asarray(right, dtype=np.float32)
    down = np.asarray(down, dtype=np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return right, down


def _border_l1(tiles: np.ndarray, strip: int, *, chunk_size: int) -> CompatibilityMatrices:
    if strip not in {1, 2, 4}:
        raise ValueError("strip must be one of 1, 2, 4")
    values = tiles.astype(np.float32) / 255.0
    right = _pairwise_l1(values[:, :, -strip:, :], values[:, :, :strip, :], chunk_size=chunk_size)
    down = _pairwise_l1(values[:, -strip:, :, :], values[:, :strip, :, :], chunk_size=chunk_size)
    right, down = _mask_self(right, down)
    return CompatibilityMatrices(f"rgb_l1_w{strip}", right, down)


def _tone_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32).reshape(len(features), -1)
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-4)


def _tone_l1(tiles: np.ndarray, strip: int, *, chunk_size: int) -> CompatibilityMatrices:
    values = tiles.astype(np.float32) / 255.0
    right_query = _tone_normalize(values[:, :, -strip:, :])
    right_key = _tone_normalize(values[:, :, :strip, :])
    down_query = _tone_normalize(values[:, -strip:, :, :])
    down_key = _tone_normalize(values[:, :strip, :, :])
    right = _pairwise_l1(right_query, right_key, chunk_size=chunk_size)
    down = _pairwise_l1(down_query, down_key, chunk_size=chunk_size)
    right, down = _mask_self(right, down)
    return CompatibilityMatrices(f"tone_l1_w{strip}", right, down)


def _prediction_compatibility(tiles: np.ndarray, *, chunk_size: int) -> CompatibilityMatrices:
    """Symmetric one-pixel linear continuation across a proposed seam."""
    values = tiles.astype(np.float32) / 255.0

    right_prediction = np.clip(2.0 * values[:, :, -1, :] - values[:, :, -2, :], 0.0, 1.0)
    left_boundary = values[:, :, 0, :]
    reverse_left_prediction = np.clip(2.0 * values[:, :, 0, :] - values[:, :, 1, :], 0.0, 1.0)
    right_boundary = values[:, :, -1, :]
    right = 0.5 * (
        _pairwise_l1(right_prediction, left_boundary, chunk_size=chunk_size)
        + _pairwise_l1(right_boundary, reverse_left_prediction, chunk_size=chunk_size)
    )

    down_prediction = np.clip(2.0 * values[:, -1, :, :] - values[:, -2, :, :], 0.0, 1.0)
    top_boundary = values[:, 0, :, :]
    reverse_top_prediction = np.clip(2.0 * values[:, 0, :, :] - values[:, 1, :, :], 0.0, 1.0)
    bottom_boundary = values[:, -1, :, :]
    down = 0.5 * (
        _pairwise_l1(down_prediction, top_boundary, chunk_size=chunk_size)
        + _pairwise_l1(bottom_boundary, reverse_top_prediction, chunk_size=chunk_size)
    )
    right, down = _mask_self(right, down)
    return CompatibilityMatrices("pbc", right, down)


def prediction_compatibility(
    tiles: np.ndarray, *, prefix: str = "denoised", chunk_size: int = 64
) -> CompatibilityMatrices:
    tiles = _validate_tiles(tiles)
    score = _prediction_compatibility(tiles, chunk_size=chunk_size)
    return CompatibilityMatrices(f"{prefix}_pbc", score.right, score.down)


def _gradient_covariance_inverse(gradients: np.ndarray) -> np.ndarray:
    flattened = np.asarray(gradients, dtype=np.float64).reshape(-1, 3)
    covariance = np.cov(flattened, rowvar=False)
    regularizer = max(float(np.trace(covariance)) / 3.0, 1e-6) * 1e-3
    return np.linalg.inv(covariance + regularizer * np.eye(3)).astype(np.float32)


def _mahalanobis_gradient_compatibility(
    tiles: np.ndarray, *, chunk_size: int
) -> CompatibilityMatrices:
    """MGC-style seam difference under global within-tile gradient covariance."""
    values = tiles.astype(np.float32) / 255.0
    horizontal_inverse = _gradient_covariance_inverse(np.diff(values, axis=2))
    vertical_inverse = _gradient_covariance_inverse(np.diff(values, axis=1))
    right = _pairwise_mahalanobis(
        values[:, :, -1, :],
        values[:, :, 0, :],
        horizontal_inverse,
        chunk_size=chunk_size,
    )
    down = _pairwise_mahalanobis(
        values[:, -1, :, :],
        values[:, 0, :, :],
        vertical_inverse,
        chunk_size=chunk_size,
    )
    right, down = _mask_self(right, down)
    return CompatibilityMatrices("mgc", right, down)


def _c2_direction(
    query_prediction: np.ndarray,
    query_boundary: np.ndarray,
    key_boundary: np.ndarray,
    key_prediction: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    forward = _pairwise_bounded_affine(
        query_prediction, key_boundary, chunk_size=chunk_size
    )
    reverse = _pairwise_bounded_affine(
        key_prediction, query_boundary, chunk_size=chunk_size
    ).T
    affine = 0.5 * (forward + reverse)

    query_tangent = _tone_normalize(np.diff(query_boundary, axis=1))
    key_tangent = _tone_normalize(np.diff(key_boundary, axis=1))
    tangent = _pairwise_l1(query_tangent, key_tangent, chunk_size=chunk_size)
    census = _pairwise_hamming(
        np.diff(query_boundary, axis=1) >= 0.0,
        np.diff(key_boundary, axis=1) >= 0.0,
        chunk_size=chunk_size,
    )
    # Rank-domain combination prevents one nuisance term from winning purely
    # because its physical scale differs from the others.
    return (
        0.60 * rank_normalize(affine)
        + 0.25 * rank_normalize(tangent)
        + 0.15 * rank_normalize(census)
    ).astype(np.float32)


def _corruption_nuisance_compatibility(
    tiles: np.ndarray, *, chunk_size: int
) -> CompatibilityMatrices:
    """C2: bounded affine tone fitting plus tangential gradient/census evidence."""
    values = tiles.astype(np.float32) / 255.0
    right = _c2_direction(
        np.clip(2.0 * values[:, :, -1, :] - values[:, :, -2, :], 0.0, 1.0),
        values[:, :, -1, :],
        values[:, :, 0, :],
        np.clip(2.0 * values[:, :, 0, :] - values[:, :, 1, :], 0.0, 1.0),
        chunk_size=chunk_size,
    )
    down = _c2_direction(
        np.clip(2.0 * values[:, -1, :, :] - values[:, -2, :, :], 0.0, 1.0),
        values[:, -1, :, :],
        values[:, 0, :, :],
        np.clip(2.0 * values[:, 0, :, :] - values[:, 1, :, :], 0.0, 1.0),
        chunk_size=chunk_size,
    )
    right, down = _mask_self(right, down)
    return CompatibilityMatrices("c2", right, down)


def _lab_l1(tiles: np.ndarray, strip: int, *, chunk_size: int) -> CompatibilityMatrices:
    flattened = np.ascontiguousarray(tiles.reshape(TILE_COUNT * TILE, TILE, 3))
    lab = cv2.cvtColor(flattened, cv2.COLOR_RGB2LAB).reshape(TILE_COUNT, TILE, TILE, 3)
    result = _border_l1(lab, strip, chunk_size=chunk_size)
    return CompatibilityMatrices(f"lab_l1_w{strip}", result.right, result.down)


def _normalized_sobel_features(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return contrast-normalized Sobel vectors and a binary edge mask per tile.

    Normalizing by each tile's maximum magnitude makes this representation less
    sensitive to the documented per-tile contrast corruption.  The binary mask
    intentionally mirrors ``SideEmbeddingNet(input_mode='binary_edges')`` so the
    filter-only and learned experiments test the same hypothesis.
    """
    values = tiles.astype(np.float32) / 255.0
    gray = values.mean(axis=3)
    gradient_x = np.empty_like(gray)
    gradient_y = np.empty_like(gray)
    for index in range(TILE_COUNT):
        gradient_x[index] = cv2.Sobel(
            gray[index], cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE
        )
        gradient_y[index] = cv2.Sobel(
            gray[index], cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE
        )
    magnitude = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y + 1e-8)
    scale = np.maximum(magnitude.max(axis=(1, 2), keepdims=True), 1e-4)
    gradient_x /= scale
    gradient_y /= scale
    magnitude /= scale
    vectors = np.stack([gradient_x, gradient_y, magnitude], axis=3).astype(np.float32)
    return vectors, magnitude >= 0.12


def build_edge_filter_score_bank(
    tiles: np.ndarray,
    *,
    prefix: str = "denoised",
    strip: int = 2,
    chunk_size: int = 64,
) -> dict[str, CompatibilityMatrices]:
    """Build direct Sobel-vector and binary-edge seam scores without learning."""
    tiles = _validate_tiles(tiles)
    if strip not in {1, 2, 4}:
        raise ValueError("strip must be one of 1, 2, 4")
    sobel, binary = _normalized_sobel_features(tiles)
    right_sobel = _pairwise_l1(
        sobel[:, :, -strip:, :], sobel[:, :, :strip, :], chunk_size=chunk_size
    )
    down_sobel = _pairwise_l1(
        sobel[:, -strip:, :, :], sobel[:, :strip, :, :], chunk_size=chunk_size
    )
    right_binary = _pairwise_hamming(
        binary[:, :, -strip:], binary[:, :, :strip], chunk_size=chunk_size
    )
    down_binary = _pairwise_hamming(
        binary[:, -strip:, :], binary[:, :strip, :], chunk_size=chunk_size
    )
    right_sobel, down_sobel = _mask_self(right_sobel, down_sobel)
    right_binary, down_binary = _mask_self(right_binary, down_binary)
    sobel_name = f"{prefix}_sobel_l1_w{strip}"
    binary_name = f"{prefix}_binary_edge_hamming_w{strip}"
    return {
        sobel_name: CompatibilityMatrices(sobel_name, right_sobel, down_sobel),
        binary_name: CompatibilityMatrices(binary_name, right_binary, down_binary),
    }


def build_classical_score_bank(
    tiles: np.ndarray,
    *,
    prefix: str = "denoised",
    chunk_size: int = 64,
) -> dict[str, CompatibilityMatrices]:
    """Build the first reproducible denoised classical score bank.

    The bank intentionally starts with cheap, independently interpretable scores.
    MGC and learned scores are added as separate hypotheses rather than hidden in
    this baseline.
    """
    tiles = _validate_tiles(tiles)
    scorers = [
        _border_l1(tiles, 1, chunk_size=chunk_size),
        _border_l1(tiles, 2, chunk_size=chunk_size),
        _border_l1(tiles, 4, chunk_size=chunk_size),
        _lab_l1(tiles, 2, chunk_size=chunk_size),
        _tone_l1(tiles, 2, chunk_size=chunk_size),
        _prediction_compatibility(tiles, chunk_size=chunk_size),
        _mahalanobis_gradient_compatibility(tiles, chunk_size=chunk_size),
        _corruption_nuisance_compatibility(tiles, chunk_size=chunk_size),
    ]
    return {
        f"{prefix}_{score.name}": CompatibilityMatrices(
            f"{prefix}_{score.name}", score.right, score.down
        )
        for score in scorers
    }


def rank_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError("matrix must be 576x576")
    order = np.argsort(matrix, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    rows = np.arange(TILE_COUNT)[:, None]
    ranks[rows, order] = np.arange(TILE_COUNT, dtype=np.int32)[None, :]
    normalized = ranks.astype(np.float32) / float(TILE_COUNT - 2)
    np.fill_diagonal(normalized, np.inf)
    return normalized


def fuse_ranked_scores(
    score_bank: Mapping[str, CompatibilityMatrices],
    *,
    names: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    name: str = "rank_fusion",
) -> CompatibilityMatrices:
    selected = list(names) if names is not None else sorted(score_bank)
    if not selected:
        raise ValueError("at least one score is required")
    missing = [key for key in selected if key not in score_bank]
    if missing:
        raise KeyError(f"unknown score names: {missing}")
    resolved_weights = {
        key: float(weights[key]) if weights is not None and key in weights else 1.0
        for key in selected
    }
    if any(not np.isfinite(value) or value < 0 for value in resolved_weights.values()):
        raise ValueError("fusion weights must be finite and non-negative")
    total = sum(resolved_weights.values())
    if total <= 0:
        raise ValueError("at least one fusion weight must be positive")
    right = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    down = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    for key in selected:
        weight = resolved_weights[key] / total
        right += weight * rank_normalize(score_bank[key].right)
        down += weight * rank_normalize(score_bank[key].down)
    right, down = _mask_self(right, down)
    return CompatibilityMatrices(name, right, down)
