"""Bit-for-bit E2 raw-tile MGC+SSD score construction (commit 63c1456)."""
from __future__ import annotations

import numpy as np
from scipy.special import log_softmax

GRID, TILE, N = 24, 20, 576
ALPHA = 0.2
DUMMY_DIFFS = np.asarray(
    [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [0, 0, 1], [0, 1, 0],
     [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
    dtype=np.float32,
)


def _mahalanobis_gradient_cost(
    source_boundary: np.ndarray,
    source_inner: np.ndarray,
    target_boundary: np.ndarray,
    *,
    batch_size: int = 24,
) -> np.ndarray:
    source_boundary = np.asarray(source_boundary, np.float32)
    source_inner = np.asarray(source_inner, np.float32)
    target_boundary = np.asarray(target_boundary, np.float32)
    gradients = source_boundary - source_inner
    means = gradients.mean(axis=1)
    dummy = np.broadcast_to(DUMMY_DIFFS, (N, *DUMMY_DIFFS.shape))
    samples = np.concatenate((gradients, dummy), axis=1).astype(np.float64)
    centered = samples - samples.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True) / (
        samples.shape[1] - 1
    )
    precisions = np.linalg.inv(covariance).astype(np.float32)
    costs = np.empty((N, N), np.float32)
    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        residual = (
            target_boundary[None, :, :, :]
            - source_boundary[start:stop, None, :, :]
            - means[start:stop, None, None, :]
        )
        costs[start:stop] = np.einsum(
            "btkc,bcd,btkd->bt", residual, precisions[start:stop], residual,
            optimize=True,
        )
    return costs


def _ssd_cost(
    source_boundary: np.ndarray,
    target_boundary: np.ndarray,
    *,
    batch_size: int = 24,
) -> np.ndarray:
    source_boundary = np.asarray(source_boundary, np.float32)
    target_boundary = np.asarray(target_boundary, np.float32)
    costs = np.empty((N, N), np.float32)
    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        residual = source_boundary[start:stop, None] - target_boundary[None]
        costs[start:stop] = np.einsum(
            "btkc,btkc->bt", residual, residual, optimize=True
        )
    return costs


def _row_robust_dissimilarity(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, np.float32)
    off_diagonal = cost[~np.eye(N, dtype=bool)].reshape(N, N - 1)
    median = np.median(off_diagonal, axis=1, keepdims=True)
    mad = np.median(np.abs(off_diagonal - median), axis=1, keepdims=True)
    scaled = (cost - median) / np.maximum(mad, 1e-6)
    np.fill_diagonal(scaled, np.inf)
    return scaled


def _dissimilarity_logp(dissimilarity: np.ndarray) -> np.ndarray:
    dissimilarity = np.asarray(dissimilarity, np.float32)
    off_diagonal = dissimilarity[~np.eye(N, dtype=bool)].reshape(N, N - 1)
    median = np.median(off_diagonal, axis=1, keepdims=True)
    mad = np.median(np.abs(off_diagonal - median), axis=1, keepdims=True)
    z = -(dissimilarity - median) / np.maximum(mad, 1e-6)
    np.fill_diagonal(z, -1e4)
    return log_softmax(z, axis=1).astype(np.float32)


def classical_mgc_ssd_scores(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tiles = np.asarray(tiles)
    if tiles.shape != (N, TILE, TILE, 3):
        raise ValueError(f"expected {(N, TILE, TILE, 3)} tiles, got {tiles.shape}")
    pixel = tiles.astype(np.float32)
    left, left_inner = pixel[:, :, 0, :], pixel[:, :, 1, :]
    right, right_inner = pixel[:, :, -1, :], pixel[:, :, -2, :]
    top, top_inner = pixel[:, 0, :, :], pixel[:, 1, :, :]
    bottom, bottom_inner = pixel[:, -1, :, :], pixel[:, -2, :, :]
    right_mgc = _mahalanobis_gradient_cost(right, right_inner, left)
    right_mgc += _mahalanobis_gradient_cost(left, left_inner, right).T
    down_mgc = _mahalanobis_gradient_cost(bottom, bottom_inner, top)
    down_mgc += _mahalanobis_gradient_cost(top, top_inner, bottom).T
    right_dissimilarity = 0.5 * (
        _row_robust_dissimilarity(right_mgc)
        + _row_robust_dissimilarity(_ssd_cost(right, left))
    )
    down_dissimilarity = 0.5 * (
        _row_robust_dissimilarity(down_mgc)
        + _row_robust_dissimilarity(_ssd_cost(bottom, top))
    )
    return _dissimilarity_logp(right_dissimilarity), _dissimilarity_logp(
        down_dissimilarity
    )


def fuse_scores(
    learned: np.ndarray, classical: np.ndarray, *, alpha: float = ALPHA
) -> np.ndarray:
    if alpha != ALPHA:
        raise ValueError(f"E14 locks alpha={ALPHA}, got {alpha}")
    fused = (1.0 - alpha) * np.asarray(learned) + alpha * np.asarray(classical)
    fused = np.asarray(fused, np.float32)
    np.fill_diagonal(fused, -1e4)
    return fused
