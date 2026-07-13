"""Weak content-based spatial priors for puzzle component placement."""

from __future__ import annotations

import numpy as np

from .geometry import GRID, TILE, TILE_COUNT


def tile_spatial_features(tiles: np.ndarray) -> np.ndarray:
    """Extract compact colour, texture and coarse-shape features per tile."""
    values = np.asarray(tiles)
    if values.shape != (TILE_COUNT, TILE, TILE, 3):
        raise ValueError(f"expected {(TILE_COUNT, TILE, TILE, 3)}, got {values.shape}")
    values = values.astype(np.float32) / 255.0
    flat = values.reshape(TILE_COUNT, -1, 3)
    features = [
        flat.mean(axis=1),
        flat.std(axis=1),
        np.quantile(flat, 0.10, axis=1).astype(np.float32),
        np.quantile(flat, 0.50, axis=1).astype(np.float32),
        np.quantile(flat, 0.90, axis=1).astype(np.float32),
    ]
    luminance = (
        0.2126 * values[..., 0]
        + 0.7152 * values[..., 1]
        + 0.0722 * values[..., 2]
    )
    horizontal = np.diff(luminance, axis=2)
    vertical = np.diff(luminance, axis=1)
    texture = np.stack(
        [
            np.mean(np.abs(horizontal), axis=(1, 2)),
            np.mean(np.abs(vertical), axis=(1, 2)),
            np.std(horizontal, axis=(1, 2)),
            np.std(vertical, axis=(1, 2)),
            np.mean(np.max(values, axis=3) - np.min(values, axis=3), axis=(1, 2)),
            np.std(luminance, axis=(1, 2)),
        ],
        axis=1,
    ).astype(np.float32)
    features.append(texture)

    # A 4x4 pooled RGB thumbnail retains weak semantic/layout cues (sky, floor,
    # skin, text orientation) while staying cheap enough for hundreds of
    # thousands of training tiles.
    pooled = values.reshape(TILE_COUNT, 4, 5, 4, 5, 3).mean(axis=(2, 4))
    features.append(pooled.reshape(TILE_COUNT, -1))
    gradient = np.sqrt(
        np.pad(horizontal * horizontal, ((0, 0), (0, 0), (0, 1)))
        + np.pad(vertical * vertical, ((0, 0), (0, 1), (0, 0)))
    )
    pooled_gradient = gradient.reshape(TILE_COUNT, 4, 5, 4, 5).mean(axis=(2, 4))
    features.append(pooled_gradient.reshape(TILE_COUNT, -1))
    return np.concatenate(features, axis=1).astype(np.float32)


def spatial_prior_cost(model: object, tiles: np.ndarray) -> np.ndarray:
    """Convert predicted normalised row/column into a position x tile cost."""
    predictions = np.asarray(model.predict(tile_spatial_features(tiles)), dtype=np.float32)
    if predictions.shape != (TILE_COUNT, 2) or not np.all(np.isfinite(predictions)):
        raise ValueError("spatial prior model must predict a finite 576x2 array")
    predictions = np.clip(predictions, 0.0, 1.0)
    positions = np.arange(TILE_COUNT, dtype=np.int32)
    rows = (positions // GRID).astype(np.float32) / float(GRID - 1)
    columns = (positions % GRID).astype(np.float32) / float(GRID - 1)
    costs = 0.5 * (
        (rows[:, None] - predictions[None, :, 0]) ** 2
        + (columns[:, None] - predictions[None, :, 1]) ** 2
    )
    return costs.astype(np.float32)
