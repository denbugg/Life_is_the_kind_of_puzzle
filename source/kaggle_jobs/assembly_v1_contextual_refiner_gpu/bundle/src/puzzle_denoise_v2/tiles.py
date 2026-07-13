"""Tile layout helpers that never mix pixels across shuffled boundaries."""

from __future__ import annotations

import numpy as np
import torch

GRID = 24
TILE = 20
IMAGE_SIZE = GRID * TILE


def split_tiles_numpy(image: np.ndarray) -> np.ndarray:
    """Convert an HWC 480x480 RGB image to 576 HWC tiles in row-major order."""
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"expected {(IMAGE_SIZE, IMAGE_SIZE, 3)}, got {image.shape}")
    return (
        image.reshape(GRID, TILE, GRID, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(GRID * GRID, TILE, TILE, 3)
    )


def merge_tiles_numpy(tiles: np.ndarray) -> np.ndarray:
    """Convert 576 row-major HWC tiles to one HWC 480x480 RGB image."""
    if tiles.shape != (GRID * GRID, TILE, TILE, 3):
        raise ValueError(f"expected {(GRID * GRID, TILE, TILE, 3)}, got {tiles.shape}")
    return (
        tiles.reshape(GRID, GRID, TILE, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    )


def split_tiles_torch(images: torch.Tensor) -> torch.Tensor:
    """Convert BCHW images to Bx576xCx20x20 without cross-tile interpolation."""
    if images.ndim != 4 or tuple(images.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected BCHW with spatial {(IMAGE_SIZE, IMAGE_SIZE)}, got {tuple(images.shape)}")
    batch, channels = images.shape[:2]
    return (
        images.reshape(batch, channels, GRID, TILE, GRID, TILE)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, GRID * GRID, channels, TILE, TILE)
    )


def merge_tiles_torch(tiles: torch.Tensor) -> torch.Tensor:
    """Convert Bx576xCx20x20 tiles to BCHW images."""
    if tiles.ndim != 5 or tiles.shape[1] != GRID * GRID or tuple(tiles.shape[-2:]) != (TILE, TILE):
        raise ValueError(f"expected Bx576xCx20x20, got {tuple(tiles.shape)}")
    batch, _, channels = tiles.shape[:3]
    return (
        tiles.reshape(batch, GRID, GRID, channels, TILE, TILE)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, IMAGE_SIZE, IMAGE_SIZE)
    )
