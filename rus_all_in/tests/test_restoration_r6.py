from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.restoration_r6 import (
    HistoricalRestoreNet,
    TileAwareDualNAFNet,
    assemble_square_tiles,
    distort_tiles,
    multi_scale_ssim_loss,
    split_square_tiles,
    tile_coordinate_channels,
)


def test_square_tile_round_trip_and_corruption() -> None:
    image = np.arange(40 * 40 * 3, dtype=np.uint16).reshape(40, 40, 3).astype(np.uint8)
    tiles = split_square_tiles(image)
    assert tiles.shape == (4, 20, 20, 3)
    np.testing.assert_array_equal(assemble_square_tiles(tiles), image)
    dirty = distort_tiles(tiles, np.random.default_rng(9))
    assert dirty.shape == tiles.shape
    assert dirty.dtype == np.uint8
    assert not np.array_equal(dirty, tiles)


def test_models_preserve_geometry_and_dual_model_starts_from_nlm() -> None:
    torch.manual_seed(3)
    source = torch.rand(1, 3, 40, 40)
    nlm = torch.rand(1, 3, 40, 40)
    historical = HistoricalRestoreNet(base=8, depth=3)
    dual = TileAwareDualNAFNet(base=8, depth=2, blocks=1)
    assert historical(source).shape == source.shape
    torch.testing.assert_close(dual(source, nlm), nlm)
    coordinates = tile_coordinate_channels(source)
    assert coordinates.shape == (1, 2, 40, 40)
    torch.testing.assert_close(coordinates[:, :, :20, :20], coordinates[:, :, 20:, 20:])


def test_restoration_loss_is_finite_and_prefers_identity() -> None:
    target = torch.rand(1, 3, 160, 160)
    same = multi_scale_ssim_loss(target, target)
    noisy = multi_scale_ssim_loss((target + 0.1 * torch.randn_like(target)).clamp(0, 1), target)
    assert torch.isfinite(same)
    assert torch.isfinite(noisy)
    assert float(same) < 1e-4
    assert float(noisy) > float(same)
