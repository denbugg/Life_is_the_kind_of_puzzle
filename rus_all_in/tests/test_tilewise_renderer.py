from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.restoration_r6 import TileAwareDualNAFNet
from aiijc_puzzle.tilewise_renderer import render_tiles_independently


def random_tiles(seed: int = 11) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(576, 20, 20, 3), dtype=np.uint8)


def tiny_model() -> TileAwareDualNAFNet:
    torch.manual_seed(5)
    model = TileAwareDualNAFNet(base=4, depth=2, blocks=1)
    with torch.no_grad():
        model.head.weight.normal_(0.0, 0.01)
    return model


def test_tilewise_renderer_preserves_roster_and_dtype() -> None:
    source = random_tiles()
    rendered, diagnostics = render_tiles_independently(
        tiny_model(), source, torch.device("cpu"), nlm_h=10, batch_size=192
    )
    assert rendered.shape == source.shape
    assert rendered.dtype == np.uint8
    assert diagnostics.tile_count == 576
    assert diagnostics.batch_size == 192


def test_tilewise_renderer_has_no_cross_tile_dependency() -> None:
    source = random_tiles()
    changed = source.copy()
    changed[0] = 255 - changed[0]
    model = tiny_model()
    first, _ = render_tiles_independently(
        model, source, torch.device("cpu"), nlm_h=10, batch_size=144
    )
    second, _ = render_tiles_independently(
        model, changed, torch.device("cpu"), nlm_h=10, batch_size=144
    )
    assert np.array_equal(first[1:], second[1:])


def test_tilewise_renderer_is_batch_partition_invariant() -> None:
    source = random_tiles()
    model = tiny_model()
    first, _ = render_tiles_independently(
        model, source, torch.device("cpu"), nlm_h=10, batch_size=576
    )
    second, _ = render_tiles_independently(
        model, source, torch.device("cpu"), nlm_h=10, batch_size=73
    )
    assert np.array_equal(first, second)
