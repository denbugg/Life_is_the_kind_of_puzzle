from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import torch
from torch import nn

from aiijc_puzzle.pretrained_tile_denoiser import (
    DrunetColor,
    blend_uint8_fraction,
    board_safety_diagnostics,
    candidate_safety_ratios,
    render_drunet_tiles,
)


class EchoRgb(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :3]


def test_checkpoint_compatible_topology_preserves_spatial_shape() -> None:
    model = DrunetColor(channels=(4, 8, 16, 32), blocks=1)
    value = torch.rand(2, 4, 24, 24)
    output = model(value)
    assert output.shape == (2, 3, 24, 24)
    assert "m_down1.0.res.0.weight" in model.state_dict()
    assert "m_up3.0.weight" in model.state_dict()


def test_renderer_reflects_then_crops_without_pixel_change_for_echo() -> None:
    rng = np.random.default_rng(4)
    tiles = rng.integers(0, 256, size=(3, 20, 20, 3), dtype=np.uint8)
    restored, diagnostics = render_drunet_tiles(
        EchoRgb(),
        tiles,
        sigma_255=25,
        device=torch.device("cpu"),
        batch_size=2,
    )
    np.testing.assert_array_equal(restored, tiles)
    assert diagnostics.tile_count == 3
    assert diagnostics.padding_bottom == 4
    assert diagnostics.padding_right == 4
    assert diagnostics.maximum_abs_change == 0


def test_renderer_has_no_cross_tile_dependency() -> None:
    tiles = np.full((2, 20, 20, 3), 60, dtype=np.uint8)
    changed = tiles.copy()
    changed[1] = 200
    first, _ = render_drunet_tiles(
        EchoRgb(), tiles, sigma_255=10, device=torch.device("cpu"), batch_size=2
    )
    second, _ = render_drunet_tiles(
        EchoRgb(), changed, sigma_255=10, device=torch.device("cpu"), batch_size=1
    )
    np.testing.assert_array_equal(first[0], second[0])


def test_fractional_blend_uses_half_up_integer_arithmetic() -> None:
    left = np.array([0, 10, 100, 255], dtype=np.uint8)
    right = np.array([4, 14, 108, 247], dtype=np.uint8)
    blended = blend_uint8_fraction(left, right, Fraction(1, 8))
    np.testing.assert_array_equal(blended, np.array([1, 11, 101, 254], dtype=np.uint8))
    with pytest.raises(ValueError):
        blend_uint8_fraction(left, right, Fraction(0, 1))


def test_safety_ratios_are_identity_for_the_same_board() -> None:
    row = np.arange(480, dtype=np.uint16) % 256
    board = np.stack(np.meshgrid(row, row, indexing="ij"), axis=-1)
    board = np.concatenate((board, board[..., :1]), axis=2).astype(np.uint8)
    diagnostics = board_safety_diagnostics(board)
    ratios = candidate_safety_ratios(board, board.copy())
    assert diagnostics["within_tile_luma_gradient_mean_abs"] > 0
    assert diagnostics["grid_luma_seam_mean_abs"] > 0
    assert ratios["gradient_ratio_vs_h28"] == pytest.approx(1.0)
    assert ratios["laplacian_ratio_vs_h28"] == pytest.approx(1.0)
    assert ratios["grid_seam_ratio_vs_h28"] == pytest.approx(1.0)
    assert ratios["maximum_abs_rgb_mean_shift_vs_h28"] == 0
    assert ratios["mean_abs_pixel_change_vs_h28"] == 0
