from __future__ import annotations

import numpy as np

from aiijc_puzzle.denoise_safety import (
    coarse_tile_identity_rate,
    cross_board_diagnostics,
    restoration_diagnostics,
)


def patterned_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    image[:240, :240] //= 3
    return image


def test_identity_restoration_preserves_all_diagnostics() -> None:
    image = patterned_image(7)
    metrics = restoration_diagnostics(image, image.copy())
    assert metrics["phase_shift_pixels"] < 1e-6
    assert metrics["raw_structural_ssim"] == 1.0
    assert metrics["global_std_ratio"] == 1.0
    assert metrics["tile_mean_correlation"] > 0.999999
    assert metrics["coarse_tile_descriptor_top1"] == 1.0
    assert metrics["tile_texture_correlation"] > 0.999999


def test_constant_collapse_is_detected() -> None:
    image = patterned_image(11)
    collapsed = np.full_like(image, 127)
    metrics = restoration_diagnostics(image, collapsed)
    assert metrics["global_std_ratio"] == 0.0
    assert metrics["near_constant_tile_fraction_std_lt_2"] == 1.0
    assert metrics["entropy_bits"] == 0.0
    assert coarse_tile_identity_rate(image, collapsed) < 1.0


def test_cross_board_identity_and_diversity() -> None:
    images = [patterned_image(seed) for seed in (2, 3, 5)]
    metrics = cross_board_diagnostics(images, [image.copy() for image in images])
    assert metrics["own_raw_board_top1_count"] == 3
    assert metrics["pairwise_board_distance_ratio"] > 0.999999
    assert metrics["cross_board_pixel_variance_ratio"] > 0.999999
