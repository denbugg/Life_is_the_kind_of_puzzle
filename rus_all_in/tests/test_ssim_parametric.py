from __future__ import annotations

import numpy as np

from aiijc_puzzle.protocol import assemble_tiles, contest_ssim, split_tiles
from aiijc_puzzle.ssim_parametric import (
    constant_channel_ssim,
    extract_invariant_features,
    input_median_rgb,
    paired_bootstrap_interval,
    render_constant_rgb,
    ssim_optimal_constant_rgb,
)


def test_constant_formula_matches_contest_metric() -> None:
    rng = np.random.default_rng(17)
    target = rng.integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    color = np.asarray((37, 128, 241), dtype=np.float64)
    expected = contest_ssim(target, render_constant_rgb(color))
    actual = np.mean(
        [constant_channel_ssim(target[:, :, channel], color[channel]) for channel in range(3)]
    )
    assert abs(expected - actual) < 1e-12


def test_oracle_recovers_a_constant_target() -> None:
    target = render_constant_rgb(np.asarray((23, 119, 231)))
    optimum = ssim_optimal_constant_rgb(target)
    np.testing.assert_allclose(optimum, (23, 119, 231), atol=1e-4)
    assert contest_ssim(target, render_constant_rgb(optimum)) == 1.0


def test_features_are_invariant_to_complete_tile_permutation() -> None:
    rng = np.random.default_rng(29)
    image = rng.integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    permuted = assemble_tiles(split_tiles(image)[rng.permutation(576)])
    np.testing.assert_allclose(
        extract_invariant_features(image),
        extract_invariant_features(permuted),
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_array_equal(input_median_rgb(image), input_median_rgb(permuted))


def test_bootstrap_is_deterministic() -> None:
    differences = np.asarray((0.01, 0.02, -0.005, 0.03))
    first = paired_bootstrap_interval(differences, replicates=500, seed=11)
    second = paired_bootstrap_interval(differences, replicates=500, seed=11)
    assert first == second
    assert first.wins == 3
    assert first.count == 4
