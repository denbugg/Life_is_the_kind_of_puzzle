from __future__ import annotations

import numpy as np

from aiijc_puzzle.bm3d_screen import (
    all_predictions_distinct,
    apply_bm3d_rgb,
    blend50_uint8,
    render_arms,
    structure_diagnostics,
)


def constant_image(value: int) -> np.ndarray:
    return np.full((480, 480, 3), value, dtype=np.uint8)


def test_bm3d_conversion_and_frozen_arguments() -> None:
    calls: list[tuple[float, str, str]] = []

    def fake(value: np.ndarray, sigma: float, *, profile: str, colorspace: str) -> np.ndarray:
        assert value.dtype == np.float64
        assert value.min() >= 0 and value.max() <= 1
        calls.append((sigma, profile, colorspace))
        return value + 0.1

    restored = apply_bm3d_rgb(constant_image(100), 0.16, denoiser=fake)
    assert calls == [(0.16, "np", "opp")]
    assert restored.dtype == np.uint8
    assert np.all(restored == 126)


def test_half_up_blend_is_overflow_safe() -> None:
    first = constant_image(255)
    second = constant_image(0)
    assert np.all(blend50_uint8(first, second) == 128)


def test_structure_diagnostics_exclude_puzzle_seams() -> None:
    grid = np.indices((24, 24)).sum(axis=0) % 2
    tiles = np.repeat(np.repeat(grid, 20, axis=0), 20, axis=1)
    image = np.repeat((tiles * 255).astype(np.uint8)[..., None], 3, axis=2)
    diagnostics = structure_diagnostics(image)
    assert diagnostics["within_tile_luma_gradient_mean_abs"] == 0.0
    assert diagnostics["within_tile_luma_laplacian_mean_abs"] == 0.0


def test_render_three_bm3d_arms_makes_three_distinct_calls() -> None:
    calls: list[float] = []

    def fake(value: np.ndarray, sigma: float, **_: str) -> np.ndarray:
        calls.append(sigma)
        return np.clip(value + sigma / 10, 0, 1)

    names = (
        "bm3d_rgb_sigma_0_12",
        "bm3d_rgb_sigma_0_16",
        "bm3d_rgb_sigma_0_20",
    )
    predictions, diagnostics = render_arms(constant_image(90), names, denoiser=fake)
    assert tuple(predictions) == names
    assert calls == [0.12, 0.16, 0.20]
    assert diagnostics["bm3d_calls"] == 3
    assert all_predictions_distinct(predictions)
