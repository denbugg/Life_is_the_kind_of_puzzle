from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from aiijc_puzzle.postassembly_harmonizer import (
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.protocol import contest_ssim, split_tiles

ROOT = Path(__file__).resolve().parents[1]


def smooth_target() -> np.ndarray:
    y, x = np.mgrid[:480, :480]
    return (
        np.stack(
            (
                30.0 + 0.25 * x + 0.05 * y,
                40.0 + 0.10 * x + 0.20 * y,
                50.0 + 0.15 * x + 0.10 * y,
            ),
            axis=-1,
        )
        .clip(0, 255)
        .astype(np.uint8)
    )


def test_seam_graph_recovers_synthetic_additive_tile_bias() -> None:
    target = split_tiles(smooth_target())
    rng = np.random.default_rng(7)
    bias = rng.uniform(-8.0, 8.0, size=(576, 3))
    observed = np.clip(np.rint(target.astype(np.float64) + bias[:, None, None, :]), 0, 255).astype(
        np.uint8
    )
    offsets, diagnostics = seam_graph_rgb_offsets(observed, SeamGraphConfig())
    expected = -bias + np.median(bias, axis=0, keepdims=True)
    corrected = apply_rgb_offsets(observed, offsets)
    assert offsets.shape == (576, 3)
    assert np.median(np.abs(offsets - expected)) < 0.8
    assert diagnostics["edge_count"] == 1104.0
    assert contest_ssim(smooth_target(), _merge(corrected)) > 0.995
    assert contest_ssim(smooth_target(), _merge(corrected)) > contest_ssim(
        smooth_target(), _merge(observed)
    )


def _merge(tiles: np.ndarray) -> np.ndarray:
    return tiles.reshape(24, 24, 20, 20, 3).transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


def test_bounded_luminance_gain_recovers_synthetic_gain() -> None:
    target_image = (smooth_target().astype(np.float64) * 0.65 + 40.0).clip(20, 220).astype(np.uint8)
    target = split_tiles(target_image)
    rng = np.random.default_rng(9)
    nuisance_gain = rng.uniform(0.97, 1.03, size=576)
    observed = np.clip(
        np.rint(target.astype(np.float64) * nuisance_gain[:, None, None, None]),
        0,
        255,
    ).astype(np.uint8)
    gains, diagnostics = seam_graph_luminance_gains(observed, LuminanceGainConfig())
    corrected = apply_luminance_gains(observed, gains)
    assert gains.shape == (576,)
    assert gains.min() >= 0.96 - 1e-6
    assert gains.max() <= 1.04 + 1e-6
    assert diagnostics["gain_min"] >= 0.96 - 1e-6
    assert contest_ssim(target_image, _merge(corrected)) > contest_ssim(
        target_image, _merge(observed)
    )


def test_public_inference_api_has_no_target_or_source_argument() -> None:
    for function in (seam_graph_rgb_offsets, seam_graph_luminance_gains):
        parameters = set(inspect.signature(function).parameters)
        assert parameters.isdisjoint({"target", "clean", "source", "source_name"})


def test_ported_configs_match_frozen_defaults_and_provenance() -> None:
    rgb = json.loads((ROOT / "configs/postassembly_rgb_offset_v1.json").read_text())
    luma = json.loads((ROOT / "configs/postassembly_luminance_gain_v1.json").read_text())
    assert rgb["method"] == {
        "extrapolation_band": 3,
        "confidence_scale": 12.0,
        "confidence_floor": 0.05,
        "ridge": 0.2,
        "huber_delta": 4.0,
        "irls_steps": 4,
        "max_abs_offset": 12.0,
        "global_gauge": "per-channel median offset equals zero",
    }
    assert luma["method"] == {
        "extrapolation_band": 3,
        "confidence_scale": 0.08,
        "confidence_floor": 0.05,
        "ridge": 0.5,
        "huber_delta": 0.025,
        "irls_steps": 4,
        "max_fractional_gain": 0.04,
        "luminance_floor": 12.0,
        "luminance_ceiling": 243.0,
        "global_gauge": "median log gain equals zero",
    }
    assert rgb["origin"] == luma["origin"]
    assert rgb["origin"]["source_blob"] == "9d8d01c0f48d0e1473c1ff48285b06ab786a5dd8"
