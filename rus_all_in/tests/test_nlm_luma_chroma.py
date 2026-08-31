from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.nlm_luma_chroma import (
    NLMArm,
    apply_nlm_luma_chroma,
    paired_t_interval,
    safety_summary,
    structure_diagnostics,
)


def _image() -> np.ndarray:
    axis = np.arange(480, dtype=np.uint16)
    grid = (axis[:, None] + axis[None, :]) % 256
    return np.stack((grid, np.roll(grid, 3, axis=1), np.roll(grid, 7, axis=0)), axis=2).astype(
        np.uint8
    )


def test_arm_rejects_forbidden_strengths() -> None:
    with pytest.raises(ValueError):
        NLMArm("bad", 30, 20, "candidate")
    with pytest.raises(ValueError):
        NLMArm("bad", 20, 0, "candidate")
    with pytest.raises(ValueError):
        NLMArm("bad", 20, 20, "unknown")


def test_decoupled_nlm_returns_strict_rgb_and_is_deterministic() -> None:
    image = _image()
    first = apply_nlm_luma_chroma(image, h=3, h_color=5)
    second = apply_nlm_luma_chroma(image, h=3, h_color=5)
    assert first.shape == (480, 480, 3)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)


def test_structure_diagnostics_and_identity_safety() -> None:
    metrics = structure_diagnostics(_image())
    assert metrics["within_tile_luminance_gradient"] > 0
    assert metrics["within_tile_chroma_gradient"] > 0
    summary = safety_summary([metrics, metrics], [metrics, metrics])
    assert all(value == pytest.approx(1.0) for value in summary.values())


def test_paired_t_interval_contains_constant_mean_exactly() -> None:
    result = paired_t_interval([0.01] * 8)
    assert result["mean"] == pytest.approx(0.01)
    assert result["lower"] == pytest.approx(0.01)
    assert result["upper"] == pytest.approx(0.01)
