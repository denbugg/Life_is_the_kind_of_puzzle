from __future__ import annotations

import numpy as np
import pytest
import torch

from aiijc_puzzle.corruption_border_encoder import (
    CORRUPTION_MODES,
    CorruptionAwareBorderEncoder,
    canonical_borders,
    corrupt_e13_tiles,
    corruption_aware_training_loss,
    directional_score_matrices,
    e13_curriculum_severity,
)


def test_canonical_borders_preserve_tangent_and_point_edge_inward() -> None:
    row = torch.arange(20).float()[:, None]
    column = torch.arange(20).float()[None, :]
    image = (row * 20 + column) / 399.0
    tiles = image.expand(1, 1, 3, -1, -1).contiguous()
    borders = canonical_borders(tiles, border=4)
    normalised = 2.0 * image - 1.0
    assert borders.shape == (1, 1, 4, 3, 4, 20)
    torch.testing.assert_close(borders[0, 0, 0, 0], normalised[:, :4].T)
    torch.testing.assert_close(borders[0, 0, 1, 0], normalised[:, -4:].flip(1).T)
    torch.testing.assert_close(borders[0, 0, 2, 0], normalised[:4])
    torch.testing.assert_close(borders[0, 0, 3, 0], normalised[-4:].flip(0))


def test_encoder_and_scores_are_tile_permutation_equivariant() -> None:
    torch.manual_seed(5)
    model = CorruptionAwareBorderEncoder(dimension=16).eval()
    tiles = torch.rand(1, 9, 3, 20, 20)
    permutation = torch.tensor([4, 1, 8, 0, 6, 2, 7, 3, 5])
    with torch.no_grad():
        reference = directional_score_matrices(model(tiles))
        observed = directional_score_matrices(model(tiles[:, permutation]))
    expected = reference[:, :, permutation][:, :, :, permutation]
    torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-6)


def test_historical_clean_corrupt_loss_has_finite_gradient() -> None:
    torch.manual_seed(7)
    model = CorruptionAwareBorderEncoder(dimension=16)
    clean = torch.rand(1, 9, 3, 20, 20)
    corrupt = (clean + 0.1 * torch.randn_like(clean)).clamp(0, 1)
    clean_sides = model(clean)
    corrupt_sides = model(corrupt)
    loss, diagnostics = corruption_aware_training_loss(
        clean_sides,
        corrupt_sides,
        grid=3,
    )
    assert torch.isfinite(loss)
    assert diagnostics["corrupt_retrieval_loss"] > 0
    assert 0 <= diagnostics["corrupt_r1"] <= 1
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)


def test_curriculum_reaches_exactly_one_without_overshoot() -> None:
    values = [e13_curriculum_severity(step, 400) for step in range(400)]
    assert values[0] == pytest.approx(0.2)
    assert values[-1] == 1.0
    assert all(
        left <= right <= 1.0 for left, right in zip(values, values[1:], strict=False)
    )


@pytest.mark.parametrize("mode", CORRUPTION_MODES)
def test_historical_corruption_is_deterministic_and_keeps_raw_tiles(mode: str) -> None:
    clean = np.arange(16 * 20 * 20 * 3, dtype=np.uint32).reshape(16, 20, 20, 3)
    clean = np.asarray(clean % 256, dtype=np.uint8)
    pristine = clean.copy()
    first = corrupt_e13_tiles(
        clean,
        np.random.default_rng(11),
        severity=0.8,
        mode=mode,
    )
    second = corrupt_e13_tiles(
        clean,
        np.random.default_rng(11),
        severity=0.8,
        mode=mode,
    )
    assert first.shape == clean.shape and first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.array_equal(clean, pristine)
    assert not np.array_equal(first, clean)
