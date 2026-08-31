from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from aiijc_puzzle.fullres_boundary_denoiser import (
    FullResolutionBoundaryDenoiser,
    FullResolutionDenoiserConfig,
    FullResolutionNAFBlock,
    boundary_denoising_loss,
    boundary_mask,
    restore_matcher_view,
)


def test_every_learned_block_preserves_twenty_by_twenty() -> None:
    model = FullResolutionBoundaryDenoiser()
    observed: list[tuple[int, int]] = []
    hooks = [
        block.register_forward_hook(
            lambda _module, _inputs, output: observed.append(tuple(output.shape[-2:]))
        )
        for block in model.modules()
        if isinstance(block, FullResolutionNAFBlock)
    ]
    value = torch.rand(3, 3, 20, 20)
    assert model(value).shape == value.shape
    for hook in hooks:
        hook.remove()
    assert observed == [(20, 20)] * model.config.blocks
    assert not any(
        isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.ConvTranspose2d))
        for module in model.modules()
    )
    assert all(
        tuple(module.stride) == (1, 1)
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    )


def test_zero_initialisation_is_exact_identity() -> None:
    model = FullResolutionBoundaryDenoiser()
    dirty = torch.rand(4, 3, 20, 20)
    assert torch.equal(model(dirty), dirty)


def test_boundary_loss_responds_to_boundary_not_untouched_interior() -> None:
    dirty = torch.full((1, 3, 20, 20), 0.5)
    clean = dirty.clone()
    clean[..., 0, :] = 0.9
    identity_loss, identity_terms = boundary_denoising_loss(dirty, clean, dirty)
    fixed = dirty.clone()
    fixed[..., 0, :] = 0.9
    fixed_loss, fixed_terms = boundary_denoising_loss(fixed, clean, dirty)
    assert fixed_loss < identity_loss
    assert fixed_terms["border"] < identity_terms["border"]

    interior_only = dirty.clone()
    interior_only[..., 9:11, 9:11] = 0.0
    interior_loss, interior_terms = boundary_denoising_loss(dirty, interior_only, dirty)
    assert interior_terms["border"].item() == pytest.approx(0.0)
    assert interior_terms["gradient"].item() == pytest.approx(0.0)
    assert interior_loss.item() == pytest.approx(0.0)


def test_boundary_mask_has_requested_perimeter_only() -> None:
    mask = boundary_mask(width=3)[0, 0]
    assert bool(mask[0, 10])
    assert bool(mask[10, 0])
    assert not bool(mask[3, 3])
    assert not bool(mask[10, 10])


def test_matcher_view_preserves_roster_shape_but_is_not_canvas_api() -> None:
    model = FullResolutionBoundaryDenoiser(
        FullResolutionDenoiserConfig(width=8, blocks=1)
    )
    raw = np.random.default_rng(4).integers(
        0,
        256,
        size=(7, 20, 20, 3),
        dtype=np.uint8,
    )
    restored = restore_matcher_view(
        model,
        raw,
        device=torch.device("cpu"),
        batch_size=3,
    )
    assert restored.shape == raw.shape
    assert restored.dtype == np.uint8
    assert np.array_equal(restored, raw)
    assert not hasattr(model, "assemble")
    assert not hasattr(model, "layout")


def test_boundary_loss_backpropagates_into_non_identity_output() -> None:
    model = FullResolutionBoundaryDenoiser(
        FullResolutionDenoiserConfig(width=8, blocks=1)
    )
    dirty = torch.rand(2, 3, 20, 20)
    clean = torch.rand(2, 3, 20, 20)
    prediction = model(dirty)
    loss, _ = boundary_denoising_loss(prediction, clean, dirty)
    loss.backward()
    assert model.ending.weight.grad is not None
    assert torch.count_nonzero(model.ending.weight.grad).item() > 0
