from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.component_absolute_placer import (
    ComponentAbsoluteConfig,
    ComponentAbsolutePlacerModel,
    align_components_across_corruptions,
    average_precision,
    component_absolute_loss,
    component_absolute_targets,
    paired_component_consistency_loss,
    place_one_component_anchor,
    render_native_component_mosaic,
)


def test_native_component_mosaic_preserves_exact_pixels_and_holes() -> None:
    raw = torch.rand(16, 3, 20, 20)
    component = {2: (0, 0), 7: (1, 1)}
    mosaic = render_native_component_mosaic(raw, component)
    assert mosaic.shape == (7, 40, 40)
    assert torch.equal(mosaic[:3, :20, :20], raw[2])
    assert torch.equal(mosaic[:3, 20:, 20:], raw[7])
    assert torch.count_nonzero(mosaic[:, :20, 20:]) == 0
    assert torch.all(mosaic[6, :20, :20] == 1)
    assert torch.all(mosaic[6, 20:, 20:] == 1)


def test_component_model_is_equivariant_to_component_enumeration() -> None:
    torch.manual_seed(4)
    config = ComponentAbsoluteConfig(
        grid=4,
        pixel_width=16,
        pixel_blocks=1,
        lattice_blocks=1,
        model_dimension=32,
        set_layers=1,
        set_heads=4,
    )
    model = ComponentAbsolutePlacerModel(config).eval()
    raw = torch.rand(16, 3, 20, 20)
    components = (
        {0: (0, 0), 1: (0, 1)},
        {4: (0, 0), 8: (1, 0)},
        {10: (0, 0), 11: (0, 1), 15: (1, 1)},
    )
    mosaics = [render_native_component_mosaic(raw, item) for item in components]
    geometry = torch.zeros(3, 12)
    geometry[:, 2:4] = torch.tensor([[0.25, 0.50], [0.50, 0.25], [0.50, 0.50]])
    with torch.inference_mode():
        purity, offset = model(mosaics, geometry)
        order = torch.tensor([2, 0, 1])
        shuffled_purity, shuffled_offset = model(
            [mosaics[index] for index in order.tolist()],
            geometry[order],
        )
    assert torch.allclose(shuffled_purity, purity[order], atol=1e-5)
    assert torch.allclose(shuffled_offset, offset[order], atol=1e-5)
    assert torch.isfinite(offset[0]).sum() == 12
    assert torch.isfinite(offset[2]).sum() == 9


def test_exact_component_targets_and_loss_ignore_impure_offsets() -> None:
    components = (
        {0: (0, 0), 1: (0, 1)},
        {4: (0, 0), 6: (0, 1)},
    )
    purity, offsets, support = component_absolute_targets(
        components,
        np.arange(16),
        grid=4,
    )
    assert purity.tolist() == [True, False]
    assert offsets.tolist() == [0, -1]
    assert torch.allclose(support, torch.tensor([1.0, 0.5]))
    purity_logits = torch.tensor([2.0, -2.0], requires_grad=True)
    offset_logits = torch.zeros(2, 16, requires_grad=True)
    loss, diagnostics = component_absolute_loss(
        purity_logits,
        offset_logits,
        purity,
        offsets,
        component_sizes=torch.tensor([2.0, 2.0]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert diagnostics["pure_fraction"] == 0.5
    assert offset_logits.grad is not None
    assert torch.count_nonzero(offset_logits.grad[1]) == 0


def test_one_anchor_packer_is_strict_and_places_anchor_exactly() -> None:
    components = (
        {0: (0, 0), 1: (0, 1)},
        {2: (0, 0), 3: (0, 1)},
        *({tile: (0, 0)} for tile in range(4, 16)),
    )
    baseline = np.arange(16, dtype=np.int32)
    layout, diagnostics = place_one_component_anchor(
        components,
        baseline,
        anchor_component_index=0,
        anchor_offset=14,
        grid=4,
    )
    assert np.array_equal(np.sort(layout), np.arange(16))
    assert layout.reshape(4, 4)[3, 2:].tolist() == [0, 1]
    assert diagnostics.strict_permutation
    assert diagnostics.anchor_size == 2


def test_average_precision_has_stable_exact_values() -> None:
    assert average_precision([1, 0, 1], [3.0, 2.0, 1.0]) == (1.0 + 2 / 3) / 2
    assert average_precision([0, 0], [1.0, 0.0]) == 0.0


def test_train_only_paired_alignment_uses_true_members_and_geometry() -> None:
    first_components = (
        {0: (0, 0), 1: (0, 1)},
        {4: (0, 0), 8: (1, 0)},
    )
    second_components = (
        {2: (0, 0), 6: (1, 0)},
        {10: (0, 0), 11: (0, 1)},
    )
    first_positions = np.arange(16)
    second_positions = (np.arange(16) + 6) % 16
    first, second = align_components_across_corruptions(
        first_components,
        first_positions,
        second_components,
        second_positions,
        grid=4,
    )
    assert first.tolist() == [0]
    assert second.tolist() == [1]
    first_purity = torch.tensor([0.5, -0.5])
    second_purity = torch.tensor([-0.5, 0.5])
    first_offsets = torch.zeros(2, 16)
    second_offsets = torch.zeros(2, 16)
    loss, diagnostics = paired_component_consistency_loss(
        first_purity,
        first_offsets,
        second_purity,
        second_offsets,
        first,
        second,
    )
    assert loss == 0
    assert diagnostics["aligned_components"] == 1
