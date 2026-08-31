from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from aiijc_puzzle.fullres_twin_side_matcher import (
    FullResolutionFieldBlock,
    FullResolutionTwinSideMatcher,
    directional_neighbour_targets,
    directional_score_matrices,
    dual_corruption_retrieval_loss,
    twin_right_down_scores,
)


def test_field_and_sequence_keep_full_ordered_resolution() -> None:
    model = FullResolutionTwinSideMatcher(
        dimension=16,
        field_blocks=2,
        sequence_blocks=1,
    )
    observed: list[tuple[int, int]] = []
    hooks = [
        module.register_forward_hook(
            lambda _module, _inputs, output: observed.append(tuple(output.shape[-2:]))
        )
        for module in model.modules()
        if isinstance(module, FullResolutionFieldBlock)
    ]
    output = model(torch.rand(1, 9, 3, 20, 20))
    for hook in hooks:
        hook.remove()
    assert output.field.shape == (1, 9, 16, 20, 20)
    assert output.sides.shape == (1, 9, 4, 20, 16)
    assert output.scores.shape == (1, 4, 9, 9)
    assert observed == [(20, 20), (20, 20)]
    assert not any(
        isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.ConvTranspose2d))
        for module in model.modules()
    )
    assert all(
        tuple(module.stride) == (1, 1)
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    )


def test_ordered_score_is_not_a_pooled_side_vector() -> None:
    sides = torch.zeros(1, 3, 4, 20, 4)
    pattern = torch.nn.functional.one_hot(torch.arange(20) % 4, 4).float()
    sides[0, 0, 1] = pattern
    sides[0, 1, 0] = pattern
    sides[0, 2, 0] = pattern.flip(0)
    scores = directional_score_matrices(sides, horizontal_log_scale=0.0)
    assert scores[0, 1, 0, 1] > scores[0, 1, 0, 2] + 0.5
    # Both candidates have exactly the same pooled vector; only position order differs.
    torch.testing.assert_close(sides[0, 1, 0].mean(0), sides[0, 2, 0].mean(0))


def test_matcher_scores_are_tile_permutation_equivariant() -> None:
    torch.manual_seed(4)
    model = FullResolutionTwinSideMatcher(
        dimension=16,
        field_blocks=1,
        sequence_blocks=1,
    ).eval()
    tiles = torch.rand(1, 9, 3, 20, 20)
    permutation = torch.tensor([4, 1, 8, 0, 6, 2, 7, 3, 5])
    with torch.no_grad():
        reference = model(tiles).scores
        observed = model(tiles[:, permutation]).scores
    expected = reference[:, :, permutation][:, :, :, permutation]
    torch.testing.assert_close(observed, expected, atol=3e-6, rtol=3e-6)


def test_arbitrary_shuffle_targets_are_exact_and_border_masked() -> None:
    layout = torch.tensor([[5, 0, 8, 2, 6, 1, 4, 7, 3]])
    targets = directional_neighbour_targets(layout, grid=3)
    assert targets.shape == (1, 4, 9)
    # At position zero tile 5 has tile 0 on its right and tile 2 below.
    assert targets[0, 1, 5].item() == 0
    assert targets[0, 3, 5].item() == 2
    # At top-right position tile 8 has no right neighbour.
    assert targets[0, 1, 8].item() == -1


def test_dual_corruption_objective_has_finite_nonzero_gradient() -> None:
    torch.manual_seed(9)
    model = FullResolutionTwinSideMatcher(
        dimension=16,
        field_blocks=1,
        sequence_blocks=1,
    )
    first_tiles = torch.rand(1, 16, 3, 20, 20)
    second_tiles = (first_tiles + 0.08 * torch.randn_like(first_tiles)).clamp(0, 1)
    first = model(first_tiles)
    second = model(second_tiles)
    layout = torch.randperm(16).unsqueeze(0)
    loss, terms = dual_corruption_retrieval_loss(
        model,
        first,
        second,
        layout,
        grid=4,
    )
    assert torch.isfinite(loss)
    assert terms["cross_entropy"] > 0
    assert terms["consistency"] >= 0
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert sum(int(torch.count_nonzero(value)) for value in gradients) > 0


def test_inference_returns_matcher_scores_without_pixel_or_layout_api() -> None:
    model = FullResolutionTwinSideMatcher(
        dimension=16,
        field_blocks=1,
        sequence_blocks=1,
    ).eval()
    tiles = np.random.default_rng(7).integers(
        0,
        256,
        size=(9, 20, 20, 3),
        dtype=np.uint8,
    )
    right, down = twin_right_down_scores(model, tiles, device=torch.device("cpu"))
    assert right.shape == down.shape == (9, 9)
    assert np.isfinite(right).all() and np.isfinite(down).all()
    assert not hasattr(model, "restore")
    assert not hasattr(model, "assemble")
    assert not hasattr(model, "layout")


def test_rejects_non_permutation_targets() -> None:
    with pytest.raises(ValueError, match="strict permutation"):
        directional_neighbour_targets(torch.zeros(1, 16), grid=4)
