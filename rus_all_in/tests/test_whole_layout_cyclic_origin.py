from __future__ import annotations

import numpy as np
import pytest
import torch

from aiijc_puzzle.component_anchor_diagnostic import DecoderComponentBuild
from aiijc_puzzle.socket_matcher import SocketOutput
from aiijc_puzzle.whole_layout_cyclic_origin import (
    WholeLayoutCyclicOriginCNN,
    WholeLayoutOriginConfig,
    anchor_to_roll_logits,
    assemble_feature_grid,
    best_roll_nll,
    combine_tile_features,
    cyclic_exact_counts,
    parameter_count,
    select_learned_cyclic_origin,
    topk_hits_best_rolls,
    uniform_best_roll_nll,
)


def test_circular_cnn_is_shift_equivariant_and_has_no_position_parameters() -> None:
    torch.manual_seed(3)
    model = WholeLayoutCyclicOriginCNN(
        WholeLayoutOriginConfig(input_channels=7, width=16, dilations=(1, 2, 4))
    ).eval()
    value = torch.randn(2, 7, 9, 9)
    baseline_anchor = model.anchor_logits(value)
    shifted_anchor = model.anchor_logits(torch.roll(value, shifts=(2, -3), dims=(2, 3)))
    assert torch.allclose(
        shifted_anchor,
        torch.roll(baseline_anchor, shifts=(2, -3), dims=(1, 2)),
        atol=2e-6,
        rtol=1e-5,
    )
    assert parameter_count(model) < 100_000
    assert not any(
        "position" in name or "embedding" in name
        for name, _ in model.named_parameters()
    )


def test_anchor_roll_mapping_and_strict_selector_use_numpy_roll_convention() -> None:
    grid = 4
    anchor = torch.full((1, grid, grid), -10.0)
    anchor[0, 1, 3] = 5.0
    roll = anchor_to_roll_logits(anchor)
    assert divmod(int(roll.flatten().argmax()), grid) == (3, 1)
    layout = np.arange(grid * grid, dtype=np.int32)
    result = select_learned_cyclic_origin(layout, roll[0], grid=grid)
    expected = np.roll(layout.reshape(grid, grid), shift=(3, 1), axis=(0, 1)).reshape(-1)
    assert result.selected_roll == (3, 1)
    assert result.strict_permutation
    assert np.array_equal(result.layout, expected)
    with pytest.raises(ValueError, match="strict tile permutation"):
        select_learned_cyclic_origin(np.zeros(grid * grid), roll[0], grid=grid)


def test_exact_roll_targets_support_ties_nll_and_topk() -> None:
    grid = 3
    reference = np.arange(grid * grid, dtype=np.int32)
    predicted = np.roll(reference.reshape(grid, grid), shift=(1, 2), axis=(0, 1)).reshape(-1)
    counts = cyclic_exact_counts(predicted, reference, grid=grid)
    assert counts.max() == grid * grid
    assert divmod(int(counts.argmax()), grid) == (2, 1)
    logits = torch.zeros((1, grid, grid))
    logits[0, 2, 1] = 8.0
    loss = best_roll_nll(logits, torch.from_numpy(counts).unsqueeze(0))
    assert loss.item() < 0.01
    hits = topk_hits_best_rolls(logits[0].numpy(), counts)
    assert hits == {1: True, 5: True}
    assert uniform_best_roll_nll(counts) == pytest.approx(np.log(grid * grid))


def test_feature_grid_is_invariant_to_consistent_tile_relabelling() -> None:
    grid = 3
    count = grid * grid
    generator = np.random.default_rng(5)
    features = generator.normal(size=(count, 6)).astype(np.float32)
    layout = generator.permutation(count)
    baseline = assemble_feature_grid(features, layout, grid=grid)
    old_to_new = generator.permutation(count)
    relabelled_features = np.empty_like(features)
    relabelled_features[old_to_new] = features
    relabelled_layout = old_to_new[layout]
    assert np.array_equal(
        baseline,
        assemble_feature_grid(relabelled_features, relabelled_layout, grid=grid),
    )


def test_combined_features_are_target_free_finite_and_have_no_tile_id_channel() -> None:
    grid = 3
    count = grid * grid
    generator = np.random.default_rng(7)
    tiles = generator.integers(0, 256, size=(count, 20, 20, 3), dtype=np.uint8)
    context = torch.randn(1, count, 8)
    assignment = torch.randn(1, count + 1, count + 1)
    border = torch.randn(1, count)
    output = SocketOutput(
        right_raw=assignment[:, :count, :count],
        down_raw=assignment[:, :count, :count],
        right_log_assignment=assignment,
        down_log_assignment=assignment + 0.1,
        right_out_border_logits=border,
        left_in_border_logits=border + 0.1,
        bottom_out_border_logits=border + 0.2,
        top_in_border_logits=border + 0.3,
    )
    build = DecoderComponentBuild(
        components=tuple({tile: (0, 0)} for tile in range(count)),
        constraints=(),
        status_counts={},
    )
    features, names = combine_tile_features(
        tiles,
        context,
        output,
        build,
        grid=grid,
    )
    assert features.shape == (count, len(names))
    assert np.isfinite(features).all()
    assert all("tile_id" not in name and "absolute" not in name for name in names)
