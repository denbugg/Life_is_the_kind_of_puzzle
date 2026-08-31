from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.frame_side_origin import (
    FrameSideClassifier,
    FrameSideConfig,
    frame_side_loss,
    frame_side_targets,
    frame_topk_metrics,
    select_frame_cyclic_translation,
    top_frame_sets,
)


def _model() -> FrameSideClassifier:
    return FrameSideClassifier(
        FrameSideConfig(width=16, blocks=2, sequence_dilations=(1, 2))
    )


def test_full_resolution_field_and_tile_permutation_equivariance() -> None:
    torch.manual_seed(1)
    model = _model().eval()
    raw = torch.rand(1, 16, 3, 20, 20)
    restored = torch.rand_like(raw)
    context = torch.rand(1, 16, 64)
    border = torch.rand(1, 16, 4)
    shapes: list[tuple[int, ...]] = []
    hooks = [
        block.register_forward_hook(lambda _module, _input, output: shapes.append(output.shape))
        for block in model.blocks
    ]
    with torch.no_grad():
        first = model(raw, restored, context, border)
        permutation = torch.randperm(16)
        second = model(
            raw[:, permutation],
            restored[:, permutation],
            context[:, permutation],
            border[:, permutation],
        )
    for hook in hooks:
        hook.remove()
    assert first.shape == (1, 16, 4)
    torch.testing.assert_close(second, first[:, permutation], atol=1e-5, rtol=1e-5)
    assert shapes and all(shape[-2:] == (20, 20) for shape in shapes)


def test_exact_frame_targets_loss_and_stable_top_sets() -> None:
    grid = 4
    positions = torch.arange(grid * grid).unsqueeze(0)
    targets = frame_side_targets(positions, grid=grid)
    assert torch.all(targets.sum(dim=1) == grid)
    logits = torch.where(targets, torch.tensor(4.0), torch.tensor(-4.0)).requires_grad_()
    loss, diagnostics = frame_side_loss(logits, targets)
    loss.backward()
    predicted = top_frame_sets(logits.detach()[0], grid=grid)
    metrics = frame_topk_metrics(predicted, positions.numpy()[0], grid=grid)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert diagnostics["loss"] > 0
    assert metrics["macro_f1"] == 1.0


def test_frame_placer_recovers_known_cyclic_origin_and_is_strict() -> None:
    grid = 4
    count = grid * grid
    identity = np.arange(count, dtype=np.int32)
    shifted = np.roll(
        identity.reshape(grid, grid), shift=(1, 2), axis=(0, 1)
    ).reshape(-1)
    positions = np.arange(count)
    row, column = divmod(positions, grid)
    sets = np.stack(
        (
            np.flatnonzero(row == 0),
            np.flatnonzero(row == grid - 1),
            np.flatnonzero(column == 0),
            np.flatnonzero(column == grid - 1),
        )
    )
    assignments = np.zeros((count + 1, count + 1), dtype=np.float64)
    result = select_frame_cyclic_translation(
        shifted,
        sets,
        assignments,
        assignments,
        grid=grid,
    )
    np.testing.assert_array_equal(result.layout, identity)
    assert result.diagnostics.selected_frame_hits == 4 * grid
    assert result.diagnostics.strict_permutation
