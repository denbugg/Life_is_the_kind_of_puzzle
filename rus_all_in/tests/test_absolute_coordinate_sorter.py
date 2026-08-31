from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.absolute_coordinate_sorter import (
    AbsoluteCoordinateSorter,
    component_translation_loss,
    coordinate_sorting_loss,
    decode_coordinate_logits,
    square_log_sinkhorn,
    train_consistent_component_unary,
    truth_consistent_component_targets,
)
from aiijc_puzzle.socket_matcher import SocketMatcher


def _model(*, grid: int = 3) -> AbsoluteCoordinateSorter:
    backbone = SocketMatcher(
        dimension=8,
        heads=2,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=3,
    )
    return AbsoluteCoordinateSorter(
        backbone,
        grid=grid,
        head_dimension=8,
        heads=2,
        set_layers=1,
        sinkhorn_iterations=6,
    )


def test_square_sinkhorn_is_doubly_stochastic() -> None:
    torch.manual_seed(71)
    assignment = square_log_sinkhorn(torch.randn(2, 9, 9), iterations=100).exp()
    assert torch.allclose(assignment.sum(2), torch.ones(2, 9), atol=2e-5)
    assert torch.allclose(assignment.sum(1), torch.ones(2, 9), atol=2e-5)


def test_absolute_coordinate_model_is_input_permutation_equivariant() -> None:
    torch.manual_seed(73)
    model = _model().eval()
    tiles = torch.rand(1, 9, 3, 20, 20)
    permutation = torch.tensor([5, 0, 8, 2, 7, 3, 1, 6, 4])
    with torch.no_grad():
        reference = model(tiles)
        shuffled = model(tiles[:, permutation])
        reference_tokens, _ = model.encode_coordinate_tokens(tiles)
        shuffled_tokens, _ = model.encode_coordinate_tokens(tiles[:, permutation])
    torch.testing.assert_close(
        shuffled_tokens,
        reference_tokens[:, permutation],
        atol=3e-5,
        rtol=3e-5,
    )
    for field in ("row_logits", "column_logits", "slot_logits", "slot_log_assignment"):
        torch.testing.assert_close(
            getattr(shuffled, field),
            getattr(reference, field)[:, permutation],
            atol=3e-5,
            rtol=3e-5,
        )


def test_coordinate_loss_has_finite_head_gradient_and_frozen_backbone() -> None:
    torch.manual_seed(79)
    model = _model()
    tiles = torch.rand(1, 9, 3, 20, 20)
    target = torch.randperm(9).reshape(1, 9)
    output = model(tiles)
    loss, diagnostics = coordinate_sorting_loss(output, target, grid=3)
    assert torch.isfinite(loss)
    assert 0 <= diagnostics["row_argmax_accuracy"] <= 1
    assert 0 <= diagnostics["column_argmax_accuracy"] <= 1
    loss.backward()
    head_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.grad is not None
    ]
    assert head_gradients and all(torch.isfinite(value).all() for value in head_gradients)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_hungarian_decoder_returns_exact_tile_at_position_permutation() -> None:
    target_position = np.array([3, 0, 8, 2, 6, 1, 5, 7, 4])
    scores = np.full((9, 9), -5.0)
    scores[np.arange(9), target_position] = 5.0
    layout = decode_coordinate_logits(scores)
    expected = np.argsort(target_position).astype(np.int32)
    assert np.array_equal(layout, expected)
    assert np.array_equal(np.sort(layout), np.arange(9))


def test_component_translation_targets_skip_false_bridges_and_train_exact_shift() -> None:
    components = (
        {0: (0, 0), 1: (0, 1)},
        {3: (0, 0), 4: (1, 0)},  # False relative vertical edge under identity truth.
        {3: (0, 0), 6: (1, 0)},
        {8: (0, 0)},  # Singleton deliberately excluded.
    )
    targets = truth_consistent_component_targets(
        components,
        np.arange(9),
        grid=3,
    )
    assert len(targets) == 2
    assert targets[0].tiles == (0, 1)
    assert (targets[0].target_row_shift, targets[0].target_column_shift) == (0, 0)
    assert targets[1].tiles == (3, 6)
    assert (targets[1].target_row_shift, targets[1].target_column_shift) == (1, 0)

    logits = torch.zeros(1, 9, 9, requires_grad=True)
    with torch.no_grad():
        logits[0, torch.arange(9), torch.arange(9)] = 5.0
    loss, diagnostics = component_translation_loss(logits, targets)
    assert float(loss.detach()) < 0.1
    assert diagnostics["supervised_component_count"] == 2
    assert diagnostics["supervised_component_tiles"] == 4
    assert diagnostics["component_translation_shift_top1_accuracy"] == 1.0
    assert (
        diagnostics["component_translation_shift_top1_accuracy"]
        > diagnostics["component_translation_shift_chance_accuracy"]
    )
    assert diagnostics["component_translation_nll_ratio_to_uniform"] < 0.1
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_train_consistent_unary_preserves_component_shift_argmax() -> None:
    generator = np.random.default_rng(83)
    logits = generator.normal(size=(9, 9))
    normalised = train_consistent_component_unary(logits)
    target = truth_consistent_component_targets(
        ({0: (0, 0), 1: (0, 1), 3: (1, 0)},),
        np.arange(9),
        grid=3,
    )
    assert len(target) == 1
    _, raw = component_translation_loss(
        torch.tensor(logits, dtype=torch.float32).unsqueeze(0),
        target,
    )
    _, scaled = component_translation_loss(
        torch.tensor(normalised, dtype=torch.float32).unsqueeze(0),
        target,
    )
    assert (
        raw["component_translation_shift_top1_accuracy"]
        == scaled["component_translation_shift_top1_accuracy"]
    )
    np.testing.assert_allclose(normalised.mean(axis=1), 0.0, atol=1e-12)
    assert np.isclose(normalised.std(), 1.0)
