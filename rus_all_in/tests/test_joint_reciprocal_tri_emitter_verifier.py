from __future__ import annotations

import math

import numpy as np
import torch

from aiijc_puzzle.joint_reciprocal_tri_emitter_verifier import (
    CONFIDENCE_BCE_WEIGHT,
    DELTA_REGULARIZATION_WEIGHT,
    SOFTMIN_TAU,
    JointReciprocalTriEmitterVerifier,
    build_joint_axis_output,
    dense_two_sided_confidence,
    exact_joint_targets,
    fixed_fraction_reciprocal_head,
    joint_assignment_loss,
)
from scripts.run_joint_reciprocal_tri_emitter_capacity import (
    make_collision_capacity_case,
)


def _small_roster() -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.tensor(
        [
            [1, 2, 3],
            [2, 3, 0],
            [3, 1, 0],
            [0, 1, 2],
        ],
        dtype=torch.long,
    )
    return candidates, torch.ones_like(candidates, dtype=torch.bool)


def test_exact_joint_targets_use_none_for_border_and_absent_truth() -> None:
    candidates, valid = _small_roster()
    # Source 2's exact target 2 is absent (and self); source 3 is a border.
    targets = exact_joint_targets(candidates, valid, torch.tensor([1, 2, 2, -1]))
    assert targets.row_slots.tolist() == [0, 0, -1, -1]
    assert targets.column_sources.tolist() == [-1, 0, 1, -1]
    assert targets.edge_truth.sum().item() == 2


def test_joint_objective_is_exact_fixed_weight_sum_and_backpropagates() -> None:
    candidates, valid = _small_roster()
    edge_logits = torch.nn.Parameter(
        torch.tensor(
            [
                [1.0, -0.5, -1.0],
                [0.8, -0.2, -0.7],
                [0.1, 0.2, -0.4],
                [0.3, 0.0, -0.1],
            ]
        )
    )
    delta = edge_logits * 0.2
    row_none = torch.nn.Parameter(torch.full((4,), 0.1))
    column_none = torch.nn.Parameter(torch.full((4,), -0.1))
    bias = torch.nn.Parameter(torch.zeros(()))
    temperature = torch.nn.Parameter(torch.ones(()))
    output = build_joint_axis_output(
        candidates,
        valid,
        edge_logits,
        delta,
        row_none,
        column_none,
        bias,
        temperature,
    )
    targets = exact_joint_targets(candidates, valid, torch.tensor([1, 2, -1, -1]))
    loss = joint_assignment_loss(output, targets, valid)
    expected = (
        loss.row_cross_entropy
        + loss.column_cross_entropy
        + CONFIDENCE_BCE_WEIGHT * loss.confidence_bce
        + DELTA_REGULARIZATION_WEIGHT * loss.delta_regularization
    )
    torch.testing.assert_close(loss.total, expected)
    loss.total.backward()
    for parameter in (edge_logits, row_none, column_none, bias, temperature):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_padded_slots_keep_every_model_parameter_gradient_finite() -> None:
    torch.manual_seed(7)
    candidates = torch.tensor(
        [
            [1, 2, -1, -1],
            [2, 3, 0, -1],
            [3, 0, -1, -1],
            [0, 1, 2, -1],
        ],
        dtype=torch.long,
    )
    valid = candidates >= 0
    model = JointReciprocalTriEmitterVerifier(
        dino_dim=3,
        width=4,
        hidden=8,
    )
    output = model(
        torch.randn(4, 4, 5, 6),
        torch.randn(4, 4, 5, 3),
        candidates,
        valid,
        torch.randn(4, 4, 19),
        torch.randn(4, 4),
        direction=0,
    )
    targets = exact_joint_targets(candidates, valid, torch.tensor([1, 2, 3, -1]))
    loss = joint_assignment_loss(output, targets, valid)

    assert torch.isfinite(loss.total)
    assert torch.isneginf(output.calibrated_confidence_logits[~valid]).all()
    loss.total.backward()

    missing_or_nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all()
    ]
    assert missing_or_nonfinite == []
    assert torch.isfinite(model.raw_confidence_temperature.grad)


def test_two_sided_confidence_is_transpose_and_relabel_equivariant() -> None:
    generator = np.random.default_rng(21)
    logits = torch.from_numpy((20 * generator.normal(size=(7, 7))).astype(np.float32))
    valid = torch.from_numpy(generator.random((7, 7)) > 0.35)
    valid.fill_diagonal_(False)
    row_none = torch.from_numpy(generator.normal(size=7).astype(np.float32))
    column_none = torch.from_numpy(generator.normal(size=7).astype(np.float32))
    _, _, confidence = dense_two_sided_confidence(
        logits, valid, row_none, column_none, tau=SOFTMIN_TAU
    )
    _, _, transposed = dense_two_sided_confidence(
        logits.T,
        valid.T,
        column_none,
        row_none,
        tau=SOFTMIN_TAU,
    )
    torch.testing.assert_close(confidence[valid], transposed.T[valid])

    order = torch.tensor([5, 2, 0, 6, 1, 4, 3])
    _, _, relabelled = dense_two_sided_confidence(
        logits[order][:, order],
        valid[order][:, order],
        row_none[order],
        column_none[order],
        tau=SOFTMIN_TAU,
    )
    expected = confidence[order][:, order]
    mask = valid[order][:, order]
    torch.testing.assert_close(relabelled[mask], expected[mask])


def test_fixed_five_percent_head_is_exact_per_axis_and_deterministic() -> None:
    count = 20
    candidates = np.empty((count, 2), dtype=np.int64)
    for source in range(count):
        candidates[source] = ((source + 1) % count, (source + 2) % count)
    valid = np.ones_like(candidates, dtype=bool)
    logits = torch.tensor(np.tile([4.0, -1.0], (count, 1)), dtype=torch.float32)
    output = build_joint_axis_output(
        torch.from_numpy(candidates),
        torch.from_numpy(valid),
        logits,
        torch.zeros_like(logits),
        torch.full((count,), -2.0),
        torch.full((count,), -2.0),
        torch.zeros(()),
        torch.ones(()),
    )
    first = fixed_fraction_reciprocal_head(output, candidates, valid)
    second = fixed_fraction_reciprocal_head(output, candidates, valid)
    assert first.requested_count == math.ceil(0.05 * count) == 1
    assert first.selected.sum() == first.requested_count
    assert first.reciprocal.sum() == count
    np.testing.assert_array_equal(first.selected, second.selected)
    np.testing.assert_array_equal(first.targets, (first.sources + 1) % count)


def test_signed_capacity_generator_contains_many_to_one_collisions() -> None:
    case = make_collision_capacity_case(seed=20260916)
    assert case["candidates"].shape == (2, 16, 8)
    assert case["hard_collision"].sum(axis=(1, 2)).min() >= 12
    for axis, target in enumerate(case["collision_targets"]):
        sources = np.argwhere(
            case["hard_collision"][axis]
            & (case["candidates"][axis] == target)
        )[:, 0]
        assert len(np.unique(sources)) >= 12
