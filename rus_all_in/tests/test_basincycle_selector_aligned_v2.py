from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from aiijc_puzzle.basincycle_selector_aligned_v2 import (
    BasinCycleSelectorAlignedV2,
    SelectorAlignedLabels,
    SelectorAlignedOutput,
    changed_contact_mask,
    compositional_changed_edge_features,
    select_selector_aligned_action,
    selector_aligned_loss,
    selector_aligned_static_ledger,
    selector_aligned_targets,
    vectorized_pair_delta_and_loss,
)
from aiijc_puzzle.basincycle_stage_b import ProposalBank, materialize_candidate_layouts
from aiijc_puzzle.basincycle_stage_b_runner import pair_loss_labels
from aiijc_puzzle.basincycle_synthetic import apply_cycle, is_strict_permutation, true_pair_count


def _tiny_model() -> BasinCycleSelectorAlignedV2:
    return BasinCycleSelectorAlignedV2(
        feature_channels=16,
        retrieval_dim=16,
        state_dim=32,
        encoder_blocks=1,
        state_blocks=1,
        proposal_top_k=4,
        proposal_candidate_cap=2,
        proposal_seed_count=3,
        proposal_cap=16,
    )


def _manual_output(
    *,
    safe_logits: torch.Tensor,
    gain_scores: torch.Tensor,
) -> SelectorAlignedOutput:
    control = torch.arange(9).unsqueeze(0)
    positions = torch.tensor([[[-1, -1, -1], [0, 1, -1], [2, 3, 4], [-1, -1, -1]]])
    bank = ProposalBank(
        positions=positions,
        lengths=torch.tensor([[0, 2, 3, 0]]),
        valid=torch.tensor([[True, True, True, False]]),
    )
    candidates = materialize_candidate_layouts(control, bank)
    return SelectorAlignedOutput(
        pair_logits=torch.zeros(1, 2, 9, 9),
        boundary_prediction=torch.zeros(1, 9, 4, 20, 6),
        proposal_bank=bank,
        candidate_layouts=candidates,
        safe_improvement_logits=safe_logits,
        pair_gain_scores=gain_scores,
        compositional_features=torch.zeros(1, 4, 2),
    )


def test_architecture_has_no_downsampling_and_only_two_selector_outputs() -> None:
    model = BasinCycleSelectorAlignedV2()
    assert model.trainable_parameter_count() == 140_744
    ledger = selector_aligned_static_ledger(model, batch_size=4)
    assert ledger["forward_learned_macs_per_board"] == 261_542_016
    assert ledger["forward_learned_macs_per_batch"] == 1_046_168_064
    assert ledger["per_board"]["selector_head"] == 18_284_544
    assert "action_head" not in ledger["per_board"]
    assert all(
        module.stride == (1, 1) for module in model.modules() if isinstance(module, nn.Conv2d)
    )
    assert model.selector_head[-1].out_features == 2
    assert "action_head" not in dict(model.named_modules())


def test_forward_keeps_strict_candidates_and_analytical_keep() -> None:
    torch.manual_seed(101)
    model = _tiny_model().eval()
    tiles = torch.rand(1, 36, 3, 20, 20)
    layout = torch.randperm(36).unsqueeze(0)
    with torch.no_grad():
        output = model(tiles, layout)
    assert output.safe_improvement_logits.shape == (1, 16)
    assert output.pair_gain_scores.shape == (1, 16)
    assert output.compositional_features.shape == (1, 16, 2)
    assert torch.isneginf(output.safe_improvement_logits[:, 0]).all()
    assert torch.equal(output.pair_gain_scores[:, 0], torch.zeros(1))
    assert torch.equal(output.compositional_features[:, 0], torch.zeros(1, 2))
    assert torch.equal(output.candidate_layouts[:, 0], layout)
    assert all(is_strict_permutation(row.numpy()) for row in output.candidate_layouts[0])
    invalid = ~output.proposal_bank.valid
    assert torch.isneginf(output.safe_improvement_logits[invalid]).all()
    assert torch.isneginf(output.pair_gain_scores[invalid]).all()


def test_changed_contact_composition_has_exact_keep_zero_and_finite_gradients() -> None:
    control = torch.arange(4).unsqueeze(0)
    bank = ProposalBank(
        positions=torch.tensor([[[-1, -1], [1, 2]]]),
        lengths=torch.tensor([[0, 2]]),
        valid=torch.tensor([[True, True]]),
    )
    candidates = materialize_candidate_layouts(control, bank)
    logits = torch.randn(1, 2, 4, 4, generator=torch.Generator().manual_seed(4))
    logits.requires_grad_()
    features = compositional_changed_edge_features(logits, candidates, grid_size=2)
    changed = changed_contact_mask(candidates, grid_size=2)
    assert features.shape == (1, 2, 2)
    assert torch.equal(features[:, 0], torch.zeros(1, 2))
    assert not changed[:, 0].any() and changed[:, 1].any()
    assert torch.all((features[..., 1] >= 0) & (features[..., 1] <= 1))
    features[:, 1].sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_vectorized_pair_labels_match_reference_set_implementation() -> None:
    rng = np.random.default_rng(8)
    truths = np.stack((np.arange(36), rng.permutation(36)))
    controls = np.stack(
        (
            apply_cycle(truths[0], (0, 1, 7)),
            apply_cycle(truths[1], (2, 11)),
        )
    )
    cycles = ((), (0, 1), (2, 3, 8), (5, 29), (12, 13, 14))
    candidates = []
    for control in controls:
        candidates.append(
            np.stack([control if not cycle else apply_cycle(control, cycle) for cycle in cycles])
        )
    candidate_values = np.stack(candidates)
    valid = np.ones((2, len(cycles)), dtype=bool)
    valid[1, -1] = False
    delta, loses = vectorized_pair_delta_and_loss(
        candidate_values,
        truths,
        valid,
        grid_size=6,
    )
    for batch_index in range(2):
        baseline = true_pair_count(controls[batch_index], truths[batch_index], grid_size=6)
        expected_delta = np.asarray(
            [
                true_pair_count(candidate, truths[batch_index], grid_size=6) - baseline
                if valid[batch_index, proposal_index]
                else 0
                for proposal_index, candidate in enumerate(candidate_values[batch_index])
            ]
        )
        expected_loss = pair_loss_labels(
            controls[batch_index],
            truths[batch_index],
            candidate_values[batch_index],
            valid[batch_index],
        )
        assert np.array_equal(delta[batch_index], expected_delta)
        assert np.array_equal(loses[batch_index], expected_loss)


def test_pair_loss_preserves_directed_identity_not_original_bond_position() -> None:
    truth = np.arange(36, dtype=np.int64)
    control = np.array(
        [
            33,
            2,
            6,
            32,
            15,
            9,
            26,
            30,
            34,
            14,
            10,
            28,
            3,
            23,
            12,
            25,
            5,
            8,
            18,
            20,
            22,
            35,
            7,
            0,
            1,
            17,
            4,
            13,
            31,
            21,
            24,
            29,
            16,
            19,
            27,
            11,
        ],
        dtype=np.int64,
    )
    candidate = apply_cycle(control, (15, 27, 33))
    candidates = np.stack((control, candidate))[None]
    valid = np.ones((1, 2), dtype=bool)
    delta, loses = vectorized_pair_delta_and_loss(
        candidates,
        truth[None],
        valid,
        grid_size=6,
    )
    assert delta.tolist() == [[0, 1]]
    assert loses.tolist() == [[False, True]]
    assert pair_loss_labels(control, truth, candidates[0], valid[0]).tolist() == [False, True]


@pytest.mark.parametrize("seed", [0, 1, 7, 19, 113])
def test_vectorized_pair_loss_seeded_fuzz_matches_scalar_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    batch_size, proposal_count, tile_count = 3, 29, 36
    truths = np.stack([rng.permutation(tile_count) for _ in range(batch_size)])
    controls = np.stack(
        [
            apply_cycle(truth, tuple(int(value) for value in rng.choice(tile_count, 3, False)))
            for truth in truths
        ]
    )
    candidates = np.empty((batch_size, proposal_count, tile_count), dtype=np.int64)
    candidates[:, 0] = controls
    for batch_index in range(batch_size):
        for proposal_index in range(1, proposal_count):
            length = int(rng.integers(2, 4))
            cycle = tuple(int(value) for value in rng.choice(tile_count, length, False))
            candidates[batch_index, proposal_index] = apply_cycle(
                controls[batch_index],
                cycle,
            )
    valid = rng.random((batch_size, proposal_count)) > 0.2
    valid[:, 0] = True
    delta, loses = vectorized_pair_delta_and_loss(
        candidates,
        truths,
        valid,
        grid_size=6,
    )
    for batch_index in range(batch_size):
        baseline = true_pair_count(
            controls[batch_index],
            truths[batch_index],
            grid_size=6,
        )
        expected_delta = np.asarray(
            [
                true_pair_count(candidate, truths[batch_index], grid_size=6) - baseline
                if valid[batch_index, proposal_index]
                else 0
                for proposal_index, candidate in enumerate(candidates[batch_index])
            ],
            dtype=np.int16,
        )
        expected_loses = pair_loss_labels(
            controls[batch_index],
            truths[batch_index],
            candidates[batch_index],
            valid[batch_index],
        )
        assert np.array_equal(delta[batch_index], expected_delta)
        assert np.array_equal(loses[batch_index], expected_loses)


def test_safe_target_is_exact_positive_gain_and_no_existing_pair_loss() -> None:
    delta = np.array([[0, 2, 1, 0, -1]])
    loses = np.array([[False, False, True, False, True]])
    valid = np.array([[True, True, True, True, False]])
    safe = selector_aligned_targets(delta, loses, valid)
    assert safe.tolist() == [[False, True, False, False, False]]


def test_selector_uses_trained_safe_and_gain_heads_then_abstains() -> None:
    output = _manual_output(
        safe_logits=torch.tensor([[-torch.inf, 0.4, 1.2, -torch.inf]]),
        gain_scores=torch.tensor([[0.0, 2.0, 1.0, -torch.inf]]),
    )
    selected, layout = select_selector_aligned_action(output)
    assert selected.item() == 1
    assert torch.equal(layout, output.candidate_layouts[:, 1])

    abstaining = _manual_output(
        safe_logits=torch.tensor([[-torch.inf, -0.1, 2.0, -torch.inf]]),
        gain_scores=torch.tensor([[0.0, 2.0, -0.1, -torch.inf]]),
    )
    selected, layout = select_selector_aligned_action(abstaining)
    assert selected.item() == 0
    assert torch.equal(layout, abstaining.candidate_layouts[:, 0])


def test_loss_is_finite_differentiable_and_uses_consistent_safe_label() -> None:
    torch.manual_seed(103)
    model = _tiny_model().train()
    output = model(torch.rand(1, 36, 3, 20, 20), torch.arange(36).unsqueeze(0))
    delta = torch.zeros_like(output.pair_gain_scores)
    loses = torch.zeros_like(output.proposal_bank.valid)
    safe = torch.zeros_like(output.proposal_bank.valid)
    edge_targets = torch.full((1, 2, 36), -1, dtype=torch.long)
    board = torch.arange(36).reshape(6, 6)
    edge_targets[0, 0, board[:, :-1].reshape(-1)] = board[:, 1:].reshape(-1)
    edge_targets[0, 1, board[:-1, :].reshape(-1)] = board[1:, :].reshape(-1)
    labels = SelectorAlignedLabels(
        safe_improvement=safe,
        pair_delta=delta,
        loses_existing_true_pair=loses,
        edge_targets=edge_targets,
        clean_boundary_targets=torch.zeros_like(output.boundary_prediction),
    )
    loss, parts = selector_aligned_loss(output, labels, hard_negatives_per_stratum=4)
    assert torch.isfinite(loss)
    assert set(parts) == {"safe_bce", "listwise", "gain", "edge", "restore"}
    loss.backward()
    assert model.selector_head[-1].weight.grad is not None
    assert torch.isfinite(model.selector_head[-1].weight.grad).all()
    assert model.image_stem[0].weight.grad is not None
    assert torch.isfinite(model.image_stem[0].weight.grad).all()

    inconsistent = SelectorAlignedLabels(
        safe_improvement=safe.clone(),
        pair_delta=delta.clone(),
        loses_existing_true_pair=loses.clone(),
        edge_targets=edge_targets,
        clean_boundary_targets=torch.zeros_like(output.boundary_prediction),
    )
    first_nonkeep = int(torch.nonzero(output.proposal_bank.valid[0])[1].item())
    inconsistent.pair_delta[0, first_nonkeep] = 1.0
    with pytest.raises(ValueError, match="safe labels"):
        selector_aligned_loss(output, inconsistent)
