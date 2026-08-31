from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from aiijc_puzzle.basincycle_stage_b import (
    METRIC_NAMES,
    QUANTILE_LEVELS,
    BasinCycleStageB,
    OracleDiagnostic,
    ProposalBank,
    StageBLabels,
    StageBOutput,
    aggregate_oracle_diagnostics,
    build_target_free_proposal_bank,
    frozen_positive_action_mask,
    materialize_candidate_layouts,
    metric_deltas_for_bank,
    model_static_ledger,
    proposal_oracle_diagnostic,
    select_hard_action,
    stage_b_loss,
)
from aiijc_puzzle.basincycle_synthetic import (
    is_strict_permutation,
    make_synthetic_case,
    relabel_instance,
)


def _tiny_model() -> BasinCycleStageB:
    return BasinCycleStageB(
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


def test_default_architecture_is_full_resolution_and_has_exact_static_ledger() -> None:
    model = BasinCycleStageB()
    assert model.trainable_parameter_count() == 141_073
    assert all(
        module.stride == (1, 1)
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    )
    assert not any(
        isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.ConvTranspose2d))
        for module in model.modules()
    )
    ledger = model_static_ledger(model, batch_size=4)
    assert ledger["trainable_parameters"] == 141_073
    assert ledger["forward_learned_macs_per_board"] == 261_623_936
    assert ledger["forward_learned_macs_per_batch"] == 1_046_495_744


def test_forward_keeps_hard_identity_and_all_candidates_strict() -> None:
    torch.manual_seed(11)
    model = _tiny_model().eval()
    tiles = torch.rand(1, 36, 3, 20, 20)
    layout = torch.randperm(36).unsqueeze(0)
    with torch.no_grad():
        output = model(tiles, layout)
    assert output.pair_logits.shape == (1, 2, 36, 36)
    assert output.boundary_prediction.shape == (1, 36, 4, 20, 6)
    assert output.quantiles.shape == (1, 16, 3, 3)
    assert torch.equal(output.candidate_layouts[:, 0], layout)
    assert all(is_strict_permutation(row.numpy()) for row in output.candidate_layouts[0])
    assert torch.all(output.quantiles[..., 0] <= output.quantiles[..., 1])
    assert torch.all(output.quantiles[..., 1] <= output.quantiles[..., 2])
    assert METRIC_NAMES == ("pair", "exact", "radius2")
    assert QUANTILE_LEVELS == (0.10, 0.50, 0.90)
    assert output.proposal_bank.valid[0, 0]
    assert output.proposal_bank.lengths[0, 0] == 0
    invalid = ~output.proposal_bank.valid
    assert torch.isneginf(output.action_logits[invalid]).all()
    assert torch.isposinf(output.risk_logits[invalid]).all()


def test_proposal_bank_is_tile_relabel_equivariant_and_target_free() -> None:
    case = make_synthetic_case(
        grid_size=6,
        seed=91,
        true_edge_score=8.0,
        false_edge_sigma=0.1,
        true_edge_noise_sigma=0.05,
        distractor_probability=0.0,
        distractor_boost=0.0,
        corruption_cycle_count=4,
        corruption_cycle_length=3,
    )
    mapping = np.random.default_rng(5).permutation(36)
    relabeled, right, down = relabel_instance(
        case.control,
        case.right_scores,
        case.down_scores,
        mapping,
    )
    scores = torch.tensor(np.stack((case.right_scores, case.down_scores)))[None]
    relabeled_scores = torch.tensor(np.stack((right, down)))[None]
    kwargs = {
        "grid_size": 6,
        "top_k": 8,
        "candidate_cap": 6,
        "seed_count": 18,
        "max_cycle_length": 3,
        "proposal_cap": 128,
    }
    original_bank = build_target_free_proposal_bank(
        scores,
        torch.tensor(case.control)[None],
        **kwargs,
    )
    relabeled_bank = build_target_free_proposal_bank(
        relabeled_scores,
        torch.tensor(relabeled)[None],
        **kwargs,
    )
    assert torch.equal(original_bank.positions, relabeled_bank.positions)
    assert torch.equal(original_bank.lengths, relabeled_bank.lengths)
    assert torch.equal(original_bank.valid, relabeled_bank.valid)
    original_layouts = materialize_candidate_layouts(
        torch.tensor(case.control)[None],
        original_bank,
    )
    relabeled_layouts = materialize_candidate_layouts(
        torch.tensor(relabeled)[None],
        relabeled_bank,
    )
    assert torch.equal(relabeled_layouts, torch.tensor(mapping)[original_layouts])


def test_selector_falls_back_to_keep_and_never_indexes_padding() -> None:
    control = torch.arange(9).unsqueeze(0)
    bank = ProposalBank(
        positions=torch.tensor([[[-1, -1, -1], [0, 1, -1], [-1, -1, -1]]]),
        lengths=torch.tensor([[0, 2, 0]]),
        valid=torch.tensor([[True, True, False]]),
    )
    candidates = materialize_candidate_layouts(control, bank)
    quantiles = torch.zeros(1, 3, 3, 3)
    quantiles[:, :, :, 0] = -1.0
    quantiles[:, :, :, 1] = 0.0
    quantiles[:, :, :, 2] = 1.0
    output = StageBOutput(
        pair_logits=torch.zeros(1, 2, 9, 9),
        boundary_prediction=torch.zeros(1, 9, 4, 20, 6),
        proposal_bank=bank,
        candidate_layouts=candidates,
        action_logits=torch.tensor([[0.0, 0.0, -torch.inf]]),
        quantiles=quantiles,
        risk_logits=torch.tensor([[0.0, -10.0, torch.inf]]),
    )
    selected, layout = select_hard_action(output)
    assert selected.item() == 0
    assert torch.equal(layout, control)

    quantiles[0, 1, 0] = torch.tensor([1.0, 2.0, 3.0])
    selected, layout = select_hard_action(output)
    assert selected.item() == 1
    assert torch.equal(layout, candidates[:, 1])
    assert is_strict_permutation(layout[0].numpy())


def test_frozen_oracle_path_scores_after_identity_freeze() -> None:
    case = make_synthetic_case(
        grid_size=3,
        seed=94,
        true_edge_score=8.0,
        false_edge_sigma=0.1,
        true_edge_noise_sigma=0.05,
        distractor_probability=0.0,
        distractor_boost=0.0,
        corruption_cycle_count=1,
        corruption_cycle_length=3,
    )
    scores = torch.tensor(np.stack((case.right_scores, case.down_scores)))[None]
    bank = build_target_free_proposal_bank(
        scores,
        torch.tensor(case.control)[None],
        grid_size=3,
        top_k=5,
        candidate_cap=4,
        seed_count=9,
        max_cycle_length=3,
        proposal_cap=64,
    )
    positions = bank.positions[0].numpy()
    lengths = bank.lengths[0].numpy()
    valid = bank.valid[0].numpy()
    frozen_positions = positions.copy()
    frozen_lengths = lengths.copy()
    frozen_valid = valid.copy()
    deltas = metric_deltas_for_bank(
        case.control,
        case.truth,
        positions,
        lengths,
        valid,
        grid_size=3,
    )
    diagnostic = proposal_oracle_diagnostic(
        case.control,
        case.truth,
        positions,
        lengths,
        valid,
        grid_size=3,
    )
    assert diagnostic.exhaustive_has_benefit
    assert diagnostic.proposal_has_benefit
    assert diagnostic.proposal_best_pair_delta == diagnostic.exhaustive_best_pair_delta
    positive = frozen_positive_action_mask(deltas, valid)
    assert positive.any()
    assert np.all(valid[positive])
    assert np.array_equal(positions, frozen_positions)
    assert np.array_equal(lengths, frozen_lengths)
    assert np.array_equal(valid, frozen_valid)


def test_proposal_oracle_coverage_has_explicit_opportunity_denominator() -> None:
    diagnostics = (
        OracleDiagnostic(True, True, 4, 2, 0, 20),
        OracleDiagnostic(True, False, 3, 0, 0, 18),
        OracleDiagnostic(False, False, 0, 0, 0, 17),
    )
    summary = aggregate_oracle_diagnostics(diagnostics)
    assert summary == {
        "state_count": 3,
        "beneficial_exhaustive_state_count": 2,
        "beneficial_proposal_state_count": 1,
        "proposal_oracle_coverage": 0.5,
    }


def test_stage_b_objective_is_finite_and_differentiable() -> None:
    torch.manual_seed(13)
    model = _tiny_model().train()
    output = model(torch.rand(1, 36, 3, 20, 20), torch.arange(36).unsqueeze(0))
    positive = torch.zeros_like(output.proposal_bank.valid)
    positive[:, 0] = True
    edge_targets = torch.full((1, 2, 36), -1, dtype=torch.long)
    board = torch.arange(36).reshape(6, 6)
    edge_targets[0, 0, board[:, :-1].reshape(-1)] = board[:, 1:].reshape(-1)
    edge_targets[0, 1, board[:-1, :].reshape(-1)] = board[1:, :].reshape(-1)
    labels = StageBLabels(
        positive_actions=positive,
        metric_deltas=torch.zeros(1, 16, 3),
        loses_true_pair=torch.zeros(1, 16, dtype=torch.bool),
        edge_targets=edge_targets,
        clean_boundary_targets=torch.zeros_like(output.boundary_prediction),
    )
    loss, parts = stage_b_loss(output, labels)
    assert torch.isfinite(loss)
    assert set(parts) == {"policy", "edge", "restore", "quantile", "risk"}
    loss.backward()
    assert model.image_stem[0].weight.grad is not None
    assert torch.isfinite(model.image_stem[0].weight.grad).all()


def test_loss_rejects_reference_labels_for_padding() -> None:
    model = _tiny_model().eval()
    model.proposal_cap = 64
    with torch.no_grad():
        output = model(torch.rand(1, 36, 3, 20, 20), torch.arange(36).unsqueeze(0))
    positive = torch.zeros_like(output.proposal_bank.valid)
    padding = torch.nonzero(~output.proposal_bank.valid[0], as_tuple=False)[0, 0]
    positive[0, padding] = True
    labels = StageBLabels(
        positive_actions=positive,
        metric_deltas=torch.zeros(1, 64, 3),
        loses_true_pair=torch.zeros(1, 64, dtype=torch.bool),
        edge_targets=torch.full((1, 2, 36), -1, dtype=torch.long),
        clean_boundary_targets=torch.zeros_like(output.boundary_prediction),
    )
    with pytest.raises(ValueError, match="padding"):
        stage_b_loss(output, labels)


def test_output_dataclass_does_not_change_when_copied() -> None:
    bank = ProposalBank(
        positions=torch.full((1, 1, 3), -1),
        lengths=torch.zeros(1, 1, dtype=torch.long),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    output = StageBOutput(
        pair_logits=torch.zeros(1, 2, 1, 1),
        boundary_prediction=torch.zeros(1, 1, 4, 20, 6),
        proposal_bank=bank,
        candidate_layouts=torch.zeros(1, 1, 1, dtype=torch.long),
        action_logits=torch.zeros(1, 1),
        quantiles=torch.zeros(1, 1, 3, 3),
        risk_logits=torch.zeros(1, 1),
    )
    assert copy.copy(output) == output
