from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_current_finetune import (
    FocalTrainingBoard,
    board_pair_ranking_loss,
    exact_harvest_edge_labels,
    train_fixed_focal_model,
)
from aiijc_puzzle.taska_focal_verifier import SeamVerifier


def test_exact_harvest_labels_follow_reference_geometry_not_bag_ids() -> None:
    grid = 3
    reference = np.asarray([5, 0, 8, 2, 7, 3, 1, 6, 4], dtype=np.int32)
    edges = (
        RawTailEdge(5, 0, "right"),
        RawTailEdge(0, 8, "right"),
        RawTailEdge(8, 2, "right"),  # row wrap: false
        RawTailEdge(5, 2, "down"),
        RawTailEdge(0, 7, "down"),
        RawTailEdge(4, 5, "down"),  # bottom row source: false
    )
    assert np.array_equal(
        exact_harvest_edge_labels(edges, reference, grid=grid),
        np.asarray([1, 1, 0, 1, 1, 0], dtype=np.uint8),
    )


def test_pair_ranking_loss_rewards_true_edges_above_false_edges() -> None:
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    good = board_pair_ranking_loss(torch.tensor([3.0, 2.0, -2.0, -3.0]), labels)
    bad = board_pair_ranking_loss(torch.tensor([-3.0, -2.0, 2.0, 3.0]), labels)
    assert good < bad


def test_fixed_training_changes_residual_but_preserves_raw_prior_exactly() -> None:
    torch.manual_seed(17)
    model = SeamVerifier(ch=4, blocks=0, feats=6, strip=4)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.prior.requires_grad_(False)
    prior_before = model.prior.detach().clone()
    weight_before = model.out.weight.detach().clone()
    rng = np.random.default_rng(19)
    boards = []
    for index in range(2):
        boards.append(
            FocalTrainingBoard(
                patches=rng.normal(size=(6, 3, 20, 8)).astype(np.float32),
                features=rng.normal(size=(6, 6)).astype(np.float32),
                labels=np.asarray([1, 1, 1, 0, 0, 0], dtype=np.uint8),
                source_filename=f"board-{index}",
            )
        )
    history = train_fixed_focal_model(model, boards, device="cpu")
    assert len(history) == 2
    assert torch.equal(model.prior.detach(), prior_before)
    assert not torch.equal(model.out.weight.detach(), weight_before)
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
