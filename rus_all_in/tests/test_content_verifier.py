from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.candidate_supply import blur3
from aiijc_puzzle.content_verifier import (
    CandidateRow,
    ContentListwiseVerifier,
    build_candidate_board,
    multi_positive_listwise_loss,
    summarize_choices,
    summarize_oracle,
)


def test_build_candidate_board_separates_dirty_features_from_clean_labels() -> None:
    rng = np.random.default_rng(7)
    clean = rng.integers(0, 256, size=(4, 8, 8, 3), dtype=np.uint8)
    permutation = np.array([2, 0, 3, 1])
    dirty = np.clip(blur3(clean), 0, 255).astype(np.uint8)[permutation]

    board = build_candidate_board(dirty, clean, views=("raw",), candidate_k=1)

    assert len(board.rows) == 4  # two right and two down labelled relations
    assert board.tiles.dtype == np.uint8
    for row in board.rows:
        assert row.features.shape == (len(row.candidates), 3)
        assert row.candidate_rmse.shape == (len(row.candidates),)
        assert row.candidate_mapping_margin.shape == (len(row.candidates),)
        assert row.anchor not in row.candidates
        assert np.isfinite(row.features).all()


def test_listwise_model_is_candidate_permutation_equivariant() -> None:
    torch.manual_seed(4)
    model = ContentListwiseVerifier(feature_dim=6, dim=16, heads=4).eval()
    assert model.spatial_position.shape == (1, 25, 16)
    assert bool(torch.any(model.spatial_position != 0))
    torch.nn.init.normal_(model.score[-1].weight, std=0.1)
    anchors = torch.randint(0, 256, (2, 3, 20, 20), dtype=torch.uint8)
    candidates = torch.randint(0, 256, (2, 4, 3, 20, 20), dtype=torch.uint8)
    features = torch.randn(2, 4, 6)
    valid = torch.ones(2, 4, dtype=torch.bool)
    directions = torch.tensor([0, 1])
    permutation = torch.tensor([2, 0, 3, 1])

    with torch.inference_mode():
        original = model(anchors, candidates, features, valid, directions)
        permuted = model(
            anchors,
            candidates[:, permutation],
            features[:, permutation],
            valid[:, permutation],
            directions,
        )
        padded = model(
            anchors,
            torch.cat((candidates, torch.randint_like(candidates[:, :1], 0, 256)), dim=1),
            torch.cat((features, torch.randn(2, 1, 6)), dim=1),
            torch.cat((valid, torch.zeros(2, 1, dtype=torch.bool)), dim=1),
            directions,
        )

    torch.testing.assert_close(permuted, original[:, permutation], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(padded[:, :4], original, atol=2e-5, rtol=2e-5)
    assert torch.all(padded[:, 4] == -1e4)


def test_multi_positive_loss_and_metrics() -> None:
    logits = torch.tensor(
        [[0.0, 3.0, 1.0], [1.0, -2.0, -10_000.0], [0.0, 1.0, -10_000.0]],
        requires_grad=True,
    )
    positives = torch.tensor([[False, True, True], [True, False, False], [False, False, False]])
    valid = torch.tensor([[True, True, True], [True, True, False], [True, True, False]])
    loss = multi_positive_listwise_loss(logits, positives, valid)
    assert 0.0 < float(loss.detach()) < 0.2
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    rows = [
        CandidateRow(
            anchor=0,
            candidates=np.array([1, 2]),
            features=np.zeros((2, 3), dtype=np.float32),
            candidate_rmse=np.array([0.0, 15.0], dtype=np.float32),
            candidate_mapping_margin=np.array([2.0, 0.5], dtype=np.float32),
            mapping_confidence_cut=1.0,
            exact_candidate=0,
            baseline_choices=(1,),
            ensemble_choice=1,
            direction=0,
            trusted=True,
        ),
        CandidateRow(
            anchor=1,
            candidates=np.array([0, 2]),
            features=np.zeros((2, 3), dtype=np.float32),
            candidate_rmse=np.array([30.0, 8.0], dtype=np.float32),
            candidate_mapping_margin=np.array([2.0, 0.5], dtype=np.float32),
            mapping_confidence_cut=1.0,
            exact_candidate=-1,
            baseline_choices=(0,),
            ensemble_choice=0,
            direction=1,
            trusted=False,
        ),
    ]
    chosen = summarize_choices(rows, [0, 1], scope="all")
    oracle = summarize_oracle(rows, scope="all")
    strict_low_margin = summarize_choices(rows[:1], [1], scope="trusted")
    query_low_margin = summarize_choices(rows[:1], [1], scope="trusted_query")
    assert rows[0].training_positives(20.0).tolist() == [True, False]
    assert rows[1].training_positives(20.0).tolist() == [False, False]
    assert chosen["exact"] == 0.5
    assert chosen["content_rmse_le_10"] == 1.0
    assert strict_low_margin["content_rmse_le_20"] == 0.0
    assert query_low_margin["content_rmse_le_20"] == 1.0
    assert oracle["exact"] == 0.5
    assert oracle["content_rmse_le_20"] == 1.0
