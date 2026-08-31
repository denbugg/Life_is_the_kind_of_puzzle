from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.seam_transformer import (
    OrderedSeamTransformer,
    SeamCandidateBoard,
    SeamCandidateRow,
    augment_ordered_pairs,
    listwise_hard_negative_loss,
    rerank_score_matrices,
)


def test_pair_scores_are_candidate_permutation_equivariant() -> None:
    torch.manual_seed(3)
    model = OrderedSeamTransformer(feature_dim=6, dim=32, heads=4, layers=2, dropout=0.0)
    model.eval()
    anchors = torch.randint(0, 256, (5, 3, 20, 20), dtype=torch.uint8)
    candidates = torch.randint(0, 256, (5, 3, 20, 20), dtype=torch.uint8)
    features = torch.randn(5, 6)
    directions = torch.tensor((0, 1, 0, 1, 0))
    permutation = torch.tensor((3, 0, 4, 1, 2))
    with torch.inference_mode():
        original = model(anchors, candidates, features, directions)
        permuted = model(
            anchors[permutation],
            candidates[permutation],
            features[permutation],
            directions[permutation],
        )
    torch.testing.assert_close(permuted, original[permutation])


def test_listwise_loss_backpropagates() -> None:
    logits = torch.tensor((0.1, -0.2, 0.4, 0.3, -0.5), requires_grad=True)
    row_ids = torch.tensor((0, 0, 0, 1, 1))
    exact = torch.tensor((2, 3))
    loss = listwise_hard_negative_loss(logits, row_ids, exact)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_reranking_preserves_score_multisets_and_strict_shapes() -> None:
    rng = np.random.default_rng(7)
    right = rng.normal(size=(576, 576)).astype(np.float32)
    down = rng.normal(size=(576, 576)).astype(np.float32)
    row = SeamCandidateRow(
        anchor=4,
        candidates=np.asarray((8, 9, 10)),
        features=np.zeros((3, 3), dtype=np.float32),
        direction=0,
        baseline_choice=0,
    )
    board = SeamCandidateBoard(
        filename="synthetic.png",
        tiles=np.zeros((576, 20, 20, 3), dtype=np.uint8),
        rows=(row,),
        right_scores=right,
        down_scores=down,
        views=("raw",),
    )
    learned_right, learned_down = rerank_score_matrices(board, [np.asarray((0.0, 2.0, 1.0))])
    np.testing.assert_array_equal(
        np.sort(learned_right[4, (8, 9, 10)]), np.sort(right[4, (8, 9, 10)])
    )
    np.testing.assert_array_equal(learned_down, down)
    assert np.argmax(learned_right[4, (8, 9, 10)]) == 1


def test_augmentation_is_deterministic_and_shape_safe() -> None:
    anchors = np.full((2, 20, 20, 3), 128, dtype=np.uint8)
    candidates = np.full((4, 20, 20, 3), 128, dtype=np.uint8)
    row_ids = np.asarray((0, 0, 1, 1))
    first = augment_ordered_pairs(
        anchors, candidates, row_ids, rng=np.random.default_rng(11), jpeg_probability=1.0
    )
    second = augment_ordered_pairs(
        anchors, candidates, row_ids, rng=np.random.default_rng(11), jpeg_probability=1.0
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0].shape == anchors.shape and first[0].dtype == np.uint8
    assert first[1].shape == candidates.shape and first[1].dtype == np.uint8
