from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.restored_border_ranker import (
    RestoredBorderRanker,
    build_candidate_union,
    pad_candidate_rows,
    restored_descriptor_scores,
    restored_seam_features,
    unpack_candidate_logits,
)


def test_candidate_union_is_target_blind_self_excluded_and_keeps_raw_topk() -> None:
    rng = np.random.default_rng(20)
    raw = rng.normal(size=(9, 9)).astype(np.float32)
    restored = rng.normal(size=(9, 9)).astype(np.float32)
    union = build_candidate_union(raw, restored, topk=3)
    assert len(union.rows) == 9
    for anchor, candidates in enumerate(union.rows):
        assert anchor not in candidates
        assert set(union.raw_topk[anchor]).issubset(candidates)
        assert set(union.restored_topk[anchor]).issubset(candidates)
        assert 3 <= len(candidates) <= 6
        assert union.scalar_features[anchor].shape == (len(candidates), 8)


def test_restored_descriptor_scores_are_finite_and_mask_self() -> None:
    rng = np.random.default_rng(21)
    tiles = rng.integers(0, 256, size=(9, 20, 20, 3), dtype=np.uint8)
    for direction in (0, 1):
        scores = restored_descriptor_scores(tiles, direction=direction)
        assert scores.shape == (9, 9)
        assert np.isfinite(scores).all()
        np.testing.assert_array_equal(np.diag(scores), np.full(9, -4.0, np.float32))


def test_zero_initialised_cross_ranker_preserves_raw_baseline() -> None:
    rng = np.random.default_rng(22)
    tiles = torch.from_numpy(
        rng.random((9, 3, 20, 20), dtype=np.float32)
    )
    anchors = torch.tensor([0, 0, 1, 1])
    candidates = torch.tensor([1, 2, 0, 2])
    directions = torch.tensor([0, 0, 1, 1])
    scalar = torch.randn(4, 8)
    baseline = torch.tensor([2.0, 1.0, 0.5, -0.5])
    model = RestoredBorderRanker(base=8)
    scores, residual = model(
        tiles,
        anchors,
        candidates,
        directions,
        scalar,
        baseline,
    )
    torch.testing.assert_close(scores, baseline)
    torch.testing.assert_close(residual, torch.zeros_like(residual))


def test_historical_seam_channels_and_listwise_padding_have_gradients() -> None:
    tiles = torch.rand(4, 3, 20, 20)
    anchors = torch.tensor([0, 0, 1, 1])
    candidates = torch.tensor([1, 2, 0, 2])
    directions = torch.tensor([0, 1, 0, 1])
    seams = restored_seam_features(tiles, anchors, candidates, directions)
    assert seams.shape == (4, 7, 20, 12)

    rows = [
        (
            0,
            0,
            np.asarray([1, 2, 3], np.int32),
            np.zeros((3, 8), np.float32),
            np.asarray([2.0, 1.0, 0.0], np.float32),
            0,
        ),
        (
            1,
            1,
            np.asarray([0, 2], np.int32),
            np.zeros((2, 8), np.float32),
            np.asarray([0.0, 1.0], np.float32),
            1,
        ),
    ]
    packed = pad_candidate_rows(rows, device=torch.device("cpu"))
    flat = packed["baseline"].clone().requires_grad_(True)
    logits = unpack_candidate_logits(flat, packed)
    loss = torch.nn.functional.cross_entropy(logits, packed["targets"])
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(flat.grad).all()
    assert logits[1, 2] < -1e3
