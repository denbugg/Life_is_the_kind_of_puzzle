from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.candidate_supply import split_tiles
from aiijc_puzzle.edge_ranker import (
    EdgeRow,
    PairwiseEdgeRanker,
    attach_target_labels,
    build_inference_board,
    edge_listwise_loss,
    pack_rows,
    prepare_tile_channels,
    score_board,
    unpack_logits,
)


def _smooth_tiles() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:40, :40]
    clean_image = np.stack(
        ((3 * xx + yy) % 256, (xx + 4 * yy) % 256, (2 * xx + 3 * yy) % 256),
        axis=-1,
    ).astype(np.uint8)
    clean = split_tiles(clean_image, grid=2)
    permutation = np.asarray([2, 0, 3, 1])
    dirty = clean[permutation].astype(np.int16)
    offsets = np.asarray([8, -5, 4, -7])[:, None, None, None]
    dirty = np.clip(dirty + offsets, 0, 255).astype(np.uint8)
    return dirty, clean


def test_target_attachment_does_not_change_inference_visible_rows() -> None:
    dirty, clean = _smooth_tiles()
    inference = build_inference_board(dirty, views=("raw", "bilateral"), candidate_k=3)
    labelled = attach_target_labels(inference, clean)
    assert len(inference.rows) == 8
    assert len(labelled.rows) == 4
    lookup = {(row.direction, row.anchor): row for row in inference.rows}
    for row in labelled.rows:
        original = lookup[(row.direction, row.anchor)]
        np.testing.assert_array_equal(row.candidates, original.candidates)
        np.testing.assert_array_equal(row.features, original.features)
        np.testing.assert_array_equal(row.baseline_scores, original.baseline_scores)
        assert row.exact_candidate >= 0
        assert row.teacher_scores is not None


def test_zero_initialised_ranker_is_exact_bilateral_identity() -> None:
    dirty, _ = _smooth_tiles()
    board = build_inference_board(dirty, views=("raw", "bilateral"), candidate_k=3)
    model = PairwiseEdgeRanker(feature_dim=6, view_mode="dual", width=8, hidden=12)
    right, down, diagnostics = score_board(model, board, device=torch.device("cpu"), pair_batch=16)
    np.testing.assert_allclose(right, board.right_baseline, atol=1e-6)
    np.testing.assert_allclose(down, board.down_baseline, atol=1e-6)
    assert diagnostics["delta_abs_max"] == 0.0


def test_listwise_loss_and_variable_padding_have_finite_gradients() -> None:
    rows = (
        EdgeRow(
            anchor=0,
            candidates=np.asarray([1, 2, 3]),
            features=np.zeros((3, 6), dtype=np.float32),
            baseline_scores=np.asarray([-1.0, -2.0, -3.0], dtype=np.float32),
            direction=0,
            exact_candidate=0,
            trusted_query=True,
            candidate_mapping_margin=np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
            mapping_confidence_cut=0.5,
            teacher_scores=np.asarray([2.0, 0.0, -1.0], dtype=np.float32),
        ),
        EdgeRow(
            anchor=1,
            candidates=np.asarray([0, 2]),
            features=np.zeros((2, 6), dtype=np.float32),
            baseline_scores=np.asarray([-2.0, -1.0], dtype=np.float32),
            direction=1,
            exact_candidate=1,
            trusted_query=True,
            candidate_mapping_margin=np.asarray([1.0, 1.0], dtype=np.float32),
            mapping_confidence_cut=0.5,
            teacher_scores=np.asarray([-1.0, 2.0], dtype=np.float32),
        ),
    )
    packed = pack_rows(rows, device=torch.device("cpu"))
    flat = packed["baseline"].clone().requires_grad_(True)
    logits = unpack_logits(flat, packed)
    loss, parts = edge_listwise_loss(logits, packed)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(flat.grad).all()
    assert parts["teacher_ce"] > 0
    assert logits[1, 2] < -1e3


def test_guarded_dual_view_keeps_raw_and_adds_denoised_residual() -> None:
    dirty, _ = _smooth_tiles()
    raw = prepare_tile_channels(dirty, view_mode="raw")
    dual = prepare_tile_channels(dirty, view_mode="dual")
    assert raw.shape == (4, 6, 20, 20)
    assert dual.shape == (4, 15, 20, 20)
    torch.testing.assert_close(dual[:, :6], raw)
