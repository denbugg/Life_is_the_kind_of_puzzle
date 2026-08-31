from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.hbt_recovery import (
    HISTORICAL_COMMIT,
    SideEmbeddingNet,
    dense_scores,
    direction_labels,
    embedding_hard_triplet_loss,
    make_synthetic_panel,
    view_tiles,
)
from aiijc_puzzle.protocol import GRID_SIZE, TILE_COUNT, TILE_SIZE


def test_direction_labels_identity_has_exact_grid_neighbours() -> None:
    labels = direction_labels(np.arange(TILE_COUNT, dtype=np.int32))
    assert len(labels.right_queries) == GRID_SIZE * (GRID_SIZE - 1)
    assert len(labels.down_queries) == GRID_SIZE * (GRID_SIZE - 1)
    np.testing.assert_array_equal(labels.right_targets, labels.right_queries + 1)
    np.testing.assert_array_equal(labels.down_targets, labels.down_queries + GRID_SIZE)
    np.testing.assert_array_equal(labels.outside[0], [1, 0, 1, 0])
    np.testing.assert_array_equal(labels.outside[-1], [0, 1, 0, 1])


def test_exact_hard_triplet_loss_is_finite_and_backpropagates() -> None:
    generator = torch.Generator().manual_seed(7)
    tensors = {
        name: torch.randn(TILE_COUNT, 8, generator=generator, requires_grad=True)
        for name in ("raw_q_right", "raw_k_left", "raw_q_down", "raw_k_up")
    }
    outputs = {
        "q_right": torch.nn.functional.normalize(tensors["raw_q_right"], dim=1),
        "k_left": torch.nn.functional.normalize(tensors["raw_k_left"], dim=1),
        "q_down": torch.nn.functional.normalize(tensors["raw_q_down"], dim=1),
        "k_up": torch.nn.functional.normalize(tensors["raw_k_up"], dim=1),
        **tensors,
        "outside_logits": torch.randn(TILE_COUNT, 4, generator=generator, requires_grad=True),
    }
    labels = direction_labels(np.arange(TILE_COUNT, dtype=np.int32))
    loss, metrics = embedding_hard_triplet_loss(outputs, labels, temperature=0.07)
    assert torch.isfinite(loss)
    assert set(metrics) == {
        "cross_entropy_loss",
        "embedding_l2",
        "loss",
        "outside_loss",
        "recall_at_1",
        "triplet_loss",
    }
    loss.backward()
    assert all(tensor.grad is not None for tensor in tensors.values())


def test_synthetic_panel_is_deterministic_and_bilateral_is_order_preserving() -> None:
    rows, columns = np.indices((480, 480))
    target = np.stack([rows % 256, columns % 256, (rows + columns) % 256], axis=2).astype(np.uint8)
    first = make_synthetic_panel(target, seed=123)
    second = make_synthetic_panel(target, seed=123)
    assert first.slot_tiles.shape == (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    np.testing.assert_array_equal(first.slot_to_target, second.slot_to_target)
    np.testing.assert_array_equal(first.slot_tiles, second.slot_tiles)
    assert len(np.unique(first.slot_to_target)) == TILE_COUNT
    guarded = view_tiles(first.slot_tiles, view="bilateral")
    assert guarded.shape == first.slot_tiles.shape
    assert guarded.dtype == np.uint8
    np.testing.assert_array_equal(view_tiles(first.slot_tiles, view="raw"), first.slot_tiles)


def test_historical_rgb_sobel_d320_dense_scores_are_valid() -> None:
    assert len(HISTORICAL_COMMIT) == 40
    model = SideEmbeddingNet(
        channels=64,
        embedding_dim=320,
        side_band=4,
        tangent_bins=10,
        temperature=0.07,
        input_mode="rgb_sobel",
    )
    tiles = np.zeros((TILE_COUNT, TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    right, down, outside = dense_scores(model, tiles, device=torch.device("cpu"))
    assert right.shape == down.shape == (TILE_COUNT, TILE_COUNT)
    assert outside.shape == (TILE_COUNT, 4)
    assert np.isfinite(right).all() and np.isfinite(down).all()
    assert np.all(np.diag(right) < -1_000) and np.all(np.diag(down) < -1_000)
    np.testing.assert_allclose(np.exp(right).sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.exp(down).sum(axis=1), 1.0, atol=1e-5)
