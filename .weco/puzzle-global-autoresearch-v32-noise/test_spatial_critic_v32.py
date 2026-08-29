import numpy as np
import torch

import spatial_critic_v32 as s


def matrices(seed=3):
    rng = np.random.default_rng(seed)
    right = rng.normal(size=(576, 576)).astype(np.float32)
    down = rng.normal(size=(576, 576)).astype(np.float32)
    unary = rng.normal(size=(576, 576)).astype(np.float32)
    return right, down, unary


def test_identity_targets_and_features():
    board = np.arange(576)
    right, down, unary = matrices()
    target_r, target_d, target_cell, adjacency = s.board_targets(board)
    assert adjacency == 1.0
    assert target_r[:, :-1].min() == 1 and target_d[:-1].min() == 1
    features = s.board_tensor(board, right, down, unary)
    assert features.shape == (32, 24, 24)
    assert np.isfinite(features).all()


def test_model_shape_and_size():
    model = s.SpatialBoardCritic(72, 104)
    global_score, local = model(torch.randn(2, 32, 24, 24))
    assert global_score.shape == (2,)
    assert local.shape == (2, 3, 24, 24)
    assert 900_000 <= s.parameter_count(model) <= 1_150_000
