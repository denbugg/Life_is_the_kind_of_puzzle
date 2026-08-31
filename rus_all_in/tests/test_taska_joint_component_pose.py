from __future__ import annotations

import inspect

import numpy as np
import torch

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_joint_component_pose import (
    CANDIDATE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    PAIR_FEATURE_DIM,
    JointComponentPoseTransformer,
    build_joint_pose_board,
    joint_pose_loss,
    joint_pose_targets,
    pack_multiple_component_anchors,
    tile_visible_descriptors,
)


def _grid4_board():
    layout = np.arange(16, dtype=np.int32)
    dirty = np.zeros((16, 20, 20, 3), dtype=np.uint8)
    dirty[..., 0] = np.arange(16, dtype=np.uint8)[:, None, None]
    right = np.full((16, 16), 20.0, dtype=np.float32)
    down = np.full((16, 16), 20.0, dtype=np.float32)
    for tile in range(16):
        right[tile] += np.arange(16, dtype=np.float32) / 100
        down[tile] += np.arange(16, dtype=np.float32) / 100
    right[0, 1] = 0.0
    down[0, 8] = 0.0
    board = build_joint_pose_board(
        layout=layout,
        dirty_tiles=dirty,
        cost_right=right,
        cost_down=down,
        selected_edges=(RawTailEdge(0, 1, "right"), RawTailEdge(0, 8, "down")),
        selected_logits=np.asarray([4.0, -1.0], dtype=np.float32),
        grid=4,
        dense_topk=2,
        candidate_cap=16,
    )
    return board


def test_dirty_visible_descriptor_and_graph_contract() -> None:
    board = _grid4_board()
    assert tile_visible_descriptors(np.zeros((16, 20, 20, 3))).shape == (16, 27)
    assert board.node_features.shape[1] == NODE_FEATURE_DIM
    assert board.pair_index.shape == (2, len(board.pair_features))
    assert board.pair_features.shape[1] == PAIR_FEATURE_DIM
    assert board.candidate_features.shape[1] == CANDIDATE_FEATURE_DIM
    assert np.array_equal(np.sort(board.layout), np.arange(16))


def test_dense_contact_roster_contains_exact_component_shift() -> None:
    board = _grid4_board()
    reference = np.asarray(
        [4, 5, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        dtype=np.int32,
    )
    targets = joint_pose_targets(board, reference, grid=4)
    component = int(board.component_of_tile[0])
    assert board.component_sizes[component] == 2
    assert targets.dominant_shift[component].tolist() == [1, 0]
    assert targets.dominant_support[component] == 2
    assert targets.covered[component]
    assert targets.positive_candidate.sum() >= 1


def test_model_loss_is_finite_and_backpropagates() -> None:
    board = _grid4_board()
    reference = np.asarray(
        [4, 5, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        dtype=np.int32,
    )
    targets = joint_pose_targets(board, reference, grid=4)
    model = JointComponentPoseTransformer(width=32, layers=1, heads=4)
    tensors = {
        "node_features": torch.from_numpy(board.node_features),
        "pair_index": torch.from_numpy(board.pair_index).long(),
        "pair_features": torch.from_numpy(board.pair_features),
        "candidate_component": torch.from_numpy(board.candidate_component).long(),
        "candidate_features": torch.from_numpy(board.candidate_features),
        "candidate_raw_score": torch.from_numpy(board.candidate_raw_score),
    }
    output = model(**tensors)
    loss, parts = joint_pose_loss(
        output,
        candidate_component=tensors["candidate_component"],
        positive_candidate=torch.from_numpy(targets.positive_candidate),
        component_sizes=torch.from_numpy(board.component_sizes).long(),
        dominant_support=torch.from_numpy(targets.dominant_support).float(),
        purity=torch.from_numpy(targets.purity),
        covered=torch.from_numpy(targets.covered),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_multi_anchor_packer_is_strict_and_inference_api_is_target_blind() -> None:
    board = _grid4_board()
    component = int(board.component_of_tile[0])
    layout, diagnostics = pack_multiple_component_anchors(
        board,
        ((component, (1, 0), 1.0),),
        grid=4,
    )
    assert np.array_equal(np.sort(layout), np.arange(16))
    assert diagnostics.placed_anchor_count == 1
    assert diagnostics.strict_permutation
    parameters = set(inspect.signature(build_joint_pose_board).parameters)
    assert not parameters & {"reference", "target", "source_filename", "tile_id"}
