from __future__ import annotations

from itertools import permutations

import numpy as np

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES
from aiijc_puzzle.taska_six_arm_learned_selector import (
    FEATURE_NAMES,
    FrozenPairwiseRidgeSelector,
    SixArmTargetFreeBoard,
    board_relative_design,
    fit_pairwise_ridge_selector,
    prepare_six_arm_target_free_board,
    select_with_frozen_ridge,
)


def test_prepare_six_arm_board_replays_independent_control() -> None:
    layout = np.arange(4)
    layouts = {arm: layout for arm in FUSION_ARM_NAMES}
    edges = {arm: (RawTailEdge(0, 1, "right"),) for arm in FUSION_ARM_NAMES}
    logits = {arm: np.asarray([1.0]) for arm in FUSION_ARM_NAMES}
    cost = np.zeros((4, 4))

    board = prepare_six_arm_target_free_board(
        pre_tail_layouts=layouts,
        cost_right=cost,
        cost_down=cost,
        arm_edges=edges,
        arm_logits=logits,
        control_choice="raw",
        frozen_control_layout=layout,
        grid=2,
    )

    assert board.features.shape == (6, len(FEATURE_NAMES))
    assert np.isfinite(board.features).all()
    assert all(not value.flags.writeable for value in board.layouts)
    np.testing.assert_array_equal(board.control_layout, layout)


def test_pairwise_ridge_learns_fixed_linear_board_ranking() -> None:
    rng = np.random.default_rng(20260831)
    features = rng.normal(size=(24, 6, len(FEATURE_NAMES)))
    arm_bias = np.linspace(-2.0, 2.0, 6)
    labels = 3.0 * features[:, :, 0] - features[:, :, 1] + arm_bias

    model = fit_pairwise_ridge_selector(features, labels)
    agreement = np.mean(
        [
            np.argmax(model.scores(board)) == np.argmax(target)
            for board, target in zip(features, labels, strict=True)
        ]
    )

    assert agreement >= 0.9
    assert np.isfinite(model.coefficients).all()


def test_board_relative_design_centers_features_and_appends_arm_identity() -> None:
    features = np.arange(6 * len(FEATURE_NAMES), dtype=float).reshape(6, -1)
    design = board_relative_design(features)

    np.testing.assert_allclose(design[:, : len(FEATURE_NAMES)].mean(axis=0), 0.0)
    np.testing.assert_array_equal(design[:, len(FEATURE_NAMES) :], np.eye(6))


def test_exact_score_tie_retains_control_arm() -> None:
    layouts = tuple(
        np.asarray(value) for value in list(permutations(range(4)))[:6]
    )
    control_choice = FUSION_ARM_NAMES[1]
    board = SixArmTargetFreeBoard(
        layouts=layouts,
        features=np.zeros((6, len(FEATURE_NAMES))),
        diagnostics=tuple({} for _ in FUSION_ARM_NAMES),
        control_choice=control_choice,
        control_layout=layouts[1],
        grid_size=2,
    )
    size = len(FEATURE_NAMES) + len(FUSION_ARM_NAMES)
    model = FrozenPairwiseRidgeSelector(
        scaler_mean=np.zeros(size),
        scaler_scale=np.ones(size),
        coefficients=np.zeros(size),
    )

    choice, layout, scores = select_with_frozen_ridge(board, model)

    assert choice == control_choice
    np.testing.assert_array_equal(layout, layouts[1])
    np.testing.assert_array_equal(scores, np.zeros(6))
