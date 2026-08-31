from __future__ import annotations

from itertools import permutations

import numpy as np

from aiijc_puzzle.taska_relation_truth_selector import (
    FEATURE_NAMES,
    MODEL_PARAMETERS,
    PROVENANCE_NAMES,
    RelationFeatureBoard,
    fit_relation_truth_classifier,
    realised_edges,
    relation_feature_board,
    select_relation_truth_layout,
)
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES


def _layouts() -> dict[str, np.ndarray]:
    values = [np.asarray(value) for value in list(permutations(range(4)))[:6]]
    return dict(zip(FUSION_ARM_NAMES, values, strict=True))


def test_relation_features_cover_every_realised_edge_of_every_arm() -> None:
    layouts = _layouts()
    rng = np.random.default_rng(20260831)
    cost_right = rng.uniform(0.1, 4.0, size=(4, 4))
    cost_down = rng.uniform(0.1, 4.0, size=(4, 4))
    edges = {
        arm: (realised_edges(layout, grid=2)[0],)
        for arm, layout in layouts.items()
    }
    logits = {arm: np.asarray([index - 2.0]) for index, arm in enumerate(layouts)}
    all_supply = tuple(dict.fromkeys(edge[0] for edge in edges.values()))
    provenance = {
        PROVENANCE_NAMES[0]: all_supply,
        PROVENANCE_NAMES[1]: (),
        PROVENANCE_NAMES[2]: (),
    }

    board = relation_feature_board(
        post_tail_layouts=layouts,
        pre_tail_layouts=layouts,
        cost_right=cost_right,
        cost_down=cost_down,
        arm_edges=edges,
        arm_logits=logits,
        provenance=provenance,
        control_choice=FUSION_ARM_NAMES[2],
        grid=2,
    )

    assert board.features.shape == (6, 4, len(FEATURE_NAMES))
    assert np.isfinite(board.features).all()
    assert all(
        tuple(edge) == realised_edges(layout, grid=2)
        for edge, layout in zip(board.edges, board.layouts, strict=True)
    )
    labels = board.labels(board.edges[0])
    assert labels.shape == (6, 4)
    assert labels[0].sum() == 4


def test_fixed_histogram_classifier_learns_relation_signal() -> None:
    rng = np.random.default_rng(2026083101)
    layouts = tuple(_layouts().values())
    edges = tuple(realised_edges(layout, grid=2) for layout in layouts)
    boards: list[RelationFeatureBoard] = []
    labels: list[np.ndarray] = []
    for _ in range(40):
        features = rng.normal(size=(6, 4, len(FEATURE_NAMES)))
        target = (features[:, :, 0] + 0.25 * features[:, :, 1] > 0.0).astype(np.uint8)
        boards.append(
            RelationFeatureBoard(
                layouts=layouts,
                edges=edges,
                features=features,
                control_choice=FUSION_ARM_NAMES[0],
                grid_size=2,
            )
        )
        labels.append(target)

    model = fit_relation_truth_classifier(boards, labels)
    features = np.concatenate(
        [board.features.reshape(-1, len(FEATURE_NAMES)) for board in boards]
    )
    target = np.concatenate([value.reshape(-1) for value in labels])
    probability = model.predict_proba(features)[:, 1]

    assert model.get_params()["max_iter"] == MODEL_PARAMETERS["max_iter"]
    assert probability[target == 1].mean() > probability[target == 0].mean() + 0.5


def test_exact_score_tie_retains_control_whole_layout() -> None:
    class UniformModel:
        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            return np.full((len(features), 2), 0.5)

    layouts = tuple(_layouts().values())
    control = FUSION_ARM_NAMES[4]
    board = RelationFeatureBoard(
        layouts=layouts,
        edges=tuple(realised_edges(layout, grid=2) for layout in layouts),
        features=np.zeros((6, 4, len(FEATURE_NAMES))),
        control_choice=control,
        grid_size=2,
    )

    choice, layout, scores = select_relation_truth_layout(board, UniformModel())

    assert choice == control
    np.testing.assert_array_equal(layout, layouts[4])
    np.testing.assert_array_equal(scores, np.full(6, 2.0))
