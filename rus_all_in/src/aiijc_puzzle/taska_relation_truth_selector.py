"""Relation-level truth model for selecting one frozen six-arm TASKA layout.

Every candidate arm contributes all 1,104 realised right/down relations.  The
model estimates the probability that each relation is correct using only
inference-visible local evidence and scores a whole layout by the sum of those
probabilities.  It never composes a new layout or changes tile pixels.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES, strict_layout

GRID = 24
COUNT = GRID * GRID
RELATION_COUNT = 2 * GRID * (GRID - 1)
MODEL_PARAMETERS: dict[str, Any] = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 160,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 128,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": 2_026_083_131,
}
PROVENANCE_NAMES = ("current", "selective_new", "unique_fullres")
FEATURE_NAMES = (
    "raw_cost",
    "outgoing_rank_fraction",
    "incoming_rank_fraction",
    "outgoing_margin_from_best",
    "incoming_margin_from_best",
    "outgoing_best_signed_gap",
    "incoming_best_signed_gap",
    "outgoing_cost_zscore",
    "incoming_cost_zscore",
    "axis_is_down",
    "post_six_arm_support",
    "post_six_arm_support_fraction",
    "pre_six_arm_support",
    "pre_tail_realised",
    "tail_created",
    "arm_supply_member",
    "arm_supply_logit",
    "arm_supply_focal_positive",
    "provenance_current",
    "provenance_selective_new",
    "provenance_unique_fullres",
    "realised_by_control",
    "arm_is_control",
    "arm_raw",
    "arm_logistic",
    "arm_focal_top5",
    "arm_nonlinear",
    "arm_selective_vote500_focal",
    "arm_combined_union_focal",
)


class ProbabilityModel(Protocol):
    """Minimal interface needed by the strict whole-layout selector."""

    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...


def _strict(value: Any, *, grid: int, name: str) -> np.ndarray:
    try:
        return strict_layout(value, grid=grid)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a strict {grid * grid}-tile layout") from error


def realised_edges(value: Any, *, grid: int = GRID) -> tuple[RawTailEdge, ...]:
    """Return the fixed horizontal-then-vertical realised relation order."""

    board = _strict(value, grid=grid, name="layout").reshape(grid, grid)
    result = (
        *(
            RawTailEdge(int(board[row, column]), int(board[row, column + 1]), "right")
            for row in range(grid)
            for column in range(grid - 1)
        ),
        *(
            RawTailEdge(int(board[row, column]), int(board[row + 1, column]), "down")
            for row in range(grid - 1)
            for column in range(grid)
        ),
    )
    expected = 2 * grid * (grid - 1)
    if len(result) != expected or len(set(result)) != expected:
        raise RuntimeError("realised relation count changed")
    return result


def _matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.shape != (count, count) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite {count}x{count} matrix")
    return result


def _cost_feature_lookup(
    edges: Sequence[RawTailEdge],
    *,
    cost_right: np.ndarray,
    cost_down: np.ndarray,
) -> dict[RawTailEdge, tuple[float, ...]]:
    """Vectorise two-sided rank and margin features over unique relations."""

    unique = tuple(dict.fromkeys(edges))
    result: dict[RawTailEdge, tuple[float, ...]] = {}
    for axis, matrix in (("right", cost_right), ("down", cost_down)):
        selected = tuple(edge for edge in unique if edge.axis == axis)
        if not selected:
            continue
        count = len(matrix)
        masked = matrix.copy()
        np.fill_diagonal(masked, np.inf)
        row_first_two = np.partition(masked, kth=1, axis=1)[:, :2]
        column_first_two = np.partition(masked.T, kth=1, axis=1)[:, :2]
        row_best = row_first_two.min(axis=1)
        row_second = row_first_two.max(axis=1)
        column_best = column_first_two.min(axis=1)
        column_second = column_first_two.max(axis=1)
        diagonal = np.diag(matrix)
        denominator = float(count - 1)
        row_mean = (matrix.sum(axis=1) - diagonal) / denominator
        column_mean = (matrix.sum(axis=0) - diagonal) / denominator
        row_second_moment = ((matrix * matrix).sum(axis=1) - diagonal * diagonal) / denominator
        column_second_moment = (
            (matrix * matrix).sum(axis=0) - diagonal * diagonal
        ) / denominator
        row_std = np.sqrt(np.maximum(row_second_moment - row_mean * row_mean, 1e-12))
        column_std = np.sqrt(
            np.maximum(column_second_moment - column_mean * column_mean, 1e-12)
        )
        sources = np.fromiter((edge.source for edge in selected), dtype=np.int32)
        targets = np.fromiter((edge.target for edge in selected), dtype=np.int32)
        costs = matrix[sources, targets]
        outgoing_rank = np.empty(len(selected), dtype=np.float64)
        incoming_rank = np.empty(len(selected), dtype=np.float64)
        for start in range(0, len(selected), 512):
            stop = min(start + 512, len(selected))
            values = costs[start:stop, None]
            outgoing_rank[start:stop] = np.count_nonzero(
                masked[sources[start:stop]] < values, axis=1
            )
            incoming_rank[start:stop] = np.count_nonzero(
                values > masked[:, targets[start:stop]].T, axis=1
            )
        outgoing_rank /= denominator
        incoming_rank /= denominator
        outgoing_signed = np.where(
            outgoing_rank == 0.0,
            row_second[sources] - costs,
            row_best[sources] - costs,
        )
        incoming_signed = np.where(
            incoming_rank == 0.0,
            column_second[targets] - costs,
            column_best[targets] - costs,
        )
        for index, edge in enumerate(selected):
            result[edge] = (
                float(costs[index]),
                float(outgoing_rank[index]),
                float(incoming_rank[index]),
                float(costs[index] - row_best[sources[index]]),
                float(costs[index] - column_best[targets[index]]),
                float(outgoing_signed[index]),
                float(incoming_signed[index]),
                float((costs[index] - row_mean[sources[index]]) / row_std[sources[index]]),
                float(
                    (costs[index] - column_mean[targets[index]])
                    / column_std[targets[index]]
                ),
                float(axis == "down"),
            )
    if len(result) != len(unique):
        raise RuntimeError("cost feature lookup omitted relations")
    return result


def _supply_map(
    edges: Sequence[RawTailEdge], logits: Any, *, name: str
) -> dict[RawTailEdge, float]:
    edge_values = tuple(edges)
    score_values = np.ascontiguousarray(logits, dtype=np.float64)
    if len(set(edge_values)) != len(edge_values) or score_values.shape != (len(edge_values),):
        raise ValueError(f"{name} supply edges/logits are duplicated or misaligned")
    if not np.isfinite(score_values).all():
        raise ValueError(f"{name} supply logits are non-finite")
    return {
        edge: float(logit)
        for edge, logit in zip(edge_values, score_values, strict=True)
    }


@dataclass(frozen=True)
class RelationFeatureBoard:
    """All local rows for the six strict layouts of one target-free board."""

    layouts: tuple[np.ndarray, ...]
    edges: tuple[tuple[RawTailEdge, ...], ...]
    features: np.ndarray
    control_choice: str
    grid_size: int = GRID

    def __post_init__(self) -> None:
        relation_count = 2 * self.grid_size * (self.grid_size - 1)
        if len(self.layouts) != len(FUSION_ARM_NAMES) or len(self.edges) != len(
            FUSION_ARM_NAMES
        ):
            raise ValueError("six-arm roster changed")
        layouts: list[np.ndarray] = []
        for arm, layout, edges in zip(FUSION_ARM_NAMES, self.layouts, self.edges, strict=True):
            current = _strict(layout, grid=self.grid_size, name=f"layouts[{arm!r}]")
            current.setflags(write=False)
            layouts.append(current)
            if len(edges) != relation_count or tuple(edges) != realised_edges(
                current, grid=self.grid_size
            ):
                raise ValueError(f"edges for {arm!r} do not match its layout")
        object.__setattr__(self, "layouts", tuple(layouts))
        values = np.ascontiguousarray(self.features, dtype=np.float64)
        expected = (len(FUSION_ARM_NAMES), relation_count, len(FEATURE_NAMES))
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"features must be finite with shape {expected}")
        values.setflags(write=False)
        object.__setattr__(self, "features", values)
        if self.control_choice not in FUSION_ARM_NAMES:
            raise ValueError("control choice is outside the fixed arm roster")

    def labels(self, truth_edges: Sequence[RawTailEdge]) -> np.ndarray:
        """Return binary truth for every arm relation in fixed row order."""

        truth = set(truth_edges)
        values = np.asarray(
            [[edge in truth for edge in edges] for edges in self.edges], dtype=np.uint8
        )
        expected = self.features.shape[:2]
        if values.shape != expected:
            raise RuntimeError("truth labels do not align with relation rows")
        return values


def relation_feature_board(
    *,
    post_tail_layouts: Mapping[str, Any],
    pre_tail_layouts: Mapping[str, Any],
    cost_right: Any,
    cost_down: Any,
    arm_edges: Mapping[str, Sequence[RawTailEdge]],
    arm_logits: Mapping[str, Any],
    provenance: Mapping[str, Sequence[RawTailEdge]],
    control_choice: str,
    grid: int = GRID,
) -> RelationFeatureBoard:
    """Extract the one fixed inference-visible feature table for a board."""

    roster = tuple(FUSION_ARM_NAMES)
    if tuple(post_tail_layouts) != roster or tuple(pre_tail_layouts) != roster:
        raise ValueError("pre/post layout roster or order changed")
    if tuple(arm_edges) != roster or tuple(arm_logits) != roster:
        raise ValueError("arm supply roster or order changed")
    if tuple(provenance) != PROVENANCE_NAMES:
        raise ValueError("provenance roster or order changed")
    if control_choice not in roster:
        raise ValueError("control choice is outside the fixed arm roster")
    count = grid * grid
    matrices = {
        "right": _matrix(cost_right, count=count, name="cost_right"),
        "down": _matrix(cost_down, count=count, name="cost_down"),
    }
    post_layout_values = tuple(
        _strict(post_tail_layouts[arm], grid=grid, name=f"post_tail_layouts[{arm!r}]")
        for arm in roster
    )
    pre_layout_values = tuple(
        _strict(pre_tail_layouts[arm], grid=grid, name=f"pre_tail_layouts[{arm!r}]")
        for arm in roster
    )
    post_edges = tuple(realised_edges(layout, grid=grid) for layout in post_layout_values)
    pre_edges = tuple(realised_edges(layout, grid=grid) for layout in pre_layout_values)
    post_support = Counter(edge for edges in post_edges for edge in edges)
    pre_support = Counter(edge for edges in pre_edges for edge in edges)
    all_edges = tuple(edge for edges in post_edges for edge in edges)
    cost_lookup = _cost_feature_lookup(
        all_edges,
        cost_right=matrices["right"],
        cost_down=matrices["down"],
    )
    supply_maps = {
        arm: _supply_map(arm_edges[arm], arm_logits[arm], name=arm) for arm in roster
    }
    provenance_sets = {name: set(provenance[name]) for name in PROVENANCE_NAMES}
    control_edges = set(post_edges[roster.index(control_choice)])
    relation_count = 2 * grid * (grid - 1)
    features = np.empty(
        (len(roster), relation_count, len(FEATURE_NAMES)), dtype=np.float64
    )
    for arm_index, (arm, arm_post_edges, arm_pre_edges) in enumerate(
        zip(roster, post_edges, pre_edges, strict=True)
    ):
        pre_set = set(arm_pre_edges)
        supply = supply_maps[arm]
        arm_values = [float(index == arm_index) for index in range(len(roster))]
        for relation_index, edge in enumerate(arm_post_edges):
            supply_member = edge in supply
            supply_logit = supply.get(edge, 0.0)
            pre_realised = edge in pre_set
            features[arm_index, relation_index] = (
                *cost_lookup[edge],
                float(post_support[edge]),
                post_support[edge] / float(len(roster)),
                float(pre_support[edge]),
                float(pre_realised),
                float(not pre_realised),
                float(supply_member),
                float(supply_logit),
                float(supply_member and supply_logit >= 0.0),
                *(float(edge in provenance_sets[name]) for name in PROVENANCE_NAMES),
                float(edge in control_edges),
                float(arm == control_choice),
                *arm_values,
            )
    return RelationFeatureBoard(
        layouts=post_layout_values,
        edges=post_edges,
        features=features,
        control_choice=control_choice,
        grid_size=grid,
    )


def fit_relation_truth_classifier(
    boards: Sequence[RelationFeatureBoard], labels: Sequence[np.ndarray]
) -> HistGradientBoostingClassifier:
    """Fit exactly one fixed nonlinear classifier over all realised seams."""

    board_values = tuple(boards)
    label_values = tuple(np.asarray(value, dtype=np.uint8) for value in labels)
    if not board_values or len(board_values) != len(label_values):
        raise ValueError("boards and labels must be non-empty and aligned")
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for board, target in zip(board_values, label_values, strict=True):
        if target.shape != board.features.shape[:2] or not np.isin(target, (0, 1)).all():
            raise ValueError("binary labels do not align with relation rows")
        rows.append(board.features.reshape(-1, len(FEATURE_NAMES)))
        targets.append(target.reshape(-1))
    y = np.concatenate(targets)
    if len(np.unique(y)) != 2:
        raise ValueError("training labels must contain both truth classes")
    model = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    model.fit(np.concatenate(rows), y)
    if model.classes_.tolist() != [0, 1]:
        raise RuntimeError("unexpected classifier class order")
    return model


def expected_correct_scores(
    board: RelationFeatureBoard, model: ProbabilityModel
) -> np.ndarray:
    """Sum expected-correct relation probabilities for every whole layout."""

    probabilities = np.asarray(
        model.predict_proba(board.features.reshape(-1, len(FEATURE_NAMES))),
        dtype=np.float64,
    )
    expected_rows = int(np.prod(board.features.shape[:2]))
    if probabilities.shape != (expected_rows, 2) or not np.isfinite(probabilities).all():
        raise RuntimeError("classifier returned invalid probabilities")
    positive = probabilities[:, 1].reshape(board.features.shape[:2])
    scores = np.ascontiguousarray(positive.sum(axis=1))
    scores.setflags(write=False)
    return scores


def select_relation_truth_layout(
    board: RelationFeatureBoard, model: ProbabilityModel
) -> tuple[str, np.ndarray, np.ndarray]:
    """Select exactly one arm; preserve control on an exact expected-score tie."""

    scores = expected_correct_scores(board, model)
    maximum = float(np.max(scores))
    tied = np.flatnonzero(scores == maximum)
    control_index = FUSION_ARM_NAMES.index(board.control_choice)
    choice_index = control_index if control_index in tied else int(tied[0])
    layout = board.layouts[choice_index].copy()
    layout.setflags(write=False)
    return FUSION_ARM_NAMES[choice_index], layout, scores


__all__ = [
    "COUNT",
    "FEATURE_NAMES",
    "GRID",
    "MODEL_PARAMETERS",
    "PROVENANCE_NAMES",
    "RELATION_COUNT",
    "RelationFeatureBoard",
    "expected_correct_scores",
    "fit_relation_truth_classifier",
    "realised_edges",
    "relation_feature_board",
    "select_relation_truth_layout",
]
