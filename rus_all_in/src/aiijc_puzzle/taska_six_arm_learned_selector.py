"""Fixed board-relative ridge selector for six post-tail TASKA layouts.

The module never reads a target.  It independently applies the already fixed
focal-gated tail96 to each arm, extracts inference-visible per-arm features,
and exposes one deterministic StandardScaler + pairwise Ridge contract.  The
training labels and staged evaluation live in the research runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_gated_protected_tail import polish_taska_tail_with_focal_gate
from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES, strict_layout
from aiijc_puzzle.taska_six_arm_consensus_selector import directed_adjacencies

RIDGE_ALPHA = 1.0
FEATURE_NAMES = (
    "pre_raw_cost",
    "post_raw_cost",
    "tail_raw_gain",
    "accepted_swaps",
    "protected_tiles",
    "free_tiles",
    "initial_realised_focal_edges",
    "final_realised_focal_edges",
    "focal_kept_edges",
    "focal_kept_fraction",
    "focal_kept_logit_sum",
    "focal_kept_logit_mean",
    "pre_realised_focal_logit_sum",
    "post_realised_focal_logit_sum",
    "mean_tile_position_agreement",
    "mean_directed_adjacency_agreement",
)


def _edges(value: Sequence[RawTailEdge], *, name: str) -> tuple[RawTailEdge, ...]:
    result = tuple(value)
    if not all(isinstance(edge, RawTailEdge) for edge in result):
        raise TypeError(f"{name} must contain only RawTailEdge values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate edges")
    return result


def _logits(value: Any, *, count: int, name: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain one finite logit per edge")
    return result


def _realised_positive(
    layout: np.ndarray,
    edges: tuple[RawTailEdge, ...],
    logits: np.ndarray,
    *,
    grid: int,
) -> tuple[int, float]:
    realised = directed_adjacencies(layout, grid=grid)
    selected = [
        float(logit)
        for edge, logit in zip(edges, logits, strict=True)
        if logit >= 0.0 and (edge.axis, edge.source, edge.target) in realised
    ]
    return len(selected), float(np.sum(selected, dtype=np.float64))


@dataclass(frozen=True)
class SixArmTargetFreeBoard:
    """Six independently polished strict layouts and their target-free features."""

    layouts: tuple[np.ndarray, ...]
    features: np.ndarray
    diagnostics: tuple[dict[str, Any], ...]
    control_choice: str
    control_layout: np.ndarray
    grid_size: int = 24

    def __post_init__(self) -> None:
        if len(self.layouts) != len(FUSION_ARM_NAMES):
            raise ValueError("layout roster must be the fixed six-arm roster")
        frozen_layouts: list[np.ndarray] = []
        for _arm, layout in zip(FUSION_ARM_NAMES, self.layouts, strict=True):
            current = strict_layout(layout, grid=self.grid_size)
            current.setflags(write=False)
            frozen_layouts.append(current)
        object.__setattr__(self, "layouts", tuple(frozen_layouts))
        matrix = np.ascontiguousarray(self.features, dtype=np.float64)
        expected = (len(FUSION_ARM_NAMES), len(FEATURE_NAMES))
        if matrix.shape != expected or not np.isfinite(matrix).all():
            raise ValueError(f"features must be finite with shape {expected}")
        matrix.setflags(write=False)
        object.__setattr__(self, "features", matrix)
        if self.control_choice not in FUSION_ARM_NAMES:
            raise ValueError("control_choice is outside the fixed six-arm roster")
        control = strict_layout(self.control_layout, grid=self.grid_size)
        control.setflags(write=False)
        object.__setattr__(self, "control_layout", control)


def prepare_six_arm_target_free_board(
    *,
    pre_tail_layouts: Mapping[str, Any],
    cost_right: Any,
    cost_down: Any,
    arm_edges: Mapping[str, Sequence[RawTailEdge]],
    arm_logits: Mapping[str, Any],
    control_choice: str,
    frozen_control_layout: Any,
    grid: int = 24,
) -> SixArmTargetFreeBoard:
    """Independently tail-polish all six arms and extract fixed features."""

    roster = tuple(FUSION_ARM_NAMES)
    if tuple(pre_tail_layouts) != roster:
        raise ValueError("pre_tail_layouts roster or order changed")
    if tuple(arm_edges) != roster or tuple(arm_logits) != roster:
        raise ValueError("edge/logit roster or order changed")
    if control_choice not in roster:
        raise ValueError("control_choice is outside the fixed six-arm roster")
    polished: list[np.ndarray] = []
    partial: list[list[float]] = []
    diagnostics: list[dict[str, Any]] = []
    for arm in roster:
        before = strict_layout(pre_tail_layouts[arm], grid=grid)
        edges = _edges(arm_edges[arm], name=f"arm_edges[{arm!r}]")
        logits = _logits(arm_logits[arm], count=len(edges), name=f"arm_logits[{arm!r}]")
        result = polish_taska_tail_with_focal_gate(
            before,
            cost_right,
            cost_down,
            edges,
            logits,
            grid=grid,
        )
        after = result.layout
        pre_cost = total_taska_adjacent_seam_cost(
            before, cost_right, cost_down, grid=grid
        )
        post_cost = total_taska_adjacent_seam_cost(
            after, cost_right, cost_down, grid=grid
        )
        keep = logits >= 0.0
        kept = logits[keep]
        pre_count, pre_logit_sum = _realised_positive(
            before, edges, logits, grid=grid
        )
        post_count, post_logit_sum = _realised_positive(
            after, edges, logits, grid=grid
        )
        tail = result.diagnostics.tail
        partial.append(
            [
                pre_cost,
                post_cost,
                pre_cost - post_cost,
                float(tail.accepted_swap_count),
                float(tail.protected_tile_count),
                float(tail.free_tile_count),
                float(pre_count),
                float(post_count),
                float(keep.sum()),
                float(keep.mean()) if len(keep) else 0.0,
                float(kept.sum(dtype=np.float64)),
                float(kept.mean()) if len(kept) else 0.0,
                pre_logit_sum,
                post_logit_sum,
            ]
        )
        polished.append(after)
        diagnostics.append(
            {
                "arm": arm,
                "pre_realised_focal_positive_edges": pre_count,
                "post_realised_focal_positive_edges": post_count,
                "pre_realised_focal_logit_sum": pre_logit_sum,
                "post_realised_focal_logit_sum": post_logit_sum,
                "tail": asdict(result.diagnostics),
            }
        )
    adjacency_sets = [directed_adjacencies(layout, grid=grid) for layout in polished]
    full_features: list[list[float]] = []
    for index, layout in enumerate(polished):
        position_agreement = np.mean(
            [
                np.mean(layout == other)
                for other_index, other in enumerate(polished)
                if other_index != index
            ]
        )
        adjacency_agreement = np.mean(
            [
                len(adjacency_sets[index] & other) / len(adjacency_sets[index])
                for other_index, other in enumerate(adjacency_sets)
                if other_index != index
            ]
        )
        full_features.append(
            [*partial[index], float(position_agreement), float(adjacency_agreement)]
        )
        diagnostics[index]["mean_tile_position_agreement"] = float(position_agreement)
        diagnostics[index]["mean_directed_adjacency_agreement"] = float(
            adjacency_agreement
        )
    control = strict_layout(frozen_control_layout, grid=grid)
    control_index = roster.index(control_choice)
    if not np.array_equal(polished[control_index], control):
        raise RuntimeError("independent control arm does not replay frozen fusion")
    return SixArmTargetFreeBoard(
        layouts=tuple(polished),
        features=np.asarray(full_features, dtype=np.float64),
        diagnostics=tuple(diagnostics),
        control_choice=control_choice,
        control_layout=control,
        grid_size=grid,
    )


@dataclass(frozen=True)
class FrozenPairwiseRidgeSelector:
    """Serializable fixed StandardScaler + Ridge parameters."""

    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficients: np.ndarray
    alpha: float = RIDGE_ALPHA

    def __post_init__(self) -> None:
        size = len(FEATURE_NAMES) + len(FUSION_ARM_NAMES)
        for name in ("scaler_mean", "scaler_scale", "coefficients"):
            value = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {(size,)}")
            if name == "scaler_scale" and np.any(value <= 0.0):
                raise ValueError("scaler_scale must be positive")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if float(self.alpha) != RIDGE_ALPHA:
            raise ValueError("ridge alpha changed")

    def scores(self, board_features: Any) -> np.ndarray:
        design = board_relative_design(board_features)
        transformed = (design - self.scaler_mean) / self.scaler_scale
        result = np.ascontiguousarray(transformed @ self.coefficients)
        if result.shape != (len(FUSION_ARM_NAMES),) or not np.isfinite(result).all():
            raise RuntimeError("selector produced invalid scores")
        result.setflags(write=False)
        return result


def board_relative_design(board_features: Any) -> np.ndarray:
    """Center continuous features within a board and append arm contrasts."""

    features = np.ascontiguousarray(board_features, dtype=np.float64)
    expected = (len(FUSION_ARM_NAMES), len(FEATURE_NAMES))
    if features.shape != expected or not np.isfinite(features).all():
        raise ValueError(f"board_features must be finite with shape {expected}")
    centered = features - features.mean(axis=0, keepdims=True)
    return np.concatenate((centered, np.eye(len(FUSION_ARM_NAMES))), axis=1)


def fit_pairwise_ridge_selector(
    feature_boards: Any,
    pair_labels: Any,
) -> FrozenPairwiseRidgeSelector:
    """Fit the one preregistered ordered-pair StandardScaler + Ridge model."""

    features = np.ascontiguousarray(feature_boards, dtype=np.float64)
    labels = np.ascontiguousarray(pair_labels, dtype=np.float64)
    expected_tail = (len(FUSION_ARM_NAMES), len(FEATURE_NAMES))
    if features.ndim != 3 or features.shape[1:] != expected_tail:
        raise ValueError("feature_boards has an invalid shape")
    if labels.shape != features.shape[:2] or not np.isfinite(labels).all():
        raise ValueError("pair_labels must align with feature_boards")
    if len(features) < 2 or not np.isfinite(features).all():
        raise ValueError("at least two finite training boards are required")
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for board_features, board_labels in zip(features, labels, strict=True):
        design = board_relative_design(board_features)
        for left in range(len(FUSION_ARM_NAMES)):
            for right in range(len(FUSION_ARM_NAMES)):
                if left == right:
                    continue
                rows.append(design[left] - design[right])
                targets.append(float(board_labels[left] - board_labels[right]))
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(scaler.transform(x), y)
    return FrozenPairwiseRidgeSelector(
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        coefficients=model.coef_,
    )


def select_with_frozen_ridge(
    board: SixArmTargetFreeBoard,
    model: FrozenPairwiseRidgeSelector,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Choose a whole arm; retain the control arm on an exact score tie."""

    scores = model.scores(board.features)
    maximum = float(np.max(scores))
    tied = np.flatnonzero(scores == maximum)
    control_index = FUSION_ARM_NAMES.index(board.control_choice)
    choice_index = control_index if control_index in tied else int(tied[0])
    layout = board.layouts[choice_index].copy()
    layout.setflags(write=False)
    return FUSION_ARM_NAMES[choice_index], layout, scores


__all__ = [
    "FEATURE_NAMES",
    "RIDGE_ALPHA",
    "FrozenPairwiseRidgeSelector",
    "SixArmTargetFreeBoard",
    "board_relative_design",
    "fit_pairwise_ridge_selector",
    "prepare_six_arm_target_free_board",
    "select_with_frozen_ridge",
]
