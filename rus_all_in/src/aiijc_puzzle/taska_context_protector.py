# ruff: noqa: E501
"""Target-free context filter for realised six-arm TASKA supply edges.

This is deliberately a *post-solver* research primitive.  It sees only an
already selected pre-tail layout and frozen matcher/solver evidence.  It never
constructs new edges, changes a seam matrix, or looks at a target.  Its binary
output says which of the already realised focal-positive supply edges should
freeze their endpoint tiles for the unchanged raw non-adjacent tail.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES, strict_layout
from aiijc_puzzle.taska_six_arm_consensus_selector import directed_adjacencies

GRID = 24
COUNT = GRID * GRID
DECISION_THRESHOLD = 0.5
FEATURE_NAMES = (
    "focal_logit",
    "raw_priority_negative_cost",
    "raw_outgoing_rank_fraction",
    "raw_incoming_rank_fraction",
    "raw_outgoing_margin_from_best",
    "raw_incoming_margin_from_best",
    "axis_is_down",
    "provenance_current",
    "provenance_selective_new",
    "provenance_unique_fullres",
    "selected_arm_raw",
    "selected_arm_logistic",
    "selected_arm_focal_top5",
    "selected_arm_nonlinear",
    "selected_arm_selective_vote500_focal",
    "selected_arm_combined_union_focal",
    "six_arm_realisation_count",
    "six_arm_realisation_fraction",
    "realised_component_tile_count",
    "source_realised_degree",
    "target_realised_degree",
    "component_mean_six_arm_support",
    "same_source_axis_conflict_count",
    "same_target_axis_conflict_count",
    "source_competing_logit_margin",
    "target_competing_logit_margin",
    "incident_conflict_density",
)


def _matrix(value: Any, *, name: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.shape != (COUNT, COUNT) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite {COUNT}x{COUNT} matrix")
    return result


def _validate_edges(
    edges: Sequence[RawTailEdge], logits: Any
) -> tuple[tuple[RawTailEdge, ...], np.ndarray]:
    result = tuple(edges)
    values = np.ascontiguousarray(logits, dtype=np.float64)
    if len(set(result)) != len(result) or values.shape != (len(result),):
        raise ValueError("supply edges/logits have duplicate or misaligned rows")
    if not np.isfinite(values).all():
        raise ValueError("supply logits must be finite")
    for edge in result:
        if not isinstance(edge, RawTailEdge) or edge.axis not in {"right", "down"}:
            raise TypeError("supply must contain directed right/down RawTailEdge values")
        if (
            not 0 <= edge.source < COUNT
            or not 0 <= edge.target < COUNT
            or edge.source == edge.target
        ):
            raise ValueError("supply edge is out of range or self-referential")
    return result, values


@dataclass(frozen=True)
class ContextEdgeRows:
    """Frozen rows and their corresponding realised edges, in solver order."""

    edges: tuple[RawTailEdge, ...]
    features: np.ndarray
    selected_arm: str

    def __post_init__(self) -> None:
        if self.selected_arm not in FUSION_ARM_NAMES:
            raise ValueError("selected arm is outside the frozen six-arm roster")
        values = np.ascontiguousarray(self.features, dtype=np.float64)
        if values.shape != (len(self.edges), len(FEATURE_NAMES)) or not np.isfinite(values).all():
            raise ValueError("context features have an invalid shape or non-finite value")
        values.setflags(write=False)
        object.__setattr__(self, "features", values)


def _component_statistics(
    edges: tuple[RawTailEdge, ...], arm_support: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-edge component size, endpoint degree, component mean support."""

    parent = np.arange(COUNT, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def join(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    degrees = np.zeros(COUNT, dtype=np.int32)
    for edge in edges:
        join(edge.source, edge.target)
        degrees[edge.source] += 1
        degrees[edge.target] += 1
    members: dict[int, list[int]] = defaultdict(list)
    for tile in range(COUNT):
        members[find(tile)].append(tile)
    comp_edges: dict[int, list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        comp_edges[find(edge.source)].append(index)
    sizes = np.empty(len(edges), dtype=np.float64)
    means = np.empty(len(edges), dtype=np.float64)
    for index, edge in enumerate(edges):
        root = find(edge.source)
        sizes[index] = len(members[root])
        means[index] = float(np.mean(arm_support[comp_edges[root]]))
    return sizes, degrees, means, parent


def realised_context_features(
    *,
    selected_layout: Any,
    selected_arm: str,
    selected_edges: Sequence[RawTailEdge],
    selected_logits: Any,
    provenance: Mapping[str, Sequence[RawTailEdge]],
    pre_tail_layouts: Mapping[str, Any],
    cost_right: Any,
    cost_down: Any,
    grid: int = GRID,
) -> ContextEdgeRows:
    """Make the exact fixed target-free feature table for one selected board."""

    if grid != GRID:
        raise ValueError("this fixed experiment is registered for a 24x24 board")
    if selected_arm not in FUSION_ARM_NAMES or tuple(pre_tail_layouts) != FUSION_ARM_NAMES:
        raise ValueError("six-arm roster/order changed")
    edges, logits = _validate_edges(selected_edges, selected_logits)
    expected_provenance = ("current", "selective_new", "unique_fullres")
    if tuple(provenance) != expected_provenance:
        raise ValueError("provenance roster/order changed")
    provenance_sets = {name: set(provenance[name]) for name in expected_provenance}
    if provenance_sets["current"] & provenance_sets["selective_new"]:
        raise ValueError("current and selective-new supply must be disjoint")
    if (provenance_sets["current"] | provenance_sets["selective_new"]) & provenance_sets[
        "unique_fullres"
    ]:
        raise ValueError("unique-fullres supply must be novel")
    if not set(edges).issubset(set().union(*provenance_sets.values())):
        raise ValueError("selected supply edge is absent from provenance")

    selected = strict_layout(selected_layout, grid=grid)
    realised = directed_adjacencies(selected, grid=grid)
    selected_indices = np.asarray(
        [
            logit >= 0.0 and (edge.axis, edge.source, edge.target) in realised
            for edge, logit in zip(edges, logits, strict=True)
        ],
        dtype=bool,
    )
    kept_edges = tuple(edge for edge, keep in zip(edges, selected_indices, strict=True) if keep)
    kept_logits = logits[selected_indices]
    if not kept_edges:
        return ContextEdgeRows(
            edges=(), features=np.empty((0, len(FEATURE_NAMES))), selected_arm=selected_arm
        )

    matrices = {
        "right": _matrix(cost_right, name="cost_right"),
        "down": _matrix(cost_down, name="cost_down"),
    }
    arm_realised = {
        arm: directed_adjacencies(strict_layout(layout, grid=grid), grid=grid)
        for arm, layout in pre_tail_layouts.items()
    }
    arm_support = np.asarray(
        [
            sum((edge.axis, edge.source, edge.target) in value for value in arm_realised.values())
            for edge in kept_edges
        ],
        dtype=np.float64,
    )
    component_sizes, degrees, component_means, _ = _component_statistics(kept_edges, arm_support)

    # Conflicts are alternative focal-positive selected-supply hypotheses with
    # the same oriented source or target, never labels or layout coordinates.
    positive_by_source: dict[tuple[int, str], list[tuple[RawTailEdge, float]]] = defaultdict(list)
    positive_by_target: dict[tuple[int, str], list[tuple[RawTailEdge, float]]] = defaultdict(list)
    for edge, logit in zip(edges, logits, strict=True):
        if logit >= 0.0:
            positive_by_source[(edge.source, edge.axis)].append((edge, float(logit)))
            positive_by_target[(edge.target, edge.axis)].append((edge, float(logit)))

    output = np.empty((len(kept_edges), len(FEATURE_NAMES)), dtype=np.float64)
    arm_index = FUSION_ARM_NAMES.index(selected_arm)
    denominator = float(COUNT - 1)
    positive_total = float(sum(len(value) for value in positive_by_source.values()))
    for index, (edge, logit) in enumerate(zip(kept_edges, kept_logits, strict=True)):
        matrix = matrices[edge.axis]
        cost = float(matrix[edge.source, edge.target])
        outgoing = np.concatenate(
            (matrix[edge.source, : edge.source], matrix[edge.source, edge.source + 1 :])
        )
        incoming = np.concatenate(
            (matrix[: edge.target, edge.target], matrix[edge.target + 1 :, edge.target])
        )
        source_logits = positive_by_source[(edge.source, edge.axis)]
        target_logits = positive_by_target[(edge.target, edge.axis)]
        source_other = [value for other, value in source_logits if other != edge]
        target_other = [value for other, value in target_logits if other != edge]
        source_count, target_count = len(source_other), len(target_other)
        # No competitor means the edge has its own logit as a finite fixed
        # reference; this avoids injecting a tuned sentinel into the model.
        source_margin = float(logit) - (max(source_other) if source_other else 0.0)
        target_margin = float(logit) - (max(target_other) if target_other else 0.0)
        provenance_values = [float(edge in provenance_sets[name]) for name in expected_provenance]
        arm_values = [float(position == arm_index) for position in range(len(FUSION_ARM_NAMES))]
        output[index] = (
            float(logit),
            -cost,
            np.count_nonzero(outgoing < cost) / denominator,
            np.count_nonzero(incoming < cost) / denominator,
            cost - float(np.min(outgoing)),
            cost - float(np.min(incoming)),
            float(edge.axis == "down"),
            *provenance_values,
            *arm_values,
            arm_support[index],
            arm_support[index] / float(len(FUSION_ARM_NAMES)),
            component_sizes[index],
            float(degrees[edge.source]),
            float(degrees[edge.target]),
            component_means[index],
            float(source_count),
            float(target_count),
            source_margin,
            target_margin,
            (source_count + target_count) / max(1.0, positive_total),
        )
    if not np.isfinite(output).all():
        raise RuntimeError("context features contain non-finite values")
    return ContextEdgeRows(edges=kept_edges, features=output, selected_arm=selected_arm)


@dataclass(frozen=True)
class ContextProtector:
    """Portable fixed StandardScaler + unweighted C=1 logistic head."""

    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficient: np.ndarray
    intercept: float

    def __post_init__(self) -> None:
        for name in ("scaler_mean", "scaler_scale", "coefficient"):
            value = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            if value.shape != (len(FEATURE_NAMES),) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite and match the feature roster")
            if name == "scaler_scale" and np.any(value <= 0.0):
                raise ValueError("scaler scale must be positive")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isfinite(self.intercept):
            raise ValueError("intercept must be finite")

    def predict_probability(self, features: Any) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError("feature matrix has the wrong shape")
        score = (
            (values - self.scaler_mean) / self.scaler_scale
        ) @ self.coefficient + self.intercept
        return np.where(
            score >= 0.0, 1.0 / (1.0 + np.exp(-score)), np.exp(score) / (1.0 + np.exp(score))
        )

    def keep_mask(self, features: Any) -> np.ndarray:
        return self.predict_probability(features) >= DECISION_THRESHOLD


def fit_context_protector(features: Any, labels: Any) -> ContextProtector:
    """Fit exactly the preregistered unweighted scaler + lbfgs logistic head."""

    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.uint8)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("feature matrix has the wrong shape")
    if (
        target.shape != (len(matrix),)
        or not np.isin(target, (0, 1)).all()
        or len(np.unique(target)) != 2
    ):
        raise ValueError("labels must align and contain both binary classes")
    scaler = StandardScaler().fit(matrix)
    estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
    estimator.fit(scaler.transform(matrix), target)
    if estimator.classes_.tolist() != [0, 1]:
        raise RuntimeError("unexpected class order")
    return ContextProtector(
        scaler.mean_, scaler.scale_, estimator.coef_[0], float(estimator.intercept_[0])
    )


__all__ = [
    "ContextEdgeRows",
    "ContextProtector",
    "DECISION_THRESHOLD",
    "FEATURE_NAMES",
    "fit_context_protector",
    "realised_context_features",
]
