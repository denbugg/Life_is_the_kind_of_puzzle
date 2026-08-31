"""Cross-query confidence calibration for frozen component relations.

The v1 component relation head ranks candidate attachments within each
component/direction query.  Its scores and margins were not trained to be
comparable *between* queries.  This module leaves every within-query candidate
ranking frozen and learns only whether the already-selected learned top-1 is
likely to be correct.

Feature extraction is deliberately target blind.  Exact synthetic labels are
accepted only by :func:`fit_confidence_calibrator` and the diagnostic
observation helper, never by inference feature construction.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from aiijc_puzzle import socket_decoder
from aiijc_puzzle.component_relation_reranker import (
    CONTACT_FEATURE_DIMENSION,
    DIRECTION_TO_INDEX,
    DIRECTIONS,
    ComponentRelationCandidate,
    ComponentTruthProfile,
    RelationCandidateLabel,
)
from aiijc_puzzle.component_shift_head import ComponentDescriptor
from aiijc_puzzle.socket_decoder import (
    SocketEdge,
    hard_partial_axis_matching,
    prioritise_component_edges,
)

GRID = 24
QUERY_SCORE_FEATURES = (
    "top_score",
    "top2_margin",
    "top_z_score",
    "normalised_entropy",
    "top_probability",
    "score_standard_deviation",
    "score_range",
)
COMPONENT_STRUCTURE_FEATURES = (
    "size_fraction",
    "log_size_fraction",
    "height_fraction",
    "width_fraction",
    "density",
    "confidence_tanh",
    "is_singleton",
    "boundary_member_fraction",
)
RELATION_GEOMETRY_FEATURES = (
    "direction_right",
    "direction_down",
    "direction_left",
    "direction_up",
    "target_row_offset",
    "target_column_offset",
    "centroid_row_delta",
    "centroid_column_delta",
    "combined_row_span",
    "combined_column_span",
    "contact_member_fraction",
    "log_contact_count",
    "log_source_target_size_ratio",
    "normalised_log_proposal_count",
)
FEATURE_NAMES = (
    *(f"learned_{name}" for name in QUERY_SCORE_FEATURES),
    *(f"raw_{name}" for name in QUERY_SCORE_FEATURES),
    "candidate_count_fraction",
    "log_candidate_count_fraction",
    "learned_raw_same_winner",
    "learned_winner_raw_reciprocal_rank",
    "raw_winner_learned_reciprocal_rank",
    "learned_winner_residual",
    "raw_winner_residual",
    *(f"selected_contact_mean_{index}" for index in range(CONTACT_FEATURE_DIMENSION)),
    *(f"selected_contact_max_{index}" for index in range(CONTACT_FEATURE_DIMENSION)),
    *(f"source_{name}" for name in COMPONENT_STRUCTURE_FEATURES),
    *(f"target_{name}" for name in COMPONENT_STRUCTURE_FEATURES),
    *RELATION_GEOMETRY_FEATURES,
)


@dataclass(frozen=True)
class QueryConfidenceFeatures:
    """One target-blind query row and its frozen learned/raw winners."""

    board_id: str
    source_component: int
    direction: str
    learned_top_candidate: int
    raw_top_candidate: int
    learned_margin: float
    raw_margin: float
    values: tuple[float, ...]

    @property
    def query_key(self) -> tuple[int, str]:
        return self.source_component, self.direction


@dataclass(frozen=True)
class LogisticConfidenceCalibrator:
    """Portable standardized logistic calibration parameters."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def __post_init__(self) -> None:
        dimension = len(self.feature_names)
        if dimension == 0 or not (
            len(self.mean) == len(self.scale) == len(self.coefficients) == dimension
        ):
            raise ValueError("calibrator arrays must share one non-zero dimension")
        if len(set(self.feature_names)) != dimension:
            raise ValueError("calibrator feature names must be unique")
        numeric = (*self.mean, *self.scale, *self.coefficients, self.intercept)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("calibrator parameters must be finite")
        if any(value <= 0 for value in self.scale):
            raise ValueError("calibrator scales must be positive")

    @property
    def parameter_count(self) -> int:
        return len(self.coefficients) + 1

    def predict_logits(self, values: Any) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        expected = len(self.feature_names)
        if matrix.ndim != 2 or matrix.shape[1] != expected:
            raise ValueError(f"features must have shape rows x {expected}")
        if not np.isfinite(matrix).all():
            raise ValueError("features must be finite")
        normalised = (matrix - np.asarray(self.mean)) / np.asarray(self.scale)
        return normalised @ np.asarray(self.coefficients) + self.intercept

    def predict_probabilities(self, values: Any) -> np.ndarray:
        logits = self.predict_logits(values)
        # Stable sigmoid without depending on the sklearn estimator at inference.
        positive = logits >= 0
        result = np.empty_like(logits)
        result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_value = np.exp(logits[~positive])
        result[~positive] = exp_value / (1.0 + exp_value)
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": "standardized-l2-logistic-query-confidence-v1.1",
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "parameters": self.parameter_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LogisticConfidenceCalibrator:
        return cls(
            feature_names=tuple(str(item) for item in value["feature_names"]),
            mean=tuple(float(item) for item in value["mean"]),
            scale=tuple(float(item) for item in value["scale"]),
            coefficients=tuple(float(item) for item in value["coefficients"]),
            intercept=float(value["intercept"]),
        )


def _component_structure(component: ComponentDescriptor, *, grid: int) -> tuple[float, ...]:
    area = component.height * component.width
    boundary = sum(
        row in {0, component.height - 1} or column in {0, component.width - 1}
        for row, column in zip(
            component.relative_rows,
            component.relative_columns,
            strict=True,
        )
    )
    return (
        component.size / (grid * grid),
        math.log1p(component.size) / math.log1p(grid * grid),
        component.height / grid,
        component.width / grid,
        component.size / area,
        math.tanh(component.confidence / 5.0),
        float(component.size == 1),
        boundary / component.size,
    )


def _score_statistics(scores: np.ndarray, order: Sequence[int]) -> tuple[float, ...]:
    values = np.asarray([scores[index] for index in order], dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("one finite non-empty score vector is required")
    top = float(values[0])
    margin = 0.0 if len(values) == 1 else top - float(values[1])
    standard_deviation = float(values.std())
    top_z = (top - float(values.mean())) / max(standard_deviation, 1e-8)
    shifted = values - values.max()
    probability = np.exp(shifted)
    probability /= probability.sum()
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-12))))
    normalised_entropy = entropy / math.log(len(values)) if len(values) > 1 else 0.0
    return (
        top,
        margin,
        top_z,
        normalised_entropy,
        float(probability[0]),
        standard_deviation,
        float(values.max() - values.min()),
    )


def _relation_geometry(
    source: ComponentDescriptor,
    target: ComponentDescriptor,
    candidate: ComponentRelationCandidate,
    *,
    grid: int,
) -> tuple[float, ...]:
    normaliser = float(max(grid - 1, 1))
    source_row = float(np.mean(source.relative_rows))
    source_column = float(np.mean(source.relative_columns))
    target_row = float(np.mean(target.relative_rows)) + candidate.target_row_offset
    target_column = (
        float(np.mean(target.relative_columns)) + candidate.target_column_offset
    )
    combined_rows = (
        *source.relative_rows,
        *(row + candidate.target_row_offset for row in target.relative_rows),
    )
    combined_columns = (
        *source.relative_columns,
        *(column + candidate.target_column_offset for column in target.relative_columns),
    )
    direction = [0.0] * len(DIRECTIONS)
    direction[DIRECTION_TO_INDEX[candidate.direction]] = 1.0
    return tuple(direction) + (
        candidate.target_row_offset / normaliser,
        candidate.target_column_offset / normaliser,
        (target_row - source_row) / normaliser,
        (target_column - source_column) / normaliser,
        (max(combined_rows) - min(combined_rows) + 1) / grid,
        (max(combined_columns) - min(combined_columns) + 1) / grid,
        len(candidate.contacts) / max(source.size + target.size, 1),
        math.log1p(len(candidate.contacts)) / math.log1p(grid),
        math.log((source.size + 1.0) / (target.size + 1.0)),
        math.log1p(candidate.proposal_count) / math.log1p(
            max(source.size, target.size) + 1
        ),
    )


def build_query_confidence_features(
    logits: torch.Tensor | np.ndarray,
    candidates: tuple[ComponentRelationCandidate, ...],
    components: tuple[ComponentDescriptor, ...],
    *,
    board_id: str,
    grid: int = GRID,
) -> tuple[QueryConfidenceFeatures, ...]:
    """Build target-free features for frozen learned top-1 correctness.

    Candidate identities are used only to produce deterministic tie breaking
    and to recover component/contact evidence.  Neither shuffled tile IDs nor
    component IDs are included in the numeric feature vector.
    """

    if not board_id:
        raise ValueError("board_id must be non-empty")
    value: Any = logits
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    learned = np.asarray(value, dtype=np.float64)
    if learned.shape != (len(candidates),) or not np.isfinite(learned).all():
        raise ValueError("logits and candidates must form one finite vector")
    if not candidates or not components or grid < 2:
        raise ValueError("candidates/components must be non-empty and grid >=2")
    raw = np.asarray([candidate.baseline_score for candidate in candidates], dtype=np.float64)
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        grouped[candidate.query_key].append(index)

    rows: list[QueryConfidenceFeatures] = []
    for source_component, direction in sorted(
        grouped,
        key=lambda item: (item[0], DIRECTION_TO_INDEX[item[1]]),
    ):
        indices = grouped[(source_component, direction)]
        learned_order = sorted(
            indices,
            key=lambda index: (-float(learned[index]), candidates[index].relation_key),
        )
        raw_order = sorted(
            indices,
            key=lambda index: (-float(raw[index]), candidates[index].relation_key),
        )
        learned_top = learned_order[0]
        raw_top = raw_order[0]
        selected = candidates[learned_top]
        source = components[selected.source_component]
        target = components[selected.target_component]
        contacts = np.asarray(
            [contact.features for contact in selected.contacts],
            dtype=np.float64,
        )
        if contacts.ndim != 2 or contacts.shape[1] != CONTACT_FEATURE_DIMENSION:
            raise RuntimeError("selected relation contact feature contract changed")
        learned_stats = _score_statistics(learned, learned_order)
        raw_stats = _score_statistics(raw, raw_order)
        raw_rank = {index: rank for rank, index in enumerate(raw_order, start=1)}
        learned_rank = {
            index: rank for rank, index in enumerate(learned_order, start=1)
        }
        numeric = (
            *learned_stats,
            *raw_stats,
            len(indices) / 64.0,
            math.log1p(len(indices)) / math.log1p(64),
            float(learned_top == raw_top),
            1.0 / raw_rank[learned_top],
            1.0 / learned_rank[raw_top],
            float(learned[learned_top] - raw[learned_top]),
            float(learned[raw_top] - raw[raw_top]),
            *contacts.mean(axis=0).tolist(),
            *contacts.max(axis=0).tolist(),
            *_component_structure(source, grid=grid),
            *_component_structure(target, grid=grid),
            *_relation_geometry(source, target, selected, grid=grid),
        )
        if len(numeric) != len(FEATURE_NAMES) or not all(map(math.isfinite, numeric)):
            raise RuntimeError("query confidence feature contract changed")
        rows.append(
            QueryConfidenceFeatures(
                board_id=board_id,
                source_component=source_component,
                direction=direction,
                learned_top_candidate=learned_top,
                raw_top_candidate=raw_top,
                learned_margin=learned_stats[1],
                raw_margin=raw_stats[1],
                values=tuple(float(item) for item in numeric),
            )
        )
    return tuple(rows)


def fit_confidence_calibrator(
    rows: Sequence[QueryConfidenceFeatures],
    labels: Sequence[bool | int],
    *,
    regularization_c: float = 1.0,
    maximum_iterations: int = 1000,
    random_seed: int = 20260910,
) -> LogisticConfidenceCalibrator:
    """Fit the tiny calibrator; labels enter only through this training API."""

    if len(rows) != len(labels) or not rows:
        raise ValueError("non-empty rows and labels must align")
    if regularization_c <= 0 or maximum_iterations <= 0:
        raise ValueError("regularization and iteration count must be positive")
    matrix = np.asarray([row.values for row in rows], dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if matrix.shape != (len(rows), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("calibration feature matrix is malformed")
    if not np.isin(target, (0, 1)).all() or len(np.unique(target)) != 2:
        raise ValueError("calibration labels must contain both binary classes")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalised = (matrix - mean) / scale
    estimator = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        max_iter=maximum_iterations,
        random_state=random_seed,
        solver="lbfgs",
    )
    estimator.fit(normalised, target)
    if estimator.n_iter_[0] >= maximum_iterations:
        raise RuntimeError("confidence calibration did not converge cleanly")
    return LogisticConfidenceCalibrator(
        feature_names=FEATURE_NAMES,
        mean=tuple(float(item) for item in mean),
        scale=tuple(float(item) for item in scale),
        coefficients=tuple(float(item) for item in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
    )


def confidence_query_observations(
    rows: Sequence[QueryConfidenceFeatures],
    calibrator: LogisticConfidenceCalibrator,
    candidates: tuple[ComponentRelationCandidate, ...],
    labels: tuple[RelationCandidateLabel, ...],
    oracle_relations: frozenset[tuple[int, str, int, int, int]],
    profiles: tuple[ComponentTruthProfile, ...],
) -> list[dict[str, Any]]:
    """Attach exact truth after inference features/probabilities are frozen."""

    if len(candidates) != len(labels):
        raise ValueError("candidate labels must align")
    if tuple(calibrator.feature_names) != FEATURE_NAMES:
        raise ValueError("calibrator feature contract differs from v1.1")
    matrix = np.asarray([row.values for row in rows], dtype=np.float64)
    probabilities = calibrator.predict_probabilities(matrix)
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        grouped[candidate.query_key].append(index)
    oracle_queries = {(relation[0], relation[1]) for relation in oracle_relations}
    observations: list[dict[str, Any]] = []
    for row, probability in zip(rows, probabilities, strict=True):
        indices = grouped[row.query_key]
        positives = {index for index in indices if labels[index].positive}
        profile = profiles[row.source_component]
        observations.append(
            {
                "board_id": row.board_id,
                "source_component": row.source_component,
                "direction": row.direction,
                "has_oracle_relation": row.query_key in oracle_queries,
                "has_candidates": True,
                "has_supplied_positive": bool(positives),
                "candidate_count": len(indices),
                "calibrated_confidence": float(probability),
                "calibrated_top1_correct": row.learned_top_candidate in positives,
                "learned_margin": row.learned_margin,
                "learned_top1_correct": row.learned_top_candidate in positives,
                "raw_margin": row.raw_margin,
                "raw_top1_correct": row.raw_top_candidate in positives,
                "source_purity": profile.purity,
                "source_size": profile.size,
            }
        )
    return observations


def _high_confidence(
    records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    caps: tuple[int, ...],
) -> dict[str, Any]:
    score_key = {
        "calibrated": "calibrated_confidence",
        "learned_margin": "learned_margin",
        "raw": "raw_margin",
    }[method]
    correct_key = "raw_top1_correct" if method == "raw" else "learned_top1_correct"
    by_board: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_board[str(record["board_id"])].append(record)
    result: dict[str, Any] = {}
    for cap in caps:
        correct: list[int] = []
        selected: list[int] = []
        for board in by_board.values():
            ordered = sorted(
                board,
                key=lambda record: (
                    -float(record[score_key]),
                    int(record["source_component"]),
                    DIRECTION_TO_INDEX[str(record["direction"])],
                ),
            )[:cap]
            correct.append(sum(bool(record[correct_key]) for record in ordered))
            selected.append(len(ordered))
        total = sum(selected)
        result[f"top{cap}"] = {
            "boards": len(by_board),
            "correct_per_board": float(np.mean(correct)) if correct else None,
            "selected_per_board": float(np.mean(selected)) if selected else None,
            "precision": sum(correct) / total if total else None,
        }
    return result


def aggregate_confidence_observations(
    records: Sequence[Mapping[str, Any]],
    *,
    caps: tuple[int, ...] = (32, 144),
) -> dict[str, Any]:
    """Compare cross-query selection only; candidate rankings stay frozen."""

    if not records or not caps or any(cap <= 0 for cap in caps):
        raise ValueError("records and positive caps are required")
    return {
        "board_count": len({str(record["board_id"]) for record in records}),
        "query_count": len(records),
        "calibrated": {
            "high_confidence": _high_confidence(
                records, method="calibrated", caps=caps
            )
        },
        "learned_margin_diagnostic": {
            "high_confidence": _high_confidence(
                records, method="learned_margin", caps=caps
            )
        },
        "raw_socket_component_baseline": {
            "high_confidence": _high_confidence(records, method="raw", caps=caps)
        },
    }


def calibrated_component_edge_priorities(
    right_log_assignment: Any,
    down_log_assignment: Any,
    rows: Sequence[QueryConfidenceFeatures],
    probabilities: Any,
    candidates: tuple[ComponentRelationCandidate, ...],
    *,
    grid: int = GRID,
    top_cap: int = 32,
    bonus_scale: float = 0.25,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Conservatively boost only hard edges supported by calibrated relations.

    Every hard projected edge starts at its unchanged Socket confidence.  For
    the top calibrated queries, an induced contact receives at most one bonus
    (the maximum supporting probability) and only when that exact directed
    contact is present in the corresponding hard matching.  Thus this helper
    can reprioritise decoder component constraints but cannot invent an edge or
    alter the partial matching.
    """

    if grid < 2 or top_cap <= 0 or not math.isfinite(bonus_scale) or bonus_scale < 0:
        raise ValueError("grid/top_cap/bonus_scale contract is invalid")
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != (len(rows),) or not np.isfinite(probability).all():
        raise ValueError("probabilities must align with confidence rows")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    right = hard_partial_axis_matching(right_log_assignment, grid=grid, axis="right")
    down = hard_partial_axis_matching(down_log_assignment, grid=grid, axis="down")
    count = grid * grid
    priorities = {
        "right": np.zeros((count, count), dtype=np.float64),
        "down": np.zeros((count, count), dtype=np.float64),
    }
    hard_edges: set[tuple[str, int, int]] = set()
    confidence: list[float] = []
    for matching in (right, down):
        for edge in matching.edges:
            priorities[edge.axis][edge.source, edge.target] = edge.confidence
            hard_edges.add((edge.axis, edge.source, edge.target))
            confidence.append(edge.confidence)
    standard_deviation = float(np.std(confidence))
    selected = sorted(
        range(len(rows)),
        key=lambda index: (
            -float(probability[index]),
            rows[index].source_component,
            DIRECTION_TO_INDEX[rows[index].direction],
        ),
    )[:top_cap]
    support: dict[tuple[str, int, int], float] = {}
    for row_index in selected:
        row = rows[row_index]
        candidate = candidates[row.learned_top_candidate]
        if candidate.query_key != row.query_key:
            raise ValueError("confidence row winner does not belong to its query")
        forward = candidate.direction in {"right", "down"}
        axis = "right" if candidate.direction in {"right", "left"} else "down"
        for contact in candidate.contacts:
            source, target = (
                (contact.source_tile, contact.target_tile)
                if forward
                else (contact.target_tile, contact.source_tile)
            )
            key = (axis, source, target)
            if key in hard_edges:
                support[key] = max(support.get(key, 0.0), float(probability[row_index]))
    bonus = bonus_scale * standard_deviation
    for (axis, source, target), value in support.items():
        priorities[axis][source, target] += bonus * value
    return priorities, {
        "top_query_cap": top_cap,
        "selected_queries": len(selected),
        "hard_edge_count": len(hard_edges),
        "boosted_hard_edges": len(support),
        "hard_edge_priority_standard_deviation": standard_deviation,
        "bonus_scale": bonus_scale,
        "maximum_absolute_bonus": bonus * max(support.values(), default=0.0),
    }


def relation_forest_score_substitution(
    right_log_assignment: Any,
    down_log_assignment: Any,
    rows: Sequence[QueryConfidenceFeatures],
    probabilities: Any,
    candidates: tuple[ComponentRelationCandidate, ...],
    *,
    grid: int = GRID,
    top_cap: int = 32,
    component_edge_budget_per_axis: int = 144,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Promote real learned contacts through a checked relation forest.

    Unlike :func:`calibrated_component_edge_priorities`, this development
    primitive can introduce a contact that was absent from the frozen hard
    matching.  Candidate relations are consumed atomically in calibrated order.
    New contacts obey per-axis outgoing/incoming capacity and are trial-added to
    the baseline component graph, whose exact coordinate-cycle, collision and
    span rules must accept the full relation.

    Accepted contacts replace their matrix rank with the better of the current
    row-best and column-best finite real score (plus only ``nextafter`` tie
    breaking).  This preserves the frozen score scale; dustbins and every other
    score remain bitwise unchanged.  The normal partial matching and decoder
    are run later on these substituted matrices.
    """

    if grid < 2 or top_cap <= 0:
        raise ValueError("grid must be >=2 and top_cap must be positive")
    count = grid * grid
    maximum_edges = count - grid
    if not 1 <= component_edge_budget_per_axis <= maximum_edges:
        raise ValueError("component edge budget is outside the decoder range")
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != (len(rows),) or not np.isfinite(probability).all():
        raise ValueError("probabilities must align with confidence rows")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must lie in [0,1]")

    def matrix(value: Any, *, name: str) -> np.ndarray:
        item = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        result = np.asarray(item, dtype=np.float64)
        if result.ndim == 3 and result.shape[0] == 1:
            result = result[0]
        if result.shape != (count + 1, count + 1):
            raise ValueError(f"{name} assignment has an invalid shape")
        usable = result.copy()
        usable[count, count] = 0.0
        if not np.isfinite(usable).all():
            raise ValueError(f"{name} assignment contains invalid usable values")
        return result.copy()

    substituted = {
        "right": matrix(right_log_assignment, name="right"),
        "down": matrix(down_log_assignment, name="down"),
    }
    right = hard_partial_axis_matching(substituted["right"], grid=grid, axis="right")
    down = hard_partial_axis_matching(substituted["down"], grid=grid, axis="down")
    seed_edges = prioritise_component_edges(
        right,
        down,
        edge_budget_per_axis=component_edge_budget_per_axis,
        tile_count=count,
    )
    builder = socket_decoder._TranslationComponents(count=count, grid=grid)
    seed_status = defaultdict(int)
    for edge in seed_edges:
        seed_status[builder.add(edge)] += 1
    hard_edge_keys = {
        (edge.axis, edge.source, edge.target)
        for matching in (right, down)
        for edge in matching.edges
    }
    selected = sorted(
        range(len(rows)),
        key=lambda index: (
            -float(probability[index]),
            rows[index].source_component,
            DIRECTION_TO_INDEX[rows[index].direction],
        ),
    )[:top_cap]
    used_outgoing = {"right": set(), "down": set()}
    used_incoming = {"right": set(), "down": set()}
    accepted_edges: list[SocketEdge] = []
    accepted_relations = 0
    rejected_capacity = 0
    rejected_geometry = defaultdict(int)
    for row_index in selected:
        row = rows[row_index]
        candidate = candidates[row.learned_top_candidate]
        if candidate.query_key != row.query_key:
            raise ValueError("confidence row winner does not belong to its query")
        axis = "right" if candidate.direction in {"right", "left"} else "down"
        forward = candidate.direction in {"right", "down"}
        proposed: list[SocketEdge] = []
        for contact in candidate.contacts:
            source, target = (
                (contact.source_tile, contact.target_tile)
                if forward
                else (contact.target_tile, contact.source_tile)
            )
            proposed.append(
                SocketEdge(
                    source=source,
                    target=target,
                    delta_row=int(axis == "down"),
                    delta_column=int(axis == "right"),
                    confidence=float(probability[row_index]),
                    axis=axis,
                )
            )
        if len({(edge.axis, edge.source, edge.target) for edge in proposed}) != len(
            proposed
        ):
            raise RuntimeError("one relation produced duplicate canonical contacts")
        if any(
            edge.source in used_outgoing[edge.axis]
            or edge.target in used_incoming[edge.axis]
            for edge in proposed
        ):
            rejected_capacity += 1
            continue
        trial = copy.deepcopy(builder)
        statuses = [trial.add(edge) for edge in proposed]
        invalid = next(
            (status for status in statuses if status not in {"added", "consistent"}),
            None,
        )
        if invalid is not None:
            rejected_geometry[invalid] += 1
            continue
        builder = trial
        accepted_relations += 1
        accepted_edges.extend(proposed)
        for edge in proposed:
            used_outgoing[edge.axis].add(edge.source)
            used_incoming[edge.axis].add(edge.target)

    changed_edges = 0
    for edge in accepted_edges:
        value = substituted[edge.axis]
        real = value[:count, :count]
        row = real[edge.source].copy()
        column = real[:, edge.target].copy()
        row[edge.source] = -np.inf
        column[edge.target] = -np.inf
        target = max(
            float(real[edge.source, edge.target]),
            float(np.max(row[np.isfinite(row)])),
            float(np.max(column[np.isfinite(column)])),
        )
        promoted = float(np.nextafter(target, math.inf))
        if promoted > real[edge.source, edge.target]:
            changed_edges += 1
            real[edge.source, edge.target] = promoted

    projected = {
        "right": hard_partial_axis_matching(
            substituted["right"], grid=grid, axis="right"
        ),
        "down": hard_partial_axis_matching(substituted["down"], grid=grid, axis="down"),
    }
    projected_keys = {
        (edge.axis, edge.source, edge.target)
        for matching in projected.values()
        for edge in matching.edges
    }
    accepted_keys = {
        (edge.axis, edge.source, edge.target) for edge in accepted_edges
    }
    forest_components = builder.complete_components()
    return substituted, {
        "top_query_cap": top_cap,
        "selected_queries": len(selected),
        "accepted_relations": accepted_relations,
        "accepted_contacts": len(accepted_edges),
        "new_contacts_absent_from_original_hard_matching": len(
            accepted_keys - hard_edge_keys
        ),
        "changed_matrix_contacts": changed_edges,
        "accepted_contacts_surviving_new_hard_matching": len(
            accepted_keys & projected_keys
        ),
        "rejected_relations_out_in_capacity": rejected_capacity,
        "rejected_relations_geometry": dict(sorted(rejected_geometry.items())),
        "seed_status_counts": dict(sorted(seed_status.items())),
        "checked_forest_component_count": len(forest_components),
        "checked_forest_largest_component": max(map(len, forest_components)),
        "score_substitution": (
            "max(current,row-best,column-best) then nextafter(+inf); dustbins and "
            "all other cells unchanged"
        ),
    }


__all__ = [
    "FEATURE_NAMES",
    "LogisticConfidenceCalibrator",
    "QueryConfidenceFeatures",
    "aggregate_confidence_observations",
    "build_query_confidence_features",
    "calibrated_component_edge_priorities",
    "confidence_query_observations",
    "fit_confidence_calibrator",
    "relation_forest_score_substitution",
]
