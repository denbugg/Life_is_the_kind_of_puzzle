"""Fixed target-free selectors over an existing strict six-arm layout roster.

The module is deliberately a consumer, not a decoder.  It projects the full
joint-verifier evidence onto each of the six already frozen layouts and may
return only one of those layouts.  No edge, tile, orientation, or pixel is
created or changed.

Two fixed rules complement the historical fixed-head rule:

* union/dense dominance keeps only arms whose candidate-union coverage and
  mean dense two-sided confidence do not regress on either axis;
* source-normalized dominance scores every realised edge after normalising
  logits and confidence inside its source row.  An edge absent from the union
  receives a deterministic source-local floor.

Both rules retain the exact incumbent unless there is a strict model-space
improvement.  Their arithmetic and tie breaks depend on relations and frozen
arm order, never on the numeric spelling of a tile identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aiijc_puzzle.joint_relation_selector_consumer import (
    ArmJointEvidence,
    FrozenSixArmRoster,
    JointRelationEvidence,
    JointRelationSelection,
    map_joint_evidence_to_six_arms,
    select_fixed_head_dominant_arm,
)
from aiijc_puzzle.taska_relation_truth_selector import realised_edges
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES, strict_layout

AXIS_NAMES = ("right", "down")
PORTFOLIO_MEMBER_NAMES = (
    "incumbent_keep",
    "fixed_head_comparator",
    "union_dense_dominance",
    "source_normalized_dominance",
)
UNION_DENSE_RULE = "axiswise-union-coverage-dense-confidence-dominance-v1"
SOURCE_NORMALIZED_RULE = "axiswise-source-normalized-full-evidence-floor-v1"
MISSING_EDGE_OFFSET = 1.0
NORMALIZATION_TANH_SCALE = 2.0
NORMALIZATION_EPSILON = 1e-12


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _bounded_source_normalize(values: np.ndarray) -> np.ndarray:
    """Return a bounded, slot-order-independent source-row normalization."""

    current = np.asarray(values, dtype=np.float64)
    if current.ndim != 1 or len(current) == 0 or not np.isfinite(current).all():
        raise ValueError("source-row values must be a non-empty finite vector")
    scale = float(current.std())
    if scale <= NORMALIZATION_EPSILON:
        return np.zeros_like(current)
    zscore = (current - float(current.mean())) / scale
    return np.tanh(zscore / NORMALIZATION_TANH_SCALE)


@dataclass(frozen=True)
class PortfolioArmEvidence:
    """Inference-only full-joint diagnostics for every arm and axis."""

    union_coverage_counts: np.ndarray
    union_dense_confidence_sums: np.ndarray
    union_dense_confidence_means: np.ndarray
    normalized_logit_sums: np.ndarray
    normalized_confidence_sums: np.ndarray
    normalized_combined_sums: np.ndarray
    missing_edge_counts: np.ndarray
    legacy_head: ArmJointEvidence


def map_full_joint_evidence_to_six_arms(
    evidence: JointRelationEvidence,
    roster: FrozenSixArmRoster,
) -> PortfolioArmEvidence:
    """Project full candidate evidence onto all six unchanged layouts."""

    if evidence.grid_size != roster.grid_size:
        raise ValueError("joint evidence and six-arm roster use different grids")
    count = evidence.grid_size**2
    slot_lookups: list[list[dict[int, int]]] = []
    normalized_logit_lookups: list[list[dict[int, float]]] = []
    normalized_confidence_lookups: list[list[dict[int, float]]] = []
    logit_floors = np.empty((2, count), dtype=np.float64)
    confidence_floors = np.empty((2, count), dtype=np.float64)
    for axis in range(2):
        axis_slots: list[dict[int, int]] = []
        axis_logits: list[dict[int, float]] = []
        axis_confidences: list[dict[int, float]] = []
        for source in range(count):
            slots = np.flatnonzero(evidence.valid[axis, source])
            if len(slots) == 0:
                raise ValueError("joint candidate row is empty")
            targets = evidence.candidates[axis, source, slots]
            normalized_logits = _bounded_source_normalize(
                evidence.learned_logits[axis][source, slots]
            )
            normalized_confidences = _bounded_source_normalize(
                evidence.learned_joint_confidence[axis][source, slots]
            )
            axis_slots.append(
                {int(target): int(slot) for target, slot in zip(targets, slots, strict=True)}
            )
            axis_logits.append(
                {
                    int(target): float(value)
                    for target, value in zip(targets, normalized_logits, strict=True)
                }
            )
            axis_confidences.append(
                {
                    int(target): float(value)
                    for target, value in zip(
                        targets, normalized_confidences, strict=True
                    )
                }
            )
            logit_floors[axis, source] = (
                float(normalized_logits.min()) - MISSING_EDGE_OFFSET
            )
            confidence_floors[axis, source] = (
                float(normalized_confidences.min()) - MISSING_EDGE_OFFSET
            )
        slot_lookups.append(axis_slots)
        normalized_logit_lookups.append(axis_logits)
        normalized_confidence_lookups.append(axis_confidences)

    shape = (len(FUSION_ARM_NAMES), 2)
    coverage = np.zeros(shape, dtype=np.int32)
    dense_confidence_sums = np.zeros(shape, dtype=np.float64)
    normalized_logit_sums = np.zeros(shape, dtype=np.float64)
    normalized_confidence_sums = np.zeros(shape, dtype=np.float64)
    missing = np.zeros(shape, dtype=np.int32)
    for arm_index, layout in enumerate(roster.layouts):
        for edge in realised_edges(layout, grid=roster.grid_size):
            axis = AXIS_NAMES.index(edge.axis)
            slot = slot_lookups[axis][edge.source].get(edge.target)
            if slot is None:
                missing[arm_index, axis] += 1
                normalized_logit_sums[arm_index, axis] += logit_floors[
                    axis, edge.source
                ]
                normalized_confidence_sums[arm_index, axis] += confidence_floors[
                    axis, edge.source
                ]
                continue
            coverage[arm_index, axis] += 1
            dense_confidence_sums[arm_index, axis] += float(
                evidence.learned_joint_confidence[axis][edge.source, slot]
            )
            normalized_logit_sums[arm_index, axis] += normalized_logit_lookups[
                axis
            ][edge.source][edge.target]
            normalized_confidence_sums[arm_index, axis] += (
                normalized_confidence_lookups[axis][edge.source][edge.target]
            )

    confidence_means = np.full(shape, -np.inf, dtype=np.float64)
    np.divide(
        dense_confidence_sums,
        coverage,
        out=confidence_means,
        where=coverage > 0,
    )
    normalized_combined = 0.5 * (
        normalized_logit_sums + normalized_confidence_sums
    )
    return PortfolioArmEvidence(
        union_coverage_counts=_readonly(coverage),
        union_dense_confidence_sums=_readonly(dense_confidence_sums),
        union_dense_confidence_means=_readonly(confidence_means),
        normalized_logit_sums=_readonly(normalized_logit_sums),
        normalized_confidence_sums=_readonly(normalized_confidence_sums),
        normalized_combined_sums=_readonly(normalized_combined),
        missing_edge_counts=_readonly(missing),
        legacy_head=map_joint_evidence_to_six_arms(evidence, roster),
    )


@dataclass(frozen=True)
class PortfolioRuleSelection:
    """One exact frozen arm and the diagnostics used to select it."""

    layout: np.ndarray
    selected_index: int
    incumbent_index: int
    arm_evidence: PortfolioArmEvidence
    selection_rule: str

    @property
    def selected_arm(self) -> str:
        return FUSION_ARM_NAMES[self.selected_index]

    @property
    def incumbent_arm(self) -> str:
        return FUSION_ARM_NAMES[self.incumbent_index]

    @property
    def changed(self) -> bool:
        return self.selected_index != self.incumbent_index


def _selection(
    roster: FrozenSixArmRoster,
    evidence: PortfolioArmEvidence,
    selected: int,
    rule: str,
) -> PortfolioRuleSelection:
    layout = strict_layout(roster.layouts[selected], grid=roster.grid_size).copy()
    layout.setflags(write=False)
    return PortfolioRuleSelection(
        layout=layout,
        selected_index=selected,
        incumbent_index=roster.incumbent_index,
        arm_evidence=evidence,
        selection_rule=rule,
    )


def select_union_dense_dominant_arm(
    evidence: JointRelationEvidence,
    roster: FrozenSixArmRoster,
) -> PortfolioRuleSelection:
    """Require axiswise union-coverage and dense-confidence nonregression."""

    mapped = map_full_joint_evidence_to_six_arms(evidence, roster)
    incumbent = roster.incumbent_index
    base_coverage = mapped.union_coverage_counts[incumbent]
    base_confidence = mapped.union_dense_confidence_means[incumbent]
    if not np.isfinite(base_confidence).all():
        # A mean over an empty incumbent projection is undefined.  Do not let
        # +/-inf arithmetic turn that absence of evidence into a switch.
        return _selection(roster, mapped, incumbent, UNION_DENSE_RULE)
    eligible = []
    for index in range(len(FUSION_ARM_NAMES)):
        coverage = mapped.union_coverage_counts[index]
        confidence = mapped.union_dense_confidence_means[index]
        nonregressing = (
            np.isfinite(confidence).all()
            and np.all(coverage >= base_coverage)
            and np.all(confidence >= base_confidence)
        )
        strict = np.any(coverage > base_coverage) or np.any(
            confidence > base_confidence
        )
        if nonregressing and strict:
            eligible.append(index)

    if eligible:

        def key(index: int) -> tuple[float, ...]:
            coverage_delta = mapped.union_coverage_counts[index] - base_coverage
            confidence_delta = (
                mapped.union_dense_confidence_means[index] - base_confidence
            )
            return (
                float(coverage_delta.sum()),
                float(coverage_delta.min()),
                float(confidence_delta.sum()),
                float(confidence_delta.min()),
                float(roster.expected_correct_scores[index]),
                float(-index),
            )

        selected = max(eligible, key=key)
    else:
        selected = incumbent
    return _selection(roster, mapped, selected, UNION_DENSE_RULE)


def select_source_normalized_dominant_arm(
    evidence: JointRelationEvidence,
    roster: FrozenSixArmRoster,
) -> PortfolioRuleSelection:
    """Require axiswise nonregression in full normalized edge evidence."""

    mapped = map_full_joint_evidence_to_six_arms(evidence, roster)
    incumbent = roster.incumbent_index
    base_score = mapped.normalized_combined_sums[incumbent]
    eligible = [
        index
        for index in range(len(FUSION_ARM_NAMES))
        if np.all(mapped.normalized_combined_sums[index] >= base_score)
        and np.any(mapped.normalized_combined_sums[index] > base_score)
    ]
    if eligible:

        def key(index: int) -> tuple[float, ...]:
            delta = mapped.normalized_combined_sums[index] - base_score
            return (
                float(delta.sum()),
                float(delta.min()),
                float(-mapped.missing_edge_counts[index].sum()),
                float(roster.expected_correct_scores[index]),
                float(-index),
            )

        selected = max(eligible, key=key)
    else:
        selected = incumbent
    return _selection(roster, mapped, selected, SOURCE_NORMALIZED_RULE)


@dataclass(frozen=True)
class FrozenSelectorPortfolio:
    """Transparent KEEP, legacy comparator, and two fixed new candidates."""

    incumbent_layout: np.ndarray
    incumbent_index: int
    fixed_head: JointRelationSelection
    union_dense: PortfolioRuleSelection
    source_normalized: PortfolioRuleSelection

    @property
    def layouts(self) -> tuple[np.ndarray, ...]:
        return (
            self.incumbent_layout,
            self.fixed_head.layout,
            self.union_dense.layout,
            self.source_normalized.layout,
        )

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return (
            self.incumbent_index,
            self.fixed_head.selected_index,
            self.union_dense.selected_index,
            self.source_normalized.selected_index,
        )


def build_frozen_selector_portfolio(
    evidence: JointRelationEvidence,
    roster: FrozenSixArmRoster,
) -> FrozenSelectorPortfolio:
    """Build all preregistered members without labels or candidate synthesis."""

    incumbent = strict_layout(
        roster.layouts[roster.incumbent_index], grid=roster.grid_size
    ).copy()
    incumbent.setflags(write=False)
    result = FrozenSelectorPortfolio(
        incumbent_layout=incumbent,
        incumbent_index=roster.incumbent_index,
        fixed_head=select_fixed_head_dominant_arm(evidence, roster),
        union_dense=select_union_dense_dominant_arm(evidence, roster),
        source_normalized=select_source_normalized_dominant_arm(evidence, roster),
    )
    for layout, index in zip(result.layouts, result.selected_indices, strict=True):
        if not np.array_equal(layout, roster.layouts[index]):
            raise RuntimeError("portfolio member is not an exact frozen six-arm layout")
    return result
