"""Target-free bridge from joint edge evidence to the frozen six-arm roster.

The joint verifier and the TASKA relation selector use the same shuffled tile
bag identity space, but they solve different problems.  This module never
turns joint logits into a new graph layout: the already confirmed six-arm
decoder remains the only layout producer.  Joint evidence may select one
whole, frozen arm only when its fixed five-percent head weakly dominates the
incumbent on both axes and strictly improves the total supported-head count.

This deliberately avoids the previously rejected all-edge HGB-ranked union.
Every returned layout is therefore one of six audited permutations of all
original upright tile identities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.taska_relation_truth_selector import FEATURE_NAMES, realised_edges
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES, strict_layout
from aiijc_puzzle.tri_emitter_edge_verifier import candidate_pool_digest

AXIS_NAMES = ("right", "down")
FIXED_HEAD_FRACTION = 0.05
SELECTION_RULE = "axiswise-fixed-head-dominance-then-frozen-lexicographic-v1"

FORBIDDEN_TARGET_ARRAY_TOKENS = (
    "target_slots",
    "truth",
    "reference",
    "labels",
    "clean_tiles",
    "output_pixels",
)

JOINT_REQUIRED_CASE_KEYS = frozenset(
    {
        "union_candidates",
        "union_valid",
        "emitter_topk",
        "union_identity_digest_ascii",
        "learned_logits__right",
        "learned_joint_confidence__right",
        "learned_head_sources__right",
        "learned_head_targets__right",
        "learned_head_confidences__right",
        "learned_head_requested__right",
        "learned_reciprocal_count__right",
        "learned_logits__down",
        "learned_joint_confidence__down",
        "learned_head_sources__down",
        "learned_head_targets__down",
        "learned_head_confidences__down",
        "learned_head_requested__down",
        "learned_reciprocal_count__down",
    }
)
JOINT_OPTIONAL_KNOWN_CASE_KEYS = frozenset(
    {
        "raw_top32",
        "learned_top32",
        "raw_head_sources__right",
        "raw_head_targets__right",
        "raw_head_confidences__right",
        "raw_head_requested__right",
        "raw_reciprocal_count__right",
        "raw_head_sources__down",
        "raw_head_targets__down",
        "raw_head_confidences__down",
        "raw_head_requested__down",
        "raw_reciprocal_count__down",
    }
)
RELATION_ROSTER_CASE_KEYS = frozenset(
    {
        "relation_features",
        "relation_expected_correct_scores",
        "relation_truth_selector_layout",
        *(f"relation_arm_{arm}_layout" for arm in FUSION_ARM_NAMES),
    }
)


def reject_target_bearing_array_names(names: Sequence[str]) -> None:
    """Reject label/reference arrays while allowing head target identities."""

    offenders = []
    for name in names:
        # ``relation_truth_selector`` is the historical, frozen model name; it
        # is not a label array.  Remove only that exact provenance phrase before
        # checking the remaining key text.
        checked = name.lower().replace("relation_truth_selector", "relation_selector")
        if any(token in checked for token in FORBIDDEN_TARGET_ARRAY_TOKENS):
            offenders.append(name)
    offenders.sort()
    if offenders:
        raise RuntimeError(f"target-free archive contains forbidden arrays: {offenders}")


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _decode_digest(value: Any) -> str:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 1:
        raise ValueError("union identity digest must be a uint8 ASCII vector")
    try:
        result = bytes(array).decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("union identity digest is not ASCII") from error
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError("union identity digest is not a lowercase SHA-256")
    return result


@dataclass(frozen=True)
class JointRelationEvidence:
    """Validated joint logits and fixed heads in immutable tile-bag identity."""

    candidates: np.ndarray
    valid: np.ndarray
    emitter_topk: np.ndarray
    learned_logits: tuple[np.ndarray, np.ndarray]
    learned_joint_confidence: tuple[np.ndarray, np.ndarray]
    head_sources: tuple[np.ndarray, np.ndarray]
    head_targets: tuple[np.ndarray, np.ndarray]
    head_confidences: tuple[np.ndarray, np.ndarray]
    union_identity_digest: str
    grid_size: int = 24

    @classmethod
    def from_case_arrays(
        cls,
        arrays: Mapping[str, Any],
        *,
        grid_size: int = 24,
    ) -> JointRelationEvidence:
        reject_target_bearing_array_names(tuple(arrays))
        keys = set(arrays)
        missing = JOINT_REQUIRED_CASE_KEYS - keys
        unexpected = keys - JOINT_REQUIRED_CASE_KEYS - JOINT_OPTIONAL_KNOWN_CASE_KEYS
        if missing or unexpected:
            raise ValueError(
                "joint target-free case schema changed; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        count = grid_size * grid_size
        candidates = np.asarray(arrays["union_candidates"])
        valid = np.asarray(arrays["union_valid"])
        emitter_topk = np.asarray(arrays["emitter_topk"])
        if candidates.ndim != 3 or candidates.shape[:2] != (2, count):
            raise ValueError("union candidates must have shape 2 x tile_count x K")
        if candidates.dtype not in (np.int32, np.int64):
            raise ValueError("union candidate identities must be integer")
        if valid.shape != candidates.shape or valid.dtype != np.bool_:
            raise ValueError("union valid mask must be aligned boolean")
        if emitter_topk.ndim != 4 or emitter_topk.shape[1:3] != (2, count):
            raise ValueError("emitter_topk must have shape E x 2 x tile_count x K")
        if emitter_topk.dtype not in (np.int32, np.int64):
            raise ValueError("emitter_topk identities must be integer")
        if np.any(valid & ((candidates < 0) | (candidates >= count))):
            raise ValueError("valid union candidate identity is out of range")
        for axis in range(2):
            for source in range(count):
                row = candidates[axis, source, valid[axis, source]]
                if len(row) == 0 or len(row) != len(np.unique(row)):
                    raise ValueError("union rows must be non-empty and identity-unique")
                if source in row:
                    raise ValueError("union rows must not contain self edges")
                available = set(int(target) for target in row)
                emitter_targets = set(
                    int(target)
                    for target in emitter_topk[:, axis, source].reshape(-1)
                )
                if not emitter_targets.issubset(available):
                    raise ValueError("emitter_topk identity is absent from the frozen union")

        digest = _decode_digest(arrays["union_identity_digest_ascii"])
        observed_digest = candidate_pool_digest(candidates, valid, emitter_topk)
        if digest != observed_digest:
            raise ValueError("union candidate identity digest mismatch")

        logits: list[np.ndarray] = []
        confidences: list[np.ndarray] = []
        head_sources: list[np.ndarray] = []
        head_targets: list[np.ndarray] = []
        head_confidences: list[np.ndarray] = []
        requested_count = max(1, math.ceil(FIXED_HEAD_FRACTION * count))
        for axis_index, axis_name in enumerate(AXIS_NAMES):
            current_logits = np.asarray(arrays[f"learned_logits__{axis_name}"])
            current_confidence = np.asarray(
                arrays[f"learned_joint_confidence__{axis_name}"]
            )
            if current_logits.shape != candidates.shape[1:] or not np.isfinite(
                current_logits
            ).all():
                raise ValueError(f"{axis_name} learned logits changed shape or are non-finite")
            if current_confidence.shape != candidates.shape[1:]:
                raise ValueError(f"{axis_name} joint confidence shape changed")
            axis_valid = valid[axis_index]
            if not np.isfinite(current_confidence[axis_valid]).all() or not np.isneginf(
                current_confidence[~axis_valid]
            ).all():
                raise ValueError(
                    f"{axis_name} joint confidence must be finite only on valid union slots"
                )

            sources = np.asarray(
                arrays[f"learned_head_sources__{axis_name}"], dtype=np.int64
            )
            targets = np.asarray(
                arrays[f"learned_head_targets__{axis_name}"], dtype=np.int64
            )
            selected_confidence = np.asarray(
                arrays[f"learned_head_confidences__{axis_name}"], dtype=np.float64
            )
            requested = np.asarray(
                arrays[f"learned_head_requested__{axis_name}"]
            )
            reciprocal = np.asarray(
                arrays[f"learned_reciprocal_count__{axis_name}"]
            )
            if requested.shape != () or int(requested.item()) != requested_count:
                raise ValueError(f"{axis_name} fixed-head requested count changed")
            if reciprocal.shape != () or int(reciprocal.item()) < requested_count:
                raise ValueError(f"{axis_name} reciprocal head cannot fill fixed coverage")
            expected_shape = (requested_count,)
            if (
                sources.shape != expected_shape
                or targets.shape != expected_shape
                or selected_confidence.shape != expected_shape
                or not np.isfinite(selected_confidence).all()
            ):
                raise ValueError(f"{axis_name} fixed-head arrays changed shape or finiteness")
            if (
                len(np.unique(sources)) != requested_count
                or len(np.unique(targets)) != requested_count
                or np.any((sources < 0) | (sources >= count))
                or np.any((targets < 0) | (targets >= count))
            ):
                raise ValueError(f"{axis_name} fixed head is not reciprocal-unique")
            gathered: list[float] = []
            for source, target in zip(sources, targets, strict=True):
                slots = np.flatnonzero(
                    axis_valid[source] & (candidates[axis_index, source] == target)
                )
                if len(slots) != 1:
                    raise ValueError(f"{axis_name} head edge is absent from immutable union")
                gathered.append(float(current_confidence[source, slots[0]]))
            if not np.allclose(
                np.asarray(gathered), selected_confidence, rtol=1e-6, atol=1e-6
            ):
                raise ValueError(f"{axis_name} head confidence is misaligned with union slots")

            logits.append(_readonly(current_logits.astype(np.float32, copy=False)))
            confidences.append(
                _readonly(current_confidence.astype(np.float32, copy=False))
            )
            head_sources.append(_readonly(sources.astype(np.int32, copy=False)))
            head_targets.append(_readonly(targets.astype(np.int32, copy=False)))
            head_confidences.append(
                _readonly(selected_confidence.astype(np.float32, copy=False))
            )
        return cls(
            candidates=_readonly(candidates.astype(np.int32, copy=False)),
            valid=_readonly(valid),
            emitter_topk=_readonly(emitter_topk.astype(np.int32, copy=False)),
            learned_logits=(logits[0], logits[1]),
            learned_joint_confidence=(confidences[0], confidences[1]),
            head_sources=(head_sources[0], head_sources[1]),
            head_targets=(head_targets[0], head_targets[1]),
            head_confidences=(head_confidences[0], head_confidences[1]),
            union_identity_digest=digest,
            grid_size=grid_size,
        )


@dataclass(frozen=True)
class FrozenSixArmRoster:
    """The existing relation selector's six strict whole-layout candidates."""

    layouts: tuple[np.ndarray, ...]
    incumbent_index: int
    expected_correct_scores: np.ndarray
    relation_features: np.ndarray
    grid_size: int = 24

    @classmethod
    def from_case_arrays(
        cls,
        arrays: Mapping[str, Any],
        metadata_row: Mapping[str, Any],
        *,
        grid_size: int = 24,
    ) -> FrozenSixArmRoster:
        reject_target_bearing_array_names(tuple(arrays))
        if set(arrays) != RELATION_ROSTER_CASE_KEYS:
            raise ValueError("normalized relation-roster target-free case schema changed")
        arm_names = tuple(metadata_row.get("arm_names", FUSION_ARM_NAMES))
        if arm_names != tuple(FUSION_ARM_NAMES):
            raise ValueError("frozen six-arm roster or order changed")
        choice = metadata_row.get("choice")
        if choice not in FUSION_ARM_NAMES:
            raise ValueError("relation-selector incumbent arm is missing")
        incumbent_index = FUSION_ARM_NAMES.index(choice)
        layouts = tuple(
            strict_layout(arrays[f"relation_arm_{arm}_layout"], grid=grid_size)
            for arm in FUSION_ARM_NAMES
        )
        candidate = strict_layout(arrays["relation_truth_selector_layout"], grid=grid_size)
        if not np.array_equal(candidate, layouts[incumbent_index]):
            raise ValueError("frozen relation-selector layout differs from incumbent arm")
        relation_count = 2 * grid_size * (grid_size - 1)
        features = np.asarray(arrays["relation_features"])
        expected_feature_shape = (
            len(FUSION_ARM_NAMES),
            relation_count,
            len(FEATURE_NAMES),
        )
        if features.shape != expected_feature_shape or not np.isfinite(features).all():
            raise ValueError("frozen relation features changed shape or finiteness")
        scores = np.asarray(arrays["relation_expected_correct_scores"], dtype=np.float64)
        if scores.shape != (len(FUSION_ARM_NAMES),) or not np.isfinite(scores).all():
            raise ValueError("frozen relation-selector expected scores changed")
        frozen_layouts: list[np.ndarray] = []
        for layout in layouts:
            value = layout.copy()
            value.setflags(write=False)
            frozen_layouts.append(value)
        return cls(
            layouts=tuple(frozen_layouts),
            incumbent_index=incumbent_index,
            expected_correct_scores=_readonly(scores),
            relation_features=_readonly(features.astype(np.float32, copy=False)),
            grid_size=grid_size,
        )


@dataclass(frozen=True)
class ArmJointEvidence:
    """Fixed joint evidence projected onto every realised six-arm relation."""

    mapped_union_counts: np.ndarray
    mapped_logit_sums: np.ndarray
    head_hits: np.ndarray
    head_confidence_sums: np.ndarray
    head_logit_sums: np.ndarray


def map_joint_evidence_to_six_arms(
    evidence: JointRelationEvidence,
    roster: FrozenSixArmRoster,
) -> ArmJointEvidence:
    """Map immutable joint pairs onto six layouts without creating new edges."""

    if evidence.grid_size != roster.grid_size:
        raise ValueError("joint evidence and six-arm roster use different grids")
    slot_lookup: list[list[dict[int, int]]] = []
    head_lookup: list[dict[tuple[int, int], tuple[float, float]]] = []
    for axis in range(2):
        rows: list[dict[int, int]] = []
        for source in range(evidence.grid_size**2):
            rows.append(
                {
                    int(target): int(slot)
                    for slot, target in enumerate(evidence.candidates[axis, source])
                    if evidence.valid[axis, source, slot]
                }
            )
        slot_lookup.append(rows)
        selected: dict[tuple[int, int], tuple[float, float]] = {}
        for source, target, confidence in zip(
            evidence.head_sources[axis],
            evidence.head_targets[axis],
            evidence.head_confidences[axis],
            strict=True,
        ):
            slot = rows[int(source)][int(target)]
            selected[(int(source), int(target))] = (
                float(confidence),
                float(evidence.learned_logits[axis][int(source), slot]),
            )
        head_lookup.append(selected)

    shape = (len(FUSION_ARM_NAMES), 2)
    mapped_counts = np.zeros(shape, dtype=np.int32)
    mapped_logit_sums = np.zeros(shape, dtype=np.float64)
    head_hits = np.zeros(shape, dtype=np.int32)
    head_confidence_sums = np.zeros(shape, dtype=np.float64)
    head_logit_sums = np.zeros(shape, dtype=np.float64)
    for arm_index, layout in enumerate(roster.layouts):
        for edge in realised_edges(layout, grid=roster.grid_size):
            axis = AXIS_NAMES.index(edge.axis)
            slot = slot_lookup[axis][edge.source].get(edge.target)
            if slot is not None:
                mapped_counts[arm_index, axis] += 1
                mapped_logit_sums[arm_index, axis] += float(
                    evidence.learned_logits[axis][edge.source, slot]
                )
            selected = head_lookup[axis].get((edge.source, edge.target))
            if selected is not None:
                head_hits[arm_index, axis] += 1
                head_confidence_sums[arm_index, axis] += selected[0]
                head_logit_sums[arm_index, axis] += selected[1]
    return ArmJointEvidence(
        mapped_union_counts=_readonly(mapped_counts),
        mapped_logit_sums=_readonly(mapped_logit_sums),
        head_hits=_readonly(head_hits),
        head_confidence_sums=_readonly(head_confidence_sums),
        head_logit_sums=_readonly(head_logit_sums),
    )


@dataclass(frozen=True)
class JointRelationSelection:
    """One unchanged arm chosen by the fixed target-free dominance rule."""

    layout: np.ndarray
    selected_index: int
    incumbent_index: int
    arm_evidence: ArmJointEvidence
    selection_rule: str = SELECTION_RULE

    @property
    def selected_arm(self) -> str:
        return FUSION_ARM_NAMES[self.selected_index]

    @property
    def incumbent_arm(self) -> str:
        return FUSION_ARM_NAMES[self.incumbent_index]

    @property
    def changed(self) -> bool:
        return self.selected_index != self.incumbent_index


def select_fixed_head_dominant_arm(
    evidence: JointRelationEvidence,
    roster: FrozenSixArmRoster,
) -> JointRelationSelection:
    """Select only an axiswise non-regressing, total-head-improving whole arm."""

    mapped = map_joint_evidence_to_six_arms(evidence, roster)
    incumbent = roster.incumbent_index
    incumbent_hits = mapped.head_hits[incumbent]
    eligible = [
        index
        for index in range(len(FUSION_ARM_NAMES))
        if np.all(mapped.head_hits[index] >= incumbent_hits)
        and int(mapped.head_hits[index].sum()) > int(incumbent_hits.sum())
    ]
    if eligible:
        def key(index: int) -> tuple[float, ...]:
            delta = mapped.head_hits[index] - incumbent_hits
            return (
                float(mapped.head_hits[index].sum()),
                float(delta.min()),
                float(mapped.head_confidence_sums[index].sum()),
                float(mapped.head_logit_sums[index].sum()),
                float(roster.expected_correct_scores[index]),
                float(-index),
            )

        selected = max(eligible, key=key)
    else:
        selected = incumbent
    layout = strict_layout(roster.layouts[selected], grid=roster.grid_size).copy()
    layout.setflags(write=False)
    return JointRelationSelection(
        layout=layout,
        selected_index=selected,
        incumbent_index=incumbent,
        arm_evidence=mapped,
    )


__all__ = [
    "AXIS_NAMES",
    "ArmJointEvidence",
    "FIXED_HEAD_FRACTION",
    "FrozenSixArmRoster",
    "JointRelationEvidence",
    "JointRelationSelection",
    "SELECTION_RULE",
    "map_joint_evidence_to_six_arms",
    "reject_target_bearing_array_names",
    "select_fixed_head_dominant_arm",
]
