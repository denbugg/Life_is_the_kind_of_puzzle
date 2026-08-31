from __future__ import annotations

import json

import numpy as np
import pytest

from aiijc_puzzle.joint_relation_selector_consumer import (
    FrozenSixArmRoster,
    JointRelationEvidence,
)
from aiijc_puzzle.joint_relation_selector_portfolio import (
    PORTFOLIO_MEMBER_NAMES,
    build_frozen_selector_portfolio,
    map_full_joint_evidence_to_six_arms,
    select_source_normalized_dominant_arm,
    select_union_dense_dominant_arm,
)
from aiijc_puzzle.taska_relation_truth_selector import FEATURE_NAMES, realised_edges
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES
from aiijc_puzzle.tri_emitter_edge_verifier import candidate_pool_digest
from scripts import freeze_joint_relation_selector_scale256_portfolio as freezer


def _layouts() -> list[np.ndarray]:
    base = np.arange(9, dtype=np.int32)
    return [
        base,
        np.roll(base, 1),
        np.roll(base, 2),
        base[::-1],
        np.asarray([0, 3, 6, 1, 4, 7, 2, 5, 8], dtype=np.int32),
        np.roll(base, 3),
    ]


def _roster_arrays() -> dict[str, np.ndarray]:
    layouts = _layouts()
    arrays: dict[str, np.ndarray] = {
        "relation_features": np.zeros(
            (6, 12, len(FEATURE_NAMES)), dtype=np.float32
        ),
        "relation_expected_correct_scores": np.asarray(
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64
        ),
        "relation_truth_selector_layout": layouts[0],
    }
    arrays.update(
        {
            f"relation_arm_{arm}_layout": layout
            for arm, layout in zip(FUSION_ARM_NAMES, layouts, strict=True)
        }
    )
    return arrays


def _roster(arrays: dict[str, np.ndarray] | None = None) -> FrozenSixArmRoster:
    return FrozenSixArmRoster.from_case_arrays(
        arrays or _roster_arrays(),
        {"choice": "raw", "arm_names": list(FUSION_ARM_NAMES)},
        grid_size=3,
    )


def _edge_sets(layout: np.ndarray) -> tuple[set[tuple[int, int]], ...]:
    result = (set(), set())
    for edge in realised_edges(layout, grid=3):
        axis = 0 if edge.axis == "right" else 1
        result[axis].add((edge.source, edge.target))
    return result


def _joint_arrays(
    *,
    preferred_index: int = 1,
    axis_confidence_scale: tuple[float, float] = (5.0, 5.0),
    sparse: bool = False,
) -> dict[str, np.ndarray]:
    count = 9
    if sparse:
        width = 2
        candidates = np.empty((2, count, width), dtype=np.int32)
        base_edges = _edge_sets(_layouts()[0])
        for axis in range(2):
            by_source = {source: target for source, target in base_edges[axis]}
            for source in range(count):
                first = by_source.get(source, (source + 1) % count)
                second = (first + 1) % count
                if second in (source, first):
                    second = (second + 1) % count
                candidates[axis, source] = (first, second)
    else:
        width = count - 1
        candidates = np.empty((2, count, width), dtype=np.int32)
        for axis in range(2):
            for source in range(count):
                candidates[axis, source] = [
                    target for target in range(count) if target != source
                ]
    valid = np.ones_like(candidates, dtype=bool)
    emitter_topk = candidates[None, :, :, :1].copy()
    arrays: dict[str, np.ndarray] = {
        "union_candidates": candidates,
        "union_valid": valid,
        "emitter_topk": emitter_topk,
    }
    preferred = _edge_sets(_layouts()[preferred_index])
    incumbent = _edge_sets(_layouts()[0])
    for axis, axis_name in enumerate(("right", "down")):
        logits = np.full((count, width), -4.0, dtype=np.float32)
        confidence = np.full((count, width), -3.0, dtype=np.float32)
        for source in range(count):
            for slot, target in enumerate(candidates[axis, source]):
                edge = (source, int(target))
                if edge in incumbent[axis]:
                    logits[source, slot] = 0.5
                    confidence[source, slot] = 0.5
                if edge in preferred[axis]:
                    logits[source, slot] = 4.0
                    confidence[source, slot] = axis_confidence_scale[axis]
        exclusive = sorted(preferred[axis] - incumbent[axis])
        if not exclusive:
            exclusive = sorted(preferred[axis])
        source, target = exclusive[0]
        slots = np.flatnonzero(candidates[axis, source] == target)
        if len(slots) == 0:
            source, target = sorted(incumbent[axis])[0]
            slots = np.flatnonzero(candidates[axis, source] == target)
        slot = int(slots[0])
        arrays[f"learned_logits__{axis_name}"] = logits
        arrays[f"learned_joint_confidence__{axis_name}"] = confidence
        arrays[f"learned_head_sources__{axis_name}"] = np.asarray(
            [source], dtype=np.int32
        )
        arrays[f"learned_head_targets__{axis_name}"] = np.asarray(
            [target], dtype=np.int32
        )
        arrays[f"learned_head_confidences__{axis_name}"] = np.asarray(
            [confidence[source, slot]], dtype=np.float32
        )
        arrays[f"learned_head_requested__{axis_name}"] = np.asarray(
            1, dtype=np.int32
        )
        arrays[f"learned_reciprocal_count__{axis_name}"] = np.asarray(
            1, dtype=np.int32
        )
    digest = candidate_pool_digest(candidates, valid, emitter_topk)
    arrays["union_identity_digest_ascii"] = np.frombuffer(
        digest.encode("ascii"), dtype=np.uint8
    )
    return arrays


def _relabel_joint(
    arrays: dict[str, np.ndarray], permutation: np.ndarray
) -> dict[str, np.ndarray]:
    candidates = arrays["union_candidates"]
    result: dict[str, np.ndarray] = {}
    relabeled_candidates = np.empty_like(candidates)
    relabeled_valid = np.empty_like(arrays["union_valid"])
    relabeled_emitter = np.empty_like(arrays["emitter_topk"])
    for old_source, new_source in enumerate(permutation):
        relabeled_candidates[:, new_source] = permutation[candidates[:, old_source]]
        relabeled_valid[:, new_source] = arrays["union_valid"][:, old_source]
        relabeled_emitter[:, :, new_source] = permutation[
            arrays["emitter_topk"][:, :, old_source]
        ]
    result.update(
        {
            "union_candidates": relabeled_candidates,
            "union_valid": relabeled_valid,
            "emitter_topk": relabeled_emitter,
        }
    )
    for axis_name in ("right", "down"):
        for key in ("learned_logits", "learned_joint_confidence"):
            value = np.empty_like(arrays[f"{key}__{axis_name}"])
            for old_source, new_source in enumerate(permutation):
                value[new_source] = arrays[f"{key}__{axis_name}"][old_source]
            result[f"{key}__{axis_name}"] = value
        result[f"learned_head_sources__{axis_name}"] = permutation[
            arrays[f"learned_head_sources__{axis_name}"]
        ].astype(np.int32)
        result[f"learned_head_targets__{axis_name}"] = permutation[
            arrays[f"learned_head_targets__{axis_name}"]
        ].astype(np.int32)
        for suffix in (
            "learned_head_confidences",
            "learned_head_requested",
            "learned_reciprocal_count",
        ):
            result[f"{suffix}__{axis_name}"] = arrays[
                f"{suffix}__{axis_name}"
            ].copy()
    digest = candidate_pool_digest(
        relabeled_candidates, relabeled_valid, relabeled_emitter
    )
    result["union_identity_digest_ascii"] = np.frombuffer(
        digest.encode("ascii"), dtype=np.uint8
    )
    return result


def test_portfolio_contains_only_exact_frozen_layouts_and_keep() -> None:
    evidence = JointRelationEvidence.from_case_arrays(_joint_arrays(), grid_size=3)
    roster = _roster()
    portfolio = build_frozen_selector_portfolio(evidence, roster)

    assert PORTFOLIO_MEMBER_NAMES[0] == "incumbent_keep"
    assert portfolio.selected_indices[0] == roster.incumbent_index
    for layout, selected in zip(
        portfolio.layouts, portfolio.selected_indices, strict=True
    ):
        np.testing.assert_array_equal(layout, roster.layouts[selected])
        np.testing.assert_array_equal(np.sort(layout), np.arange(9))
        assert not layout.flags.writeable


def test_freeze_payload_archives_every_member_and_full_six_arm_diagnostics() -> None:
    evidence = JointRelationEvidence.from_case_arrays(_joint_arrays(), grid_size=3)
    roster = _roster()
    payload, indices, arm_names = freezer.build_case_freeze_payload(evidence, roster)

    assert set(payload) == {
        "portfolio_layouts",
        "selected_arm_indices",
        "arm_union_coverage_counts",
        "arm_union_dense_confidence_sums",
        "arm_union_dense_confidence_means",
        "arm_normalized_logit_sums",
        "arm_normalized_confidence_sums",
        "arm_normalized_combined_sums",
        "arm_missing_edge_counts",
        "arm_legacy_head_hits",
    }
    assert payload["portfolio_layouts"].shape == (len(PORTFOLIO_MEMBER_NAMES), 9)
    assert payload["selected_arm_indices"].shape == (len(PORTFOLIO_MEMBER_NAMES),)
    for key, value in payload.items():
        if key not in {"portfolio_layouts", "selected_arm_indices"}:
            assert value.shape == (len(FUSION_ARM_NAMES), 2)
    assert tuple(payload["selected_arm_indices"]) == indices
    assert arm_names == tuple(FUSION_ARM_NAMES[index] for index in indices)
    np.testing.assert_array_equal(payload["portfolio_layouts"][0], roster.layouts[0])


def test_union_dense_rule_refuses_a_right_for_down_trade() -> None:
    arrays = _joint_arrays(axis_confidence_scale=(9.0, -2.0))
    evidence = JointRelationEvidence.from_case_arrays(arrays, grid_size=3)
    roster = _roster()
    mapped = map_full_joint_evidence_to_six_arms(evidence, roster)
    preferred = 1
    assert (
        mapped.union_dense_confidence_means[preferred, 0]
        > mapped.union_dense_confidence_means[roster.incumbent_index, 0]
    )
    assert (
        mapped.union_dense_confidence_means[preferred, 1]
        < mapped.union_dense_confidence_means[roster.incumbent_index, 1]
    )
    assert select_union_dense_dominant_arm(evidence, roster).selected_index != preferred


def test_source_normalized_rule_penalizes_every_missing_realised_edge() -> None:
    evidence = JointRelationEvidence.from_case_arrays(
        _joint_arrays(sparse=True), grid_size=3
    )
    roster = _roster()
    mapped = map_full_joint_evidence_to_six_arms(evidence, roster)
    relation_count_per_axis = 3 * 2

    np.testing.assert_array_equal(
        mapped.union_coverage_counts + mapped.missing_edge_counts,
        np.full((6, 2), relation_count_per_axis),
    )
    assert np.all(np.isfinite(mapped.normalized_combined_sums))
    selection = select_source_normalized_dominant_arm(evidence, roster)
    assert selection.selected_index in range(6)
    np.testing.assert_array_equal(selection.layout, roster.layouts[selection.selected_index])


def test_source_normalized_rule_can_switch_on_two_axis_full_evidence_gain() -> None:
    evidence = JointRelationEvidence.from_case_arrays(
        _joint_arrays(preferred_index=3), grid_size=3
    )
    roster = _roster()
    selection = select_source_normalized_dominant_arm(evidence, roster)

    assert selection.selected_index == 3
    assert np.all(
        selection.arm_evidence.normalized_combined_sums[3]
        > selection.arm_evidence.normalized_combined_sums[roster.incumbent_index]
    )
    np.testing.assert_array_equal(selection.layout, roster.layouts[3])


def test_all_rules_are_equivariant_to_a_tile_identity_bijection() -> None:
    arrays = _joint_arrays()
    roster_arrays = _roster_arrays()
    evidence = JointRelationEvidence.from_case_arrays(arrays, grid_size=3)
    roster = _roster(roster_arrays)
    original = build_frozen_selector_portfolio(evidence, roster)
    permutation = np.asarray([4, 8, 1, 6, 0, 3, 7, 2, 5], dtype=np.int32)

    relabeled_roster_arrays = {
        key: value.copy() for key, value in roster_arrays.items()
    }
    relabeled_roster_arrays["relation_truth_selector_layout"] = permutation[
        roster_arrays["relation_truth_selector_layout"]
    ]
    for arm in FUSION_ARM_NAMES:
        key = f"relation_arm_{arm}_layout"
        relabeled_roster_arrays[key] = permutation[roster_arrays[key]]
    relabeled_evidence = JointRelationEvidence.from_case_arrays(
        _relabel_joint(arrays, permutation), grid_size=3
    )
    relabeled_roster = _roster(relabeled_roster_arrays)
    relabeled = build_frozen_selector_portfolio(relabeled_evidence, relabeled_roster)

    assert relabeled.selected_indices == original.selected_indices
    for before, after in zip(original.layouts, relabeled.layouts, strict=True):
        np.testing.assert_array_equal(after, permutation[before])


def test_unsigned_scale256_template_fails_before_any_bundle_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads(freezer.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config["selection"] == freezer.selection_contract()
    assert config["rule_commitment_sha256"] == freezer.rule_commitment_sha256(config)
    touched = False

    def forbidden_lookup(*_args: object, **_kwargs: object) -> None:
        nonlocal touched
        touched = True
        raise AssertionError("blocked template attempted an artifact lookup")

    monkeypatch.setattr(freezer, "_verified_record", forbidden_lookup)
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        freezer._load_signed_config(freezer.DEFAULT_CONFIG)
    assert not touched
