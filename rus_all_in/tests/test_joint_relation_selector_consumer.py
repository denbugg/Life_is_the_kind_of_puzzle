from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from aiijc_puzzle.joint_relation_selector_consumer import (
    FrozenSixArmRoster,
    JointRelationEvidence,
    reject_target_bearing_array_names,
    select_fixed_head_dominant_arm,
)
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_relation_truth_selector import FEATURE_NAMES
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES
from aiijc_puzzle.tri_emitter_edge_verifier import candidate_pool_digest
from scripts import run_joint_relation_selector_consumer as runner


def _joint_arrays(
    right_head: tuple[int, int] = (8, 0),
    down_head: tuple[int, int] = (8, 2),
) -> dict[str, np.ndarray]:
    count = 9
    width = count - 1
    candidates = np.empty((2, count, width), dtype=np.int32)
    for axis in range(2):
        for source in range(count):
            candidates[axis, source] = [
                target for target in range(count) if target != source
            ]
    valid = np.ones_like(candidates, dtype=bool)
    emitter_topk = candidates[None].copy()
    digest = candidate_pool_digest(candidates, valid, emitter_topk)
    arrays: dict[str, np.ndarray] = {
        "union_candidates": candidates,
        "union_valid": valid,
        "emitter_topk": emitter_topk,
        "union_identity_digest_ascii": np.frombuffer(digest.encode(), dtype=np.uint8),
    }
    for axis_index, (axis_name, edge) in enumerate(
        zip(("right", "down"), (right_head, down_head), strict=True)
    ):
        logits = np.zeros((count, width), dtype=np.float32)
        confidence = np.zeros_like(logits)
        source, target = edge
        slot = int(np.flatnonzero(candidates[axis_index, source] == target)[0])
        logits[source, slot] = 4.0
        confidence[source, slot] = 3.0
        arrays[f"learned_logits__{axis_name}"] = logits
        arrays[f"learned_joint_confidence__{axis_name}"] = confidence
        arrays[f"learned_head_sources__{axis_name}"] = np.asarray(
            [source], dtype=np.int32
        )
        arrays[f"learned_head_targets__{axis_name}"] = np.asarray(
            [target], dtype=np.int32
        )
        arrays[f"learned_head_confidences__{axis_name}"] = np.asarray(
            [3.0], dtype=np.float32
        )
        arrays[f"learned_head_requested__{axis_name}"] = np.asarray(1, dtype=np.int32)
        arrays[f"learned_reciprocal_count__{axis_name}"] = np.asarray(
            1, dtype=np.int32
        )
    return arrays


def _roster_arrays(*, duplicate_other_arms: bool = False) -> dict[str, np.ndarray]:
    base = np.arange(9, dtype=np.int32)
    layouts = [
        base,
        np.roll(base, 1),
        np.roll(base, 2),
        base[::-1],
        np.asarray([0, 3, 6, 1, 4, 7, 2, 5, 8], dtype=np.int32),
        np.roll(base, 3),
    ]
    if duplicate_other_arms:
        layouts[2:] = [base.copy() for _ in layouts[2:]]
    arrays = {
        "relation_features": np.zeros(
            (6, 12, len(FEATURE_NAMES)), dtype=np.float32
        ),
        "relation_expected_correct_scores": np.asarray(
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0], dtype=np.float64
        ),
        "relation_truth_selector_layout": base,
    }
    arrays.update(
        {
            f"relation_arm_{arm}_layout": layout
            for arm, layout in zip(FUSION_ARM_NAMES, layouts, strict=True)
        }
    )
    return arrays


def _roster(arrays: dict[str, np.ndarray]) -> FrozenSixArmRoster:
    return FrozenSixArmRoster.from_case_arrays(
        arrays,
        {"choice": "raw", "arm_names": list(FUSION_ARM_NAMES)},
        grid_size=3,
    )


def test_fixed_joint_head_selects_only_a_dominant_existing_whole_arm() -> None:
    evidence = JointRelationEvidence.from_case_arrays(_joint_arrays(), grid_size=3)
    roster = _roster(_roster_arrays())
    result = select_fixed_head_dominant_arm(evidence, roster)

    assert result.incumbent_arm == "raw"
    assert result.selected_arm == "logistic"
    assert result.changed
    assert result.arm_evidence.head_hits[0].tolist() == [0, 0]
    assert result.arm_evidence.head_hits[1].tolist() == [1, 1]
    np.testing.assert_array_equal(result.layout, roster.layouts[1])
    np.testing.assert_array_equal(np.sort(result.layout), np.arange(9))


def test_axiswise_head_loss_refuses_switch_even_when_other_axis_improves() -> None:
    # The incumbent realises 5->8 down; the rolled arm realises 8->0 right but
    # loses the incumbent's down head.  No one-axis trade is authorized.
    evidence = JointRelationEvidence.from_case_arrays(
        _joint_arrays(right_head=(8, 0), down_head=(5, 8)), grid_size=3
    )
    roster = _roster(_roster_arrays(duplicate_other_arms=True))
    result = select_fixed_head_dominant_arm(evidence, roster)
    assert result.selected_arm == "raw"
    assert not result.changed


def test_joint_schema_rejects_misaligned_head_and_target_arrays() -> None:
    arrays = _joint_arrays()
    arrays["learned_head_targets__right"] = np.asarray([7], dtype=np.int32)
    with pytest.raises(ValueError, match="misaligned"):
        JointRelationEvidence.from_case_arrays(arrays, grid_size=3)

    with pytest.raises(RuntimeError, match="forbidden arrays"):
        reject_target_bearing_array_names(("case_0000__target_slots",))
    # Historical model provenance is not confused with a truth label.
    reject_target_bearing_array_names(("case_0000__relation_truth_selector_layout",))

    arrays = _joint_arrays()
    arrays["opaque_y"] = np.asarray([1], dtype=np.uint8)
    with pytest.raises(ValueError, match="unexpected=.*opaque_y"):
        JointRelationEvidence.from_case_arrays(arrays, grid_size=3)


def _write_selection_freeze(tmp_path, *, config_sha: str = "fixed") -> None:
    archive = tmp_path / runner.OUTPUT_ARCHIVE
    with archive.open("wb") as stream:
        np.savez_compressed(stream, case_0000__candidate_layout=np.arange(576))
    metadata = tmp_path / runner.OUTPUT_METADATA
    metadata.write_text(
        json.dumps(
            {
                "schema": runner.OUTPUT_METADATA_SCHEMA,
                "config_sha256": config_sha,
                "contains_exact_references_or_labels": False,
                "contains_pixels": False,
                "layout_only_original_upright_tile_identities": True,
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    freeze = {
        "schema": runner.OUTPUT_FREEZE_SCHEMA,
        "created_before_exact_reference_reconstruction": True,
        "contains_evaluation_references_or_labels": False,
        "config_sha256": config_sha,
        "artifacts": {
            "archive": {"sha256": sha256_file(archive)},
            "metadata": {"sha256": sha256_file(metadata)},
        },
    }
    (tmp_path / runner.OUTPUT_FREEZE).write_text(json.dumps(freeze), encoding="utf-8")


def test_reference_loader_runs_only_after_selection_hash_verification(tmp_path) -> None:
    _write_selection_freeze(tmp_path)
    calls: list[str] = []
    result = runner.score_after_verified_freeze(
        tmp_path,
        "fixed",
        reference_loader=lambda _verified: calls.append("labels") or 7,
        scorer=lambda _verified, value: value + 1,
    )
    assert result == 8
    assert calls == ["labels"]

    with (tmp_path / runner.OUTPUT_ARCHIVE).open("ab") as stream:
        stream.write(b"tampered")
    calls.clear()
    with pytest.raises(RuntimeError, match="changed after freeze"):
        runner.score_after_verified_freeze(
            tmp_path,
            "fixed",
            reference_loader=lambda _verified: calls.append("labels"),
            scorer=lambda _verified, value: value,
        )
    assert calls == []


def test_scoring_reports_pairs_exact_absolute_manhattan_and_radius2(tmp_path) -> None:
    reference = np.arange(576, dtype=np.int32)
    control = reference.copy()
    control[[0, 1]] = control[[1, 0]]
    archive_path = tmp_path / "selection.npz"
    with archive_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            case_0000__incumbent_layout=control,
            case_0000__candidate_layout=reference,
        )
    metadata_path = tmp_path / "selection.json"
    metadata_path.write_text("{}", encoding="utf-8")
    verified = runner.VerifiedBundle(
        archive=archive_path,
        metadata=metadata_path,
        freeze=tmp_path / "freeze.json",
        rows=(
            {
                "prefix": "case_0000",
                "case_id": "case",
                "source_filename": "source.png",
                "draw_index": 0,
                "changed": True,
                "incumbent_arm": "raw",
                "selected_arm": "logistic",
            },
        ),
        archive_sha256=sha256_file(archive_path),
        metadata_sha256=sha256_file(metadata_path),
    )
    metrics = runner.score_frozen_selection(
        verified,
        {"case": SimpleNamespace(tile_at_position=reference)},
        {
            "mean_satisfied_pairs_delta_minimum": 0.0,
            "mean_exact_tiles_delta_minimum": 0.0,
            "mean_absolute_manhattan_delta_maximum": 0.0,
            "mean_radius2_recall_delta_minimum": 0.0,
            "require_at_least_one_changed_case": True,
            "require_at_least_one_strict_aggregate_improvement": True,
        },
    )
    assert metrics["aggregate"]["satisfied_pairs"]["delta"] > 0
    assert metrics["aggregate"]["exact_tiles"]["delta"] == 2
    assert metrics["aggregate"]["absolute_mean_manhattan"]["delta"] < 0
    assert metrics["aggregate"]["radius2_recall"]["delta"] >= 0
    assert metrics["gate"]["passed"]


def test_unsigned_template_is_populated_but_explicitly_blocked() -> None:
    config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    runner._require_exact_contract(config)
    changed = json.loads(json.dumps(config))
    changed["selection"]["fixed_head_count_per_axis"] = 28
    with pytest.raises(RuntimeError, match="selection changed"):
        runner._require_exact_contract(changed)
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        runner._load_signed_config(runner.DEFAULT_CONFIG)
