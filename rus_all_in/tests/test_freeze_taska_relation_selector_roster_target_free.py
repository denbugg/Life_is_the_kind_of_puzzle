from __future__ import annotations

import json

import numpy as np
import pytest

from aiijc_puzzle.joint_relation_selector_consumer import RELATION_ROSTER_CASE_KEYS
from aiijc_puzzle.taska_relation_truth_selector import FEATURE_NAMES
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES
from scripts import freeze_taska_relation_selector_roster_target_free as freezer


def _relation_outputs() -> dict[str, np.ndarray]:
    base = np.arange(9, dtype=np.int32)
    layouts = (
        base,
        np.roll(base, 1),
        np.roll(base, 2),
        base[::-1],
        np.asarray([0, 3, 6, 1, 4, 7, 2, 5, 8], dtype=np.int32),
        np.roll(base, 3),
    )
    result: dict[str, np.ndarray] = {
        "relation_features": np.zeros(
            (6, 12, len(FEATURE_NAMES)), dtype=np.float32
        ),
        "relation_expected_correct_scores": np.asarray(
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64
        ),
        "relation_truth_selector_layout": base,
        # Real confirmation inference has additional target-free arrays.  The
        # freezer must not copy them into the normalized sibling contract.
        "parent_cost_right": np.zeros((9, 9), dtype=np.float32),
    }
    result.update(
        {
            f"relation_arm_{arm}_layout": layout
            for arm, layout in zip(FUSION_ARM_NAMES, layouts, strict=True)
        }
    )
    return result


def test_normalizer_keeps_exact_label_free_allowlist_and_strict_roster() -> None:
    arrays, row = freezer.normalize_relation_case(
        _relation_outputs(),
        {"choice": "raw", "control_choice": "raw", "changed_from_control": False},
        grid_size=3,
    )
    assert set(arrays) == RELATION_ROSTER_CASE_KEYS
    assert "parent_cost_right" not in arrays
    assert row == {
        "choice": "raw",
        "control_choice": "raw",
        "changed_from_control": False,
        "arm_names": list(FUSION_ARM_NAMES),
    }
    for arm in FUSION_ARM_NAMES:
        layout = arrays[f"relation_arm_{arm}_layout"]
        np.testing.assert_array_equal(np.sort(layout), np.arange(9))


def test_normalizer_rejects_hidden_labels_and_non_permutation() -> None:
    arrays = _relation_outputs()
    arrays["target_slots"] = np.zeros((2, 9), dtype=np.int16)
    with pytest.raises(RuntimeError, match="forbidden arrays"):
        freezer.normalize_relation_case(
            arrays,
            {"choice": "raw"},
            grid_size=3,
        )

    arrays = _relation_outputs()
    arrays["relation_arm_raw_layout"] = np.zeros(9, dtype=np.int32)
    with pytest.raises(ValueError, match="strict"):
        freezer.normalize_relation_case(
            arrays,
            {"choice": "raw"},
            grid_size=3,
        )


def test_normalizer_rejects_incumbent_layout_or_choice_drift() -> None:
    arrays = _relation_outputs()
    with pytest.raises(ValueError, match="differs from incumbent"):
        freezer.normalize_relation_case(
            arrays,
            {"choice": "logistic"},
            grid_size=3,
        )
    with pytest.raises(ValueError, match="incumbent arm"):
        freezer.normalize_relation_case(
            arrays,
            {"choice": "unknown"},
            grid_size=3,
        )


def test_unsigned_roster_template_fixes_v2_source_and_is_blocked() -> None:
    config = json.loads(freezer.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    freezer._require_exact_contract(config)
    assert config["rule_commitment_sha256"] == freezer.rule_commitment_sha256(config)
    assert config["source_protocol"]["dev_digest"] == (
        "93112f89096f8e9555172f10f6934fd8dd5abf48a8029b86a8803d507e79e87e"
    )
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        freezer._load_signed_config(freezer.DEFAULT_CONFIG)

    changed = json.loads(json.dumps(config))
    changed["rule_commitment_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="rule commitment"):
        freezer._require_exact_contract(changed)


def test_exclusive_target_free_writers_refuse_overwrite(tmp_path) -> None:
    json_path = tmp_path / "metadata.json"
    freezer._write_json(json_path, {"contains_exact_references_or_labels": False})
    with pytest.raises(FileExistsError):
        freezer._write_json(json_path, {})

    archive_path = tmp_path / "roster.npz"
    freezer._write_npz(archive_path, {"layout": np.arange(9)})
    with pytest.raises(FileExistsError):
        freezer._write_npz(archive_path, {"layout": np.arange(9)})
