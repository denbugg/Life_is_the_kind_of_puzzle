from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_joint_native_head_arm_fit_v2 as v2  # noqa: E402
import score_joint_native_head_arm_fit_v3 as v3  # noqa: E402


def test_signed_overlay_changes_exactly_one_manifest_bound_leaf() -> None:
    overlay, _ = v3._load_signed_overlay(v3.DEFAULT_CONFIG)
    base, _ = v2._load_signed_config(ROOT / overlay["bound_inputs"]["v2_config"]["path"])
    repaired, proof = v3._apply_one_field_overlay(
        base,
        overlay,
        ROOT / "data/interim/validation_manifest.json",
    )
    differences = v3._json_leaf_differences(base, repaired)
    assert len(differences) == proof["changed_leaf_count"] == 1
    assert differences[0][0] == v3.EXPECTED_CORRECTION_PATH
    assert proof["derived_from_target_pixels"] is False


def test_overlay_fails_closed_if_any_second_change_is_requested() -> None:
    overlay, _ = v3._load_signed_overlay(v3.DEFAULT_CONFIG)
    base, _ = v2._load_signed_config(ROOT / overlay["bound_inputs"]["v2_config"]["path"])
    broken = copy.deepcopy(overlay)
    broken["correction"]["filename"] = "img_000934.png"
    with pytest.raises(RuntimeError, match="preregistered one-field"):
        v3._apply_one_field_overlay(
            base,
            broken,
            ROOT / "data/interim/validation_manifest.json",
        )


def test_authoritative_value_is_existing_manifest_metadata() -> None:
    overlay, _ = v3._load_signed_overlay(v3.DEFAULT_CONFIG)
    manifest = json.loads((ROOT / "data/interim/validation_manifest.json").read_text())
    row = next(
        row
        for row in manifest["splits"]["train"]
        if row["filename"] == overlay["correction"]["filename"]
    )
    assert row["target_sha256"] == overlay["correction"]["to_sha256"]
    assert overlay["correction"]["derived_from_target_pixels"] is False
