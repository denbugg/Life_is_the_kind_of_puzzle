from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from aiijc_puzzle import dualnaf_stack as stack


def test_preregistered_rosters_and_historical_exposure() -> None:
    config = stack.load_config()
    primary = stack.panel_records("primary")
    confirmation = stack.panel_records("confirmation")

    assert stack.sha256_file(stack.CONFIG_PATH) == stack.CONFIG_SHA256
    assert len(primary) == len(confirmation) == 32
    assert primary[0]["filename"] == "img_001431.png"
    assert primary[-1]["filename"] == "img_001145.png"
    assert confirmation[0]["filename"] == "img_001856.png"
    assert confirmation[-1]["filename"] == "img_004775.png"
    assert {row["filename"] for row in primary}.isdisjoint(row["filename"] for row in confirmation)
    assert (
        stack._filename_digest(primary) == config["protocol"]["primary"]["filename_nul_join_sha256"]
    )
    exposure = stack.audit_historical_exposure()
    assert exposure["freshness_claim"] is False
    assert exposure["legacy_exact_matches_to_current_calibration"] == 700
    assert exposure["primary_historically_exposed"] == 32
    assert exposure["confirmation_historically_exposed"] == 32


def test_prepare_cannot_accept_target_directory() -> None:
    assert "targets_dir" not in inspect.signature(stack.prepare_panel).parameters


def test_alpha0125_blend_is_same_index_rounded_uint8() -> None:
    original = np.zeros((576, 20, 20, 3), dtype=np.uint8)
    rendered = np.full_like(original, 255)
    blended = stack.blend_tiles_alpha0125(original, rendered)
    assert blended.dtype == np.uint8
    assert blended.shape == original.shape
    assert np.all(blended == 32)

    original[1] = 80
    rendered[1] = 160
    blended = stack.blend_tiles_alpha0125(original, rendered)
    assert np.all(blended[1] == 90)
    assert np.all(blended[0] == 32)


def test_paired_bootstrap_constant_vector() -> None:
    low, high = stack.paired_bootstrap([0.0125] * 32)
    assert low == pytest.approx(0.0125)
    assert high == pytest.approx(0.0125)


def test_numeric_gate_requires_D_to_beat_both_controls() -> None:
    score_summary = {
        stack.ARM_D: {"mean_ssim": 0.275},
        f"{stack.ARM_D}__minus__{stack.ARM_A}": {
            "paired_t_ci95": [0.01, 0.02],
            "paired_bootstrap_ci95": [0.01, 0.02],
            "wins_ties_losses": [32, 0, 0],
        },
        f"{stack.ARM_D}__minus__{stack.ARM_B}": {
            "paired_t_ci95": [0.001, 0.005],
            "paired_bootstrap_ci95": [0.001, 0.005],
            "wins_ties_losses": [24, 0, 8],
        },
    }
    safety_summary = {
        stack.ARM_D: {
            "within_tile_gradient_retention_vs_A": {
                "mean": 0.9,
                "min": 0.8,
                "max": 0.95,
            },
            "laplacian_retention_vs_A": {"mean": 0.8, "min": 0.7, "max": 0.9},
            "grid_ratio_relative_to_A": {"mean": 0.9, "min": 0.8, "max": 1.0},
            "all_predictions_distinct_across_boards": True,
        }
    }
    result = stack.evaluate_numeric_gate(score_summary, safety_summary)
    assert result["all_passed"] is True

    score_summary[f"{stack.ARM_D}__minus__{stack.ARM_B}"]["paired_t_ci95"][0] = -0.001
    result = stack.evaluate_numeric_gate(score_summary, safety_summary)
    assert result["all_passed"] is False


def test_confirmation_requires_numeric_and_manual_pass(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "report.json").write_text(
        json.dumps({"numeric_gate": {"all_passed": True}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="manual review"):
        stack._require_confirmation_authorized(tmp_path)

    (primary / "manual-review.json").write_text(
        json.dumps({"overall_verdict": "FAIL"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="manual review did not pass"):
        stack._require_confirmation_authorized(tmp_path)

    (primary / "manual-review.json").write_text(
        json.dumps({"overall_verdict": "PASS"}),
        encoding="utf-8",
    )
    stack._require_confirmation_authorized(tmp_path)
