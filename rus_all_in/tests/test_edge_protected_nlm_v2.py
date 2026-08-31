from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from aiijc_puzzle.edge_protected_nlm import protected_masks
from aiijc_puzzle.edge_protected_nlm_v2 import SOBEL_THRESHOLD, blend_h28safe_h40flat

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts/run_edge_protected_nlm_v2.py"
SPEC = importlib.util.spec_from_file_location("edge_protected_runner_v2", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _image(value: int = 100) -> np.ndarray:
    image = np.full((480, 480, 3), value, dtype=np.uint8)
    image[200:280, 200:280] = min(value + 80, 255)
    return image


def test_v2_blend_uses_exact_h20_derived_t40_mask() -> None:
    h20 = _image(80)
    h28 = _image(100)
    h40 = _image(140)
    output, dilated, soft, diagnostics = blend_h28safe_h40flat(h20, h28, h40)
    expected_dilated, expected_soft, fraction = protected_masks(
        h20,
        sobel_threshold=SOBEL_THRESHOLD,
    )
    expected = np.rint(
        expected_soft[..., None] * h28.astype(np.float32)
        + (1.0 - expected_soft[..., None]) * h40.astype(np.float32)
    ).astype(np.uint8)
    np.testing.assert_array_equal(dilated, expected_dilated)
    np.testing.assert_array_equal(soft, expected_soft)
    np.testing.assert_array_equal(output, expected)
    assert diagnostics["binary_dilated_protected_fraction"] == pytest.approx(fraction)


def test_v2_safe_and_aggressive_identity_is_preserved() -> None:
    source = _image(80)
    shared = _image(120)
    output, _, _, _ = blend_h28safe_h40flat(source, shared, shared)
    np.testing.assert_array_equal(output, shared)


def test_runner_contract_hash_rosters_and_panel_disjointness() -> None:
    config, primary = runner.load_contract(runner.MANIFEST, "primary")
    _, confirmation = runner.load_contract(runner.MANIFEST, "confirmation")
    assert runner.sha256_file(runner.CONFIG) == runner.CONFIG_SHA256
    assert len(primary) == len(confirmation) == 60
    assert primary[0]["filename"] == "img_002280.png"
    assert primary[-1]["filename"] == "img_005813.png"
    assert confirmation[0]["filename"] == "img_000390.png"
    assert confirmation[-1]["filename"] == "img_003624.png"
    assert {row["filename"] for row in primary}.isdisjoint(row["filename"] for row in confirmation)
    assert runner.names_digest(primary) == config["protocol"]["primary"]["filenames_newline_sha256"]
    assert (
        runner.roster_digest(primary)
        == config["protocol"]["primary"]["filename_input_roster_sha256"]
    )
    confirmation_config = config["protocol"]["confirmation_if_and_only_if_primary_passes"]
    assert runner.names_digest(confirmation) == confirmation_config["filenames_newline_sha256"]
    assert runner.roster_digest(confirmation) == confirmation_config["filename_input_roster_sha256"]


def test_config_agreement_rejects_post_freeze_arm_change() -> None:
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    runner.validate_config_agreement(config)
    config["arms"][2]["aggressive_h"] = 50
    with pytest.raises(RuntimeError, match="executable arms"):
        runner.validate_config_agreement(config)


def test_numeric_gate_names_exactly_match_immutable_config() -> None:
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    thresholds = config["primary_promotion_gate"]
    safety = {
        "mean_luminance_gradient_retention": 0.9,
        "minimum_luminance_gradient_retention": 0.8,
        "mean_chroma_gradient_retention": 0.9,
        "minimum_chroma_gradient_retention": 0.8,
        "mean_laplacian_retention": 0.8,
        "minimum_laplacian_retention": 0.7,
        "mean_grid_ratio_relative_to_baseline": 0.9,
        "maximum_grid_ratio_relative_to_baseline": 1.0,
        "mean_protected_pixel_fraction": 0.5,
        "minimum_protected_pixel_fraction": 0.4,
        "maximum_protected_pixel_fraction": 0.7,
        "maximum_clipped_fraction_increase": 0.0,
        "distinct_from_A_and_B_on_every_board": True,
    }
    result = runner.numeric_gate(
        [0.29] * 60,
        [0.25] * 60,
        [0.27] * 60,
        safety,
        thresholds,
    )
    assert result["all_passed"] is True
    assert set(result["checks"]) == set(thresholds) - {"manual_severe_new_artifacts_allowed"}


def test_commitment_reload_detects_artifact_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {"filename": "synthetic.png", "input_sha256": "a" * 64}
    image = _image()
    arrays = {
        "dirty": image,
        **{f"prediction__{name}": image.copy() for name in runner.PREDICTION_NAMES},
    }
    artifact_path = tmp_path / "artifacts" / "synthetic.npz"
    runner.write_npz_exclusive(artifact_path, arrays)
    artifact = {
        "path": "artifacts/synthetic.npz",
        "file_sha256": runner.sha256_file(artifact_path),
        "array_sha256": {name: runner.array_digest(value) for name, value in arrays.items()},
    }
    prediction_hashes = {
        name.removeprefix("prediction__"): runner.image_digest(value)
        for name, value in arrays.items()
        if name.startswith("prediction__")
    }
    fixed_sources = {"runner.py": "b" * 64}
    monkeypatch.setattr(runner, "source_hashes", lambda: fixed_sources)
    commitment = {
        "source_sha256": fixed_sources,
        "filenames_newline_sha256": runner.names_digest([record]),
        "filename_input_roster_sha256": runner.roster_digest([record]),
        "prediction_names": list(runner.PREDICTION_NAMES),
        "per_board": [
            {
                "filename": record["filename"],
                "input_sha256": record["input_sha256"],
                "prediction_sha256": prediction_hashes,
                "artifact": artifact,
            }
        ],
    }
    commitment["commitment_sha256"] = runner.canonical_digest(commitment)
    assert len(runner.reload_committed_predictions(commitment, [record], tmp_path)) == 1

    os.chmod(artifact_path, 0o644)
    with artifact_path.open("ab") as stream:
        stream.write(b"mutation")
    with pytest.raises(RuntimeError, match="artifact file changed"):
        runner.reload_committed_predictions(commitment, [record], tmp_path)


def test_confirmation_fails_closed_without_bound_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "PRIMARY_ROOT", tmp_path / "primary")
    with pytest.raises(RuntimeError, match="manual review"):
        runner.authorized_confirmation()
