from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from aiijc_puzzle.edge_protected_nlm import (
    ProtectedArm,
    blend_protected,
    colored_nlm,
    protected_masks,
    protected_weight,
)

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_edge_protected_nlm.py"
SPEC = importlib.util.spec_from_file_location("edge_protected_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _image() -> np.ndarray:
    image = np.full((480, 480, 3), 100, dtype=np.uint8)
    image[200:280, 200:280] = 220
    return image


def test_arm_validation() -> None:
    with pytest.raises(ValueError):
        ProtectedArm("bad", 29, 30.0)
    with pytest.raises(ValueError):
        ProtectedArm("bad", 40, 0.0)


def test_grid_and_content_edges_are_protected() -> None:
    weight, fraction = protected_weight(_image(), sobel_threshold=30.0)
    assert weight.shape == (480, 480)
    assert weight.dtype == np.float32
    assert 0 < fraction < 1
    assert weight[199:202, 240].max() > 0.9
    assert weight[239, 19:22].max() > 0.9
    assert weight[110, 110] < 0.1


def test_persisted_mask_components_reproduce_public_weight() -> None:
    dilated, soft, fraction = protected_masks(_image(), sobel_threshold=30.0)
    weight, public_fraction = protected_weight(_image(), sobel_threshold=30.0)
    assert dilated.dtype == np.bool_ and dilated.shape == (480, 480)
    assert soft.dtype == np.float32 and soft.shape == (480, 480)
    np.testing.assert_array_equal(soft, weight)
    assert fraction == public_fraction == pytest.approx(float(dilated.mean()))


def test_identical_sources_remain_exact_identity() -> None:
    image = _image()
    output, diagnostics = blend_protected(image, image, sobel_threshold=30.0)
    np.testing.assert_array_equal(output, image)
    assert 0 < diagnostics["binary_dilated_protected_fraction"] < 1


def test_colored_nlm_is_strict_and_deterministic() -> None:
    image = _image()
    first = colored_nlm(image, 20)
    second = colored_nlm(image, 20)
    assert first.dtype == np.uint8 and first.shape == image.shape
    np.testing.assert_array_equal(first, second)


def test_runner_contract_uses_exact_shared_selector_and_digests() -> None:
    config, primary = runner.load_contract(runner.MANIFEST, "primary")
    _, confirmation = runner.load_contract(runner.MANIFEST, "confirmation")
    assert runner.sha256_file(runner.CONFIG) == runner.CONFIG_SHA256
    assert len(primary) == len(confirmation) == 24
    assert primary[0]["filename"] == "img_002972.png"
    assert primary[-1]["filename"] == "img_002690.png"
    assert confirmation[0]["filename"] == "img_004344.png"
    assert confirmation[-1]["filename"] == "img_003835.png"
    assert {row["filename"] for row in primary}.isdisjoint(row["filename"] for row in confirmation)
    assert runner.names_digest(primary) == config["protocol"]["primary"]["filenames_newline_sha256"]
    assert (
        runner.roster_digest(primary)
        == config["protocol"]["primary"]["filename_input_roster_sha256"]
    )
    confirmation_config = config["protocol"]["confirmation_if_and_only_if_primary_passes"]
    assert runner.names_digest(confirmation) == confirmation_config["filenames_newline_sha256"]
    assert runner.roster_digest(confirmation) == confirmation_config["filename_input_roster_sha256"]


def test_executable_constants_agree_with_config_and_fail_on_arm_drift() -> None:
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    runner.validate_config_agreement(config)
    config["arms"][2]["aggressive_h"] = 36
    with pytest.raises(RuntimeError, match="arm roster"):
        runner.validate_config_agreement(config)


def test_commitment_reload_detects_artifact_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filename = "synthetic.png"
    record = {"filename": filename, "input_sha256": "a" * 64}
    image = _image()
    arrays = {
        "dirty": image,
        **{
            f"prediction__{name}": image.copy()
            for name in (
                runner.CONTROL,
                runner.REFERENCE,
                *(arm.name for arm in runner.PROTECTED_ARMS),
            )
        },
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
        "protected_arms": [arm.__dict__ for arm in runner.PROTECTED_ARMS],
        "per_board": [
            {
                "filename": filename,
                "input_sha256": record["input_sha256"],
                "prediction_sha256": prediction_hashes,
                "artifact": artifact,
            }
        ],
    }
    commitment["commitment_sha256"] = runner.canonical_digest(commitment)
    reloaded = runner.reload_committed_predictions(commitment, [record], tmp_path)
    assert len(reloaded) == 1

    os.chmod(artifact_path, 0o644)
    with artifact_path.open("ab") as stream:
        stream.write(b"mutation")
    with pytest.raises(RuntimeError, match="artifact file changed"):
        runner.reload_committed_predictions(commitment, [record], tmp_path)


def test_numeric_gate_check_names_match_immutable_config() -> None:
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    thresholds = config["primary_promotion_gate"]
    safety = {
        "mean_luminance_gradient_retention": 1.0,
        "minimum_luminance_gradient_retention": 1.0,
        "mean_chroma_gradient_retention": 1.0,
        "minimum_chroma_gradient_retention": 1.0,
        "mean_laplacian_retention": 1.0,
        "minimum_laplacian_retention": 1.0,
        "mean_grid_ratio_relative_to_baseline": 1.0,
        "maximum_grid_ratio_relative_to_baseline": 1.0,
        "mean_protected_pixel_fraction": 0.5,
        "minimum_protected_pixel_fraction": 0.4,
        "maximum_protected_pixel_fraction": 0.7,
        "maximum_clipped_fraction_increase": 0.0,
        "distinct_from_A_on_every_board": True,
    }
    result = runner.numeric_gate([0.28] * 24, [0.25] * 24, safety, thresholds)
    assert result["all_passed"] is True
    assert set(result["checks"]) == set(thresholds) - {"manual_severe_new_artifacts_allowed"}


def test_confirmation_fails_closed_without_report_and_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "PRIMARY_ROOT", tmp_path / "primary")
    with pytest.raises(RuntimeError, match="explicit manual review"):
        runner.authorized_winner()
