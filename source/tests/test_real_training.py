from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from puzzle_denoise_v2.real_training import (
    FineTuneConfig,
    assess_candidate,
    deterministic_contamination_aware_split,
    deterministic_source_split,
    fine_tune_code_fingerprint,
    fine_tune_pixel_fingerprints,
    is_real_batch_step,
    learning_rate_scale,
    load_validation_quarantine,
    paired_dihedral,
    source_name_list_sha256,
    validate_fine_tune_config,
)


QUARANTINE_SHA256 = "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215"
LEGACY_SHA256 = "d1df5a4e4852c821d79f72063866cf1fe09fb1beff913a4fb1034466d6ead96e"


def _config() -> FineTuneConfig:
    return FineTuneConfig(
        data_root="puzzle",
        manifest="manifest.json",
        train_pairs="train.npz",
        val_pairs="val.npz",
        init_checkpoint="init.pt",
        legacy_checkpoint="legacy.pt",
        quarantine_artifact="quarantine.json",
        output="out.pt",
        expected_manifest_sha256="a" * 64,
        expected_train_pairs_sha256="b" * 64,
        expected_val_pairs_sha256="c" * 64,
        expected_init_checkpoint_sha256="d" * 64,
        expected_legacy_checkpoint_sha256="e" * 64,
        expected_quarantine_sha256="1" * 64,
        expected_training_pixels_sha256="f" * 64,
        expected_validation_pixels_sha256="0" * 64,
        expected_opencv_version="5.0.0",
        gate_source_count=350,
        steps=1000,
        warmup_steps=100,
    )


def test_real_batch_schedule_and_cosine_warmup_are_explicit() -> None:
    config = _config()
    assert not is_real_batch_step(1, config)
    assert is_real_batch_step(8, config)
    assert not is_real_batch_step(500, config)
    assert is_real_batch_step(504, config)
    assert learning_rate_scale(1, config) == pytest.approx(0.01)
    assert learning_rate_scale(100, config) == pytest.approx(1.0)
    assert learning_rate_scale(1000, config) == pytest.approx(config.min_lr_ratio)


def test_paired_dihedral_applies_identical_transform_to_both_tiles() -> None:
    corrupt = torch.arange(8 * 3 * 20 * 20, dtype=torch.float32).reshape(8, 3, 20, 20)
    clean = corrupt + 7.0
    transformed_corrupt, transformed_clean = paired_dihedral(
        corrupt,
        clean,
        np.random.default_rng(31),
    )
    assert torch.equal(transformed_clean - transformed_corrupt, torch.full_like(corrupt, 7.0))
    assert sorted(transformed_corrupt.flatten().tolist()) == sorted(corrupt.flatten().tolist())


def test_promotion_gate_requires_real_gain_and_synthetic_safety() -> None:
    synthetic = {
        "tile_ssim": 0.80,
        "psnr": 24.0,
        "boundary_mae": 12.0,
        "signed_bias_r": 0.2,
        "signed_bias_g": -0.3,
        "signed_bias_b": 0.1,
    }
    primary = {"tile_ssim": 0.77}
    sensitivity = {"tile_ssim": 0.75}
    raw_primary = {"tile_ssim": 0.70}
    raw_sensitivity = {"tile_ssim": 0.69}
    classical_primary = {"tile_ssim": 0.71}
    classical_sensitivity = {"tile_ssim": 0.70}
    legacy_primary = {"tile_ssim": 0.72}
    legacy_sensitivity = {"tile_ssim": 0.71}
    eligible = assess_candidate(
        synthetic,
        {**synthetic, "tile_ssim": 0.799},
        primary,
        {"tile_ssim": 0.774},
        sensitivity,
        {"tile_ssim": 0.752},
        raw_primary,
        raw_sensitivity,
        classical_primary,
        classical_sensitivity,
        legacy_primary,
        legacy_sensitivity,
        bootstrap_lower=0.001,
        raw_bootstrap_lower=0.01,
        classical_bootstrap_lower=0.008,
        legacy_bootstrap_lower=0.005,
    )
    assert eligible["eligible"]
    assert eligible["synthetic_safe"]

    unsafe = assess_candidate(
        synthetic,
        {**synthetic, "tile_ssim": 0.79},
        primary,
        {"tile_ssim": 0.774},
        sensitivity,
        {"tile_ssim": 0.752},
        raw_primary,
        raw_sensitivity,
        classical_primary,
        classical_sensitivity,
        legacy_primary,
        legacy_sensitivity,
        bootstrap_lower=0.001,
        raw_bootstrap_lower=0.01,
        classical_bootstrap_lower=0.008,
        legacy_bootstrap_lower=0.005,
    )
    assert not unsafe["eligible"]
    assert not unsafe["synthetic_safe"]


def test_fine_tune_config_validation_and_fingerprint() -> None:
    config = _config()
    validate_fine_tune_config(config)
    with pytest.raises(ValueError, match="warmup"):
        validate_fine_tune_config(replace(config, warmup_steps=1001))
    with pytest.raises(ValueError, match="confidence"):
        validate_fine_tune_config(replace(config, val_sensitivity_confidence=2.0))
    with pytest.raises(ValueError, match="expected_manifest_sha256"):
        validate_fine_tune_config(replace(config, expected_manifest_sha256="not-a-sha"))
    with pytest.raises(ValueError, match="expected_opencv_version"):
        validate_fine_tune_config(replace(config, expected_opencv_version="latest"))
    with pytest.raises(ValueError, match="gate_source_count"):
        validate_fine_tune_config(replace(config, gate_source_count=349))
    fingerprint = fine_tune_code_fingerprint()
    assert len(fingerprint) == 64
    int(fingerprint, 16)


def test_promotion_gate_rejects_candidate_below_raw_baseline() -> None:
    synthetic = {
        "tile_ssim": 0.80,
        "psnr": 24.0,
        "boundary_mae": 12.0,
        "signed_bias_r": 0.2,
        "signed_bias_g": -0.3,
        "signed_bias_b": 0.1,
    }
    result = assess_candidate(
        synthetic,
        synthetic,
        {"tile_ssim": 0.70},
        {"tile_ssim": 0.704},
        {"tile_ssim": 0.69},
        {"tile_ssim": 0.692},
        {"tile_ssim": 0.71},
        {"tile_ssim": 0.70},
        {"tile_ssim": 0.715},
        {"tile_ssim": 0.705},
        {"tile_ssim": 0.72},
        {"tile_ssim": 0.71},
        bootstrap_lower=0.001,
        raw_bootstrap_lower=-0.001,
        classical_bootstrap_lower=-0.0015,
        legacy_bootstrap_lower=-0.002,
    )
    assert not result["eligible"]
    assert not result["checks"]["real_primary_not_worse_than_raw"]
    assert not result["checks"]["real_primary_raw_bootstrap_lower_positive"]
    assert not result["checks"]["real_primary_not_worse_than_legacy"]
    assert not result["checks"]["real_primary_legacy_bootstrap_lower_positive"]


def test_promotion_gate_rejects_candidate_below_legacy_baseline() -> None:
    synthetic = {
        "tile_ssim": 0.80,
        "psnr": 24.0,
        "boundary_mae": 12.0,
        "signed_bias_r": 0.2,
        "signed_bias_g": -0.3,
        "signed_bias_b": 0.1,
    }
    result = assess_candidate(
        synthetic,
        synthetic,
        {"tile_ssim": 0.70},
        {"tile_ssim": 0.704},
        {"tile_ssim": 0.69},
        {"tile_ssim": 0.692},
        {"tile_ssim": 0.68},
        {"tile_ssim": 0.67},
        {"tile_ssim": 0.69},
        {"tile_ssim": 0.68},
        {"tile_ssim": 0.71},
        {"tile_ssim": 0.70},
        bootstrap_lower=0.001,
        raw_bootstrap_lower=0.01,
        classical_bootstrap_lower=0.005,
        legacy_bootstrap_lower=-0.001,
    )
    assert not result["eligible"]
    assert result["checks"]["real_primary_not_worse_than_raw"]
    assert result["checks"]["real_primary_raw_bootstrap_lower_positive"]
    assert not result["checks"]["real_primary_not_worse_than_legacy"]
    assert not result["checks"]["real_primary_legacy_bootstrap_lower_positive"]


def test_deterministic_source_split_is_disjoint_stable_and_order_independent() -> None:
    names = tuple(f"img_{index:06d}.png" for index in range(10))
    forward = np.arange(10, dtype=np.int64)
    reverse = forward[::-1].copy()
    cal_a, gate_a = deterministic_source_split(names, forward, 0.5, seed=77)
    cal_b, gate_b = deterministic_source_split(names, reverse, 0.5, seed=77)
    assert np.array_equal(cal_a, cal_b)
    assert np.array_equal(gate_a, gate_b)
    assert len(cal_a) == len(gate_a) == 5
    assert set(cal_a).isdisjoint(set(gate_a))
    assert sorted(np.concatenate([cal_a, gate_a]).tolist()) == list(range(10))


def test_quarantine_loader_and_contamination_aware_split_are_pinned(tmp_path) -> None:
    repo = Path(__file__).resolve().parents[1]
    artifact_path = repo / "configs" / "denoise_validation_quarantine_v1.json"
    manifest_path = repo / "configs" / "denoise_splits_seed20260710.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    artifact, actual_sha256 = load_validation_quarantine(
        artifact_path,
        QUARANTINE_SHA256,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_validation_names=manifest["splits"]["val"],
        expected_legacy_checkpoint_sha256=LEGACY_SHA256,
        expected_synthetic_validation_names=manifest["splits"]["val"][:24],
        gate_source_count=350,
        seed=20260710,
    )
    assert actual_sha256 == QUARANTINE_SHA256
    assert artifact["counts"] == {
        "calibration": 257,
        "eligible_after_quarantine": 607,
        "frozen_gate": 350,
        "legacy_train_seen": 87,
        "legacy_validation_seen": 6,
        "quarantine": 93,
        "synthetic_validation_seen": 24,
    }

    # Reverse the pair-table order to prove that exclusion and assignment map by name.
    source_names = tuple(reversed(manifest["splits"]["val"]))
    calibration, gate = deterministic_contamination_aware_split(
        source_names,
        np.arange(len(source_names), dtype=np.int64),
        tuple(artifact["quarantine_names"]),
        350,
        20260710,
    )
    calibration_names = sorted(source_names[int(index)] for index in calibration)
    gate_names = sorted(source_names[int(index)] for index in gate)
    quarantine_names = set(artifact["quarantine_names"])
    assert len(calibration_names) == 257
    assert len(gate_names) == 350
    assert not quarantine_names.intersection(calibration_names)
    assert not quarantine_names.intersection(gate_names)
    assert source_name_list_sha256(calibration_names) == artifact["name_sha256"]["calibration"]
    assert source_name_list_sha256(gate_names) == artifact["name_sha256"]["frozen_gate"]

    tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
    tampered["unexpected"] = True
    tampered_path = tmp_path / "tampered_quarantine.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    tampered_sha256 = hashlib.sha256(tampered_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="top-level schema"):
        load_validation_quarantine(
            tampered_path,
            tampered_sha256,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            manifest_validation_names=manifest["splits"]["val"],
            expected_legacy_checkpoint_sha256=LEGACY_SHA256,
            expected_synthetic_validation_names=manifest["splits"]["val"][:24],
            gate_source_count=350,
            seed=20260710,
        )


def test_pixel_fingerprints_pin_decoded_training_and_validation_data(tmp_path) -> None:
    inputs = tmp_path / "train" / "inputs"
    targets = tmp_path / "train" / "targets"
    inputs.mkdir(parents=True)
    targets.mkdir(parents=True)
    synthetic_name = "img_000001.png"
    train_name = "img_000002.png"
    validation_name = "img_000003.png"
    zeros = np.zeros((480, 480, 3), dtype=np.uint8)
    Image.fromarray(zeros).save(targets / synthetic_name)
    for name, value in ((train_name, 10), (validation_name, 20)):
        Image.fromarray(np.full_like(zeros, value)).save(inputs / name)
        Image.fromarray(np.full_like(zeros, value + 1)).save(targets / name)

    first = fine_tune_pixel_fingerprints(
        tmp_path,
        [synthetic_name],
        (train_name,),
        (validation_name,),
    )
    assert first == fine_tune_pixel_fingerprints(
        tmp_path,
        [synthetic_name],
        (train_name,),
        (validation_name,),
    )

    changed = np.full_like(zeros, 20)
    changed[0, 0, 0] = 21
    Image.fromarray(changed).save(inputs / validation_name)
    second = fine_tune_pixel_fingerprints(
        tmp_path,
        [synthetic_name],
        (train_name,),
        (validation_name,),
    )
    assert second["training_pixels_sha256"] == first["training_pixels_sha256"]
    assert second["validation_pixels_sha256"] != first["validation_pixels_sha256"]


def test_real_finetune_kaggle_job_is_pinned_and_self_verifies_output() -> None:
    root = Path(__file__).resolve().parents[1]
    job_dir = root / "kaggle_jobs" / "denoise_v2_real_finetune"
    metadata = json.loads((job_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] is True
    assert metadata["kernel_sources"] == ["rusyalain/vsos-denoise-v2-synthetic-50k"]
    source = (job_dir / "train.py").read_text(encoding="utf-8")
    for pin in (
        "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
        fine_tune_code_fingerprint(),
        "2.6.0+cu124",
        '"cuda": "12.4"',
        '"jpeg_codec": "6.2"',
        '"libjpeg_turbo_version": "3.1.1"',
        '"promotion_status": "promoted"',
        '"promotion_status": "rollback_safe"',
        'gate_validation.get("selected_step")',
        'assessment.get("eligible") is not True',
    ):
        assert pin in source
