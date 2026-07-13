from __future__ import annotations

import inspect
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest
import torch

from puzzle_assembly.contextual_refiner import (
    CONTEXT_FEATURE_CHANNELS,
    ContextualResidualNAF,
    bilateral_tile_consensus_residual,
    broadcast_tile_grid,
    build_context_features,
    internal_seam_mask,
    maximum_rgb_change,
    model_parameter_count,
    tile_mean_grid,
)
from puzzle_assembly.contextual_refiner_training import ContextualRefinerLoss
from puzzle_denoise_v2.tiles import GRID, IMAGE_SIZE, TILE
from puzzle_assembly.protocol import source_names_for_split


ROOT = Path(__file__).resolve().parents[1]


def _image(batch: int = 1, value: float = 0.5) -> torch.Tensor:
    return torch.full((batch, 3, IMAGE_SIZE, IMAGE_SIZE), value)


def _confidence(batch: int = 1, value: float = 1.0) -> torch.Tensor:
    return torch.full((batch, 1, GRID, GRID), value)


def test_tile_mean_grid_and_broadcast_are_exact() -> None:
    grid = torch.arange(GRID * GRID, dtype=torch.float32).reshape(1, 1, GRID, GRID)
    image = broadcast_tile_grid(grid).repeat(1, 3, 1, 1) / float(GRID * GRID)
    recovered = tile_mean_grid(image)
    assert recovered.shape == (1, 3, GRID, GRID)
    assert torch.allclose(
        recovered[:, :1], grid / float(GRID * GRID), rtol=0.0, atol=1e-6
    )


def test_bilateral_5x5_prior_preserves_a_strong_semantic_edge() -> None:
    image = _image(value=0.1)
    image[:, :, :, IMAGE_SIZE // 2 :] = 0.9
    residual = bilateral_tile_consensus_residual(
        image,
        radius=2,
        sigma_spatial=1.5,
        sigma_colour=0.05,
    )
    assert residual.shape == (1, 3, GRID, GRID)
    # A plain 5x5 mean would mix the two halves near the boundary.  The range
    # kernel should make that leakage numerically negligible.
    assert float(residual.abs().max()) < 1e-5


def test_internal_seam_mask_excludes_frame_and_marks_both_sides() -> None:
    mask = internal_seam_mask(device="cpu", dtype=torch.float32, band=2)
    assert mask.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert mask[0, 0, 0, 0] == 0
    assert mask[0, 0, TILE - 2, 100] == 1
    assert mask[0, 0, TILE + 1, 100] == 1
    assert mask[0, 0, TILE + 2, TILE + 2] == 0


def test_context_features_are_target_blind_and_confidence_gated() -> None:
    harmonized = _image(batch=2, value=0.5)
    preanalytic = _image(batch=2, value=0.45)
    seam_confidence = _confidence(batch=2, value=0.8)
    layout_confidence = _confidence(batch=2, value=0.25)
    features, gate, seam = build_context_features(
        harmonized,
        preanalytic,
        seam_confidence,
        layout_confidence,
    )
    assert features.shape == (2, CONTEXT_FEATURE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    assert gate.shape == seam.shape == (2, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.allclose(gate, torch.full_like(gate, 0.2))
    assert not ({"target", "clean", "source", "source_name"} & set(
        inspect.signature(build_context_features).parameters
    ))


def test_zero_initialization_is_exact_identity_and_zero_gate_stays_identity() -> None:
    model = ContextualResidualNAF(width=8, blocks=2)
    harmonized = _image(value=0.5)
    features, gate, seam = build_context_features(
        harmonized,
        _image(value=0.45),
        _confidence(),
        _confidence(),
    )
    with torch.no_grad():
        prediction = model(harmonized, features, gate, seam)
    assert torch.equal(prediction, harmonized)

    with torch.no_grad():
        model.base_tail.bias.fill_(10.0)
        model.seam_tail.bias.fill_(10.0)
        prediction = model(harmonized, features, torch.zeros_like(gate), seam)
    assert torch.equal(prediction, harmonized)


def test_refiner_has_a_hard_rgb_change_bound() -> None:
    model = ContextualResidualNAF(width=8, blocks=2)
    harmonized = _image(value=0.5)
    features, gate, seam = build_context_features(
        harmonized,
        harmonized,
        _confidence(),
        _confidence(),
    )
    with torch.no_grad():
        model.base_tail.bias.fill_(50.0)
        model.seam_tail.bias.fill_(-50.0)
        prediction = model(harmonized, features, gate, seam)
    assert float((prediction - harmonized).abs().max()) <= maximum_rgb_change(model) + 1e-7
    assert maximum_rgb_change(model) == pytest.approx(8.0 / 255.0)
    assert model_parameter_count(model) < 100_000


def test_invalid_confidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="lie in"):
        build_context_features(
            _image(),
            _image(),
            _confidence(value=1.1),
            _confidence(),
        )


def test_contextual_loss_rewards_the_clean_target_and_reports_preservation_terms() -> None:
    target = _image(value=0.5)
    target[:, :, 100:300, 100:300] = 0.7
    worse = (target + 0.05).clamp(0, 1)
    seam = internal_seam_mask(device="cpu", dtype=torch.float32, band=2)
    loss_fn = ContextualRefinerLoss()
    exact_loss, exact_terms = loss_fn(target, target, worse, seam)
    worse_loss, worse_terms = loss_fn(worse, target, worse, seam)
    assert float(exact_loss) < float(worse_loss)
    assert float(exact_terms["ssim"]) == pytest.approx(1.0, abs=1e-6)
    assert set(worse_terms) == {
        "total",
        "pixel",
        "ssim",
        "seam",
        "gradient",
        "texture",
        "residual",
    }


def test_pilot_protocol_is_whole_source_disjoint_and_evidence_pinned() -> None:
    path = ROOT / "configs/postassembly_contextual_refiner_v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    manifest = ROOT / config["authoritative_inputs"]["manifest"]["path"]
    quarantine = ROOT / config["authoritative_inputs"]["quarantine"]["path"]
    audit_exclusion = ROOT / config["authoritative_inputs"]["audit_exclusion"]["path"]
    train = source_names_for_split(
        "edge_train",
        manifest_path=manifest,
        quarantine_path=quarantine,
        audit_exclusion_path=audit_exclusion,
    )[:512]
    validation = source_names_for_split(
        "assembly_cal",
        manifest_path=manifest,
        quarantine_path=quarantine,
        audit_exclusion_path=audit_exclusion,
    )[:32]
    actual_report_record = config["evidence_gate"]["actual_layout_report"]
    actual_report = ROOT / actual_report_record["path"]
    actual_names = json.loads(actual_report.read_text(encoding="utf-8"))["source_names"]
    assert len(train) == len(set(train)) == 512
    assert len(validation) == len(set(validation)) == 32
    assert len(actual_names) == len(set(actual_names)) == 32
    assert not set(train) & set(validation)
    assert not set(train) & set(actual_names)
    assert not set(validation) & set(actual_names)
    assert hashlib.sha256(actual_report.read_bytes()).hexdigest() == actual_report_record[
        "sha256"
    ]
    assert config["scope"]["submission_promotion_allowed"] is False
    assert config["source_protocol"]["frozen_actual_qap_gate"][
        "score_each_checkpoint_at_most_once"
    ] is True


def _load_contextual_kaggle_job():
    path = ROOT / "kaggle_jobs/assembly_v1_contextual_refiner_gpu/train.py"
    spec = importlib.util.spec_from_file_location("contextual_refiner_kaggle_job", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kaggle_job_is_t4_pinned_and_uses_only_scoped_sources() -> None:
    metadata = json.loads(
        (
            ROOT
            / "kaggle_jobs/assembly_v1_contextual_refiner_gpu/kernel-metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == [
        "pasha883/vsos-contextual-refiner-code",
        "pasha883/vsos-ai-initiative-pazzle",
        "pasha883/vsos-assembly-v1-runtime",
        "pasha883/vsos-contextual-refiner-frozen-qap-gate",
    ]
    assert metadata["kernel_sources"] == [
        "pasha883/vsos-assembly-v1-seam-denoiser-gpu"
    ]
    assert not any("oracle" in value for value in metadata["dataset_sources"])


def test_kaggle_smoke_gate_requires_every_preservation_check() -> None:
    job = _load_contextual_kaggle_job()

    def comparison() -> dict:
        return {
            "mean_ssim_delta": 0.006,
            "paired_bootstrap_95_ci": [0.003, 0.009],
            "mean_boundary_band_mae_delta": -0.1,
            "mean_target_referenced_seam_error_delta": -0.001,
            "texture_gradient_mae_ratio": 1.0,
            "large_regressions_below_minus_0_01": 0,
            "candidate_advantage_over_context_placebo": 0.002,
            "zero_confidence_byte_identity_all": True,
            "face_roi_count": 0,
            "face_roi_mean_ssim_delta": None,
        }

    correct = {"comparisons": {panel: comparison() for panel in job.PANELS}}
    actual = {
        "comparisons": {panel: comparison() for panel in job.PANELS},
        "all_layouts_unchanged": True,
    }
    assert job._gate(correct, actual)["passed"] is True
    actual["comparisons"]["primary_kornia"][
        "zero_confidence_byte_identity_all"
    ] = False
    failed = job._gate(correct, actual)
    assert failed["passed"] is False
    assert failed["continuation_to_10000_allowed"] is False
