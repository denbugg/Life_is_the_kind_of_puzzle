from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from puzzle_denoise_v2.final_gate_audit import (
    FinalGateAuditConfig,
    assess_final_gate,
    final_gate_code_fingerprint,
    validate_final_gate_config,
)


def _config() -> FinalGateAuditConfig:
    return FinalGateAuditConfig(
        data_root="puzzle",
        manifest="manifest.json",
        val_pairs="val.npz",
        checkpoint="selected.pt",
        legacy_checkpoint="legacy.pt",
        quarantine_artifact="quarantine.json",
        selection_manifest="selected.json",
        output="gate.json",
        expected_manifest_sha256="a" * 64,
        expected_val_pairs_sha256="b" * 64,
        expected_checkpoint_sha256="c" * 64,
        expected_legacy_checkpoint_sha256="d" * 64,
        expected_quarantine_sha256="e" * 64,
        expected_selection_manifest_sha256="f" * 64,
        expected_code_sha256="0" * 64,
        expected_opencv_version="4.11.0",
    )


def _assessment_fixture(lower: float = 0.01) -> tuple[dict, dict]:
    metrics = {}
    bootstraps = {}
    for panel in ("primary", "sensitivity"):
        metrics[panel] = {
            "selected_ema": {"source_macro": {"tile_ssim": 0.80}},
            "raw": {"source_macro": {"tile_ssim": 0.65}},
            "opencv_nlm": {"source_macro": {"tile_ssim": 0.70}},
            "legacy_q90": {"source_macro": {"tile_ssim": 0.76}},
        }
        bootstraps[panel] = {}
        for name, value in (("raw", 0.65), ("opencv_nlm", 0.70), ("legacy_q90", 0.76)):
            delta = 0.80 - value
            bootstraps[panel][f"selected_minus_{name}"] = {
                "candidate_minus_baseline": delta,
                "lower": lower,
                "upper": delta + 0.01,
            }
    return metrics, bootstraps


def test_config_and_code_fingerprint_are_strict() -> None:
    validate_final_gate_config(_config())
    with pytest.raises(ValueError, match="exactly 5000"):
        validate_final_gate_config(replace(_config(), bootstrap_resamples=4999))
    digest = final_gate_code_fingerprint()
    assert len(digest) == 64
    int(digest, 16)


def test_final_gate_assessment_requires_every_positive_lower_bound() -> None:
    metrics, bootstraps = _assessment_fixture()
    result = assess_final_gate(metrics, bootstraps)
    assert result["passes_final_gate"]
    assert result["failed_checks"] == []
    bootstraps["sensitivity"]["selected_minus_legacy_q90"]["lower"] = -0.001
    result = assess_final_gate(metrics, bootstraps)
    assert not result["passes_final_gate"]
    assert result["failed_checks"] == [
        "sensitivity_beats_legacy_q90_bootstrap_lower_positive"
    ]


def test_final_gate_assessment_rejects_inconsistent_delta() -> None:
    metrics, bootstraps = _assessment_fixture()
    bootstraps["primary"]["selected_minus_raw"]["candidate_minus_baseline"] = 0.1
    with pytest.raises(ValueError, match="deltas disagree"):
        assess_final_gate(metrics, bootstraps)


def test_final_gate_kaggle_job_is_cpu_only_and_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    job = root / "kaggle_jobs" / "denoise_v2_final_gate_cpu"
    metadata = json.loads((job / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] is False
    assert metadata["kernel_sources"] == ["rusyalain/vsos-denoise-v2-synthetic-50k"]
    source = (job / "evaluate.py").read_text(encoding="utf-8")
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = ""' in source
    assert final_gate_code_fingerprint() in source
    assert "ce244ce8c9759be859262fd16560f8318814022883ec52cdc380ad490a924080" in source
    selection = root / "kaggle_datasets" / "denoise_v2_code" / "selected_model.json"
    assert hashlib.sha256(selection.read_bytes()).hexdigest() == (
        "ce244ce8c9759be859262fd16560f8318814022883ec52cdc380ad490a924080"
    )
