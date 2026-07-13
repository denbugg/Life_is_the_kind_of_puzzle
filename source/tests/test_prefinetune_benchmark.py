from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pytest

from puzzle_denoise_v2.prefinetune_benchmark import (
    CALIBRATION_SOURCE_COUNT,
    QUARANTINE_SOURCE_COUNT,
    SEALED_GATE_SOURCE_COUNT,
    PreFineTuneBenchmarkConfig,
    assess_prefinetune_diagnostic,
    classical_nlm_tiles_uint8,
    prefinetune_benchmark_code_fingerprint,
    select_frozen_calibration_sources,
    validate_prefinetune_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> PreFineTuneBenchmarkConfig:
    return PreFineTuneBenchmarkConfig(
        data_root="puzzle",
        manifest="manifest.json",
        val_pairs="val.npz",
        init_checkpoint="init.pt",
        legacy_checkpoint="legacy.pt",
        quarantine_artifact="quarantine.json",
        output="report.json",
        expected_manifest_sha256="a" * 64,
        expected_val_pairs_sha256="b" * 64,
        expected_init_checkpoint_sha256="c" * 64,
        expected_legacy_checkpoint_sha256="d" * 64,
        expected_quarantine_sha256="e" * 64,
        expected_validation_pixels_sha256="f" * 64,
        expected_code_sha256="0" * 64,
        expected_opencv_version="4.11.0",
        gate_source_count=350,
    )


def _diagnostic_fixture(
    *, candidate: float = 0.780, legacy: float = 0.785, legacy_lower: float | None = None
) -> tuple[dict, dict]:
    metrics = {}
    bootstraps = {}
    for panel in ("primary", "sensitivity"):
        baselines = {"raw": 0.65, "opencv_nlm": 0.72, "legacy_q90": legacy}
        metrics[panel] = {
            name: {"source_macro": {"tile_ssim": score}}
            for name, score in baselines.items()
        }
        metrics[panel]["synthetic_ema"] = {
            "source_macro": {"tile_ssim": candidate}
        }
        bootstraps[panel] = {}
        for name, score in baselines.items():
            delta = candidate - score
            lower = delta - 0.002
            if name == "legacy_q90" and legacy_lower is not None:
                lower = legacy_lower
            bootstraps[panel][f"candidate_minus_{name}"] = {
                "candidate_minus_baseline": delta,
                "lower": lower,
                "upper": delta + 0.002,
            }
    return metrics, bootstraps


def test_source_protocol_is_name_based_exact_93_257_350_and_order_independent() -> None:
    names = tuple(f"img_{index:06d}.png" for index in range(700))
    quarantine = tuple(sorted(names[:QUARANTINE_SOURCE_COUNT]))
    active = np.arange(700, dtype=np.int64)
    calibration_a, sealed_a = select_frozen_calibration_sources(
        names, active, active, quarantine, gate_source_count=350
    )
    reversed_names = tuple(reversed(names))
    calibration_b, sealed_b = select_frozen_calibration_sources(
        reversed_names,
        active,
        active,
        quarantine,
        gate_source_count=350,
    )

    calibration_names_a = {names[index] for index in calibration_a}
    calibration_names_b = {reversed_names[index] for index in calibration_b}
    sealed_names_a = {names[index] for index in sealed_a}
    sealed_names_b = {reversed_names[index] for index in sealed_b}
    assert len(calibration_a) == len(calibration_b) == CALIBRATION_SOURCE_COUNT == 257
    assert CALIBRATION_SOURCE_COUNT * 8 == 2056
    assert len(sealed_a) == len(sealed_b) == SEALED_GATE_SOURCE_COUNT == 350
    assert calibration_names_a == calibration_names_b
    assert sealed_names_a == sealed_names_b
    assert not np.intersect1d(calibration_a, sealed_a).size
    assert not calibration_names_a.intersection(quarantine)
    assert not sealed_names_a.intersection(quarantine)
    assert calibration_names_a | sealed_names_a | set(quarantine) == set(names)

    with pytest.raises(ValueError, match="cover every validation source"):
        select_frozen_calibration_sources(
            names, active[:-1], active, quarantine, gate_source_count=350
        )


def test_spending_diagnostic_allows_only_bounded_non_significant_legacy_gap() -> None:
    metrics, bootstraps = _diagnostic_fixture()
    result = assess_prefinetune_diagnostic(metrics, bootstraps)
    assert result["proceed_to_finetune"]
    assert result["decision_kind"] == "fine_tune_spending_and_headroom"
    assert result["not_a_model_promotion_decision"]
    assert not result["launches_training"]

    metrics, bootstraps = _diagnostic_fixture(candidate=0.770, legacy=0.785)
    too_far_behind = assess_prefinetune_diagnostic(metrics, bootstraps)
    assert not too_far_behind["proceed_to_finetune"]
    assert any("legacy_deficit" in name for name in too_far_behind["failed_checks"])

    metrics, bootstraps = _diagnostic_fixture(legacy_lower=-0.0101)
    statistically_behind = assess_prefinetune_diagnostic(metrics, bootstraps)
    assert not statistically_behind["proceed_to_finetune"]
    assert any("legacy_noninferiority" in name for name in statistically_behind["failed_checks"])


def test_config_and_code_fingerprint_are_strict_but_quick() -> None:
    validate_prefinetune_config(_config())
    with pytest.raises(ValueError, match="max_legacy_ssim_deficit"):
        validate_prefinetune_config(replace(_config(), max_legacy_ssim_deficit=0.021))
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        validate_prefinetune_config(replace(_config(), bootstrap_resamples=999))
    with pytest.raises(ValueError, match="gate_source_count"):
        validate_prefinetune_config(replace(_config(), gate_source_count=349))
    digest = prefinetune_benchmark_code_fingerprint()
    assert len(digest) == 64
    int(digest, 16)


def test_fixed_nlm_is_tile_isolated_uint8_and_validates_shape() -> None:
    tiles = np.zeros((1, 20, 20, 3), dtype=np.uint8)
    restored = classical_nlm_tiles_uint8(tiles)
    assert restored.dtype == np.uint8
    assert restored.shape == tiles.shape
    assert np.array_equal(restored, tiles)
    with pytest.raises(TypeError, match="uint8"):
        classical_nlm_tiles_uint8(tiles.astype(np.float32))


def test_kaggle_job_is_cpu_only_and_uses_only_required_sources() -> None:
    job_dir = ROOT / "kaggle_jobs" / "denoise_v2_prefinetune_cpu"
    metadata = json.loads((job_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] is False
    assert metadata["dataset_sources"] == [
        "pasha883/vsos-ai-initiative-pazzle",
        "rusyalain/vsos-denoise-v2-code",
        "rusyalain/vsos-denoise-v2-real-pairs",
        "rusyalain/vsos-denoise-legacy-baseline",
    ]
    assert metadata["kernel_sources"] == ["rusyalain/vsos-denoise-v2-synthetic-50k"]
    source = (job_dir / "evaluate.py").read_text(encoding="utf-8")
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = ""' in source
    assert "package_root = package_dir.parent" in source
    sentinel_match = re.search(
        r'EXPECTED_INIT_CHECKPOINT_SHA256\s*=\s*"([^"]+)"', source
    )
    assert sentinel_match is not None
    checkpoint_pin = sentinel_match.group(1)
    assert checkpoint_pin == "REPLACE_WITH_COMPLETED_SYNTH_CHECKPOINT_SHA256" or re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_pin
    )
    assert "run_prefinetune_benchmark" in source
    assert prefinetune_benchmark_code_fingerprint() in source
    assert "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215" in source
    assert "fe573ed28b74b45e8b0302ad51c53ff0f7ad5ad907aa3d4d9332c87010e42bd5" in source
    assert "torch==2.6.0" in source
    for version in (
        "3.12.13",
        "2.0.2",
        "11.3.0",
        "1.16.3",
        "0.25.2",
        "2.6.0+cpu",
        "0.8.3",
        "4.11.0",
    ):
        assert version in source
    assert "source_code_sha256" in (
        ROOT / "src" / "puzzle_denoise_v2" / "prefinetune_benchmark.py"
    ).read_text(encoding="utf-8")
    docs = (ROOT / "DENOISE_V2.md").read_text(encoding="utf-8")
    assert "Do not run the prepared CPU diagnostic yet" not in docs
    assert "257 clean calibration" in docs


def test_code_dataset_expanded_payload_checksums_match_staged_zip() -> None:
    staging = ROOT / "kaggle_datasets" / "denoise_v2_code"
    with zipfile.ZipFile(staging / "denoise_v2_code.zip") as archive:
        for line in (staging / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            expected, remote_path = line.split("  ", 1)
            if remote_path.startswith("denoise_v2_code/"):
                payload = archive.read(remote_path.removeprefix("denoise_v2_code/"))
            else:
                payload = (staging / remote_path).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == expected
