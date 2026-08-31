from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
from pathlib import Path
from typing import Any

from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    select_manifest_records,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_RUNNER = PROJECT_ROOT / "scripts/run_pretrained_drunet_protected_stack_all700.py"
V2_RUNNER = PROJECT_ROOT / "scripts/run_pretrained_drunet_protected_stack_all700_v2.py"
CONFIG = PROJECT_ROOT / "configs/pretrained_drunet_protected_stack_all700_measurement_v2.json"
SIDECAR = Path(f"{CONFIG}.sha256")
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def names_digest(records: tuple[dict[str, Any], ...]) -> str:
    return hashlib.sha256("\n".join(record["filename"] for record in records).encode()).hexdigest()


def input_digest(records: tuple[dict[str, Any], ...]) -> str:
    return hashlib.sha256(
        "\n".join(f"{record['filename']} {record['input_sha256']}" for record in records).encode()
    ).hexdigest()


def test_v2_config_is_readonly_hash_bound_and_covers_both_full_splits() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(CONFIG)
    assert SIDECAR.read_text(encoding="utf-8").split()[0] == config_sha256
    assert not CONFIG.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert not SIDECAR.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for split in ("calibration", "holdout"):
        records = select_manifest_records(
            manifest,
            split,
            limit=700,
            seed=EXPERIMENT_SUBSET_SEED,
            namespace=EXPERIMENT_SUBSET_NAMESPACE,
        )
        assert len(records) == 700
        assert names_digest(records) == config["data"][split]["filenames_sha256"]
        assert input_digest(records) == config["data"][split]["input_roster_sha256"]
        assert config["data"][split]["historical_target_exposure"]["freshness_claim"] is False


def test_v2_source_roster_and_json_audit_fix_are_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for relative, expected in config["source_sha256"].items():
        assert sha256_file(PROJECT_ROOT / relative) == expected
    wrapper = load_module(V2_RUNNER, "all700_v2_test")
    value = {"missing": (), "duplicates": (1, 2), "passed": True}
    assert wrapper._json_compatible(value) == {
        "missing": [],
        "duplicates": [1, 2],
        "passed": True,
    }


def test_v1_failed_before_target_access_and_v2_rule_is_fail_closed() -> None:
    v1_root = (
        PROJECT_ROOT
        / "outputs/pretrained-drunet-protected-stack/all700-measurement-v1/calibration700"
    )
    assert not (v1_root / "prediction-commitment.json").exists()
    assert not (v1_root / "commitment-receipt.json").exists()
    assert not (v1_root / "targets-opened-receipt.json").exists()
    assert not (v1_root / "report.json").exists()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rule = config["broad_completion_rule_fixed_before_calibration_score"]
    assert rule["calibration700_mean_ssim_closed_interval"] == [0.27, 0.28]
    assert rule["all_700_strict_provenance_audits_must_pass"] is True
    assert rule["preexisting_target_free_safety_gate_must_pass"] is True
    assert rule["holdout_no_method_parameter_or_threshold_change"] is True


def test_frozen_safety_gate_boundaries_and_quantile_labels() -> None:
    runner = load_module(V1_RUNNER, "all700_v1_test")
    summary = {
        "mean_luminance_gradient_retention": 0.90,
        "minimum_luminance_gradient_retention": 0.80,
        "mean_chroma_gradient_retention": 0.80,
        "minimum_chroma_gradient_retention": 0.60,
        "mean_laplacian_retention": 0.90,
        "minimum_laplacian_retention": 0.80,
        "mean_grid_ratio_relative_to_baseline": 1.08,
        "maximum_grid_ratio_relative_to_baseline": 1.15,
        "protected_fraction_mean_min_max": [0.40, 0.30, 0.85],
        "maximum_rgb_mean_shift": 3.0,
        "global_rgb_std_ratio_mean_min_max": [0.97, 0.90, 1.05],
        "mean_abs_pixel_change_mean_max": [4.0, 8.0],
        "maximum_clipping_increase": 0.01,
    }
    assert runner.safety_gate(summary, True)["passed"] is True
    assert runner.safety_gate(summary, False)["passed"] is False
    assert list(runner.quantile_map(runner.np.arange(10.0))) == [
        "q000",
        "q005",
        "q010",
        "q025",
        "q050",
        "q075",
        "q090",
        "q095",
        "q100",
    ]
