from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

import aiijc_puzzle.compliant_fixed_b_standard_submission as production
import aiijc_puzzle.compliant_fixed_b_standard_validation as independent
from aiijc_puzzle.compliant_submission import InputSnapshot, array_sha256, atomic_write_json
from aiijc_puzzle.edge_protected_nlm import protected_masks
from aiijc_puzzle.legacy_upgrade import deterministic_submission_zip, png_bytes
from aiijc_puzzle.pretrained_tile_denoiser import render_drunet_tiles
from aiijc_puzzle.protocol import IMAGE_SIZE, TILE_COUNT, assemble_tiles, split_tiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def zeros_board() -> np.ndarray:
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


def pixel_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@dataclass
class FakeSolve:
    layout: np.ndarray
    runtime_seconds: float = 0.0


def fake_drunet_diagnostics() -> dict[str, int | float]:
    return {
        "tile_count": 576,
        "sigma_255": 50.0,
        "batch_size": 144,
        "padding_bottom": 4,
        "padding_right": 4,
        "runtime_seconds": 0.0,
        "mean_abs_change": 0.0,
        "q99_abs_change": 0.0,
        "maximum_abs_change": 0,
        "clipped_fraction": 1.0,
    }


def fake_prediction() -> production.FixedBSubmissionPrediction:
    board = zeros_board()
    binary = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    soft = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    audit = {
        "grid_rows": 24,
        "grid_columns": 24,
        "tile_count": 576,
        "unique_tile_indices": 576,
        "missing_tile_indices": [],
        "duplicate_tile_indices": [],
        "exact_reassembly_from_declared_layout": True,
        "input_output_tile_multiset_equal": True,
        "raw_input_pixels_preserved": True,
        "restoration_applied_after_audit": True,
        "passed": True,
    }
    return production.FixedBSubmissionPrediction(
        layout=np.arange(TILE_COUNT, dtype=np.int32),
        raw=board,
        harmonized=board,
        restored=board,
        audit=audit,
        tile_multiset_sha256=production.tile_multiset_sha256(board),
        restoration={
            "harmonized_array_sha256": array_sha256(board),
            "safety_reference_h28_array_sha256": array_sha256(board),
            "drunet50_canvas_pixel_sha256": pixel_sha256(board),
            "nlm_h20_pixel_sha256": pixel_sha256(board),
            "nlm_h28_pixel_sha256": pixel_sha256(board),
            "nlm_h50_pixel_sha256": pixel_sha256(board),
            "binary_mask_array_sha256": independent._typed_array_digest(binary),
            "soft_mask_array_sha256": independent._typed_array_digest(soft),
            "protected_fraction": 0.0,
            "drunet_diagnostics": fake_drunet_diagnostics(),
            "output_array_sha256": array_sha256(board),
        },
        score_seconds=0.0,
        solve_seconds=0.0,
        restoration_seconds=0.0,
    )


def fake_promotion() -> production.PromotionEvidence:
    artifacts: dict[str, dict[str, str]] = {}
    for index, name in enumerate(production.EVIDENCE_NAMES):
        record = {
            "path": (
                production.MEASUREMENT_CONFIG_PATH
                if name == "measurement_config"
                else (
                    production.RUNTIME_PREFLIGHT_RELATIVE_PATH
                    if name == "production_runtime_manifest"
                    else f"outputs/future/{name}.json"
                )
            ),
            "sha256": f"{index + 2:064x}",
        }
        if name == "production_runtime_manifest":
            record["digest_sha256"] = "f" * 64
        artifacts[name] = record
    return production.PromotionEvidence(
        config_sha256="1" * 64,
        artifacts=artifacts,
    )


def fake_runtime_manifest() -> dict[str, object]:
    content: dict[str, object] = {
        "files": {"runtime.py": "a" * 64},
        "assets": {f"asset-{index}": f"{index + 1:064x}" for index in range(4)},
        "harmonizers": {"rgb": "b" * 64, "luma": "c" * 64},
        "versions": {
            "python": "3.11",
            "numpy": "2",
            "opencv": "4",
            "torch": "2",
            "scipy": "1",
            "pillow": "12",
            "scikit_image": "0.26",
            "scikit_learn": "1.9",
            "jsonschema": "4",
        },
        "host": {
            "platform": "macOS-test-arm64",
            "mac_ver": ["test", ["", "", ""], "arm64"],
            "machine": "arm64",
            "mps_available": True,
        },
        "canonical_device": "mps",
    }
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**content, "digest_sha256": digest}


def write_readonly_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    path.chmod(0o444)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_safety() -> dict[str, object]:
    return {
        "checks": {
            name: True for name in sorted(production.EXPECTED_BROAD_SAFETY_CHECK_NAMES)
        },
        "passed": True,
        "summary": {"synthetic_fixture": True},
    }


def synthetic_committed_boards() -> list[dict[str, object]]:
    return [
        {
            "filename": f"img_{index:06d}.png",
            "layout_sha256": f"{index + 1:064x}",
            "candidate_pixel_sha256": f"{index + 701:064x}",
            "raw_permutation_audit_passed": True,
        }
        for index in range(700)
    ]


def promotion_fixture(
    tmp_path: Path,
    *,
    candidate: str = production.PROMOTED_ARM,
    safety_reference: str = production.SAFETY_REFERENCE,
    measurement_source_sha256: dict[str, str] | None = None,
    measurement_asset_sha256: dict[str, str] | None = None,
    commitment_source_sha256: dict[str, str] | None = None,
    commitment_asset_sha256: dict[str, str] | None = None,
) -> Path:
    measurement_sources = (
        dict(production.EXPECTED_BROAD_SOURCE_SHA256)
        if measurement_source_sha256 is None
        else dict(measurement_source_sha256)
    )
    measurement_assets = (
        dict(production.EXPECTED_BROAD_ASSET_SHA256)
        if measurement_asset_sha256 is None
        else dict(measurement_asset_sha256)
    )
    measurement = json.loads(
        (PROJECT_ROOT / production.MEASUREMENT_CONFIG_PATH).read_text(encoding="utf-8")
    )
    measurement["source_sha256"] = measurement_sources
    measurement["asset_sha256"] = measurement_assets
    paths = {
        name: (
            Path(production.MEASUREMENT_CONFIG_PATH)
            if name == "measurement_config"
            else (
                Path(production.RUNTIME_PREFLIGHT_RELATIVE_PATH)
                if name == "production_runtime_manifest"
                else Path(f"outputs/fixed-b-future/{name}.json")
            )
        )
        for name in production.EVIDENCE_NAMES
    }
    measurement_path = tmp_path / paths["measurement_config"]
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    if measurement_source_sha256 is None and measurement_asset_sha256 is None:
        shutil.copy2(PROJECT_ROOT / production.MEASUREMENT_CONFIG_PATH, measurement_path)
        measurement_path.chmod(0o444)
        measurement_hash = hashlib.sha256(measurement_path.read_bytes()).hexdigest()
    else:
        measurement_hash = write_readonly_json(measurement_path, measurement)
    runtime_manifest = production.build_runtime_manifest()
    runtime_hash = write_readonly_json(
        tmp_path / paths["production_runtime_manifest"], runtime_manifest
    )
    hashes: dict[str, str] = {
        "measurement_config": measurement_hash,
        "production_runtime_manifest": runtime_hash,
    }
    measurement_sha = hashes["measurement_config"]
    fixed_source = (
        dict(production.EXPECTED_BROAD_SOURCE_SHA256)
        if commitment_source_sha256 is None
        else dict(commitment_source_sha256)
    )
    fixed_assets = (
        dict(production.EXPECTED_BROAD_ASSET_SHA256)
        if commitment_asset_sha256 is None
        else dict(commitment_asset_sha256)
    )
    fixed_model = {
        "sigma_255": production.DRUNET_SIGMA_255,
        "batch_size": production.DRUNET_BATCH_SIZE,
        "parameter_count": production.MODEL_PARAMETER_COUNT,
    }
    committed_boards = synthetic_committed_boards()
    candidate_roster_sha256 = "c" * 64
    safety = synthetic_safety()
    for stage in ("calibration", "holdout"):
        commitment_name = f"{stage}_commitment"
        commitment = {
            "schema": "aiijc-drunet-sigma50-protected-all700-commitment-v1",
            "stage": stage,
            "count": 700,
            "config_sha256": measurement_sha,
            "fixed_candidate": candidate,
            "target_free_safety_reference_only": safety_reference,
            "all_700_strict_raw_permutation_audits_pass": True,
            "target_free_safety": safety,
            "source_sha256": fixed_source,
            "asset_sha256": fixed_assets,
            "model": fixed_model,
            "candidate_roster_sha256": candidate_roster_sha256,
            "boards": committed_boards,
        }
        hashes[commitment_name] = write_readonly_json(
            tmp_path / paths[commitment_name], commitment
        )
        receipt_name = f"{stage}_commitment_receipt"
        receipt = {
            "schema": "aiijc-drunet-sigma50-protected-all700-receipt-v1",
            "status": "commitment_created_before_any_target_decode_in_this_measurement_stage",
            "stage": stage,
            "count": 700,
            "config_sha256": measurement_sha,
            "commitment_sha256": hashes[commitment_name],
            "candidate_roster_sha256": candidate_roster_sha256,
            "targets_decoded_before_receipt": False,
            "competition_test_access": False,
        }
        hashes[receipt_name] = write_readonly_json(tmp_path / paths[receipt_name], receipt)
        target_name = f"{stage}_target_access_receipt"
        target_receipt = {
            "schema": "aiijc-drunet-sigma50-protected-all700-target-access-v1",
            "status": (
                "written_after_full_prediction_verification_and_immediately_before_target_decode"
            ),
            "stage": stage,
            "count": 700,
            "config_sha256": measurement_sha,
            "commitment_sha256": hashes[commitment_name],
            "commitment_receipt_sha256": hashes[receipt_name],
            "candidate_roster_sha256": candidate_roster_sha256,
            "predictions_were_committed_before_current_target_decode": True,
            "historical_workspace_target_exposure_acknowledged": True,
            "freshness_claim": False,
        }
        hashes[target_name] = write_readonly_json(
            tmp_path / paths[target_name], target_receipt
        )
        report_name = f"{stage}_report"
        score = 0.275
        rows = [
            {
                "filename": board["filename"],
                "layout_sha256": board["layout_sha256"],
                "candidate_pixel_sha256": board["candidate_pixel_sha256"],
                "ssim": score,
            }
            for board in committed_boards
        ]
        report = {
            "schema": "aiijc-drunet-sigma50-protected-all700-report-v1",
            "status": "exact_all700_measurement_from_precommitted_predictions",
            "stage": stage,
            "count": 700,
            "fixed_candidate": candidate,
            "mean_ssim": float(np.asarray([score] * 700, dtype=np.float64).mean()),
            "config_sha256": measurement_sha,
            "commitment_sha256": hashes[commitment_name],
            "commitment_receipt_sha256": hashes[receipt_name],
            "target_access_receipt_sha256": hashes[target_name],
            "candidate_roster_sha256": candidate_roster_sha256,
            "rows": rows,
            "strict_provenance": {"all_700_pass": True},
            "broad_completion_gate": {"passed": True},
            "target_free_safety": safety,
            "competition_test_access": False,
        }
        hashes[report_name] = write_readonly_json(tmp_path / paths[report_name], report)
        review_name = f"{stage}_manual_review"
        review = {
            "schema": "aiijc-fixed-b-standard-all700-manual-review-v1",
            "reviewer": "root",
            "stage": stage,
            "reviewed_arm": candidate,
            "reviewed_board_count": 700,
            "reviewed_all_700_outputs": True,
            "severe_artifacts": 0,
            "material_face_text_or_object_loss": False,
            "mask_halo_or_boundary_damage": False,
            "constant_or_near_flat_tile_substitution": False,
            "passed": True,
            "report_sha256": hashes[report_name],
            "commitment_sha256": hashes[commitment_name],
        }
        hashes[review_name] = write_readonly_json(tmp_path / paths[review_name], review)

    config = {
        "schema": "aiijc-fixed-b-standard-production-authorization-v1",
        "status": (
            "CALIBRATION700_AND_UNCHANGED_HOLDOUT700_NUMERIC_PROVENANCE_"
            "SAFETY_FLATNESS_MANUAL_PASS"
        ),
        "production_authorized_by_root": True,
        "canonical_device": "mps",
        "promoted_arm": candidate,
        "pipeline": production.frozen_pipeline_record(),
        "evidence": {},
    }
    for name in production.EVIDENCE_NAMES:
        record = {"path": str(paths[name]), "sha256": hashes[name]}
        if name == "production_runtime_manifest":
            record["digest_sha256"] = runtime_manifest["digest_sha256"]
        config["evidence"][name] = record
    config_path = tmp_path / "promotion.json"
    write_readonly_json(config_path, config)
    return config_path


def test_exact_frozen_b_candidate_and_r_reference_commitment_loads(tmp_path: Path) -> None:
    assert production.PROMOTED_ARM == "B_drunet50_protected_h28_h50_t60"
    assert production.SAFETY_REFERENCE == "R_drunet50_h28_safety_reference"
    exact_commitment = {
        "schema": "aiijc-drunet-sigma50-protected-all700-commitment-v1",
        "stage": "calibration",
        "count": 700,
        "config_sha256": "a" * 64,
        "fixed_candidate": "B_drunet50_protected_h28_h50_t60",
        "target_free_safety_reference_only": "R_drunet50_h28_safety_reference",
        "all_700_strict_raw_permutation_audits_pass": True,
        "target_free_safety": synthetic_safety(),
        "source_sha256": dict(production.EXPECTED_BROAD_SOURCE_SHA256),
        "asset_sha256": dict(production.EXPECTED_BROAD_ASSET_SHA256),
        "candidate_roster_sha256": "c" * 64,
        "boards": synthetic_committed_boards(),
        "model": {
            "sigma_255": 50.0,
            "batch_size": 144,
            "parameter_count": 32_640_960,
        },
    }
    production._validate_commitment(
        exact_commitment,
        stage="calibration",
        measurement_sha256="a" * 64,
    )
    alias_commitment = {
        **exact_commitment,
        "fixed_candidate": "F_drunet50_protected_h28_h50_t60",
    }
    with pytest.raises(ValueError, match="candidate changed"):
        production._validate_commitment(
            alias_commitment,
            stage="calibration",
            measurement_sha256="a" * 64,
        )

    config_path = promotion_fixture(tmp_path)
    evidence = production.load_promotion_evidence(config_path, project_root=tmp_path)
    assert evidence.artifacts["measurement_config"] == {
        "path": production.MEASUREMENT_CONFIG_PATH,
        "sha256": production.MEASUREMENT_CONFIG_SHA256,
    }
    assert evidence.artifacts["calibration_commitment"]["sha256"]
    assert evidence.artifacts["holdout_report"]["sha256"]

    alias_names = tmp_path / "alias-names"
    failed_config = promotion_fixture(
        alias_names,
        candidate="F_drunet50_protected_h28_h50_t60",
        safety_reference="B_drunet50_h28_safety_reference",
    )
    with pytest.raises(ValueError, match="promoted arm changed"):
        production.load_promotion_evidence(failed_config, project_root=alias_names)


def test_measurement_config_alias_path_fails_closed(tmp_path: Path) -> None:
    config_path = promotion_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    alias_relative = "configs/measurement-alias.json"
    alias_path = tmp_path / alias_relative
    shutil.copy2(PROJECT_ROOT / production.MEASUREMENT_CONFIG_PATH, alias_path)
    alias_path.chmod(0o444)
    config["evidence"]["measurement_config"]["path"] = alias_relative
    config_path.unlink()
    write_readonly_json(config_path, config)
    with pytest.raises(ValueError, match="measurement config path changed"):
        production.load_promotion_evidence(config_path, project_root=tmp_path)


def test_in_root_symlink_evidence_alias_fails_closed(tmp_path: Path) -> None:
    config_path = promotion_fixture(tmp_path)
    measurement_path = tmp_path / production.MEASUREMENT_CONFIG_PATH
    real_path = measurement_path.with_name("real-measurement.json")
    measurement_path.rename(real_path)
    measurement_path.symlink_to(real_path.name)
    with pytest.raises(ValueError, match="symlink paths are forbidden"):
        production.load_promotion_evidence(config_path, project_root=tmp_path)


def test_in_root_symlink_evidence_ancestor_fails_closed(tmp_path: Path) -> None:
    config_path = promotion_fixture(tmp_path)
    configs = tmp_path / "configs"
    real_configs = tmp_path / "real-configs"
    configs.rename(real_configs)
    configs.symlink_to(real_configs.name, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink paths are forbidden"):
        production.load_promotion_evidence(config_path, project_root=tmp_path)


def _one_hash_drift(mapping: dict[str, str]) -> dict[str, str]:
    changed = dict(mapping)
    first = next(iter(changed))
    changed[first] = "0" * 64
    return changed


@pytest.mark.parametrize(
    ("keyword", "expected_error"),
    (
        ("measurement_source_sha256", "measurement config hash changed"),
        ("measurement_asset_sha256", "measurement config hash changed"),
        ("commitment_source_sha256", "commitment frozen broad source map changed"),
        ("commitment_asset_sha256", "commitment frozen broad asset map changed"),
    ),
)
def test_config_and_commitment_source_asset_map_drift_fail_closed(
    tmp_path: Path,
    keyword: str,
    expected_error: str,
) -> None:
    base = (
        production.EXPECTED_BROAD_SOURCE_SHA256
        if "source" in keyword
        else production.EXPECTED_BROAD_ASSET_SHA256
    )
    config_path = promotion_fixture(tmp_path, **{keyword: _one_hash_drift(dict(base))})
    with pytest.raises(ValueError, match=expected_error):
        production.load_promotion_evidence(config_path, project_root=tmp_path)


def test_runtime_preflight_tamper_and_current_runtime_drift_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tamper_root = tmp_path / "tamper"
    config_path = promotion_fixture(tamper_root)
    runtime_path = tamper_root / production.RUNTIME_PREFLIGHT_RELATIVE_PATH
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["versions"]["python"] = "tampered-runtime"
    content = {key: value for key, value in runtime.items() if key != "digest_sha256"}
    runtime["digest_sha256"] = production._canonical_json_sha256(content)
    runtime_path.unlink()
    runtime_file_sha = write_readonly_json(runtime_path, runtime)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["evidence"]["production_runtime_manifest"] = {
        "path": production.RUNTIME_PREFLIGHT_RELATIVE_PATH,
        "sha256": runtime_file_sha,
        "digest_sha256": runtime["digest_sha256"],
    }
    config_path.unlink()
    write_readonly_json(config_path, config)
    with pytest.raises(ValueError, match="runtime differs from frozen preflight"):
        production.load_promotion_evidence(config_path, project_root=tamper_root)

    drift_root = tmp_path / "current-drift"
    drift_config = promotion_fixture(drift_root)
    current = production.build_runtime_manifest()
    changed_current = copy.deepcopy(current)
    source_name = "src/aiijc_puzzle/compliant_fixed_b_standard_submission.py"
    changed_current["files"][source_name] = "0" * 64
    current_content = {
        key: value for key, value in changed_current.items() if key != "digest_sha256"
    }
    changed_current["digest_sha256"] = production._canonical_json_sha256(current_content)
    monkeypatch.setattr(production, "build_runtime_manifest", lambda: changed_current)
    with pytest.raises(ValueError, match="runtime differs from frozen preflight"):
        production.load_promotion_evidence(drift_config, project_root=drift_root)


@pytest.mark.parametrize(
    ("section", "key"),
    (
        ("versions", "scipy"),
        ("versions", "pillow"),
        ("versions", "jsonschema"),
        ("host", "platform"),
        ("host", "machine"),
    ),
)
def test_runtime_preflight_dependency_or_host_drift_fail_closed(
    tmp_path: Path,
    section: str,
    key: str,
) -> None:
    config_path = promotion_fixture(tmp_path)
    runtime_path = tmp_path / production.RUNTIME_PREFLIGHT_RELATIVE_PATH
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime[section][key] = f"tampered-{section}-{key}"
    content = {name: value for name, value in runtime.items() if name != "digest_sha256"}
    runtime["digest_sha256"] = production._canonical_json_sha256(content)
    runtime_path.unlink()
    runtime_file_sha = write_readonly_json(runtime_path, runtime)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["evidence"]["production_runtime_manifest"] = {
        "path": production.RUNTIME_PREFLIGHT_RELATIVE_PATH,
        "sha256": runtime_file_sha,
        "digest_sha256": runtime["digest_sha256"],
    }
    config_path.unlink()
    write_readonly_json(config_path, config)
    with pytest.raises(ValueError, match="runtime differs from frozen preflight"):
        production.load_promotion_evidence(config_path, project_root=tmp_path)


def test_runtime_preflight_freezer_is_readonly_and_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(production, "DEFAULT_PROMOTION_CONFIG", tmp_path / "missing-promotion")
    destination = tmp_path / "runtime-manifest.json"
    result = production.freeze_production_runtime_preflight(path=destination)
    assert result["competition_test_access"] is False
    assert result["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert result["digest_sha256"] == json.loads(destination.read_text())["digest_sha256"]
    assert json.loads(destination.read_text(encoding="utf-8")) == (
        production.build_runtime_manifest()
    )
    assert destination.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        production.freeze_production_runtime_preflight(path=destination)


def test_runtime_preflight_freezer_rejects_unavailable_mps_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(production, "DEFAULT_PROMOTION_CONFIG", tmp_path / "missing-promotion")
    current = production.build_runtime_manifest()
    current["host"]["mps_available"] = False
    content = {key: value for key, value in current.items() if key != "digest_sha256"}
    current["digest_sha256"] = production._canonical_json_sha256(content)
    monkeypatch.setattr(production, "build_runtime_manifest", lambda: current)
    destination = tmp_path / "runtime-manifest.json"
    with pytest.raises(RuntimeError, match="requires an available MPS backend"):
        production.freeze_production_runtime_preflight(path=destination)
    assert not destination.exists()


def test_runtime_manifest_is_complete_and_non_self_referential() -> None:
    schema_relative = str(production.DEFAULT_SCHEMA_PATH.relative_to(PROJECT_ROOT))
    promotion_relative = str(production.DEFAULT_PROMOTION_CONFIG.relative_to(PROJECT_ROOT))
    assert production.RUNTIME_PREFLIGHT_RELATIVE_PATH not in production.RUNTIME_FILE_RELATIVE_PATHS
    assert promotion_relative not in production.RUNTIME_FILE_RELATIVE_PATHS
    assert schema_relative in production.RUNTIME_FILE_RELATIVE_PATHS

    manifest = production.build_runtime_manifest()
    assert manifest["files"][schema_relative] == production.PINNED_SCHEMA_SHA256
    assert set(manifest["versions"]) == {
        "python",
        "numpy",
        "opencv",
        "torch",
        "scipy",
        "pillow",
        "scikit_image",
        "scikit_learn",
        "jsonschema",
    }
    assert manifest["versions"]["scipy"] == production.scipy.__version__
    assert manifest["versions"]["pillow"] == production.PILLOW_VERSION
    assert manifest["versions"]["scikit_image"] == production.skimage.__version__
    assert manifest["versions"]["scikit_learn"] == production.sklearn.__version__
    assert manifest["versions"]["jsonschema"] == production.distribution_version(
        "jsonschema"
    )
    assert set(manifest["host"]) == {
        "platform",
        "mac_ver",
        "machine",
        "mps_available",
    }
    mac_release, mac_version_info, mac_machine = production.platform.mac_ver()
    assert manifest["host"]["platform"] == production.platform.platform()
    assert manifest["host"]["mac_ver"] == [
        mac_release,
        list(mac_version_info),
        mac_machine,
    ]
    assert manifest["host"]["machine"] == production.platform.machine()
    assert manifest["host"]["mps_available"] is bool(
        production.torch.backends.mps.is_available()
    )


def _copy_frozen_broad_tree(destination: Path) -> None:
    for relative in production.EXPECTED_BROAD_SOURCE_SHA256:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, target)
    asset_root = destination / production.KAIR_ASSET_ROOT_RELATIVE
    for relative in production.EXPECTED_BROAD_ASSET_SHA256:
        source = PROJECT_ROOT / production.KAIR_ASSET_ROOT_RELATIVE / relative
        target = asset_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == "drunet_color.pth":
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        else:
            shutil.copy2(source, target)


def test_current_frozen_broad_file_and_asset_one_byte_drift_fail_closed(tmp_path: Path) -> None:
    _copy_frozen_broad_tree(tmp_path)
    observed = production.verify_broad_measurement_integrity(project_root=tmp_path)
    assert observed["source_sha256"] == production.EXPECTED_BROAD_SOURCE_SHA256
    assert observed["asset_sha256"] == production.EXPECTED_BROAD_ASSET_SHA256

    source_path = tmp_path / "src/aiijc_puzzle/drunet_sigma50_protected_broad.py"
    original_source = source_path.read_bytes()
    source_path.write_bytes(original_source + b"\n")
    with pytest.raises(ValueError, match="current frozen broad source hash drift"):
        production.verify_broad_measurement_integrity(project_root=tmp_path)
    source_path.write_bytes(original_source)

    license_path = tmp_path / production.KAIR_ASSET_ROOT_RELATIVE / "LICENSE"
    license_path.write_bytes(license_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="current frozen broad asset hash drift"):
        production.verify_broad_measurement_integrity(project_root=tmp_path)


def test_report_rows_mean_roster_provenance_and_safety_are_commitment_bound() -> None:
    for stage in ("calibration", "holdout"):
        boards = synthetic_committed_boards()
        safety = synthetic_safety()
        commitment = {
            "candidate_roster_sha256": "c" * 64,
            "boards": boards,
            "target_free_safety": safety,
        }
        score = 0.275
        report = {
            "schema": "aiijc-drunet-sigma50-protected-all700-report-v1",
            "status": "exact_all700_measurement_from_precommitted_predictions",
            "stage": stage,
            "count": 700,
            "fixed_candidate": production.PROMOTED_ARM,
            "mean_ssim": float(np.asarray([score] * 700, dtype=np.float64).mean()),
            "config_sha256": "a" * 64,
            "commitment_sha256": "b" * 64,
            "commitment_receipt_sha256": "d" * 64,
            "target_access_receipt_sha256": "e" * 64,
            "candidate_roster_sha256": "c" * 64,
            "rows": [
                {
                    "filename": board["filename"],
                    "layout_sha256": board["layout_sha256"],
                    "candidate_pixel_sha256": board["candidate_pixel_sha256"],
                    "ssim": score,
                }
                for board in boards
            ],
            "strict_provenance": {"all_700_pass": True},
            "broad_completion_gate": {"passed": True},
            "target_free_safety": safety,
            "competition_test_access": False,
        }
        kwargs = {
            "commitment": commitment,
            "stage": stage,
            "measurement_sha256": "a" * 64,
            "commitment_sha256": "b" * 64,
            "receipt_sha256": "d" * 64,
            "target_access_sha256": "e" * 64,
        }
        production._validate_report(report, **kwargs)

        mutations: list[tuple[str, dict[str, object], str]] = []
        changed = copy.deepcopy(report)
        changed["mean_ssim"] = float(report["mean_ssim"]) + 0.0001
        mutations.append(("mean", changed, "mean does not match"))
        changed = copy.deepcopy(report)
        changed["rows"][317]["candidate_pixel_sha256"] = "0" * 64
        mutations.append(("row", changed, "row/commitment mismatch"))
        changed = copy.deepcopy(report)
        changed["candidate_roster_sha256"] = "0" * 64
        mutations.append(("roster", changed, "candidate roster binding"))
        changed = copy.deepcopy(report)
        changed["rows"][0]["ssim"] = float("nan")
        mutations.append(("nonfinite", changed, "SSIM is invalid"))
        changed = copy.deepcopy(report)
        changed["rows"][0]["ssim"] = 1.0001
        mutations.append(("out-of-range", changed, "SSIM is invalid"))
        changed = copy.deepcopy(report)
        changed["strict_provenance"]["all_700_pass"] = False
        mutations.append(("provenance", changed, "provenance gate"))
        changed = copy.deepcopy(report)
        safety_key = next(iter(production.EXPECTED_BROAD_SAFETY_CHECK_NAMES))
        changed["target_free_safety"]["checks"][safety_key] = False
        mutations.append(("safety", changed, "safety differs"))
        for _label, mutated, message in mutations:
            with pytest.raises(ValueError, match=message):
                production._validate_report(mutated, **kwargs)


def test_frozen_calibration_manual_review_path_hash_and_extras_are_accepted() -> None:
    review_path = (
        PROJECT_ROOT
        / "outputs/drunet-sigma50-protected/all700-measurement-v1/calibration700/"
        "manual-review.json"
    )
    assert hashlib.sha256(review_path.read_bytes()).hexdigest() == (
        "cd347ab92d037b6c086b8a4946d1bfaf1ffde9f6d08c2d7a421639ce9540e3d8"
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review["review_material"]["coverage"].startswith("seven target-free")
    assert "does not prove" in review["limitation"]
    production._validate_manual_review(
        review,
        stage="calibration",
        report_sha256="06934d5a2450ef3752a88c0d9f8ab90b17b5f0a3d859ef8cc67dbb4da3392590",
        commitment_sha256=(
            "13a61821d616ad01eee79f45db51dc223d1c041fe502ff84525e6259e623853c"
        ),
    )


def test_frozen_holdout_manual_review_path_hash_and_extras_are_accepted() -> None:
    review_path = (
        PROJECT_ROOT
        / "outputs/drunet-sigma50-protected/all700-measurement-v1/holdout700/"
        "manual-review.json"
    )
    assert hashlib.sha256(review_path.read_bytes()).hexdigest() == (
        "30880261fe19afee448c0cf997e1d1e7261e348bf9aa6c51f299d78bc86d759e"
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review["review_material"]["coverage"].startswith("seven target-free")
    assert review["independent_review"]["severe_restoration_induced_artifacts"] == 0
    assert "does not prove" in review["limitation"]
    production._validate_manual_review(
        review,
        stage="holdout",
        report_sha256="05a4f2fab4d0dd07624e5a54c143004302cf4df78a15460388775fa0b1013d07",
        commitment_sha256=(
            "6df1f7b6d8d318a6dce6d81d6c35890cdb05e1c6481c1351b92d2393341298f2"
        ),
    )


def test_promotion_remains_absent_fail_closed_and_precedes_test_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "configs/missing-promotion.json"
    monkeypatch.setattr(production, "DEFAULT_PROMOTION_CONFIG", missing)
    status = production.dry_run_status()
    assert status["production_authorized"] is False
    assert status["competition_test_access"] is False
    assert not missing.exists()

    snapshot_called = False

    def snapshot_spy(*args, **kwargs):
        nonlocal snapshot_called
        snapshot_called = True
        raise AssertionError("test snapshot must not be opened before promotion")

    monkeypatch.setattr(
        independent,
        "load_promotion_evidence",
        lambda: (_ for _ in ()).throw(FileNotFoundError("blocked")),
    )
    monkeypatch.setattr(independent, "build_official_input_snapshot", snapshot_spy)
    with pytest.raises(FileNotFoundError, match="blocked"):
        independent.validate_submission(
            inputs_dir=tmp_path / "test",
            source_archive=tmp_path / "test.zip",
            submission_zip=tmp_path / "submission.zip",
            attestation_path=tmp_path / "attestation.json",
        )
    assert snapshot_called is False


def test_schema_is_parallel_pinned_and_rejects_geometry_or_tail_changes() -> None:
    schema_path = production.DEFAULT_SCHEMA_PATH
    assert hashlib.sha256(schema_path.read_bytes()).hexdigest() == production.PINNED_SCHEMA_SHA256
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)

    names = tuple(f"img_{index:06d}.png" for index in range(700))
    snapshot = InputSnapshot(
        source_archive_sha256="d" * 64,
        filenames=names,
        filenames_sha256="e" * 64,
        input_sha256=tuple((name, "f" * 64) for name in names),
    )
    template = production.board_attestation(
        filename=names[0],
        input_sha256="f" * 64,
        prediction=fake_prediction(),
        output_png_sha256="9" * 64,
    )
    records = []
    for name in names:
        record = copy.deepcopy(template)
        record["filename"] = name
        records.append(record)
    payload = production.build_attestation(
        snapshot=snapshot,
        archive_sha256="8" * 64,
        per_board=records,
        promotion=fake_promotion(),
        runtime_manifest=fake_runtime_manifest(),
    )
    Draft202012Validator(schema).validate(payload)
    assert payload["status"] == "METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN"

    for key, value in (
        (("per_board", 0, "restoration", "drunet", "sigma_255"), 40.0),
        (("per_board", 0, "restoration", "nlm", "independent_single_pass_strengths"), [20, 28]),
        (("per_board", 0, "restoration", "protected_mask", "sobel_threshold"), 40.0),
        (("policy", "constant_or_near_flat_tile_substitution_used"), True),
    ):
        changed = copy.deepcopy(payload)
        cursor = changed
        for part in key[:-1]:
            cursor = cursor[part]
        cursor[key[-1]] = value
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(changed)

    for section, key in (
        ("versions", "scipy"),
        ("versions", "pillow"),
        ("versions", "jsonschema"),
        ("host", "platform"),
        ("host", "mps_available"),
    ):
        changed = copy.deepcopy(payload)
        del changed["runtime_manifest"][section][key]
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(changed)

    changed = copy.deepcopy(payload)
    changed["runtime_manifest"]["host"]["unexpected"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(changed)

    duplicated = copy.deepcopy(payload)
    duplicated["per_board"][0]["tile_at_position"][1] = 0
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(duplicated)


def test_public_production_has_no_tail_or_model_selection_parameters() -> None:
    prediction_parameters = set(inspect.signature(production.predict_fixed_b_standard).parameters)
    assert prediction_parameters == {"input_image", "model", "device"}
    run_parameters = set(inspect.signature(production.run_production_submission).parameters)
    assert run_parameters == {"inputs_dir", "source_archive"}
    assert run_parameters.isdisjoint(
        {"sigma", "h", "threshold", "layout", "checkpoint", "device", "target"}
    )
    assert production.OUTPUT_ROOT.name == "compliant-fixed-b-standard-submission-v1"
    assert production.DEFAULT_OUTPUT_ZIP.parent == production.OUTPUT_ROOT


def test_prediction_preserves_strict_raw_bijection_and_calls_one_frozen_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, columns = np.indices((IMAGE_SIZE, IMAGE_SIZE))
    image = np.stack((rows % 256, columns % 256, (rows + columns) % 256), axis=2).astype(np.uint8)
    layout = np.arange(TILE_COUNT, dtype=np.int32)[::-1]
    calls = 0
    monkeypatch.setattr(
        production,
        "directional_scores",
        lambda *args, **kwargs: {
            "bilateral": (
                np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float32),
                np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float32),
            )
        },
    )
    monkeypatch.setattr(production, "solve_buddies", lambda *args, **kwargs: FakeSolve(layout))
    monkeypatch.setattr(production, "_apply_frozen_harmonizers", lambda value: value.copy())

    def frozen_renderer(model, tiles, *, device):
        nonlocal calls
        calls += 1
        board = assemble_tiles(tiles)
        h28 = np.full_like(board, 28)
        candidate = np.full_like(board, 50)
        binary = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        soft = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        diagnostics = {
            "neural_intermediate_pixel_sha256": {
                "drunet50_canvas": pixel_sha256(board),
                "drunet50_then_h20_mask_source": pixel_sha256(board),
                "drunet50_then_h28_reference_and_safe": pixel_sha256(h28),
                "drunet50_then_h50_flat": pixel_sha256(candidate),
            },
            "mask": {
                "binary_mask_sha256": independent._typed_array_digest(binary),
                "soft_mask_sha256": independent._typed_array_digest(soft),
                "binary_dilated_protected_fraction": 0.0,
            },
            "drunet": fake_drunet_diagnostics(),
        }
        return h28, candidate, diagnostics

    monkeypatch.setattr(production, "render_sigma50_protected", frozen_renderer)
    prediction = production.predict_fixed_b_standard(
        image, torch.nn.Identity(), device=torch.device("cpu")
    )
    np.testing.assert_array_equal(prediction.raw, assemble_tiles(split_tiles(image)[layout]))
    assert prediction.audit["passed"]
    assert prediction.tile_multiset_sha256 == production.tile_multiset_sha256(image)
    assert calls == 1
    assert np.all(prediction.restored == 50)


class EchoModel(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :3]


class ParameterCountOnlyModel:
    def parameters(self):
        class Count:
            @staticmethod
            def numel() -> int:
                return production.MODEL_PARAMETER_COUNT

        return (Count(),)


def test_independent_tile_renderer_and_t60_mask_match_frozen_math() -> None:
    rng = np.random.default_rng(905)
    tiles = rng.integers(0, 256, size=(TILE_COUNT, 20, 20, 3), dtype=np.uint8)
    model = EchoModel()
    expected, _ = render_drunet_tiles(
        model,
        tiles,
        sigma_255=50,
        device=torch.device("cpu"),
        batch_size=144,
    )
    observed = independent._independent_drunet_tiles(
        model,
        tiles,
        device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(observed, expected)

    board = assemble_tiles(tiles)
    binary, soft, fraction = protected_masks(board, sobel_threshold=60.0)
    observed_binary, observed_soft, observed_fraction = independent._independent_masks(board)
    np.testing.assert_array_equal(observed_binary, binary)
    np.testing.assert_array_equal(observed_soft, soft)
    assert observed_fraction == fraction


def test_independent_validator_recomputes_all_700_boards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    names = tuple(f"img_{index:06d}.png" for index in range(700))
    board = zeros_board()
    payload = png_bytes(board)
    payload_sha = hashlib.sha256(payload).hexdigest()
    for name in names:
        (inputs / name).write_bytes(payload)
    source_archive = tmp_path / "source.zip"
    submission_zip = tmp_path / "submission.zip"
    deterministic_submission_zip(inputs, list(names), source_archive)
    deterministic_submission_zip(inputs, list(names), submission_zip)
    snapshot = InputSnapshot(
        source_archive_sha256=hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        filenames=names,
        filenames_sha256=hashlib.sha256(
            b"".join(name.encode() + b"\0" for name in names)
        ).hexdigest(),
        input_sha256=tuple((name, payload_sha) for name in names),
    )
    prediction = fake_prediction()
    template = production.board_attestation(
        filename=names[0],
        input_sha256=payload_sha,
        prediction=prediction,
        output_png_sha256=payload_sha,
    )
    records = []
    for name in names:
        record = copy.deepcopy(template)
        record["filename"] = name
        records.append(record)
    promotion = fake_promotion()
    runtime = fake_runtime_manifest()
    attestation = production.build_attestation(
        snapshot=snapshot,
        archive_sha256=hashlib.sha256(submission_zip.read_bytes()).hexdigest(),
        per_board=records,
        promotion=promotion,
        runtime_manifest=runtime,
    )
    attestation_path = tmp_path / "attestation.json"
    atomic_write_json(attestation_path, attestation)

    monkeypatch.setattr(
        independent,
        "_independent_layout",
        lambda image: np.arange(TILE_COUNT, dtype=np.int32),
    )
    monkeypatch.setattr(independent, "_independent_harmonize", lambda image: image.copy())
    fake_derived = {
        "drunet_canvas": board,
        "h20": board,
        "h28": board,
        "h50": board,
        "binary": np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool),
        "soft": np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
        "protected_fraction": 0.0,
        "output": board,
    }
    calls = 0

    def fake_restore(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_derived

    monkeypatch.setattr(independent, "_independent_restore", fake_restore)
    monkeypatch.setattr(
        independent,
        "load_drunet_color",
        lambda *args, **kwargs: ParameterCountOnlyModel(),
    )
    report = independent.validate_against_snapshot(
        snapshot=snapshot,
        inputs_dir=inputs,
        submission_zip=submission_zip,
        attestation_path=attestation_path,
        promotion=promotion,
        runtime_manifest=runtime,
        device=torch.device("mps"),
    )
    assert calls == 700
    assert report["boards_fully_recomputed"] == 700
    assert report["all_h20_h28_h50_t60_masks_and_blends_recomputed"]
    assert report["status"] == "METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN"


def test_existing_h20_and_drunet40_artifacts_are_byte_immutable() -> None:
    expected = {
        "outputs/compliant-submission/submission.zip": (
            "7c36307af0ea821c8a5fbf3139323ece332744dcf59a413198dd96d5a2f619bf"
        ),
        "outputs/compliant-submission/compliance-attestation.json": (
            "5323d05b71b56645a7ad2acab5276187035c4e1e9de07c3fb34821b60c688c8f"
        ),
        "src/aiijc_puzzle/compliant_drunet_protected_submission.py": (
            "86d1e86b31d1c672857650622abe79b6915b9116f8ff82c55a251a1b5a9c3816"
        ),
        "src/aiijc_puzzle/compliant_drunet_protected_validation.py": (
            "bbb0062514bee1ad16db053cac8b6f2156449e8132a03bc41382ee12cfd17880"
        ),
        "configs/compliant-drunet-protected-submission-v3.schema.json": (
            "a5581b56604671cee44747ede095a2666f2375ebd74962b8ffaa5484fcc5bf69"
        ),
        "notebooks/reproduce_compliant_drunet_protected_colab.ipynb": (
            "524e2f92a48cea5f298a0b26b185c18ab63a5b6dc2e301355f6b6fb804baf7bd"
        ),
    }
    observed = {
        relative: hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
        for relative in expected
    }
    assert observed == expected
    assert production.OUTPUT_ROOT != PROJECT_ROOT / "outputs/compliant-submission"
    assert production.OUTPUT_ROOT != (
        PROJECT_ROOT / "outputs/compliant-drunet-protected-submission-v1"
    )


@pytest.mark.parametrize("failure_at", (2, 3))
def test_publish_failure_rolls_back_every_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_at: int,
) -> None:
    source_directory = tmp_path / "source-predictions"
    source_directory.mkdir()
    (source_directory / "one.png").write_bytes(b"one")
    sources = [
        source_directory,
        tmp_path / "source.zip",
        tmp_path / "source-attestation.json",
        tmp_path / "source-validation.json",
    ]
    for index, source in enumerate(sources[1:], start=1):
        source.write_bytes(bytes([index]))
    destinations = [
        tmp_path / "final/predictions",
        tmp_path / "final/submission.zip",
        tmp_path / "final/compliance-attestation.json",
        tmp_path / "final/independent-validation.json",
    ]
    destinations[0].parent.mkdir()
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("injected publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(production.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected publish failure"):
        production._publish_validated_artifacts(tuple(zip(sources, destinations, strict=True)))
    assert calls == failure_at
    assert not any(destination.exists() for destination in destinations)


def test_colab_never_chmods_config_supplied_evidence_before_validation() -> None:
    notebook = json.loads(
        (PROJECT_ROOT / "notebooks/reproduce_compliant_fixed_b_standard_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    authorization_cell = next(
        cell for cell in notebook["cells"] if cell.get("id") == "authorization"
    )
    source = "".join(authorization_cell["source"])
    assert "chmod" not in source
    assert "EXPECTED STOP" in source
    assert source.index("load_promotion_evidence()") < source.index("raise RuntimeError(")
