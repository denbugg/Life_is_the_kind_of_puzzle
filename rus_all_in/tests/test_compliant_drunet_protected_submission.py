from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

import aiijc_puzzle.compliant_drunet_protected_submission as production
import aiijc_puzzle.compliant_drunet_protected_validation as independent
from aiijc_puzzle.compliant_submission import InputSnapshot, array_sha256, atomic_write_json
from aiijc_puzzle.legacy_upgrade import deterministic_submission_zip, png_bytes
from aiijc_puzzle.protocol import IMAGE_SIZE, TILE_COUNT, assemble_tiles, split_tiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def zeros_board() -> np.ndarray:
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


@dataclass
class FakeSolve:
    layout: np.ndarray
    runtime_seconds: float = 0.0


@dataclass
class FakeDrunetDiagnostics:
    def as_dict(self) -> dict[str, int | float]:
        return {
            "tile_count": 576,
            "sigma_255": 40.0,
            "batch_size": 144,
            "padding_bottom": 4,
            "padding_right": 4,
            "runtime_seconds": 0.0,
            "mean_abs_change": 0.0,
            "q99_abs_change": 0.0,
            "maximum_abs_change": 0,
            "clipped_fraction": 1.0,
        }


def fake_prediction() -> production.ProtectedSubmissionPrediction:
    board = zeros_board()
    tiles = split_tiles(board)
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
    return production.ProtectedSubmissionPrediction(
        layout=np.arange(TILE_COUNT, dtype=np.int32),
        raw=board,
        harmonized=board,
        restored=board,
        audit=audit,
        tile_multiset_sha256=production.tile_multiset_sha256(board),
        restoration={
            "harmonized_array_sha256": array_sha256(board),
            "drunet_tiles_array_sha256": array_sha256(tiles),
            "drunet_canvas_array_sha256": array_sha256(board),
            "drunet_diagnostics": FakeDrunetDiagnostics().as_dict(),
            "nlm_h20_array_sha256": array_sha256(board),
            "nlm_h28_array_sha256": array_sha256(board),
            "nlm_h40_array_sha256": array_sha256(board),
            "binary_mask_array_sha256": array_sha256(binary),
            "soft_mask_array_sha256": array_sha256(soft),
            "protected_fraction": 0.0,
            "output_array_sha256": array_sha256(board),
        },
        score_seconds=0.0,
        solve_seconds=0.0,
        restoration_seconds=0.0,
    )


def fake_promotion() -> production.PromotionEvidence:
    return production.PromotionEvidence(
        config_sha256="1" * 64,
        artifacts={
            name: {"path": relative, "sha256": f"{index + 2:064x}"}
            for index, (name, relative) in enumerate(production.PROMOTION_EVIDENCE_PATHS.items())
        },
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
        },
        "canonical_device": "mps",
    }
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**content, "digest_sha256": digest}


def write_readonly_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def promotion_fixture(tmp_path: Path, *, confirmation_pass: bool = True) -> Path:
    preregistration = {
        "schema": "aiijc-pretrained-drunet-protected-stack-preregistration-v1",
        "arm_names": ["A", "B", "C", production.PROMOTED_ARM],
        "geometry_contract": {
            "strict_shared_layout": True,
            "all_576_upright_tiles_preserved_one_to_one": True,
            "same_board_pixels_only": True,
            "cross_tile_neural_context": False,
            "cross_board_context": False,
            "resize": False,
            "warp": False,
            "rotation": False,
            "flip": False,
            "external_reference_or_template_pixels": False,
            "generation_or_substitution": False,
        },
    }
    payloads: dict[str, object] = {"preregistration": preregistration}
    evidence_hashes: dict[str, str] = {}
    prereg_path = tmp_path / production.PROMOTION_EVIDENCE_PATHS["preregistration"]
    evidence_hashes["preregistration"] = write_readonly_json(prereg_path, preregistration)
    prereg_sha = evidence_hashes["preregistration"]
    for stage in ("primary", "confirmation"):
        commitment_name = f"{stage}_commitment"
        receipt_name = f"{stage}_commitment_receipt"
        commitment = {"schema": "commitment", "stage": stage}
        receipt = {"schema": "receipt", "stage": stage}
        evidence_hashes[commitment_name] = write_readonly_json(
            tmp_path / production.PROMOTION_EVIDENCE_PATHS[commitment_name],
            commitment,
        )
        evidence_hashes[receipt_name] = write_readonly_json(
            tmp_path / production.PROMOTION_EVIDENCE_PATHS[receipt_name],
            receipt,
        )
        report = {
            "schema": "aiijc-pretrained-drunet-protected-stack-report-v1",
            "status": "scored_from_frozen_predictions",
            "stage": stage,
            "count": 120,
            "offset": 264 if stage == "primary" else 408,
            "preregistration_sha256": prereg_sha,
            "commitment_sha256": evidence_hashes[commitment_name],
            "commitment_receipt_sha256": evidence_hashes[receipt_name],
            "quantitative_pass": True if stage == "primary" else confirmation_pass,
            "selected_passing_winner": production.PROMOTED_ARM,
            "competition_test_access": False,
            "holdout_access": False,
            "quantitative_checks": {"all": True},
        }
        report_name = f"{stage}_report"
        evidence_hashes[report_name] = write_readonly_json(
            tmp_path / production.PROMOTION_EVIDENCE_PATHS[report_name],
            report,
        )
        review = {
            "reviewer": "root",
            "reviewed_arm": production.PROMOTED_ARM,
            "reviewed_board_count": 120,
            "reviewed_all_full_canvas_triplets": True,
            "passed": True,
            "severe_artifacts": 0,
            "material_face_text_or_object_loss": False,
            "mask_halo_or_boundary_damage": False,
            "preregistration_sha256": prereg_sha,
            f"{stage}_report_sha256": evidence_hashes[report_name],
        }
        review_name = f"{stage}_manual_review"
        evidence_hashes[review_name] = write_readonly_json(
            tmp_path / production.PROMOTION_EVIDENCE_PATHS[review_name],
            review,
        )
        payloads[report_name] = report
        payloads[review_name] = review
    config = {
        "schema": "aiijc-drunet-protected-production-authorization-v1",
        "status": "PRIMARY_AND_CONFIRMATION_NUMERIC_AND_MANUAL_PASS",
        "production_authorized_by_root": True,
        "canonical_device": "mps",
        "promoted_arm": production.PROMOTED_ARM,
        "pipeline": production.frozen_pipeline_record(),
        "evidence": {
            name: {"path": relative, "sha256": evidence_hashes[name]}
            for name, relative in production.PROMOTION_EVIDENCE_PATHS.items()
        },
    }
    config_path = tmp_path / "promotion.json"
    write_readonly_json(config_path, config)
    return config_path


def test_new_schema_is_parallel_pinned_and_rejects_parameter_changes() -> None:
    schema_path = production.DEFAULT_SCHEMA_PATH
    assert hashlib.sha256(schema_path.read_bytes()).hexdigest() == (production.PINNED_SCHEMA_SHA256)
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
    assert payload["correct_hidden_layout_proven"] is False

    changed = copy.deepcopy(payload)
    changed["per_board"][0]["restoration"]["drunet"]["sigma_255"] = 39.0
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(changed)
    changed = copy.deepcopy(payload)
    changed["per_board"][0]["restoration"]["nlm"]["independent_single_pass_strengths"] = [20, 28]
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(changed)


def test_promotion_evidence_is_readonly_hash_bound_and_fail_closed(tmp_path: Path) -> None:
    config_path = promotion_fixture(tmp_path)
    evidence = production.load_promotion_evidence(config_path, project_root=tmp_path)
    assert evidence.artifacts["primary_report"]["sha256"]
    assert evidence.artifacts["confirmation_report"]["sha256"]

    failed_root = tmp_path / "failed"
    failed_config = promotion_fixture(failed_root, confirmation_pass=False)
    with pytest.raises(ValueError, match="quantitative gate did not pass"):
        production.load_promotion_evidence(failed_config, project_root=failed_root)

    config_path.chmod(0o644)
    with pytest.raises(PermissionError, match="writable"):
        production.load_promotion_evidence(config_path, project_root=tmp_path)


def test_public_production_has_no_tuning_or_device_parameters() -> None:
    prediction_parameters = set(inspect.signature(production.predict_drunet_protected).parameters)
    assert prediction_parameters == {"input_image", "model", "device"}
    run_parameters = set(inspect.signature(production.run_production_submission).parameters)
    assert run_parameters == {"inputs_dir", "source_archive"}
    assert run_parameters.isdisjoint(
        {"sigma", "h", "threshold", "layout", "checkpoint", "device", "target"}
    )
    assert production.OUTPUT_ROOT.name == "compliant-drunet-protected-submission-v1"
    assert production.DEFAULT_OUTPUT_ZIP.parent == production.OUTPUT_ROOT


def test_prediction_preserves_strict_raw_bijection_and_uses_frozen_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, columns = np.indices((IMAGE_SIZE, IMAGE_SIZE))
    image = np.stack((rows % 256, columns % 256, (rows + columns) % 256), axis=2).astype(np.uint8)
    layout = np.arange(TILE_COUNT, dtype=np.int32)[::-1]
    calls: list[int] = []
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
    monkeypatch.setattr(
        production,
        "solve_buddies",
        lambda *args, **kwargs: FakeSolve(layout=layout),
    )
    monkeypatch.setattr(production, "_apply_frozen_harmonizers", lambda value: value.copy())
    monkeypatch.setattr(
        production,
        "render_drunet_tiles",
        lambda model, tiles, **kwargs: (tiles.copy(), FakeDrunetDiagnostics()),
    )

    def fake_nlm(value: np.ndarray, h: int) -> np.ndarray:
        calls.append(h)
        return np.full_like(value, h)

    soft = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    soft[:, : IMAGE_SIZE // 2] = 1.0
    monkeypatch.setattr(production, "colored_nlm", fake_nlm)
    monkeypatch.setattr(
        production,
        "protected_masks",
        lambda value, sobel_threshold: (soft.astype(bool), soft, 0.5),
    )
    prediction = production.predict_drunet_protected(
        image,
        torch.nn.Identity(),
        device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(prediction.raw, assemble_tiles(split_tiles(image)[layout]))
    assert prediction.audit["passed"]
    assert prediction.tile_multiset_sha256 == production.tile_multiset_sha256(image)
    assert calls == [20, 28, 40]
    assert np.all(prediction.restored[:, : IMAGE_SIZE // 2] == 28)
    assert np.all(prediction.restored[:, IMAGE_SIZE // 2 :] == 40)


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


def test_independent_tile_renderer_and_mask_match_frozen_math() -> None:
    rng = np.random.default_rng(902)
    tiles = rng.integers(0, 256, size=(TILE_COUNT, 20, 20, 3), dtype=np.uint8)
    model = EchoModel()
    expected, _ = production.render_drunet_tiles(
        model,
        tiles,
        sigma_255=40,
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
    binary, soft, fraction = production.protected_masks(board, sobel_threshold=40.0)
    independent_binary, independent_soft, independent_fraction = independent._independent_masks(
        board
    )
    np.testing.assert_array_equal(independent_binary, binary)
    np.testing.assert_array_equal(independent_soft, soft)
    assert independent_fraction == fraction


def test_independent_validator_recomputes_every_one_of_700_boards(
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
        filenames_sha256=production.hashlib.sha256(
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
        "drunet_tiles": split_tiles(board),
        "drunet_canvas": board,
        "h20": board,
        "h28": board,
        "h40": board,
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
    assert report["all_h20_h28_h40_masks_and_blends_recomputed"]
    assert report["status"] == "METHOD_COMPLIANT_LAYOUT_ACCURACY_UNPROVEN"

    mutated = copy.deepcopy(attestation)
    mutated["per_board"][0]["restoration"]["nlm"]["h20_array_sha256"] = "0" * 64
    mutated_path = tmp_path / "mutated.json"
    atomic_write_json(mutated_path, mutated)
    with pytest.raises(ValueError, match="NLM h20 hash changed"):
        independent.validate_against_snapshot(
            snapshot=snapshot,
            inputs_dir=inputs,
            submission_zip=submission_zip,
            attestation_path=mutated_path,
            promotion=promotion,
            runtime_manifest=runtime,
            device=torch.device("mps"),
        )


def test_dry_run_never_creates_or_modifies_production_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-promotion.json"
    monkeypatch.setattr(production, "DEFAULT_PROMOTION_CONFIG", missing)
    status = production.dry_run_status()
    assert status["status"] == "BLOCKED_AWAITING_IMMUTABLE_PROMOTION_AUTHORIZATION"
    assert status["production_executed"] is False
    assert not missing.exists()
    assert not (tmp_path / "predictions").exists()


def test_old_fallback_runtime_files_and_output_root_are_not_repurposed() -> None:
    assert production.OUTPUT_ROOT != PROJECT_ROOT / "outputs/compliant-submission"
    assert "src/aiijc_puzzle/compliant_submission.py" in production.RUNTIME_FILE_RELATIVE_PATHS
    assert "configs/submission-compliance.schema.json" not in production.RUNTIME_FILE_RELATIVE_PATHS
    assert not any(
        "frozen_submission_h20x1" in path for path in production.RUNTIME_FILE_RELATIVE_PATHS
    )
