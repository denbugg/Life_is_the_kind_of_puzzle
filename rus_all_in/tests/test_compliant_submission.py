from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from aiijc_puzzle.compliant_submission import (
    EXPECTED_POLICY,
    FINAL_NLM_H,
    FINAL_NLM_H_COLOR,
    FINAL_NLM_PASSES,
    METHOD_STATUS,
    OFFICIAL_FILENAMES_SHA256,
    OFFICIAL_TEST_ARCHIVE_SHA256,
    PINNED_SCHEMA_SHA256,
    PROOF_SCOPE,
    InputSnapshot,
    _apply_frozen_harmonizers,
    _independent_frozen_harmonizers,
    _independent_layout_digest,
    _independent_nlm_h20_once,
    _independent_no_atlas_buddies96_layout,
    _independent_raw_assembly,
    _inspect_submission_members,
    _load_compliance_schema,
    _method_declaration,
    _proper_rgb_nlm_h20_once,
    _validate_submission_against_snapshot,
    array_sha256,
    atomic_write_json,
    board_attestation,
    build_attestation,
    build_input_snapshot,
    build_runtime_manifest,
    filenames_digest,
    guard_artifact_paths,
    load_frozen_tail_evidence,
    predict_frozen_submission,
    restoration_name,
    run_production_submission,
    validate_submission,
)
from aiijc_puzzle.legacy_upgrade import (
    atomic_write_png,
    deterministic_submission_zip,
    png_bytes,
)
from aiijc_puzzle.protocol import IMAGE_SIZE, TILE_COUNT, assemble_tiles, split_tiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synthetic_board(seed: int = 301) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


def make_input_archive(tmp_path: Path, *, count: int = 2) -> tuple[Path, Path, list[str]]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    names = [f"img_{index:06d}.png" for index in range(count)]
    for index, name in enumerate(names):
        image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 20 + index, dtype=np.uint8)
        atomic_write_png(inputs / name, image)
    archive = tmp_path / "test.zip"
    deterministic_submission_zip(inputs, names, archive)
    return inputs, archive, names


def test_input_snapshot_binds_extraction_to_source_archive(tmp_path: Path) -> None:
    inputs, archive, names = make_input_archive(tmp_path)
    snapshot = build_input_snapshot(inputs, archive, expected_count=2)
    assert snapshot.filenames == tuple(names)
    assert snapshot.filenames_sha256 == filenames_digest(names)
    assert snapshot.source_archive_sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert (
        dict(snapshot.input_sha256)[names[0]]
        == hashlib.sha256((inputs / names[0]).read_bytes()).hexdigest()
    )

    atomic_write_png(
        inputs / names[0],
        np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 99, dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="differs from source archive"):
        build_input_snapshot(inputs, archive, expected_count=2)


def test_input_snapshot_rejects_symlink(tmp_path: Path) -> None:
    inputs, archive, names = make_input_archive(tmp_path)
    (inputs / names[-1]).unlink()
    os.symlink(inputs / names[0], inputs / names[-1])
    with pytest.raises(ValueError, match="non-regular entry"):
        build_input_snapshot(inputs, archive, expected_count=2)


def test_board_evidence_and_independent_reassembly_preserve_exact_tiles() -> None:
    image = synthetic_board()
    layout = np.random.default_rng(302).permutation(TILE_COUNT).astype(np.int32)
    raw = assemble_tiles(split_tiles(image)[layout])
    harmonized = _apply_frozen_harmonizers(raw)
    restored = _proper_rgb_nlm_h20_once(harmonized)
    evidence = load_frozen_tail_evidence()
    independently_rebuilt = _independent_raw_assembly(image, layout.tolist())
    assert np.array_equal(independently_rebuilt, raw)
    record = board_attestation(
        filename="img_000001.png",
        input_sha256="1" * 64,
        layout=layout,
        raw=raw,
        harmonized=harmonized,
        restored=restored,
        output_png_sha256="2" * 64,
        tail_evidence=evidence,
    )
    assert record["tile_at_position"] == layout.tolist()
    assert record["raw_assembly_sha256"] == array_sha256(raw)
    assert record["restoration"]["name"] == restoration_name()
    assert record["restoration"]["harmonizers"] == evidence.harmonizers_record()
    assert record["restoration"]["harmonized_array_sha256"] == array_sha256(harmonized)
    assert record["restoration"]["nlm"] == {
        "name": "opencv_fast_nl_means_colored",
        "proper_rgb_bgr_roundtrip": True,
        "h": 20,
        "h_color": 20,
        "template_window_size": 7,
        "search_window_size": 21,
        "passes": 1,
    }

    invalid = layout.copy()
    invalid[-1] = invalid[0]
    with pytest.raises(ValueError, match="exact permutation"):
        board_attestation(
            filename="img_000001.png",
            input_sha256="1" * 64,
            layout=invalid,
            raw=raw,
            harmonized=harmonized,
            restored=restored,
            output_png_sha256="2" * 64,
            tail_evidence=evidence,
        )


def test_independent_frozen_tail_matches_production_and_is_not_configurable() -> None:
    rows, columns = np.indices((IMAGE_SIZE, IMAGE_SIZE))
    image = np.stack(
        (
            rows % 256,
            columns % 256,
            (3 * rows + 5 * columns) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    harmonized = _apply_frozen_harmonizers(image)
    independently_harmonized = _independent_frozen_harmonizers(image)
    assert np.array_equal(independently_harmonized, harmonized)
    production = _proper_rgb_nlm_h20_once(harmonized)
    independently_recomputed = _independent_nlm_h20_once(independently_harmonized)
    assert np.array_equal(independently_recomputed, production)
    assert (FINAL_NLM_H, FINAL_NLM_H_COLOR, FINAL_NLM_PASSES) == (20, 20, 1)
    assert tuple(inspect.signature(predict_frozen_submission).parameters) == ("input_image",)
    production_parameters = set(inspect.signature(run_production_submission).parameters)
    assert production_parameters.isdisjoint(
        {"atlas", "atlas_path", "atlas_weight", "nlm_h", "nlm_passes", "schema_path"}
    )
    validation_parameters = set(inspect.signature(validate_submission).parameters)
    assert validation_parameters == {
        "inputs_dir",
        "source_archive",
        "submission_zip",
        "attestation_path",
    }
    with pytest.raises(TypeError):
        predict_frozen_submission(image, nlm_passes=2)  # type: ignore[call-arg]
    layout = np.random.default_rng(303).permutation(TILE_COUNT).astype(np.int32)
    assert (
        _independent_layout_digest(layout)
        == hashlib.sha256(layout.astype("<i4").tobytes()).hexdigest()
    )


def test_frozen_tail_configs_are_target_blind_and_content_addressed(tmp_path: Path) -> None:
    evidence = load_frozen_tail_evidence()
    assert evidence.rgb_config_sha256 == (
        "4adfd9b614e8556b7de5c1f527d759d15d29c0f74e20aa26ff87900dd773ec9a"
    )
    assert evidence.luma_config_sha256 == (
        "7488cad2ae7cc75792d6ff0ff2ea0a38fa778979083ffd5c161c857b68fd550f"
    )
    assert evidence.pipeline_config_sha256 == (
        "7609987c9d9b817c48cc893d58f2a77fc37b8c1a2911574bed0013e01e38a042"
    )

    rgb = json.loads((PROJECT_ROOT / "configs/postassembly_rgb_offset_v1.json").read_text())
    rgb["target_access"] = True
    tampered = tmp_path / "tampered-rgb.json"
    tampered.write_text(json.dumps(rgb))
    with pytest.raises(ValueError, match="target-blind"):
        load_frozen_tail_evidence(rgb_config_path=tampered)


def test_attestation_conforms_to_checked_in_schema() -> None:
    names = tuple(f"img_{index:06d}.png" for index in range(700))
    hashes = tuple((name, f"{index:064x}") for index, name in enumerate(names))
    snapshot = InputSnapshot(
        source_archive_sha256="a" * 64,
        filenames=names,
        filenames_sha256=filenames_digest(names),
        input_sha256=hashes,
    )
    layout = np.arange(TILE_COUNT, dtype=np.int32)
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    evidence = load_frozen_tail_evidence()
    runtime_manifest = build_runtime_manifest()
    template = board_attestation(
        filename=names[0],
        input_sha256=hashes[0][1],
        layout=layout,
        raw=image,
        harmonized=image,
        restored=image,
        output_png_sha256="b" * 64,
        tail_evidence=evidence,
    )
    records = []
    for name, input_hash in hashes:
        record = copy.copy(template)
        record["filename"] = name
        record["input_sha256"] = input_hash
        records.append(record)
    payload = build_attestation(
        snapshot=snapshot,
        archive_sha256="c" * 64,
        per_board=records,
        method=_method_declaration(evidence, runtime_manifest),
        runtime_manifest=runtime_manifest,
    )
    schema = json.loads((PROJECT_ROOT / "configs/submission-compliance.schema.json").read_text())
    Draft202012Validator(schema).validate(payload)
    assert payload["status"] == METHOD_STATUS
    assert payload["scope"] == PROOF_SCOPE
    assert payload["correct_hidden_layout_proven"] is False
    assert "does not prove the hidden ground-truth permutation" in payload["proof_limitation"]
    assert payload["policy"] == EXPECTED_POLICY
    forbidden = copy.deepcopy(payload)
    forbidden["per_board"][0]["restoration"]["nlm"]["passes"] = 2
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(forbidden)


def test_zip_inspector_requires_exact_root_roster_and_rgb_pngs(tmp_path: Path) -> None:
    inputs, archive_path, names = make_input_archive(tmp_path)
    del inputs
    with zipfile.ZipFile(archive_path) as archive:
        members = _inspect_submission_members(archive, expected_names=names)
        assert list(members) == names

    nested = tmp_path / "nested.zip"
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("folder/img_000000.png", b"not a png")
    with zipfile.ZipFile(nested) as archive, pytest.raises(ValueError, match="roster/order"):
        _inspect_submission_members(archive, expected_names=names[:1])


def test_output_guards_reject_overlap_and_existing_artifacts(tmp_path: Path) -> None:
    inputs, source, _ = make_input_archive(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        guard_artifact_paths(
            inputs_dir=inputs,
            source_archive=source,
            output_dir=inputs / "predictions",
            output_zip=tmp_path / "submission.zip",
            attestation_path=tmp_path / "attestation.json",
        )

    existing = tmp_path / "existing.zip"
    existing.write_bytes(b"owned")
    with pytest.raises(FileExistsError, match="overwrite"):
        guard_artifact_paths(
            inputs_dir=inputs,
            source_archive=source,
            output_dir=tmp_path / "predictions",
            output_zip=existing,
            attestation_path=tmp_path / "attestation.json",
        )


def test_independent_validator_recomputes_complete_700_file_bundle(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    names = [f"img_{index:06d}.png" for index in range(700)]
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    payload = png_bytes(image)
    payload_hash = hashlib.sha256(payload).hexdigest()
    for name in names:
        (inputs / name).write_bytes(payload)
    source_archive = tmp_path / "source.zip"
    submission_zip = tmp_path / "submission.zip"
    deterministic_submission_zip(inputs, names, source_archive)
    deterministic_submission_zip(inputs, names, submission_zip)
    snapshot = build_input_snapshot(inputs, source_archive)

    prediction = predict_frozen_submission(image)
    assert np.array_equal(prediction.raw, image)
    assert np.array_equal(prediction.harmonized, image)
    assert np.array_equal(prediction.restored, image)
    assert np.array_equal(
        _independent_no_atlas_buddies96_layout(image),
        prediction.layout,
    )
    evidence = load_frozen_tail_evidence()
    runtime_manifest = build_runtime_manifest()
    template = board_attestation(
        filename=names[0],
        input_sha256=payload_hash,
        layout=prediction.layout,
        raw=prediction.raw,
        harmonized=prediction.harmonized,
        restored=prediction.restored,
        output_png_sha256=payload_hash,
        tail_evidence=evidence,
    )
    records = []
    for name in names:
        record = copy.copy(template)
        record["filename"] = name
        records.append(record)
    attestation = build_attestation(
        snapshot=snapshot,
        archive_sha256=hashlib.sha256(submission_zip.read_bytes()).hexdigest(),
        per_board=records,
        method=_method_declaration(evidence, runtime_manifest),
        runtime_manifest=runtime_manifest,
    )
    attestation_path = tmp_path / "attestation.json"
    atomic_write_json(attestation_path, attestation)
    report = _validate_submission_against_snapshot(
        snapshot=snapshot,
        inputs_dir=inputs,
        submission_zip=submission_zip,
        attestation_path=attestation_path,
    )
    assert report["status"] == METHOD_STATUS
    assert report["file_count"] == 700
    assert report["all_raw_assemblies_recomputed"]
    assert report["all_solver_layouts_recomputed"]
    assert report["restoration_recomputed"]
    assert report["unique_input_derivations_recomputed"] == 1
    assert report["correct_hidden_layout_proven"] is False

    wrong_layout = np.roll(prediction.layout, 1)
    mutated = copy.deepcopy(attestation)
    mutated["per_board"][0]["tile_at_position"] = wrong_layout.tolist()
    mutated["per_board"][0]["layout_sha256"] = _independent_layout_digest(wrong_layout)
    mutated_path = tmp_path / "wrong-layout-attestation.json"
    atomic_write_json(mutated_path, mutated)
    with pytest.raises(ValueError, match="differs from frozen solver layout"):
        _validate_submission_against_snapshot(
            snapshot=snapshot,
            inputs_dir=inputs,
            submission_zip=submission_zip,
            attestation_path=mutated_path,
        )

    with pytest.raises(ValueError, match="pinned official test.zip"):
        validate_submission(
            inputs_dir=inputs,
            source_archive=source_archive,
            submission_zip=submission_zip,
            attestation_path=attestation_path,
        )
    with pytest.raises(TypeError):
        validate_submission(  # type: ignore[call-arg]
            inputs_dir=inputs,
            source_archive=source_archive,
            submission_zip=submission_zip,
            attestation_path=attestation_path,
            recompute_restoration=False,
        )


def test_official_archive_schema_and_runtime_are_exactly_pinned(tmp_path: Path) -> None:
    assert OFFICIAL_TEST_ARCHIVE_SHA256 == (
        "62d365c45fe85c3da06e96f83390e7bb056935036a9b5dee7a99d32f11483c89"
    )
    assert OFFICIAL_FILENAMES_SHA256 == (
        "312e8c46b2ccfa27e525d607d046d0e3676688f8c71533b8498c377d71805376"
    )
    schema_path = PROJECT_ROOT / "configs/submission-compliance.schema.json"
    assert hashlib.sha256(schema_path.read_bytes()).hexdigest() == PINNED_SCHEMA_SHA256
    _load_compliance_schema()

    schema = json.loads(schema_path.read_text())
    schema["description"] = "alternate schema must be rejected"
    tampered_schema = tmp_path / "submission-compliance.schema.json"
    tampered_schema.write_text(json.dumps(schema))
    with pytest.raises(ValueError, match="schema hash differs"):
        _load_compliance_schema(tampered_schema)

    manifest = build_runtime_manifest()
    assert set(manifest) == {"files", "versions", "digest_sha256"}
    assert manifest["files"]["configs/submission-compliance.schema.json"] == (PINNED_SCHEMA_SHA256)
    assert set(manifest["versions"]) == {
        "python",
        "numpy",
        "opencv",
        "scipy",
        "pillow",
        "scikit_image",
        "scikit_learn",
    }
