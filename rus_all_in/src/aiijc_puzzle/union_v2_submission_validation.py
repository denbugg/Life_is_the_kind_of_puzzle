"""Independent full-roster validator for the frozen Union-v2 submission.

Unlike the packager, this module ignores its assembled raw/output arrays.  It
re-runs the public SHA-locked Union-v2 adapter, independently reconstructs the
declared tile permutation, independently composes both historical harmonizers
and the single colored-NLM h20 pass, and compares those arrays with the root-only
ZIP, prediction directory, board records and attestation.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from aiijc_puzzle.compliant_submission import (
    EXPECTED_TEST_FILES,
    InputSnapshot,
    array_sha256,
    atomic_write_json,
    build_official_input_snapshot,
    decode_rgb_png,
    load_rgb_png,
)
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.postassembly_harmonizer import (
    DEFAULT_LUMINANCE_GAIN_CONFIG,
    DEFAULT_SEAM_GRAPH_CONFIG,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.protocol import (
    GRID_SIZE,
    IMAGE_SIZE,
    RGB_CHANNELS,
    TILE_COUNT,
    TILE_SIZE,
    assemble_tiles,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_pixel_tails import NLM_H, historical_rgb_luma_nlm_h20_contract
from aiijc_puzzle.union_v2_submission import (
    ATTESTATION_SCHEMA,
    DEFAULT_CONFIG,
    EXPECTED_POLICY,
    METHOD_STATUS,
    PROOF_LIMITATION,
    PROOF_SCOPE,
    RECORD_SCHEMA,
    FrozenUnionSubmissionConfig,
    _absolute,
    _digest_json,
    _load_json,
    _output_state,
    _require_directory,
    _require_regular_file,
    _run_identity,
    _stable_record,
    build_pipeline_contract,
    load_union_submission_config,
    load_union_v2_engine,
)

VALIDATION_SCHEMA = "aiijc-union-v2-submission-validation-progress-v1"


def _strict_layout(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError("attested tile layout must contain integers")
    layout = np.ascontiguousarray(array, dtype=np.int32)
    if layout.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(layout), np.arange(TILE_COUNT, dtype=np.int32)
    ):
        raise ValueError("attested tile layout is not one strict 0..575 permutation")
    return layout


def _independent_layout_digest(layout: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(layout, dtype="<i4").tobytes()
    ).hexdigest()


def _independent_raw_assembly(image: np.ndarray, layout: np.ndarray) -> np.ndarray:
    """Reassemble without calling the production assembly/audit helper."""

    value = np.asarray(image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"independent assembly requires uint8 RGB {expected}")
    order = _strict_layout(layout)
    tiles = (
        value.reshape(GRID_SIZE, TILE_SIZE, GRID_SIZE, TILE_SIZE, RGB_CHANNELS)
        .transpose(0, 2, 1, 3, 4)
        .reshape(TILE_COUNT, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
    )
    selected = tiles[order]
    raw = (
        selected.reshape(GRID_SIZE, GRID_SIZE, TILE_SIZE, TILE_SIZE, RGB_CHANNELS)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    )
    # Byte-multiset equality independently establishes corresponding-input
    # identity even if a future production audit implementation drifts.
    source_hashes = sorted(
        hashlib.sha256(np.ascontiguousarray(tile).tobytes()).digest()
        for tile in tiles
    )
    raw_hashes = sorted(
        hashlib.sha256(np.ascontiguousarray(tile).tobytes()).digest()
        for tile in split_tiles(raw)
    )
    if source_hashes != raw_hashes:
        raise RuntimeError("independent raw assembly changed the input tile multiset")
    return np.ascontiguousarray(raw)


def _independent_tail(raw: np.ndarray) -> np.ndarray:
    """Compose the frozen tail without calling the production tail function."""

    value = np.asarray(raw)
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"independent tail requires uint8 RGB {expected}")
    tiles = split_tiles(value)
    offsets, _ = seam_graph_rgb_offsets(tiles, DEFAULT_SEAM_GRAPH_CONFIG)
    rgb_corrected = apply_rgb_offsets(tiles, offsets)
    gains, _ = seam_graph_luminance_gains(
        rgb_corrected,
        DEFAULT_LUMINANCE_GAIN_CONFIG,
    )
    harmonized = assemble_tiles(apply_luminance_gains(rgb_corrected, gains))
    return np.ascontiguousarray(apply_nlm_color(harmonized, h=NLM_H).image)


def _load_attestation(
    path: Path,
    config: FrozenUnionSubmissionConfig,
) -> dict[str, Any]:
    attestation_path = _require_regular_file(path)
    payload = _load_json(attestation_path)
    schema_path = config.artifacts["attestation_schema"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    if schema.get("properties", {}).get("schema", {}).get("const") != ATTESTATION_SCHEMA:
        raise ValueError("attestation schema does not pin the Union-v2 contract")
    return payload


def _zip_members(
    archive: zipfile.ZipFile,
    expected_names: Sequence[str],
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if names != list(expected_names) or len(names) != len(set(names)):
        raise ValueError("submission ZIP is not the exact sorted official roster")
    result: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if (
            info.is_dir()
            or Path(info.filename).name != info.filename
            or not info.filename.endswith(".png")
            or info.flag_bits & 0x1
            or info.compress_type != zipfile.ZIP_DEFLATED
            or unix_mode != 0o100644
            or info.date_time != (1980, 1, 1, 0, 0, 0)
            or info.file_size <= 0
        ):
            raise ValueError(f"submission ZIP contains a foreign/unsafe member: {info.filename}")
        result[info.filename] = info
    return result


def _attestation_identity(
    *,
    snapshot: InputSnapshot,
    pipeline: Mapping[str, Any],
    submission_zip_sha256: str,
    attestation_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "pipeline_digest": pipeline["pipeline_digest"],
        "source_archive_sha256": snapshot.source_archive_sha256,
        "filenames_sha256": snapshot.filenames_sha256,
        "submission_zip_sha256": submission_zip_sha256,
        "attestation_sha256": attestation_sha256,
    }
    payload["identity_digest"] = _digest_json(payload)
    return payload


def _receipt(
    *,
    record: Mapping[str, Any],
    input_sha256: str,
    zip_payload_sha256: str,
    independently_predicted_layout: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "filename": record["filename"],
        "input_sha256": input_sha256,
        "zip_payload_sha256": zip_payload_sha256,
        "record_digest": _digest_json(_stable_record(record)),
        "layout_sha256": record["layout"]["sha256_int32_le"],
        "raw_assembly_sha256": record["raw_assembly"]["array_sha256"],
        "output_array_sha256": record["pixel_tail"]["output_array_sha256"],
        "independently_predicted_layout": independently_predicted_layout,
        "tail_independently_recomputed": True,
    }
    payload["receipt_digest"] = _digest_json(payload)
    return payload


def _validate_receipt(value: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(value, Mapping):
        return False
    receipt = dict(value)
    digest = receipt.pop("receipt_digest", None)
    if digest != _digest_json(receipt):
        return False
    return receipt.get("record_digest") == _digest_json(_stable_record(expected)) and bool(
        receipt.get("independently_predicted_layout")
    )


def _validate_declared_record(
    *,
    record: Mapping[str, Any],
    filename: str,
    input_sha256: str,
    input_image: np.ndarray,
    predicted_layout: np.ndarray,
    raw: np.ndarray,
    output: np.ndarray,
    output_payload: bytes,
    directory_payload: bytes,
    pipeline: Mapping[str, Any],
) -> None:
    if record.get("schema") != RECORD_SCHEMA or record.get("filename") != filename:
        raise ValueError(f"attestation board identity mismatch: {filename}")
    if record.get("input") != {
        "file_sha256": input_sha256,
        "decoded_rgb_sha256": array_sha256(input_image),
    }:
        raise ValueError(f"attestation input identity mismatch: {filename}")
    if record.get("lineage") != {"pipeline_digest": pipeline["pipeline_digest"]}:
        raise ValueError(f"attestation pipeline mismatch: {filename}")
    layout_record = record.get("layout")
    raw_record = record.get("raw_assembly")
    tail_record = record.get("pixel_tail")
    if not all(isinstance(value, Mapping) for value in (layout_record, raw_record, tail_record)):
        raise ValueError(f"attestation board structure is malformed: {filename}")
    declared = _strict_layout(layout_record.get("tile_at_position"))
    if not np.array_equal(declared, predicted_layout):
        raise ValueError(f"attested layout differs from independent Union-v2 rerun: {filename}")
    if layout_record.get("sha256_int32_le") != _independent_layout_digest(declared):
        raise ValueError(f"attested layout hash mismatch: {filename}")
    audit = raw_record.get("audit")
    if (
        raw_record.get("array_sha256") != array_sha256(raw)
        or raw_record.get("audited_before_restoration") is not True
        or not isinstance(audit, Mapping)
        or audit.get("passed") is not True
    ):
        raise ValueError(f"attested pre-tail raw audit/hash mismatch: {filename}")
    if (
        tail_record.get("contract") != historical_rgb_luma_nlm_h20_contract()
        or tail_record.get("layout_changed") is not False
        or tail_record.get("output_array_sha256") != array_sha256(output)
    ):
        raise ValueError(f"attested historical tail mismatch: {filename}")
    if output_payload != directory_payload:
        raise ValueError(f"ZIP and prediction-directory PNG bytes differ: {filename}")
    png_sha = hashlib.sha256(output_payload).hexdigest()
    if record.get("output_png_sha256") != png_sha:
        raise ValueError(f"attested output PNG hash mismatch: {filename}")
    decoded = decode_rgb_png(output_payload, context=f"Union-v2 ZIP:{filename}")
    if not np.array_equal(decoded, output):
        raise ValueError(f"output differs from independent historical tail: {filename}")


def validate_union_v2_submission(
    *,
    source_dir: Path,
    source_archive: Path,
    output_dir: Path,
    submission_zip: Path,
    attestation_path: Path,
    validation_state_path: Path,
    config_path: Path = DEFAULT_CONFIG,
    device_name: str = "cpu",
    allow_nondeterministic_mps: bool = False,
    force_full_layout_recompute: bool = False,
) -> dict[str, Any]:
    """Validate/resume the exact official 700-board Union-v2 bundle."""

    source = _require_directory(source_dir)
    output_root = _require_directory(output_dir)
    records_dir = _require_directory(output_root / "records")
    archive_path = _require_regular_file(submission_zip)
    attestation_file = _require_regular_file(attestation_path)
    state_path = _absolute(validation_state_path)
    if (
        state_path in {archive_path, attestation_file}
        or state_path.is_relative_to(output_root)
        or state_path.is_relative_to(source)
        or source.is_relative_to(state_path)
    ):
        raise ValueError("validation progress must be a separate artifact")
    snapshot = build_official_input_snapshot(source, source_archive)
    if snapshot.file_count != EXPECTED_TEST_FILES:
        raise ValueError("Union-v2 validator requires the full official 700-board roster")
    config = load_union_submission_config(config_path)
    engine = load_union_v2_engine(
        config,
        device_name=device_name,
        allow_nondeterministic_mps=allow_nondeterministic_mps,
    )
    pipeline = build_pipeline_contract(
        config,
        engine,
        allow_nondeterministic_mps=allow_nondeterministic_mps,
    )
    attestation = _load_attestation(attestation_file, config)
    if (
        attestation.get("status") != METHOD_STATUS
        or attestation.get("scope") != PROOF_SCOPE
        or attestation.get("correct_hidden_layout_proven") is not False
        or attestation.get("proof_limitation") != PROOF_LIMITATION
        or attestation.get("policy") != EXPECTED_POLICY
        or attestation.get("input_snapshot") != snapshot.attestation_record()
        or attestation.get("pipeline") != pipeline
    ):
        raise ValueError("attestation overclaims or differs from the frozen contract")
    archive_sha = sha256_file(archive_path)
    attestation_sha = sha256_file(attestation_file)
    archive_record = attestation.get("archive")
    if not isinstance(archive_record, Mapping) or archive_record.get("sha256") != archive_sha:
        raise ValueError("attested archive SHA-256 mismatch")
    if archive_record.get("filenames") != list(snapshot.filenames):
        raise ValueError("attested archive roster differs from official input")
    board_records = attestation.get("per_board")
    if not isinstance(board_records, list) or [
        record.get("filename") if isinstance(record, Mapping) else None
        for record in board_records
    ] != list(snapshot.filenames):
        raise ValueError("attested per-board roster/order differs from official input")
    _output_state(output_root, records_dir, snapshot.filenames, require_complete=True)
    run = _load_json(output_root / "run.json")
    run_identity = _run_identity(snapshot, pipeline, source)
    if any(run.get(key) != value for key, value in run_identity.items()):
        raise ValueError("prediction run.json identity differs from the validated bundle")

    identity = _attestation_identity(
        snapshot=snapshot,
        pipeline=pipeline,
        submission_zip_sha256=archive_sha,
        attestation_sha256=attestation_sha,
    )
    receipts: dict[str, Any] = {}
    if state_path.exists() and not force_full_layout_recompute:
        state = _load_json(state_path)
        if state.get("schema") != VALIDATION_SCHEMA or any(
            state.get(key) != value for key, value in identity.items()
        ):
            raise ValueError("validation progress belongs to another immutable bundle")
        raw_receipts = state.get("receipts")
        if not isinstance(raw_receipts, Mapping):
            raise ValueError("validation progress receipt mapping is malformed")
        receipts = dict(raw_receipts)
        if any(name not in snapshot.filenames for name in receipts):
            raise ValueError("validation progress contains a foreign filename")
    progress: dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        **identity,
        "status": "IN_PROGRESS",
        "receipts": receipts,
    }
    atomic_write_json(state_path, progress)
    input_hashes = snapshot.hashes_by_name
    recomputed_layouts = 0
    resumed_layouts = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = _zip_members(archive, snapshot.filenames)
        for index, raw_record in enumerate(board_records, start=1):
            record = dict(raw_record)
            name = str(record["filename"])
            expected_input_hash = input_hashes[name]
            image = load_rgb_png(source / name, expected_sha256=expected_input_hash)
            output_payload = archive.read(members[name])
            directory_payload = _require_regular_file(output_root / name).read_bytes()
            existing_receipt = receipts.get(name)
            can_resume = (
                not force_full_layout_recompute
                and _validate_receipt(existing_receipt, record)
                and existing_receipt.get("input_sha256") == expected_input_hash
                and existing_receipt.get("zip_payload_sha256")
                == hashlib.sha256(output_payload).hexdigest()
            )
            if can_resume:
                predicted_layout = _strict_layout(record["layout"]["tile_at_position"])
                resumed_layouts += 1
            else:
                predicted_layout, diagnostics = engine.predict_layout(image)
                if diagnostics.get("selected_variant") != "raw-twin-union-v2":
                    raise RuntimeError("independent Union-v2 validator observed fallback")
                recomputed_layouts += 1
            raw = _independent_raw_assembly(image, predicted_layout)
            output = _independent_tail(raw)
            _validate_declared_record(
                record=record,
                filename=name,
                input_sha256=expected_input_hash,
                input_image=image,
                predicted_layout=predicted_layout,
                raw=raw,
                output=output,
                output_payload=output_payload,
                directory_payload=directory_payload,
                pipeline=pipeline,
            )
            disk_record = _load_json(records_dir / f"{name}.json")
            if _stable_record(disk_record) != _stable_record(record):
                raise ValueError(f"prediction record differs from attestation: {name}")
            receipts[name] = _receipt(
                record=record,
                input_sha256=expected_input_hash,
                zip_payload_sha256=hashlib.sha256(output_payload).hexdigest(),
                independently_predicted_layout=True,
            )
            progress["receipts"] = receipts
            atomic_write_json(state_path, progress)
            print(
                json.dumps(
                    {
                        "event": "union_v2_validation_board",
                        "index": index,
                        "count": snapshot.file_count,
                        "filename": name,
                        "layout_recomputed": not can_resume,
                        "tail_recomputed": True,
                    }
                ),
                flush=True,
            )
    if len(receipts) != snapshot.file_count:
        raise RuntimeError("validation progress is incomplete after full roster traversal")
    progress["status"] = "COMPLETE"
    progress["layouts_recomputed_this_invocation"] = recomputed_layouts
    progress["layouts_resumed_from_verified_receipts"] = resumed_layouts
    progress["tails_recomputed_this_invocation"] = snapshot.file_count
    atomic_write_json(state_path, progress)
    return {
        "status": "PASS",
        "scope": PROOF_SCOPE,
        "correct_hidden_layout_proven": False,
        "proof_limitation": PROOF_LIMITATION,
        "file_count": snapshot.file_count,
        "source_archive_sha256": snapshot.source_archive_sha256,
        "filenames_sha256": snapshot.filenames_sha256,
        "submission_zip_sha256": archive_sha,
        "attestation_sha256": attestation_sha,
        "pipeline_digest": pipeline["pipeline_digest"],
        "all_layouts_independently_recomputed_across_validation_state": True,
        "layouts_recomputed_this_invocation": recomputed_layouts,
        "layouts_resumed_from_verified_receipts": resumed_layouts,
        "all_tails_independently_recomputed_this_invocation": True,
        "all_layouts_strict_original_upright_permutations": True,
        "foreign_files_rejected": True,
        "validation_state": str(state_path),
    }


__all__ = [
    "VALIDATION_SCHEMA",
    "validate_union_v2_submission",
]
