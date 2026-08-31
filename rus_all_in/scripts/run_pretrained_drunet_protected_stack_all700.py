#!/usr/bin/env python3
"""Commit and measure the already-frozen legal DRUNet-protected stack on 700 boards.

This runner is deliberately not an arm-selection harness.  It renders exactly
the previously preregistered D arm, plus the already-frozen original-h28 image
used only for target-free preservation diagnostics.  Prediction preparation
never opens a target.  Scoring is refused until every board is immutable and
bound by a stage commitment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.legacy_upgrade import directional_scores, layout_digest, solve_buddies
from aiijc_puzzle.nlm_luma_chroma import safety_summary
from aiijc_puzzle.postassembly_harmonizer import (
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.pretrained_drunet_protected_stack import (
    ARM_COMBINED,
    ARM_ORIGINAL_H28,
    DRUNET_SIGMA,
    MODEL_BATCH_SIZE,
    image_digest,
    render_combined_arms,
)
from aiijc_puzzle.pretrained_tile_denoiser import load_drunet_color
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    IMAGE_SIZE,
    TILE_COUNT,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/pretrained_drunet_protected_stack_all700_measurement_v1.json"
CONFIG_SIDECAR = Path(f"{CONFIG}.sha256")
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
TARGETS = PROJECT_ROOT / "data/raw/train/targets"
ASSET_ROOT = PROJECT_ROOT / "artifacts/pretrained-denoisers/kair-fc1732f"
CHECKPOINT = ASSET_ROOT / "drunet_color.pth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/pretrained-drunet-protected-stack/all700-measurement-v1"
STAGE_ROOTS = {
    "calibration": OUTPUT_ROOT / "calibration700",
    "holdout": OUTPUT_ROOT / "holdout700",
}

EXPECTED_MANIFEST_SHA256 = "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da"
EXPECTED_PROTOCOL_DIGEST = "2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4"
EXPECTED_ASSET_SHA256 = {
    "LICENSE": "448e69b705d64f21bf8cb86562301e0edd99ac79026064ddd75af8242b067be5",
    "drunet_color.pth": "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4",
    "models/basicblock.py": "48406db8867394ac5ae233ebeec7711ac10acfc3a6bbf0072c33aa77d659b6fd",
    "models/network_unet.py": "8043b6350f1589d5f08892e3be0b4d12c5a502058014285107b7360696d12bf5",
}
MODEL_PARAMETER_COUNT = 32_640_960
EDGE_BUDGET = 96
BOARD_COUNT = 700
FOLD_COUNT = 10
FOLD_SIZE = 70
QUANTILES = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "score"), required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_ROOTS), default="calibration")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def require_readonly(path: Path) -> None:
    if path.stat().st_mode & WRITE_BITS:
        raise PermissionError(f"integrity-bound artifact is writable: {path}")


def write_bytes_exclusive_readonly(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o444)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    require_readonly(path)
    return hashlib.sha256(payload).hexdigest()


def write_json_exclusive_readonly(path: Path, payload: Mapping[str, Any]) -> str:
    return write_bytes_exclusive_readonly(path, canonical_json_bytes(payload))


def png_bytes(image: np.ndarray) -> bytes:
    if image.dtype != np.uint8 or image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("prediction must be uint8 RGB 480x480")
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def load_rgb_verified(path: Path, expected_hash: str) -> np.ndarray:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or image.size
            != (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def names_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode("utf-8")
    ).hexdigest()


def input_roster_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(f"{record['filename']} {record['input_sha256']}" for record in records).encode(
            "utf-8"
        )
    ).hexdigest()


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/aiijc_puzzle/pretrained_drunet_protected_stack.py",
        PROJECT_ROOT / "src/aiijc_puzzle/pretrained_tile_denoiser.py",
        PROJECT_ROOT / "src/aiijc_puzzle/edge_protected_nlm.py",
        PROJECT_ROOT / "src/aiijc_puzzle/nlm_luma_chroma.py",
        PROJECT_ROOT / "src/aiijc_puzzle/legacy_upgrade.py",
        PROJECT_ROOT / "src/aiijc_puzzle/postassembly_harmonizer.py",
        PROJECT_ROOT / "src/aiijc_puzzle/protocol.py",
        PROJECT_ROOT / "src/aiijc_puzzle/compliant_atlas_decoder.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def verify_assets() -> dict[str, str]:
    observed = {relative: sha256_file(ASSET_ROOT / relative) for relative in EXPECTED_ASSET_SHA256}
    if observed != EXPECTED_ASSET_SHA256:
        raise ValueError(f"official KAIR assets changed: {observed}")
    license_text = (ASSET_ROOT / "LICENSE").read_text(encoding="utf-8")
    if "MIT License" not in license_text or "Copyright (c) 2019 Kai Zhang" not in license_text:
        raise ValueError("official KAIR MIT license text changed")
    return observed


def load_manifest() -> dict[str, Any]:
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("validation manifest file changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest self-digest is invalid")
    if manifest.get("protocol_digest") != EXPECTED_PROTOCOL_DIGEST:
        raise ValueError("validation protocol changed")
    return manifest


def select_all_records(manifest: Mapping[str, Any], stage: str) -> tuple[Mapping[str, Any], ...]:
    records = tuple(
        select_manifest_records(
            manifest,
            stage,
            limit=BOARD_COUNT,
            seed=EXPERIMENT_SUBSET_SEED,
            namespace=EXPERIMENT_SUBSET_NAMESPACE,
        )
    )
    if len(records) != BOARD_COUNT:
        raise RuntimeError("selected record count drifted")
    return records


def config_sha256() -> str:
    require_readonly(CONFIG)
    require_readonly(CONFIG_SIDECAR)
    observed = sha256_file(CONFIG)
    if CONFIG_SIDECAR.read_text(encoding="utf-8").split()[0] != observed:
        raise ValueError("measurement config sidecar changed")
    return observed


def calibration_report_allows_holdout() -> bool:
    report_path = STAGE_ROOTS["calibration"] / "report.json"
    if not report_path.is_file():
        return False
    require_readonly(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return bool(
        report.get("schema") == "aiijc-pretrained-drunet-protected-all700-report-v1"
        and report.get("stage") == "calibration"
        and report.get("broad_completion_gate", {}).get("passed") is True
        and report.get("config_sha256") == config_sha256()
    )


def load_context(
    stage: str,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...], str]:
    digest = config_sha256()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("schema") != "aiijc-pretrained-drunet-protected-all700-measurement-v1":
        raise ValueError("measurement config schema changed")
    if config.get("source_sha256") != source_hashes():
        raise ValueError("measurement source differs from immutable config")
    if config.get("asset_sha256") != verify_assets():
        raise ValueError("measurement model assets differ from immutable config")
    if stage == "holdout" and not calibration_report_allows_holdout():
        raise RuntimeError("holdout is fail-closed: calibration700 broad gate did not pass")
    records = select_all_records(load_manifest(), stage)
    expected = config["data"][stage]
    if names_digest(records) != expected["filenames_sha256"]:
        raise ValueError(f"{stage} filename roster differs from immutable config")
    if input_roster_digest(records) != expected["input_roster_sha256"]:
        raise ValueError(f"{stage} input roster differs from immutable config")
    return config, records, digest


def apply_rgb_luma(ordered_tiles: np.ndarray) -> np.ndarray:
    offsets, _ = seam_graph_rgb_offsets(ordered_tiles, SeamGraphConfig())
    rgb_tiles = apply_rgb_offsets(ordered_tiles, offsets)
    gains, _ = seam_graph_luminance_gains(rgb_tiles, LuminanceGainConfig())
    return apply_luminance_gains(rgb_tiles, gains)


def infer_board(dirty: np.ndarray, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    input_tiles = split_tiles(dirty)
    right, down = directional_scores(input_tiles, views=("bilateral",))["bilateral"]
    solved = solve_buddies(right, down, max_edges=EDGE_BUDGET)
    layout = np.asarray(solved.layout, dtype=np.int32)
    ordered = np.ascontiguousarray(input_tiles[layout])
    raw = assemble_tiles(ordered)
    audit = audit_raw_permutation(dirty, raw, layout, restoration_applied_after_audit=True)
    if not audit.passed:
        raise RuntimeError("strict raw permutation audit failed")
    harmonized = apply_rgb_luma(ordered)
    predictions, diagnostics = render_combined_arms(model, harmonized, device=device)
    reference = predictions[ARM_ORIGINAL_H28]
    candidate = predictions[ARM_COMBINED]
    if np.array_equal(reference, candidate):
        raise RuntimeError("candidate unexpectedly equals safety reference")
    return {
        "layout": layout,
        "raw": raw,
        "audit": audit.as_dict(),
        "solver": solved.solver,
        "objective": float(solved.objective),
        "reference": reference,
        "candidate": candidate,
        "diagnostics": diagnostics,
    }


def board_safety(result: Mapping[str, Any]) -> dict[str, Any]:
    reference = result["reference"]
    candidate = result["candidate"]
    structures = result["diagnostics"]["structure"]
    reference_float = reference.astype(np.float64)
    candidate_float = candidate.astype(np.float64)
    shift = candidate_float.mean(axis=(0, 1)) - reference_float.mean(axis=(0, 1))
    return {
        "reference_structure": structures[ARM_ORIGINAL_H28],
        "candidate_structure": structures[ARM_COMBINED],
        "protected_fraction": result["diagnostics"]["mask"]["binary_dilated_protected_fraction"],
        "maximum_abs_rgb_mean_shift_vs_reference": float(np.max(np.abs(shift))),
        "global_rgb_std_ratio_vs_reference": float(candidate_float.std() / reference_float.std()),
        "mean_abs_pixel_change_vs_reference": float(
            np.abs(candidate.astype(np.int16) - reference.astype(np.int16)).mean()
        ),
        "clipped_fraction_increase_vs_reference": float(
            structures[ARM_COMBINED]["clipped_fraction"]
            - structures[ARM_ORIGINAL_H28]["clipped_fraction"]
        ),
    }


def stage_board_directory(stage: str, filename: str) -> Path:
    return STAGE_ROOTS[stage] / "boards" / Path(filename).stem


def load_board_record(
    stage: str,
    record: Mapping[str, Any],
    expected_config_sha256: str,
    *,
    verify_input_provenance: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    directory = stage_board_directory(stage, str(record["filename"]))
    metadata_path = directory / "record.json"
    require_readonly(directory)
    require_readonly(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("filename") != record["filename"]
        or metadata.get("input_sha256") != record["input_sha256"]
        or metadata.get("config_sha256") != expected_config_sha256
        or metadata.get("stage") != stage
    ):
        raise ValueError(f"frozen board binding changed: {directory}")
    images: dict[str, np.ndarray] = {}
    for key in ("safety_reference", "candidate"):
        image_record = metadata["images"][key]
        path = directory / image_record["filename"]
        require_readonly(path)
        if sha256_file(path) != image_record["png_sha256"]:
            raise ValueError(f"frozen PNG changed: {path}")
        image = load_rgb_verified(path, image_record["png_sha256"])
        if image_digest(image) != image_record["pixel_sha256"]:
            raise ValueError(f"frozen pixels changed: {path}")
        images[key] = image
    if verify_input_provenance:
        dirty = load_rgb_verified(INPUTS / record["filename"], record["input_sha256"])
        layout = np.asarray(metadata["layout"], dtype=np.int32)
        if layout.shape != (TILE_COUNT,) or layout_digest(layout) != metadata["layout_sha256"]:
            raise ValueError(f"frozen layout changed: {directory}")
        raw = assemble_tiles(split_tiles(dirty)[layout])
        audit = audit_raw_permutation(dirty, raw, layout, restoration_applied_after_audit=True)
        if not audit.passed or audit.as_dict() != metadata["raw_permutation_audit"]:
            raise ValueError(f"full provenance audit failed: {directory}")
        if image_digest(raw) != metadata["raw_pixel_sha256"]:
            raise ValueError(f"frozen raw reassembly hash changed: {directory}")
    return metadata, images["safety_reference"], images["candidate"]


def write_board(
    stage: str,
    record: Mapping[str, Any],
    result: Mapping[str, Any],
    expected_config_sha256: str,
) -> None:
    final_directory = stage_board_directory(stage, str(record["filename"]))
    boards_root = final_directory.parent
    boards_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final_directory.name}.", dir=boards_root))
    try:
        images: dict[str, Any] = {}
        for key, filename, image in (
            ("safety_reference", "B_original_h28.png", result["reference"]),
            ("candidate", "D_frozen_stack.png", result["candidate"]),
        ):
            payload = png_bytes(image)
            path = temporary / filename
            path.write_bytes(payload)
            os.chmod(path, 0o444)
            images[key] = {
                "filename": filename,
                "png_sha256": hashlib.sha256(payload).hexdigest(),
                "pixel_sha256": image_digest(image),
            }
        metadata = {
            "schema": "aiijc-pretrained-drunet-protected-all700-board-v1",
            "status": "target_blind_prediction_frozen",
            "stage": stage,
            "filename": record["filename"],
            "input_sha256": record["input_sha256"],
            "config_sha256": expected_config_sha256,
            "layout": np.asarray(result["layout"], dtype=np.int32).tolist(),
            "layout_sha256": layout_digest(result["layout"]),
            "raw_pixel_sha256": image_digest(result["raw"]),
            "raw_permutation_audit": result["audit"],
            "solver": result["solver"],
            "objective": result["objective"],
            "images": images,
            "safety": board_safety(result),
            "drunet": result["diagnostics"]["drunet"],
            "neural_intermediate_pixel_sha256": result["diagnostics"][
                "neural_intermediate_pixel_sha256"
            ],
            "mask": result["diagnostics"]["mask"],
            "targets_decoded_during_board_prepare": False,
        }
        metadata_path = temporary / "record.json"
        metadata_path.write_bytes(canonical_json_bytes(metadata))
        os.chmod(metadata_path, 0o444)
        os.chmod(temporary, 0o555)
        os.rename(temporary, final_directory)
    finally:
        if temporary.exists():
            os.chmod(temporary, 0o755)
            shutil.rmtree(temporary)


def aggregate_safety(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = safety_summary(
        [row["candidate_structure"] for row in rows],
        [row["reference_structure"] for row in rows],
    )
    protected = np.asarray([row["protected_fraction"] for row in rows])
    color_shift = np.asarray([row["maximum_abs_rgb_mean_shift_vs_reference"] for row in rows])
    spread = np.asarray([row["global_rgb_std_ratio_vs_reference"] for row in rows])
    pixel = np.asarray([row["mean_abs_pixel_change_vs_reference"] for row in rows])
    clipping = np.asarray([row["clipped_fraction_increase_vs_reference"] for row in rows])
    return {
        **summary,
        "protected_fraction_mean_min_max": [
            float(protected.mean()),
            float(protected.min()),
            float(protected.max()),
        ],
        "maximum_rgb_mean_shift": float(color_shift.max()),
        "global_rgb_std_ratio_mean_min_max": [
            float(spread.mean()),
            float(spread.min()),
            float(spread.max()),
        ],
        "mean_abs_pixel_change_mean_max": [float(pixel.mean()), float(pixel.max())],
        "maximum_clipping_increase": float(clipping.max()),
    }


def safety_gate(summary: Mapping[str, Any], all_provenance_pass: bool) -> dict[str, Any]:
    protected = summary["protected_fraction_mean_min_max"]
    spread = summary["global_rgb_std_ratio_mean_min_max"]
    checks = {
        "luma_gradient_mean_at_least_0_90": summary["mean_luminance_gradient_retention"] >= 0.90,
        "luma_gradient_min_at_least_0_80": summary["minimum_luminance_gradient_retention"] >= 0.80,
        "chroma_gradient_mean_at_least_0_80": summary["mean_chroma_gradient_retention"] >= 0.80,
        "chroma_gradient_min_at_least_0_60": summary["minimum_chroma_gradient_retention"] >= 0.60,
        "laplacian_mean_at_least_0_90": summary["mean_laplacian_retention"] >= 0.90,
        "laplacian_min_at_least_0_80": summary["minimum_laplacian_retention"] >= 0.80,
        "grid_ratio_mean_at_most_1_08": summary["mean_grid_ratio_relative_to_baseline"] <= 1.08,
        "grid_ratio_max_at_most_1_15": summary["maximum_grid_ratio_relative_to_baseline"] <= 1.15,
        "protected_fraction_mean_between_0_40_and_0_75": 0.40 <= protected[0] <= 0.75,
        "protected_fraction_every_board_between_0_30_and_0_85": protected[1] >= 0.30
        and protected[2] <= 0.85,
        "rgb_mean_shift_at_most_3": summary["maximum_rgb_mean_shift"] <= 3.0,
        "rgb_std_ratio_every_board_between_0_90_and_1_05": spread[1] >= 0.90 and spread[2] <= 1.05,
        "mean_abs_pixel_change_max_at_most_8": summary["mean_abs_pixel_change_mean_max"][1] <= 8.0,
        "clipping_increase_at_most_0_01": summary["maximum_clipping_increase"] <= 0.01,
        "all_700_strict_raw_provenance_audits_pass": all_provenance_pass,
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


def prepare(stage: str, device: torch.device) -> None:
    config, records, digest = load_context(stage)
    root = STAGE_ROOTS[stage]
    commitment_path = root / "prediction-commitment.json"
    receipt_path = root / "commitment-receipt.json"
    if commitment_path.exists() or receipt_path.exists():
        raise FileExistsError(f"stage is already committed: {root}")
    canonical_device = config["runtime"]["canonical_prediction_device"]
    if device.type != canonical_device:
        raise RuntimeError(f"canonical prediction device is {canonical_device}, got {device.type}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    model = load_drunet_color(CHECKPOINT, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != MODEL_PARAMETER_COUNT:
        raise ValueError("DRUNet parameter count changed")
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    for index, record in enumerate(records, start=1):
        directory = stage_board_directory(stage, str(record["filename"]))
        if directory.exists():
            load_board_record(
                stage,
                record,
                digest,
                verify_input_provenance=True,
            )
            state = "resumed_verified"
        else:
            dirty = load_rgb_verified(INPUTS / record["filename"], record["input_sha256"])
            result = infer_board(dirty, model, device)
            write_board(stage, record, result, digest)
            load_board_record(
                stage,
                record,
                digest,
                verify_input_provenance=True,
            )
            state = "prepared"
        print(
            json.dumps(
                {
                    "phase": "target_blind_prepare",
                    "stage": stage,
                    "done": index,
                    "total": len(records),
                    "filename": record["filename"],
                    "state": state,
                }
            ),
            flush=True,
        )
    boards: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    roster_lines: list[str] = []
    for record in records:
        directory = stage_board_directory(stage, str(record["filename"]))
        metadata, _, _ = load_board_record(stage, record, digest, verify_input_provenance=True)
        metadata_path = directory / "record.json"
        metadata_sha256 = sha256_file(metadata_path)
        boards.append(
            {
                "filename": record["filename"],
                "record_relative_path": str(metadata_path.relative_to(root)),
                "record_sha256": metadata_sha256,
                "layout_sha256": metadata["layout_sha256"],
                "candidate_png_sha256": metadata["images"]["candidate"]["png_sha256"],
                "candidate_pixel_sha256": metadata["images"]["candidate"]["pixel_sha256"],
                "raw_permutation_audit_passed": metadata["raw_permutation_audit"]["passed"],
            }
        )
        safety_rows.append(metadata["safety"])
        roster_lines.append(
            f"{record['filename']} {metadata_sha256} "
            f"{metadata['images']['candidate']['pixel_sha256']}"
        )
    summary = aggregate_safety(safety_rows)
    provenance_pass = all(row["raw_permutation_audit_passed"] for row in boards)
    commitment = {
        "schema": "aiijc-pretrained-drunet-protected-all700-commitment-v1",
        "status": "all_predictions_committed_before_any_target_decode_in_this_measurement_stage",
        "purpose": "measurement_only_no_tuning_no_arm_selection",
        "stage": stage,
        "split": stage,
        "count": len(records),
        "selection_digest": names_digest(records),
        "input_roster_sha256": input_roster_digest(records),
        "historical_target_exposure": config["data"][stage]["historical_target_exposure"],
        "targets_decoded_during_prepare": False,
        "holdout_is_not_claimed_fresh": True,
        "competition_test_access": False,
        "config_sha256": digest,
        "manifest_sha256": sha256_file(MANIFEST),
        "source_sha256": source_hashes(),
        "asset_sha256": verify_assets(),
        "model": {
            "architecture": "KAIR colour DRUNet UNetRes",
            "parameter_count": parameter_count,
            "sigma_255": DRUNET_SIGMA,
            "batch_size": MODEL_BATCH_SIZE,
            "strict_state_dict_load": True,
        },
        "fixed_candidate": ARM_COMBINED,
        "target_free_safety_reference_only": ARM_ORIGINAL_H28,
        "candidate_roster_sha256": hashlib.sha256(
            "\n".join(roster_lines).encode("utf-8")
        ).hexdigest(),
        "all_700_strict_raw_permutation_audits_pass": provenance_pass,
        "target_free_safety": safety_gate(summary, provenance_pass),
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "runtime_seconds_including_resume_verification": perf_counter() - started,
        "boards": boards,
    }
    commitment_sha256 = write_json_exclusive_readonly(commitment_path, commitment)
    receipt = {
        "schema": "aiijc-pretrained-drunet-protected-all700-receipt-v1",
        "status": "commitment_created_before_any_target_decode_in_this_measurement_stage",
        "stage": stage,
        "count": len(records),
        "config_sha256": digest,
        "commitment_relative_path": str(commitment_path.relative_to(PROJECT_ROOT)),
        "commitment_sha256": commitment_sha256,
        "selection_digest": names_digest(records),
        "input_roster_sha256": input_roster_digest(records),
        "candidate_roster_sha256": commitment["candidate_roster_sha256"],
        "targets_decoded_before_receipt": False,
        "competition_test_access": False,
    }
    receipt_sha256 = write_json_exclusive_readonly(receipt_path, receipt)
    print(
        json.dumps(
            {
                "commitment": str(commitment_path),
                "commitment_sha256": commitment_sha256,
                "receipt": str(receipt_path),
                "receipt_sha256": receipt_sha256,
                "candidate_roster_sha256": commitment["candidate_roster_sha256"],
                "safety": commitment["target_free_safety"],
            },
            indent=2,
        )
    )


def load_commitment(
    stage: str,
    records: Sequence[Mapping[str, Any]],
    digest: str,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], np.ndarray]]]:
    root = STAGE_ROOTS[stage]
    commitment_path = root / "prediction-commitment.json"
    receipt_path = root / "commitment-receipt.json"
    for path in (commitment_path, receipt_path):
        if not path.is_file():
            raise FileNotFoundError(f"target-blind commitment artifact is missing: {path}")
        require_readonly(path)
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        commitment.get("stage") != stage
        or commitment.get("count") != BOARD_COUNT
        or commitment.get("config_sha256") != digest
        or commitment.get("source_sha256") != source_hashes()
        or commitment.get("asset_sha256") != verify_assets()
        or commitment.get("selection_digest") != names_digest(records)
    ):
        raise ValueError("commitment binding changed")
    if (
        receipt.get("commitment_sha256") != sha256_file(commitment_path)
        or receipt.get("candidate_roster_sha256") != commitment.get("candidate_roster_sha256")
        or receipt.get("targets_decoded_before_receipt") is not False
    ):
        raise ValueError("commitment receipt binding changed")
    frozen: list[tuple[dict[str, Any], np.ndarray]] = []
    roster_lines: list[str] = []
    for record, board in zip(records, commitment["boards"], strict=True):
        if board["filename"] != record["filename"]:
            raise ValueError("commitment board order changed")
        metadata, _, candidate = load_board_record(
            stage, record, digest, verify_input_provenance=True
        )
        metadata_path = STAGE_ROOTS[stage] / board["record_relative_path"]
        metadata_sha256 = sha256_file(metadata_path)
        if metadata_sha256 != board["record_sha256"]:
            raise ValueError("committed board metadata changed")
        roster_lines.append(
            f"{record['filename']} {metadata_sha256} "
            f"{metadata['images']['candidate']['pixel_sha256']}"
        )
        frozen.append((metadata, candidate))
    roster_digest = hashlib.sha256("\n".join(roster_lines).encode("utf-8")).hexdigest()
    if roster_digest != commitment["candidate_roster_sha256"]:
        raise ValueError("candidate roster changed after commitment")
    return commitment, frozen


def quantile_map(values: np.ndarray) -> dict[str, float]:
    return {
        f"q{int(round(level * 100)):03d}": float(np.quantile(values, level)) for level in QUANTILES
    }


def score(stage: str) -> None:
    config, records, digest = load_context(stage)
    root = STAGE_ROOTS[stage]
    report_path = root / "report.json"
    target_receipt_path = root / "targets-opened-receipt.json"
    if report_path.exists() or target_receipt_path.exists():
        raise FileExistsError(f"stage scoring already started: {root}")
    commitment, frozen = load_commitment(stage, records, digest)
    commitment_path = root / "prediction-commitment.json"
    receipt_path = root / "commitment-receipt.json"
    target_receipt = {
        "schema": "aiijc-pretrained-drunet-protected-all700-target-access-v1",
        "status": "written_after_full_prediction_verification_and_immediately_before_target_decode",
        "stage": stage,
        "count": len(records),
        "config_sha256": digest,
        "commitment_sha256": sha256_file(commitment_path),
        "commitment_receipt_sha256": sha256_file(receipt_path),
        "candidate_roster_sha256": commitment["candidate_roster_sha256"],
        "predictions_were_committed_before_current_target_decode": True,
        "historical_workspace_target_exposure_acknowledged": True,
        "freshness_claim": False,
    }
    target_receipt_sha256 = write_json_exclusive_readonly(target_receipt_path, target_receipt)
    started = perf_counter()
    rows: list[dict[str, Any]] = []
    for index, (record, (metadata, candidate)) in enumerate(
        zip(records, frozen, strict=True), start=1
    ):
        target = load_rgb_verified(TARGETS / record["filename"], record["target_sha256"])
        value = contest_ssim(target, candidate)
        rows.append(
            {
                "ranked_index": index - 1,
                "filename": record["filename"],
                "input_sha256": record["input_sha256"],
                "target_sha256": record["target_sha256"],
                "layout_sha256": metadata["layout_sha256"],
                "candidate_pixel_sha256": metadata["images"]["candidate"]["pixel_sha256"],
                "ssim": value,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "score_frozen_predictions",
                    "stage": stage,
                    "done": index,
                    "total": len(records),
                    "filename": record["filename"],
                    "ssim": value,
                }
            ),
            flush=True,
        )
    values = np.asarray([row["ssim"] for row in rows], dtype=np.float64)
    folds = []
    for fold in range(FOLD_COUNT):
        start = fold * FOLD_SIZE
        stop = start + FOLD_SIZE
        fold_values = values[start:stop]
        folds.append(
            {
                "fold": fold,
                "ranked_start_inclusive": start,
                "ranked_stop_exclusive": stop,
                "count": len(fold_values),
                "mean_ssim": float(fold_values.mean()),
                "minimum_ssim": float(fold_values.min()),
                "maximum_ssim": float(fold_values.max()),
            }
        )
    provenance_pass = bool(
        commitment["all_700_strict_raw_permutation_audits_pass"]
        and all(board["raw_permutation_audit_passed"] for board in commitment["boards"])
    )
    safety_pass = bool(commitment["target_free_safety"]["passed"])
    mean_ssim = float(values.mean())
    interval_pass = 0.27 <= mean_ssim <= 0.28
    broad_gate = {
        "fixed_before_calibration700_score": True,
        "mean_ssim_in_closed_interval_0_27_0_28": interval_pass,
        "all_700_strict_provenance_audits_pass": provenance_pass,
        "target_free_safety_gate_pass": safety_pass,
        "passed": interval_pass and provenance_pass and safety_pass,
        "consequence": (
            "exact_unchanged_holdout700_prepare_then_score_authorized"
            if interval_pass and provenance_pass and safety_pass
            else "holdout700_must_remain_unopened_by_this_measurement"
        ),
    }
    report = {
        "schema": "aiijc-pretrained-drunet-protected-all700-report-v1",
        "status": "exact_all700_measurement_from_precommitted_predictions",
        "purpose": "measurement_only_no_tuning_no_new_arm",
        "stage": stage,
        "split": stage,
        "count": len(rows),
        "config_sha256": digest,
        "commitment_sha256": sha256_file(commitment_path),
        "commitment_receipt_sha256": sha256_file(receipt_path),
        "target_access_receipt_sha256": target_receipt_sha256,
        "candidate_roster_sha256": commitment["candidate_roster_sha256"],
        "fixed_candidate": ARM_COMBINED,
        "mean_ssim": mean_ssim,
        "standard_deviation_ssim": float(values.std(ddof=1)),
        "standard_error_ssim": float(values.std(ddof=1) / np.sqrt(len(values))),
        "quantiles": quantile_map(values),
        "fixed_ranked_folds_10x70": folds,
        "fold_mean_quantiles": quantile_map(np.asarray([fold["mean_ssim"] for fold in folds])),
        "boards_at_or_above_0_27": int(np.sum(values >= 0.27)),
        "boards_at_or_above_0_28": int(np.sum(values >= 0.28)),
        "broad_completion_gate": broad_gate,
        "strict_provenance": {
            "all_700_pass": provenance_pass,
            "audit_contract": (
                "576 unique upright source tiles, exact declared reassembly, equal "
                "tile multiset, restoration only after audit"
            ),
        },
        "target_free_safety": commitment["target_free_safety"],
        "historical_target_exposure": config["data"][stage]["historical_target_exposure"],
        "freshness_claim": False,
        "holdout_access_by_this_measurement": stage == "holdout",
        "competition_test_access": False,
        "runtime_seconds": perf_counter() - started,
        "rows": rows,
    }
    report_sha256 = write_json_exclusive_readonly(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": report_sha256,
                "stage": stage,
                "mean_ssim": mean_ssim,
                "quantiles": report["quantiles"],
                "folds": folds,
                "broad_completion_gate": broad_gate,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("Refusing to run without --run")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.phase == "prepare":
        prepare(args.stage, device)
    else:
        score(args.stage)


if __name__ == "__main__":
    main()
